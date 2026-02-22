# Performance Optimization Ideas for UFC ML Agents

## Executive Summary

This document outlines potential performance optimizations for the UFC ML Agents feature pipeline, categorized by priority, implementation complexity, and expected impact. The goal is to continue the 10-100x speedup achieved by batch loading.

---

## 🔴 HIGH IMPACT / HIGH PRIORITY

### 1. Persistent Fighter Feature Cache (2-5x speedup)

**Problem:** Fighter features are recomputed for every fight they appear in. A fighter with 30 fights has their features calculated 30 times during dataset creation.

**Solution:** Implement persistent caching of fighter features using LRU cache or Redis.

**Implementation:**
```python
from functools import lru_cache
import pickle
from pathlib import Path

class CachedFeatureBuilder(FeatureBuilder):
    def __init__(self, session, rolling_windows, cache_dir='cache/fighter_features'):
        super().__init__(session, rolling_windows)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build_features(self, fighter_id, feature_set, as_of_date=None):
        cache_key = f"{fighter_id}_{as_of_date}_{hash(tuple(feature_set))}"
        cache_path = self.cache_dir / f"{cache_key}.pkl"

        # Try to load from cache
        if cache_path.exists():
            return pickle.load(open(cache_path, 'rb'))

        # Compute features
        features = super().build_features(fighter_id, feature_set, as_of_date)

        # Save to cache
        pickle.dump(features, open(cache_path, 'wb'))
        return features
```

**Expected Impact:** 2-5x speedup for dataset creation
**Implementation Effort:** Medium (1-2 days)
**Risk:** Low - cache can be invalidated/cleared

---

### 2. Batch Fight History Queries (3-5x speedup)

**Problem:** Each fighter's fight history is queried individually, causing N database queries for N fighters.

**Solution:** Batch query all fight histories upfront and index by fighter ID.

**Implementation:**
```python
def get_all_fight_histories_batch(session: Session, as_of_date=None) -> Dict[int, pd.DataFrame]:
    """
    Query all fight histories in a single batch operation.
    Returns dict mapping fighter_id -> fight_history DataFrame
    """
    # Single query to get all relevant fights
    query = session.query(
        Fight.id,
        Fight.fighter_1_id,
        Fight.fighter_2_id,
        Fight.result,
        Fight.event_id,
        Fight.method,
        Event.date.label('event_date')
    ).join(Event)

    # Filter by date if needed
    if as_of_date:
        query = query.filter(Event.date < as_of_date)

    # Execute single query
    all_fights = query.all()

    # Build dict of fighter histories
    fighter_histories = {}

    for fight in all_fights:
        # Process fighter_1 perspective
        if fight.fighter_1_id not in fighter_histories:
            fighter_histories[fight.fighter_1_id] = []
        fighter_histories[fight.fighter_1_id].append({
            'fight_id': fight.id,
            'opponent_id': fight.fighter_2_id,
            'result': 'win' if fight.result == 'fighter_1' else 'loss' if fight.result == 'fighter_2' else 'draw',
            'method': fight.method,
            'date': fight.event_date
        })

        # Process fighter_2 perspective
        if fight.fighter_2_id not in fighter_histories:
            fighter_histories[fight.fighter_2_id] = []
        fighter_histories[fight.fighter_2_id].append({
            'fight_id': fight.id,
            'opponent_id': fight.fighter_1_id,
            'result': 'win' if fight.result == 'fighter_2' else 'loss' if fight.result == 'fighter_1' else 'draw',
            'method': fight.method,
            'date': fight.event_date
        })

    # Convert to DataFrames
    for fighter_id in fighter_histories:
        fighter_histories[fighter_id] = pd.DataFrame(fighter_histories[fighter_id])

    return fighter_histories
```

**Expected Impact:** 3-5x speedup for dataset creation
**Implementation Effort:** Medium (2-3 days)
**Risk:** Medium - need to handle memory for large datasets

---

### 3. Pre-computed Rolling Statistics (2-3x speedup)

**Problem:** Rolling statistics (last 3 fights, last 5 fights, etc.) are recalculated for every fight.

**Solution:** Pre-compute rolling windows and store as database columns.

**Implementation:**
```python
def compute_and_cache_rolling_stats(session: Session):
    """
    Pre-compute rolling statistics for all fighters and store in database.
    Run this once after data update.
    """
    fighters = session.query(Fighter).all()

    for fighter in fighters:
        history = get_fight_history(session, fighter.id)

        # Pre-compute all rolling windows
        for window in [3, 5, 10]:
            rolling = history.rolling(window)

            stats = {
                f'wins_last_{window}': (history['result'] == 'win').rolling(window).sum(),
                f'ko_rate_last_{window}': (history['method'].str.contains('KO').rolling(window).mean(),
                # ... more stats
            }

            # Store in Fighter model or new FighterRollingStats table
            for stat_name, stat_values in stats.items():
                setattr(fighter, stat_name, stat_values.iloc[-1] if len(stat_values) > 0 else 0)

    session.commit()
```

**Expected Impact:** 2-3x speedup for feature extraction
**Implementation Effort:** High (3-5 days)
**Risk:** High - need cache invalidation strategy

---

### 4. Parallel Feature Extraction (4-8x speedup with multiprocessing)

**Problem:** Feature extraction is sequential, wasting CPU cores.

**Solution:** Use multiprocessing to extract features in parallel.

**Implementation:**
```python
from multiprocessing import Pool, cpu_count
from functools import partial

def extract_matchup_features_parallel(
    matchups: List[Tuple[int, int, Optional[datetime]]],
    num_workers: int = None
) -> List[Dict]:
    """
    Extract matchup features in parallel using multiprocessing.
    """
    if num_workers is None:
        num_workers = cpu_count() - 1  # Leave one core free

    logger.info(f"Using {num_workers} workers for parallel feature extraction")

    # Create a session per worker to avoid sharing issues
    def worker_init():
        global worker_session
        worker_session = DatabaseManager().get_session()

    def extract_single_matchup(fighter_1_id, fighter_2_id, as_of_date):
        """Worker function to extract features for a single matchup"""
        extractor = MatchupFeatureExtractor(worker_session)
        return extractor.extract_matchup_features(
            fighter_1_id, fighter_2_id, as_of_date
        )

    # Extract in parallel
    with Pool(processes=num_workers, initializer=worker_init) as pool:
        results = pool.starmap(extract_single_matchup, matchups)

    return results
```

**Expected Impact:** 4-8x speedup on multi-core machines
**Implementation Effort:** Medium (2-3 days)
**Risk:** High - SQLite locking issues, need connection pooling

---

## 🟡 MEDIUM IMPACT / MEDIUM PRIORITY

### 5. Database Query Optimization (1.5-2x speedup)

**Problem:** Unoptimized SQL queries with unnecessary joins and fetching all columns.

**Solution:** Use indexed queries and select only needed columns.

**Implementation:**
```python
# BEFORE (slow)
fights = (
    session.query(Fight)
    .filter(Fight.result != None)
    .join(Fight.event)
    .order_by(Fight.id)
    .all()
)

# AFTER (fast)
fights = (
    session.query(
        Fight.id,
        Fight.fighter_1_id,
        Fight.fighter_2_id,
        Fight.result,
        Fight.weight_class,
        Fight.is_title_fight,
        Fight.method,
        Event.date.label('event_date')
    )
    .filter(Fight.result != None)
    .join(Event)
    .order_by(Fight.id)
    .all()
)
```

**Add indexes:**
```python
# In database/schema.py
class Fight(Base):
    # ... existing fields ...

    # Add compound indexes for common queries
    __table_args__ = (
        Index('idx_fight_result_date', 'result', 'event_id'),
        Index('idx_fighter_1_result', 'fighter_1_id', 'result'),
        Index('idx_fighter_2_result', 'fighter_2_id', 'result'),
    )
```

**Expected Impact:** 1.5-2x speedup
**Implementation Effort:** Low (1 day)
**Risk:** Low

---

### 6. Lazy Loading of FightStats (1.5-2x speedup)

**Problem:** FightStats JSON is loaded even when not needed for basic features.

**Solution:** Only load FightStats when explicitly needed.

**Implementation:**
```python
# In database/schema.py
class Fight(Base):
    # ... existing fields ...

    # Change relationship to lazy loading
    fight_stats = relationship('FightStats', back_populates='fight',
                          uselist=False, lazy='joined')

# OR make it explicit
class Fight(Base):
    # ... existing fields ...
    fight_stats = relationship('FightStats', back_populates='fight',
                          uselist=False, lazy='select')  # Load on access
```

**Expected Impact:** 1.5-2x speedup for features not using FightStats
**Implementation Effort:** Low (0.5 days)
**Risk:** Low

---

### 7. Pre-computed Opponent Quality (1.5-2x speedup)

**Problem:** Opponent quality requires recursive calculation (opponents' opponents, etc.).

**Solution:** Pre-compute and cache opponent quality scores.

**Implementation:**
```python
def compute_opponent_quality_scores(session: Session):
    """
    Pre-compute opponent quality scores for all fighters.
    Store in Fighter.opponent_quality_score column.
    """
    fighters = session.query(Fighter).all()

    for fighter in fighters:
        quality_score = compute_opponent_quality_recursive(
            session, fighter.id, max_depth=2
        )
        fighter.opponent_quality_score = quality_score

    session.commit()

# Then in features/opponent_quality.py:
def extract_opponent_quality_features(fighter_id, fight_history, session):
    """Use pre-computed score instead of recalculating"""
    fighter = session.query(Fighter).filter_by(id=fighter_id).first()
    return {
        'opponent_quality_score': fighter.opponent_quality_score or 0.0,
    }
```

**Expected Impact:** 1.5-2x speedup
**Implementation Effort:** Medium (2-3 days)
**Risk:** Medium - need cache invalidation

---

### 8. Chunked Dataset Processing (1.5-2x memory efficiency)

**Problem:** Entire dataset loaded into memory before saving.

**Solution:** Process and save in chunks.

**Implementation:**
```python
def create_training_dataset_chunked(
    session: Session,
    output_path: str = 'data/processed/training_data.csv',
    chunk_size: int = 1000
):
    """
    Create training dataset in chunks to reduce memory usage.
    """
    # Get all fights
    fights = get_all_fights(session)

    # Process in chunks
    chunk = []
    for i, fight in enumerate(fights, 1):
        # Extract features for this fight
        features_win = extract_features_for_fight(fight, winner=True)
        features_lose = extract_features_for_fight(fight, winner=False)

        chunk.extend([features_win, features_lose])

        # Save chunk when full
        if len(chunk) >= chunk_size:
            df_chunk = pd.DataFrame(chunk)
            write_mode = 'w' if i == 1 else 'a'  # Write or append
            header = i == 1  # Only write header once
            df_chunk.to_csv(output_path, mode=write_mode, header=header, index=False)
            chunk = []
            logger.info(f"Saved chunk {i // chunk_size}")

    # Save remaining
    if chunk:
        pd.DataFrame(chunk).to_csv(output_path, mode='a', header=False, index=False)
```

**Expected Impact:** 1.5-2x memory reduction (faster for large datasets)
**Implementation Effort:** Low (1 day)
**Risk:** Low

---

## 🟢 LOW IMPACT / LOW PRIORITY

### 9. Use NumPy instead of Pandas for Aggregations (1.2-1.5x speedup)

**Problem:** Pandas overhead for simple aggregations.

**Solution:** Use NumPy for numeric operations.

**Implementation:**
```python
# BEFORE
win_rate = (fight_history['result'] == 'win').mean()

# AFTER
win_rate = np.mean(fight_history['result'] == 'win')
```

**Expected Impact:** 1.2-1.5x speedup for numeric operations
**Implementation Effort:** Low (1 day)
**Risk:** Low

---

### 10. Pre-compiled Feature Functions (1.2-1.5x speedup)

**Problem:** Feature function lookups happen for every feature.

**Solution:** Pre-compile feature functions once.

**Implementation:**
```python
class FeatureBuilderOptimized(FeatureBuilder):
    def __init__(self, session, rolling_windows):
        super().__init__(session, rolling_windows)

        # Pre-compile feature functions
        self.feature_functions = {}
        for feature_name in FeatureRegistry.FEATURE_SET_FULL:
            self.feature_functions[feature_name] = FeatureRegistry.get_feature_function(feature_name)

    def build_features(self, fighter_id, feature_set, as_of_date=None):
        """Use pre-compiled functions"""
        # ... rest of implementation
```

**Expected Impact:** 1.2-1.5x speedup
**Implementation Effort:** Low (0.5 days)
**Risk:** Low

---

### 11. Avoid Repeated String Parsing (1.2-1.5x speedup)

**Problem:** Fight methods, dates, etc. parsed repeatedly.

**Solution:** Parse once and store as structured data.

**Implementation:**
```python
# In database schema, add parsed fields
class Fight(Base):
    # ... existing fields ...

    # Add parsed/cached fields
    method_is_ko = Column(Boolean)
    method_is_submission = Column(Boolean)
    method_is_decision = Column(Boolean)
    event_date_parsed = Column(DateTime)  # Parse date string once

# Pre-populate these on data load
def populate_parsed_fight_data(session: Session):
    fights = session.query(Fight).all()
    for fight in fights:
        fight.method_is_ko = 'KO' in fight.method
        fight.method_is_submission = 'SUB' in fight.method
        fight.method_is_decision = 'DEC' in fight.method
        fight.event_date_parsed = parse_date(fight.event.date)
    session.commit()
```

**Expected Impact:** 1.2-1.5x speedup
**Implementation Effort:** Low (1 day)
**Risk:** Low

---

### 12. Optimize SQL Session Management (1.1-1.3x speedup)

**Problem:** Session overhead and connection management.

**Solution:** Use connection pooling and reuse sessions.

**Implementation:**
```python
# In database/db_manager.py
class DatabaseManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        # ... existing code ...

        # Add connection pooling
        from sqlalchemy.pool import QueuePool
        self.engine = create_engine(
            connection_string,
            echo=False,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Check connection health
            pool_recycle=3600  # Recycle connections after 1 hour
        )
```

**Expected Impact:** 1.1-1.3x speedup
**Implementation Effort:** Low (0.5 days)
**Risk:** Low

---

## 📊 IMPACT MATRIX

| # | Optimization | Speedup | Effort | Risk | Priority |
|---|--------------|----------|---------|------|----------|
| 1 | Persistent Fighter Cache | 2-5x | Medium | Low | 🔴 HIGH |
| 2 | Batch Fight History Queries | 3-5x | Medium | Medium | 🔴 HIGH |
| 3 | Pre-computed Rolling Stats | 2-3x | High | High | 🔴 HIGH |
| 4 | Parallel Feature Extraction | 4-8x | Medium | High | 🔴 HIGH |
| 5 | Database Query Optimization | 1.5-2x | Low | Low | 🟡 MEDIUM |
| 6 | Lazy Loading FightStats | 1.5-2x | Low | Low | 🟡 MEDIUM |
| 7 | Pre-computed Opponent Quality | 1.5-2x | Medium | Medium | 🟡 MEDIUM |
| 8 | Chunked Dataset Processing | 1.5-2x | Low | Low | 🟡 MEDIUM |
| 9 | NumPy Aggregations | 1.2-1.5x | Low | Low | 🟢 LOW |
| 10 | Pre-compiled Feature Functions | 1.2-1.5x | Low | Low | 🟢 LOW |
| 11 | Avoid Repeated String Parsing | 1.2-1.5x | Low | Low | 🟢 LOW |
| 12 | Optimize SQL Session Management | 1.1-1.3x | Low | Low | 🟢 LOW |

---

## 🚀 RECOMMENDED IMPLEMENTATION ROADMAP

### Phase 1: Quick Wins (Week 1)
1. **#5 Database Query Optimization** - Low effort, immediate impact
2. **#6 Lazy Loading FightStats** - Low effort, immediate impact
3. **#12 Optimize SQL Session Management** - Low effort, baseline improvement

**Expected Phase 1 Impact:** 1.5-2x total speedup

---

### Phase 2: Medium Effort, High Impact (Week 2-3)
4. **#1 Persistent Fighter Feature Cache** - Medium effort, high impact
5. **#2 Batch Fight History Queries** - Medium effort, high impact

**Expected Phase 2 Impact:** Additional 3-5x total speedup (cumulative: 5-10x)

---

### Phase 3: Advanced Optimizations (Week 4-5)
6. **#8 Chunked Dataset Processing** - Low effort, memory efficiency
7. **#7 Pre-computed Opponent Quality** - Medium effort, specific improvement

**Expected Phase 3 Impact:** Additional 1.5-2x total speedup (cumulative: 8-20x)

---

### Phase 4: Experimental/High-Risk (Week 6+)
8. **#3 Pre-computed Rolling Stats** - High effort, high impact, high risk
9. **#4 Parallel Feature Extraction** - Medium effort, very high impact, high risk

**Expected Phase 4 Impact:** Additional 4-8x total speedup (cumulative: 32-160x)

---

## 🎯 PROJECTIONS

### Current Baseline
- Dataset creation time (5,000 fights): ~10 minutes
- Feature vector building: ~12 seconds

### After Phase 1 (Quick Wins)
- Dataset creation: ~5-7 minutes (1.5-2x speedup)
- Feature vector building: ~8-10 seconds

### After Phase 2 (Medium Effort)
- Dataset creation: ~30 seconds - 2 minutes (5-10x speedup)
- Feature vector building: ~4-6 seconds

### After Phase 3 (Advanced)
- Dataset creation: ~15-30 seconds (8-20x speedup)
- Feature vector building: ~2-3 seconds

### After Phase 4 (Experimental)
- Dataset creation: **5-20 seconds** (32-160x speedup)
- Feature vector building: **<1 second**

---

## 🛠️ IMPLEMENTATION TIPS

### Profiling First
Before implementing any optimization, profile to find actual bottlenecks:

```python
import cProfile
import pstats
from io import StringIO

def profile_feature_pipeline():
    pr = cProfile.Profile()
    pr.enable()

    # Run feature pipeline
    create_training_dataset(session)

    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    print(s.getvalue())

profile_feature_pipeline()
```

### Benchmark After Each Change
```python
import time

def benchmark(func, *args, iterations=10):
    times = []
    for _ in range(iterations):
        start = time.time()
        func(*args)
        times.append(time.time() - start)
    avg = sum(times) / len(times)
    print(f"{func.__name__}: {avg:.2f}s avg (over {iterations} runs)")
    return avg
```

### A/B Testing
Always test optimized version against baseline to ensure correctness:

```python
def test_optimization():
    # Baseline
    baseline = extract_features_baseline(fighter_id)

    # Optimized
    optimized = extract_features_optimized(fighter_id)

    # Verify results are identical (within floating point tolerance)
    assert np.allclose(baseline, optimized, rtol=1e-5)
    print("✅ Optimization verified - results are identical")
```

---

## 🔍 MONITORING & METRICS

Add performance logging to track improvements:

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    """Context manager to time operations"""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"[PERF] {name}: {elapsed:.2f}s")

# Usage:
with timer("Dataset Creation"):
    create_training_dataset(session)

with timer("Feature Extraction"):
    features = extract_features(fighter_id)
```

---

## 📝 CAVEATS & CONSIDERATIONS

1. **SQLite Limitations:** SQLite has limited write concurrency. For multiprocessing (#4), consider PostgreSQL or use read-only workers.

2. **Memory vs Speed:** Some optimizations trade memory for speed (#2, #3). Monitor memory usage on your machine.

3. **Cache Invalidation:** With caching (#1, #3, #7), need strategy to invalidate when data changes.

4. **Development Time vs Impact:** Quick wins give better ROI. Don't over-optimize early.

5. **Maintainability:** Complex optimizations (#3, #4) make code harder to maintain. Document well.

6. **Testing:** Every optimization needs comprehensive tests to ensure correctness.

---

## 🎓 FUTURE DIRECTIONS

### Advanced Optimizations (Not covered above)

- **GPU Acceleration:** Use CuPy for NumPy operations on GPU
- **Just-In-Time Compilation:** Use Numba to compile Python functions
- **Feature Selection:** Remove redundant features to reduce computation
- **Incremental Updates:** Only recompute changed fighters
- **Database Materialized Views:** Pre-compute complex joins
- **Redis Cache:** Distributed caching for fighter features
- **Async I/O:** Use asyncpg for PostgreSQL async queries

---

## 📚 REFERENCES

- SQLAlchemy Performance Tips: https://docs.sqlalchemy.org/en/14/core/performance.html
- Pandas Performance: https://pandas.pydata.org/docs/user_guide/enhancingperf.html
- Python Multiprocessing: https://docs.python.org/3/library/multiprocessing.html
- NumPy Performance: https://numpy.org/doc/stable/user/whatsnew2.0.html#array-iteration-performance-improvements

---

## ✅ CHECKLIST FOR IMPLEMENTATION

For each optimization:
- [ ] Profile to confirm it's a bottleneck
- [ ] Implement optimization
- [ ] Write unit tests
- [ ] Benchmark before/after
- [ ] Update documentation
- [ ] Add performance monitoring
- [ ] Code review
- [ ] Deploy to production
- [ ] Monitor post-deployment

---

**Last Updated:** January 29, 2026
**Maintainer:** Feature Pipeline Team
**Version:** 1.0

