"""Predictions are supposed to be order-invariant via symmetric scoring.

For any fight, predict(A, B) and predict(B, A) should satisfy:
    P(A wins | f1=A) ≈ 1 − P(B wins | f1=B)

If this breaks, the symmetric averaging in _score_row was bypassed or
feature ordering depends on f1/f2 in a way it shouldn't.
"""
import os, re, requests, pytest
from datetime import datetime

SITE_URL = os.environ.get("SITE_URL", "http://107.175.94.166:8002")


def _to_iso_date(s: str | None) -> str | None:
    """Convert 'May 10th, 2026' / 'January 25th' / '2026-05-10' → ISO YYYY-MM-DD."""
    if not s:
        return None
    s = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", s).strip()
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%B %d", "%b %d"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.year == 1900:    # year-less → stamp current year
                dt = dt.replace(year=datetime.now().year)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# Pick a handful of fights from /api/events at module load.
@pytest.fixture(scope="module")
def fight_pairs():
    r = requests.get(f"{SITE_URL}/api/events", timeout=30)
    r.raise_for_status()
    pairs = []
    for ev in r.json():
        if not ev.get("fights"): continue
        iso_date = _to_iso_date(ev.get("event_date"))
        if not iso_date: continue
        for f in ev["fights"][:2]:
            if f.get("f1_odds") is None or f.get("f2_odds") is None: continue
            pairs.append({
                "fighter1": f["fighter1"], "fighter2": f["fighter2"],
                "f1_odds":  f["f1_odds"],  "f2_odds":  f["f2_odds"],
                "date":     iso_date,
            })
        if len(pairs) >= 12: break
    return pairs[:12]


def _predict(f1, f2, o1, o2, date):
    r = requests.post(f"{SITE_URL}/api/predict", json={
        "fighter1": f1, "fighter2": f2,
        "fighter1_odds": o1, "fighter2_odds": o2,
        "fight_date": date,
    }, timeout=30)
    r.raise_for_status()
    return r.json()


def test_predict_swap_symmetry(fight_pairs):
    """P(A) when A=f1 must equal 1 − P(A) when B=f1."""
    bad = []
    for fp in fight_pairs:
        a = _predict(fp["fighter1"], fp["fighter2"], fp["f1_odds"], fp["f2_odds"], fp["date"])
        b = _predict(fp["fighter2"], fp["fighter1"], fp["f2_odds"], fp["f1_odds"], fp["date"])
        if a.get("model_prob_f1") is None or b.get("model_prob_f1") is None:
            continue
        # In b, f1 is the original fighter2, so P(orig fighter1) = 1 − model_prob_f1
        delta = abs(a["model_prob_f1"] - (100 - b["model_prob_f1"]))
        if delta > 0.5:
            bad.append(f"{fp['fighter1']} vs {fp['fighter2']}: a={a['model_prob_f1']} "
                       f"b_inv={100-b['model_prob_f1']:.1f} delta={delta:.2f}pp")
    assert not bad, "Swap-symmetry violations:\n  " + "\n  ".join(bad)
