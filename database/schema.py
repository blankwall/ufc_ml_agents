"""
Database schema definitions using SQLAlchemy
"""

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Fighter(Base):
    """Fighter information and career statistics"""
    __tablename__ = 'fighters'
    
    # Primary identifiers
    id = Column(Integer, primary_key=True, autoincrement=True)
    fighter_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False, index=True)
    nickname = Column(String(200))
    
    # Physical attributes
    height_cm = Column(Float)
    weight_lbs = Column(Float)
    reach_inches = Column(Float)
    stance = Column(String(50))
    date_of_birth = Column(String(50))
    age = Column(Integer)
    
    # Record
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    no_contests = Column(Integer, default=0)
    
    # Career statistics (averages)
    sig_strikes_landed_per_min = Column(Float)
    striking_accuracy = Column(Float)
    sig_strikes_absorbed_per_min = Column(Float)
    striking_defense = Column(Float)
    takedown_avg_per_15min = Column(Float)
    takedown_accuracy = Column(Float)
    takedown_defense = Column(Float)
    submission_avg_per_15min = Column(Float)
    
    # Metadata
    url = Column(String(500))
    scraped_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    fights_as_fighter_1 = relationship('Fight', foreign_keys='Fight.fighter_1_id', back_populates='fighter_1')
    fights_as_fighter_2 = relationship('Fight', foreign_keys='Fight.fighter_2_id', back_populates='fighter_2')
    
    def __repr__(self):
        return f"<Fighter(name='{self.name}', record={self.wins}-{self.losses}-{self.draws})>"


class Event(Base):
    """UFC Event information"""
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(500), nullable=False)
    date = Column(String(100))
    location = Column(String(500))
    venue = Column(String(500))
    
    # Metadata
    url = Column(String(500))
    scraped_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    fights = relationship('Fight', back_populates='event')
    
    def __repr__(self):
        return f"<Event(name='{self.name}', date='{self.date}')>"


class Fight(Base):
    """Individual fight information"""
    __tablename__ = 'fights'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fight_id = Column(String(50), unique=True, index=True)
    
    # Event relationship
    event_id = Column(Integer, ForeignKey('events.id'), nullable=False, index=True)
    event = relationship('Event', back_populates='fights')
    fight_number = Column(Integer)  # Order on the card (1 = main event)
    
    # Fighters
    fighter_1_id = Column(Integer, ForeignKey('fighters.id'), nullable=False, index=True)
    fighter_1 = relationship('Fighter', foreign_keys=[fighter_1_id], back_populates='fights_as_fighter_1')
    
    fighter_2_id = Column(Integer, ForeignKey('fighters.id'), nullable=False, index=True)
    fighter_2 = relationship('Fighter', foreign_keys=[fighter_2_id], back_populates='fights_as_fighter_2')
    
    # Fight details
    weight_class = Column(String(100))
    is_title_fight = Column(Boolean, default=False)
    scheduled_rounds = Column(Integer, default=3)
    
    # Result
    winner_id = Column(Integer, ForeignKey('fighters.id'))  # NULL if draw/NC
    result = Column(String(50))  # 'fighter_1', 'fighter_2', 'draw', 'no_contest'
    method = Column(String(200))  # KO/TKO, Submission, Decision, etc.
    method_detail = Column(String(500))  # Specific technique
    round_finished = Column(Integer)
    time = Column(String(20))  # Time in round (MM:SS)
    
    # URLs
    fight_detail_url = Column(String(500))
    
    # Metadata
    scraped_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    fight_stats = relationship('FightStats', back_populates='fight', uselist=False)
    predictions = relationship('Prediction', back_populates='fight')
    betting_odds = relationship('BettingOdds', back_populates='fight')
    
    def __repr__(self):
        return f"<Fight({self.fighter_1.name if self.fighter_1 else '?'} vs {self.fighter_2.name if self.fighter_2 else '?'})>"


class FightStats(Base):
    """Detailed round-by-round fight statistics"""
    __tablename__ = 'fight_stats'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fight_id = Column(Integer, ForeignKey('fights.id'), unique=True, nullable=False)
    fight = relationship('Fight', back_populates='fight_stats')
    
    # Overall totals (JSON for flexibility)
    fighter_1_totals = Column(JSON)  # Total strikes, takedowns, etc.
    fighter_2_totals = Column(JSON)
    
    # Round-by-round breakdown
    round_by_round = Column(JSON)  # Array of round statistics
    
    # Significant strikes breakdown
    significant_strikes = Column(JSON)  # Head, body, leg, distance, clinch, ground
    
    scraped_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<FightStats(fight_id={self.fight_id})>"


class Prediction(Base):
    """Model predictions for fights"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fight_id = Column(Integer, ForeignKey('fights.id'), nullable=False, index=True)
    fight = relationship('Fight', back_populates='predictions')
    
    # Model information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    
    # Predictions
    fighter_1_win_probability = Column(Float, nullable=False)
    fighter_2_win_probability = Column(Float, nullable=False)
    draw_probability = Column(Float, default=0.0)
    
    # Predicted outcome
    predicted_winner = Column(String(50))  # 'fighter_1' or 'fighter_2'
    confidence = Column(Float)  # 0-1 scale
    
    # Method predictions
    predicted_method = Column(String(100))  # KO/TKO, Submission, Decision
    predicted_round = Column(Integer)
    
    # Feature importance (for interpretability)
    feature_importance = Column(JSON)
    
    # Metadata
    predicted_at = Column(DateTime, default=datetime.now, index=True)
    
    # Evaluation (after fight happens)
    was_correct = Column(Boolean)
    log_loss = Column(Float)
    brier_score = Column(Float)
    
    def __repr__(self):
        return f"<Prediction(fight_id={self.fight_id}, model={self.model_name}, winner={self.predicted_winner})>"


class BettingOdds(Base):
    """Betting odds from various bookmakers"""
    __tablename__ = 'betting_odds'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fight_id = Column(Integer, ForeignKey('fights.id'), nullable=False, index=True)
    fight = relationship('Fight', back_populates='betting_odds')
    
    # Bookmaker info
    bookmaker = Column(String(100), nullable=False)
    
    # Odds (American format)
    fighter_1_odds = Column(Integer)  # e.g., -150, +200
    fighter_2_odds = Column(Integer)
    
    # Implied probabilities
    fighter_1_implied_prob = Column(Float)
    fighter_2_implied_prob = Column(Float)
    
    # Opening vs closing lines
    is_opening_line = Column(Boolean, default=False)
    is_closing_line = Column(Boolean, default=False)
    
    # Timestamp
    odds_timestamp = Column(DateTime, default=datetime.now, index=True)
    scraped_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<BettingOdds(fight_id={self.fight_id}, bookmaker={self.bookmaker})>"


class BettingRecommendation(Base):
    """Betting recommendations based on edge detection"""
    __tablename__ = 'betting_recommendations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fight_id = Column(Integer, ForeignKey('fights.id'), nullable=False, index=True)
    
    # Recommendation
    recommended_bet = Column(String(50))  # 'fighter_1', 'fighter_2', 'no_bet'
    recommended_stake = Column(Float)  # As percentage of bankroll
    
    # Edge calculation
    model_probability = Column(Float)
    market_probability = Column(Float)
    edge = Column(Float)  # Difference between model and market
    expected_value = Column(Float)  # EV of the bet
    
    # Odds at time of recommendation
    recommended_odds = Column(Integer)
    bookmaker = Column(String(100))
    
    # Metadata
    recommended_at = Column(DateTime, default=datetime.now, index=True)
    
    # Outcome tracking
    bet_placed = Column(Boolean, default=False)
    actual_stake = Column(Float)
    profit_loss = Column(Float)
    
    def __repr__(self):
        return f"<BettingRecommendation(fight_id={self.fight_id}, bet={self.recommended_bet}, edge={self.edge:.2%})>"


class ModelPerformance(Base):
    """Track model performance over time"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    
    # Evaluation period
    evaluation_start_date = Column(DateTime)
    evaluation_end_date = Column(DateTime)
    num_fights = Column(Integer)
    
    # Performance metrics
    accuracy = Column(Float)
    log_loss = Column(Float)
    brier_score = Column(Float)
    roc_auc = Column(Float)
    
    # Betting performance
    roi = Column(Float)
    total_profit_loss = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    
    # Calibration metrics
    calibration_error = Column(Float)
    
    # Metadata
    evaluated_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<ModelPerformance(model={self.model_name}, accuracy={self.accuracy:.2%}, roi={self.roi:.2%})>"


def create_all_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(engine)


def drop_all_tables(engine):
    """Drop all tables from the database"""
    Base.metadata.drop_all(engine)

