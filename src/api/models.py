from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import PrimaryKeyConstraint

db = SQLAlchemy()


class NewsSentimentHistory(db.Model):
    __tablename__ = 'news_sentiment_history'

    ticker = db.Column(db.String(10), primary_key=True)
    news_link = db.Column(db.String(512), primary_key=True)

    news_title = db.Column(db.Text, nullable=False)
    news_publication_date = db.Column(db.DateTime(timezone=True))
    sentiment_score = db.Column(db.Float, nullable=False)
    context_text = db.Column(db.Text)
    calculation_date = db.Column(db.DateTime(timezone=True), nullable=False)

    # Define the composite primary key constraint for the table.
    __table_args__ = (
        PrimaryKeyConstraint('ticker', 'news_link'),
    )

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class SentimentThermometer(db.Model):
    __tablename__ = 'sentiment_thermometer'

    ticker = db.Column(db.String(10), primary_key=True)
    calculation_date = db.Column(db.DateTime(timezone=True), primary_key=True)

    avg_sentiment = db.Column(db.Float, nullable=False)
    news_count = db.Column(db.BigInteger, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint('ticker', 'calculation_date'),
    )

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}