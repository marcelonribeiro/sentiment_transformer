import pendulum
from flask import Flask, jsonify
from sqlalchemy import func

from src.config import settings
from src.api.models import db, SentimentThermometer, NewsSentimentHistory


# --- Flask App Initialization ---
def create_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = settings.APP_DATABASE_URL
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)

    # --- API Endpoints ---

    @app.route("/api/v1/sentiments/summary", methods=["GET"])
    def get_all_sentiments_summary():
        """Fetches the LATEST sentiment summary for all tickers."""
        latest_calc_subquery = db.session.query(
            SentimentThermometer.ticker,
            func.max(SentimentThermometer.calculation_date).label('max_date')
        ).group_by(SentimentThermometer.ticker).subquery()

        all_data = db.session.query(SentimentThermometer).join(
            latest_calc_subquery,
            (SentimentThermometer.ticker == latest_calc_subquery.c.ticker) &
            (SentimentThermometer.calculation_date == latest_calc_subquery.c.max_date)
        ).order_by(SentimentThermometer.avg_sentiment.desc()).all()

        return jsonify([item.to_dict() for item in all_data])

    @app.route("/api/v1/sentiment/<string:ticker_code>", methods=["GET"])
    def get_current_sentiment(ticker_code: str):
        data = SentimentThermometer.query.filter(
            SentimentThermometer.ticker.ilike(ticker_code)
        ).order_by(
            SentimentThermometer.calculation_date.desc()
        ).first()

        if data:
            return jsonify(data.to_dict())
        return jsonify({"error": "Data not found"}), 404

    @app.route("/api/v1/sentiment/<string:ticker_code>/news", methods=["GET"])
    def get_news_history(ticker_code: str):
        three_months_ago = pendulum.now('UTC').subtract(months=3)

        history = NewsSentimentHistory.query.filter(
            NewsSentimentHistory.ticker.ilike(ticker_code),
            NewsSentimentHistory.news_publication_date >= three_months_ago
        ).order_by(NewsSentimentHistory.news_publication_date.desc()).all()

        return jsonify([item.to_dict() for item in history])

    @app.route("/api/v1/sentiment/<string:ticker_code>/timeseries", methods=["GET"])
    def get_sentiment_timeseries(ticker_code: str):
        timeseries_data = SentimentThermometer.query.filter(
            SentimentThermometer.ticker.ilike(ticker_code)
        ).order_by(SentimentThermometer.calculation_date.asc()).all()

        return jsonify([item.to_dict() for item in timeseries_data])

    @app.route("/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "ok"}), 200

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)