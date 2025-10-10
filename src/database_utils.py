import os
from flask import Flask
from sqlalchemy import create_engine

# --- Application Imports ---
# Import the 'db' object and application settings
from src.api.models import db
from src.config import settings


def get_db_engine():
    """
    Creates a SQLAlchemy engine from the APP_DATABASE_URL environment variable.
    This function remains useful for scripts needing a direct connection.
    """
    db_url = os.getenv("APP_DATABASE_URL")
    if not db_url:
        raise ValueError("APP_DATABASE_URL environment variable is not set.")
    return create_engine(db_url)


def create_tables_if_not_exist():
    """
    REFACTORED: Connects to the database and creates tables using the ORM models
    from 'src/api/models.py' as the single source of truth.
    """
    print("Connecting to the database to ensure tables exist based on ORM models...")

    # Create a temporary Flask app instance to provide the necessary context
    temp_app = Flask(__name__)
    temp_app.config["SQLALCHEMY_DATABASE_URI"] = settings.APP_DATABASE_URL
    temp_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Associate our application's db object with the temporary app
    db.init_app(temp_app)

    # Use the application context to run database operations
    with temp_app.app_context():
        # Import the models here to ensure they are registered in the 'db' metadata
        # within the correct context.
        from src.api.models import NewsSentimentHistory, SentimentThermometer

        # SQLAlchemy now inspects the models and creates any tables that do not exist.
        db.create_all()

    print("Database tables are synchronized with the ORM models.")


if __name__ == '__main__':
    # Allows running this script directly to initialize the database
    # Example usage: python src/database_utils.py
    print("Running database setup directly...")
    create_tables_if_not_exist()
    print("Setup complete.")