import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from the project root
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)


class Settings:
    # Project Paths
    SENTIMENT_PROJECT_DIR: str = os.getenv("SENTIMENT_PROJECT_DIR", str(project_root))
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
    MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

    # Database URL
    APP_DATABASE_URL: str = os.getenv("APP_DATABASE_URL")

    # API URL
    API_BASE_URL: str = os.getenv("API_BASE_URL")

    # MLflow Model Registry
    MODEL_NAME_REGISTRY: str = "sentiment-infomoney-custom"

    # External API Credentials
    STRATEGIA_INVEST_API_USERNAME = os.getenv("STRATEGIA_INVEST_API_USERNAME")
    STRATEGIA_INVEST_API_PASSWORD = os.getenv("STRATEGIA_INVEST_API_PASSWORD")
    STRATEGIA_INVEST_API_BASE_URL = os.getenv("STRATEGIA_INVEST_API_BASE_URL")

    # AWS Credentials
    AWS_REGION = os.getenv("AWS_REGION")
    AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")

settings = Settings()