from __future__ import annotations

from io import StringIO

import numpy as np
import pendulum
import pandas as pd
import json
import os

from airflow.providers.standard.operators.bash import BashOperator
from sqlalchemy import create_engine, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
import mlflow
from airflow.models.dag import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from sqlalchemy.orm import sessionmaker

from src.api.models import NewsSentimentHistory, SentimentThermometer
from src.database_utils import create_tables_if_not_exist

# --- Dynamic and Portable Paths ---
from src.config import settings

def calculate_weighted_sentiment(text: str, ticker: str, sentiment_model,
                                 window_size: int = 200, stride: int = 50, sigma: float = 100.0) -> tuple[float | None, str | None]:
    """
    Calculates a proximity-weighted sentiment score for a ticker within a text.

    Args:
        text (str): The full text of the news article.
        ticker (str): The stock ticker to analyze.
        sentiment_model: The loaded MLflow pyfunc model.
        window_size (int): The size of each text window (in words).
        stride (int): The step size for the sliding window.
        sigma (float): The standard deviation for the Gaussian weighting function.
                       Controls how quickly the weight decays with distance.

    Returns:
        float | None: The final weighted sentiment score, or None if the ticker is not found.
    """
    words = text.split()
    if not words:
        return None, None

    # Find all positions of the ticker
    ticker_positions = [i for i, word in enumerate(words) if ticker in word.upper()]
    if not ticker_positions:
        return None, None

    # Generate sliding windows and run batch prediction
    windows = []
    for i in range(0, len(words), stride):
        window_text = " ".join(words[i: i + window_size])
        windows.append(window_text)

    if not windows:
        return None, None

    # Run sentiment prediction on all windows at once for efficiency
    window_scores = sentiment_model.predict(pd.DataFrame(windows))

    # Calculate proximity weights for each window
    raw_weights = []
    for i, _ in enumerate(windows):
        window_center = i * stride + (window_size / 2)

        # Find the distance to the NEAREST ticker mention
        min_distance = min([abs(window_center - pos) for pos in ticker_positions])

        # Convert distance to a weight using a Gaussian (bell curve) function
        weight = np.exp(- (min_distance ** 2) / (2 * sigma ** 2))
        raw_weights.append(weight)

    # Normalize weights to sum to 1
    total_weight = sum(raw_weights)
    if total_weight == 0:
        return np.mean(window_scores), None  # Fallback to simple average if no weight is significant

    normalized_weights = [w / total_weight for w in raw_weights]

    if not raw_weights:
        return None, None

    best_window_index = np.argmax(raw_weights)
    best_context = " ".join(text.split()[best_window_index * stride: best_window_index * stride + window_size])

    final_score = np.dot(window_scores, normalized_weights)
    return float(final_score), best_context

# --- Python Functions for Airflow Tasks ---

def setup_database():
    """Ensures that the target table exists in the database."""
    create_tables_if_not_exist()

def load_local_dvc_data(**kwargs):
    """
    Reads the DVC-tracked files directly from the local project directory.
    """
    print(f"Loading data from local project directory: {settings.SENTIMENT_PROJECT_DIR}")

    news_path = os.path.join(settings.SENTIMENT_PROJECT_DIR, 'data', 'raw', 'infomoney_news.csv')
    codes_path = os.path.join(settings.SENTIMENT_PROJECT_DIR, 'data', 'raw', 'stock_codes.json')

    news_df = pd.read_csv(news_path)
    news_df.dropna(subset=['main_text'], inplace=True)

    with open(codes_path, 'r', encoding='utf-8') as f:
        stock_codes = set(json.load(f)['stock_codes'])

    print(f"Loaded {len(news_df)} news articles and {len(stock_codes)} stock codes.")

    def find_tickers_in_text(text, all_tickers):
        found_tickers = {ticker for ticker in all_tickers if f"({ticker})" in text or f" {ticker} " in text}
        return list(found_tickers)

    news_df['tickers'] = news_df['main_text'].apply(lambda text: find_tickers_in_text(text, stock_codes))

    news_with_tickers = news_df[news_df['tickers'].apply(len) > 0].copy()
    print(f"Found tickers in {len(news_with_tickers)} news items.")

    kwargs['ti'].xcom_push(key='news_with_tickers_json', value=news_with_tickers.to_json(orient='split'))


def run_sentiment_inference(**kwargs):
    """
    Loads the production model and runs the advanced, weighted-average
    sentiment inference algorithm for each (article, ticker) pair.
    """
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    ti = kwargs['ti']
    news_json = ti.xcom_pull(key='news_with_tickers_json', task_ids='load_local_data_task')
    if not news_json:
        print("No news with tickers found. Skipping inference.")
        return

    news_df = pd.read_json(StringIO(news_json), orient='split')

    print("Loading Production model from MLflow Registry...")
    model_uri = "models:/sentiment-infomoney-custom@production"
    sentiment_model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully.")

    print("Running advanced sentiment inference for each (article, ticker) pair...")

    # Explode the DataFrame to have one row per (article, ticker) mention
    exploded_df = news_df.explode('tickers').dropna(subset=['tickers'])

    # Apply the advanced sentiment calculation to each row
    results = exploded_df.apply(
        lambda row: calculate_weighted_sentiment(row['main_text'], row['tickers'], sentiment_model),
        axis=1,
        result_type='expand'
    )
    exploded_df[['sentiment_score', 'context_text']] = results

    # Drop rows where sentiment could not be calculated and prepare for next task
    final_results_df = exploded_df.dropna(subset=['sentiment_score'])

    print(f"Calculated sentiment for {len(final_results_df)} ticker mentions.")
    ti.xcom_push(key='inference_results_json', value=final_results_df.to_json(orient='split'))


def store_results_in_db(**kwargs):
    ti = kwargs['ti']
    inference_json = ti.xcom_pull(key='inference_results_json', task_ids='run_inference_task')

    print("Connecting to the database...")
    connection = BaseHook.get_connection('app_postgres_conn')
    db_uri = f"postgresql://{connection.login}:{connection.password}@{connection.host}:{connection.port}/{connection.schema}"
    engine = create_engine(db_uri)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # UPSERT raw inference results into the history table
        if inference_json:
            df = pd.read_json(StringIO(inference_json), orient='split')
            print(f"Processing {len(df)} ticker-article records for the history table...")

            # Prepare data for the UPSERT operation
            history_values_to_upsert = [
                {
                    'ticker': row['tickers'],
                    'news_link': row['link'],
                    'news_title': row['title'],
                    'news_publication_date': pd.to_datetime(row['publication_date']),
                    'sentiment_score': row['sentiment_score'],
                    'context_text': row['context_text'],
                    'calculation_date': pendulum.now('UTC')
                }
                for _, row in df.iterrows()
            ]

            # Create the UPSERT statement for the history table
            history_insert_stmt = pg_insert(NewsSentimentHistory).values(history_values_to_upsert)
            history_upsert_stmt = history_insert_stmt.on_conflict_do_update(
                index_elements=['ticker', 'news_link'],  # The composite primary key
                set_={
                    'sentiment_score': history_insert_stmt.excluded.sentiment_score,
                    'context_text': history_insert_stmt.excluded.context_text,
                    'calculation_date': history_insert_stmt.excluded.calculation_date
                }
            )
            session.execute(history_upsert_stmt)
            print("Successfully upserted records into 'news_sentiment_history'.")
        else:
            print("No new inference results found. Proceeding to recalculate summary.")

        # Recalculate and update the summary table (thermometer)
        print("Recalculating 3-month rolling average sentiment for all tickers...")
        today = pendulum.now('UTC')
        three_months_ago = today.subtract(months=3)

        summary_query = session.query(
            NewsSentimentHistory.ticker,
            func.avg(NewsSentimentHistory.sentiment_score).label('avg_sentiment'),
            func.count().label('news_count')  # Corrected to count all rows in the group
        ).filter(
            NewsSentimentHistory.news_publication_date >= three_months_ago
        ).group_by(NewsSentimentHistory.ticker).all()

        if not summary_query:
            print("No recent news found to calculate summary. Skipping summary update.")
            session.commit()
            return

        # Execute an UPSERT operation for today's summary
        calculation_date_today = today.start_of('day')
        summary_values_to_upsert = [
            {
                'ticker': row.ticker,
                'calculation_date': calculation_date_today,
                'avg_sentiment': row.avg_sentiment,
                'news_count': row.news_count
            }
            for row in summary_query
        ]

        summary_insert_stmt = pg_insert(SentimentThermometer).values(summary_values_to_upsert)
        summary_upsert_stmt = summary_insert_stmt.on_conflict_do_update(
            index_elements=['ticker', 'calculation_date'],
            set_={
                'avg_sentiment': summary_insert_stmt.excluded.avg_sentiment,
                'news_count': summary_insert_stmt.excluded.news_count
            }
        )
        session.execute(summary_upsert_stmt)
        print("Successfully upserted rolling average sentiments into 'sentiment_thermometer'.")

        session.commit()
        print("Database transaction committed successfully.")

    except Exception as e:
        print(f"Error during database operation: {e}. Rolling back transaction.")
        session.rollback()
        raise
    finally:
        session.close()


# DAG Definition
with DAG(
    dag_id="sentiment_inference_pipeline",
    start_date=pendulum.datetime(2025, 10, 1, tz="UTC"),
    schedule="0 */4 * * *",
    catchup=False,
    tags={"mlops", "sentiment", "inference"},
    doc_md="DAG to calculate the stock sentiment thermometer from local DVC-tracked files.",
) as dag:

    update_raw_data_task = BashOperator(
        task_id="update_raw_data_task",
        bash_command=(
            f"cd {settings.SENTIMENT_PROJECT_DIR} && "

            "echo '--- Running DVC data acquisition stages ---' && "
            "dvc repro --force scrape_articles fetch_stock_codes -v && "

            "echo '--- Pushing new raw data to DVC remote ---' && "

            "dvc push data/raw/infomoney_news.csv data/raw/stock_codes.json"
        )
    )

    setup_database_task = PythonOperator(
        task_id="setup_database_task",
        python_callable=setup_database,
    )

    load_local_data_task = PythonOperator(
        task_id="load_local_data_task",
        python_callable=load_local_dvc_data,
    )

    run_inference_task = PythonOperator(
        task_id="run_inference_task",
        python_callable=run_sentiment_inference,
    )

    store_results_task = PythonOperator(
        task_id="store_results_task",
        python_callable=store_results_in_db,
    )

    [setup_database_task, update_raw_data_task] >> load_local_data_task >> run_inference_task >> store_results_task