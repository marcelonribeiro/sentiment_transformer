from __future__ import annotations
import pendulum
import pandas as pd
import json
import os
from pathlib import Path
from sqlalchemy import create_engine, text
import mlflow
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook

# --- Dynamic and Portable Paths ---
PROJECT_DIR = str(Path(__file__).parent.parent.parent)
MLFLOW_TRACKING_URI = f"file://{PROJECT_DIR}/mlruns"

# --- Python Functions for Airflow Tasks ---

def setup_database():
    """Ensures that the target table exists in the database."""
    import sys
    sys.path.insert(0, str(Path(PROJECT_DIR) / "src"))
    from database_utils import create_tables_if_not_exist
    create_tables_if_not_exist()

def load_local_dvc_data(**kwargs):
    """
    Reads the DVC-tracked files directly from the local project directory.
    This task assumes the training DAG has run 'dvc pull' or 'dvc repro'
    to ensure the files are physically present and up-to-date.
    """
    print(f"Loading data from local project directory: {PROJECT_DIR}")

    news_path = os.path.join(PROJECT_DIR, 'data', 'raw', 'infomoney_news.csv')
    codes_path = os.path.join(PROJECT_DIR, 'data', 'raw', 'stock_codes.json')

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
    """Loads the production model from MLflow and runs inference."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    ti = kwargs['ti']
    news_json = ti.xcom_pull(key='news_with_tickers_json', task_ids='load_local_data_task')
    if not news_json:
        print("No news with tickers found. Skipping inference.")
        return

    news_df = pd.read_json(news_json, orient='split')

    print("Loading Production model from MLflow Registry...")
    model_uri = "models:/sentiment-infomoney-custom/Production"
    sentiment_model = mlflow.pyfunc.load_model(model_uri)

    print("Running sentiment prediction...")
    predictions = sentiment_model.predict(pd.DataFrame(news_df['main_text']))
    news_df['sentiment_score'] = predictions

    ti.xcom_push(key='inference_results_json', value=news_df.to_json(orient='split'))

def calculate_and_store_thermometer(**kwargs):
    """Aggregates results by ticker and stores them in the database."""
    ti = kwargs['ti']
    inference_json = ti.xcom_pull(key='inference_results_json', task_ids='run_inference_task')
    if not inference_json:
        print("No inference results found. Skipping database update.")
        return

    df = pd.read_json(inference_json, orient='split')
    df_exploded = df.explode('tickers')

    sentiment_thermometer = df_exploded.groupby('tickers')['sentiment_score'].mean().reset_index()
    sentiment_thermometer.rename(columns={'tickers': 'ticker', 'sentiment_score': 'avg_sentiment'}, inplace=True)
    sentiment_thermometer['last_updated'] = pendulum.now('UTC')

    print("Storing sentiment thermometer in the database...")
    connection = BaseHook.get_connection('app_postgres_conn')
    db_uri = f"postgresql://{connection.login}:{connection.password}@{connection.host}:{connection.port}/{connection.schema}"
    engine = create_engine(db_uri)

    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE sentiment_thermometer;"))
        sentiment_thermometer.to_sql('sentiment_thermometer', con=conn, if_exists='append', index=False)
        conn.commit()

    print(f"{len(sentiment_thermometer)} tickers updated in the database.")

# --- DAG Definition ---
with DAG(
    dag_id="sentiment_inference_pipeline",
    start_date=pendulum.datetime(2025, 10, 1, tz="UTC"),
    schedule_interval="0 */4 * * *",
    catchup=False,
    tags=["mlops", "sentiment", "inference"],
    doc_md="DAG to calculate the stock sentiment thermometer from local DVC-tracked files.",
) as dag:

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
        python_callable=calculate_and_store_thermometer,
    )

    setup_database_task >> load_local_data_task >> run_inference_task >> store_results_task