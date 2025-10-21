from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.providers.standard.operators.bash import BashOperator

from src.config import settings

with DAG(
        dag_id="dvc_training_pipeline",
        start_date=pendulum.datetime(2025, 10, 1, tz="UTC"),
        schedule="@weekly",
        catchup=False,
        tags={"mlops", "sentiment", "training"},
        doc_md="DAG to trigger the full DVC pipeline for model retraining.",
) as dag:
    trigger_dvc_pipeline_task = BashOperator(
        task_id="trigger_full_dvc_pipeline_task",
        # This command is simple because all dependencies are in the same environment.
        # It assumes this DAG runs in an environment where the project dependencies
        # (from requirements.txt) are already installed.
        bash_command=(
            f"cd {settings.SENTIMENT_PROJECT_DIR} && "
            "echo '--- DVC PULLING ALL DATA ---' && "
            "dvc pull -v --force && "

            "echo '--- DVC REPRODUCING PIPELINE ---' && "
            "dvc repro evaluate -v --force"
        ),
    )
