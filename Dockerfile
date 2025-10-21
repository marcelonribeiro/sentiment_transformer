FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y python3.12 python3-pip python3.12-dev build-essential libpq-dev curl gnupg && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

ENV APP_DIR=/app
WORKDIR ${APP_DIR}
ENV AIRFLOW_HOME=${APP_DIR}/airflow
ENV PYTHONPATH=${APP_DIR}

ARG AIRFLOW_USER_ID=50000

RUN groupadd --gid ${AIRFLOW_USER_ID} airflow && \
    useradd --uid ${AIRFLOW_USER_ID} --gid 0 --create-home airflow

# 1. Copy requirements to install dependencies first (leverages Docker cache)
COPY --chown=airflow:0 requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt
RUN playwright install-deps && playwright install

# 2. Copy the application source and specific required assets
COPY --chown=airflow:0 src/ ./src/
COPY --chown=airflow:0 airflow/dags/ ./airflow/dags/
COPY --chown=airflow:0 dvc.yaml .
COPY --chown=airflow:0 dvc.lock .
COPY --chown=airflow:0 .dvc/ ./.dvc/
COPY --chown=airflow:0 .git/ ./.git/
COPY --chown=airflow:0 data/ ./data/
# Add any other root-level files your project needs here
# COPY other_file.txt .

RUN mkdir -p /app/artifacts \
             /app/metrics \
             /app/data/processed \
             /app/data/embeddings \
             /app/data/raw/sitemaps_infomoney \
             /app/mlruns && \
    touch /app/data/raw/stock_codes.json \
          /app/data/raw/infomoney_news.csv && \
    chown -R airflow:0 /app/artifacts \
                       /app/metrics \
                       /app/data/processed \
                       /app/data/embeddings \
                       /app/data/raw/sitemaps_infomoney \
                       /app/data/raw/stock_codes.json \
                       /app/data/raw/infomoney_news.csv \
                       /app/mlruns && \
    chmod -R u+w /app/data /app/artifacts /app/metrics /app/mlruns

# Switch to the non-root user that now owns everything
USER airflow

# Expose the ports for the services
EXPOSE 8080 8000 8501 5000