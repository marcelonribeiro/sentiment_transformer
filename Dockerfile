# Use a specific Python version for reproducibility
FROM python:3.10-slim

# Set a single, unified application directory
ENV APP_DIR=/app

# All subsequent commands will run from this directory
WORKDIR ${APP_DIR}

# Set Airflow's home as a subdirectory and add the app root to PYTHONPATH
ENV AIRFLOW_HOME=${APP_DIR}/airflow
ENV PYTHONPATH=${APP_DIR}
ENV PYTHONUNBUFFERED=1

# Install all system-level dependencies as root
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Create the non-root user that will own everything
ARG AIRFLOW_USER_ID=50000
RUN groupadd --gid ${AIRFLOW_USER_ID} airflow && \
    useradd --uid ${AIRFLOW_USER_ID} --gid 0 --create-home airflow

# --- Explicit Copying ---
# Copy only the necessary files and directories. This avoids copying
# local runtime files like airflow.db or logs.

# 1. Copy requirements to install dependencies first (leverages Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install-deps && playwright install

# 2. Copy the application source and specific required assets
COPY src/ ./src/
COPY airflow/dags/ ./airflow/dags/
COPY dvc.yaml .
COPY dvc.lock .
COPY .dvc/ ./.dvc/
COPY .git/ ./.git/
COPY data/ ./data/
# Add any other root-level files your project needs here
# COPY other_file.txt .

# ===================================================================
# THE FINAL PERMISSION FIX:
# After all files are in place, recursively change the ownership
# of the entire application directory to the 'airflow' user.
# ===================================================================
RUN chown -R airflow:0 ${APP_DIR}

# Switch to the non-root user that now owns everything
USER airflow

# Expose the ports for the services
EXPOSE 8080 8000 8501 5000