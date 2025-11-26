#!/bin/bash
# Script to start the Docker Compose stack, detecting GPU availability

# Base compose file
COMPOSE_FILES="-f docker-compose.yml"

if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
  echo ">>> GPU NVIDIA detectada. Incluindo docker-compose.gpu.yml..."
  COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.gpu.yml"
fi

if [ -f "docker-compose.local.yml" ]; then
  echo ">>> Configuração local detectada. Incluindo docker-compose.local.yml..."
  COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.local.yml"
fi

echo ">>> Executing: docker compose $COMPOSE_FILES up -d $@"
docker compose $COMPOSE_FILES up -d "$@"

echo ">>> Fixing volume permissions..."

sleep 5

docker compose exec -u root -T airflow-standalone bash -c "
    chown -R airflow:0 /app/.dvc &&
    chown -R airflow:0 /dvc-storage &&
    chown -R airflow:0 /app/mlruns &&
    chown -R airflow:0 /app/airflow &&
    chmod -R u+w /app/.dvc /dvc-storage /app/mlruns /app/airflow
"

echo ">>> Permissions fixed! System ready."