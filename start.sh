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

echo ">>> Script finished."