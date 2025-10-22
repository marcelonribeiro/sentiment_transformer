#!/bin/bash
# Script to start the Docker Compose stack, detecting GPU availability

# Base compose file
COMPOSE_FILES="-f docker-compose.yml"

# Check if nvidia-smi command exists AND executes successfully (returns 0)
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
  echo ">>> NVIDIA GPU detected. Including docker-compose.gpu.yml configuration..."
  COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.gpu.yml"
else
  echo ">>> No functional NVIDIA GPU detected. Using only docker-compose.yml..."
fi

# Build (if needed) and start containers in the background
# Pass along any extra arguments received by the script (e.g., --build, --force-recreate) to docker compose
echo ">>> Executing: docker compose $COMPOSE_FILES up -d $@"
docker compose $COMPOSE_FILES up -d "$@"

echo ">>> Script finished."