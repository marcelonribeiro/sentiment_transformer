#!/bin/sh
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration Paths ---
CONFIG_FILE="/etc/mlflow/auth_config.ini"
CONFIG_DIR=$(dirname "$CONFIG_FILE")
MLRUNS_DIR="/app/mlruns"
AUTH_DB_FILE="${MLRUNS_DIR}/mlflow_auth.db"

# --- Generate Auth Config File ---
echo "Generating MLflow auth config file at ${CONFIG_FILE}..."
mkdir -p "$CONFIG_DIR"

# Check required environment variables for auth config
if [ -z "$MLFLOW_TRACKING_USERNAME" ] || [ -z "$MLFLOW_TRACKING_PASSWORD" ]; then
  echo "Error: MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD must be defined." >&2
  exit 1
fi

# Create the .ini file for SQLite authentication
cat <<EOF > "$CONFIG_FILE"
[mlflow]
database_uri = sqlite:///${AUTH_DB_FILE}
default_permission = EDIT
admin_username = ${MLFLOW_TRACKING_USERNAME}
admin_password = ${MLFLOW_TRACKING_PASSWORD}
EOF

# Set permissions for the config file (readable by owner only, usually airflow)
chown airflow:0 "$CONFIG_FILE"
chmod 600 "$CONFIG_FILE"

echo "Auth config file ${CONFIG_FILE} generated successfully:"
cat "$CONFIG_FILE" # Display generated file for debugging

# --- Start MLflow Server ---
echo "Starting MLflow server..."
# Use exec to replace the script process with the mlflow server process
exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "${APP_DATABASE_URL}" \
    --app-name basic-auth \
    --default-artifact-root "/app/mlruns"