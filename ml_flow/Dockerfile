FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && apt-get clean

# Install MLflow and PostgreSQL dependencies
RUN pip install --no-cache-dir mlflow psycopg2-binary

# Create a directory for MLflow artifacts
RUN mkdir -p /mlflow/artifacts

# Expose the default MLflow port
EXPOSE 5000

# Command to run MLflow server
CMD ["mlflow", "server", "--backend-store-uri", "postgresql://mlflow_user:password@postgres:5432/mlflow_db", "--default-artifact-root", "/mlflow/artifacts", "--host", "0.0.0.0"]
