FROM python:3.8-slim

# 시스템 패키지 업데이트 및 필수 라이브러리 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app

# 필요한 라이브러리 설치
RUN pip install --no-cache-dir \
    mlflow \
    pandas \
    scikit-learn \
    spotipy \
    numpy

# MLflow 서버 실행
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]