FROM apache/airflow:2.7.3

# 루트 사용자로 전환
USER root

# 필수 시스템 패키지 설치 및 캐시 제거
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 기본 airflow 사용자로 복구
USER airflow

# 추가 Python 라이브러리 설치
RUN pip install --no-cache-dir \
    apache-airflow-providers-slack \
    pandas \
    scikit-learn \
    requests \
    numpy \
    spotipy \
    mlflow
