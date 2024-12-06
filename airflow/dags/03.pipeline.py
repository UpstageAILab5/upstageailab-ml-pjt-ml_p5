from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import mlflow
import mlflow.sklearn
import os

# Spotify 데이터 로드 및 MLflow 설정
os.environ['NO_PROXY'] = '*'
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("02_Spotify_Music_Recommendation")

default_args = {
    'owner': 'admin',
    'start_date': datetime(2024, 12, 1),
    'retries': 1,
}

# 1. 데이터 준비 함수
def prepare_music_data(**context):
    # Spotify 데이터 로드 (샘플 데이터셋 사용)
    music_data = pd.read_csv('music_recommendation_data.csv')  # 데이터 경로 수정 필요
    features = ['danceability', 'energy', 'loudness', 'valence', 'tempo']
    X = music_data[features]
    song_ids = music_data['track_id']  # 추천에 필요한 식별자
    
    # XCom으로 데이터 전달
    context['ti'].xcom_push(key='features', value=features)
    context['ti'].xcom_push(key='X', value=X.to_json())
    context['ti'].xcom_push(key='song_ids', value=song_ids.to_list())

# 2. 모델 학습 함수
def train_cosine_similarity_model(**context):
    ti = context['ti']
    X = pd.read_json(ti.xcom_pull(key='X'))

    # 코사인 유사도 기반 추천 모델 학습 (NearestNeighbors)
    model = NearestNeighbors(metric='cosine')
    model.fit(X)
    
    # 모델 저장 및 MLflow 로깅
    with mlflow.start_run(run_name="Cosine_Similarity"):
        mlflow.sklearn.log_model(model, "Cosine_Similarity_Model")
        model_path = '/tmp/cosine_similarity_model.pkl'
        mlflow.log_artifact(model_path)
        context['ti'].xcom_push(key='cosine_model_path', value=model_path)

# 3. 모델 평가 함수
def evaluate_model(**context):
    ti = context['ti']
    model_path = ti.xcom_pull(key='cosine_model_path')

    # 로드된 모델로 평가 수행
    model = joblib.load(model_path)
    X = pd.read_json(ti.xcom_pull(key='X'))
    
    # 코사인 유사도로 평가 (예: 모든 샘플에 대해 추천 생성 및 점검)
    distances, indices = model.kneighbors(X, n_neighbors=5)
    average_distance = distances.mean()

    # MLflow 로깅
    with mlflow.start_run(run_name="Model_Evaluation"):
        mlflow.log_metric("average_distance", average_distance)

    context['ti'].xcom_push(key='average_distance', value=average_distance)

# 4. Slack 메시지 전송 함수
def send_slack_notification(**context):
    ti = context['ti']
    avg_distance = ti.xcom_pull(key='average_distance')

    message = (
        f"🎵 **Music Recommendation Model Pipeline Complete!**\n\n"
        f"📈 **Average Distance:** {avg_distance}\n"
        f"✨ Check the MLflow UI for more details."
    )
    
    slack_notification = SlackWebhookOperator(
        task_id='send_slack_notification_task',
        slack_webhook_conn_id='slack_webhook',
        message=message,
        dag=context['dag']
    )
    slack_notification.execute(context=context)

# DAG 정의
dag = DAG(
    'spotify_music_recommendation_pipeline',
    default_args=default_args,
    description='Music recommendation pipeline with MLflow logging and Slack notifications',
    schedule_interval='@daily',
    catchup=False
)

# Task 정의
prepare_data_task = PythonOperator(
    task_id='prepare_music_data',
    python_callable=prepare_music_data,
    provide_context=True,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_cosine_similarity_model',
    python_callable=train_cosine_similarity_model,
    provide_context=True,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    provide_context=True,
    dag=dag,
)

slack_notification_task = PythonOperator(
    task_id='send_slack_notification',
    python_callable=send_slack_notification,
    provide_context=True,
    dag=dag
)

# Task 의존성 설정
prepare_data_task >> train_model_task >> evaluate_model_task >> slack_notification_task