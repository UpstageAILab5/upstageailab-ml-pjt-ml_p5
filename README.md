# Spotify 음악 추천 시스템

---

## **팀 구성**  
| 팀원 | 역할 |  
|------|------|  
| 조성지 | 팀장, mlflow 환경설정, 데이터 크롤링, 프론트엔드제작 |  
| 안서인 | 모델링 및 평가지표제작 |  
| 김태환 | 모델링 및 서빙 |  
| 조혜인 | 모델링 |  

---

## **프로젝트 개요**  
Spotify 오디오 특성 데이터를 기반으로 사용자 맞춤형 음악을 추천하는 콘텐츠 기반 추천 시스템을 개발합니다.

### **개발 환경**  
- **언어 및 도구:**  
  - Python 3.9  
  - Jupyter Notebook / VSCode 
- **라이브러리:**  
  - NumPy, pandas, scikit-learn, TensorFlow, Spotipy, mlflow, streamlit, fastapi, apache-airflow 

### **요구 사항**  
- Spotify API 액세스 키  
- Python 필수 라이브러리 설치  
- Spotify 오디오 특성 데이터셋  

---

## **타임라인**  
- **시작일:** 2024년 11월 25일  
- **최종 제출 마감일:** 2024년 12월 7일  

---

## **프로젝트 디렉토리 구조**  
```plaintext

├── project  
│   ├── mlflow  
│   │   └── mlflow_model.ipynb  
│   └── airflow
│   │    └── Dags
│   │          └── airflow_name 
│   └── fastapi/streamlit
│   │    └── mlflow_download_serving.ipynb
├── Input  
│   └── data  
│         ├── track_10000.data 
│         └── user_info.data
├── mlruns
│    └── training_mlflow_id
├── mlflow-artifacts
│    └── training_mlflow_id
│         ├── training_model_usingdata.data 
│         └── model       
├── Dockerfile  
├── docker-compose.yml
├── Readme.md
```
---

## 3. 데이터 설명

### 데이터 개요

- 출처: Spotify API
- 총 10,000곡 이상의 음악 정보
- 주요 특성: acousticness, danceability, energy, tempo, valence, genres 등

### EDA

1. 결측값 탐지 및 제거.
2. 오디오 특성 간 상관관계 분석.
3. 장르별 오디오 특성 분포 시각화.

### Data Processing

- 중복 제거 및 결측값 처리.
- 주요 특성을 기반으로 데이터 라벨링.
- 오디오 특성 값 표준화/정규화

## 4. 모델링

### 선택모델

- KNN
- 컨텐츠 기반 유사도
- SVN

### 코드 설명

아이템-아이템 유사도 기반 추천
오디오 특성을 기반으로 음악 간 유사도를 계산합니다.

```
python
코드 복사
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def create_item_similarity_matrix(data, feature_columns):
    """
    아이템-아이템 유사도 행렬 생성
    :param data: 데이터프레임
    :param feature_columns: 오디오 특성 컬럼 리스트
    :return: 아이템 유사도 데이터프레임
    """
    item_features = data[feature_columns]
    similarity_matrix = cosine_similarity(item_features)
    return pd.DataFrame(
        similarity_matrix, 
        index=data['track_id'], 
        columns=data['track_id']
    )
```

TF-IDF 기반 장르 유사도 추천
텍스트 데이터를 벡터화하여 사용자 선호도에 기반한 추천을 수행합니다.

```
python
코드 복사
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_recommendations_genre_similarity(select_artist, df, top):
    """
    장르 기반 추천 모델
    :param select_artist: 추천 기준 아티스트 리스트
    :param df: 데이터프레임
    :param top: 추천할 곡 수
    :return: 추천 곡 데이터프레임
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['genres'])
    artist_list = []

    for artist in select_artist:
        artist_df = df[df['artist'] == artist.lower().strip()]
        if artist_df.empty:
            print(f"Artist {artist} not found in dataset.")
            continue

        artist_tfidf = tfidf_matrix[artist_df.index.tolist()].mean(axis=0).flatten()
        genre_sim = cosine_similarity([artist_tfidf], tfidf_matrix).flatten()
        similar_indices = np.argsort(-genre_sim)[1:top+1]
        temp = df.iloc[similar_indices]
        temp['genre_similarity'] = genre_sim[similar_indices]
        artist_list.append(temp)

    return pd.concat(artist_list, ignore_index=True) if artist_list else pd.DataFrame()
```

오디오 특성 기반 코사인 유사도 추천
오디오 특성을 활용하여 사용자 선호와 유사한 트랙을 추천합니다.

```
python
코드 복사
from sklearn.metrics.pairwise import cosine_similarity

def recommend_tracks_by_audio_features(track_audio_features_list, df, audio_features_columns, count=10):
    """
    오디오 특성 기반 추천
    :param track_audio_features_list: 추천 기준 트랙 리스트
    :param df: 데이터프레임
    :param audio_features_columns: 오디오 특성 컬럼 리스트
    :param count: 추천할 곡 수
    :return: 추천 결과 데이터프레임
    """
    rec = []
    for _, track_row in track_audio_features_list.iterrows():
        playlist_vector = track_row[audio_features_columns].values.reshape(1, -1)
        similarity_scores = cosine_similarity(playlist_vector, df[audio_features_columns].values).flatten()
        sorted_indices = np.argsort(similarity_scores)[::-1][:count]
        similar_tracks = df.iloc[sorted_indices].copy()
        similar_tracks['cosine_similarity_score'] = similarity_scores[sorted_indices]
        rec.append(similar_tracks)

    return pd.concat(rec, ignore_index=True) if rec else pd.DataFrame()
```

### 모델링 프로세스 

1. 음악 선택/아티스트 선택/현재 유저의 플레이리스트 선택
2. 추천 모델 학습 : 오디오 특성 기반 유사도 계산
3. 결과 검증 : 추천 음악의 유사성평가

## 5. 결과

### 리더보드

- feature 유사도
- f1 score 
- Human Evaluation

### 발표자료
: 링크

## 기타

### 회의록
: 링크

### 참고자료
- Spotify API 문서
- 추천 시스템 관련 논문 및 튜토리얼
