
# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from scipy.sparse import csr_matrix
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os

# FastAPI 앱 생성
app = FastAPI()

# 사용자 요청 모델 정의
class UserRequest(BaseModel):
    user_id: str

# 사전 학습된 모델 및 매핑 로드
print("Loading models and mappings...")
svd = joblib.load('svd_model.joblib')
mappings = joblib.load('mappings.joblib')

track_id_to_name = mappings['track_id_to_name']
user_to_index = mappings['user_to_index']
track_to_index = mappings['track_to_index']
track_id_unique = mappings['track_id_unique']
print("Models and mappings loaded.")

# SVD의 아이템 잠재 행렬 추출
item_latent_matrix = svd.components_.T  # (n_items, n_components)


load_dotenv()

client_id = os.getenv('SPOTIPY_CLIENT_ID')
client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI')

# Spotify API 인증 부분 수정
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope="user-read-recently-played playlist-read-private"
))


# 추천 생성 함수 정의
def recommend_songs_exclude_listened(temp_user_vector, listened_tracks, n_recommendations=5):
    """
    사용자가 이미 들은 곡을 제외한 추천 곡 생성 함수.
    """
    scores = np.dot(item_latent_matrix, temp_user_vector)  # (n_items,)
    print(f"점수 분포: {scores}")  # 디버깅용 출력
    track_indices = np.argsort(-scores)  # 점수가 높은 순으로 정렬
    recommended_tracks = []
    for idx in track_indices:
        track_id = track_id_unique[idx]
        if track_id not in listened_tracks:  # 이미 들은 곡 제외
            recommended_tracks.append(track_id)
        if len(recommended_tracks) >= n_recommendations:
            break
    print(f"추천된 트랙: {recommended_tracks}")  # 디버깅용 출력
    return recommended_tracks

# Spotify API에서 사용자의 트랙 가져오기
def get_user_tracks():
    """
    Spotify API에서 사용자의 트랙을 가져오는 함수.
    """
    try:
        my_playlists = sp.current_user_playlists(limit=50)
        my_track_ids = []
        for playlist in my_playlists['items']:
            playlist_tracks = sp.playlist_tracks(playlist_id=playlist['id'])
            for item in playlist_tracks['items']:
                track = item['track']
                if track:
                    my_track_ids.append(track['id'])
        return set(my_track_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data from Spotify: {str(e)}")

# track_id를 trackname으로 변환
def convert_track_id_to_name(recommended_tracks):
    track_names = []
    for track_id in recommended_tracks:
        track_name = track_id_to_name.get(track_id, "Unknown Track")
        track_names.append(track_name)
        print(f"Track ID: {track_id}, Track Name: {track_name}")  # 디버깅용 출력
    return track_names



# API 엔드포인트
@app.post("/recommend/")
def recommend_tracks(request: UserRequest):
    user_id = request.user_id

    # Spotify API로부터 사용자가 들은 트랙 가져오기
    try:
        listened_tracks = get_user_tracks()
        print("Listened Tracks:", listened_tracks)  # 디버깅용 출력
    except HTTPException as e:
        raise e

    # 매핑에 없는 트랙을 무시하고, 이미 매핑된 트랙만 사용
    unseen_tracks = [track for track in listened_tracks if track not in track_to_index]
    if unseen_tracks:
        print(f"Unseen tracks (not in mapping and will be ignored): {unseen_tracks}")

    # 매핑된 트랙 생성: only include tracks that are already in track_to_index
    mapped_tracks = [track_to_index[track] for track in listened_tracks if track in track_to_index]
    print("Mapped Tracks:", mapped_tracks)  # 디버깅용 출력
    print("데이터셋의 트랙 ID 예시:", list(track_to_index.keys())[:5])
    print("Spotify에서 가져온 트랙 ID 예시:", list(listened_tracks)[:5])


    # 매핑된 트랙이 없을 경우 에러 반환
    if not mapped_tracks:
        raise HTTPException(status_code=400, detail="No valid tracks found in user's listened tracks.")

    # 임시 사용자-아이템 상호작용 행렬 생성
    try:
        temp_user_interactions = csr_matrix(
            (np.ones(len(mapped_tracks)),
             ([0] * len(mapped_tracks), mapped_tracks)),
            shape=(1, len(track_id_unique))
        )

        # 사용자 벡터 생성
        temp_user_vector = svd.transform(temp_user_interactions)[0]
        print("Temporary User Vector:", temp_user_vector)  # 벡터 확인
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Error in user vector generation: {str(e)}")

    # 추천 곡 생성
    try:
        recommended_track_ids = recommend_songs_exclude_listened(temp_user_vector, listened_tracks, n_recommendations=5)
        recommended_track_names = convert_track_id_to_name(recommended_track_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

    return {"recommended_tracks": recommended_track_names}


