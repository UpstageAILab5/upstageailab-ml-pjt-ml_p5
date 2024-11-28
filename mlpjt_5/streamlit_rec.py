import streamlit as st
import joblib
import numpy as np
from scipy.sparse import csr_matrix
from main import recommend_songs_exclude_listened, get_user_tracks, convert_track_id_to_name

# 모델 및 매핑 로드
st.title("스포티파이 음악 추천 서비스")

# Streamlit App Title
st.sidebar.title("음악을 추천해드림")
st.sidebar.write("스포티파이 아이디를 입력해주세요")

# 사용자 입력
user_id = st.sidebar.text_input("User ID", placeholder="Enter your User ID")
recommend_button = st.sidebar.button("입력후 이 버튼을 눌러주세요")

# 모델 및 매핑 로드
@st.cache_resource
def load_models():
    st.write("Loading model and mappings...")
    svd = joblib.load('svd_model.joblib')
    mappings = joblib.load('mappings.joblib')
    return svd, mappings

svd, mappings = load_models()
track_id_to_name = mappings['track_id_to_name']
user_to_index = mappings['user_to_index']
track_to_index = mappings['track_to_index']
track_id_unique = mappings['track_id_unique']
item_latent_matrix = svd.components_.T  # (n_items, n_components)

# 추천 곡을 생성하는 함수
def get_recommendations(user_id):
    try:
        # Spotify API를 통해 사용자의 트랙 가져오기
        listened_tracks = get_user_tracks()
        
        # 매핑된 트랙 필터링
        mapped_tracks = [track_to_index[track] for track in listened_tracks if track in track_to_index]
        
        # 매핑된 트랙이 없을 경우 처리
        if not mapped_tracks:
            return {"error": "No valid tracks found in the user's listened tracks."}
        
        # 사용자 벡터 생성
        temp_user_interactions = csr_matrix(
            (np.ones(len(mapped_tracks)),
             ([0] * len(mapped_tracks), mapped_tracks)),
            shape=(1, len(track_id_unique))
        )
        temp_user_vector = svd.transform(temp_user_interactions)[0]
        
        # 추천 곡 생성
        recommended_track_ids = recommend_songs_exclude_listened(temp_user_vector, listened_tracks, n_recommendations=5)
        recommended_track_names = convert_track_id_to_name(recommended_track_ids)
        
        return {"tracks": recommended_track_names}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# 추천 곡 생성 버튼 클릭 시
if recommend_button:
    if not user_id:
        st.error("Please enter a valid User ID!")
    else:
        st.write(f"Fetching recommendations for User ID: {user_id}...")
        recommendations = get_recommendations(user_id)
        
        if "error" in recommendations:
            st.error(recommendations["error"])
        else:
            st.write("Here are your recommended tracks:")
            for idx, track_name in enumerate(recommendations["tracks"], start=1):
                st.write(f"{idx}. {track_name}")
