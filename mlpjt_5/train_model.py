# train_model.py

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import joblib

def train_and_save_model(data_path='data_15918.csv', model_path='svd_model.joblib',
                        mappings_path='mappings.joblib', latent_matrix_path='latent_matrix.joblib'):
    # 데이터 로드
    print("Loading data...")
    data_final = pd.read_csv(data_path)
    # 필요한 컬럼만 선택할 경우:
    # data_final = pd.read_csv(data_path, usecols=['user_id', 'track_id', 'trackname'])

    # 매핑 생성
    print("Creating mappings...")
    track_id_to_name = pd.Series(data_final['trackname'].values, index=data_final['track_id']).to_dict()
    user_id_unique = data_final['user_id'].unique()
    track_id_unique = data_final['track_id'].unique()
    user_to_index = {user_id: index for index, user_id in enumerate(user_id_unique)}
    track_to_index = {track_id: index for index, track_id in enumerate(track_id_unique)}

    # 사용자-아이템 행렬 생성
    print("Creating interaction matrix...")
    data_final['user_index'] = data_final['user_id'].map(user_to_index)
    data_final['track_index'] = data_final['track_id'].map(track_to_index)
    interaction_matrix = csr_matrix(
        (np.ones(len(data_final)), (data_final['user_index'], data_final['track_index'])),
        shape=(len(user_id_unique), len(track_id_unique))
    )

    # SVD 모델 학습
    print("Training TruncatedSVD model...")
    svd = TruncatedSVD(n_components=50, random_state=42)
    latent_matrix = svd.fit_transform(interaction_matrix)

    # 모델 및 매핑 저장
    print("Saving model and mappings...")
    joblib.dump(svd, model_path)
    joblib.dump({
        'track_id_to_name': track_id_to_name,
        'user_to_index': user_to_index,
        'track_to_index': track_to_index,
        'track_id_unique': track_id_unique
    }, mappings_path)
    joblib.dump(latent_matrix, latent_matrix_path)

    print("Training and saving completed.")

if __name__ == "__main__":
    train_and_save_model()



