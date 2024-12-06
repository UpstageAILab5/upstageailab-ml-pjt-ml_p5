from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base_hook import BaseHook
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import mlflow.sklearn
from datetime import datetime


# Function for genre-based recommendations
def get_recommendations_genre_similarity(df: pd.DataFrame, sp: SpotifyOAuth, select_artist, top=5):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['genres'])

    artist_list = []
    select_artist = [artist.strip().lower() for artist in select_artist]
    
    for artist in select_artist:
        artist_df = df[df['artist'] == artist]
        
        if artist_df.empty:
            print(f"Artist {artist} not found in the DataFrame.")
            continue
        
        artist_indices = artist_df.index.tolist()        
        artist_tfidf = tfidf_matrix[artist_indices].mean(axis=0)
        artist_tfidf = np.asarray(artist_tfidf).flatten()
        genre_sim = cosine_similarity([artist_tfidf], tfidf_matrix).flatten()
        similar_indices = np.argsort(-genre_sim)[1:top+1]
        temp = df.iloc[similar_indices]
        
        temp = temp[~temp['artist'].isin(select_artist)]
        
        if not temp.empty:
            temp['genre_similarity'] = genre_sim[similar_indices][:len(temp)]
            artist_list.append(temp)
        
    if artist_list:
        result_df = pd.concat(artist_list, ignore_index=True)
        return result_df
    else:
        return pd.DataFrame()

# Function to get user playlists from Spotify
def get_user_playlists(sp):
    return sp.current_user_playlists()['items']

# Function to get recommendations based on the playlist
def get_recommendations_playlist_track_id(df, sp, playlist_id, count=5):
    audio_features_columns = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    track_audio_features_list = get_track_audio_features_list(playlist_id, df)
   
    if len(track_audio_features_list) <= 0:
        return pd.DataFrame()

    track_features = track_audio_features_list[audio_features_columns].values
    playlist_vector = track_features[0].reshape(1, -1)

    similarity_scores = cosine_similarity(playlist_vector, track_features).flatten()
    similar_indices = np.argsort(similarity_scores)[1:count+1]  
    similar_tracks = track_audio_features_list.iloc[similar_indices]
    similar_tracks['cosine_similarity_score'] = similarity_scores[similar_indices]

    return similar_tracks

# Function to handle the recommendation process (Main logic)
def recommend_music(df, sp, select_artist=['lady gaga', 'jimin']):
    user_playlists = get_user_playlists(sp)
    
    if len(user_playlists) <= 0:
        result1 = get_recommendations_genre_similarity(df, sp, select_artist, 10)
        result2 = get_recommendations_genre_similarity(df, sp, select_artist, 10)  # Example for additional artist-based recommendation
        print(result1, result2)
    else:
        for playlist in user_playlists:
            playlist_id = playlist['id']
            result = get_recommendations_playlist_track_id(df, sp, playlist_id, 5)
            print(f"Recommendations for playlist {playlist_id}:")
            print(result)

# Define your Airflow DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 12, 5),  # Set start date as per your requirement
    'retries': 1,
}

dag = DAG(
    'music_recommendation_pipeline',  # Name of the DAG
    default_args=default_args,
    description='A pipeline for generating music recommendations',
    schedule_interval=None,  # Trigger the pipeline manually or with a defined schedule
)

# Task 1: Authenticate and load data (Spotify OAuth)
def authenticate_spotify():
    sp = SpotifyOAuth(client_id="your_client_id", client_secret="your_client_secret", redirect_uri="your_redirect_uri")
    return sp

# Task 2: Data Preprocessing (You could load and preprocess your data here)
def preprocess_data():
    # Load or preprocess data here
    # For example:
    df = pd.read_csv('your_music_data.csv')  # Replace with actual data source
    return df

# Task 3: Generate Music Recommendations
def generate_recommendations():
    df = preprocess_data()  # Example: Assuming data is preloaded
    sp = authenticate_spotify()
    recommend_music(df, sp, select_artist=['lady gaga', 'jimin'])

# Define Airflow tasks
authenticate_spotify_task = PythonOperator(
    task_id='authenticate_spotify',
    python_callable=authenticate_spotify,
    dag=dag,
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

generate_recommendations_task = PythonOperator(
    task_id='generate_recommendations',
    python_callable=generate_recommendations,
    dag=dag,
)

# Set task dependencies (execution order)
authenticate_spotify_task >> preprocess_data_task >> generate_recommendations_task
