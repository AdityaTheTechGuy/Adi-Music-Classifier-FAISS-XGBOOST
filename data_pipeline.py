import pandas as pd
import re
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

GENRE_MAP = {
    'metal': 'Metal', 'grindcore': 'Metal', 'black-metal': 'Metal', 'heavy-metal': 'Metal',
    'pop': 'Pop', 'indie-pop': 'Pop', 'k-pop': 'Pop', 'mandopop': 'Pop',
    'rock': 'Rock', 'hard-rock': 'Rock', 'punk-rock': 'Rock', 'psych-rock': 'Rock',
    'house': 'Electronic', 'techno': 'Electronic', 'trance': 'Electronic', 'edm': 'Electronic', 'idm': 'Electronic',
    'classical': 'Classical', 'opera': 'Classical',
    'indian': 'Indian', 'groove': 'Indian',
    'hip-hop': 'Hip-Hop', 'rap': 'Hip-Hop',
    'jazz': 'Jazz', 'blues': 'Jazz',
    'country': 'Country', 'bluegrass': 'Country',
    'reggae': 'Reggae', 'dubstep': 'Reggae',
    'folk': 'Folk', 'acoustic': 'Folk'
}

def map_to_macro_genre(micro_genre):
    if pd.isna(micro_genre): return 'Other'
    for keyword, macro in GENRE_MAP.items():
        if keyword in micro_genre.lower(): return macro
    return 'Other'

def load_and_clean_base_data(csv_path):
    df = pd.read_csv(csv_path)
    df['track_name_norm'] = df['track_name'].str.replace(r'\s*[\(\[][^\]\)]*[\)\]]', '', regex=True).str.strip().str.lower()
    df['artists_norm'] = df['artists'].str.lower().str.strip()
    df = df.drop_duplicates(subset=['track_name_norm', 'artists_norm'], keep='first')
    df['macro_genre'] = df['track_genre'].apply(map_to_macro_genre)
    return df

def build_preprocessing_pipeline():
    scale_features = ['tempo', 'loudness']
    passthrough_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'speechiness', 'valence']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', MinMaxScaler(), scale_features),
            ('passthrough', 'passthrough', passthrough_features)
        ],
        remainder='drop'
    )
    return preprocessor

def run_data_pipeline(csv_path='dataset.csv'):
    df = load_and_clean_base_data(csv_path)
    preprocessor = build_preprocessing_pipeline()
    X_hybrid = preprocessor.fit_transform(df)
    
    scale_features = ['tempo', 'loudness']
    passthrough_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'speechiness', 'valence']
    all_engineered_features = scale_features + passthrough_features
    
    df_cleaned = pd.DataFrame(X_hybrid, columns=all_engineered_features, index=df.index)
    return df, df_cleaned, all_engineered_features