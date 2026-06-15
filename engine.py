import pandas as pd
import numpy as np
import faiss
import joblib
import os
import xgboost as xgb

def train_tuned_xgboost(X_train, y_train):
    """Initializes and trains the regularized, hyperparameter-optimized XGBoost model."""
    model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=6,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=1,
        random_state=42, eval_metric='mlogloss'
    )
    print("Training optimized XGBoost classifier...")
    model.fit(X_train, y_train)
    print("XGBoost model training locked.")
    return model

# Your 8 baseline audio features used for model training
MODEL_FEATURES = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'tempo', 'valence']

# Your 1,008 total structural columns (TF-IDF + scaling columns)
FEATURES = ['tempo', 'loudness', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'speechiness', 'valence']

def load_xgboost_classifier(model_path='xgboost_genre_artifacts.joblib'):
    """
    Loads saved model and label encoder from disk to classify genre probabilities.
    """
    if os.path.exists(model_path):
        print(f"📦 Loading pre-trained XGBoost artifacts from '{model_path}'...")
        artifacts = joblib.load(model_path)
        return artifacts['model'], artifacts['label_encoder']
    else:
        print("⚠️ Warning: 'xgboost_genre_artifacts.joblib' not found. Run main.py first to save it.")
        return None, None

def build_faiss_index(df_vectors: pd.DataFrame, feature_cols: list):
    """Constructs a fast in-memory L2 vector index."""
    # Ensure we only slice columns that exist in the DataFrame columns index
    valid_cols = [c for c in feature_cols if c in df_vectors.columns]
    matrix_values = df_vectors[valid_cols].values.astype('float32')
    matrix_values = np.ascontiguousarray(matrix_values)
    
    dimensions = matrix_values.shape[1]
    index = faiss.IndexFlatL2(dimensions)
    index.add(matrix_values)
    return index

def get_hybrid_recommendations(song_name: str, df_meta: pd.DataFrame, df_vect: pd.DataFrame, 
                               index: faiss.Index, feature_cols: list, xgb_model, label_encoder,
                               num_recs: int = 5, rejected_ids: list = None):
    if rejected_ids is None:
        rejected_ids = []

    target_rows = df_meta[df_meta['track_name'].str.lower() == song_name.lower()]
    if target_rows.empty:
        return None

    target_idx = target_rows.index[0]
    
    # 1. ⚓ ANCHORS: Grab both exact and macro genres
    actual_macro_genre = target_rows.loc[target_idx, 'macro_genre']
    actual_raw_genre = target_rows.loc[target_idx, 'track_genre'] # The specific sub-genre/language
    
    # Extract raw vectors for FAISS search space
    valid_cols = [c for c in feature_cols if c in df_vect.columns]
    target_vector = df_vect.loc[[target_idx], valid_cols].values.astype('float32')
    target_vector = np.ascontiguousarray(target_vector)

    fetch_limit = min(3000, len(df_meta)) 
    distances, indices = index.search(target_vector, fetch_limit)

    found_indices = indices[0].tolist()
    found_distances = distances[0].tolist()

    candidate_pool = []
    for idx_pos, distance_score in zip(found_indices, found_distances):
        actual_df_id = df_meta.index[idx_pos]

        if actual_df_id == target_idx or actual_df_id in rejected_ids:
            continue
            
        meta_row = df_meta.loc[actual_df_id]
        if song_name.lower() in meta_row['track_name'].lower() and meta_row['track_name'].lower() != song_name.lower():
            continue

        similarity_score = max(0.01, 1.0 - (distance_score / 8.0))

        candidate_pool.append({
            "index": int(actual_df_id),
            "track_name": meta_row['track_name'],
            "artists": meta_row['artists'],
            "macro_genre": meta_row['macro_genre'], 
            "raw_genre": meta_row['track_genre'],
            "similarity_score": float(similarity_score),
            "track_id": meta_row['track_id']
        })

    # --- THE PRODUCTION RE-RANKER (Artist Deduplication) ---
    def enforce_business_logic(matches, required_count):
        final_list = []
        seen_artists = set()
        for match in matches:
            primary_artist = match['artists'].split(';')[0].strip().lower()
            if primary_artist not in seen_artists:
                final_list.append(match)
                seen_artists.add(primary_artist)
            if len(final_list) >= required_count:
                break
        return final_list

    # 1. Standard: Must match the EXACT language/sub-genre (e.g., strictly 'hindi')
    standard_matches = [s for s in candidate_pool if s['raw_genre'] == actual_raw_genre]
    standard_matches = sorted(standard_matches, key=lambda x: x['similarity_score'], reverse=True)
    standard_matches = enforce_business_logic(standard_matches, num_recs)

    # 2. Diverse: Must strictly NOT belong to the parent macro-genre (e.g., absolutely no 'Indian' tracks)
    diverse_matches = [s for s in candidate_pool if s['macro_genre'] != actual_macro_genre]
    diverse_matches = sorted(diverse_matches, key=lambda x: x['similarity_score'], reverse=True)
    diverse_matches = enforce_business_logic(diverse_matches, num_recs)

    # Re-map keys for the frontend UI rendering
    for match in standard_matches + diverse_matches:
        match["track_genre"] = match["raw_genre"]

    return {
        "standard_matches": standard_matches,
        "diverse_matches": diverse_matches
    }