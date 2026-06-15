import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Import modular blocks including the new build_faiss_index
from data_pipeline import run_data_pipeline
from engine import train_tuned_xgboost, get_hybrid_recommendations, build_faiss_index

MODEL_PATH = 'xgboost_genre_artifacts.joblib'

def main():
    # 1. Trigger the data engineering pipeline
    print("--- Step 1: Executing Data Pipeline ---")
    df, df_cleaned, all_engineered_features = run_data_pipeline('dataset.csv')
    print(f"Data pipeline complete. Hybrid matrix shape: {df_cleaned.shape}\n")
    
    # 2. Extract features and target strings
    print("--- Step 2: Preparing Features ---")
    best_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'tempo', 'valence']
    X = df_cleaned[best_features]
    y = df['macro_genre']
    
    # 3. Check for existing trained model artifacts
    if os.path.exists(MODEL_PATH):
        print(f"Found saved model artifacts at '{MODEL_PATH}'. Loading...")
        artifacts = joblib.load(MODEL_PATH)
        classifier = artifacts['model']
        label_encoder = artifacts['label_encoder']
        print("Model and Label Encoder loaded successfully.\n")
    else:
        print("No saved model found. Initiating full training sequence...")
        
        # Encode targets dynamically
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Stratified Train-Test Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train the optimized model configuration
        print("--- Step 3: Training Model Engine ---")
        classifier = train_tuned_xgboost(X_train, y_train)
        
        # Verify generalization accuracy on validation set
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Verified Test Accuracy: {accuracy * 100:.2f}%")
        
        # Save artifacts to disk to prevent future retraining
        artifacts = {
            'model': classifier,
            'label_encoder': label_encoder
        }
        joblib.dump(artifacts, MODEL_PATH)
        print(f"Artifacts successfully saved to '{MODEL_PATH}'\n")
    
    # 4. Run the Hybrid Recommendation System with Mock Rejection
    print("--- Step 4: Running Interactive Diversity Engine ---")
    test_song = 'Kesariya (From "Brahmastra")'
    
    # Compile the FAISS index locally for testing
    print("Building FAISS index for local testing...")
    faiss_index = build_faiss_index(df_cleaned, all_engineered_features)
    
    mock_rejections = [] 
    
    # Updated function call with new arguments
    recs = get_hybrid_recommendations(
        song_name=test_song,
        df_meta=df,
        df_vect=df_cleaned,
        index=faiss_index,
        feature_cols=all_engineered_features,
        xgb_model=classifier,
        label_encoder=label_encoder,
        num_recs=5,
        rejected_ids=mock_rejections
    )
    
    if recs:
        print("\n🔥 Standard Matches (Exploitation):")
        for match in recs["standard_matches"]:
            print(f"[{match['index']}] {match['track_name']} | {match['artists']} | {match['track_genre']} | {match['similarity_score']*100:.1f}%")
        
        print("\n🎲 Diverse Matches (Exploration):")
        for match in recs["diverse_matches"]:
            print(f"[{match['index']}] {match['track_name']} | {match['artists']} | {match['track_genre']} | {match['similarity_score']*100:.1f}%")
    else:
        print("Song not found.")

if __name__ == '__main__':
    main()