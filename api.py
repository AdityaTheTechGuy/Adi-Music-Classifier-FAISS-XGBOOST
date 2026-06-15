from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from data_pipeline import run_data_pipeline
from engine import build_faiss_index, get_hybrid_recommendations, load_xgboost_classifier

app = FastAPI(title="EchoMatch Hyper-Scale Recommendation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GLOBAL APP STATE VARIABLES
df_metadata = None
df_vectors = None
faiss_index = None
engineered_feature_list = []
xgb_model = None
label_encoder = None

@app.on_event("startup")
async def startup_event():
    global df_metadata, df_vectors, faiss_index, engineered_feature_list, xgb_model, label_encoder
    print("📊 Executing data cleaning and high-dimensional TF-IDF transformations...")
    df_metadata, df_vectors, engineered_feature_list = run_data_pipeline('dataset.csv')
    
    print("⚡ Building in-memory FAISS flat vector matrix space...")
    faiss_index = build_faiss_index(df_vectors, engineered_feature_list)
    
    # 👇 NEW: Initialize XGBoost structural parameters right into RAM
    xgb_model, label_encoder = load_xgboost_classifier('xgboost_genre_artifacts.joblib')
    
    print(f"✅ Setup complete. Vector Maps: {faiss_index.ntotal} | Model Loaded: {xgb_model is not None}")

class RecommendationRequest(BaseModel):
    song_name: str
    num_recommendations: int = 5
    rejected_track_ids: list = []

@app.post("/recommend/")
def recommend(request: RecommendationRequest):
    if faiss_index is None:
        raise HTTPException(status_code=503, detail="Search index initialization incomplete.")
        
    recs = get_hybrid_recommendations(
        song_name=request.song_name,
        df_meta=df_metadata,
        df_vect=df_vectors,
        index=faiss_index,
        feature_cols=engineered_feature_list,
        xgb_model=xgb_model,         # Passing global model
        label_encoder=label_encoder, # Passing global decoder
        num_recs=request.num_recommendations,
        rejected_ids=request.rejected_track_ids
    )
    
    if recs is None:
        raise HTTPException(status_code=404, detail="Song not found in matrix parameters.")
    return recs

@app.get("/autocomplete/")
def autocomplete_search(q: str):
    if df_metadata is None:
        return {"suggestions": []}
    if not q or len(q) < 2:
        return {"suggestions": []}
    
    q_lower = q.lower()
    matches = df_metadata[df_metadata['track_name'].str.contains(q, case=False, na=False)]['track_name'].dropna().unique()
    
    def sort_key(track_name):
        name_lower = track_name.lower()
        if name_lower == q_lower:
            return (0, len(name_lower))
        elif name_lower.startswith(q_lower + " ") or name_lower.startswith(q_lower + "(") or name_lower.startswith(q_lower + "-"):
            return (1, len(name_lower))
        elif name_lower.startswith(q_lower):
            return (2, len(name_lower))
        else:
            return (3, len(name_lower))

    sorted_matches = sorted(matches, key=sort_key)
    return {"suggestions": sorted_matches[:7]}