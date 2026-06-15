from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import numpy as np
import urllib.parse
import httpx 
import os

from data_pipeline import run_data_pipeline
from engine import build_faiss_index, get_hybrid_recommendations, load_xgboost_classifier
from config import settings  

app = FastAPI(title="EchoMatch Hyper-Scale Recommendation Engine")

# 🔒 SECURITY NOTE: For production verification, swap ["*"] out with your explicit HTTPS web domain
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
    
   # 📥 1. Automatically fetch dataset if missing or empty
    if not os.path.exists('dataset.csv') or os.path.getsize('dataset.csv') == 0:
        print("📥 Dataset missing or empty. Downloading from cloud asset storage...")
        DATASET_URL = "https://github.com/AdityaTheTechGuy/Adi-Music-Classifier-FAISS-XGBOOST/releases/download/v1.0-assets/dataset.csv"
        with open('dataset.csv', 'wb') as f:
            with httpx.Client() as client:
                # 🌟 FIXED: Added follow_redirects=True
                f.write(client.get(DATASET_URL, follow_redirects=True).content)
                
    # 📥 2. Automatically fetch model artifacts if missing or empty
    if not os.path.exists('xgboost_genre_artifacts.joblib') or os.path.getsize('xgboost_genre_artifacts.joblib') == 0:
        print("📥 Model artifacts missing or empty. Downloading...")
        MODEL_URL = "https://github.com/AdityaTheTechGuy/Adi-Music-Classifier-FAISS-XGBOOST/releases/download/v1.0-assets/xgboost_genre_artifacts.joblib"
        with open('xgboost_genre_artifacts.joblib', 'wb') as f:
            with httpx.Client() as client:
                # 🌟 FIXED: Added follow_redirects=True
                f.write(client.get(MODEL_URL, follow_redirects=True).content)

    print("📊 Executing data cleaning and high-dimensional TF-IDF transformations...")
    df_metadata, df_vectors, engineered_feature_list = run_data_pipeline('dataset.csv')
    
    print("⚡ Building in-memory FAISS flat vector matrix space...")
    faiss_index = build_faiss_index(df_vectors, engineered_feature_list)
    
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
        xgb_model=xgb_model,        
        label_encoder=label_encoder, 
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

# =====================================================================
# 🔐 AUTHENTICATION ENDPOINTS (SPOTIFY & YOUTUBE PIPELINE)
# =====================================================================

def get_token_from_header(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header indicator.")
    return authorization.split(" ")[1]

@app.get("/auth/spotify/login")
def spotify_login():
    scopes = "playlist-modify-public playlist-modify-private"
    params = {
        "client_id": settings.spotify_client_id,
        "response_type": "code",
        "redirect_uri": settings.spotify_redirect_uri,
        "scope": scopes,
        "show_dialog": "true" 
    }
    url_args = urllib.parse.urlencode(params)
    auth_url = f"https://accounts.spotify.com/authorize?{url_args}"
    return RedirectResponse(auth_url)

@app.get("/auth/spotify/callback")
async def spotify_callback(code: str = None, error: str = None):
    if error:
        raise HTTPException(status_code=400, detail=f"Spotify Authorization Failed: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code from Spotify.")

    token_url = "https://accounts.spotify.com/api/token"
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": settings.spotify_redirect_uri,
        "client_id": settings.spotify_client_id,
        "client_secret": settings.spotify_client_secret,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, data=payload, headers=headers)
        
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to retrieve access token from Spotify.")
        
    token_data = response.json()
    
    # 🌟 FIXED: Target production GitHub Pages endpoint context
    frontend_url = f"https://adityathetechguy.github.io/Adi-Music-Classifier-FAISS-XGBOOST/?spotify_token={token_data['access_token']}"
    return RedirectResponse(frontend_url)
    
class PlaylistRequest(BaseModel):
    playlist_name: str
    track_ids: list[str]

@app.post("/playlist/spotify/create")
async def create_spotify_playlist(request: PlaylistRequest, access_token: str = Depends(get_token_from_header)):
    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        user_profile_url = "https://api.spotify.com/v1/me"
        user_response = await client.get(user_profile_url, headers=headers)
        
        if user_response.status_code != 200:
            raise HTTPException(status_code=user_response.status_code, detail="Failed to fetch profile.")
            
        user_id = user_response.json().get("id")
        
        create_playlist_url = f"https://api.spotify.com/v1/users/{user_id}/playlists"
        playlist_data = {
            "name": request.playlist_name,
            "description": "Custom tracks curated via EchoMatch Engine.",
            "public": False
        }
        playlist_response = await client.post(create_playlist_url, headers=headers, json=playlist_data)
        if playlist_response.status_code not in [200, 201]:
            raise HTTPException(status_code=playlist_response.status_code, detail="Failed to create playlist.")
        
        playlist_id = playlist_response.json().get("id")
        
        if request.track_ids:
            add_tracks_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
            spotify_track_uris = [f"spotify:track:{tid}" for tid in request.track_ids]
            tracks_payload = {"uris": spotify_track_uris}
            
            tracks_response = await client.post(add_tracks_url, headers=headers, json=tracks_payload)
            if tracks_response.status_code not in [200, 201]:
                raise HTTPException(status_code=tracks_response.status_code, detail="Failed to append approved tracks.")

        return {
            "status": "Playlist created and synchronized successfully!",
            "playlist_id": playlist_id,
            "tracks_added_count": len(request.track_ids)
        }

@app.get("/auth/youtube/login")
def youtube_login():
    scopes = "https://www.googleapis.com/auth/youtube"
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": scopes,
        "access_type": "offline",     
        "prompt": "consent"            
    }
    url_args = urllib.parse.urlencode(params)
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{url_args}"
    return RedirectResponse(auth_url)

@app.get("/auth/youtube/callback")
async def youtube_callback(code: str = None, error: str = None):
    if error:
        raise HTTPException(status_code=400, detail=f"Google Authorization Failed: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="Missing code from Google.")

    token_url = "https://oauth2.googleapis.com/token"
    payload = {
        "code": code,
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "redirect_uri": settings.google_redirect_uri,
        "grant_type": "authorization_code"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, data=payload)
        
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to retrieve tokens from Google.")
        
    token_data = response.json()
    
    # 🌟 FIXED: Target production GitHub Pages endpoint context
    frontend_url = f"https://adityathetechguy.github.io/Adi-Music-Classifier-FAISS-XGBOOST/?youtube_token={token_data['access_token']}"
    return RedirectResponse(frontend_url)

class YouTubePlaylistRequest(BaseModel):
    playlist_name: str
    track_names: list[str]

@app.post("/playlist/youtube/create")
async def create_youtube_playlist(request: YouTubePlaylistRequest, access_token: str = Depends(get_token_from_header)):
    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        playlist_url = "https://www.googleapis.com/youtube/v3/playlists?part=snippet,status"
        playlist_payload = {
            "snippet": {
                "title": request.playlist_name,
                "description": "Curated automatically by EchoMatch Hyper-Scale Recommendation Engine"
            },
            "status": {"privacyStatus": "public"}
        }

        playlist_res = await client.post(playlist_url, headers=headers, json=playlist_payload)
        if playlist_res.status_code not in (200, 201):
            raise HTTPException(status_code=playlist_res.status_code, detail="Failed to create YouTube playlist.")

        playlist_id = playlist_res.json().get("id")
        added_count = 0

        for track in request.track_names:
            search_url = "https://www.googleapis.com/youtube/v3/search"
            optimized_query = f"{track} Official Audio"
            search_params = {
                "part": "id",
                "q": optimized_query,
                "type": "video",
                "videoCategoryId": "10",  
                "maxResults": 1,
                "key": settings.youtube_api_key
            }
            
            search_res = await client.get(search_url, params=search_params)
            if search_res.status_code != 200:
                continue

            items = search_res.json().get("items", [])
            if not items:
                continue
                
            video_id = items[0]["id"]["videoId"]

            insert_url = "https://www.googleapis.com/youtube/v3/playlistItems?part=snippet"
            insert_payload = {
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": video_id
                    }
                }
            }

            insert_res = await client.post(insert_url, headers=headers, json=insert_payload)
            if insert_res.status_code in (200, 201):
                added_count += 1

        return {
            "status": "YouTube Music playlist synchronized successfully!",
            "playlist_id": playlist_id,
            "videos_added_count": added_count
        }

class CreateEmptyPlaylistRequest(BaseModel):
    playlist_name: str

class AddSingleTrackRequest(BaseModel):
    playlist_id: str
    track_name: str

@app.get("/playlists/youtube")
async def get_youtube_playlists(access_token: str = Depends(get_token_from_header)):
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {access_token}"}
        url = "https://www.googleapis.com/youtube/v3/playlists?part=snippet&maxResults=50&mine=true"
        res = await client.get(url, headers=headers)
        if res.status_code != 200:
            raise HTTPException(status_code=res.status_code, detail="Failed to fetch account playlists.")
        
        items = res.json().get("items", [])
        return [{"id": item["id"], "title": item["snippet"]["title"]} for item in items]

@app.post("/playlist/youtube/create-empty")
async def create_empty_playlist(request: CreateEmptyPlaylistRequest, access_token: str = Depends(get_token_from_header)):
    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        url = "https://www.googleapis.com/youtube/v3/playlists?part=snippet,status"
        payload = {
            "snippet": {
                "title": request.playlist_name,
                "description": "Curated via EchoMatch Vector Engine"
            },
            "status": {"privacyStatus": "public"}
        }
        res = await client.post(url, headers=headers, json=payload)
        if res.status_code not in (200, 201):
            raise HTTPException(status_code=res.status_code, detail="Failed to provision container.")
        return {"id": res.json().get("id"), "title": request.playlist_name}

@app.post("/playlist/youtube/add-track")
async def add_track_to_playlist(request: AddSingleTrackRequest, access_token: str = Depends(get_token_from_header)):
    async with httpx.AsyncClient() as client:
        # 1. Resolve to Official Audio Link
        search_url = "https://www.googleapis.com/youtube/v3/search"
        search_params = {
            "part": "id", "q": f"{request.track_name} Official Audio",
            "type": "video", "videoCategoryId": "10", "maxResults": 1,
            "key": settings.youtube_api_key
        }
        search_res = await client.get(search_url, params=search_params)
        items = search_res.json().get("items", [])
        if not items:
            raise HTTPException(status_code=404, detail="Track matching signature not found.")
        video_id = items[0]["id"]["videoId"]

        # 2. Inject directly into the targeted Playlist ID
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        insert_url = "https://www.googleapis.com/youtube/v3/playlistItems?part=snippet"
        payload = {
            "snippet": {
                "playlistId": request.playlist_id,
                "resourceId": {"kind": "youtube#video", "videoId": video_id}
            }
        }
        insert_res = await client.post(insert_url, headers=headers, json=payload)
        if insert_res.status_code not in (200, 201):
            raise HTTPException(status_code=insert_res.status_code, detail="Injection mapping failed.")
        return {"status": "success", "video_id": video_id}