from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # App Secret Key for securing user session cookies/state tokens
    app_secret_key: str = Field(default="super-secret-dev-key-change-in-production")

    # Spotify API Credentials
    spotify_client_id: str
    spotify_client_secret: str
    spotify_redirect_uri: str = Field(default="http://127.0.0.1:5500/auth/spotify/callback")

    # YouTube / Google API Credentials
    youtube_api_key: str
    google_client_id: str
    google_client_secret: str 
    google_redirect_uri: str = Field(default="http://127.0.0.1:5500/auth/youtube/callback")

    # Automatically load values from a .env file located in the root directory
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Instantiating the settings object to be imported globally
settings = Settings()