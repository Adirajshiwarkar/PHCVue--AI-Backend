from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Manages all application configuration from environment variables."""
    
    # Your live MongoDB connection string should be in the .env file
    # This is a safe default for local development.
    MONGO_URI: str = "mongodb+srv://info:xlhQRSPwz0RwzmXD@cluster0.rmyuiop.mongodb.net/MNR?retryWrites=true&w=majority&appName=Cluster0"
    DATABASE_NAME: str = "PHCVue"

    # OpenAI API Key and Model Configuration
    OPENAI_API_KEY: Optional[str] = "YOUR_OPENAI_API_KEY"
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TTS_MODEL: str = "tts-1"
    OPENAI_WHISPER_MODEL: str = "whisper-1" # <-- ADDED to fix the error

    # AWS Credentials for Transcribe Streaming
    AWS_ACCESS_KEY_ID: Optional[str] = "YOUR_AWS_ACCESS_KEY_ID"
    AWS_SECRET_ACCESS_KEY: Optional[str] = "YOUR_AWS_SECRET_ACCESS_KEY"
    AWS_REGION: str = "us-east-1"

    class Config:
        env_file = ".env"

settings = Settings()