import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database Settings
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "moneyball_db"
    POSTGRES_USER: str = "moneyball"
    POSTGRES_PASSWORD: str = "moneyball_secret"

    DATABASE_URL: str = "postgresql+asyncpg://moneyball:moneyball_secret@localhost:5432/moneyball_db"
    DATABASE_URL_SYNC: str = "postgresql+psycopg://moneyball:moneyball_secret@localhost:5432/moneyball_db"

    # API & UI Settings
    API_PORT: int = 8000
    UI_PORT: int = 8501

    # Gemini LLM Settings
    GEMINI_API_KEY: Optional[str] = None
    MODEL_NAME: str = "gemini-3.8-flash"

    # Environment
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"


settings = Settings()
