from functools import lru_cache
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    app_name: str = "Stock news"
    debug: bool = False
    environment: str = "development"
    OPENAI_API_KEY: str

    class Config:
        env_file = ".env"
        extra = "ignore"
        case_sensitive = True


@lru_cache()
def get_settings():
    return Settings()
