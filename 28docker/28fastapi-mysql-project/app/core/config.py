from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Project"
    API_V1_STR: str = "/api/v1"
    
    # Database Settings (defined in .env)
    MYSQL_ROOT_PASSWORD: str | None = None
    MYSQL_DATABASE: str | None = None
    MYSQL_USER: str | None = None
    MYSQL_PASSWORD: str | None = None
    DB_HOST: str | None = None
    DB_PORT: int | None = None
    
    DATABASE_URL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"  # 정의되지 않은 환경 변수는 무시 (에러 방지)
    )

settings = Settings()
