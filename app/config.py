from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Secrets: required. App fails fast on startup if any is missing. ---
    QDRANT_URL: str
    QDRANT_API_KEY: str
    GROQ_API_KEY: str
    GEMINI_API_KEY: str
    DEMO_API_KEY: str

    # --- Model IDs: pinned in Phase 0, live in exactly ONE place ---
    EMBED_MODEL: str = "intfloat/multilingual-e5-small"
    LLM_PRIMARY: str = "openai/gpt-oss-120b"
    LLM_FALLBACK: str = "gemini-3.5-flash-lite"

    # --- Limits & knobs: from Phase 0 / Phase 1 decisions ---
    MAX_FILE_MB: int = 10
    MAX_PAGES: int = 50
    TOP_K: int = 5
    SCORE_THRESHOLD: float = 0.75          # [CALIBRATE] via golden dataset
    RATE_LIMIT_PER_MIN: int = 5
    SESSION_MAX_QUESTIONS: int = 25
    SESSION_TOKEN_BUDGET: int = 80_000
    DAILY_MAX_QUESTIONS: int = 1000
    SESSION_TTL_MINUTES: int = 30

settings = Settings()

