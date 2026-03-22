import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    gcp_project: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    vertex_location: str = os.getenv("VERTEX_AI_LOCATION", "europe-north1")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview-customtools")
    gemini_fast_model: str = os.getenv("GEMINI_FAST_MODEL", "gemini-3-flash-preview")

    # Local sandbox (used by seed_sandbox.py and local curl tests)
    sandbox_url: str = os.getenv("TRIPLETEX_SANDBOX_URL", "")
    sandbox_token: str = os.getenv("TRIPLETEX_SANDBOX_TOKEN", "")

    api_key: str = os.getenv("API_KEY", "")
    dev_mode: bool = os.getenv("DEV_MODE", "false").lower() == "true"


settings = Settings()
