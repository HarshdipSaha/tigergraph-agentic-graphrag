from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    tg_host: str = os.getenv("TG_HOST", "")
    tg_graph: str = os.getenv("TG_GRAPH", "OlympicsRAG")
    tg_secret: str = os.getenv("TG_SECRET", "")
    tg_username: str = os.getenv("TG_USERNAME", "")
    tg_password: str = os.getenv("TG_PASSWORD", "")
    llm_provider: str = os.getenv("AGRAG_LLM_PROVIDER", "groq")   # "groq" (free) or "anthropic"
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    model: str = os.getenv("AGRAG_MODEL", "openai/gpt-oss-20b")
    max_tokens: int = int(os.getenv("AGRAG_MAX_TOKENS", "1024"))
    embed_model: str = os.getenv("AGRAG_EMBED_MODEL", "all-MiniLM-L6-v2")


settings = Settings()
