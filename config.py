"""Configuracao dos servicos; credenciais sao lidas somente do ambiente."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    java_url: str = "https://soulai-api.onrender.com"
    watson_apikey: str = ""
    watson_url: str = ""
    watson_assistant_id: str = ""
    watson_version: str = "2021-11-27"
    auth_secret: str = ""
    auth_ttl: int = 3600
    session_ttl: int = 1800
    max_sessions: int = 1000
    request_timeout: float = 30
    cors_origins: tuple[str, ...] = ()
    llm_url: str = ""
    llm_model: str = ""
    llm_apikey: str = ""

    @classmethod
    def from_env(cls):
        return cls(
            java_url=os.getenv("SOULIA_JAVA_URL", cls.java_url).strip().rstrip("/"),
            watson_apikey=os.getenv("WATSON_APIKEY") or os.getenv("WATSON_ASSISTANT_APIKEY", ""),
            watson_url=os.getenv("WATSON_URL") or os.getenv("WATSON_ASSISTANT_URL", ""),
            watson_assistant_id=os.getenv("WATSON_ASSISTANT_ID", ""),
            watson_version=os.getenv("WATSON_VERSION", cls.watson_version),
            auth_secret=os.getenv("SOULIA_AUTH_SECRET", ""),
            auth_ttl=max(60, int(os.getenv("SOULIA_AUTH_TTL_SECONDS", "3600"))),
            session_ttl=max(60, int(os.getenv("SOULIA_SESSION_TTL_SECONDS", "1800"))),
            max_sessions=max(1, int(os.getenv("SOULIA_MAX_SESSIONS", "1000"))),
            request_timeout=max(1, float(os.getenv("SOULIA_HTTP_TIMEOUT_SECONDS", "30"))),
            cors_origins=tuple(v.strip().rstrip("/") for v in os.getenv(
                "SOULIA_CORS_ORIGINS", "https://soul-ai-projeto.vercel.app,http://localhost:3000,http://localhost:5173"
            ).split(",") if v.strip()),
            llm_url=os.getenv("SOULIA_LLM_URL", "").strip(),
            llm_model=os.getenv("SOULIA_LLM_MODEL", "").strip(),
            llm_apikey=os.getenv("SOULIA_LLM_API_KEY", ""),
        )

    @property
    def watson_configured(self):
        return bool(self.watson_apikey and self.watson_url and self.watson_assistant_id)

    @property
    def auth_configured(self):
        return len(self.auth_secret.encode("utf-8")) >= 32
