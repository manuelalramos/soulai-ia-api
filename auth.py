"""Tokens do chat sao emitidos apenas depois do login validado pela API Java."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
from fastapi import HTTPException

from config import Settings

ISSUER = "soulia-ia-api"
AUDIENCE = "soulia-webchat"


@dataclass(frozen=True)
class Identity:
    user_id: int
    name: str = ""


def issue_token(identity: Identity, settings: Settings) -> str:
    if not settings.auth_configured:
        raise HTTPException(503, "Configure SOULIA_AUTH_SECRET com pelo menos 32 bytes no servidor.")
    now = datetime.now(timezone.utc)
    return jwt.encode({
        "sub": str(identity.user_id), "name": identity.name,
        "iat": now, "exp": now + timedelta(seconds=settings.auth_ttl),
        "iss": ISSUER, "aud": AUDIENCE, "jti": str(uuid4()),
    }, settings.auth_secret, algorithm="HS256")


def read_identity(authorization: str | None, settings: Settings) -> Identity | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token or not settings.auth_configured:
        raise HTTPException(401, "Login inválido ou expirado. Entre novamente.")
    try:
        claims = jwt.decode(
            token, settings.auth_secret, algorithms=["HS256"],
            issuer=ISSUER, audience=AUDIENCE,
            options={"require": ["sub", "iat", "exp", "iss", "aud", "jti"]},
        )
        user_id = int(claims["sub"])
        if user_id <= 0:
            raise ValueError("invalid subject")
        return Identity(user_id, str(claims.get("name", "")))
    except (jwt.InvalidTokenError, ValueError, TypeError):
        raise HTTPException(401, "Login inválido ou expirado. Entre novamente.") from None
