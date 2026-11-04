"""API SoulIA: chat web migrado do Node-RED, login Java e modelo de engajamento."""

from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, Field, SecretStr

from auth import Identity, issue_token, read_identity
from chatbot import ChatService
from clients import ItemSelector, JavaClient, ServiceError, WatsonClient
from config import Settings
from engagement import DadosUsuario, predict_engagement
from sessions import SessionCapacityError, SessionStore


class Mensagem(BaseModel):
    text: str = Field(default="", max_length=4000,
                      validation_alias=AliasChoices("text", "texto", "message", "input"))
    session_id: str | None = Field(default=None, min_length=1, max_length=128,
                                  validation_alias=AliasChoices("sessionId", "session_id", "chatId"))
    usuario_id: int | None = Field(default=None, gt=0, strict=True,
                                  validation_alias=AliasChoices("usuarioId", "usuario_id", "idUsuario"))
    engagement: Any = Field(default=None, validation_alias="dadosEngajamento")


class Login(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    senha: SecretStr = Field(min_length=1, max_length=256)


def create_app(settings=None, java=None, watson=None, selector=None, predictor=predict_engagement):
    settings = settings or Settings.from_env()
    java = java if java is not None else JavaClient(settings)
    watson = watson if watson is not None else WatsonClient(settings)
    selector = selector if selector is not None else ItemSelector(settings)
    service = ChatService(java, watson, selector, predictor)
    store = SessionStore(settings.session_ttl, settings.max_sessions)

    @asynccontextmanager
    async def lifespan(app):
        yield
        java.close()
        selector.close()

    app = FastAPI(title="SoulAI - API de IA", version="2.0.0", lifespan=lifespan,
                  description="Chat web, operacoes na API Java e previsao de engajamento.")
    app.add_middleware(
        CORSMiddleware, allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Authorization", "X-Session-Id"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Nao refletir senhas ou outros valores recebidos em erros de validacao.
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
        ]})

    @app.post("/auth/login")
    def login(credentials: Login):
        if not settings.auth_configured:
            raise HTTPException(503, "Configure SOULIA_AUTH_SECRET com pelo menos 32 bytes no servidor.")
        try:
            user = java.request("POST", "/usuario/login", {
                "email": credentials.email.strip(), "senha": credentials.senha.get_secret_value(),
            })
        except ServiceError as error:
            if error.status in {400, 401, 403}:
                raise HTTPException(401, "E-mail ou senha inválidos.") from None
            raise HTTPException(502, "Não consegui validar o login na plataforma agora.") from None
        if not isinstance(user, dict) or type(user.get("idUsuario")) is not int or user["idUsuario"] <= 0:
            raise HTTPException(502, "A plataforma não confirmou a identidade do usuário.")
        identity = Identity(user["idUsuario"], str(user.get("nome") or ""))
        return {"access_token": issue_token(identity, settings), "token_type": "bearer",
                "expires_in": settings.auth_ttl,
                "usuario": {"idUsuario": identity.user_id, "nome": identity.name}}

    @app.post("/chat")
    @app.post("/webchat/message")
    def chat(message: Mensagem, authorization: str | None = Header(default=None),
             x_session_id: str | None = Header(default=None, max_length=128)):
        identity = read_identity(authorization, settings)
        if message.usuario_id is not None:
            if identity is None:
                raise HTTPException(401, "Entre na sua conta. usuarioId sozinho não comprova identidade.")
            if message.usuario_id != identity.user_id:
                raise HTTPException(403, "O usuário informado não corresponde ao login.")
        session_id = message.session_id or x_session_id or str(uuid4())
        try:
            with store.conversation(session_id, identity.user_id if identity else None) as conversation:
                answer = service.reply(message.text.strip(), conversation, identity, session_id, message.engagement)
        except SessionCapacityError:
            raise HTTPException(503, "O chat está ocupado. Tente novamente em instantes.") from None
        answer["metadata"]["authenticated"] = identity is not None
        return {"sessionId": session_id, "sucesso": not answer["metadata"].get("unavailable", False),
                "resposta": answer["reply"], **answer}

    @app.post("/predict")
    def predict(dados: DadosUsuario):
        try:
            return predictor(dados)
        except Exception:
            raise HTTPException(400, "Não foi possível realizar a previsão.") from None

    @app.get("/health")
    def health():
        # Liveness: nao executa chamadas pagas nem verifica credenciais na IBM.
        return {"status": "ok", "servico": "SoulAI IA API", "modelo_engajamento": "Decision Tree",
                "watson_configurado": settings.watson_configured, "login_configurado": settings.auth_configured}

    return app


app = create_app()
