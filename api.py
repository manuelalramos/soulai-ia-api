# API de IA do SoulAI (unificada).
# Reune os dois servicos de IA do projeto num unico deploy:
#   - /predict : modelo de Machine Learning que preve o nivel de engajamento (Decision Tree)
#   - /chat    : conversa com o assistente (IBM Watson Assistant)
# As credenciais do Watson vem de variaveis de ambiente, nunca escritas no codigo.

import os
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


# ---------------------------------------------------------------------------
# App + CORS (libera o acesso do site)
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SoulAI - API de IA",
    version="1.0.0",
    description="Reune o modelo de engajamento (ML) e o chat (Watson) num servico so.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# PARTE 1 - MODELO DE ENGAJAMENTO (Machine Learning)
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT = joblib.load(BASE_DIR / "soulia_decision_tree.joblib")
model = ARTIFACT["model"]
FEATURES = ARTIFACT["features"]


class DadosUsuario(BaseModel):
    idade: int = Field(ge=13, le=100, examples=[27])
    dias_ativo: int = Field(ge=0, examples=[120])
    acessos: int = Field(ge=0, examples=[130])
    missoes_recebidas: int = Field(ge=0, examples=[45])
    missoes_concluidas: int = Field(ge=0, examples=[24])
    acoes_sustentaveis: int = Field(ge=0, examples=[25])
    interacoes_chatbot: int = Field(ge=0, examples=[100])
    comentarios: int = Field(ge=0, examples=[20])
    curtidas: int = Field(ge=0, examples=[80])
    participacoes_comunidade: int = Field(ge=0, examples=[8])
    streak_maximo: int = Field(ge=0, examples=[40])
    pontos: int = Field(ge=0, examples=[1800])
    recompensas_resgatadas: int = Field(ge=0, examples=[4])

    @model_validator(mode="after")
    def validar_coerencia(self):
        # Mesmas regras de qualidade validadas na base (notebook, secao 2)
        if self.missoes_concluidas > self.missoes_recebidas:
            raise ValueError("missoes_concluidas nao pode ser maior que missoes_recebidas")
        if self.streak_maximo > self.dias_ativo:
            raise ValueError("streak_maximo nao pode ser maior que dias_ativo")
        return self


@app.post("/predict")
def predict(dados: DadosUsuario):
    try:
        entrada = pd.DataFrame([[getattr(dados, campo) for campo in FEATURES]], columns=FEATURES)
        classe = str(model.predict(entrada)[0])
        probabilidades = model.predict_proba(entrada)[0]
        confianca = {str(rotulo): round(float(prob), 4) for rotulo, prob in zip(model.classes_, probabilidades)}
        return {
            "sucesso": True,
            "nivel_engajamento": classe,
            "modelo": "Decision Tree",
            "probabilidades": confianca,
            "observacao": "Previsao experimental baseada em dados sinteticos; nao representa uma avaliacao definitiva do usuario.",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Nao foi possivel realizar a previsao: {exc}")


# ===========================================================================
# PARTE 2 - CHAT (IBM Watson Assistant)
# ===========================================================================
WATSON_APIKEY = os.getenv("WATSON_APIKEY", "")
WATSON_URL = os.getenv("WATSON_URL", "")
WATSON_ASSISTANT_ID = os.getenv("WATSON_ASSISTANT_ID", "")
WATSON_VERSION = os.getenv("WATSON_VERSION", "2021-11-27")


def criar_assistant():
    autenticador = IAMAuthenticator(WATSON_APIKEY)
    assistant = AssistantV2(version=WATSON_VERSION, authenticator=autenticador)
    assistant.set_service_url(WATSON_URL)
    return assistant


assistant = criar_assistant() if WATSON_APIKEY and WATSON_URL else None


class Mensagem(BaseModel):
    texto: str = Field(min_length=1, examples=["quais sao minhas missoes?"])


@app.post("/chat")
def chat(mensagem: Mensagem):
    if assistant is None or not WATSON_ASSISTANT_ID:
        raise HTTPException(status_code=500, detail="Watson nao configurado no servidor.")
    try:
        # modo stateless: nao precisa criar/guardar sessao
        resposta = assistant.message_stateless(
            assistant_id=WATSON_ASSISTANT_ID,
            input={"message_type": "text", "text": mensagem.texto},
        ).get_result()

        textos = []
        for item in resposta.get("output", {}).get("generic", []):
            if item.get("response_type") == "text":
                textos.append(item.get("text", ""))

        resposta_final = " ".join(textos).strip() or "Desculpe, nao entendi. Pode reformular?"
        return {"sucesso": True, "resposta": resposta_final}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erro ao falar com o Watson: {exc}")


# ===========================================================================
# SAUDE DO SERVICO
# ===========================================================================
@app.get("/health")
def health():
    watson_ok = bool(WATSON_APIKEY and WATSON_URL and WATSON_ASSISTANT_ID)
    return {
        "status": "ok",
        "servico": "SoulAI IA API",
        "modelo_engajamento": "Decision Tree",
        "watson_configurado": watson_ok,
    }