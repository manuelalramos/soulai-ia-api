from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT = joblib.load(BASE_DIR / "soulia_decision_tree.joblib")
model = ARTIFACT["model"]
FEATURES = ARTIFACT["features"]

app = FastAPI(title="SoulIA - API de Classificação de Engajamento", version="1.0.0", description="Prevê a faixa de engajamento a partir de indicadores comportamentais.")

# Libera o acesso do front-end (site) a esta API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/health")
def health():
    return {"status": "ok", "servico": "SoulIA ML API", "modelo": "Decision Tree"}

@app.post("/predict")
def predict(dados: DadosUsuario):
    try:
        entrada = pd.DataFrame([[getattr(dados, campo) for campo in FEATURES]], columns=FEATURES)
        classe = str(model.predict(entrada)[0])
        probabilidades = model.predict_proba(entrada)[0]
        confianca = {str(rotulo): round(float(prob), 4) for rotulo, prob in zip(model.classes_, probabilidades)}
        return {"sucesso": True, "nivel_engajamento": classe, "modelo": "Decision Tree", "probabilidades": confianca, "observacao": "Previsao experimental baseada em dados sinteticos; nao representa uma avaliacao definitiva do usuario."}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Nao foi possivel realizar a previsao: {exc}")