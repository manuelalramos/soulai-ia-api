"""Modelo de engajamento existente, compartilhado por /predict e pelo chat."""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from pydantic import BaseModel, Field, model_validator


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
        if self.missoes_concluidas > self.missoes_recebidas:
            raise ValueError("missoes_concluidas nao pode ser maior que missoes_recebidas")
        if self.streak_maximo > self.dias_ativo:
            raise ValueError("streak_maximo nao pode ser maior que dias_ativo")
        return self


@lru_cache(maxsize=1)
def load_artifact():
    return joblib.load(Path(__file__).resolve().parent / "soulia_decision_tree.joblib")


def predict_engagement(dados: DadosUsuario):
    artifact = load_artifact()
    model, features = artifact["model"], artifact["features"]
    entrada = pd.DataFrame([[getattr(dados, campo) for campo in features]], columns=features)
    classe = str(model.predict(entrada)[0])
    probabilities = model.predict_proba(entrada)[0]
    return {
        "sucesso": True,
        "nivel_engajamento": classe,
        "modelo": "Decision Tree",
        "probabilidades": {str(label): round(float(prob), 4)
                           for label, prob in zip(model.classes_, probabilities)},
        "observacao": "Previsao experimental baseada em dados sinteticos; nao representa uma avaliacao definitiva do usuario.",
    }
