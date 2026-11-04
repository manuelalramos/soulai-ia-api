import copy
import json

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from api import create_app
from auth import AUDIENCE, ISSUER, Identity, issue_token
from clients import ItemSelector, JavaClient, ServiceError, WatsonClient
from config import Settings
from engagement import DadosUsuario, predict_engagement
from sessions import SessionStore


INDICATORS = {
    "idade": 27, "dias_ativo": 120, "acessos": 130, "missoes_recebidas": 45,
    "missoes_concluidas": 24, "acoes_sustentaveis": 25, "interacoes_chatbot": 100,
    "comentarios": 20, "curtidas": 80, "participacoes_comunidade": 8,
    "streak_maximo": 40, "pontos": 1800, "recompensas_resgatadas": 4,
}


class FakeWatson:
    def __init__(self):
        self.calls = []
        self.fail = False

    def message(self, text, context, user_id):
        self.calls.append((text, copy.deepcopy(context), user_id))
        if self.fail:
            raise ServiceError("watson")
        turn = context.get("turn", 0) + 1
        return {"context": {"turn": turn}, "output": {
            "generic": [{"response_type": "text", "text": f"Conversa {turn}"}],
            "intents": [{"intent": "inventario"}] if text == "meus pertences" else [],
        }}


@pytest.fixture
def setup():
    settings = Settings(auth_secret="somente-para-testes-" * 3, cors_origins=("https://site.example",))
    catalog = [{"idAcessorio": 10, "nome": "Óculos", "preco": 50},
               {"idAcessorio": 11, "nome": "Chapéu", "preco": 120}]
    missions = [
        {"idMissao": 1, "tituloMissao": "Economizar água", "valorMoedasMissao": 30, "ativo": "S"},
        {"idMissao": 2, "tituloMissao": "Pedalar", "valorMoedasMissao": 60, "ativo": "S"},
        {"idMissao": 3, "tituloMissao": "Jardim", "valorMoedasMissao": 80, "ativo": "N"},
        {"idMissao": 4, "tituloMissao": "Antiga", "descricaoMissao": "Encerrada", "valorMoedasMissao": 100, "ativo": "S"},
        {"idMissao": 5, "tituloMissao": "Reciclar", "valorMoedasMissao": 20, "ativo": "S"},
    ]
    progress = [{"idUsuario": 7, "idMissao": 2, "missaoConcluida": "S", "moedasGanhasMissao": 60},
                {"idUsuario": 7, "idMissao": 1, "missaoConcluida": "N", "progressoMissao": 30}]
    inventory = [{"idUsuario": 7, "idAcessorio": 11, "equipado": "S"}]
    calls, overrides = [], {}

    def upstream(request):
        path, method = request.url.path, request.method
        body = json.loads(request.content) if request.content and request.content != b"null" else None
        calls.append((method, path, body))
        if (method, path) in overrides:
            response = overrides[(method, path)]
            if isinstance(response, Exception):
                raise response
            return response
        if method == "POST" and path == "/usuario/login":
            if body == {"email": "pessoa@example.com", "senha": "senha-de-teste"}:
                return httpx.Response(200, json={"idUsuario": 7, "nome": "Pessoa", "senha": "nao-repassar"})
            return httpx.Response(401, json={"detail": "invalid"})
        if path in {"/usuario/7", "/usuario/8"}:
            return httpx.Response(200, json={"idUsuario": int(path.rsplit("/", 1)[1]),
                                             "nome": "Pessoa", "saldoMoedas": 10, "nivelOfensiva": 4})
        if method == "GET":
            data = {"/acessorio": catalog, "/missao": missions, "/missao/ativas": missions,
                    "/usuario/7/missoes": progress, "/usuario/7/inventario": inventory,
                    "/avatar/usuario/7": {"idAvatar": 20, "nome": "Sol"}}
            if path in data:
                return httpx.Response(200, json=data[path])
        if method in {"POST", "PUT"} and path.startswith("/usuario/7/"):
            parts = path.split("/")
            key = "idMissao" if parts[3] == "missoes" else "idAcessorio"
            return httpx.Response(200, json={"idUsuario": 7, key: int(parts[4]),
                                             "missaoConcluida": "S", "moedasGanhasMissao": 30,
                                             "equipado": "N" if path.endswith("/desequipar") else "S"})
        return httpx.Response(404, json={})

    java = JavaClient(settings, transport=httpx.MockTransport(upstream))
    watson = FakeWatson()
    selector = ItemSelector(settings)
    app = create_app(settings, java=java, watson=watson, selector=selector)
    with TestClient(app) as client:
        yield {"client": client, "calls": calls, "watson": watson, "overrides": overrides,
               "settings": settings, "catalog": catalog, "missions": missions,
               "inventory": inventory, "progress": progress}


def send(setup, text, logged=True, **fields):
    headers = {}
    if logged:
        headers["Authorization"] = "Bearer " + issue_token(Identity(7, "Pessoa"), setup["settings"])
    return setup["client"].post("/webchat/message", json={"text": text, "sessionId": "conversa", **fields}, headers=headers)


def test_login_uses_java_and_returns_only_sanitized_identity(setup):
    response = setup["client"].post("/auth/login", json={"email": "pessoa@example.com", "senha": "senha-de-teste"})
    assert response.status_code == 200
    body = response.json()
    assert body["usuario"] == {"idUsuario": 7, "nome": "Pessoa"}
    assert "senha" not in response.text and "nao-repassar" not in response.text
    claims = jwt.decode(body["access_token"], setup["settings"].auth_secret, algorithms=["HS256"],
                        issuer=ISSUER, audience=AUDIENCE)
    assert claims["sub"] == "7" and "email" not in claims
    assert setup["watson"].calls == []


def test_invalid_login_never_issues_token(setup):
    response = setup["client"].post("/auth/login", json={"email": "pessoa@example.com", "senha": "errada"})
    assert response.status_code == 401 and "access_token" not in response.text


def test_login_timeout_is_not_bad_password(setup):
    setup["overrides"][("POST", "/usuario/login")] = httpx.ReadTimeout("sensitive upstream data")
    response = setup["client"].post("/auth/login", json={"email": "pessoa@example.com", "senha": "x"})
    assert response.status_code == 502 and "sensitive" not in response.text


def test_forged_user_id_and_other_users_are_rejected(setup):
    assert send(setup, "saldo", logged=False, usuarioId=7).status_code == 401
    assert send(setup, "saldo", usuarioId=8).status_code == 403
    response = setup["client"].post("/chat", json={"texto": "saldo"}, headers={"Authorization": "Bearer forjado"})
    assert response.status_code == 401
    assert setup["calls"] == []


def test_expired_token_is_rejected(setup):
    token = jwt.encode({"sub": "7", "iat": 0, "exp": 1, "jti": "x", "iss": ISSUER, "aud": AUDIENCE},
                       setup["settings"].auth_secret, algorithm="HS256")
    response = setup["client"].post("/chat", json={"texto": "saldo"}, headers={"Authorization": "Bearer " + token})
    assert response.status_code == 401


def test_anonymous_private_request_does_not_call_java(setup):
    body = send(setup, "meu saldo", logged=False).json()
    assert body["metadata"]["auth_required"] is True
    assert setup["calls"] == []


@pytest.mark.parametrize("text,path", [("loja", "/acessorio"), ("missões", "/missao/ativas")])
def test_public_catalogs_work_without_watson(setup, text, path):
    body = send(setup, text, logged=False).json()
    assert body["metadata"]["source"] == "java"
    assert setup["calls"] == [("GET", path, None)]
    assert setup["watson"].calls == []


def test_profile_inventory_avatar_and_completed_missions(setup):
    assert "10 moedas" in send(setup, "qual meu saldo?").json()["reply"]
    assert "Chapéu" in send(setup, "itens equipados").json()["reply"]
    assert "Sol" in send(setup, "avatar").json()["reply"]
    reply = send(setup, "minhas missões concluídas").json()["reply"]
    assert "Pedalar" in reply and "Economizar água" not in reply


@pytest.mark.parametrize("text,method,path,label", [
    ("iniciar missão Reciclar", "POST", "/usuario/7/missoes/5", "Missão iniciada"),
    ("concluir missão Economizar água", "POST", "/usuario/7/missoes/1/concluir", "Missão concluída"),
    ("comprar Óculos", "POST", "/usuario/7/inventario/10", "Acessório comprado"),
    ("equipar Chapéu", "PUT", "/usuario/7/inventario/11/equipar", "Acessório equipado"),
    ("desequipar Chapéu", "PUT", "/usuario/7/inventario/11/desequipar", "Acessório desequipado"),
    ("comprar 10", "POST", "/usuario/7/inventario/10", "Acessório comprado"),
])
def test_operations_use_real_java_contract(setup, text, method, path, label):
    response = send(setup, text)
    assert response.status_code == 200
    assert label in response.json()["reply"]
    assert setup["calls"][-1] == (method, path, None)
    assert response.json()["metadata"]["data_changed"] is True
    assert "usuario" in response.json()["metadata"]["refresh"]


def test_ambiguous_target_does_not_write(setup):
    reply = send(setup, "comprar Óculos e Chapéu").json()["reply"]
    assert "mais de uma" in reply
    assert all(method == "GET" for method, _, _ in setup["calls"])


@pytest.mark.parametrize("text", ["não comprar Óculos", "como comprar Óculos?", "comprar item 10 e item 11"])
def test_negated_explanatory_and_multiple_id_requests_do_not_write(setup, text):
    send(setup, text)
    assert all(method == "GET" for method, _, _ in setup["calls"])


@pytest.mark.parametrize("response", [httpx.Response(409, json={}), httpx.Response(200, json={}),
    httpx.Response(200, json={"idUsuario": 8, "idAcessorio": 10}), httpx.ReadTimeout("timeout")])
def test_failed_or_unconfirmed_write_never_reports_success(setup, response):
    setup["overrides"][("POST", "/usuario/7/inventario/10")] = response
    body = send(setup, "comprar Óculos").json()
    assert body["sucesso"] is False
    assert "Acessório comprado" not in body["reply"]
    assert len([c for c in setup["calls"] if c[0] == "POST"]) == 1


def test_recommendation_excludes_completed_inactive_closed_and_duplicate_missions(setup):
    setup["missions"].append(copy.deepcopy(setup["missions"][0]))
    reply = send(setup, "quais missões preciso para conseguir Óculos?").json()["reply"]
    assert "Faltam 40 moedas" in reply and "chegar a 60 moedas" in reply
    assert "Pedalar" not in reply and "Jardim" not in reply and "Antiga" not in reply
    assert reply.count("🌿 Economizar água") == 1
    assert all(method == "GET" for method, _, _ in setup["calls"])


def test_owned_item_is_not_recommended_for_purchase(setup):
    reply = send(setup, "quais missões preciso para conseguir Chapéu?").json()["reply"]
    assert "já tem Chapéu" in reply


def test_recommendation_does_not_invent_unknown_balance(setup):
    setup["overrides"][("GET", "/usuario/7")] = httpx.Response(200, json={"idUsuario": 7, "saldoMoedas": None})
    body = send(setup, "quais missões preciso para conseguir Óculos?").json()
    assert body["sucesso"] is False and "Faltam" not in body["reply"]


def test_recommendation_when_balance_is_enough(setup):
    setup["overrides"][("GET", "/usuario/7")] = httpx.Response(200, json={"idUsuario": 7, "saldoMoedas": 50})
    reply = send(setup, "quais missões para conseguir Óculos?").json()["reply"]
    assert "moedas suficientes" in reply
    assert all(method == "GET" for method, _, _ in setup["calls"])


def test_recommendation_reports_shortfall_and_no_eligible_missions(setup):
    setup["missions"][:] = [setup["missions"][0]]
    reply = send(setup, "quais missões para conseguir Óculos?").json()["reply"]
    assert "ainda faltarão 10 moedas" in reply
    setup["missions"].clear()
    reply = send(setup, "quais missões para conseguir Óculos?").json()["reply"]
    assert "Não encontrei missões ativas" in reply


def test_engagement_needs_all_real_consistent_indicators(setup):
    assert "não recebi" in send(setup, "meu engajamento").json()["reply"]
    invalid = {**INDICATORS, "missoes_concluidas": 100}
    assert "inconsistentes" in send(setup, "engajamento", dadosEngajamento=invalid).json()["reply"]
    assert "inconsistentes" in send(setup, "engajamento", dadosEngajamento={**INDICATORS, "acessos": True}).json()["reply"]
    assert setup["calls"] == [] and setup["watson"].calls == []


def test_model_and_existing_predict_endpoint(setup):
    result = predict_engagement(DadosUsuario(**INDICATORS))
    assert result["nivel_engajamento"] in {"baixo", "medio", "alto"}
    assert abs(sum(result["probabilidades"].values()) - 1) < 0.001
    assert setup["client"].post("/predict", json=INDICATORS).json() == result
    reply = send(setup, "engajamento", dadosEngajamento=INDICATORS).json()["reply"]
    assert "dados sintéticos" in reply
    assert setup["client"].post("/predict", json={**INDICATORS, "streak_maximo": 500}).status_code == 422


def test_watson_context_is_retained_isolated_and_reset(setup):
    assert send(setup, "o que é sustentabilidade?").json()["reply"] == "Conversa 1"
    assert send(setup, "como vai?").json()["reply"] == "Conversa 2"
    assert send(setup, "o que é sustentabilidade?", logged=False).json()["reply"] == "Conversa 1"
    token = issue_token(Identity(8), setup["settings"])
    body = setup["client"].post("/chat", json={"texto": "o que é sustentabilidade?", "sessionId": "conversa"},
                                 headers={"Authorization": "Bearer " + token}).json()
    assert body["reply"] == "Conversa 1"
    send(setup, "reiniciar conversa")
    assert send(setup, "o que é sustentabilidade?").json()["reply"] == "Conversa 1"


def test_watson_intent_routes_to_java(setup):
    body = send(setup, "meus pertences").json()
    assert "Chapéu" in body["reply"] and body["metadata"]["intent"] == "inventario"


def test_watson_outage_is_friendly_and_does_not_block_catalog(setup):
    setup["watson"].fail = True
    assert send(setup, "o que é sustentabilidade?").json()["sucesso"] is False
    assert send(setup, "loja").json()["sucesso"] is True


@pytest.mark.parametrize("text", ["pessoa@example.com", "minha senha é exemplo"])
def test_credentials_are_never_used_as_chat_authentication_or_sent_to_ai(setup, text):
    body = send(setup, text, logged=False).json()
    assert "login do site" in body["reply"]
    assert setup["calls"] == [] and setup["watson"].calls == []


def test_legacy_chat_aliases_and_empty_message(setup):
    response = setup["client"].post("/chat", json={"texto": "oi"}, headers={"X-Session-Id": "legacy"})
    body = response.json()
    assert body["sessionId"] == "legacy" and body["resposta"] == body["reply"] == body["text"]
    assert "mensagem" in setup["client"].post("/webchat/message", json={}).json()["reply"]


def test_validation_errors_do_not_echo_passwords(setup):
    response = setup["client"].post("/auth/login", json={"email": "x", "senha": "secret-value" * 100})
    assert response.status_code == 422 and "secret-value" not in response.text


def test_cors_allows_only_configured_site(setup):
    headers = {"Origin": "https://site.example", "Access-Control-Request-Method": "POST",
               "Access-Control-Request-Headers": "authorization,content-type"}
    assert setup["client"].options("/chat", headers=headers).headers["access-control-allow-origin"] == "https://site.example"
    headers["Origin"] = "https://unlisted.example"
    assert "access-control-allow-origin" not in setup["client"].options("/chat", headers=headers).headers


def test_session_expiry_and_capacity():
    now = [0]
    store = SessionStore(ttl=60, capacity=1, clock=lambda: now[0])
    with store.conversation("x", 7) as session:
        session.context = {"private": True}
    now[0] = 61
    with store.conversation("x", 7) as session:
        assert session.context == {}
    with store.conversation("y", 7):
        with pytest.raises(RuntimeError):
            with store.conversation("z", 7):
                pass
    with store.conversation("x", None) as session:
        assert session.context == {}


@pytest.mark.parametrize("content", ['{"item_index":99}', '{"item_index":true}', 'not JSON'])
def test_optional_llm_invalid_results_fall_back(content):
    settings = Settings(llm_url="https://llm.example/chat/completions", llm_model="test")
    selector = ItemSelector(settings, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": content}}]})))
    assert selector.select("qual item?", [{"nome": "Óculos"}], None) is None
    selector.close()


def test_optional_llm_selects_only_a_catalog_index_and_sends_no_account_data():
    settings = Settings(llm_url="https://llm.example/chat/completions", llm_model="test")
    def upstream(request):
        payload = json.loads(request.content)
        data = json.loads(payload["messages"][1]["content"])
        assert set(data) == {"pergunta", "catalogo"}
        assert "pessoa@example.com" not in data["pergunta"]
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"item_index":0}'}}]})
    selector = ItemSelector(settings, transport=httpx.MockTransport(upstream))
    assert selector.select("quero o item pessoa@example.com", [{"nome": "Óculos"}], None) == 0
    selector.close()


def test_watson_adapter_returns_previous_context_to_sdk():
    from ibm_cloud_sdk_core.authenticators import NoAuthAuthenticator

    client = WatsonClient(Settings(watson_apikey="chave-ficticia-para-teste",
                                  watson_url="https://watson.invalid", watson_assistant_id="assistente-de-teste"))
    # A autenticacao IAM ocorre em prepare_request, antes de send. Desabilita-la
    # explicitamente garante que este teste do SDK real nao consulte a rede.
    client.assistant.authenticator = NoAuthAuthenticator()
    class Response:
        def get_result(self):
            return {"context": {"turn": 2}, "output": {}}
    def send(request, **kwargs):
        body = json.loads(request["data"])
        assert body["context"] == {"turn": 1}
        assert body["input"]["options"]["return_context"] is True
        assert body["user_id"] == "usuario:7"
        return Response()
    client.assistant.send = send
    assert client.message("oi", {"turn": 1}, "usuario:7")["context"]["turn"] == 2


def test_greetings_and_menu_follow_verified_login(setup):
    anonymous = send(setup, "Oi", logged=False).json()
    assert anonymous["metadata"]["authenticated"] is False
    assert "entre na sua conta" in anonymous["reply"].lower()
    logged = send(setup, "Olá").json()
    assert logged["metadata"]["authenticated"] is True
    assert "Pessoa" in logged["reply"] and "conta está conectada" in logged["reply"]
    assert setup["watson"].calls == [] and setup["calls"] == []


def test_claiming_to_be_logged_in_does_not_end_conversation_or_grant_access(setup):
    anonymous = send(setup, "Mas eu to logada", logged=False).json()
    assert anonymous["metadata"]["auth_required"] is True
    assert anonymous["metadata"]["login_url"] == "/login"
    assert "não recebeu um login válido" in anonymous["reply"]
    logged = send(setup, "Mas eu to logada").json()
    assert "conta está conectada" in logged["reply"]
    assert setup["watson"].calls == [] and setup["calls"] == []


@pytest.mark.parametrize("logged", [True, False])
def test_rewards_are_explained_without_incorrect_registration_prompt(setup, logged):
    body = send(setup, "Recompensas", logged=logged).json()
    assert "moedas" in body["reply"] and "acessórios" in body["reply"]
    assert "cadastro" not in body["reply"].lower()
    assert body["metadata"]["data_changed"] is False
    assert setup["watson"].calls == [] and setup["calls"] == []


def test_mission_catalog_is_personalized_and_read_only(setup):
    body = send(setup, "missões").json()
    assert "Pedalar" not in body["reply"]
    assert "Jardim" not in body["reply"] and "Antiga" not in body["reply"]
    assert "Economizar água" in body["reply"] and "Status: em andamento" in body["reply"]
    assert "iniciar missão Reciclar" in body["reply"]
    assert "iniciar missão Economizar água" not in body["reply"]
    assert body["metadata"]["data_changed"] is False
    assert setup["calls"] == [("GET", "/missao/ativas", None), ("GET", "/usuario/7/missoes", None)]


def test_anonymous_catalog_explains_that_progress_is_not_known(setup):
    body = send(setup, "missões", logged=False).json()
    assert "catálogo" in body["reply"] and "não verifica seu progresso" in body["reply"]
    assert "iniciar missão" not in body["reply"]
    assert body["metadata"]["data_changed"] is False
    assert setup["calls"] == [("GET", "/missao/ativas", None)]


@pytest.mark.parametrize("command,expected", [
    ("iniciar missão Pedalar", "já concluiu"),
    ("concluir missão Pedalar", "já concluiu"),
    ("iniciar missão Economizar água", "já está em andamento"),
    ("concluir missão Reciclar", "ainda não iniciou"),
])
def test_existing_mission_state_prevents_duplicate_or_premature_writes(setup, command, expected):
    body = send(setup, command).json()
    assert expected in body["reply"]
    assert body["metadata"]["data_changed"] is False
    assert all(method == "GET" for method, _, _ in setup["calls"])


def test_completed_or_unverifiable_progress_is_not_recommended_as_new(setup):
    setup["progress"][:] = [{"idMissao": v["idMissao"], "missaoConcluida": "S"} for v in setup["missions"]]
    body = send(setup, "missões").json()
    assert "já concluiu todas" in body["reply"]
    setup["progress"][0]["missaoConcluida"] = None
    body = send(setup, "missões").json()
    assert body["sucesso"] is False and "iniciar missão" not in body["reply"]
    assert all(method == "GET" for method, _, _ in setup["calls"])


def test_recommendation_does_not_restart_pending_mission(setup):
    body = send(setup, "quais missões preciso para conseguir Óculos?").json()
    assert "já iniciou a primeira missão" in body["reply"]
    assert "iniciar missão Economizar água" not in body["reply"]
    assert body["metadata"]["data_changed"] is False
    assert all(method == "GET" for method, _, _ in setup["calls"])
