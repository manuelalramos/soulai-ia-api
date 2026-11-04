"""Regras da aba ativa 'SoulIA — Java integrado' do flow Node-RED.

Somente o canal web de texto foi migrado. Identificacao por email foi
substituida pelo login real da API Java; o ID vem exclusivamente do token.
"""

import re
import unicodedata
from dataclasses import dataclass

from pydantic import ValidationError

from auth import Identity
from clients import ServiceError
from engagement import DadosUsuario, predict_engagement


@dataclass(frozen=True)
class ActionResult:
    text: str
    refresh: tuple[str, ...]


def normalize(value):
    return "".join(c for c in unicodedata.normalize("NFD", str(value or ""))
                   if unicodedata.category(c) != "Mn").lower().strip()


def yes(value):
    return value is True or value == 1 or normalize(value) in {"s", "sim", "true", "1"}


def integer(value):
    return type(value) is int and 0 <= value <= 2**53 - 1


def shown(value):
    return "não informado" if value is None or value == "" else str(value)


def money(value):
    return f"{value:,}".replace(",", ".") + " moedas" if integer(value) else "valor não informado"


def classify(text, intent=""):
    if re.search(r"minhas missoes|missoes.*(concluid|pendente|andamento)|quais.*(conclui|terminei)", text):
        return "minhas"
    rules = [
        ("concluir", r"(?:concluir|conclui|terminei|finalizar|finalizei).*missao|^concluir\s+\d+$", {"concluir_missao"}),
        ("iniciar", r"(?:iniciar|aceitar).*missao|^iniciar\s+\d+$", {"iniciar_missao"}),
        ("desequipar", r"\bdesequipar\b", {"desequipar_item"}),
        ("equipar", r"\bequipar\b", {"equipar_item"}),
        ("comprar", r"\bcomprar\b|resgatar.*item", {"resgatar_item", "comprar_item"}),
        ("inventario", r"inventario|meus (?:itens|acessorios)|itens.*equipados", {"inventario"}),
        ("loja", r"loja|catalogo", {"loja"}),
        ("missoes", r"missoes|missao", {"missoes"}),
        ("avatar", r"avatar", {"avatar"}),
        ("perfil", r"saldo|perfil|status|ofensiva|quantas moedas|quanto.*moedas|foguinho", {"menu_principal"}),
    ]
    for action, pattern, intents in rules:
        if re.search(pattern, text) or intent in intents:
            return action
    return ""


def is_goal(text):
    return bool(re.search(r"missao|missoes|misoes|moedas|quanto falta", text)
                and re.search(r"conseguir|comprar|ganhar|obter|adquirir|juntar|preciso|falta|chapeu", text))


class ChatService:
    def __init__(self, java, watson, selector, predictor=predict_engagement):
        self.java, self.watson, self.selector, self.predictor = java, watson, selector, predictor

    def menu(self, identity):
        if identity:
            name = f", {identity.name}" if identity.name else ""
            return self.answer(
                f"Olá{name}! Sua conta está conectada ao chat.\n\n"
                "Você pode pedir:\n• Minhas missões\n• Meu saldo\n• Meu foguinho\n"
                "• Meu inventário\n• Meu avatar\n• Loja\n• Uma dica sustentável\n\n"
                "Também posso ajudar a escolher missões para conseguir um acessório.", action="menu"
            )
        return self.answer(
            "Olá! Sou a SoulIA. Você pode conversar sobre sustentabilidade, pedir uma dica, "
            "ver as missões disponíveis ou consultar a loja.\n\n"
            "Para consultar seu saldo, foguinho, avatar, inventário e missões pessoais, "
            "entre na sua conta no site. Não envie sua senha pelo chat.", action="menu", login_url="/login"
        )

    def reply(self, raw, session, identity: Identity | None, session_id, engagement=None):
        text = normalize(raw)
        if not text:
            return self.answer("Me manda uma mensagem para eu continuar.")
        if text in {"sair", "reiniciar", "reiniciar conversa", "nova conversa", "/start"}:
            session.context.clear()
            if text != "/start":
                return self.answer("Conversa reiniciada. Seu login no site continua ativo." if identity
                                   else "Conversa reiniciada. Digite menu para ver os comandos.")
        greeting = re.sub(r"[!?.,]", "", text).strip()
        if greeting in {"/start", "menu", "ajuda", "oi", "ola", "bom dia", "boa tarde", "boa noite",
                        "oi tudo bem", "ola tudo bem"}:
            return self.menu(identity)
        # Credenciais nunca devem ser enviadas ao Watson ou ao interpretador opcional.
        if re.search(r"[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}", raw) or re.search(
            r"\b(?:senha|password|apikey|api_key)\b", text
        ):
            return self.answer("Use o login do site para identificar sua conta. Não envie sua senha pelo chat.")
        if re.search(r"\b(?:logad[ao]|login|autenticad[ao])\b", text):
            if identity:
                return self.answer("Sua conta está conectada ao chat. Pode pedir seu saldo, foguinho, "
                                   "inventário ou missões pessoais.", action="login_status")
            return self.answer("O chat ainda não recebeu um login válido da sua conta. Entre novamente "
                               "pelo site para conectar o chat. Você pode continuar conversando sobre "
                               "sustentabilidade e consultar a loja e as missões disponíveis.",
                               action="login_status", auth_required=True, login_url="/login")
        if "engajamento" in text:
            return self.engagement(engagement)
        action = "recomendacao" if is_goal(text) else classify(text)
        watson_metadata = {}
        if not action:
            if "recompensa" in text:
                return self.answer("As recompensas disponíveis são as moedas ganhas nas missões e os "
                                   "acessórios da loja. Peça “loja” para ver os itens ou “meu saldo” para "
                                   "consultar suas moedas." + (" Entre na sua conta para consultar seu saldo."
                                   if identity is None else ""), action="recompensas")
            if re.search(r"cadastr|registrar.*acao", text):
                return self.answer("Use o cadastro da plataforma para criar sua conta. Você pode "
                                   "continuar conversando sobre sustentabilidade e consultar missões e loja.")
            try:
                result = self.watson.message(
                    raw, session.context,
                    f"usuario:{identity.user_id}" if identity else f"visitante:{session_id}",
                )
            except ServiceError:
                return self.answer("Não consegui falar com o assistente agora. Você ainda pode pedir "
                                   "missões, loja, inventário, avatar ou saldo.", source="watson", unavailable=True)
            if isinstance(result.get("context"), dict):
                session.context = result["context"]
            output = result.get("output") or {}
            intents = output.get("intents") or result.get("intents") or []
            intent = str(intents[0].get("intent", "")) if intents else ""
            if intent == "menu_principal":
                return self.menu(identity)
            action = classify(text, intent)
            watson_metadata = {"intent": intent} if intent else {}
            if not action:
                texts = []
                for item in output.get("generic", []):
                    if item.get("response_type") == "text":
                        if item.get("text"):
                            texts.append(str(item["text"]))
                        texts.extend(str(v["text"]) for v in item.get("values", []) if v.get("text"))
                return self.answer("\n\n".join(texts) or "Posso consultar missões, loja, inventário, avatar e saldo. "
                                   "Digite menu para ver os comandos.", source="watson", **watson_metadata)
        if action not in {"loja", "missoes"} and identity is None:
            return self.answer("Entre na sua conta no site para eu consultar seus dados e realizar esse pedido.",
                               action=action, auth_required=True, login_url="/login")
        if action in {"iniciar", "concluir", "comprar", "equipar", "desequipar"} and re.search(
            r"\b(?:nao|nunca|evite)\b|\bcomo\b|\bo que acontece\b", text
        ):
            return self.answer("Para realizar uma operação, envie um pedido direto, como “comprar Óculos” "
                               "ou “iniciar missão Economizar água”. Nenhuma operação foi enviada.")
        try:
            if action == "recomendacao":
                answer = self.recommend(raw, identity.user_id)
            else:
                answer = self.execute(action, raw, text, identity.user_id if identity else None)
            if isinstance(answer, ActionResult):
                return self.answer(answer.text, source="java", action=action,
                                   data_changed=True, refresh=list(answer.refresh), **watson_metadata)
            return self.answer(answer, source="java", action=action, **watson_metadata)
        except ServiceError as error:
            reason = {
                400: "A plataforma recusou o pedido. Confira o saldo e o estado da missão ou do item.",
                401: "O serviço recusou o acesso.", 403: "O serviço recusou o acesso.",
                404: "O registro solicitado não foi encontrado.",
                409: "O pedido entrou em conflito com o estado atual da missão ou do item.",
            }.get(error.status, "O serviço não conseguiu atender ao pedido agora.")
            return self.answer("Não consegui confirmar o resultado. " + reason +
                               " Consulte seus dados antes de repetir uma operação.",
                               source="java", action=action, unavailable=True)

    @staticmethod
    def answer(reply, source="local", **metadata):
        return {"reply": reply, "text": reply,
                "metadata": {"source": source, "data_changed": False, **metadata}, "variables": {}}

    def execute(self, action, raw, text, user_id):
        if action == "loja":
            items = self.java.get_list("/acessorio")
            return ("🛍️ Acessórios para seu avatar\n\n" + "\n\n".join(
                f"• {shown(v.get('nome'))}\n🪙 Preço: {money(v.get('preco'))}"
                + (f"\nTipo: {v['tipo']}" if v.get("tipo") else "")
                + (f" | Raridade: {v['raridade']}" if v.get("raridade") else "") for v in items
            ) + "\n\nVocê pode dizer “comprar” e o nome do acessório." if items
                    else "A loja não tem acessórios disponíveis no momento.")
        if action == "missoes":
            return self.available_missions(user_id)
        if action == "perfil":
            user = self.java.get_object(f"/usuario/{user_id}")
            if user.get("idUsuario") != user_id:
                raise ServiceError("java")
            if re.search(r"saldo|quantas moedas|quanto.*moedas", text):
                return f"Seu saldo atual é {money(user.get('saldoMoedas'))}."
            if re.search(r"ofensiva|foguinho", text):
                return f"Seu nível de ofensiva é {shown(user.get('nivelOfensiva'))}."
            return (f"Seu perfil\n\nNome: {shown(user.get('nome'))}\nSaldo: {money(user.get('saldoMoedas'))}"
                    f"\nNível de ofensiva: {shown(user.get('nivelOfensiva'))}")
        if action == "avatar":
            avatar = self.java.get_object(f"/avatar/usuario/{user_id}")
            return (f"Seu avatar se chama {shown(avatar.get('nome'))}.\n\nPeça para ver seu inventário "
                    "para consultar os acessórios." if avatar.get("idAvatar")
                    else "Você ainda não possui um avatar cadastrado.")
        if action in {"inventario", "minhas"}:
            return self.records(action, text, user_id)
        return self.operate(action, raw, user_id)

    def available_missions(self, user_id):
        items = [v for v in self.java.get_list("/missao/ativas") if yes(v.get("ativo"))
                 and not re.search(r"encerrad[ao]", normalize(v.get("descricaoMissao")))]
        if not items:
            return "No momento, não há missões ativas disponíveis."
        progress = self.mission_progress(user_id) if user_id else []
        completed = {v["idMissao"] for v in progress
                     if integer(v.get("idMissao")) and yes(v.get("missaoConcluida"))}
        assigned = {v["idMissao"] for v in progress if integer(v.get("idMissao"))}
        items = [v for v in items if v.get("idMissao") not in completed]
        if not items:
            return "Você já concluiu todas as missões ativas disponíveis. Peça “minhas missões concluídas” " \
                "para consultar seu histórico ou aguarde novas missões."
        lines = []
        for mission in items:
            state = "\nStatus: em andamento" if mission.get("idMissao") in assigned else ""
            lines.append(f"🌿 {shown(mission.get('tituloMissao'))}\n{shown(mission.get('descricaoMissao'))}"
                         f"\n🪙 Recompensa: {money(mission.get('valorMoedasMissao'))}" + state)
        if not user_id:
            return "🌱 Missões disponíveis no catálogo\n\n" + "\n\n".join(lines) + \
                "\n\nEntre na sua conta para conferir quais você já iniciou ou concluiu. " \
                "Esta consulta pública não verifica seu progresso e não inicia missões."
        new = next((v for v in items if v.get("idMissao") not in assigned), None)
        next_step = (f"Para iniciar uma nova missão, diga “iniciar missão {new.get('tituloMissao')}”." if new else
                     "Você já iniciou essas missões. Peça “minhas missões pendentes” para acompanhar seu progresso.")
        return "🌱 Missões que você ainda não concluiu\n\n" + "\n\n".join(lines) + "\n\n" + next_step

    def mission_progress(self, user_id):
        progress = self.java.get_list(f"/usuario/{user_id}/missoes")
        for record in progress:
            state = record.get("missaoConcluida")
            if (not integer(record.get("idMissao")) or record["idMissao"] <= 0
                    or str(state).lower() not in {"s", "n", "sim", "nao", "não", "true", "false", "1", "0"}):
                raise ServiceError("java")
        return progress

    def records(self, action, text, user_id):
        inventory = action == "inventario"
        records = self.java.get_list(f"/usuario/{user_id}/" + ("inventario" if inventory else "missoes"))
        if not records:
            return ("Seu inventário está vazio. Peça para ver a loja." if inventory
                    else "Você ainda não possui missões atribuídas. Peça para ver as missões disponíveis.")
        unavailable = False
        try:
            catalog = self.java.get_list("/acessorio" if inventory else "/missao")
        except ServiceError:
            catalog, unavailable = [], True
        if inventory:
            equipped = bool(re.search(r"equipad|vestindo|usando", text))
            records = [v for v in records if not equipped or yes(v.get("equipado"))]
            if not records:
                return "Você não tem acessórios equipados no momento."
            lines = []
            for record in records:
                item = next((v for v in catalog if v.get("idAcessorio") == record.get("idAcessorio")), {})
                state = ("Não consegui consultar se está equipado" if record.get("equipado") is None else
                         "Equipado no avatar" if yes(record["equipado"]) else "Disponível no inventário")
                lines.append(f"{item.get('nome') or 'Acessório com nome indisponível'}\n{state}")
            return ("Estes são seus acessórios equipados:" if equipped else "Estes são os itens do seu inventário:") + \
                "\n\n" + "\n\n".join(lines) + ("\n\nOs nomes não puderam ser consultados agora." if unavailable else "")
        completed = bool(re.search(r"concluid|finalizad", text))
        pending = bool(re.search(r"pendente|andamento", text))
        records = [v for v in records if (not completed or yes(v.get("missaoConcluida")))
                   and (not pending or v.get("missaoConcluida") is not None and not yes(v["missaoConcluida"]))]
        if not records:
            return "Você não possui missões com esse status."
        lines = []
        for record in records:
            mission = next((v for v in catalog if v.get("idMissao") == record.get("idMissao")), {})
            state = ("não informado" if record.get("missaoConcluida") is None else
                     "concluída" if yes(record["missaoConcluida"]) else "em andamento")
            lines.append(f"{mission.get('tituloMissao') or 'Missão com nome indisponível'}\nStatus: {state}"
                         f"\nProgresso registrado: {shown(record.get('progressoMissao'))}"
                         + (f"\nMoedas ganhas: {money(record.get('moedasGanhasMissao'))}"
                            if yes(record.get("missaoConcluida")) else ""))
        return "Estas são suas missões:\n\n" + "\n\n".join(lines)

    def operate(self, action, raw, user_id):
        mission = action in {"iniciar", "concluir"}
        id_key, name_key = ("idMissao", "tituloMissao") if mission else ("idAcessorio", "nome")
        catalog = self.java.get_list("/missao" if mission else "/acessorio")
        text = normalize(raw)
        explicit_ids = re.findall(r"(?:\bid\s*|\bmissao\s+|\bitem\s+|\bacessorio\s+)(\d+)\b", text)
        if len(set(explicit_ids)) > 1:
            return "Informe apenas um item ou uma missão por operação. Nenhuma operação foi enviada."
        explicit = re.search(r"(?:\bid\s*|\bmissao\s+|\bitem\s+|\bacessorio\s+)(\d+)\b", text)
        explicit = explicit or re.fullmatch(r"(?:comprar|equipar|desequipar|iniciar|concluir)\s+(\d+)", text)
        matches = [v for v in catalog if (str(v.get(id_key)) == explicit[1] if explicit else
                   v.get(name_key) and normalize(v[name_key]) in text)]
        if len(matches) != 1:
            return ("Encontrei mais de uma opção." if matches else "Não consegui identificar o item ou a missão.") + \
                " Diga o nome completo que aparece em “missões” ou “loja”."
        target = matches[0]
        target_id = target.get(id_key)
        if not integer(target_id) or target_id <= 0:
            raise ServiceError("java")
        if mission:
            progress = self.mission_progress(user_id)
            record = next((v for v in progress if v.get("idMissao") == target_id), None)
            if record is not None and yes(record.get("missaoConcluida")):
                return f"Você já concluiu a missão “{target[name_key]}”. Não é necessário iniciá-la ou concluí-la novamente."
            if action == "iniciar" and record is not None:
                return f"A missão “{target[name_key]}” já está em andamento. Peça “minhas missões pendentes” para acompanhá-la."
            if action == "concluir" and record is None:
                return f"Você ainda não iniciou a missão “{target[name_key]}”. Diga “iniciar missão {target[name_key]}” primeiro."
        path = f"/usuario/{user_id}/" + ("missoes/" if mission else "inventario/") + str(target_id)
        suffix = {"concluir": "/concluir", "equipar": "/equipar", "desequipar": "/desequipar"}.get(action, "")
        result = self.java.request("PUT" if action in {"equipar", "desequipar"} else "POST", path + suffix)
        if (not isinstance(result, dict) or result.get(id_key) != target_id
                or result.get("idUsuario") != user_id):
            raise ServiceError("java")
        if action == "concluir" and not yes(result.get("missaoConcluida")):
            raise ServiceError("java")
        if action == "equipar" and not yes(result.get("equipado")):
            raise ServiceError("java")
        if action == "desequipar" and (result.get("equipado") is None or yes(result["equipado"])):
            raise ServiceError("java")
        labels = {"iniciar": "Missão iniciada", "concluir": "Missão concluída", "comprar": "Acessório comprado",
                  "equipar": "Acessório equipado", "desequipar": "Acessório desequipado"}
        reply = f"{labels[action]}: {target[name_key]}." + (
            f"\nRecompensa registrada: {money(result['moedasGanhasMissao'])}."
            if action == "concluir" and result.get("moedasGanhasMissao") is not None else "")
        return ActionResult(reply, ("usuario", "missoes") if mission else ("usuario", "inventario", "avatar"))

    def recommend(self, raw, user_id):
        catalog = self.java.get_list("/acessorio")
        exact = [i for i, v in enumerate(catalog) if v.get("nome") and normalize(v["nome"]) in normalize(raw)]
        # Nomes sobrepostos sao ambiguos: nao deixar o LLM escolher no lugar da pessoa.
        if len(exact) > 1:
            return "Encontrei mais de um acessório na pergunta. Informe apenas o nome de um item da loja."
        index = self.selector.select(raw, catalog, exact[0] if exact else None)
        if index is None:
            return "Não consegui identificar esse acessório com segurança. Peça para ver a loja e use o nome do item."
        item = catalog[index]
        user = self.java.get_object(f"/usuario/{user_id}")
        missions = self.java.get_list("/missao/ativas")
        progress = self.mission_progress(user_id)
        inventory = self.java.get_list(f"/usuario/{user_id}/inventario")
        price, balance = item.get("preco"), user.get("saldoMoedas")
        if user.get("idUsuario") != user_id or not integer(price) or not integer(balance):
            raise ServiceError("java")
        if any(v.get("idAcessorio") == item.get("idAcessorio") for v in inventory):
            return f"Você já tem {item.get('nome')} no seu inventário! Para usá-lo, diga “equipar {item.get('nome')}”."
        gap = max(0, price - balance)
        head = f"🎯 Vamos conseguir {item.get('nome')}!\n\n🪙 Preço: {price} moedas\nSeu saldo: {balance} moedas"
        if not gap:
            return head + f"\n\nVocê já tem moedas suficientes. Para comprar, diga “comprar {item.get('nome')}”."
        completed = {v["idMissao"] for v in progress
                     if integer(v.get("idMissao")) and yes(v.get("missaoConcluida"))}
        seen, eligible = set(), []
        for mission in missions:
            mid, reward = mission.get("idMissao"), mission.get("valorMoedasMissao")
            if (integer(mid) and mid > 0 and mid not in seen and mid not in completed
                    and yes(mission.get("ativo")) and integer(reward) and reward > 0
                    and not re.search(r"encerrad[ao]", normalize(mission.get("descricaoMissao")))):
                eligible.append(mission)
                seen.add(mid)
        earned, chosen = 0, []
        for mission in sorted(eligible, key=lambda v: v["valorMoedasMissao"], reverse=True):
            if earned >= gap:
                break
            chosen.append(mission)
            earned += mission["valorMoedasMissao"]
        if not chosen:
            return head + f"\nFaltam {gap} moedas.\n\nNão encontrei missões ativas ainda não concluídas " \
                "com recompensas confirmadas para recomendar agora."
        lines = "\n\n".join(f"🌿 {shown(v.get('tituloMissao'))}\n{v.get('descricaoMissao') or ''}"
                              f"\nRecompensa: {v['valorMoedasMissao']} moedas" for v in chosen)
        end = (f"Ao concluir essas missões, você poderá somar {earned} moedas e chegar a {balance + earned} moedas — "
               "suficiente para esse item." if earned >= gap else f"Essas missões somam {earned} moedas. "
               f"Depois delas ainda faltarão {gap - earned} moedas; será necessário aguardar outras missões.")
        already_started = any(v.get("idMissao") == chosen[0].get("idMissao") for v in progress)
        next_step = ("Você já iniciou a primeira missão sugerida. Peça “minhas missões pendentes” para acompanhar "
                     "seu progresso." if already_started else f"Para começar, diga “iniciar missão {chosen[0].get('tituloMissao')}”.")
        return head + f"\nFaltam {gap} moedas.\n\nEstas missões podem ajudar:\n\n" + lines + "\n\n" + end + \
            "\n\nConsiderei apenas missões que você ainda não concluiu. As moedas dependem da conclusão e da " \
            "confirmação na plataforma.\n" + next_step

    def engagement(self, data):
        if data is None:
            return self.answer("Ainda não recebi os indicadores de uso necessários para analisar seu engajamento. "
                               "A plataforma precisa fornecer esses dados; não vou estimar o resultado pela conversa.")
        try:
            validated = DadosUsuario.model_validate(data, strict=True)
        except ValidationError:
            return self.answer("Os indicadores recebidos estão incompletos ou inconsistentes. "
                               "Não consegui analisar seu engajamento.")
        try:
            result = self.predictor(validated)
        except Exception:
            return self.answer("Não consegui analisar seu engajamento agora. Tente novamente em instantes.",
                               source="ml", unavailable=True)
        level = result.get("nivel_engajamento")
        labels = {"baixo": "baixo", "medio": "médio", "alto": "alto"}
        tips = {"baixo": "Que tal escolher uma missão que caiba na sua rotina para começar?",
                "medio": "Você pode escolher uma próxima missão para manter sua participação.",
                "alto": "Continue escolhendo missões que façam sentido para sua rotina."}
        if result.get("sucesso") is not True or level not in labels:
            return self.answer("Não consegui analisar seu engajamento agora.", source="ml", unavailable=True)
        return self.answer(f"🌱 Seu engajamento foi classificado como {labels[level]}.\n\n{tips[level]}\n\n"
                           "Essa é uma previsão experimental de um modelo treinado com dados sintéticos, "
                           "e não uma avaliação definitiva sobre você.", source="ml", nivel_engajamento=level)
