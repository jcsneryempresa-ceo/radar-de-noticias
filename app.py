# app.py — Radar de Notícias (Streamlit)
# Recursos:
# - Busca global via Google News RSS
# - Ranking por score (relevância) + data (recência)
# - Planner simples (checklist de leitura / ação)
# - Gemini via google-genai
# - Limite diário: 3 gerações de resumo por dia (por máquina/instalação)
#
# Como rodar:
#   pip install streamlit feedparser python-dateutil google-genai
#   streamlit run app.py
#
# Chave do Gemini:
#   - Opção A (recomendado): Streamlit Secrets -> st.secrets["GEMINI_API_KEY"]
#   - Opção B: variável de ambiente GEMINI_API_KEY

from __future__ import annotations

import os
import re
import json
import math
import time
import hashlib
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import feedparser
from dateutil import parser as dateparser
from dateutil import tz

# Gemini (google-genai)
# Docs: https://pypi.org/project/google-genai/
try:
    from google import genai
except Exception:
    genai = None


# =========================
# Configuração básica
# =========================
APP_TITLE = "Radar de Notícias"
DAILY_LIMIT = 3
STATE_FILE = "radar_state.json"  # limite diário local (instalação)
TZ_LOCAL = tz.gettz("America/Fortaleza")  # seu fuso (RN)

DEFAULT_LANG = "pt-BR"

CATEGORIES = [
    "Política", "Economia", "Educação", "Tecnologia", "Saúde",
    "Esportes", "Segurança", "Cultura", "Mundo", "Brasil",
    "Ciência", "Trabalho", "Assistência Social"
]

# Filtros "pré-varredura" (o app não usa geolocalização automática; você define manualmente)
DEFAULT_LOCATION_HINT = "Brasil OR RN OR Natal OR Parnamirim"


# =========================
# Utilidades: arquivo/limite
# =========================
def _today_key() -> str:
    # chave do dia no fuso local
    now = dt.datetime.now(TZ_LOCAL)
    return now.strftime("%Y-%m-%d")


def _load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {"day": _today_key(), "used": 0}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("day") != _today_key():
            return {"day": _today_key(), "used": 0}
        if "used" not in data:
            data["used"] = 0
        return data
    except Exception:
        return {"day": _today_key(), "used": 0}


def _save_state(data: Dict[str, Any]) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        # se falhar, não derruba o app
        pass


def verificar_limite_diario() -> Tuple[bool, int, int]:
    """
    Retorna (pode_usar, usados, limite).
    """
    data = _load_state()
    used = int(data.get("used", 0))
    if used >= DAILY_LIMIT:
        return (False, used, DAILY_LIMIT)
    return (True, used, DAILY_LIMIT)


def registrar_uso() -> Tuple[int, int]:
    """
    Incrementa uso e retorna (usados, limite).
    """
    data = _load_state()
    if data.get("day") != _today_key():
        data = {"day": _today_key(), "used": 0}
    data["used"] = int(data.get("used", 0)) + 1
    _save_state(data)
    return (data["used"], DAILY_LIMIT)


# =========================
# Modelo de notícia
# =========================
@dataclass
class NewsItem:
    title: str
    link: str
    source: str
    published: Optional[dt.datetime]
    summary: str
    query: str

    # Scores
    score_relevance: float = 0.0
    score_recency: float = 0.0
    score_total: float = 0.0


# =========================
# Google News RSS
# =========================
def build_google_news_rss_url(query: str, lang: str = "pt-BR", country: str = "BR") -> str:
    # Google News RSS: https://news.google.com/rss/search?q=...
    # hl=lang, gl=country, ceid=country:lang
    from urllib.parse import quote_plus
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl={lang}&gl={country}&ceid={country}:{lang}"


def parse_datetime_safe(value: Any) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        d = dateparser.parse(str(value))
        if d.tzinfo is None:
            # assume UTC se vier sem tz
            d = d.replace(tzinfo=tz.UTC)
        return d.astimezone(TZ_LOCAL)
    except Exception:
        return None


def fetch_rss_items(query: str, max_items: int = 30, lang: str = "pt-BR", country: str = "BR") -> List[NewsItem]:
    url = build_google_news_rss_url(query=query, lang=lang, country=country)
    feed = feedparser.parse(url)

    items: List[NewsItem] = []
    for e in feed.entries[:max_items]:
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "").strip()
        published = None

        # Tentativas comuns no RSS:
        if hasattr(e, "published"):
            published = parse_datetime_safe(e.published)
        elif hasattr(e, "updated"):
            published = parse_datetime_safe(e.updated)

        summary = getattr(e, "summary", "") or getattr(e, "description", "")
        summary = re.sub(r"\s+", " ", str(summary)).strip()

        # No Google News RSS, o "source" às vezes vem em e.source.title
        source = ""
        try:
            source = (e.source.title or "").strip()  # type: ignore
        except Exception:
            source = ""

        if not title or not link:
            continue

        items.append(
            NewsItem(
                title=title,
                link=link,
                source=source,
                published=published,
                summary=summary,
                query=query,
            )
        )
    return items


# =========================
# Ranking (score + data)
# =========================
STOPWORDS_PT = set("""
a o os as um uma uns umas de do da dos das em no na nos nas para por com sem e ou
que como quando onde qual quais quem se sua seu suas seus meu meus minha minhas
é foi são ser estar está estavam esteve estive tem têm tinha tinham
mais menos muito muita muitos muitas já ainda também
""".split())


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9áéíóúâêôãõç\- ]+", " ", text, flags=re.IGNORECASE)
    parts = re.split(r"\s+", text.strip())
    toks = []
    for p in parts:
        if not p or len(p) < 3:
            continue
        if p in STOPWORDS_PT:
            continue
        toks.append(p)
    return toks


def relevance_score(item: NewsItem, query: str, extra_terms: List[str]) -> float:
    # Score simples e robusto:
    # - match de termos do query no título pesa mais
    # - match no resumo pesa menos
    q_terms = tokenize(query) + [t.lower() for t in extra_terms if t.strip()]
    q_terms = [t for t in q_terms if t and t not in STOPWORDS_PT]
    if not q_terms:
        return 0.0

    title = item.title.lower()
    blob = (item.title + " " + item.summary).lower()

    hits_title = sum(1 for t in q_terms if t in title)
    hits_blob = sum(1 for t in q_terms if t in blob)

    # bônus se o item parece "local" (ex: RN, Natal, Parnamirim)
    local_bonus = 0.0
    local_markers = ["rn", "natal", "parnamirim", "rio grande do norte"]
    if any(m in blob for m in local_markers):
        local_bonus = 0.15

    # fórmula:
    base = (hits_title * 1.6 + hits_blob * 0.6) / max(1, len(set(q_terms)))
    # normaliza suavemente
    base = math.tanh(base)  # 0..~1
    return float(min(1.0, base + local_bonus))


def recency_score(published: Optional[dt.datetime]) -> float:
    # Score 1.0 = agora; decai com meia-vida de 36h
    if not published:
        return 0.35  # neutro baixo
    now = dt.datetime.now(TZ_LOCAL)
    hours = max(0.0, (now - published).total_seconds() / 3600.0)
    half_life = 36.0
    score = 2 ** (-hours / half_life)
    return float(max(0.0, min(1.0, score)))


def rank_items(items: List[NewsItem], query: str, extra_terms: List[str], w_rel: float = 0.65, w_rec: float = 0.35) -> List[NewsItem]:
    for it in items:
        it.score_relevance = relevance_score(it, query, extra_terms)
        it.score_recency = recency_score(it.published)
        it.score_total = (w_rel * it.score_relevance) + (w_rec * it.score_recency)

    # desempate por data (mais recente primeiro), depois score total
    def sort_key(it: NewsItem):
        ts = it.published.timestamp() if it.published else 0
        return (it.score_total, ts)

    return sorted(items, key=sort_key, reverse=True)


# =========================
# Prompt “já corrigido” + Gemini
# =========================
def get_gemini_key() -> Optional[str]:
    # Streamlit secrets primeiro, depois env
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")  # type: ignore
    except Exception:
        key = None
    if not key:
        key = os.getenv("GEMINI_API_KEY")
    return key


def build_prompt(item: NewsItem, persona: str, objective: str) -> str:
    # Prompt desenhado para:
    # - Resumo factual e curto
    # - Sem alucinar: se faltarem dados, dizer "não informado"
    # - Estrutura fixa, com tags e ação
    # - Título, pontos-chave, por que importa, riscos, próximos passos
    published_str = item.published.strftime("%d/%m/%Y %H:%M") if item.published else "não informado"

    return f"""
Você é um analista de notícias extremamente cuidadoso e objetivo.

REGRAS (obrigatórias):
- Use SOMENTE as informações fornecidas abaixo. Não invente fatos, números, nomes, cargos, datas ou citações.
- Se um detalhe não estiver explícito, escreva "não informado".
- Não dê opiniões políticas. Foque em fatos, contexto mínimo e implicações práticas.
- Seja conciso: no máximo 1700 caracteres no total.

CONTEXTO DO USUÁRIO (para calibrar relevância):
- Persona: {persona}
- Objetivo: {objective}

DADOS DA NOTÍCIA (fonte bruta):
- Título: {item.title}
- Publicação: {published_str}
- Veículo/Fonte: {item.source or "não informado"}
- Link: {item.link}
- Trecho/Resumo do RSS: {item.summary or "não informado"}

ENTREGA (formato fixo):
1) Resumo (2–4 frases)
2) Pontos-chave (3 bullets)
3) Por que isso importa (1–2 frases)
4) O que falta confirmar (1–2 itens)
5) Ação sugerida (1 item, prático e pequeno)
6) Tags (até 6: tema, local, atores, tipo de evento)
""".strip()


def gemini_summarize(prompt: str, model: str = "gemini-2.0-flash") -> str:
    if genai is None:
        raise RuntimeError("Biblioteca google-genai não está instalada.")
    key = get_gemini_key()
    if not key:
        raise RuntimeError("Chave do Gemini não encontrada. Defina GEMINI_API_KEY (env) ou st.secrets['GEMINI_API_KEY'].")

    client = genai.Client(api_key=key)

    # Resposta padrão de texto
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    # Compatibilidade com retornos diferentes
    text = getattr(resp, "text", None)
    if text:
        return text.strip()

    # fallback: tenta extrair de candidates
    try:
        candidates = resp.candidates or []
        if candidates and candidates[0].content and candidates[0].content.parts:
            parts = candidates[0].content.parts
            out = "".join(getattr(p, "text", "") for p in parts)
            return out.strip()
    except Exception:
        pass

    return "Não foi possível obter texto do Gemini (resposta vazia)."


# =========================
# Planner simples
# =========================
def planner_init():
    if "planner" not in st.session_state:
        st.session_state.planner = []  # list[dict]


def planner_add(title: str, link: str):
    planner_init()
    item_id = hashlib.sha1((title + link).encode("utf-8")).hexdigest()[:10]
    st.session_state.planner.append({
        "id": item_id,
        "title": title,
        "link": link,
        "done": False,
        "created_at": dt.datetime.now(TZ_LOCAL).strftime("%d/%m/%Y %H:%M"),
    })


def planner_render():
    planner_init()
    st.subheader("🗓️ Planner (leituras/ações)")
    if not st.session_state.planner:
        st.caption("Adicione itens clicando em “Adicionar ao Planner” nas notícias.")
        return

    # Controles
    cols = st.columns([1, 1, 2])
    if cols[0].button("Marcar tudo como feito"):
        for it in st.session_state.planner:
            it["done"] = True
    if cols[1].button("Limpar concluídos"):
        st.session_state.planner = [it for it in st.session_state.planner if not it.get("done")]

    st.write("")

    # Lista
    for i, it in enumerate(list(st.session_state.planner)):
        c1, c2, c3 = st.columns([0.08, 0.72, 0.20])
        done = c1.checkbox("", value=bool(it.get("done")), key=f"pl_done_{it['id']}")
        st.session_state.planner[i]["done"] = done

        c2.markdown(f"**{it['title']}**  \n_{it['created_at']}_")
        c3.link_button("Abrir", it["link"])


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("🛰️ Radar de Notícias")

with st.sidebar:
    st.header("⚙️ Configurações")

    # Modo / motor
    st.caption("Busca global: Google News RSS")
    lang = st.selectbox("Idioma (hl)", ["pt-BR", "en-US", "es-ES"], index=0)
    country = st.selectbox("País (gl)", ["BR", "US", "PT", "AR", "MX"], index=0)

    st.divider()

    st.subheader("🔎 Pré-varredura")
    topic = st.selectbox("Tema principal", CATEGORIES, index=0)
    location_hint = st.text_input("Filtro de localização (opcional)", value=DEFAULT_LOCATION_HINT)

    st.caption("Dica: use operadores: OR, AND, aspas, etc.")
    extra_terms_text = st.text_input("Termos extras (separe por vírgula)", value="")
    extra_terms = [t.strip() for t in extra_terms_text.split(",") if t.strip()]

    st.divider()

    st.subheader("🏁 Ranking")
    w_rel = st.slider("Peso relevância", 0.0, 1.0, 0.65, 0.05)
    w_rec = st.slider("Peso recência", 0.0, 1.0, 0.35, 0.05)

    st.divider()

    st.subheader("🤖 Gemini")
    model = st.selectbox("Modelo", ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-flash"], index=0)
    persona = st.text_input("Persona (curto)", value="Educador social que precisa entender o essencial rápido.")
    objective = st.text_input("Objetivo (curto)", value="Triar notícias e decidir ações práticas em poucos minutos.")
    st.caption("Limite diário de resumos: 3")

    ok, used, lim = verificar_limite_diario()
    if ok:
        st.success(f"Resumos hoje: {used}/{lim}")
    else:
        st.error(f"Limite diário atingido: {used}/{lim}")

    st.divider()
    planner_render()

# Corpo
colA, colB = st.columns([0.62, 0.38], gap="large")

with colA:
    st.subheader("📡 Busca")

    query_base = st.text_input(
        "Consulta",
        value=f"{topic} {location_hint}".strip(),
        help="Ex.: política Parnamirim RN; ou use OR/AND/aspas"
    )

    max_items = st.slider("Quantidade (RSS)", 10, 80, 30, 5)
    buscar = st.button("Buscar agora", type="primary")

    if buscar:
        with st.spinner("Buscando no Google News RSS..."):
            raw = fetch_rss_items(query=query_base, max_items=max_items, lang=lang, country=country)
            ranked = rank_items(raw, query=query_base, extra_terms=extra_terms, w_rel=w_rel, w_rec=w_rec)
            st.session_state.last_results = ranked
            st.session_state.last_query = query_base
            st.session_state.last_ts = time.time()

    results: List[NewsItem] = st.session_state.get("last_results", [])
    if results:
        st.caption(f"Resultados: {len(results)}  •  Query: {st.session_state.get('last_query','')}")
        st.write("")

        for idx, it in enumerate(results[:25], start=1):
            published_str = it.published.strftime("%d/%m/%Y %H:%M") if it.published else "não informado"

            with st.container(border=True):
                top = st.columns([0.74, 0.26])
                top[0].markdown(f"**{idx}. {it.title}**")
                top[1].markdown(
                    f"**Score:** {it.score_total:.2f}  \n"
                    f"Rel: {it.score_relevance:.2f} • Rec: {it.score_recency:.2f}"
                )

                meta_cols = st.columns([0.34, 0.33, 0.33])
                meta_cols[0].markdown(f"**Fonte:** {it.source or 'não informado'}")
                meta_cols[1].markdown(f"**Publicado:** {published_str}")
                meta_cols[2].link_button("Abrir notícia", it.link)

                if it.summary:
                    st.caption(it.summary[:280] + ("…" if len(it.summary) > 280 else ""))

                actions = st.columns([0.34, 0.33, 0.33])

                if actions[0].button("Adicionar ao Planner", key=f"add_plan_{idx}"):
                    planner_add(it.title, it.link)
                    st.toast("Adicionado ao Planner.", icon="🗓️")

                # Resumo Gemini
                can_use, used_now, lim_now = verificar_limite_diario()
                if not can_use:
                    actions[1].button("Gerar resumo (limite atingido)", key=f"sum_disabled_{idx}", disabled=True)
                else:
                    if actions[1].button("Gerar resumo", key=f"sum_{idx}"):
                        try:
                            prompt = build_prompt(it, persona=persona, objective=objective)
                            with st.spinner("Gerando resumo no Gemini..."):
                                text = gemini_summarize(prompt, model=model)
                            registrar_uso()
                            st.session_state[f"summary_{idx}"] = text
                        except Exception as e:
                            st.error(f"Erro no Gemini: {e}")

                # Copiar prompt (útil pra debugar)
                if actions[2].button("Copiar prompt", key=f"copy_{idx}"):
                    st.session_state[f"prompt_{idx}"] = build_prompt(it, persona=persona, objective=objective)
                    st.toast("Prompt pronto (veja na coluna da direita).", icon="📋")

                # Exibir resumo se existir
                s = st.session_state.get(f"summary_{idx}")
                if s:
                    st.markdown("**Resumo Gemini**")
                    st.write(s)

with colB:
    st.subheader("📋 Painel")

    ok, used, lim = verificar_limite_diario()
    st.info(f"Resumos hoje: {used}/{lim}")

    st.markdown("### Prompt (último copiado)")
    last_prompt = None
    # pega o prompt mais recente salvo
    for k in list(st.session_state.keys())[::-1]:
        if str(k).startswith("prompt_"):
            last_prompt = st.session_state.get(k)
            break

    if last_prompt:
        st.code(last_prompt, language="text")
    else:
        st.caption("Clique em “Copiar prompt” em alguma notícia para ver aqui.")

    st.markdown("### Como o motor faz varredura (modelo simples)")
    st.write(
        "- **Pré-varredura:** você define Tema + Local + termos extras.\n"
        "- **Busca:** Google News RSS retorna itens recentes.\n"
        "- **Ranking:** Score total = (peso relevância * match de termos) + (peso recência * idade da notícia).\n"
        "- **Planner:** você marca o que vai ler/agir depois.\n"
        "- **Gemini:** gera um resumo factual e curto usando apenas os dados do RSS."
    )

    st.markdown("### Diagnóstico rápido")
    if genai is None:
        st.warning("google-genai não está instalado. Instale: `pip install google-genai`")
    else:
        key = get_gemini_key()
        if not key:
            st.warning("Falta a chave GEMINI_API_KEY (env) ou st.secrets['GEMINI_API_KEY'].")
        else:
            st.success("Gemini pronto (chave encontrada).")
