import os
import json
import datetime as dt
import streamlit as st
import feedparser
import pandas as pd
from openai import OpenAI
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ==============================
# CONFIG
# ==============================

st.set_page_config(page_title="Radar de Notícias", layout="centered")

PROFILE_FILE = "perfil.json"
USO_FILE = "uso_ia.json"
MAX_DIARIO = 3

BRT = ZoneInfo("America/Fortaleza")

TEMAS = ["Política", "Economia", "Esporte", "Moda", "Cultura", "Educação", "Segurança", "Saúde"]

PALAVRAS_TEMA = {
    "Política": ["governo", "congresso", "prefeitura", "vereador", "deputado", "ministro", "partido", "eleição"],
    "Economia": ["inflação", "juros", "pib", "mercado", "emprego", "renda", "investimento"],
    "Esporte": ["campeonato", "time", "atleta", "partida", "vitória", "gol", "copa"],
    "Moda": ["coleção", "tendência", "look", "desfile", "marca", "estilo"],
    "Cultura": ["show", "festival", "arte", "cinema", "teatro", "livro"],
    "Educação": ["escola", "professor", "aluno", "enem", "universidade", "curso"],
    "Segurança": ["polícia", "crime", "assalto", "roubo", "prisão", "investigação"],
    "Saúde": ["hospital", "sus", "vacina", "doença", "médico", "tratamento"],
}

# ==============================
# HELPERS
# ==============================

def load_profile():
    try:
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {
            "nome_portal": "Radar de Notícias",
            "assinatura": "jcsnery.empresa",
            "estilo": "jornalistico",
            "intencao_comunicativa": "neutro",
            "linhas": 10,
        }

def save_profile(p):
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)

def carregar_uso():
    try:
        with open(USO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"data": str(dt.date.today()), "contador": 0}

def salvar_uso(dados):
    with open(USO_FILE, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

def verificar_limite():
    uso = carregar_uso()
    hoje = str(dt.date.today())
    if uso.get("data") != hoje:
        uso = {"data": hoje, "contador": 0}
        salvar_uso(uso)
    return uso

def normalizar_locais(txt: str):
    return [x.strip() for x in (txt or "").split(",") if x.strip()]

def extrair_data_entry(entry):
    """
    Pega data do item RSS e devolve datetime em BRT.
    Se não existir data, devolve None.
    """
    if getattr(entry, "published_parsed", None):
        dtu = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        return dtu.astimezone(BRT)
    if getattr(entry, "updated_parsed", None):
        dtu = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
        return dtu.astimezone(BRT)
    return None

def marcador_tempo(dt_item, agora):
    if dt_item is None:
        return "SEM DATA"
    delta = agora - dt_item
    horas = delta.total_seconds() / 3600
    dias = delta.days

    if horas < 2:
        return "AGORA"
    if horas < 24:
        return "HOJE"
    if dias == 1:
        return "ONTEM"
    if dias < 7:
        return f"{dias} dias"
    if dias < 30:
        return f"{dias // 7} sem"
    return f"{dias // 30} mes"

def montar_google_news_url(assunto: str, local: str, janela: str):
    """
    Google News RSS. A 'janela' entra como parâmetro de recência (when).
    """
    consulta = " ".join([x for x in [assunto.strip(), local.strip()] if x]).strip()
    if not consulta:
        consulta = "notícias"

    # when:1d, when:7d, when:30d
    when_map = {
        "24h": "1d",
        "7 dias": "7d",
        "30 dias": "30d",
    }
    when = when_map.get(janela, None)
    if when:
        consulta = f"{consulta} when:{when}"

    q = consulta.replace(" ", "+")
    return f"https://news.google.com/rss/search?q={q}&hl=pt-BR&gl=BR&ceid=BR:pt-419"

def score_texto(texto: str, palavras: list[str], locais: list[str]):
    """
    Score simples:
    - termo do tema/assunto: +2 por ocorrência (checa presença)
    - local: +4 por presença
    """
    t = (texto or "").lower()
    score = 0

    for w in palavras:
        if w and w.lower() in t:
            score += 2

    for loc in locais:
        if loc and loc.lower() in t:
            score += 4

    return score

# ==============================
# APP
# ==============================

profile = load_profile()

st.title("📡 Radar de Notícias")
st.caption(f"{profile['nome_portal']} • {profile['assinatura']}")

tabs = st.tabs(["Identidade", "Inteligência", "Produção"])

# ==============================
# TAB: IDENTIDADE
# ==============================

with tabs[0]:
    nome = st.text_input("Nome do Portal", profile["nome_portal"])
    assinatura = st.text_input("Assinatura", profile["assinatura"])
    estilo = st.selectbox("Estilo", ["jornalistico", "analitico", "opinativo", "didatico"],
                          index=["jornalistico", "analitico", "opinativo", "didatico"].index(profile.get("estilo", "jornalistico")))
    linhas = st.slider("Quantidade aproximada de linhas", 4, 20, int(profile["linhas"]))
    intencao = st.text_area("Intenção comunicativa", value=profile.get("intencao_comunicativa", ""))

    if st.button("Salvar Perfil"):
        save_profile({
            "nome_portal": nome,
            "assinatura": assinatura,
            "estilo": estilo,
            "intencao_comunicativa": intencao,
            "linhas": linhas,
        })
        st.success("Perfil salvo!")

# ==============================
# TAB: INTELIGÊNCIA
# ==============================

with tabs[1]:
    st.subheader("Varredura")

    modo = st.selectbox("Modo", ["Busca global (resultado acima de fonte)", "Fontes fixas (RN)"])

    col1, col2 = st.columns(2)
    with col1:
        tema_do_dia = st.selectbox("Tema (opcional)", ["(livre)"] + TEMAS)
    with col2:
        janela = st.selectbox("Janela de tempo", ["24h", "7 dias", "30 dias"])

    assunto_livre = st.text_input("Assunto (ex: picolés, dengue, emprego)", placeholder="Digite um assunto…")
    local_do_dia = st.text_input("Local (opcional) (ex: RN, Natal, Montes Claros, Groenlândia)", placeholder="Pode deixar em branco…")

    itens_por_fonte = st.slider("Itens por fonte", 5, 30, 10)

    # Define assunto e palavras para score
    if assunto_livre.strip():
        assunto = assunto_livre.strip()
        # termos do assunto para score (bem simples)
        palavras_assunto = [p.strip().lower() for p in assunto.replace(",", " ").split() if len(p.strip()) >= 3]
    elif tema_do_dia != "(livre)":
        assunto = tema_do_dia
        palavras_assunto = PALAVRAS_TEMA.get(tema_do_dia, [])
    else:
        assunto = ""
        palavras_assunto = []

    if modo == "Fontes fixas (RN)":
        sites = {
            "Tribuna do Norte": "https://tribunadonorte.com.br/feed/",
            "Agora RN": "https://agorarn.com.br/feed/",
        }
    else:
        sites = {
            "Google News": montar_google_news_url(assunto or "notícias", local_do_dia, janela)
        }

    if st.button("Vamos lá 🚀 Executar varredura"):
        resultados = []
        agora = datetime.now(BRT)
        locais_norm = normalizar_locais(local_do_dia)

        for nome_site, url in sites.items():
            feed = feedparser.parse(url)

            for entry in feed.entries[:itens_por_fonte]:
                titulo = getattr(entry, "title", "").strip()
                link = getattr(entry, "link", "").strip()
                resumo = entry.get("summary", "") or ""
                dt_item = extrair_data_entry(entry)

                texto_base = f"{titulo} {resumo}"
                sc = score_texto(texto_base, palavras_assunto, locais_norm)

                resultados.append({
                    "quando": marcador_tempo(dt_item, agora),
                    "data_txt": dt_item.strftime("%d %b %Y • %H:%M (BRT)") if dt_item else "data não informada pela fonte",
                    "data": dt_item,  # datetime ou None
                    "fonte": nome_site,
                    "titulo": titulo,
                    "link": link,
                    "resumo": resumo,
                    "score": sc,
                })

        df = pd.DataFrame(resultados)

        if df.empty:
            st.warning("Nenhum resultado encontrado.")
        else:
            # Ordenação: score desc + recência desc (None vai pro fim)
            df["data_ord"] = pd.to_datetime(df["data"], utc=True, errors="coerce")
            df["data_ord"] = df["data_ord"].fillna(pd.Timestamp("1970-01-01", tz="UTC"))
            df = df.sort_values(["score", "data_ord"], ascending=[False, False]).reset_index(drop=True)

            st.session_state["ranking"] = df
            st.dataframe(df[["quando", "data_txt", "fonte", "titulo", "score", "link"]].head(15), use_container_width=True)

# ==============================
# TAB: PRODUÇÃO
# ==============================

with tabs[2]:
    api_key = st.secrets.get("OPENAI_API_KEY", None)

    if not api_key:
        st.error("Chave OPENAI_API_KEY não configurada.")
        st.stop()

    if "ranking" not in st.session_state or st.session_state["ranking"].empty:
        st.warning("Execute a varredura antes.")
        st.stop()

    df = st.session_state["ranking"]

    escolha = st.selectbox(
        "Escolha a matéria",
        range(len(df)),
        format_func=lambda i: f"{df.iloc[i]['quando']} • {df.iloc[i]['titulo']}"
    )

    materia = df.iloc[escolha]
    st.caption(f"🗓️ {materia['data_txt']}  |  📰 {materia['fonte']}")

    if st.button("Gerar texto com IA ✍️"):
        uso = verificar_limite()

        if uso["contador"] >= MAX_DIARIO:
            st.error("Limite diário de 3 gerações atingido.")
            st.stop()

        prompt = f"""
Você é redator do portal {profile['nome_portal']}.
Estilo: {profile['estilo']}
Intenção: {profile['intencao_comunicativa']}
Escreva cerca de {profile['linhas']} linhas.

Base (notícia):
Fonte: {materia['fonte']}
Data: {materia['data_txt']}
Título: {materia['titulo']}
Resumo: {materia['resumo']}

Regras:
- Não invente fatos além do que está na base.
- Se a base for insuficiente, deixe claro que o texto é um resumo a partir do feed.
"""

        client = OpenAI(api_key=api_key)

        resposta = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Você escreve textos jornalísticos em português do Brasil, sem inventar fatos."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        uso["contador"] += 1
        salvar_uso(uso)

        st.success("Texto gerado!")
        st.text_area("Resultado", resposta.choices[0].message.content, height=320)
        st.caption(f"Gerações hoje: {uso['contador']}/{MAX_DIARIO}")
