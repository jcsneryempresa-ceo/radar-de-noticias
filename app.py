import json
import re
import uuid
import datetime as dt
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from collections import Counter

import streamlit as st
import feedparser
import pandas as pd
import google.generativeai as genai

# ==============================
# CONFIG
# ==============================

st.set_page_config(page_title="Radar de Notícias", layout="centered")

BRT = ZoneInfo("America/Fortaleza")

PROFILE_FILE = "perfil.json"
USO_FILE = "uso_ia.json"
AGENDA_FILE = "agenda.json"
PUBLICACOES_FILE = "publicacoes.json"

MAX_DIARIO = 3

EDITORIAIS = ["livre", "Política", "Economia", "Esporte", "Moda", "Cultura", "Educação", "Segurança", "Saúde"]

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

STOPWORDS_PT = {
    "a", "o", "os", "as", "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
    "para", "por", "com", "sem", "um", "uma", "uns", "umas", "e", "ou", "ao", "aos", "à",
    "às", "que", "se", "sua", "seu", "suas", "seus", "são", "ser", "foi", "é", "vai", "já",
    "mais", "menos", "entre", "sobre", "contra", "após", "antes", "durante", "até", "isso",
    "essa", "esse", "este", "esta", "esses", "essas", "como", "quando", "onde", "porque",
    "pra", "pela", "pelo", "pelas", "pelos",
}

# ==============================
# JSON helpers
# ==============================

def read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ==============================
# Profile / usage
# ==============================

def load_profile():
    return read_json(PROFILE_FILE, {
        "nome_portal": "Radar de Notícias",
        "nome_redator": "José Nery",
        "assinatura": "jcsnery.empresa",
        "estilo": "jornalistico",
        "linhas": 10,
        "intencao_comunicativa": "neutro",
    })

def save_profile(p):
    write_json(PROFILE_FILE, p)

def carregar_uso():
    return read_json(USO_FILE, {"data": str(dt.date.today()), "contador": 0})

def salvar_uso(dados):
    write_json(USO_FILE, dados)

def verificar_limite():
    uso = carregar_uso()
    hoje = str(dt.date.today())
    if uso.get("data") != hoje:
        uso = {"data": hoje, "contador": 0}
        salvar_uso(uso)
    return uso

# ==============================
# Agenda / publicações
# ==============================

def load_agenda():
    return read_json(AGENDA_FILE, {"itens": []})

def save_agenda(agenda):
    write_json(AGENDA_FILE, agenda)

def load_publicacoes():
    return read_json(PUBLICACOES_FILE, {"itens": []})

def save_publicacoes(pub):
    write_json(PUBLICACOES_FILE, pub)

# ==============================
# Text helpers
# ==============================

def normalizar_locais(txt: str):
    return [x.strip() for x in (txt or "").split(",") if x.strip()]

def extrair_data_entry(entry):
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

def tokenize(texto: str):
    texto = (texto or "").lower()
    texto = re.sub(r"[^a-zà-ú0-9\s-]", " ", texto)
    parts = [p.strip() for p in texto.split() if p.strip()]
    out = []
    for p in parts:
        if len(p) < 3:
            continue
        if p in STOPWORDS_PT:
            continue
        out.append(p)
    return out

def score_texto(texto: str, palavras: list[str], locais: list[str]):
    t = (texto or "").lower()
    sc = 0
    for w in palavras:
        if w and w.lower() in t:
            sc += 2
    for loc in locais:
        if loc and loc.lower() in t:
            sc += 4
    return sc

def montar_google_news_url(assunto: str, local: str, janela: str):
    consulta = " ".join([x for x in [assunto.strip(), local.strip()] if x]).strip()
    if not consulta:
        consulta = "notícias"
    when_map = {"24h": "1d", "7 dias": "7d", "30 dias": "30d"}
    when = when_map.get(janela)
    if when:
        consulta = f"{consulta} when:{when}"
    q = consulta.replace(" ", "+")
    return f"https://news.google.com/rss/search?q={q}&hl=pt-BR&gl=BR&ceid=BR:pt-419"

# ==============================
# Gemini (Versão Estável)
# ==============================

def gemini_generate(prompt: str, temperature: float = 0.7, max_output_tokens: int = 900) -> str:
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        raise RuntimeError("Chave GEMINI_API_KEY não configurada.")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    resp = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
    )
    
    texto = (getattr(resp, "text", "") or "").strip()
    if not texto:
        raise RuntimeError("A IA não retornou texto.")
    return texto

# ==============================
# UI
# ==============================

profile = load_profile()

st.title("📡 Radar de Notícias")
st.caption(f"{profile['nome_portal']} • {profile.get('assinatura','jcsnery.empresa')}")

# Declaração Global para evitar NameError
palavras_box = st.sidebar.empty()
assuntos_box = st.sidebar.empty()

tabs = st.tabs(["Radar", "Redação", "Perfil"])

# --- PERFIL ---
with tabs[2]:
    st.subheader("Perfil editorial")
    nome_portal = st.text_input("Nome do Portal", value=profile.get("nome_portal", ""))
    nome_redator = st.text_input("Nome do Redator", value=profile.get("nome_redator", ""))
    estilo = st.selectbox("Estilo", ["jornalistico", "analitico", "opinativo"], index=0)
    linhas = st.slider("Linhas", 4, 20, int(profile.get("linhas", 10)))
    intencao = st.text_area("Intenção", value=profile.get("intencao_comunicativa", ""))
    if st.button("Salvar Perfil"):
        profile.update({"nome_portal": nome_portal, "nome_redator": nome_redator, "estilo": estilo, "linhas": linhas, "intencao_comunicativa": intencao})
        save_profile(profile)
        st.success("Salvo!")

# --- RADAR ---
with tabs[0]:
    st.subheader("Radar")
    colA, colB = st.columns(2)
    with colA:
        editorial = st.selectbox("Editorial", EDITORIAIS, index=0)
        janela = st.selectbox("Janela de tempo", ["24h", "7 dias", "30 dias"], index=0)
        local = st.text_input("Local", placeholder="Brasil, RN...")
    
    st.markdown("### Guia de Publicações")
    agenda = load_agenda()
    a_fazer = [x for x in agenda["itens"] if x.get("status") != "feita"]
    if a_fazer:
        for it in a_fazer[:3]:
            st.caption(f"📅 {it.get('data')} - {it.get('titulo')}")

    st.divider()
    assunto_livre = st.text_input("Assunto específico")

    if st.button("Buscar notícias"):
        agora = datetime.now(BRT)
        locais_norm = normalizar_locais(local)
        palavras = PALAVRAS_TEMA.get(editorial, []) if editorial != "livre" else tokenize(assunto_livre)
        url = montar_google_news_url(assunto_livre or editorial, local, janela)
        feed = feedparser.parse(url)
        
        resultados = []
        for entry in feed.entries[:15]:
            dt_item = extrair_data_entry(entry)
            titulo = getattr(entry, "title", "")
            resultados.append({
                "quando": marcador_tempo(dt_item, agora),
                "data_txt": dt_item.strftime("%d/%m %H:%M") if dt_item else "---",
                "titulo": titulo,
                "link": getattr(entry, "link", ""),
                "resumo": entry.get("summary", ""),
                "fonte": "Google News",
                "score": score_texto(titulo, palavras, locais_norm)
            })
        
        df = pd.DataFrame(resultados).sort_values("score", ascending=False)
        st.session_state["ranking"] = df
        
        tokens = tokenize(" ".join(df["titulo"]))
        top = [w[0].upper() for w in Counter(tokens).most_common(5)]
        palavras_box.info(f"EM ALTA: {', '.join(top)}")
        assuntos_box.info("\n".join(df["titulo"].head(3)))
        st.dataframe(df[["quando", "titulo", "score", "link"]], use_container_width=True)

# --- REDAÇÃO ---
with tabs[1]:
    st.subheader("Redação")
    pub_hist = load_publicacoes()
    if pub_hist["itens"]:
        st.caption(f"Última: {pub_hist['itens'][-1].get('titulo')}")

    if "ranking" not in st.session_state:
        st.warning("Busque no Radar primeiro.")
    else:
        df = st.session_state["ranking"]
        sel = st.selectbox("Escolha a matéria", range(len(df)), format_func=lambda i: df.iloc[i]["titulo"])
        materia = df.iloc[sel]
        
        c1, c2, c3 = st.columns(3)
        with c1: canal = st.selectbox("Canal", ["Instagram", "WhatsApp", "Site"])
        with c2: data_p = st.date_input("Data", value=dt.date.today())
        with c3: hora_p = st.time_input("Hora", value=dt.time(18, 0))

        uso = verificar_limite()
        st.caption(f"Uso: {uso['contador']}/{MAX_DIARIO}")

        if st.button("Gerar publicação"):
            if uso["contador"] >= MAX_DIARIO:
                st.error("Limite atingido.")
            else:
                prompt = f"Redator do portal {profile['nome_portal']}. Estilo {profile['estilo']} em {profile['linhas']} linhas para {canal} sobre: {materia['titulo']}. Link: {materia['link']}"
                try:
                    texto = gemini_generate(prompt)
                    st.text_area("Resultado", texto, height=300)
                    
                    uso["contador"] += 1
                    salvar_uso(uso)
                    
                    pub_hist["itens"].append({"id": str(uuid.uuid4()), "data": str(data_p), "titulo": materia["titulo"], "canal": canal})
                    save_publicacoes(pub_hist)
                    
                    agenda["itens"].append({"data": str(data_p), "hora": str(hora_p), "titulo": materia["titulo"][:50], "canal": canal, "status": "a_fazer"})
                    save_agenda(agenda)
                    st.success("Gerado e agendado!")
                except Exception as e:
                    st.error(f"Erro: {e}")
                       
