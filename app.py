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
# CONFIGURAÇÕES
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

STOPWORDS_PT = {"a", "o", "os", "as", "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas", "para", "por", "com", "sem", "um", "uma", "uns", "umas", "e", "ou", "ao", "aos", "à", "às", "que", "se", "sua", "seu", "suas", "seus", "são", "ser", "foi", "é", "vai", "já", "mais", "menos", "entre", "sobre", "contra", "após", "antes", "durante", "até", "isso", "essa", "esse", "este", "esta", "esses", "essas", "como", "quando", "onde", "porque", "pra", "pela", "pelo", "pelas", "pelos"}

# ==============================
# HELPERS
# ==============================

def read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except: return default

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_profile():
    return read_json(PROFILE_FILE, {"nome_portal": "Radar", "nome_redator": "José Nery", "assinatura": "jcsnery.empresa", "estilo": "jornalistico", "linhas": 10, "intencao_comunicativa": "neutro"})

def verificar_limite():
    uso = read_json(USO_FILE, {"data": str(dt.date.today()), "contador": 0})
    if uso.get("data") != str(dt.date.today()):
        uso = {"data": str(dt.date.today()), "contador": 0}
        write_json(USO_FILE, uso)
    return uso

def tokenize(texto):
    texto = (texto or "").lower()
    tokens = re.sub(r"[^a-zà-ú0-9\s-]", " ", texto).split()
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS_PT]

def score_texto(texto, palavras, locais):
    t = (texto or "").lower()
    sc = 0
    for w in palavras:
        if w.lower() in t: sc += 2
    for loc in locais:
        if loc.lower() in t: sc += 4
    return sc

def extrair_data_entry(entry):
    if getattr(entry, "published_parsed", None):
        dtu = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        return dtu.astimezone(BRT)
    return None

def marcador_tempo(dt_item, agora):
    if not dt_item: return "SEM DATA"
    delta = agora - dt_item
    horas = delta.total_seconds() / 3600
    if horas < 24: return "HOJE"
    if delta.days == 1: return "ONTEM"
    return f"{delta.days} dias"

# ==============================
# GEMINI (ESTÁVEL)
# ==============================

def gemini_generate(prompt):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: raise RuntimeError("Chave ausente.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    if not resp.text: raise RuntimeError("IA sem resposta.")
    return resp.text.strip()

# ==============================
# INTERFACE
# ==============================

profile = load_profile()
st.title("📡 Radar de Notícias")
st.caption(f"{profile['nome_portal']} • {profile.get('assinatura','jcsnery.empresa')}")

# Containers globais para evitar NameError
palavras_box = st.sidebar.empty()
assuntos_box = st.sidebar.empty()

tabs = st.tabs(["Radar", "Redação", "Perfil"])

# --- ABA PERFIL ---
with tabs[2]:
    st.subheader("Perfil editorial")
    nome_p = st.text_input("Portal", profile.get("nome_portal"))
    nome_r = st.text_input("Redator", profile.get("nome_redator"))
    estilo = st.selectbox("Estilo", ["jornalistico", "analitico", "opinativo"], index=0)
    linhas = st.slider("Linhas", 4, 25, int(profile.get("linhas", 10)))
    if st.button("Salvar Perfil"):
        profile.update({"nome_portal": nome_p, "nome_redator": nome_r, "estilo": estilo, "linhas": linhas})
        write_json(PROFILE_FILE, profile)
        st.success("Salvo!")

# --- ABA RADAR ---
with tabs[0]:
    colA, colB = st.columns(2)
    with colA:
        ed = st.selectbox("Editorial", EDITORIAIS)
        jan = st.selectbox("Janela", ["24h", "7 dias", "30 dias"])
    with colB:
        loc = st.text_input("Localização", "Brasil")
        assunto_l = st.text_input("Assunto específico")

    # Agenda rápida
    agenda = read_json(AGENDA_FILE, {"itens": []})
    a_fazer = [x for x in agenda["itens"] if x.get("status") != "feita"]
    if a_fazer:
        st.caption(f"Próximo: {a_fazer[0].get('titulo')}")

    if st.button("Buscar notícias"):
        agora = datetime.now(BRT)
        q_tempo = {"24h": "1d", "7 dias": "7d", "30 dias": "30d"}[jan]
        url = f"https://news.google.com/rss/search?q={assunto_l or ed}+{loc}+when:{q_tempo}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        feed = feedparser.parse(url)
        
        results = []
        for e in feed.entries[:15]:
            dt_item = extrair_data_entry(e)
            results.append({
                "quando": marcador_tempo(dt_item, agora),
                "titulo": e.title,
                "link": e.link,
                "score": score_texto(e.title, PALAVRAS_TEMA.get(ed, []), [loc])
            })
        
        df = pd.DataFrame(results).sort_values("score", ascending=False)
        st.session_state["ranking"] = df
        
        # Atualiza caixas globais
        tokens = tokenize(" ".join(df["titulo"]))
        top = [w[0].upper() for w in Counter(tokens).most_common(5)]
        palavras_box.info(f"EM ALTA: {', '.join(top)}")
        assuntos_box.info("\n".join(df["titulo"].head(3)))
        st.dataframe(df[["quando", "titulo", "link"]], use_container_width=True)

# --- ABA REDAÇÃO ---
with tabs[1]:
    if "ranking" not in st.session_state or st.session_state["ranking"].empty:
        st.warning("Busque no Radar antes.")
    else:
        df = st.session_state["ranking"]
        sel = st.selectbox("Matéria", range(len(df)), format_func=lambda i: df.iloc[i]["titulo"])
        materia = df.iloc[sel]
        
        canal = st.selectbox("Canal", ["Instagram", "WhatsApp", "Site"])
        uso = verificar_limite()
        st.caption(f"Cota: {uso['contador']}/{MAX_DIARIO}")

        if st.button("Gerar publicação"):
            if uso["contador"] >= MAX_DIARIO:
                st.error("Limite atingido.")
            else:
                prompt = f"Escreva um texto {profile['estilo']} de {profile['linhas']} linhas para {canal} sobre: {materia['titulo']}. Link: {materia['link']}"
                try:
                    texto = gemini_generate(prompt)
                    st.text_area("Resultado", texto, height=300)
                    
                    # Atualiza dados
                    uso["contador"] += 1
                    write_json(USO_FILE, uso)
                    
                    hist = read_json(PUBLICACOES_FILE, {"itens": []})
                    hist["itens"].append({"data": str(dt.date.today()), "titulo": materia["titulo"], "canal": canal})
                    write_json(PUBLICACOES_FILE, hist)
                    
                    st.success("Gerado!")
                except Exception as e:
                    st.error(f"Erro: {e}")
