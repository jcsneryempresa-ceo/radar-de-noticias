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
# CONFIGURAÇÕES E CONSTANTES
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
# HELPERS DE PERSISTÊNCIA (JSON)
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

def load_profile():
    return read_json(PROFILE_FILE, {
        "nome_portal": "Radar de Notícias",
        "nome_redator": "José Nery",
        "assinatura": "jcsnery.empresa",
        "estilo": "jornalistico",
        "linhas": 10,
        "intencao_comunicativa": "neutro",
    })

def verificar_limite():
    uso = read_json(USO_FILE, {"data": str(dt.date.today()), "contador": 0})
    if uso.get("data") != str(dt.date.today()):
        uso = {"data": str(dt.date.today()), "contador": 0}
        write_json(USO_FILE, uso)
    return uso

def load_agenda(): return read_json(AGENDA_FILE, {"itens": []})
def save_agenda(agenda): write_json(AGENDA_FILE, agenda)
def load_publicacoes(): return read_json(PUBLICACOES_FILE, {"itens": []})
def save_publicacoes(pub): write_json(PUBLICACOES_FILE, pub)

# ==============================
# AUXILIARES DE TEXTO E BUSCA
# ==============================

def normalizar_locais(txt: str):
    return [x.strip() for x in (txt or "").split(",") if x.strip()]

def extrair_data_entry(entry):
    if getattr(entry, "published_parsed", None):
        dtu = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        return dtu.astimezone(BRT)
    return None

def marcador_tempo(dt_item, agora):
    if dt_item is None: return "SEM DATA"
    delta = agora - dt_item
    horas = delta.total_seconds() / 3600
    if horas < 24: return "HOJE"
    if delta.days == 1: return "ONTEM"
    return f"{delta.days} dias"

def tokenize(texto: str):
    texto = (texto or "").lower()
    texto = re.sub(r"[^a-zà-ú0-9\s-]", " ", texto)
    return [p for p in texto.split() if len(p) > 2 and p not in STOPWORDS_PT]

def score_texto(texto: str, palavras: list[str], locais: list[str]):
    t = (texto or "").lower()
    sc = 0
    for w in palavras:
        if w.lower() in t: sc += 2
    for loc in locais:
        if loc.lower() in t: sc += 4
    return sc

def montar_google_news_url(assunto: str, local: str, janela: str):
    consulta = f"{assunto.strip()} {local.strip()}".strip() or "notícias"
    when = {"24h": "1d", "7 dias": "7d", "30 dias": "30d"}.get(janela, "1d")
    q = f"{consulta} when:{when}".replace(" ", "+")
    return f"https://news.google.com/rss/search?q={q}&hl=pt-BR&gl=BR&ceid=BR:pt-419"

# ==============================
# FUNÇÃO GEMINI (SDK ESTÁVEL)
# ==============================

def gemini_generate(prompt: str, temperature: float = 0.7, max_output_tokens: int = 900) -> str:
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Chave GEMINI_API_KEY não configurada nos Secrets.")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    resp = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
    )
    
    if not resp.text:
        raise RuntimeError("A IA não retornou texto.")
    return resp.text.strip()

# ==============================
# INTERFACE STREAMLIT
# ==============================

profile = load_profile()
st.title("📡 Radar de Notícias")
st.caption(f"{profile['nome_portal']} • {profile.get('assinatura', 'jcsnery.empresa')}")

# ✅ Containers declarados globalmente para evitar o NameError
palavras_box = st.sidebar.empty()
assuntos_box = st.sidebar.empty()
st.sidebar.divider()

tabs = st.tabs(["Radar", "Redação", "Perfil"])

# --- TAB: PERFIL ---
with tabs[2]:
    st.subheader("Configurações do Redator")
    nome_p = st.text_input("Nome do Portal", profile.get("nome_portal"))
    nome_r = st.text_input("Nome do Redator", profile.get("nome_redator"))
    estilo = st.selectbox("Estilo", ["jornalistico", "analitico", "opinativo"], index=0)
    linhas = st.slider("Quantidade de linhas", 4, 25, int(profile.get("linhas", 10)))
    intencao = st.text_area("Intenção Comunicativa", profile.get("intencao_comunicativa"))
    
    if st.button("Salvar Perfil"):
        profile.update({"nome_portal": nome_p, "nome_redator": nome_r, "estilo": estilo, "linhas": linhas, "intencao_comunicativa": intencao})
        write_json(PROFILE_FILE, profile)
        st.success("Perfil atualizado!")

# --- TAB: RADAR ---
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        ed = st.selectbox("Editorial", EDITORIAIS)
        janela = st.selectbox("Janela de tempo", ["24h", "7 dias", "30 dias"])
    with col2:
        local_input = st.text_input("Localização", "Brasil")
        assunto_l = st.text_input("Assunto específico")

    st.markdown("### Guia de Publicações")
    agenda = load_agenda()
    a_fazer = [x for x in agenda["itens"] if x.get("status") != "feita"]
    if a_fazer:
        for it in a_fazer[:3]:
            st.caption(f"📅 {it.get('data')} {it.get('hora')} - {it.get('titulo')}")

    if st.button("Buscar Notícias"):
        agora = datetime.now(BRT)
        locais_norm = normalizar_locais(local_input)
        palavras_score = PALAVRAS_TEMA.get(ed, []) if ed != "livre" else tokenize(assunto_l)
        
        url = montar_google_news_url(assunto_l or ed, local_input, janela)
        feed = feedparser.parse(url)
        
        resultados = []
        for entry in feed.entries[:20]:
            dt_item = extrair_data_entry(entry)
            titulo = entry.title
            sc = score_texto(titulo, palavras_score, locais_norm)
            resultados.append({
                "quando": marcador_tempo(dt_item, agora),
                "data_txt": dt_item.strftime("%d/%m %H:%M") if dt_item else "---",
                "titulo": titulo,
                "link": entry.link,
                "fonte": "Google News",
                "score": sc
            })
        
        df = pd.DataFrame(resultados).sort_values("score", ascending=False)
        st.session_state["ranking"] = df
        
        # Atualiza Sidebar
        tokens = tokenize(" ".join(df["titulo"]))
        top = [w[0].upper() for w in Counter(tokens).most_common(5)]
        palavras_box.info(f"EM ALTA: {', '.join(top)}")
        assuntos_box.info(f"ASSUNTOS:\n" + "\n".join(df["titulo"].head(3).values))
        
        st.dataframe(df[["quando", "titulo", "score", "link"]], use_container_width=True)

# --- TAB: REDAÇÃO ---
with tabs[1]:
    st.subheader("Redação Assistida")
    if "ranking" not in st.session_state or st.session_state["ranking"].empty:
        st.warning("⚠️ Realize uma busca no Radar primeiro.")
    else:
        df_red = st.session_state["ranking"]
        idx = st.selectbox("Selecione a matéria", range(len(df_red)), format_func=lambda i: df_red.iloc[i]["titulo"])
        materia = df_red.iloc[idx]
        
        c1, c2, c3 = st.columns(3)
        with c1: canal = st.selectbox("Canal", ["Instagram", "WhatsApp", "Site"])
        with c2: data_p = st.date_input("Data", dt.date.today())
        with c3: hora_p = st.time_input("Hora", dt.time(18, 0))

        uso = verificar_limite()
        st.caption(f"Cota de IA: {uso['contador']}/{MAX_DIARIO}")

        if st.button("Gerar Texto"):
            if uso["contador"] >= MAX_DIARIO:
                st.error("Limite diário esgotado.")
            else:
                prompt = f"""
                Redija como redator do {profile['nome_portal']}.
                Estilo: {profile['estilo']}. Tamanho: {profile['linhas']} linhas.
                Notícia: {materia['titulo']}
                Link: {materia['link']}
                Canal: {canal}
                """
                try:
                    texto = gemini_generate(prompt)
                    st.text_area("Resultado", texto, height=350)
                    
                    uso["contador"] += 1
                    write_json(USO_FILE, uso)
                    
                    hist = load_publicacoes()
                    hist["itens"].append({"id": str(uuid.uuid4()), "data": str(data_p), "titulo": materia["titulo"], "canal": canal})
                    save_publicacoes(hist)
                    
                    agenda["itens"].append({"data": str(data_p), "hora": str(hora_p), "titulo": materia["titulo"][:50], "canal": canal, "status": "a_fazer"})
                    save_agenda(agenda)
                    st.success("Conteúdo gerado e agendado!")
                except Exception as e:
                    st.error(f"Erro na IA: {e}")

st.divider()
st.subheader("Histórico Recente")
hist_view = load_publicacoes()
if hist_view["itens"]:
    for h in hist_view["itens"][-3:][::-1]:
        st.caption(f"✅ {h['data']} - {h['titulo']} ({h['canal']})")
