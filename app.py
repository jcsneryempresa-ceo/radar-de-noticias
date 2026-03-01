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
from google import genai

# ==============================
# CONFIG E VARIÁVEIS GLOBAIS
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
# FUNÇÕES DE PERSISTÊNCIA (JSON)
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

def verificar_uso():
    uso = read_json(USO_FILE, {"data": str(dt.date.today()), "contador": 0})
    if uso["data"] != str(dt.date.today()):
        uso = {"data": str(dt.date.today()), "contador": 0}
    return uso

# ==============================
# PROCESSAMENTO DE TEXTO E BUSCA
# ==============================

def normalizar_locais(txt):
    return [x.strip() for x in (txt or "").split(",") if x.strip()]

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

def gemini_generate(prompt):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: raise RuntimeError("API Key ausente nos Secrets.")
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
    if not resp.text: raise RuntimeError("IA não retornou texto.")
    return resp.text.strip()

# ==============================
# INTERFACE PRINCIPAL
# ==============================

profile = load_profile()
st.title("📡 Radar de Notícias")
st.caption(f"{profile['nome_portal']} • {profile['assinatura']}")

# Containers globais - Evitam NameError ao trocar de aba
palavras_box = st.sidebar.empty()
assuntos_box = st.sidebar.empty()
st.sidebar.divider()
st.sidebar.info("As sugestões aparecerão aqui após a busca.")

tabs = st.tabs(["Radar", "Redação", "Perfil"])

# --- ABA PERFIL ---
with tabs[2]:
    st.subheader("Configurações Editorial")
    with st.form("perfil_form"):
        p_nome = st.text_input("Nome do Portal", profile['nome_portal'])
        p_redator = st.text_input("Redator Responsável", profile['nome_redator'])
        p_assinatura = st.text_input("Assinatura", profile['assinatura'])
        p_estilo = st.selectbox("Estilo de Escrita", ["jornalistico", "analitico", "opinativo", "didatico"], 
                               index=["jornalistico", "analitico", "opinativo", "didatico"].index(profile['estilo']))
        p_linhas = st.slider("Quantidade de linhas", 5, 30, profile['linhas'])
        p_intencao = st.text_area("Intenção Comunicativa", profile['intencao_comunicativa'])
        
        if st.form_submit_button("Salvar Alterações"):
            profile.update({
                "nome_portal": p_nome, "nome_redator": p_redator, "assinatura": p_assinatura,
                "estilo": p_estilo, "linhas": p_linhas, "intencao_comunicativa": p_intencao
            })
            write_json(PROFILE_FILE, profile)
            st.success("Perfil atualizado com sucesso!")

# --- ABA RADAR ---
with tabs[0]:
    colA, colB = st.columns(2)
    with colA:
        editorial = st.selectbox("Editorial", EDITORIAIS)
        local_input = st.text_input("Localização", "Brasil")
    with colB:
        janela = st.selectbox("Janela de tempo", ["24h", "7 dias", "30 dias"])
        assunto_manual = st.text_input("Assunto específico (Opcional)")

    st.markdown("### Planejamento")
    agenda = read_json(AGENDA_FILE, {"itens": []})
    a_fazer = [x for x in agenda["itens"] if x.get("status") != "feita"]
    if a_fazer:
        for it in a_fazer[:3]:
            st.caption(f"📅 {it.get('data')} {it.get('hora')} - {it.get('titulo')[:50]}... ({it.get('canal')})")

    st.divider()

    if st.button("Executar Varredura"):
        agora = datetime.now(BRT)
        locais_norm = normalizar_locais(local_input)
        
        # Define busca
        palavras_score = PALAVRAS_TEMA.get(editorial, []) if editorial != "livre" else tokenize(assunto_manual)
        q_termo = assunto_manual if assunto_manual else editorial
        q_tempo = {"24h": "1d", "7 dias": "7d", "30 dias": "30d"}[janela]
        
        url = f"https://news.google.com/rss/search?q={q_termo}+{local_input}+when:{q_tempo}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        feed = feedparser.parse(url)
        
        results = []
        for entry in feed.entries[:20]:
            dt_item = extrair_data_entry(entry)
            titulo = entry.title
            sc = score_texto(titulo, palavras_score, locais_norm)
            
            results.append({
                "quando": marcador_tempo(dt_item, agora),
                "titulo": titulo,
                "link": entry.link,
                "score": sc,
                "fonte": "Google News"
            })
        
        df = pd.DataFrame(results).sort_values("score", ascending=False)
        st.session_state["ranking"] = df
        
        # Atualiza Sidebar
        tokens = tokenize(" ".join(df["titulo"]))
        top = [w[0].upper() for w in Counter(tokens).most_common(5)]
        palavras_box.info(f"EM ALTA: {', '.join(top)}")
        assuntos_box.info(f"TOP TEMAS:\n" + "\n".join(df["titulo"].head(3).values))
        
        st.dataframe(df[["quando", "titulo", "score", "link"]], use_container_width=True)

# --- ABA REDAÇÃO ---
with tabs[1]:
    if "ranking" not in st.session_state or st.session_state["ranking"].empty:
        st.warning("⚠️ Realize uma busca no Radar primeiro.")
    else:
        df_red = st.session_state["ranking"]
        idx_sel = st.selectbox("Matéria Selecionada", range(len(df_red)), format_func=lambda i: df_red.iloc[i]["titulo"])
        materia = df_red.iloc[idx_sel]
        
        c1, c2, c3 = st.columns(3)
        with c1: canal_sel = st.selectbox("Canal de Destino", ["Instagram", "WhatsApp", "Facebook", "Site"])
        with c2: data_sel = st.date_input("Data de Postagem", dt.date.today())
        with c3: hora_sel = st.time_input("Hora", dt.time(18, 0))

        uso_atual = verificar_uso()
        st.write(f"Cota de IA: {uso_atual['contador']}/{MAX_DIARIO}")

        if st.button("Gerar Postagem"):
            if uso_atual["contador"] >= MAX_DIARIO:
                st.error("Limite diário esgotado.")
            else:
                prompt_ia = f"""
                Aja como redator do {profile['nome_portal']}.
                Estilo: {profile['estilo']}. Tamanho: {profile['linhas']} linhas.
                Intenção: {profile['intencao_comunicativa']}.
                
                Notícia: {materia['titulo']}
                Link: {materia['link']}
                Canal: {canal_sel}
                Assinatura: {profile['assinatura']}
                """
                try:
                    with st.spinner("IA processando..."):
                        texto_final = gemini_generate(prompt_ia)
                    
                    st.success("Conteúdo Gerado!")
                    st.text_area("Resultado Final", texto_final, height=350)
                    
                    # Salva Publicação e Atualiza Uso
                    uso_atual["contador"] += 1
                    write_json(USO_FILE, uso_atual)
                    
                    hist = read_json(PUBLICACOES_FILE, {"itens": []})
                    hist["itens"].append({
                        "id": str(uuid.uuid4()),
                        "data": str(data_sel),
                        "titulo": materia["titulo"],
                        "texto": texto_final,
                        "canal": canal_sel
                    })
                    write_json(PUBLICACOES_FILE, hist)
                    
                    # Add ao Planner (Agenda)
                    agenda["itens"].append({
                        "data": str(data_sel),
                        "hora": str(hora_sel),
                        "titulo": materia["titulo"][:50],
                        "canal": canal_sel,
                        "status": "a_fazer"
                    })
                    write_json(AGENDA_FILE, agenda)
                    
                except Exception as e:
                    st.error(f"Erro na geração: {e}")

# Histórico Rápido na Redação
st.divider()
st.subheader("Últimas Publicações")
pubs_recentes = read_json(PUBLICACOES_FILE, {"itens": []})
if pubs_recentes["itens"]:
    for p in pubs_recentes["itens"][-3:][::-1]:
        st.caption(f"{p['data']} - {p['canal']} - {p['titulo']}")
