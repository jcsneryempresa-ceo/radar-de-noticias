import os
import json
import streamlit as st
import feedparser
import pandas as pd
import datetime

# ---------- TEMAS (dicionários de apoio) ----------
TEMAS = [
    "Política", "Economia", "Esporte", "Moda",
    "Cultura", "Educação", "Segurança", "Saúde"
]

# Palavras-base por tema (você pode ajustar com o tempo)
TEMA_KEYWORDS = {
    "Política": ["governo", "congresso", "minist", "prefeit", "vereador", "deput", "senad", "elei", "partid", "gestão"],
    "Economia": ["inflação", "juros", "mercado", "pib", "dólar", "emprego", "renda", "invest", "tribut", "orçamento"],
    "Esporte": ["campeonato", "atleta", "time", "técnico", "torneio", "gol", "jogo", "liga", "seleção"],
    "Moda": ["coleção", "tendência", "look", "desfile", "estilo", "marca", "fashion", "roupa", "acessório"],
    "Cultura": ["festival", "show", "cinema", "teatro", "música", "arte", "exposição", "livro", "literatura"],
    "Educação": ["escola", "universidade", "enem", "ifrn", "aluno", "professor", "aula", "educação", "matrícula"],
    "Segurança": ["polícia", "crime", "prisão", "roubo", "assalto", "operação", "violência", "investigação", "suspeito"],
    "Saúde": ["hospital", "vacina", "doença", "sus", "médico", "saúde", "tratamento", "paciente", "epidemia"]
}

USO_FILE = "uso_ia.json"
MAX_DIARIO = 3

def carregar_uso():
    try:
        with open(USO_FILE, "r") as f:
            return json.load(f)
    except:
        return {"data": str(datetime.date.today()), "contador": 0}

def salvar_uso(dados):
    with open(USO_FILE, "w") as f:
        json.dump(dados, f)

def verificar_limite_diario():
    uso = carregar_uso()
    hoje = str(datetime.date.today())

    if uso["data"] != hoje:
        uso = {"data": hoje, "contador": 0}
        salvar_uso(uso)

    return uso
from collections import Counter
from openai import OpenAI

st.set_page_config(page_title="Radar de Notícias", layout="centered")

# ---------- PERFIL (carrega do JSON) ----------
PROFILE_FILE = "perfil.json"

def load_profile():
    try:
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "nome_portal": "Radar de Notícias",
            "assinatura": "jcsnery.empresa",
            "estilo": "jornalistico",
            "intencao_comunicativa": "neutro",
            "tamanho_padrao": "medio",
            "linhas": 10
        }

def save_profile(p):
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)

profile = load_profile()

# ---------- HEADER ----------
st.title("📡 Radar de Notícias")
st.caption(f"{profile.get('nome_portal','')} • {profile.get('assinatura','')}")

tabs = st.tabs(["Identidade & Diretrizes", "Inteligência", "Produção"])

# ---------- CONFIG DEFAULTS ----------
sites_default = {
    "Tribuna do Norte": "https://tribunadonorte.com.br/feed/",
    "Agora RN": "https://agorarn.com.br/feed/",
    "GE RN": "https://ge.globo.com/rss/ge/rn/"
}

if "sites" not in st.session_state:
    st.session_state.sites = sites_default.copy()

if "palavras" not in st.session_state:
    st.session_state.palavras = {"RN": 3, "Natal": 2, "Parnamirim": 4, "esporte": 2, "amador": 5}

if "ultimo_df" not in st.session_state:
    st.session_state.ultimo_df = None

# ---------- TAB 1: PERFIL ----------
with tabs[0]:
    st.subheader("Identidade & Diretrizes")

    col1, col2 = st.columns(2)
    with col1:
        nome_portal = st.text_input("Nome do portal", value=profile.get("nome_portal", "Radar de Notícias"))
        assinatura = st.text_input("Assinatura", value=profile.get("assinatura", "jcsnery.empresa"))
    with col2:
        estilo = st.selectbox(
            "Estilo de linguagem",
            ["jornalistico", "analitico", "didatico", "opinativo_leve", "jovem"],
            index=["jornalistico","analitico","didatico","opinativo_leve","jovem"].index(profile.get("estilo","jornalistico"))
        )
        linhas = st.slider("Quantidade de linhas (aprox.)", 4, 20, int(profile.get("linhas", 10)))

    intencao = st.text_area(
        "Intenção comunicativa (o que você quer que o texto deixe claro)",
        value=profile.get("intencao_comunicativa", ""),
        height=90,
        placeholder="Ex.: contextualizar, cobrar transparência, mostrar que é sistêmico, etc."
    )

    tamanho = st.selectbox(
        "Tamanho padrão",
        ["curto", "medio", "longo"],
        index=["curto","medio","longo"].index(profile.get("tamanho_padrao","medio"))
    )

    if st.button("Salvar diretrizes"):
        profile = {
            "nome_portal": nome_portal,
            "assinatura": assinatura,
            "estilo": estilo,
            "intencao_comunicativa": intencao,
            "tamanho_padrao": tamanho,
            "linhas": linhas
        }
        save_profile(profile)
        st.success("Diretrizes salvas. Vamos lá! ✅")

# ---------- TAB 2: RADAR ----------
with tabs[1]:
    st.subheader("Inteligência de Curadoria")

    with st.expander("Fontes (RSS) e Palavras-chave", expanded=False):
        st.write("Fontes ativas:")
        for nome, url in list(st.session_state.sites.items()):
            st.write(f"• {nome} — {url}")

        st.write("Pesos atuais:")
        st.write(st.session_state.palavras)

    if st.button("Vamos lá 🚀 Executar varredura editorial"):
        resultados = []
        for nome, url in st.session_state.sites.items():
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                texto = (entry.title + " " + entry.get("summary", "")).lower()
                score = 0
                for palavra, peso in st.session_state.palavras.items():
                    if palavra.lower() in texto:
                        score += peso
                resultados.append({
                    "fonte": nome,
                    "titulo": entry.title,
                    "link": entry.link,
                    "resumo": entry.get("summary", ""),
                    "score": score
                })

        df = pd.DataFrame(resultados)
        if not df.empty:
            df = df.sort_values(by="score", ascending=False)
            st.session_state.ultimo_df = df
            st.success("Varredura concluída. Ranking atualizado ✅")
            st.dataframe(df.head(12))
        else:
            st.warning("Nenhum resultado encontrado nas fontes atuais.")

    st.divider()
    st.subheader("Radar Editorial (tendência)")
    if st.session_state.ultimo_df is not None:
        texto_geral = " ".join((st.session_state.ultimo_df["titulo"].fillna("") + " " + st.session_state.ultimo_df["resumo"].fillna("")).tolist()).lower()
        cont = Counter()
        for palavra in st.session_state.palavras.keys():
            if palavra.lower() in texto_geral:
                cont[palavra] += texto_geral.count(palavra.lower())
        if cont:
            dominante = cont.most_common(1)[0][0]
            st.info(f"Tendência dominante (pelas suas palavras): {dominante}")
        else:
            st.info("Sem tendência clara pelas palavras configuradas (isso pode ser bom).")
    else:
        st.caption("Execute a varredura para ver tendências.")

# ---------- TAB 3: PRODUÇÃO (OpenAI) ----------
with tabs[2]:
    st.subheader("Produção Editorial")

    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        st.error("Chave de IA não configurada (OPENAI_API_KEY em Secrets).")
        st.stop()

    # Limite simples (proteção básica)
    if "gen_count" not in st.session_state:
        st.session_state.gen_count = 0
    MAX_GEN_SESSION = 10

    if st.session_state.ultimo_df is None or st.session_state.ultimo_df.empty:
        st.warning("Primeiro execute a varredura na aba Inteligência.")
        st.stop()

    df = st.session_state.ultimo_df.head(20).reset_index(drop=True)
    escolha = st.selectbox("Escolha uma matéria do ranking", options=list(range(len(df))), format_func=lambda i: f"{df.loc[i,'titulo']} ({df.loc[i,'fonte']})")

    formato = st.selectbox("Formato de saída", ["Título + Lide", "Título + Lide + 1º parágrafo", "Legenda Instagram (curta)"])
    social_link = st.text_input("Link social (opcional, só referência)", placeholder="Cole um link de post, se quiser")
    social_texto = st.text_area("Texto do post (opcional, recomendado se quiser reaproveitar)", height=90)

if st.button("Gerar texto com IA ✍️"):

    # 🔒 Verificar limite diário
    uso = verificar_limite_diario()

    if uso["contador"] >= MAX_DIARIO:
        st.error("Limite diário de gerações atingido (3 por dia). Tente amanhã.")
        st.stop()

    materia = df.loc[escolha].to_dict()

    regras_seguranca = (
        "Regras: não afirme acusações como fato sem atribuição. "
        "Use 'segundo a matéria', 'de acordo com', 'a investigação apura' quando houver alegações. "
        "Evite difamação. Mantenha linguagem responsável."
    )

    instrucao = f"""
Você é um redator para o portal "{profile.get('nome_portal')}".
Assinatura: "{profile.get('assinatura')}".
Estilo: {profile.get('estilo')}.
Intenção comunicativa: {profile.get('intencao_comunicativa')}.
Tamanho: {profile.get('tamanho_padrao')} com cerca de {profile.get('linhas')} linhas.
Formato solicitado: {formato}.
{regras_seguranca}

Base (matéria):
Título: {materia.get('titulo')}
Fonte: {materia.get('fonte')}
Link: {materia.get('link')}
Resumo: {materia.get('resumo')}

Insumo social (se houver):
Link: {social_link}
Texto: {social_texto}
"""

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Você escreve textos jornalísticos e conteúdos para redes sociais em PT-BR."},
            {"role": "user", "content": instrucao}
        ],
        temperature=0.7
    )

    uso["contador"] += 1
    salvar_uso(uso)

    st.success("Gerado ✅")
    st.text_area("Resultado", value=resp.choices[0].message.content, height=260)
    st.caption(f"Gerações hoje: {uso['contador']}/3")
