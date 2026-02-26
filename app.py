import os
import json
import datetime
import streamlit as st
import feedparser
import pandas as pd
from collections import Counter
from openai import OpenAI

st.set_page_config(page_title="Radar de Notícias", layout="centered")

# ==============================
# CONFIGURAÇÕES GERAIS
# ==============================

PROFILE_FILE = "perfil.json"
USO_FILE = "uso_ia.json"
MAX_DIARIO = 3

TEMAS = ["Política", "Economia", "Esporte", "Moda", "Cultura", "Educação", "Segurança", "Saúde"]

PALAVRAS_TEMA = {
    "Política": ["governo","congresso","prefeitura","vereador","deputado","ministro","partido","eleição"],
    "Economia": ["inflação","juros","pib","mercado","emprego","renda","investimento"],
    "Esporte": ["campeonato","time","atleta","partida","vitória","gol","copa"],
    "Moda": ["coleção","tendência","look","desfile","marca","estilo"],
    "Cultura": ["show","festival","arte","cinema","teatro","livro"],
    "Educação": ["escola","professor","aluno","enem","universidade","curso"],
    "Segurança": ["polícia","crime","assalto","roubo","prisão","investigação"],
    "Saúde": ["hospital","sus","vacina","doença","médico","tratamento"]
}

# ==============================
# FUNÇÕES AUXILIARES
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
            "linhas": 10
        }

def save_profile(p):
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)

def carregar_uso():
    try:
        with open(USO_FILE, "r") as f:
            return json.load(f)
    except:
        return {"data": str(datetime.date.today()), "contador": 0}

def salvar_uso(dados):
    with open(USO_FILE, "w") as f:
        json.dump(dados, f)

def verificar_limite():
    uso = carregar_uso()
    hoje = str(datetime.date.today())

    if uso["data"] != hoje:
        uso = {"data": hoje, "contador": 0}
        salvar_uso(uso)

    return uso

def normalizar_locais(txt):
    return [x.strip() for x in txt.split(",") if x.strip()]

# ==============================
# INÍCIO DO APP
# ==============================

profile = load_profile()

st.title("📡 Radar de Notícias")
st.caption(f"{profile['nome_portal']} • {profile['assinatura']}")

tabs = st.tabs(["Identidade", "Inteligência", "Produção"])

# ==============================
# ABA IDENTIDADE
# ==============================

with tabs[0]:

    nome = st.text_input("Nome do Portal", profile["nome_portal"])
    assinatura = st.text_input("Assinatura", profile["assinatura"])
    estilo = st.selectbox("Estilo", ["jornalistico","analitico","opinativo","didatico"])
    linhas = st.slider("Quantidade aproximada de linhas", 4, 20, profile["linhas"])
    intencao = st.text_area("Intenção comunicativa")

    if st.button("Salvar Perfil"):
        save_profile({
            "nome_portal": nome,
            "assinatura": assinatura,
            "estilo": estilo,
            "intencao_comunicativa": intencao,
            "linhas": linhas
        })
        st.success("Perfil salvo!")

# ==============================
# ABA INTELIGÊNCIA
# ==============================

with tabs[1]:

    st.subheader("Escopo da Varredura")

    tema_do_dia = st.selectbox("Tema", TEMAS)
    local_do_dia = st.text_input("Local (ex: RN, Natal)")
    itens_por_fonte = st.slider("Itens por fonte", 5, 30, 10)

    sites = {
        "Tribuna do Norte": "https://tribunadonorte.com.br/feed/",
        "Agora RN": "https://agorarn.com.br/feed/"
    }

    if st.button("Vamos lá 🚀 Executar varredura"):

        resultados = []

        for nome_site, url in sites.items():
            feed = feedparser.parse(url)

            for entry in feed.entries[:itens_por_fonte]:

                texto = (entry.title + " " + entry.get("summary","")).lower()
                score = 0

                for w in PALAVRAS_TEMA.get(tema_do_dia, []):
                    if w in texto:
                        score += 2

                for loc in normalizar_locais(local_do_dia):
                    if loc.lower() in texto:
                        score += 4

                resultados.append({
                    "fonte": nome_site,
                    "titulo": entry.title,
                    "link": entry.link,
                    "resumo": entry.get("summary",""),
                    "score": score
                })

        df = pd.DataFrame(resultados)

        if not df.empty:
            df = df.sort_values("score", ascending=False)
            st.session_state["ranking"] = df
            st.dataframe(df.head(10))
        else:
            st.warning("Nenhum resultado encontrado.")

# ==============================
# ABA PRODUÇÃO
# ==============================

with tabs[2]:

    api_key = st.secrets.get("OPENAI_API_KEY", None)

    if not api_key:
        st.error("Chave OPENAI_API_KEY não configurada.")
        st.stop()

    if "ranking" not in st.session_state:
        st.warning("Execute a varredura antes.")
        st.stop()

    df = st.session_state["ranking"]

    escolha = st.selectbox(
        "Escolha a matéria",
        range(len(df)),
        format_func=lambda i: df.iloc[i]["titulo"]
    )

    if st.button("Gerar texto com IA ✍️"):

        uso = verificar_limite()

        if uso["contador"] >= MAX_DIARIO:
            st.error("Limite diário de 3 gerações atingido.")
            st.stop()

        materia = df.iloc[escolha]

        prompt = f"""
Você é redator do portal {profile['nome_portal']}.
Estilo: {profile['estilo']}
Intenção: {profile['intencao_comunicativa']}
Escreva cerca de {profile['linhas']} linhas.

Base:
Título: {materia['titulo']}
Resumo: {materia['resumo']}
"""

        client = OpenAI(api_key=api_key)

        resposta = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Você escreve textos jornalísticos em português do Brasil."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        uso["contador"] += 1
        salvar_uso(uso)

        st.success("Texto gerado!")
        st.text_area("Resultado", resposta.choices[0].message.content, height=300)
        st.caption(f"Gerações hoje: {uso['contador']}/3")
