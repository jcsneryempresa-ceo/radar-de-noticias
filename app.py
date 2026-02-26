import streamlit as st
import feedparser
import pandas as pd
from collections import Counter

st.set_page_config(page_title="Radar de Notícias", layout="centered")

st.title("📡 Radar de Notícias")
sites_default = {
# Configuração inicial
sites_default = {
    "Tribuna do Norte": "https://tribunadonorte.com.br/feed/",
    "Agora RN": "https://agorarn.com.br/feed/",
    "GE RN": "https://ge.globo.com/rss/ge/rn/"
}

if "sites" not in st.session_state:
st.session_state.sites = sites_default.copy()

if "palavras" not in st.session_state:
  st.session_state.sites = sites_default.copy()

if "palavras" not in st.session_state:
st.session_state.palavras = {
"RN": 3,
"Natal": 2,
"Parnamirim": 4,
"esporte": 2,
"amador": 5
}
st.header("🔎 Buscar Notícias")

if st.button("Buscar agora"):


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
            "score": score
        })

df = pd.DataFrame(resultados)

if not df.empty:
    df = df.sort_values(by="score", ascending=False)
    st.dataframe(df.head(10))

    melhor = df.iloc[0]
    st.success("📰 Notícia mais relevante:")
    st.write(melhor["titulo"])
    st.write(melhor["link"])
else:
    st.warning("Nenhuma notícia encontrada.")
  st.header("📊 Radar Editorial")

if st.button("Gerar relatório editorial"):
palavras_encontradas = Counter()

for nome, url in st.session_state.sites.items():
    feed = feedparser.parse(url)
    for entry in feed.entries[:10]:
        texto = (entry.title + " " + entry.get("summary", "")).lower()
        for palavra in st.session_state.palavras.keys():
            if palavra.lower() in texto:
                palavras_encontradas[palavra] += 1

if palavras_encontradas:
    dominante = palavras_encontradas.most_common(1)[0][0]
    st.write("Hoje o tema dominante é:", dominante)
else:
    st.write("Nenhuma tendência identificada.")
