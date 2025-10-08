# app.py
import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse

st.set_page_config(page_title="Capillary Docs Chatbot", layout="wide")

@st.cache_data
def load_docs(path="data.json"):
    with open(path, "r", encoding="utf-8") as f:
        pages = json.load(f)
    # chunk each page content into smaller pieces
    chunks = []
    def chunk_text(text, max_words=200):
        words = text.split()
        for i in range(0, len(words), max_words):
            yield " ".join(words[i:i+max_words])
    for p in pages:
        for chunk in chunk_text(p.get("content",""), 200):
            if len(chunk.strip()) > 50:
                chunks.append({"text": chunk, "url": p.get("url"), "title": p.get("title")})
    return chunks

@st.cache_data
def build_index(chunks):
    texts = [c['text'] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(texts)
    return vectorizer, doc_vectors

chunks = load_docs("data.json")
if not chunks:
    st.error("No docs found. Run scraper.py to produce data.json first.")
    st.stop()#

vectorizer, doc_vectors = build_index(chunks)

st.title("📘 Capillary Docs Chatbot")
st.write("Ask questions about Capillary documentation (scraped).")

query = st.text_input("Type your question here and press Enter:")

def get_answers(q, top_k=3):
    qv = vectorizer.transform([q])
    sims = cosine_similarity(qv, doc_vectors).flatten()
    idxs = sims.argsort()[::-1][:top_k]
    results = []
    for i in idxs:
        results.append({
            "score": float(sims[i]),
            "text": chunks[i]["text"],
            "url": chunks[i]["url"],
            "title": chunks[i]["title"]
        })
    return results

if query:
    res = get_answers(query, top_k=4)
    st.markdown("### Top results")
    for r in res:
        st.markdown(f"*Score:* {r['score']:.3f} — *Source:* [{r['title']}]({r['url']})")
        st.write(r['text'][:800] + ("..." if len(r['text'])>800 else ""))
        st.write("---")

st.sidebar.header("Testing queries")
st.sidebar.write("""
Examples:
- Where to get user activity log
- How to access Entity Audit Logs API
- Behavioral Events tracking
""")

