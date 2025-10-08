import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Capillary Docs Chatbot", layout="wide")

# --- DATA LOADING (Caching for speed) ---
@st.cache_data
def load_docs(path="data.json"):
    """Loads documentation chunks from the data.json file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            pages = json.load(f)
    except FileNotFoundError:
        st.warning(f"Warning: '{path}' not found. Please run the scraper tool first.")
        return []
    except json.JSONDecodeError:
        st.error(f"Error: '{path}' contains invalid JSON. Please check the file content.")
        return []
    
    chunks = []
    
    def chunk_text(text, max_words=200):
        words = text.split()
        # Yield chunks of text
        for i in range(0, len(words), max_words):
            yield " ".join(words[i:i+max_words])
            
    # Process each page into chunks
    for p in pages:
        for chunk in chunk_text(p.get("content",""), 200):
            # Only add chunks longer than 50 characters
            if len(chunk.strip()) > 50:
                chunks.append({"text": chunk, "url": p.get("url"), "title": p.get("title")})
    return chunks

# --- INDEX BUILDING (Caching for speed) ---
@st.cache_data
def build_index(chunks):
    """Builds a TF-IDF vectorizer and document vectors."""
    texts = [c['text'] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(texts)
    return vectorizer, doc_vectors

# --- MAIN APP LOGIC ---

chunks = load_docs("data.json")

# Check if data was loaded. If not, show error and return early.
if not chunks:
    st.error("No docs found or data.json is empty. Please ensure data.json has content.")
    # st.stop() line is removed to allow the app to display the error message.
else:
    vectorizer, doc_vectors = build_index(chunks)

st.title("📘 Capillary Docs Chatbot")
st.write("Ask questions about Capillary documentation (scraped).")

# If chunks are available, show the query box and search functionality
if chunks:
    query = st.text_input("Type your question here and press Enter:")

    def get_answers(q, top_k=3):
        """Finds top_k similar documents using Cosine Similarity."""
        # Use try-except block in case vectorizer wasn't built (shouldn't happen here)
        try:
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
        except Exception as e:
            st.error(f"Search failed: {e}")
            return []

    if query:
        res = get_answers(query, top_k=4)
        if res:
            st.markdown("### Top results")
            for r in res:
                st.markdown(f"**Score:** {r['score']:.3f} — **Source:** [{r['title']}]({r['url']})")
                st.write(r['text'][:800] + ("..." if len(r['text']) > 800 else ""))
                st.write("---")
        else:
            st.info("No matching results found in the documentation.")

    st.sidebar.header("Testing queries")
    st.sidebar.write("""
    Examples:
    - Where to get user activity log
    - How to access Entity Audit Logs API
    - Behavioral Events tracking
    """)
