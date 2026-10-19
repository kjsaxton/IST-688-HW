try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass
 
import streamlit as st
import chromadb
import pandas as pd
from openai import OpenAI
 

CSV_PATH = "news.csv"
TITLE_COL = "YOUR_TITLE_COLUMN"   # <- rename to your CSV's title column
BODY_COL = "YOUR_BODY_COLUMN"     # <- rename to your CSV's article text column
DATE_COL = "YOUR_DATE_COLUMN"     # <- rename to your CSV's date column
 
MAX_DISTANCE = 0.6   # cosine distance cutoff; tune it (lower = stricter)
 
SYSTEM = (
    "You are a news assistant for a global law firm that monitors news about its clients. "
    "Use ONLY the articles provided below. Cite article titles. "
    "If the articles don't contain the answer, say you have no relevant news. Never invent facts."
)
 
INTEREST_CRITERIA = (
    "Legal and regulatory risk (lawsuits, investigations, fines), mergers and acquisitions, "
    "leadership changes, financial distress, reputational damage, and cross-border or "
    "high-impact events affecting a client."
)
# ==========================================================
 
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
EMBED_MODEL = "text-embedding-3-small"
 
 
# ---------- build the vector DB once per server ----------
@st.cache_resource(show_spinner="Building news database...")
def build_vector_db(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path).fillna("")
    col = chromadb.Client().get_or_create_collection(
        "HW7NewsCollection", metadata={"hnsw:space": "cosine"}
    )
 
    texts = [f"{r[TITLE_COL]}\n{r[BODY_COL]}"[:8000] for _, r in df.iterrows()]
    metas = [{"title": str(r[TITLE_COL]), "date": str(r[DATE_COL])} for _, r in df.iterrows()]
    ids = [str(i) for i in range(len(df))]
 
    for s in range(0, len(texts), 50):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[s:s + 50])
        col.add(
            ids=ids[s:s + 50],
            documents=texts[s:s + 50],
            embeddings=[e.embedding for e in resp.data],
            metadatas=metas[s:s + 50],
        )
    return col
 
 
col = build_vector_db()
 
 
# ---------- retrieval + LLM helpers ----------
def search_news(query, k=5):
    emb = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    r = col.query(query_embeddings=[emb], n_results=min(k, col.count()))
    return list(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))
 
 
def format_articles(items):
    return "\n\n".join(f"[{m['title']} | {m['date']}]\n{d[:1500]}" for d, m, _ in items)
 
 
def ask_llm(model, system_text, history, user_msg):
    messages = [{"role": "system", "content": system_text}] + history + [
        {"role": "user", "content": user_msg}
    ]
    return client.chat.completions.create(model=model, messages=messages).choices[0].message.content
 
 
# ---------- UI ----------
st.title("News Information Bot")
model = st.sidebar.selectbox("Model", ["gpt-5-mini", "gpt-5"])  # cheap vs. expensive
st.sidebar.write(f"Articles loaded: {col.count()}")
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])
 
if prompt := st.chat_input("Ask about the news (e.g., 'Find the most interesting news')"):
    st.chat_message("user").write(prompt)
 
    if "interesting" in prompt.lower():
        items = search_news(INTEREST_CRITERIA, k=15)
        instruction = (
            "Rank these articles from most to least interesting for the law firm using these "
            f"criteria: {INTEREST_CRITERIA}\nReturn a numbered list. For each article give the "
            "title and 1-2 sentences explaining WHY it ranks there."
        )
    else:
        items = [x for x in search_news(prompt, k=5) if x[2] <= MAX_DISTANCE]
        instruction = "Answer the user's question using only these articles."
 
    if not items:
        reply = "I couldn't find any relevant articles in my news set for that."
    else:
        system_text = SYSTEM + "\n\n" + instruction + "\n\nARTICLES:\n" + format_articles(items)
        with st.spinner("Thinking..."):
            reply = ask_llm(model, system_text, st.session_state.messages[-6:], prompt)
 
    st.chat_message("assistant").write(reply)
    st.session_state.messages += [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reply},
    ]