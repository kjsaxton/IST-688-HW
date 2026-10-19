try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass
 
import re
import streamlit as st
import chromadb
import pandas as pd
from openai import OpenAI
 
CSV_PATH = "news.csv"          
 
SYSTEM = (
    "You are a news assistant for a global law firm that monitors news about its clients. "
    "Use ONLY the articles provided below. Cite article titles. "
    "If the articles don't contain the answer, say you have no relevant news. Never invent facts."
)
 
INTEREST_CRITERIA = (
    "Legal and regulatory risk (lawsuits, investigations, fines, regulator scrutiny), "
    "mergers and acquisitions, leadership changes, financial distress, reputational damage, "
    "and cross-border or high-impact events affecting a client."
)
 
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
EMBED_MODEL = "text-embedding-3-small"
 
 
@st.cache_resource(show_spinner="Building news database...")
def build_vector_db(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path).fillna("")
    df["company_name"] = df["company_name"].str.strip()
    df = df.drop_duplicates(subset=["Document"]).reset_index(drop=True)   # ~196 duplicate rows in the CSV
 
    col = chromadb.Client().get_or_create_collection(
        "HW7NewsCollection", metadata={"hnsw:space": "cosine"}
    )
 
    texts, metas = [], []
    for _, r in df.iterrows():
        doc = str(r["Document"])
        title = doc.split(" Description:")[0].strip()          
        texts.append(f"{r['company_name']}: {doc}"[:8000])     
        metas.append({
            "company": r["company_name"],
            "title": title,
            "date": str(r["Date"])[:10],
            "url": str(r["URL"]),
        })
    ids = [str(i) for i in range(len(df))]
 
    for s in range(0, len(texts), 100):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[s:s + 100])
        col.add(
            ids=ids[s:s + 100],
            documents=texts[s:s + 100],
            embeddings=[e.embedding for e in resp.data],
            metadatas=metas[s:s + 100],
        )
    return col
 
 
col = build_vector_db()
 
 
@st.cache_data
def company_list():
    return sorted({m["company"] for m in col.get()["metadatas"]})
 
 

def find_companies(query):
    """Return client companies mentioned by name in the query (whole-word match)."""
    q = query.lower()
    return [c for c in company_list() if re.search(r"\b" + re.escape(c.lower()) + r"\b", q)]
 
 
def get_by_company(companies):
    r = col.get(where={"company": {"$in": companies}})
    return [(d, m, 0.0) for d, m in zip(r["documents"], r["metadatas"])]
 
 
def search_news(query, k=5):
    emb = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    r = col.query(query_embeddings=[emb], n_results=min(k, col.count()))
    return list(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))
 
 
def format_articles(items):
    return "\n\n".join(
        f"[{m['company']} | {m['title']} | {m['date']} | {m['url']}]\n{d[:1500]}"
        for d, m, _ in items
    )
 
 
def ask_llm(model, system_text, history, user_msg):
    messages = [{"role": "system", "content": system_text}] + history + [
        {"role": "user", "content": user_msg}
    ]
    return client.chat.completions.create(model=model, messages=messages).choices[0].message.content
 
 

st.title("News Information Bot")
model = st.sidebar.selectbox("Model", ["gpt-5-mini", "gpt-5"])  
st.sidebar.write(f"Articles loaded: {col.count()}")
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])
 
if prompt := st.chat_input("Ask about the news (e.g., 'Find the most interesting news')"):
    st.chat_message("user").write(prompt)
 
    mentioned = find_companies(prompt)
    wants_interesting = "interesting" in prompt.lower()
 
    if wants_interesting:
        # candidate pool: a named company's articles, or the best matches to the criteria
        items = get_by_company(mentioned) if mentioned else search_news(INTEREST_CRITERIA, k=POOL_SIZE)
        instruction = (
            "Rank these articles from most to least interesting for the law firm using these "
            f"criteria: {INTEREST_CRITERIA}\nReturn a numbered list (at most 10). For each article give "
            "the company, title, link, and 1-2 sentences explaining WHY it ranks there."
        )
    else:
        # topic query: exact company match if a client is named, otherwise semantic search
        if mentioned:
            items = get_by_company(mentioned)
        else:
            items = [x for x in search_news(prompt, k=5) if x[2] <= MAX_DISTANCE]
        instruction = "Answer the user's question using only these articles. Include links."
 
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
 
