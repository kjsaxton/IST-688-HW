import streamlit as st, chromadb, pandas as pd
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def build_vector_db(csv_path="news.csv"):
    df = pd.read_csv(csv_path)
    chroma = chromadb.Client()                     
    col = chroma.get_or_create_collection("HW7NewsCollection")

    texts = [f"{r['TITLE_COL']}\n{r['BODY_COL']}"[:8000] for _, r in df.iterrows()]   # rename columns
    metas = [{"title": str(r["TITLE_COL"]), "date": str(r["DATE_COL"])} for _, r in df.iterrows()]
    ids = [str(i) for i in range(len(df))]

    
    for start in range(0, len(texts), 100):
        batch = texts[start:start+100]
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        col.add(
            ids=ids[start:start+100],
            documents=batch,
            embeddings=[e.embedding for e in resp.data],
            metadatas=metas[start:start+100],
        )
    return col

if "HW7_VectorDB" not in st.session_state:
    with st.spinner("Building news database..."):
        st.session_state.HW7_VectorDB = build_vector_db()
col = st.session_state.HW7_VectorDB
