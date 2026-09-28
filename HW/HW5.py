import streamlit as st
from openai import OpenAI
import sys
import json
import chromadb
from pathlib import Path
from bs4 import BeautifulSoup

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW')
collection = chroma_client.get_or_create_collection('HW4Collection')

if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

client = st.session_state.openai_client


def add_to_collection(collection, text, file_name):
    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )
    embedding = response.data[0].embedding
    collection.add(
        documents=[text],
        ids=[file_name],
        embeddings=[embedding]
    )


def extract_text_from_html(html_path):
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text(separator="\n ", strip=True)


def chunk_by_paragraph(text, num_chunks=2):
    paragraphs = [p for p in text.split("\n") if p.strip()]

    if len(paragraphs) < 2:
        words = text.split()
        if not words:
            return [text, ""]
        mid = len(words) // 2
        return [" ".join(words[:mid]), " ".join(words[mid:])]

    total_len = sum(len(p) for p in paragraphs)
    target = total_len / 2
    running = 0
    split_idx = 1
    for i, p in enumerate(paragraphs):
        running += len(p)
        if running >= target:
            split_idx = i + 1
            break

    chunk1 = " ".join(paragraphs[:split_idx])
    chunk2 = " ".join(paragraphs[split_idx:])
    return [chunk1, chunk2]


def load_html_to_collection(folder_path, collection):
    loaded = []
    for html_path in Path(folder_path).glob("*.html"):
        try:
            text = extract_text_from_html(html_path)
            chunks = chunk_by_paragraph(text)
            for i, chunk_text in enumerate(chunks):
                if chunk_text.strip():
                    chunk_id = f"{html_path.name}_chunk{i+1}"
                    add_to_collection(collection, chunk_text, chunk_id)
            loaded.append(html_path.name)
        except Exception as e:
            st.sidebar.write(f"{html_path.name}: {e}")
            continue
    return loaded


if collection.count() == 0:
    loaded = load_html_to_collection('HW/su_orgs/', collection)

if 'HW5_VectorDB' not in st.session_state:
    st.session_state.HW5_VectorDB = collection

model = "gpt-5-mini"


def relevant_club_info(query, n_results=3):
    response = client.embeddings.create(
        input=query,
        model='text-embedding-3-small'
    )
    query_embedding = response.data[0].embedding

    results = st.session_state.HW5_VectorDB.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    docs = results['documents'][0]
    ids = results['ids'][0]

    if not docs:
        return "No relevant student organization info was found.", []

    info = "\n\n".join(
        f"[Source: {doc_id}]\n{doc_text}" for doc_id, doc_text in zip(ids, docs)
    )
    return info, ids


tools = [
    {
        "type": "function",
        "function": {
            "name": "relevant_club_info",
            "description": (
                "Search the student organizations vector database for information "
                "relevant to a query about clubs or student organizations at the "
                "university (e.g. meeting times, purpose, how to join). Call this "
                "whenever the user asks about a specific club/organization or "
                "student activities, rewriting vague follow-up questions into a "
                "clear, self-contained search query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A focused, self-contained search query, e.g. "
                            "'robotics club meeting times' rather than 'what about that one?'."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    }
]

buffer_type = st.sidebar.radio(
    "Conversation memory type:",
    (
        "Last 5 interactions",
        "Token limit",
    )
)

max_tokens = st.sidebar.number_input(
    "Max tokens to send (token limit mode)",
    min_value=200,
    max_value=8000,
    value=1000,
    step=100,
)

st.title("HW5 - Student Org Chatbot with Tool-Calling RAG")
st.write(
    st.write(
    "Ask me about Syracuse student organizations. I will decide when a database "
    "lookup is necessary and what to search for, using a function called "
    "relevant_club_info."
)


system_prompt = {
 system_prompt = {
    "role": "system",
    "content": (
        "You are a friendly assistant that helps students learn about Syracuse "
        "student organizations. You have access to a tool called "
        "relevant_club_info(query) that searches a vector database of student "
        "organization pages. Call it whenever the user asks about a club, "
        "activity, or organization. Rewrite vague follow-ups (like 'their "
        "contact info' or 'meeting times') into a clear, standalone search "
        "query using the conversation so far. If you use information from the "
        "tool, say so clearly (e.g. 'Based on the student organization info I "
        "found...'). If the tool returns nothing relevant, say so and answer "
        "from general knowledge instead. Do not call the tool for greetings, "
        "thanks, or questions unrelated to student organizations."
    ),
}
}

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about any student organization on campus."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def estimate_tokens(text):
    return max(1, len(text) // 4)


def build_buffer(messages, buffer_type, max_tokens, system_prompt):
    if buffer_type == "Last 5 interactions":
        trimmed = messages[-10:] if len(messages) > 10 else messages
        return [system_prompt] + trimmed
    else:
        running_total = estimate_tokens(system_prompt["content"])
        kept_reversed = []
        for msg in reversed(messages):
            msg_tokens = estimate_tokens(msg["content"])
            if running_total + msg_tokens > max_tokens:
                break
            kept_reversed.append(msg)
            running_total += msg_tokens
        return [system_prompt] + list(reversed(kept_reversed))


if prompt := st.chat_input("Ask about a student organization..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages_to_send = build_buffer(
        st.session_state.messages, buffer_type, max_tokens, system_prompt
    )

    first_response = client.chat.completions.create(
        model=model,
        messages=messages_to_send,
        tools=tools,
        tool_choice="auto",
    )
    assistant_msg = first_response.choices[0].message

    source_ids = []

    if assistant_msg.tool_calls:
        messages_to_send.append({
            "role": "assistant",
            "content": assistant_msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in assistant_msg.tool_calls
            ],
        })

        for tc in assistant_msg.tool_calls:
            if tc.function.name == "relevant_club_info":
                args = json.loads(tc.function.arguments)
                query = args.get("query", prompt)
                info, ids = relevant_club_info(query)
                source_ids.extend(ids)
            else:
                info = "Unknown tool requested."

            messages_to_send.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": info,
            })

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=model,
            messages=messages_to_send,
            stream=True,
        )
        response = st.write_stream(stream)
        if source_ids:
            st.caption(f"Sources consulted: {', '.join(source_ids)}")

    st.session_state.messages.append({"role": "assistant", "content": response})
