import requests
from bs4 import BeautifulSoup
import streamlit as st
from openai import OpenAI
import anthropic
 
def read_url_content(url):
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None
 
 
MODEL_OPTIONS = {
    "OpenAI": "gpt-6-astra",
    "Claude (Anthropic)": "claude-opus-5",
}
 
openai_key = st.secrets.get("OPENAI_API_KEY", None)
claude_key = st.secrets.get("ANTHROPIC_API_KEY", None)
key_map = {"OpenAI": openai_key, "Claude (Anthropic)": claude_key}
 
 
st.title("HW3 - URL-Aware Chatbot")
st.write(
    """
   This Chatbot can read up to two Urls and answers questions about them. 
   It has a buffer of 6, meaning it will keep the most recent 6 messages in its memory
    """
)


with st.sidebar:
    st.header("URL Context")
    url1 = st.text_input("URL #1 (optional)", key="url1")
    url2 = st.text_input("URL #2 (optional)", key="url2")
 
    st.divider()
    st.header("Model Options")
    llm_choice = st.selectbox("Choose the LLM provider", list(MODEL_OPTIONS.keys()))
    selected_model = MODEL_OPTIONS[llm_choice]
    st.caption(f"Model: `{selected_model}`")
 
active_key = key_map[llm_choice]
 
if "last_urls" not in st.session_state or st.session_state.last_urls != (url1, url2):
    url_texts = []
    if url1:
        content1 = read_url_content(url1)
        if content1:
            url_texts.append(f"--- Content from URL 1 ({url1}) ---\n{content1}")
    if url2:
        content2 = read_url_content(url2)
        if content2:
            url_texts.append(f"--- Content from URL 2 ({url2}) ---\n{content2}")
 
    st.session_state.url_context = "\n\n".join(url_texts) if url_texts else ""
    st.session_state.last_urls = (url1, url2)
 
base_instructions = (
    "You are a friendly, knowledgeable assistant. Answer the user's questions "
    "clearly and concisely. If the user asks about the provided reference "
    "material below, use it to inform your answer. If no relevant URL content "
    "is available, answer from your own knowledge and say so."
)
 
if st.session_state.url_context:
    system_content = (
        f"{base_instructions}\n\n"
        f"Here is reference material from the URL(s) the user provided:\n\n"
        f"{st.session_state.url_context}"
    )
else:
    system_content = base_instructions
 
system_prompt = {"role": "system", "content": system_content}
 
BUFFER_SIZE = 6
 
 
def build_buffer(messages):
    trimmed = messages[-BUFFER_SIZE:] if len(messages) > BUFFER_SIZE else messages
    return [system_prompt] + trimmed
 
 
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything — including about the URLs you've added in the sidebar."}
    ]
 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
 
def stream_openai(messages_to_send):
    client = OpenAI(api_key=active_key)
    stream = client.chat.completions.create(
        model=selected_model,
        messages=messages_to_send,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content 
        if delta:
            yield delta
 
 
def stream_claude(messages_to_send):
    client = anthropic.Anthropic(api_key=active_key)
    # Anthropic's API takes the system prompt separately from the messages list
    system_text = messages_to_send[0]["content"]
    convo = messages_to_send[1:]
    with client.messages.stream(
        model=selected_model,
        max_tokens=1024,
        system=system_text,
        messages=convo,
    ) as stream:
        for text in stream.text_stream:
            yield text
 

if prompt := st.chat_input("What is up?"):
    if not active_key:
        st.error(f"No API key found for **{llm_choice}**. Add it in Settings > Secrets.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
 
        messages_to_send = build_buffer(st.session_state.messages)
 
        with st.chat_message("assistant"):
            if llm_choice == "OpenAI":
                response = st.write_stream(stream_openai(messages_to_send))
            else:
                response = st.write_stream(stream_claude(messages_to_send))
 
        st.session_state.messages.append({"role": "assistant", "content": response})
 