import streamlit as st
from openai import OpenAI
# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_kev = st.secrets.OPENAI_API_KEY
client = OpenAI(api_key=openai_api_kev)
# Sidebar Options
use_advanced = st.sidebar.checkbox("Use advanced model")
if use_advanced:
model = "gpt-5-mini"
else:
model = "gpt-5-nano"
buffer_type = st.sidebar.radio(
"Conversation memory type:",
(
"Last 2 questions",
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
system_prompt = {
"role": "system",
"content": (
"You are a friendly chatbot. Always explain things simply enough "
"that a 10-year-old could understand your answer - use short "
"sentences, everyday words, and simple examples.\n\n"
"Conversation flow you must follow:\n"
"1. When the user asks a question, answer it simply, then ask: "
"'Do you want more info?'\n"
"2. If the user says something like 'yes', give a bit more detail "
"(still simple), and then ask again: 'Do you want more info?'\n"
"3. If the user says something like 'no', stop giving more detail "
"and instead ask: 'What else can I help you with?'\n"
"Keep following this pattern for every new question the user asks."
),
}
if "messages" not in st.session_state:
st.session_state.messages = [
{"role": "assistant", "content": "Hi! What can I help you with today?"}
]
for message in st.session_state.messages:
with st.chat_message(message["role"]):
st.markdown(message["content"])
def estimate_tokens(text):
return max(1, len(text) // 4)
def build_buffer(messages, buffer_type, max_tokens):
if buffer_type == "Last 2 questions":
trimmed = messages[-4:] if len(messages) > 4 else messages
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
if prompt := st.chat_input("What is up?"):
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
st.markdown(prompt)

messages_to_send = build_buffer(st.session_state.messages, buffer_type, max_tokens)
stream = client.chat.completions.create(
model=model,
messages=messages_to_send,
stream=True,
)
with st.chat_message("assistant"):
response = st.write_stream(stream)
st.session_state.messages.append({"role": "assistant", "content": response})
