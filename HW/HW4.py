import streamlit as st
from openai import OpenAI
import sys
import chromadb
from pathlib import Path
from pypdf import PdfReader
 
# A fix for working with ChromaDB on streamlit community cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
 
# create ChromaDB client
chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_Lab')
collection = chroma_client.get_or_create_collection('Lab4Collection')
 
## USING CHROMA DB WITH OPENAI EMBEDDINGS ####
 
# Create OpenAI client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)
 
# A function that will add documents to the collection
# collection = ChromaDB collection, already established
# text = extracted text from PDF files
# Embeddings inserted into the collection from OpenAI
def add_to_collection(collection, text, file_name):
 
    # Create an embedding
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )
 
    # Get the Embedding
    embedding= response.data[0].embedding
 
    # Add embedding and document to ChromaDB
    collection.add(
        documents=[text],
        ids=[file_name],
        embeddings=[embedding]
    )
 
#### EXTRACT TEXT FROM PDF ####
# This function extracts text from each syllabus
# to pass to add_to_collection
def extract_text_from_pdf(pdf_path):
  reader = PdfReader(pdf_path)
  text = ""
  for page in reader.pages:
      page_text = page.extract_text()
      if page_text:
          text += page_text + "\n"
  return text
 
#### POPULATE COLLECTION WITH PDFs 
# This function uses extract_text_from_pdf
# and add_to_collection to put syllabi in ChromaDB collection
def load_pdfs_to_collection(folder_path, collection):
    loaded = []
    for pdf_path in Path(folder_path).glob("*.pdf"):
        text = extract_text_from_pdf(pdf_path)
        add_to_collection(collection, text, pdf_path.name)
        loaded.append(pdf_path.name)
    return loaded
 
# Check if collection is empty and load PDFs
if collection.count() == 0:
    loaded = load_pdfs_to_collection('./Lab-04-Data/', collection)
 
if 'Lab4_VectorDB' not in st.session_state:
    st.session_state.Lab4_VectorDB = collection
 
# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_kev = st.secrets.OPENAI_API_KEY
 
client = OpenAI(api_key=openai_api_kev)
 
#### QUERYING A COLLECTION -- ONLY USED FOR TESTING ####
# Uncomment this section to validate Part A (that the vectorDB returns
# sensible results), then comment it back out for Part B.
 
# topic = st.sidebar.text_input('Topic', placeholder='Type your topic (e.g., GenAI)...')
#
# if topic:
#     response = client.embeddings.create(
#         input=topic,
#         model='text-embedding-3-small'
#     )
#
#     # Get the embedding
#     query_embedding = response.data[0].embedding
#
#     # Get the text related to this question (this prompt)
#     results = st.session_state.Lab4_VectorDB.query(
#         query_embeddings=[query_embedding],
#         n_results=3  # The number of closest documents to return
#     )
#
#     # Display the results
#     st.subheader(f'Results for: {topic}')
#
#     for i in range(len(results['documents'][0])):
#         doc = results['documents'][0][i]
#         doc_id = results['ids'][0][i]
#
#         st.write(f'**{i+1}. {doc_id}**')
# else:
#     st.info('Enter a topic in the sidebar to search the collection')
 
model = "gpt-5-mini"
 
def get_info_from_vectorDB(myVectorDB, prompt, n_results=3):
    response = client.embeddings.create(
        input=prompt,
        model='text-embedding-3-small'
    )
    query_embedding = response.data[0].embedding
 
    results = myVectorDB.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
 
    docs = results['documents'][0]
    ids = results['ids'][0]
 
    extra_info = "\n\n".join(
        f"[Source: {doc_id}]\n{doc_text}" for doc_id, doc_text in zip(ids, docs)
    )
    return extra_info, ids
 
# Sidebar Options
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
        "Keep following this pattern for every new question the user asks.\n\n"
        "You will sometimes be given course syllabus excerpts as context "
        "below the conversation. If you use that context to answer, say "
        "so clearly (e.g., 'Based on the course syllabus...'). If the "
        "context doesn't have relevant info, say so and answer from your "
        "general knowledge instead."
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
 
 
def build_buffer(messages, buffer_type, max_tokens, system_prompt):
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
 
        extra_info, source_ids = get_info_from_vectorDB(
        st.session_state.Lab4_VectorDB, prompt
    )
 
    rag_system_prompt = {
        "role": "system",
        "content": (
            system_prompt["content"]
            + "\n\nRelevant course syllabus excerpts:\n\n"
            + extra_info
        ),
    }
 
    messages_to_send = build_buffer(
        st.session_state.messages, buffer_type, max_tokens, rag_system_prompt
    )
 
    stream = client.chat.completions.create(
        model=model,
        messages=messages_to_send,
        stream=True,
    )
 
    with st.chat_message("assistant"):
        response = st.write_stream(stream)
        st.caption(f"Sources consulted: {', '.join(source_ids)}")
    st.session_state.messages.append({"role": "assistant", "content": response})