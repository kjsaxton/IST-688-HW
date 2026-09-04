import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

# Show title and description.
st.title("MY Document summarizer")
st.write(
    "Upload a PDF document below and get a summary – GPT will answer! "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_kev = st.secrets.OPENAI_API_KEY

client = OpenAI(api_key=openai_api_kev)

# Sidebar Options
summary_type = st.sidebar.selectbox(
    "Choose summary type:",
    (
        "Summarize in 100 words",
        "Summarize in 2 connecting paragraphs",
        "Summarize in 5 bullet points",
    )
)

use_advanced = st.sidebar.checkbox("Use advanced model")

if use_advanced:
    model = "gpt-5-mini"
else:
    model = "gpt-5-nano"

# Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
    "Upload a document (.pdf)", type=("pdf")
)

if uploaded_file:

    # Process the uploaded file.
    reader = PdfReader(uploaded_file)
    document = ""
    for page in reader.pages:
        document += page.extract_text() or ""

    if summary_type == "Summarize in 100 words":
        instruction = "Summarize the document in exactly 100 words."
    elif summary_type == "Summarize in 2 connecting paragraphs":
        instruction = "Summarize the document in 2 connecting paragraphs."
    else:
        instruction = "Summarize the document in 5 concise bullet points."

    messages = [
        {
            "role": "user",
            "content": f"Here's a document: {document} \n\n---\n\n {instruction}",
        }
    ]

    # Generate an answer using the OpenAI API.
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    # Stream the response to the app using `st.write_stream`.
    st.write_stream(stream)
