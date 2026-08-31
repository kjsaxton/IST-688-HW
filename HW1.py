import streamlit as st
from openai import OpenAI
from pypdf import PdfReader


def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


st.title("MY Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

openai_api_key = st.text_input("OpenAI API Key", type="password")

check_key = False

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    client = OpenAI(api_key=openai_api_key)

    try:
        client.models.list()
        check_key = True
    except Exception:
        st.error("Could not validate the API key")

    if check_key:
        uploaded_file = st.file_uploader(
            "Upload a document (.pdf or .txt)", type=("pdf", "txt")
        )

        question = st.text_area(
            "Now ask a question about the document!",
            placeholder="Can you give me a short summary?",
            disabled=not uploaded_file,
        )

        if uploaded_file and question:

            file_extension = uploaded_file.name.split('.')[-1]
            if file_extension == 'txt':
                document = uploaded_file.read().decode()
            elif file_extension == 'pdf':
                document = read_pdf(uploaded_file)
            else:
                st.error("Unsupported file type.")
                document = None

            if document:
                messages = [
                    {
                        "role": "user",
                        "content": f"Here's a document: {document} \n\n---\n\n {question}",
                    }
                ]

                stream = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=messages,
                    stream=True,
                )

                st.write_stream(stream)