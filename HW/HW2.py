import requests
from bs4 import BeautifulSoup
import streamlit as st
from openai import OpenAI

def read_url_content(url):
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None


MODEL_OPTIONS = {"basic": "gpt-4o-mini", "advanced": "gpt-4o"}

SUMMARY_TYPES = {
    "Summarize in 100 words": "Summarize the text in about 100 words.",
    "Summarize in 2 connecting paragraphs": "Summarize the text in exactly two connecting paragraphs.",
    "Summarize in 5 bullet points": "Summarize the text as 5 concise bullet points.",
}

LANGUAGES = ["English", "French", "Spanish", "German", "Japanese"]


def build_prompt(text, summary_instruction, language):
    return (
        f"{summary_instruction}\n\n"
        f"Write the ENTIRE summary in {language}, regardless of the "
        f"language of the source text.\n\n"
        f"Here is the text to summarize:\n\n{text}"
    )


def call_openai(api_key, model, prompt):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


st.title("HW2 - URL Summarizer")
st.caption("Enter a web page URL below, then configure the summary in the sidebar.")

url = st.text_input("Enter a URL to summarize", placeholder="https://example.com/article")

with st.sidebar:
    st.header("Summary Options")
    summary_choice = st.radio("Type of summary", list(SUMMARY_TYPES.keys()))
    language = st.selectbox("Output language", LANGUAGES, index=0)
    st.divider()
    st.header("Model Options")
    use_advanced = st.checkbox("Use advanced model", value=False)
    tier = "advanced" if use_advanced else "basic"
    selected_model = MODEL_OPTIONS[tier]
    st.caption(f"Model: `{selected_model}`")
    generate = st.button("Generate Summary", type="primary", use_container_width=True)

openai_key = st.secrets.get("OPENAI_API_KEY", None)

if generate:
    if not url:
        st.warning("Please enter a URL first.", icon="⚠️")
    elif not openai_key:
        st.error("No OpenAI API key found. Add OPENAI_API_KEY in Settings > Secrets.", icon="🗝️")
    else:
        with st.spinner(f"Reading page and summarizing ({selected_model})..."):
            page_text = read_url_content(url)
            if page_text:
                prompt = build_prompt(page_text, SUMMARY_TYPES[summary_choice], language)
                summary = call_openai(openai_key, selected_model, prompt)
                st.subheader("Summary")
                st.write(summary)
                with st.expander("Show raw extracted page text"):
                    st.write(page_text[:5000] + ("..." if len(page_text) > 5000 else ""))
