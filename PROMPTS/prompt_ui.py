from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
#dynamic prompts with langchain
load_dotenv()

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7
)

st.header("Research Assistant")

paper_input = st.selectbox(
    "Select a research paper",
    [
        "No Silver Bullet",
        "The Mythical Man-Month",
        "The Cathedral and the Bazaar"
    ]
)

style_input = st.selectbox(
    "Select a writing style",
    ["Formal", "Informal", "Technical"]
)

length_input = st.slider(
    "Select the length of the summary",
    50,
    500,
    150
)

template = """
You are a research assistant.

Your task is to summarize the research paper titled "{paper_title}" in a {writing_style} style.

The summary should be approximately {summary_length} words long.

Please provide a concise and informative summary that captures the key points of the paper.
"""

prompt = PromptTemplate.from_template(template)

if st.button("Generate Summary"):

    final_prompt = prompt.invoke({
        "paper_title": paper_input,
        "writing_style": style_input,
        "summary_length": length_input
    })

    response = model.invoke(final_prompt)

    st.subheader("Summary:")
    st.write(response.content)