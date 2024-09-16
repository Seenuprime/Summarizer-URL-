import streamlit as st 
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader


st.set_page_config(page_title="LangChain: Summarize YouTube Video")
st.title("Langchain: Summarize Youtube video")
st.subheader("Summarize URL")


with st.sidebar:
    api = st.sidebar.text_input('Enter the Groq API key:', type="password")

llm = ChatGroq(model="gemma-7b-It", api_key=api)
prompt_template = """
Please provide the summary for the given content,
content: {text}
"""
prompt = PromptTemplate(input_variables=['text'], template=prompt_template)


url = st.text_input("Enter the URL: ", label_visibility='collapsed')

if st.button("Summarize"):
    if not api.strip() or not url.strip():
        st.error("Please Provide the Info")
    elif not validators.url(url):
        st.error("Please enter a valid error")
    else:
        try:
            with st.spinner("Waiting...."):
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False)
                
                docs = loader.load()

                chain = load_summarize_chain(llm, chain_type='stuff', prompt=prompt)
                summary = chain.run(docs)

                st.success(summary)

        except Exception as e:
            st.exception(e)

