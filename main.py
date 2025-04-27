import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
import os
from dotenv import load_dotenv

load_dotenv()
## setting up the StreamLit App
st.set_page_config(page_title="Youtube Video or Website summarizer powered by Langchain")
st.title("Summarize Youtube Videos or Website Pages")
st.subheader('Summarize URL')


input_url = st.text_input("URL", label_visibility="collapsed")

#Groq gemma model
groq_api=os.getenv("GROQ_API")
llm=ChatGroq(model="mistral-saba-24b", groq_api_key=groq_api)


prompt_template="""
Just Give me a summary of the given below text in 500 words
Content:{text}
Go through above given content and only generate me a detailed summary of 500 words, first understand the topic and then generate the summary, don't ask any further questions.
This is not a chat, just give me the summary as I asked.
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Get the Summary from the URL"):
    if not input_url.strip():
        st.error("Please enter the URL to get started")
    elif not validators.url(input_url):
        st.error("Please enter a valid URL, can be a Youtube Video or a Web URL")
    else:
        try:
            with st.spinner("Loading...."):
                # Check if YouTube or Webpage
                if "youtube.com" in input_url or "youtu.be" in input_url:
                    # YouTube: fetch transcript
                    try:
                        if "v=" in input_url:
                            video_id = input_url.split("v=")[1].split("&")[0]
                        else:
                            video_id = input_url.split("/")[-1]
                        
                        transcript = YouTubeTranscriptApi.get_transcript(video_id)
                        text = " ".join([t['text'] for t in transcript])
                    except Exception as e:
                        st.error("Failed to fetch YouTube transcript. Maybe captions are disabled.")
                        st.exception(e)
                        text = None
                else:
                    # Webpage: fetch content
                    loader = UnstructuredURLLoader(
                        urls=[input_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                                          "Chrome/116.0.0.0 Safari/537.36"
                        }
                    )
                    docs = loader.load()
                    text = docs[0].page_content if docs else None

                # If text was fetched
                if text:
                    doc = Document(page_content=text)  # <-- wrap text inside a Document
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run([doc])  # <-- pass a list of Document(s)
                    st.success(output_summary)
                else:
                    st.error("Failed to load content from the provided URL.")
        except Exception as e:
            st.exception(e)


