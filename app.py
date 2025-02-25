import os
import subprocess

# Define the expected path of the browser executable.
# (You might need to adjust this path if it differs in your environment)
browser_path = "/home/appuser/.cache/ms-playwright/chromium-1155/chrome-linux/chrome"

# Check if the executable exists; if not, install browsers.
if not os.path.exists(browser_path):
    st.write("Installing Playwright browsers...")
    subprocess.run(["playwright", "install"], check=True)

import streamlit as st
import asyncio
import nest_asyncio
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from streamlit_chat import message as st_message

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

load_dotenv()
nest_asyncio.apply()


# ---------------------------
# Initialize Session State Variables
# ---------------------------

if "url_submitted" not in st.session_state:
    st.session_state.url_submitted = False

if "extraction_done" not in st.session_state:
    st.session_state.extraction_done = False
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "embedding_done" not in st.session_state:
    st.session_state.embedding_done = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = ""

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(layout="wide", page_title="WebSage")
st.title("Project WebSage")

page = st.sidebar.selectbox("Navigation",["Home","AI Engine","Contact"])

if page == "Home":
    st.markdown("""
    ## Welcome to WebSage
    **WebSage** is a cutting-edge RAG Chatbot application that allows you to extract content from any URL, generate detailed summaries, and interact with the content using advanced language models.  
    With options to choose between **GROQ API** (GROQ) and **Local LLMs** (Ollama) engines for both summarization and conversation, WebSage gives you the flexibility to explore and deploy the best AI solutions for your needs.
    
    **Features:**
    - **Website Extraction:** Crawl and extract web page content.
    - **Summarization:** Generate detailed summaries of the extracted content.
    - **Embeddings & Retrieval:** Create embeddings with FAISS for intelligent document retrieval.
    - **Chatbot Interface:** Interact with your content via a conversational agent.
    
    Get started by selecting **AI Engine** from the sidebar.
    """)

elif page == "AI Engine":
    # ---------------------------
    # URL Input Form
    # ---------------------------
    with st.form("url_form"):
        url_input=st.text_input("Enter a URL to Crawl:")
        submit_url=st.form_submit_button("Submit URL")
        if submit_url and url_input:
            st.session_state.url_submitted = True

            st.session_state.extraction_done = False
            st.session_state.embedding_done = False
            st.session_state.chat_history = []
            st.session_state.summary = ""
     # ---------------------------
    # If URL has been submitted, divide layout into three columns
    # ---------------------------
    if st.session_state.url_submitted:
        col1, col2, col3 = st.columns(3)
         # ---------------------------
        # Column 1: Website Extraction & Summarization using crawl4ai
        # ---------------------------
        with col1:
            st.header("1. Website Extraction")
            if not st.session_state.extraction_done:
                with st.spinner("Extracting Website..."):
                    async def simple_crawl(url):
                        crawler_run_config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
                        async with AsyncWebCrawler() as crawler:
                            result=await crawler.arun(url=url, config=crawler_run_config)
                            return result.markdown
                    
                
                    extracted=asyncio.run(simple_crawl(url_input))
                    st.session_state.extracted_text=extracted
                    st.session_state.extraction_done = True
            
                st.success("Extraction Complete!")

            preview = "\n".join(
                    [line for line in st.session_state.extracted_text.splitlines() if line.strip()][:5]
                )
            
            st.text_area("Extracted Text Preview", value=preview, height=300)

            st.download_button(
                label="Download Extracted Text",
                data=st.session_state.extracted_text,
                file_name="extracted_text.txt",
                mime="text/plain",
            )

            st.markdown("----")
            st.subheader("Summarize Web Page")
            if st.button("Summarize Web Page", key="summarize_button"):
                with st.spinner('Summarizing Web Page...'):
                    summary_prompt_template = """
                    You are an AI assistant that is tasked with summarizing a web page.
                    Your summary should be detailed and cover all key points mentioned in the web page.
                    Below is the extracted content of the web page:
                    {content}

                    Please provide a comprehensive and detailed summary in Markdown format.
                    """

                    summary_prompt = PromptTemplate(template=summary_prompt_template, input_variables=["content"])
                    prompt_text=summary_prompt.format(content=st.session_state.extracted_text)
                    summarizer=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.3,max_tokens=1500)
                    summary_response = summarizer.invoke(prompt_text).content
                    st.session_state.summary = summary_response
                
                st.success("Summarization Complete!")
            
            if st.session_state.summary:
                st.subheader("Summary Output")
                st.markdown(st.session_state.summary, unsafe_allow_html=False)
        
        with col2:
            st.subheader("2. Create Embeddings:")
            if st.session_state.extraction_done and not st.session_state.embedding_done:
                if st.button("Create Embeddings..."):
                    with st.spinner("Creating Embeddings..."):
                        with open("output.md", "w", encoding="utf-8") as f:
                            f.write(st.session_state.extracted_text)

                        loader = UnstructuredMarkdownLoader("output.md")
                        data = loader.load()

                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        texts = text_splitter.split_documents(data)

                        embeddings = HuggingFaceEmbeddings()

                        vectorstore = FAISS.from_documents(texts, embeddings)
                        vectorstore.save_local("faiss_index")

                        st.session_state.vectorstore = vectorstore
                        st.session_state.embedding_done = True  # Only mark embeddings as done after the process is complete
      
                st.success("Vector Embeddings Created!")
            
            elif st.session_state.embedding_done:
                st.success("Vector Embeddings Already Created!")

        
        with col3:
            st.header("3. Chatbot Interface")
            if st.session_state.embedding_done:
                llm_choice = st.radio("Select LLM Type", ("Groq API","Local LLMs Ollama"), index=0, key="llm_choice")

                vectorstore=st.session_state.vectorstore
                retriever=vectorstore.as_retriever(search_kwargs={"k":5})
                prompt_template = """
                    You are an AI assistant tasked with answering questions based solely
                    on the provided context. Your goal is to generate a comprehensive answer
                    for the given question using only the information available in the context.

                    context: {context}

                    question: {question}

                    <response> Your answer in Markdown format. </response>
                    """
                
                prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                chain_type_kwargs={"prompt":prompt}
                if llm_choice=="Groq API":
                    llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.3,max_tokens=1500)
                else:
                    llm=ChatOllama(model="gemma:2b", base_url="http://localhost:11434", temperature=0.3)
                
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs=chain_type_kwargs,
                    return_source_documents=True,
                    verbose=True
                )

                user_input=st.text_input("Your Message:", key='chat_input')
                if st.button("Send", key="send_button") and user_input:
                    response=qa(user_input)
                    bot_answer=response["result"]
                    st.session_state.chat_history.append({"user": user_input, "bot": bot_answer})

                    chat_file_content="\n\n".join(
                        [f"User: {chat['user']}\nBot: {chat['bot']}" for chat in st.session_state.chat_history]
                    )

                    with open("chat_history.md","w",encoding="utf-8") as f:
                        f.write(chat_file_content)
                    
                if st.session_state.chat_history:
                    for chat in st.session_state.chat_history:
                        st_message(chat["user"], is_user=True)
                        st_message(chat["bot"], is_user=False)
            else:
                st.info("Please create embeddings to activate the chat.")
elif page == "Contact":
    st.markdown("""
    ## Contact
    - **GitHub:** [SunkaraboinaPraveenKumar](https://github.com/SunkaraboinaPraveenKumar)  
      _Contributions are welcome – fork the repository and submit a PR!_

    - **LinkedIn:** [SunkaraboinaPraveenKumar](https://www.linkedin.com/in/sunkaraboina-praveen-84594a256/)  
      _Feel free to reach out on LinkedIn for collaborations or questions.
    """)

                



        