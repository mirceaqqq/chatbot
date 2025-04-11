import os
import streamlit as st
st.set_page_config(page_title="Chatbot", page_icon="🤖")

from dotenv import load_dotenv
from groq import Groq

# Importuri actualizate pentru a evita avertismentele de depreciere
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq


# Load API key
load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize Groq client for LangChain
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model = "llama3-70b-8192" 
)

# Load PDF documents
@st.cache_resource
def load_documents():
    all_docs = []
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("data", file))
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

# Vectorstore + Retriever
@st.cache_resource
def get_retriever():
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    texts = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Înlocuim Chroma cu FAISS
    db = FAISS.from_documents(texts, embeddings)
    
    # Salvăm indexul pentru utilizare viitoare (opțional)
    db.save_local("./faiss_index")
    
    return db.as_retriever()

retriever = get_retriever()

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Streamlit UI
st.title(" ATM Chatbot")

prompt = st.text_input("Intrebare:")

if prompt:
    with st.spinner("Thinking..."):
        result = qa_chain(prompt)
        st.markdown("Răspuns:")
        st.write(result["result"])

        with st.expander("Surse:"):
            for doc in result["source_documents"]:
                st.markdown(f"**Pagina:** {doc.metadata.get('page', '?')}")
                st.markdown(doc.page_content)
                st.markdown("---")
 

