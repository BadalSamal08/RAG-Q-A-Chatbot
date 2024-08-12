import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")


prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

def create_vector_embedding(uploaded_files):
    """Function to create vector embeddings from uploaded PDFs."""
   
    documents = []
    for uploaded_file in uploaded_files:
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        
        pdf_loader = PyPDFLoader(tmp_file_path)
        documents.extend(pdf_loader.load())

        
        os.remove(tmp_file_path)

    
    st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


st.title("PDF Query Application")
uploaded_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
import time
if st.button("Create Document Embedding"):
    if uploaded_files:
        create_vector_embedding(uploaded_files)
        st.write("Vector Database is ready")
    else:
        st.warning("Please upload at least one PDF file.")

user_prompt = st.text_input("Enter your query")
if user_prompt and 'vectors' in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    print(f"Response time: {time.process_time() - start}")

    st.write(response['answer'])
else:
    st.warning("Please create the document embedding first before asking a query.")
