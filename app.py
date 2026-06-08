import streamlit as st
import os

from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HUGGING_FACE")
groq_api_key = os.getenv("GROQ_API_KEY")

## Model defined
llm=ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")

## Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant for answering questions about the documents provided.
    Use only the following retrieved documents to answer the question. If you don't know the answer, say you don't know.
    <context>
    {context}
    <context>
    Question: {input}
    Answer:
    """
)

## Create vecotors
def create_vector_embeddings(): 
    ## We will be using session state to store the vector embeddings and avoid creating them every time the user asks a question
    ## This will speed up the response time for the user
    ## This function will be called when the user uploads a new document and we need to create new vector embeddings for the new document
    ## This can be used in the other code snippets where we need to create vector embeddings for the documents

    if 'vector_embeddings' not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        st.session_state.loader = PyPDFDirectoryLoader("files")

        st.session_state.docs = st.session_state.loader.load() ## Document loaders

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs) ##we are only using the first 50 documents for this example, you can change this to use all the documents

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    ## No need to return anything as we are using session state to store the vector embeddings

user_prompt = st.text_input("Ask me Something Realted to the documents you uploaded : TRANSFORMERS")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.success("Vector embeddings created successfully!")

import time

if user_prompt:
    if 'vectors' not in st.session_state:
        st.warning("Please create vector embeddings first by clicking the 'Document Embedding' button.")
    else:
        document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        ## by using create_stuff_documents_chain we are creating a chain that will provide the retrieved_docs in the prompt template
        retriever = st.session_state.vectors.as_retriever()
        ##It works as an interface to retrieve relevant documents from the vector store based on the user's query. It uses the vector embeddings to find and return the most relevant documents that can help answer the user's question.
        retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

        start=time.process_time()
        response = retrieval_chain.invoke({
            "input": user_prompt
        })
        print(f"Time taken to get the response: {time.process_time() - start} seconds")

        st.write("Answer: ", response["answer"])

        ##With a stremalit expander
        with st.expander("Documents similarity search"):
            for i,doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('-----------------------')