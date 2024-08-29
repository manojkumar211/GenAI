import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")



## Document Loaders:-

pdf_documents=PyPDFLoader("C:/DataScience/attentionneed.pdf")
pdf_load=pdf_documents.load()
print(pdf_load[0].page_content)


## Text Splitter (Chunks):-

text_split=RecursiveCharacterTextSplitter(separators="\n\n",chunk_size=1000,chunk_overlap=100)
text_chunk=text_split.split_documents(pdf_load)
print(text_chunk[0].page_content)
print(text_chunk[1].page_content)


## Text Embedding (Vector):-

text_embedding=OllamaEmbeddings(model="gemma2:2b")


## VEctor Store Data Base:-

text_vectoreDB=FAISS.from_documents(text_chunk,text_embedding)
query="tell about agent"
text_query=text_vectoreDB.similarity_search(query)
print(text_query[0].page_content)