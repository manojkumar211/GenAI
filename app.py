# Import the neccessary Libraries

import os
from dotenv import load_dotenv
from langchain_community.llms import ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# LangSmith Tracking

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

# Prompt Template

prompt=ChatPromptTemplate.from_message(
    [
        ("system","you are a helpful assistant. please respond to the question asked"),
        ("user","Question:{question}")
    ]
)

# Streamlit Framework

st.title("Langchain Project with Ollama")
input_text=st.text_input("what question you have in mind")

# Ollama LLama2 model

llm=ollama(model="gemma2:2b")
out_parser=StrOutputParser()
chain=prompt | llm | out_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))

if __name__=="__main__":
    import uvicorn
    uvicorn.run(host="0.0.0.0", port=8000)
