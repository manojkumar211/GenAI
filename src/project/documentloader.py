from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import arxiv



text_loader=TextLoader("C:/Projects/GenerativeAI/telugu.txt")
text_data=text_loader.load()
print(text_data[0].page_content)
print(len(text_data[0].page_content))