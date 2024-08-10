from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from documentloader import text_data


text_split=RecursiveCharacterTextSplitter(separators=".",chunk_size=100,chunk_overlap=20)
text_chunk=text_split.split_documents(text_data)
print(text_chunk[0].page_content)
print(text_chunk[1].page_content)
print(text_chunk[2].page_content)