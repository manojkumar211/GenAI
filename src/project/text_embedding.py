from documentloader import text_data
from text_chunks import text_chunk
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings

text_embedding=OllamaEmbeddings(model="gemma2:2b")
text_vector=text_embedding.embed_documents(text_chunk)

print(text_vector[0])