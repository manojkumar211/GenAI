from documentloader import text_data
from text_chunks import text_chunk
from text_embedding import text_embedding
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import chroma

db=FAISS.from_documents(text_chunk,text_embedding)
query="Collaborated with domain experts/ business"
db_query=db.similarity_search(query)
print(db_query[0].page_content)

# As Retriever

retrivers=db.as_retriever()
db_retriver=retrivers.invoke(query)

print(db_retriver[0].page_content)

# Similarity Search

similarity_results=db.similarity_search(query)
print(similarity_results[0])


# Model Saving & Loading

db.save_to_disk('faiss_db')
loaded_db=FAISS.load_from_disk('faiss_db',text_embedding,allow_dangerous_deserialization=True)
print(loaded_db.similarity_search(query)[0].page_content)

