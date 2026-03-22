from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
     model_name='sentence-transformers/all-MiniLM-L6-v2',
     task="text-generation",
     max_new_tokens=512,
     temperature=0.1
)

vectore = embeddings.embed_query('What is the refund policy')
print(f'Dimensions: {len(vectore)}')

embedding_large = HuggingFaceEmbeddings(
    model_name ='BAAI/bge-large-en-v1.5'
)