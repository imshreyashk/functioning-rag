import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

load_dotenv()

# 1. Fixed LLM Connection
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API"), # Fixed name
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1
)

# 2. Embedding Model (This part was perfect)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Test it
query_result = embedding.embed_query('This is a test document for my Rag System')

print(f"Embedding length: {len(query_result)}")
print("Setup Verified!")
