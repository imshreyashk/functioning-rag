from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()
loader = TextLoader('E:\\my-rag-system\\data\\company_policy.txt', encoding='utf-8')
documents = loader.load()
print(f'Loaded {len(documents)} documents')


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,  # Ensure this is 'overlap', not 'overlay'
)


chunk = splitter.split_documents(documents)
print(f'Created {len(chunk)} chunks')


print('Creating embeddings and storing in ChromaDB....')
print('(This call the Hugging Face API once per chunk. Takes 10 seconds)')

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = Chroma(
    persist_directory='./chroma_db',
    embedding_function=embedding,
    collection_name='company_docs',
)

count = vectorstore._collection.count()
print(f'Reloaded vectore store with {count} chunks')

results = vectorstore.similarity_search('What is the shipping cost')
for doc in results:
    print(doc.page_content[:200])
    print()


import os 
if os.path.exists('./chroma_db'):
    print('Loading existing vecotr store...')
    vectorstore = Chroma(
        persist_directory='./chroma_db',
        embedding_function=embedding,
        collection_name='company_docs'
    )

else:
    print('Building new vectore store')
    vectore = Chroma.from_documents(
        documents=chunk,
        embedding=embedding,
        persist_directory='./chroma_db',
        collection_name='company_docs'
    )

results_with_scores = vectorstore.similarity_search_with_score(
    'What is the refund policy',
    k=4
)

print('Result with similarity scores:')
print('(Lower score = more similar in ChromaDB, it uses L2 distance)')
print()
for doc , score in results_with_scores:
    print(f'Score: {score:.4f}')
    print(f'Text: {doc.page_content[:100]}')
    print()

