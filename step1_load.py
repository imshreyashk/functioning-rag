from langchain_community.document_loaders import TextLoader

#Load the text file 
loader = TextLoader('E:\my-rag-system\data\company_policy.txt', encoding='utf-8')
documents = loader.load()

#Inspect what we loaded
print(f'Number of documents: {len(documents)}')
print(f'Type: {type(documents[0])}')
print(f'Metadata: {documents[0].metadata}')
print(f'First 200 chars of content:')
print(documents[0].page_content[:200])

