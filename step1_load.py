from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader


#Load the text file 
loader = TextLoader('E:\my-rag-system\data\company_policy.txt', encoding='utf-8')
documents = loader.load()

print(f"Number of documents :{len(documents)}")
print(f'Type {type(documents[0])}')
print(f'Metadata {documents[0].metadata}')
print(f'First 200 chars of content:')
print(documents[0].page_content[:200])

#For single pdf loader
loader = PyPDFLoader('data/document1.pdf')
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators = [
        '',
        '',
        '. ',
        '',
    ]
)
#Split all documents into chunk 
chunks = splitter.split_documents(documents)
print(f'Original documents: {len(documents)}')
print(f'Chunks created: {len(chunks)}')
print()
for i, chunk in enumerate(chunks):
    print(f'--- Chunk {i+1} ---')
    print(f'Characters: {len(chunk.page_content)}')
    print(f'Metadata: {chunk.metadata}')
    print(f'Content preview: {chunk.page_content}')
    print()



for i, doc in enumerate (documents):
    print(f'Page {i+1}: {len(doc.page_content)} characters')
    print(f'Metadata: {doc.metadata}')

def inspect_chunks(chunks):
    sizes = [len(c.page_content) for c in chunks]

    print('=== CHUNK QUALITY REPORT ===')
    print(f'Total chunks {len(chunks)}')
    print(f'Average size: {sum(sizes)/len(sizes):.0f} chars')
    print(f'Smalles Chunk: {min(sizes)} chars')
    print(f'Largest Chunk: {max(sizes)} chars')


    empty = [c for c in chunks if len(c.page_content.strip()) < 50]
    if empty:
        print(f'WARNINGS: {len(empty)} very small chunks found !')
        print('Consider increasing chunk_size or preprocessing documents.')

        print('===First 3 Chunks ===')
        for i, chunk in enumerate(chunks):
            print(f'{i+1} Chunks')
            print(chunk.page_content)
            print('----')

#For multiple loader 
new_loader = DirectoryLoader(
    path= 'data/',
    glob='**/*.txt',
    loader_cls=TextLoader,
    show_progress=True
)

all_docs = new_loader.load()

print(f'Total documents loaded: {len(all_docs)}')
all_documents = []

data_folder = Path('data')

for pdf_path in data_folder.glob('*.pdf'):
    loader = PyPDFLoader(str(pdf_path))
    all_documents.extend(loader.load())
    print(f'Loaded PDF: {pdf_path.name}')


for txt_path in data_folder.glob('*.txt'):
    loader = TextLoader(str(txt_path), encoding='utf-8')
    all_documents.extend(loader.load())
    print(f'Loaded TXT: {txt_path.name}')
