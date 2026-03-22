from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader('data/company_policy.txt')
document = loader.load()


spliter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap = 100,
    length_function = len,
    separators=[
        '',
        '',
        '. ',
        ' ',
        '',
    ]
)

chunks = spliter.split_text(document)

print(f'Original documents: {len(document)}')
print(f'Chunk Created: {len(chunks)}')
print()

for i , chunk in enumerate(chunks):
    print(f'Chunks {i+1}')
    print(f'Characters: {len(chunk.page_content)}')
    print(f'Metadata: {chunk.metadata}')
    print(f'Content preview: {chunk.page_content[:100]}')
    print()

def inspect_chunk(chunks):
    size = [len(c.page_content) for c in chunks]
    print("===CHUNK QUALITY REPORT===")
    print(f'Total Chunks: {len(chunks)}')
    print(f'Average size {sum(size)/len(size):.0f}chars')
    print(f'Smallest Chunk: {min(chunks)}')
    print(f'Largest chunk: {max(size)} chars')
    print()

    empty = [c for c in chunks if len(c.page_content.stip()) < 50]
    if empty:
        print(f"WARNING: {len(empty)} very small chunks found!")
        print('Consider increasing chunk_size or pre-processing documents.')

    for i, chunk in enumerate(chunks):
        print(f'Chunks {i+1}')
        print(chunk.page_content)
        print('----')

inspect_chunk(chunks)