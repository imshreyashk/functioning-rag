from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from pydantic import BaseModel, Field



load_dotenv()


embedding = HuggingFaceEmbeddings(
    model='sentence-transformers/all-MiniLM-L6-v2'
)

#1. LOAD EXISTING VECTOR STORE 
vectorstore = Chroma(
    persist_directory='./chroma_db',
    embedding_function=embedding,
    collection_name='company_docs'
)

#2. CREATE THE RETRIEVER
retriver = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 3}
)

#3. CREATE THE PROMPT TEMPLATE
prompt = ChatPromptTemplate.from_messages([
    ('system', '''You are a helpful customer service assistant.
     Answer the user's question based ONLY on the context provided below.
     If the anaswer is not in the context, say 'I dont't have information about that in my 
     knowledge base.
     Do not make up information .
     
    Context: {context}'''),
    ('human', '{question}')
])

#4. CREATE THE LLM

llm = HuggingFacePipeline.from_model_id(
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens":512,
        "temperature":0.1,
        "do_sample":False,
    },
)

chat = ChatHuggingFace(llm=llm)

#5. HELPER FUNCTION

def format_docs(docs):
    formatted = []
    for i , doc in enumerate(docs):
        source = doc.metadata.get('source', 'unkown')
        formatted.append(f'[Source {i+1} : {source}]\n {doc.page_content}')
    return '\n\n'.join(formatted)

#6. BUILD THE CHAIN 

rag_chain = (
    {
        'context': retriver | format_docs,
        'question': RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

#7. TEST THE CHAIN

test_question = [
    'What is the refund policy',
    'How long does standard shipping take',
    'Is my personal data safe',
    'What is the CEO s salary',
]
for question in test_question:
    print(f'Questions {question}')
    answer = rag_chain.invoke(question)
    print(f'Answer: {answer}')
    print()


class RAGResponse (BaseModel):
    answer: str = Field(description='Answer to the question')
    sources: list[str] = Field(description='List of source document names used')
    confidence: str = Field(description='high, medium, or low')

structured_llm = llm.with_structured_output(RAGResponse)

structured_prompt = ChatPromptTemplate.from_messages([
    ('system', '''Answer the question using ONLY the context below.
    Return your response in the required format with answer, sources, and confidence.
    If the answer is not in the context, say so and set confidence to low.
    
     Context: {context}'''),
     ('human', '{question}')
])

#Build the structured chain

structure_rag_chain = (
    {
        'context' : retriver | format_docs,
        'question' : RunnablePassthrough()
    }
    | structured_prompt
    | structured_llm
)

# Test it 
response = structure_rag_chain.invoke('What is the refund policy')
print(f'Answer: {response.answer}')
print(f'Source: {response.sources}')
print(f'Confidence: {response.confidence}')

