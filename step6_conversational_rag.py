from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings , ChatHuggingFace, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ZepChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory, RunnableParallel, RunnableLambda
from operator import itemgetter

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

vectore = Chroma(
    persist_directory='./Chromadb',
    embedding_function=embedding,
    collection_name='company_docs'
)

retrival = vectore.as_retriever(search_kwargs={'k':3})

llm = HuggingFacePipeline.from_model_id(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    task="text generation",
    pipeline_kwargs={
        "max_new_token":512,
        "temperature":0,
        "do_sample":False
    },
)

chat = ChatHuggingFace(llm)

contextual_prompt = ChatPromptTemplate.from_messages([
    ('system', '''Given a chat histroy and the latest user question
    reformulate the question to be stanalone and self-contained.
    Do not anwer the questio -just reformulate it.
    If the questio is already standalone, retun it unchanged.'''),

MessagesPlaceholder('chat_history'), 
('human', '{input}')

])

contextualize_q_chain = contextual_prompt | chat | StrOutputParser

def get_contextual_retriever_input(input_dict):
    '''If ther's chat history, reformulate question first.'''
    if input_dict.get('chat_histroy'):
        standalon_q = contextualize_q_chain.invoke({
            'input': input_dict['input'],
            'chat_history': input_dict['chat_history']
        })
        return standalon_q
    else:
        return input_dict['input']
    

qa_prompt = ChatPromptTemplate.from_messages([
    ('system', '''You are a helpful assistant.
    Answer the question using ONLY the context below.
    If the answer is not in the content , say don't know.
    Context:
    {context}'''),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')

])

def format_docs(docs):
    formulated = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'unkown')
        formulated.append(f'[Source {i+1}: {source}] {doc.page_content}')
        return '\n\n.join'
    
conversation_rag = (
    RunnableParallel ({
        'context' : RunnableLambda(get_contextual_retriever_input) | retrival | format_docs,
        'input': itemgetter('input'),
        'chat_history': itemgetter('chat_history')
    })
    |qa_prompt
    | chat
    | StrOutputParser()
)


session_histories = {}

def get_session_histore(session_id : str):
    if session_id not in session_histories:
        session_histories[session_id] = ZepChatMessageHistory
    return session_histories[session_id]

chain_with_memory = RunnableWithMessageHistory(
    conversation_rag,
    get_session_histore,
    input_messages_key='input',
    history_messages_key='chat_history'
)


session_id = 'user 001'
config = {'configurable': {'session_id': session_id}}

questions = [
    'What is the refund policy',
    'What about digital products',
    'How long does the refund take to process',
    'And what about international shipping',
]

for question in questions:
    print(f'uswer: {question}')
    response = chain_with_memory.invoke({'input': question}, config=config)
    print(f'Bot: {response}')
    print()

    