from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()


persistent_directory = 'database_dir'

embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

database =  Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={'hnsw:space':'cosine'}
)

retriever = database.as_retriever(kwargs={"k":3})

def query(q):
    answer = retriever.invoke(q)
    for _,doc in enumerate(answer):
        
        print (f'------------------------------ DOC {_}------------------------ \n {doc.page_content}')

query('what were the problems with RNNs')