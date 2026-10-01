from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
load_dotenv()


persistent_directory = 'database_dir'

embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

database =  Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={'hnsw:space':'cosine'}
)


model = ChatOpenAI(model='gpt-4o')

chat_history = []

rules = """1 ) DON'T Use vague sources,
2 ) BE clear and cut and concise,
3 ) If you dont know something thats outside of document, make sure you are honest about it
4 ) Dont Try to be overconfident on answers that you dont know
5 ) If any question is somehow ambiguous, ask the user to explain the question clearly
6 ) BE confidnet in answers"""

while True:

    query = input('YOU :')
    if query.lower() in ['bye','exit']:
        break

    retriever = database.as_retriever(kwargs=4)

    doc = retriever.invoke(query)
    for document in doc:

        context = document.page_content

    messages = [SystemMessage(content=f'''Answer the following questions with the given context :' \
     {context}  || Maintain the following rules : {rules}'''),*chat_history,HumanMessage(content=query)]

    response = model.invoke(messages)

    print(f'AI : {response.content}')


    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response.content))
    
