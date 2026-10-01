from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
load_dotenv()


# This is the exact same function as the retrieval pipeline , its just an extension of it by adding : Chat_history/memory 

# This is important to pick up context from the chat more effectively

persistent_directory = 'database_dir'

embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

database =  Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={'hnsw:space':'cosine'}
)


model = ChatOpenAI(model='gpt-4o')

chat_history = [] #chat history is empty at first

# Set up rules for the model to follow
rules = """1 ) DON'T Use vague sources,
2 ) BE clear and cut and concise,
3 ) If you dont know something thats outside of document, make sure you are honest about it
4 ) Dont Try to be overconfident on answers that you dont know
5 ) If any question is somehow ambiguous, ask the user to explain the question clearly
6 ) BE confident  in your answers"""



while True: #Creating a chat loop

    query = input('YOU :')
    if query.lower() in ['bye','exit']: # If the input query has "bye" or "exit", the conversation will terminate
        break

    retriever = database.as_retriever(kwargs=4) # Use the top 4 related chunks WRT query

    doc = retriever.invoke(query)
    for document in doc:

        context = document.page_content #take up the relevant docs separately as context

    messages = [SystemMessage(content=f'''Answer the following questions with the given context :' \
     {context}  || Maintain the following rules : {rules}'''),*chat_history,HumanMessage(content=query)]

    response = model.invoke(messages)

    print(f'AI : {response.content}')

    # Append each and every time chat is done , to store context
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response.content))
    
