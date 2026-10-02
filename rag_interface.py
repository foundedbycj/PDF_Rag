import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
load_dotenv()
import streamlit as st


st.set_page_config(page_title='PDF_Summariser',page_icon="🤖")
st.title('PDF_Summariser',text_alignment='center')

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
6 ) BE confident  in your answers
7 ) IF there is any answer out of the context/chunks , say (THis is out of the scope),
8 ) Only answer whats withing the docs and given . Nothing beyond that"""

retriever = database.as_retriever(kwargs=4) # choose the most relevant chunks (here 4)




if 'messages' not in st.session_state: 
    st.session_state.messages = []

for messages in st.session_state.messages :

    with st.chat_message(messages['role']):
        st.markdown(messages['content'])

query = st.chat_input('start typing.....')

if query: # if its a query
    with st.chat_message('user'):
        st.markdown(query) #the question of the user will exactly be the query

    st.session_state.messages.append({ # append the interaction of the user
        'role':'user',
        'content':query
    })

    docs = retriever.invoke(query) # Pull the relevant chunk

    for doc in docs :
        context = ''
        context = context +  doc.page_content # loop through the docs and add them up
        
    message  = [SystemMessage(content=f'''Answer the following query with the given context :' \
         {context}  || Maintain the following rules : {rules}'''),*chat_history,HumanMessage(content=query)] # Pack up the messages


    response  = model.invoke(message)# query the model
    answer    = response.content # Take out the textual content of the query


    with st.chat_message('assistant'):   # For any message of AI Assistant, the reply will be the response
        st.markdown(answer)

    st.session_state.messages.append({  # append each interaction of the AI 
        'role':'assistant',
        'content' : answer
    })
    


