from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
load_dotenv()


persistent_directory = 'database_dir'

embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

database =  Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={'hnsw:space':'cosine'}
)

retriever = database.as_retriever(kwargs={"k":3})






model = ChatOpenAI(model='gpt-4o')




def query(q):

    contents = ''

    sources = retriever.invoke(q)
    
    for _,doc in enumerate(sources ):

        contents += doc.page_content

    combined_input = contents
    messages = [
    SystemMessage(content="""You are a researcher at a top university, studying groundbreaking topics,
      and you can simplify research papers so they become useful for future work and easy to understand.
        Most papers today are highly technical and hard for laypeople.I want you to take the top‑page contents provided and, 
        based on the query, craft an answer that a five‑year‑old can understand. Use only the page content returned by the retriever;
          do not add external knowledge, alter the information, or introduce any fluff - If I write' #1', you should explain it as if to a 5‑year‑old, clear and very simple.
        - If I write #2, explain it in a mature, academic style that remains accessible and not overly technical.
        - If I write #3, respond as a PhD‑level researcher, providing detailed explanations with mathematical formulas
                  
                  
                  
        Be strict with the rules:
        1. Use only the PDF and the combined input provided.
        2. Avoid vague explanations.
        3. If the PDF lacks contextual information, acknowledge it and ask the user to consult other sources.
        4. When faced with a very complex question, provide a clear, thorough answer that remains understandable and avoid beating around the bush.),"""),
                  
    HumanMessage(content = contents)]

    answer = model.invoke(messages)

    print(answer.content)

    
query('whats the formula for attention? why was it used #2')                                                                                                                                                                                                                      