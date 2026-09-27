import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import glob
load_dotenv()
from pypdf import PdfReader

from langchain_core.documents import Document

def extract_pdf(file):
    

    reader = PdfReader(file)
    text   = ''
    for page in reader.pages:

      
        text = text + page.extract_text()
    return text



def text_splitter(text,chunks=600,chunk_overlap=2):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunks,
        chunk_overlap= chunk_overlap
    )

    document = Document(page_content=text)
    return splitter.split_documents([document])


def create_vector_store(chunk,persist_directory = 'database_dir'): # enter whatever persist directory you need

    embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
    vector_store = Chroma.from_documents(
       documents  = chunk,
       embedding  = embedding_model,   
       persist_directory = persist_directory,
       collection_metadata= {'hnsw:space':'cosine'})
    
    return vector_store


    
    


def integrate (file):

    text =extract_pdf(file)#write your own file name
    chunks = text_splitter(text)
    vector_store = create_vector_store(chunks)
    return vector_store



integrate('docs/1706.03762v7.pdf')
