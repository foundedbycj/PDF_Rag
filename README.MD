# HI Welcome To PDF_Summariser RAG


<h1> PREFACE </h1>
<br>
<h3> PDF_Summariser RAG is a small mini RAG agent that you can use as a researcher to dump all the documents (upon which you all are working) and ask queries,summary and review of the respective document without needing to slide through pages manually.</h3>

<h1> What we will be doing </h1>
<ol> 
<li> Creating <code>ingestion pipeline </code> to convert the pdf textual data into vectors in a vector DB.</li>
<li> Creating <code>retrieval pipeline </code> to convert user queries into a vectors and apply the search rule </li>
<li> Creating <code>RAG interface </code> to make a visual interface for the RAG Agent to be used
<li> Integrating them together to form a RAG </li>

</ol>


# NOTE (It is fully local rag and is not deployed anywhere)

<H1 > Tech-stack/Tools used </h1>
<ul>
<li> ChromaDB
<li> LangChain
<li> OpenAI
<li> Streamlit
