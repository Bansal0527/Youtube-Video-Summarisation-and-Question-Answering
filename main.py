from langchain.document_loaders import YoutubeLoader
from langchain.llms import  Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import GPT4AllEmbeddings




embeddings = GPT4AllEmbeddings()

# Load
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=byJgRDFdUj0", add_video_info=True)

docs =loader.load()


# Splitting into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
# LLM

llm = Ollama(base_url='http://localhost:11434',model="llama2")






# Embeddings
persist_directory = 'docs/YT-1/'
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=GPT4AllEmbeddings(),
    persist_directory=persist_directory
)
vectordb.persist()



# Summarizer
chain = load_summarize_chain(llm=llm,chain_type='map_reduce',)
response = chain.run(splits)
print(response)

while True:
    question=input("Ask the question :")
      # chain
    if(question == exit):
        exit()
    qa_chain = RetrievalQA.from_chain_type(
       llm,
       retriever=vectordb.as_retriever()
     )

    result = qa_chain({"query": question})
    print(result["result"])