from uuid import uuid4

from dotenv import load_dotenv
from pathlib import Path
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

CHUNK_SIZE=100
COLLECTION_NAME="real_estate"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"

llm=None
vector_store=None

def initialize_components():
    global llm,vector_store

    if llm is None:
        llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.9,max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs = {"trust_remote_code":True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )

def process_urls(urls): ## scraps data from url and stores in vectordb

    yield "Initializing components..."
    initialize_components()

    yield("Resetting the vector Store....")
    vector_store.reset_collection()
    
    yield("Loading data....")
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    yield("splitting text into chunks....")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=50  # Adjusted chunk_overlap to be smaller than chunk_size
    )

    docs = text_splitter.split_documents(data)

    yield("Add chunks to vector database...")
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs,ids=uuids)

    yield("Done adding docs to vector database...")

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")
    

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vector_store.as_retriever())

    results = chain.invoke({"question":query},return_only_outputs=True)
    
    sources = results.get("sources","")
    return results['answer'],sources

if __name__=="__main__":
    urls=["https://www.britannica.com/money/Tesla-Motors"]


    process_urls(urls)

    answer,sources =generate_answer("Who is Tesla's chief executive officer")

    print(f"Answer: {answer}")
    print(f"Source: {sources}")
    # results = vector_store.similarity_search(
    #     "Tesla Motors was formed to develop an electric sports car",
    #     k=2
    # )

    # print(results)