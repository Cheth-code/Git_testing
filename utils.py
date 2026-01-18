import os
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings


def get_vector_db_retriever():
    persist_path = os.path.join(tempfile.gettempdir(), "finops.parquet")
    embeddings = OpenAIEmbeddings()

    # -------------------------------------------------
    # Load existing vector store if present
    # -------------------------------------------------
    if os.path.exists(persist_path):
        vectorstore = SKLearnVectorStore(
            embedding=embeddings,
            persist_path=persist_path,
            serializer="parquet",
        )
        num_docs = len(vectorstore._texts)
        k = min(4, num_docs)
        return vectorstore.as_retriever(search_kwargs={"k": k})

    # -------------------------------------------------
    # Load FinOps pages directly (WEB BASED)
    # -------------------------------------------------
    urls = [
        "https://learn.finops.org/introduction-to-finops",
        "https://learn.finops.org/introduction-to-focus",
    ]

    loader = WebBaseLoader(urls)
    docs = loader.load()

    if not docs:
        raise RuntimeError("No documents loaded from FinOps URLs")

    # -------------------------------------------------
    # Chunk documents
    # -------------------------------------------------
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=100,
    )
    doc_splits = splitter.split_documents(docs)

    # -------------------------------------------------
    # Build vector store
    # -------------------------------------------------
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embeddings,
        persist_path=persist_path,
        serializer="parquet",
    )
    vectorstore.persist()

    num_docs = len(vectorstore._texts)
    k = min(4, num_docs)
    return vectorstore.as_retriever(search_kwargs={"k": k})
