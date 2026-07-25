from langchain_community.vectorstores import FAISS

from services.embeddings import get_embedding_model


def create_vector_store(chunks):
    """
    Creates a FAISS vector database from chunks.
    """

    embeddings = get_embedding_model()

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store