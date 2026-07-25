from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from services.llm import get_llm


def create_rag_chain(vector_store):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return qa_chain