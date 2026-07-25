import streamlit as st
from pathlib import Path

from services.pdf_loader import load_pdfs
from services.splitter import split_documents
from services.vector_store import create_vector_store
from services.rag import create_rag_chain
from services.image_loader import load_images


# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🤖",
    layout="wide"
)


# ----------------------------
# Load CSS
# ----------------------------

def load_css():

    css_path = Path("assets/style.css")

    if css_path.exists():

        css = css_path.read_text()

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True
        )


load_css()



# ----------------------------
# Session State
# ----------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# ----------------------------
# Header
# ----------------------------

st.markdown(
"""
<div class="main-title">
    🤖 Multimodal RAG Assistant
</div>

<div class="subtitle">
    Chat with PDFs, Images and Documents using AI
</div>

<br>
""",
unsafe_allow_html=True
)



# ----------------------------
# Sidebar
# ----------------------------

with st.sidebar:


    st.markdown(
    """
    <div class="upload-box">

    <h2>
    📂 Knowledge Base
    </h2>

    <p>
    Upload documents and build your AI knowledge base.
    </p>

    </div>
    """,
    unsafe_allow_html=True
    )


    pdfs = st.file_uploader(
        "📄 Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )


    images = st.file_uploader(
        "🖼 Upload Images",
        type=["png","jpg","jpeg"],
        accept_multiple_files=True
    )


    st.markdown(
    """
    <div class="upload-box">

    <h3>
    🧠 AI Pipeline
    </h3>

    <p>✔ PDF Loader</p>
    <p>✔ Image Extractor</p>
    <p>✔ Text Splitter</p>
    <p>✔ FAISS Vector Search</p>
    <p>✔ RAG Generation</p>

    </div>
    """,
    unsafe_allow_html=True
    )


    st.divider()


    process = st.button(
        "🚀 Build Knowledge Base"
    )


    clear_chat = st.button(
        "🗑️ Clear Chat"
    )


    reset_kb = st.button(
        "♻️ Reset Knowledge Base"
    )



# ----------------------------
# Clear Chat
# ----------------------------

if clear_chat:

    st.session_state.chat_history = []

    if "qa_chain" in st.session_state:

        if hasattr(
            st.session_state.qa_chain,
            "memory"
        ):

            st.session_state.qa_chain.memory.clear()

    st.rerun()



# ----------------------------
# Reset Knowledge Base
# ----------------------------

if reset_kb:


    keys = [
        "documents",
        "chunks",
        "vector_store",
        "qa_chain"
    ]


    for key in keys:

        if key in st.session_state:

            del st.session_state[key]


    st.session_state.chat_history = []

    st.rerun()



# ----------------------------
# Stats
# ----------------------------

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "📄 PDFs",
        len(pdfs) if pdfs else 0
    )


with col2:

    st.metric(
        "🖼 Images",
        len(images) if images else 0
    )


with col3:

    st.metric(
        "💬 Messages",
        len(st.session_state.chat_history)
    )


st.divider()



# ----------------------------
# Build Knowledge Base
# ----------------------------

if process:


    if not pdfs and not images:

        st.warning(
            "Please upload PDFs or Images first."
        )

        st.stop()



    progress = st.progress(0)

    status = st.empty()


    status.write(
        "📄 Loading documents..."
    )

    progress.progress(20)



    pdf_documents = []

    if pdfs:

        pdf_documents = load_pdfs(
            pdfs
        )



    status.write(
        "🖼 Processing images..."
    )

    progress.progress(35)



    image_documents = []

    if images:

        image_documents = load_images(
            images
        )



    documents = (
        pdf_documents +
        image_documents
    )



    status.write(
        "🧩 Splitting documents..."
    )

    progress.progress(55)



    chunks = split_documents(
        documents
    )



    status.write(
        "🔍 Creating embeddings..."
    )

    progress.progress(75)



    vector_store = create_vector_store(
        chunks
    )



    status.write(
        "🤖 Building RAG chain..."
    )

    progress.progress(90)



    qa_chain = create_rag_chain(
        vector_store
    )


    progress.progress(100)


    status.success(
        "Knowledge Base Ready 🚀"
    )



    st.session_state.qa_chain = qa_chain

    st.session_state.documents = documents

    st.session_state.chunks = chunks

    st.session_state.vector_store = vector_store



    st.success(
    f"""
    ✅ Knowledge Base Created

    📄 PDF Pages: {len(pdf_documents)}

    🖼 Images: {len(image_documents)}

    🧩 Chunks: {len(chunks)}

    🔍 FAISS Vector Store Ready
    """
    )



# ----------------------------
# Status
# ----------------------------

if "qa_chain" in st.session_state:

    st.success(
        "🟢 AI is ready. Ask questions below."
    )

else:

    st.info(
        "Upload files and build knowledge base first."
    )



# ----------------------------
# Chat Input
# ----------------------------

user_question = st.chat_input(
    "Ask anything about your documents..."
)



if user_question:


    st.session_state.chat_history.append(
        {
            "role":"user",
            "content":user_question
        }
    )


    if "qa_chain" not in st.session_state:


        answer = (
            "Please process your documents first."
        )

        sources = []



    else:


        with st.spinner(
            "Thinking..."
        ):


            response = (
                st.session_state
                .qa_chain
                .invoke(
                    {
                        "question": user_question
                    }
                )
            )


        answer = response["answer"]

        sources = response.get(
            "source_documents",
            []
        )



    st.session_state.chat_history.append(
        {
            "role":"assistant",
            "content":answer,
            "sources":sources
        }
    )



# ----------------------------
# Chat + Sources Layout
# ----------------------------

chat_col, source_col = st.columns(
    [2,1]
)



with chat_col:


    st.markdown(
        "## 💬 Conversation"
    )


    for chat in st.session_state.chat_history:


        with st.chat_message(
            chat["role"]
        ):

            st.markdown(
                chat["content"]
            )



with source_col:


    st.markdown(
        "## 📚 Retrieved Sources"
    )


    found_sources = []


    for chat in st.session_state.chat_history:


        if (
            chat["role"] == "assistant"
            and chat.get("sources")
        ):

            found_sources.extend(
                chat["sources"]
            )



    if found_sources:

        for index, doc in enumerate(found_sources):

            source_type = doc.metadata.get(
                "type",
                "pdf"
            )

            icon = "🖼️" if source_type == "image" else "📄"

            page_number = doc.metadata.get(
                "page",
                doc.metadata.get(
                    "page_number",
                    "N/A"
                )
            )

            st.markdown(
            f"""
            <div class="source-card">

            <h4>
            {icon} Source {index+1}
            </h4>

            <p>
            <b>File:</b>
            {doc.metadata.get("source","Unknown")}
            </p>

            <p>
            <b>Type:</b>
            {source_type.upper()}
            </p>

            <p>
            <b>Page:</b>
            {page_number}
            </p>

            </div>
            """,
            unsafe_allow_html=True
            )

            with st.expander(
                "View Content"
            ):

                st.write(
                    doc.page_content
                )

    else:

        st.info(
            "Sources will appear after asking questions."
        )