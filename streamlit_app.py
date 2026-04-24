import fitz
import streamlit as st
from ingestion_pipeline import create_vector_db
from answer_generation import ask_question
from chunking_strategies import semantic_chunking
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage


# ---- PAGE CONFIG ----
st.set_page_config(page_title="RAG Chat", page_icon="🧠", layout="wide")

st.title("🧠 Conversational RAG System")

# ---- SESSION STATE ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ---- SIDEBAR (DOCUMENT UPLOAD) ----
with st.sidebar:
    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    def parse_pdf(uploaded_file):
        """
        Input: Streamlit uploaded PDF file
        Output: List of text chunks with metadata
        """

        # ---- READ PDF ----
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        raw_docs = []

        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")

            # Skip empty pages
            if not text.strip():
                continue

            raw_docs.append(
                Document(
                page_content=text,
                metadata={
                    "page": page_num,
                    "source": uploaded_file.name
                    }
                )
            )

        chunks = semantic_chunking(raw_docs)

        return chunks
    
    if uploaded_file is not None and st.session_state.vectorstore is None:
        with st.spinner("Processing PDF..."):
            text_chunks = parse_pdf(uploaded_file)
            vectorstore = create_vector_db(text_chunks, f"db/{uploaded_file.name}")

            st.session_state.vectorstore = vectorstore
            st.success("Document processed!")
    else:
        st.success("Document processed!")

# ---- MAIN CHAT UI ----
st.subheader("💬 Chat")

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)

    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ---- USER INPUT ----
user_query = st.chat_input("Ask something about the document...")

if user_query:
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload a document first.")
    else:
        # Display user message
        st.chat_message("user").markdown(user_query)

        # Generate response
        with st.spinner("Thinking..."):
            response = ask_question(
                user_query, 
                st.session_state.chat_history,
                st.session_state.vectorstore
                )

        # Display assistant response
        st.chat_message("assistant").markdown(response)