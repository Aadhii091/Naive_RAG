import streamlit as st

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

    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            text_chunks = parse_pdf(uploaded_file)   
            vectorstore = create_vectorstore(text_chunks)  

            st.session_state.vectorstore = vectorstore
            st.success("Document processed!")

# ---- MAIN CHAT UI ----
st.subheader("💬 Chat")

# Display chat history
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# ---- USER INPUT ----
user_query = st.chat_input("Ask something about the document...")

if user_query:
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload a document first.")
    else:
        # Display user message
        st.chat_message("user").markdown(user_query)
        st.session_state.chat_history.append(("user", user_query))

        # Generate response
        with st.spinner("Thinking..."):
            response = rag_query(
                query=user_query,
                chat_history=st.session_state.chat_history,
                vectorstore=st.session_state.vectorstore
            )

        # Display assistant response
        st.chat_message("assistant").markdown(response)
        st.session_state.chat_history.append(("assistant", response))