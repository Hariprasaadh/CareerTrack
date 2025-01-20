import os
from dotenv import load_dotenv
import streamlit as st
from gitingest import ingest
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

@st.cache_data
def process_repository(repo_url_or_path):
    summary, tree, content = ingest(repo_url_or_path)
    return summary, tree, content

def setup_vectorstore_from_repo(content, vectorstore_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(content, embeddings)
    vectorstore.save_local(vectorstore_path)
    return vectorstore

def load_vectorstore(vectorstore_path):
    if os.path.exists(vectorstore_path):
        return FAISS.load_local(vectorstore_path)
    else:
        return None

# Create Conversational Chain
def create_chain(vectorstore):
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True,
        max_content_messages=4
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False
    )
    return chain

# Streamlit App Layout
st.set_page_config(
    page_title="GitHub Repo Assistant",
    page_icon="🗂️",
    layout="centered"
)
st.title("🤖 Chat with GitHub Repo")
st.markdown("### **Understand and Explore Repositories with Ease**")
st.markdown("_**No Ads | Free to Use**_ ✨")

# Sidebar for additional information
with st.sidebar:
    st.header("About GitHub Repo Assistant")
    st.markdown(
        """
        - 🛠 **Purpose**: Helps developers analyze and understand GitHub repositories efficiently.
        - 📚 **Features**:
          - Analyze the repository's structure.
          - Ask questions about the codebase.
          - Get summaries of the repository's content.
        """
    )

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

repo_url_or_path = st.text_input("Enter the GitHub Repo URL or Local Path", "")

vectorstore_path = "vectorstore_repo"

if repo_url_or_path:
    with st.spinner("Processing the repository..."):
        summary, tree, content = process_repository(repo_url_or_path)
        st.success("Repository processed successfully!")

        # Display summary and tree structure
        st.subheader("Repository Summary")
        st.markdown(summary)
        st.subheader("Repository Tree")
        st.code(tree)

        # Check and set up vectorstore
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = load_vectorstore(vectorstore_path)
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = setup_vectorstore_from_repo(content, vectorstore_path)

        if "conversation_chain" not in st.session_state:
            st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("Chat History")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input for questions
user_input = st.chat_input("💬 Ask your question about the repository here...")

if user_input:
    # Add the user's input to the chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Prepare the chat history for the chain
        formatted_chat_history = [
            (message["role"], message["content"])
            for message in st.session_state.chat_history
        ]
        response = st.session_state.conversation_chain({
            "question": user_input,
            "chat_history": formatted_chat_history
        })
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        # Add the assistant's response to the chat history
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

st.markdown("---")
