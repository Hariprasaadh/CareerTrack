import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

# Uploaded file will get added to this directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Function to load document
@st.cache_data
def load_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()  # Read the data, extract the text and store it in a document format
    return docs

# To store the document data chunks into a vector database
def setup_vectorstore(docs,vectorstore_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=100,
    )
    doc_chunks = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    vectorstore.save_local(vectorstore_path)
    return vectorstore

# Function to load the vectorstore locally
def load_vectorstore(vectorstore_path):
    if os.path.exists(vectorstore_path):
        return FAISS.load_local(vectorstore_path)
    else:
        return None


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

# Adding Streamlit UI
st.set_page_config(
    page_title="Research Bot",
    page_icon="📄",
    layout="centered"  # Put all content at the center of the page
)
st.title("🤖 Research Bot")
st.markdown("### **Your Research Paper Companion**")
st.markdown("_**No Ads | Free to Use**_ ✨")

# Sidebar for additional information
with st.sidebar:
    st.header("About Research Bot")
    st.markdown(
        """
        - 📚 **Purpose**: Helps students and researchers analyze and understand research papers efficiently.
        - 🛠 **Features**:
          - Upload PDF research papers.
          - Ask questions and get detailed answers.
          - AI-powered insights and clarifications.
        """
    )

# Initialize the chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # All the user questions and models' answers will be stored here

uploaded_file = st.file_uploader(label="📂 Upload your Research Paper (PDF only)", type=["pdf"])

vectorstore_path = "vectorstore"

if uploaded_file:
    file_path = f"{working_dir}/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Check if vectorstore already exists and load it
    if "vectorstore" not in st.session_state:  # To avoid rendering of Streamlit page
        st.session_state.vectorstore = load_vectorstore(vectorstore_path)
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = setup_vectorstore(load_document(file_path), vectorstore_path)

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

    st.success("📄 Research paper uploaded successfully! You can now ask your questions.")

else:
    st.warning("Please upload a PDF file to proceed.")

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("Chat History")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input box for questions
user_input = st.chat_input("💬 Ask your question here...")

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
