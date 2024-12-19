import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()


#Uploaded file will get added to this directory
working_dir=os.path.dirname(os.path.abspath(__file__))

#Function to load document
def load_document(file_path):
    loader = UnstructuredPDFLoader(file_path)
    docs = loader.load()  #Read the data,extract the text and store it in a document format
    return docs

#To store the document data chunks into a vector database
def setup_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    doc_chunks = text_splitter.split_documents(docs)
    vectorstore=FAISS.from_documents(doc_chunks,embeddings)
    return vectorstore



def create_chain(vectorstore):
    llm=ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")

    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        memory=memory,
        verbose=False
    )
    return chain

#Adding Streamlit UI
st.set_page_config(
    page_title="Research Paper Analyzer",
    page_icon="📄",
    layout="centered"  #Put all content at the center of the page
)
st.title("🤖 Research Bot")

#Initialize the chat history in streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  #All the user questions and models answer will be stored here

uploaded_file = st.file_uploader(label="Upload your Research Paper", type=["pdf"])

if uploaded_file:
    file_path = f"{working_dir}/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "vectorstore" not in st.session_state:   #To avoid rendering of streamlit page
        st.session_state.vectorstore = setup_vectorstore(load_document(file_path))

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

#To display previous messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask AI...")

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






