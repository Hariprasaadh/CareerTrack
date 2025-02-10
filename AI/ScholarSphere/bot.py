import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
GOOGLE_API_KEY=st.secrets['GOOGLE_API_KEY']


load_dotenv()
genai.configure(api_key=os.getenv(GOOGLE_API_KEY))


def get_pdf_text(pdf_docs):
    text = ""
    if not isinstance(pdf_docs, list):
        pdf_docs = [pdf_docs]

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, use your knowledge and answer from that.

    Context:
    {context}?

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def get_ai_response(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response['output_text']


def chat_interface():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_ai_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    st.set_page_config(page_title="Research Bot", page_icon=":books:", layout="wide")

    st.title("🤖 Research Bot")
    st.markdown("### **Your Research Paper Companion**")
    st.markdown("_**No Ads | Free to Use**_ ✨")

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

    if "messages" not in st.session_state:
        st.session_state.messages = []

    uploaded_file = st.file_uploader(label="📂 Upload your Research Paper (PDF only)", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Submit & Process"):
            with st.spinner("Processing your files..."):
                try:
                    raw_text = get_pdf_text(uploaded_file)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed and indexed successfully!")
                    st.session_state.messages = []
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    chat_interface()

    st.sidebar.info(
        """Unveil the secrets of knowledge with Research Bot, 
    your mystical guide through the world of research."""
    )


if __name__ == "__main__":
    main()