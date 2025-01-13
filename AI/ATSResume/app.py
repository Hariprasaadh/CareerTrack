from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
import os
import PyPDF2 as pdf
from langchain_core.prompts import PromptTemplate
import json

load_dotenv()

def get_response(text, jd):
    llm = ChatGroq(
        model_name="llama-3.1-70b-versatile",
        temperature=0.5,
        groq_api_key=os.getenv("GROQCLOUD_API")
    )
    prompt_extract = PromptTemplate.from_template(
        """
        Act Like a skilled or very experienced ATS(Application Tracking System) with a deep understanding of tech fields such as software engineering, data science, data analysis, big data engineering, web development, app development, and more. Your task is to evaluate the resume based on the given job description. You must consider that the job market is highly competitive and provide the best assistance for improving the resumes. Assign a percentage match based on the job description (JD) and highlight the missing keywords with high accuracy.
        resume:{text}
        description:{jd}
        Please provide the response in the following structured format:
        1. **Job Description Match**: "%" Give 4 reasons
        2. **Missing Keywords**: [] (In points)
        3. **Profile Summary**: ""   (In a descriptive manner)
        4. **Suggestions**: Provide 5 tips for improving the resume from the perspective of an experienced employer. Ensure that the output is clearly formatted with headings in **bold** and use a larger font for better readability.
        """
    )
    chain = prompt_extract | llm
    res = chain.invoke(input={'text': text, 'jd': jd})
    return res.content

def extract_text(uploaded_file):
    try:
        reader = pdf.PdfReader(uploaded_file)
        pages = len(reader.pages)
        text = ""
        for page_num in range(pages):
            page = reader.pages[page_num]
            text += str(page.extract_text())
        return text
    except Exception as e:
        st.error(f"Error occurred while extracting text: {str(e)}")
        return ""


st.set_page_config(page_title="Smart ATS", page_icon="📄", layout="wide")
st.title("Smart ATS - Enhance Your Resume 📈")
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4caf50;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stTextInput>label {
            font-size: 18px;
        }
        .stTextArea>label {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# Input fields
st.text("Welcome to the Smart ATS tool! Upload your resume and paste the job description to receive feedback.")
jd = st.text_area("Paste the Job Description 📋", height=200)
uploaded_file = st.file_uploader("Upload Your Resume (PDF) 📄", type="pdf", help="Upload a PDF resume for analysis.")

# Submit button
submit = st.button("Submit 📥")

# Handling the response when the user submits the resume
if submit:
    if uploaded_file is not None and jd:
        text = extract_text(uploaded_file)
        response = get_response(text, jd)
        st.write(response)


