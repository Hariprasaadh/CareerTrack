from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
import os
import PyPDF2 as pdf
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

def get_response(resume_content, job_content):
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.5,
        groq_api_key=os.getenv("GROQCLOUD_API")
    )
    prompt_extract =    PromptTemplate.from_template(
        f"""
        
        ###Role Definition:
            Act as a highly skilled ATS (Application Tracking System) professional with expertise in evaluating resumes 
            across various tech fields, including software engineering, data science, data analysis, big data engineering, web development, and app development. 
            Compare resumes against job descriptions, ensuring high precision and actionable insights.
            
        ###Task Objective:
            Your task is to provide a comprehensive evaluation of a resume based on the job content, 
            web-scraped from the provided job link. The aim is to assist candidates in optimizing their resumes for a 
            competitive job market by identifying gaps and offering targeted improvement suggestions.    
           
           
        ###Response Format
            #Job Description Match:
                Match Percentage: "%"
                Reasons for Match Percentage:
                    Highlight specific strengths in the resume.
                    Explain gaps in responsibilities or skills.
                    Identify alignment with role-specific requirements.
                    Highlight the importance of missing keywords. 
                    
            #Missing Keywords
                Provide a detailed list of missing or weakly represented keywords.

            #Suggestions
                Offer five actionable tips to improve the resume from an employer's perspective. 
                These should include advice on content, structure, formatting, and keyword optimization.
                Add some Coursera and Udemy certifications to do with links to certificate course. (Provide only valid links and make sure all are valid course links and accessible)
                Prioritize "must-have" skills and qualifications from the job description in the evaluation.
        
        
        Resume Content : {resume_content}
        Content scarped from Job Application Link : {job_content}
        
            
        """
    )
    chain = prompt_extract | llm
    res = chain.invoke(input={'resume_content': resume_content, 'job_content': job_content})
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

st.set_page_config(
    page_title="Smart ATS Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');

        /* Main Background with animated gradient */
        .stApp {
            #background-image: linear-gradient(to left, #ff8177 0%, #ff867a 0%, #ff8c7f 21%, #f99185 52%, #cf556c 78%, #b12a5b 100%);
            background-image: linear-gradient(to top, #f43b47 0%, #453a94 100%);
            animation: gradient 15s ease infinite;
        }

        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Glass morphism effect for containers */
        .glass-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
        }

        /* Neon Title */
        .neon-title {
            font-family: 'Orbitron', sans-serif;
            color: #fff;
            text-shadow: 0 0 5px #fff,
                         0 0 10px #fff,
                         0 0 20px #0ff,
                         0 0 30px #0ff,
                         0 0 40px #0ff;
            animation: neon 1.5s ease-in-out infinite alternate;
        }

        @keyframes neon {
            from { text-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 20px #0ff; }
            to { text-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 20px #0ff, 0 0 30px #0ff, 0 0 40px #0ff; }
        }

        /* Modern Info Box */
        .info-box {
            background: rgba(0, 0, 0, 0.7);
            color: #fff;
            border-left: 10px solid #00f7ff;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transform: perspective(1000px) rotateX(5deg);
            transition: transform 0.3s ease;
        }

        .info-box:hover {
            transform: perspective(1000px) rotateX(0deg);
        }

        /* Animated Upload Button */
        .stFileUploader > div > button {
            background: linear-gradient(45deg, #FF512F, #DD2476);
            color: white;
            border: none;
            border-radius: 15px;
            padding: 15px;
            transition: all 0.3s ease;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        /* Submit Button */
        .stButton > button {
            background: linear-gradient(45deg, #00f7ff, #00ff95);
            color: black;
            font-weight: bold;
            padding: 15px 30px;
            border-radius: 25px;
            border: none;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }

        /* Text Area Styling */
        .stTextArea > div > div > textarea {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(5px);
            border: 2px solid rgba(255, 255, 255, 0.2);
            color: white;
            border-radius: 15px;
        }

        /* Response Container */
        .response-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            margin-top: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            animation: slideUp 0.5s ease-out;
        }

        @keyframes slideUp {
            from { transform: translateY(50px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        /* Progress Bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(to right, #00f7ff, #00ff95);
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(45deg, #00f7ff, #00ff95);
            border-radius: 5px;
        }

        /* Loading Animation */
        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 5px solid #f3f3f3;
            border-top: 5px solid #00f7ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        
    </style>
""", unsafe_allow_html=True)

# Modified Header with Neon Effect
st.markdown("<h1 class='neon-title' style='text-align: center; color: black;'>🎯 Smart ATS Resume Analyzer</h1>", unsafe_allow_html=True)

# Info Box with Glass Morphism
st.markdown("""
    <div class="info-box glass-container">
        <h4 style='color: #00f7ff;'>📌 How to use:</h4>
        <p style='color: #fff;'>1. Upload your resume in PDF format</p>
        <p style='color: #fff;'>2. Paste the job application link</p>
        <p style='color: #fff;'>3. Click submit and get instant AI-powered feedback!</p>
    </div>
""", unsafe_allow_html=True)

# [Rest of the code remains the same, but wrapped in glass-container class where appropriate]



# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📄 Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload Resume",
        type="pdf",
        help="Upload a PDF resume for analysis"
    )

with col2:
    st.markdown("### 🔗 Job Details")
    job_link = st.text_area(
        "Job Link",
        placeholder="Paste the job application link here...",
        height=100
    )

# Center the submit button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    submit = st.button("Analyze Resume 🚀")

# Loading spinner and response
if submit:
    if uploaded_file is not None and job_link:
        with st.spinner('Analyzing your resume... Please wait'):
            text = extract_text(uploaded_file)
            loader = WebBaseLoader(job_link)
            jl_content = loader.load().pop().page_content
            response = get_response(text, jl_content)

            # Display response in a styled container
            st.markdown("""
                <div class="response-container">
                    <h3 style="color:black;">📊 Analysis Results</h3>
                </div>
            """, unsafe_allow_html=True)
            st.write(response)
    else:
        st.error("Please upload both a resume and provide a job link.")

