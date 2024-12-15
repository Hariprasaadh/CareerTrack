import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio
from utils import clean_text

# Set page configuration as the first Streamlit command
st.set_page_config(layout="wide", page_title="Job Email Generator", page_icon="📧")

# Custom CSS for styling
st.markdown(
    """
    <style>
    
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stCode {
        border-radius: 5px;
        background-color: #f9f9f9;
        padding: 10px;
    }
    .sidebar .stTextInput {
        border: 1px solid #ccc;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def create_streamlit_app(llm, portfolio, clean_text):
    st.title("📧 AI-Powered Job Application Email Generator")
    st.subheader("Streamline your job application process with AI")

    with st.sidebar:
        st.header("🔗 Input Settings")
        url_input = st.text_input("Enter Job URL",
                                  value="https://jobs.nike.com/job/R-46630?from=job%20search%20funnel",
                                  help="Paste the URL of the job listing here.")
        submit_button = st.button("Submit 🚀")

    if submit_button:
        with st.spinner("Analyzing job posting..."):
            try:
                # Load job data
                loader = WebBaseLoader([url_input])
                raw_data = loader.load().pop().page_content

                # Extract jobs
                data = clean_text(raw_data)
                portfolio.load_portfolio()
                jobs = llm.extract_jobs(data)


                # Display job details
                for idx, job in enumerate(jobs):
                    links = portfolio.query_links(job.get('skills', []))
                    email = llm.write_mail(job, links)
                    st.code(email, language='markdown')
                    st.download_button(
                        "📩 Download Email",
                        email,
                        file_name=f"job_email_{idx + 1}.md",
                        key=f"download_{idx}"
                    )
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    create_streamlit_app(chain, portfolio, clean_text)
