import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configure Streamlit theme
st.set_page_config(
    page_title="Job Search Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with new color scheme
st.markdown("""
<style>
    /* Main theme - New gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e1b4b);
        color: #ffffff;
    }

    /* Header styling - Enhanced glow */
    .main-header {
        background: linear-gradient(90deg, #312e81, #1e1b4b);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: 0 0 40px rgba(139, 92, 246, 0.2);
        animation: headerGlow 4s infinite alternate;
    }

    /* Multiple animation keyframes for different elements */
    @keyframes headerGlow {
        0% {
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.2);
            transform: translateY(0);
        }
        50% {
            box-shadow: 0 0 40px rgba(139, 92, 246, 0.4);
            transform: translateY(-2px);
        }
        100% {
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.2);
            transform: translateY(0);
        }
    }

    @keyframes cardFloat {
        0% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-5px);
        }
        100% {
            transform: translateY(0);
        }
    }

    /* Enhanced search box */
    .stTextInput input {
        background-color: rgba(30, 27, 75, 0.7);
        color: #ffffff;
        border: 2px solid rgba(139, 92, 246, 0.3);
        border-radius: 15px;
        padding: 1.2rem;
        font-size: 1.1rem;
        transition: all 0.4s ease;
        backdrop-filter: blur(5px);
    }

    .stTextInput input:focus {
        border-color: #8b5cf6;
        box-shadow: 0 0 25px rgba(139, 92, 246, 0.4);
        background-color: rgba(30, 27, 75, 0.9);
    }

    /* Redesigned button */
    .stButton button {
        background: linear-gradient(45deg, #8b5cf6, #6d28d9);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        border-radius: 15px;
        font-weight: bold;
        transition: all 0.4s ease;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.5);
        background: linear-gradient(45deg, #6d28d9, #8b5cf6);
    }

    /* Enhanced job card */
    .job-card {
        background: linear-gradient(145deg, rgba(30, 27, 75, 0.9), rgba(49, 46, 129, 0.7));
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        border: 1px solid rgba(139, 92, 246, 0.2);
        transition: all 0.4s ease;
        box-shadow: 0 0 25px rgba(139, 92, 246, 0.1);
        backdrop-filter: blur(10px);
        animation: cardFloat 6s infinite ease-in-out;
    }

    .job-card:hover {
        transform: translateY(-7px) scale(1.01);
        box-shadow: 0 0 35px rgba(139, 92, 246, 0.3);
        border-color: rgba(139, 92, 246, 0.5);
        background: linear-gradient(145deg, rgba(49, 46, 129, 0.9), rgba(30, 27, 75, 0.7));
    }

    /* Improved metrics card */
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 27, 75, 0.8), rgba(49, 46, 129, 0.6));
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        border: 1px solid rgba(139, 92, 246, 0.2);
        transition: all 0.4s ease;
        backdrop-filter: blur(8px);
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.3);
        border-color: rgba(139, 92, 246, 0.4);
    }

    /* Enhanced link styling */
    a {
        color: #a78bfa;
        text-decoration: none;
        transition: all 0.3s ease;
    }

    a:hover {
        color: #c4b5fd;
        text-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(30, 27, 75, 0.7);
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #8b5cf6, #6d28d9);
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #6d28d9, #8b5cf6);
    }

    /* Download button enhancement */
    .download-button {
        background: linear-gradient(45deg, #7c3aed, #6d28d9);
        padding: 0.8rem 2rem;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        transition: all 0.4s ease;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
        text-align: center;
        margin-top: 2rem;
        display: inline-block;
    }

    .download-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.5);
        background: linear-gradient(45deg, #6d28d9, #7c3aed);
    }
</style>
""", unsafe_allow_html=True)


def create_search_url(job_name):
    """Create search URL from job name."""
    job_name_split = job_name.split()
    job_name_title = job_name.title()
    keywords = '+'.join(job_name_split)
    return f'https://www.timesjobs.com/candidate/job-search.html?searchType=personalizedSearch&from=submit&searchTextSrc=&searchTextText="{job_name_title}"&txtKeywords={keywords}&txtLocation='


def scrape_jobs(url):
    """Scrape job data from TimesJobs."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "lxml")

        job_titles_list = []
        job_titles = soup.find_all("h2", class_="heading-trun")
        for job_title in job_titles[:5]:
            job_titles_list.append(job_title["title"])

        company_names_list = []
        company_names = soup.find_all("h3", class_="joblist-comp-name")
        for company_name in company_names[:5]:
            company_name = company_name.text
            cleaned_company_name = re.sub(r'\s+', ' ', company_name).strip()
            company_names_list.append(cleaned_company_name)

        hyperlinks_list = []
        hyperlinks = soup.find_all("a", class_="posoverlay_srp")
        for hyperlink in hyperlinks[:5]:
            hyperlinks_list.append(hyperlink["href"])

        return pd.DataFrame({
            "Job Title": job_titles_list,
            "Company Name": company_names_list,
            "Link": hyperlinks_list,
            "Posted Date": [datetime.now().strftime("%Y-%m-%d")] * len(job_titles_list)
        })
    except Exception as e:
        st.error(f"Error scraping data: {str(e)}")
        return None


def create_company_distribution(df):
    """Create company distribution visualization."""
    company_counts = df['Company Name'].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=company_counts.index,
        values=company_counts.values,
        hole=0.5,
        marker=dict(colors=px.colors.sequential.Plasma)
    )])
    fig.update_layout(
        title="Company Distribution",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True
    )
    return fig


def main():
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("💼 Job Search Dashboard")
    st.markdown("""
    <p style='font-size: 1.2rem; color: #a5b4fc;'>
        Find your dream job with real-time job market insights
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Search section
    col1, col2 = st.columns([3, 1])
    with col1:
        job_name = st.text_input("", placeholder="Enter job title (e.g., Data Scientist)")
    with col2:
        search_button = st.button("🔍 Search Jobs")

    if search_button and job_name:
        with st.spinner("🔍 Searching for jobs..."):
            url = create_search_url(job_name)
            df = scrape_jobs(url)

            if df is not None and not df.empty:
                # Display metrics
                st.markdown("### 📊 Job Market Insights")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Total Jobs Found", len(df))
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Unique Companies", df['Company Name'].nunique())
                    st.markdown('</div>', unsafe_allow_html=True)
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Latest Update", df['Posted Date'].iloc[0])
                    st.markdown('</div>', unsafe_allow_html=True)

                # Visualizations
                st.markdown("### 📈 Analytics")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_company_distribution(df), use_container_width=True)
                with col2:
                    # Word cloud of job titles
                    st.markdown("#### 🏢 Popular Job Titles")
                    for title in df['Job Title'].unique():
                        st.markdown(f"• {title}")

                # Job listings
                st.markdown("### 🎯 Job Listings")
                for _, row in df.iterrows():
                    st.markdown('<div class="job-card">', unsafe_allow_html=True)
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"""
                        <h3>{row['Job Title']}</h3>
                        <p style='color: #a5b4fc;'>{row['Company Name']}</p>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <a href="{row['Link']}" target="_blank">
                            <div style='background: linear-gradient(45deg, #5271ff, #5c4dff);
                                      padding: 0.5rem 1rem;
                                      border-radius: 5px;
                                      text-align: center;
                                      color: white;'>
                                View Job
                            </div>
                        </a>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Export option
                st.download_button(
                    label="📥 Download Job List",
                    data=df.to_csv(index=False),
                    file_name=f"job_search_results_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.error("😕 No jobs found. Please try a different search term.")
    elif search_button:
        st.warning("⚠ Please enter a job title to search.")


if __name__ == "__main__":
    main()