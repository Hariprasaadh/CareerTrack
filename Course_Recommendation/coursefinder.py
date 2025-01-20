import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

st.set_page_config(
    page_title="Course Finder",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #2d2d2d, #1a1a1a);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 0 20px rgba(0, 128, 255, 0.2);
    }

    /* Search box styling */
    .stTextInput input {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #4a4a4a;
        border-radius: 5px;
    }

    /* Button styling */
    .stButton button {
        background: linear-gradient(45deg, #0066cc, #0099ff);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(0, 128, 255, 0.3);
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(0, 128, 255, 0.5);
    }

    /* Course card styling */
    .course-card {
        background-color: #2d2d2d;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 0 15px rgba(0, 128, 255, 0.1);
        transition: all 0.3s ease;
    }

    .course-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 20px rgba(0, 128, 255, 0.2);
    }

    /* Link styling */
    a {
        color: #0099ff;
        text-decoration: none;
        transition: color 0.3s ease;
    }

    a:hover {
        color: #00ccff;
    }

    /* Rating badge styling */
    .rating-badge {
        background: linear-gradient(45deg, #00cc66, #00ff80);
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        color: #1a1a1a;
        font-weight: bold;
    }

    /* Level badge styling */
    .level-badge {
        background: linear-gradient(45deg, #ff6600, #ff9933);
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        color: #1a1a1a;
        font-weight: bold;
    }

    /* Skills section styling */
    .skills-section {
        background-color: #363636;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 0.5rem;
    }

    /* Download button styling */
    .download-button {
        background: linear-gradient(45deg, #009933, #00cc44);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 0 15px rgba(0, 204, 68, 0.3);
    }
</style>
""", unsafe_allow_html=True)


def create_search_url(course_name):
    course_split = course_name.split()
    search_terms = '%20'.join(course_split)
    return f"https://www.coursera.org/search?query={search_terms}"


def get_course_data(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        educators_list = [educator.text for educator in
                          soup.find_all("p", class_="cds-ProductCard-partnerNames css-vac8rf")[:5]]
        course_titles_list = [title.text for title in soup.find_all("h3", class_="cds-CommonCard-title css-6ecy9b")[:5]]
        skills_list = [skill.text for skill in soup.find_all("div", class_="cds-CommonCard-bodyContent")[:5]]
        ratings = soup.find_all("div", class_="cds-RatingStat-meter")
        ratings_list = [rate.text.split("Rating")[0].strip() for rate in ratings[:5]]
        hyperlinks_list = ["https://www.coursera.org" + hyperlink["href"] for hyperlink in
                           soup.find_all("a",
                                         class_="cds-119 cds-113 cds-115 cds-CommonCard-titleLink css-vflzcf cds-142")[
                           :5]]
        levels_list = [level.text.split()[0] for level in soup.find_all("div", class_="cds-CommonCard-metadata")[:5]]

        images_div = soup.find_all("img")
        images_list = []
        for i in images_div[5:10]:
            if 'src' in i.attrs:
                images_list.append(i['src'])
            else:
                images_list.append('')

        return pd.DataFrame({
            "Course Title": course_titles_list,
            "Educator": educators_list,
            "Skills": skills_list,
            "Rating": ratings_list,
            "Link": hyperlinks_list,
            "Level": levels_list,
            "Images Link": images_list
        })

    except requests.RequestException as e:
        st.error(f"Error fetching data: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None


def main():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🎓 Course Finder")
    st.markdown("""
    <p style='font-size: 1.2rem; color: #cccccc;'>
        Discover your next learning opportunity on Coursera with our enhanced search tool.
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Search section
    col1, col2 = st.columns([3, 1])
    with col1:
        course_name = st.text_input("", placeholder="Enter course name (e.g., Python Programming)")
    with col2:
        search_button = st.button("🔍 Search Courses")

    if search_button and course_name:
        with st.spinner("🔍 Searching for the best courses..."):
            url = create_search_url(course_name)
            course_data = get_course_data(url)

            if course_data is not None and not course_data.empty:
                st.markdown("### 🎯 Search Results", unsafe_allow_html=True)

                # Display courses in enhanced cards
                for idx, row in course_data.iterrows():
                    st.markdown('<div class="course-card">', unsafe_allow_html=True)
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.image(row["Images Link"], use_container_width=True)

                    with col2:
                        st.markdown(f"""
                        <h3><a href="{row['Link']}" target="_blank">{row['Course Title']}</a></h3>
                        <p><strong>🏫 Educator:</strong> {row['Educator']}</p>
                        <span class="level-badge">📚 {row['Level']}</span>
                        <span class="rating-badge">⭐ {row['Rating']}</span>
                        <div class="skills-section">
                            <strong>🎯 Skills you'll gain:</strong><br>
                            {row['Skills']}
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                # Download section
                st.markdown('<div class="download-button">', unsafe_allow_html=True)
                csv = course_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="coursera_search_results.csv",
                    mime="text/csv"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("😕 No courses found. Please try a different search term.")
    elif search_button:
        st.warning("⚠ Please enter a course name to search.")


if __name__ == "__main__":
    main()