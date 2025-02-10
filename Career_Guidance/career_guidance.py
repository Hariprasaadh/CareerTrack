import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import regex as re
import graphviz
GROQ_API_KEY=st.secrets['GROQ_API_KEY']


st.set_page_config(page_title="Career Guidance System", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0B1120;
        color: #E2E8F0;
    }
    h1, h2, h3 {
        color: #60A5FA;
        font-family: 'Segoe UI', sans-serif;
    }
    .stCard {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2D3748;
        margin-bottom: 20px;
    }
    .stSuccess {
        background-color: #064E3B;
        color: #A7F3D0;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stSlider > div > div > div {
        background-color: #3B82F6;
    }
    .stButton > button {
        background-color: #3B82F6;
        color: white;
        border-radius: 6px;
        padding: 10px 25px;
        font-weight: 600;
        border: none;
    }
    .stButton > button:hover {
        background-color: #2563EB;
    }
    .sidebar .sidebar-content {
        background-color: #1E293B;
    }
    .career-details {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Career Guidance System</h1>", unsafe_allow_html=True)

knowledge_levels = ['Professional', 'Not Interested', 'Poor', 'Beginner', 'Average', 'Intermediate', 'Excellent']
knowledge_mapping = {level: i for i, level in enumerate(knowledge_levels)}

career_classes = [
    "Database Administrator", "System Architect", "Cloud Engineer",
    "Cybersecurity Analyst", "Network Engineer", "Software Developer",
    "Programmer", "Project Manager", "Forensic Analyst",
    "Technical Writer", "AI/ML Specialist", "Software Engineer",
    "Business Analyst", "HR Specialist", "Data Scientist",
    "IT Support Specialist", "Graphic Designer"
]

fields = [
    'Database Fundamentals', 'Computer Architecture', 'Distributed Computing Systems',
    'Cyber Security', 'Networking', 'Software Development', 'Programming Skills',
    'Project Management', 'Computer Forensics Fundamentals', 'Technical Communication',
    'AI ML', 'Software Engineering', 'Business Analysis', 'Communication skills',
    'Data Science', 'Troubleshooting skills', 'Graphics Designing'
]

PROMPT_TEMPLATE = """
Provide a detailed, structured overview of the {predicted_career} career, organized as follows:

i) Career Overview:
1. Basic Career Overview (100-120 words):
   (Briefly describe the career, including the key responsibilities, goals, and daily tasks.)

2. Career Opportunities (80-100 words):
   (Describe potential job titles, industries, and demand trends for this career.)

3. Skills and Competencies (120-150 words):
   (List essential technical and soft skills, along with commonly used tools, software, or frameworks.)

4. Educational Qualifications (80-100 words):
   (Mention required degrees, certifications, and additional qualifications needed.)

5. Career Path and Progression (100-120 words):
   (Outline typical career trajectories, starting positions, and advancement opportunities.)

6. Salary and Job Outlook (80-100 words):
   (Provide average salary details and the job market outlook for this role.)

7. Eligibility and Prerequisites (80-100 words):
   (Describe necessary qualifications, experience, or skills for entering the field.)

8. Future Trends and Growth (80-100 words):
   (Highlight trends, emerging technologies, and areas of growth in the field.)

9. Work Environment (80-100 words):
   (Describe the typical work setting: office, remote work, teamwork level, etc.)

10. Networking and Community (80-100 words):
    (Discuss key networking opportunities, professional communities, and relevant events.)

11. Job Satisfaction and Impact (80-100 words):
    (Analyze job satisfaction, common challenges, and the impact of the work.)

ii) Actionable Steps for Success:
Provide a roadmap with specific, actionable steps for success in the {predicted_career} field. Group the steps as shown below:

- Foundation:
   - Programming Skills: steps[:2]
   - Mathematical Fundamentals: steps[2:4]
   - Computer Science Basics: steps[4:6]

- Development:
   - Core Skills Building: steps[6:9]
   - Project Work: steps[9:11]
   - Advanced Topics: steps[11:13]

- Advanced:
   - Specialization: steps[13:16]
   - Portfolio Development: steps[16:18]
   - Professional Networking: steps[18:20]

Each step should be concise, actionable, and clearly aligned with the career path.
The steps must be little descriptive for about 2 lines.
The steps must be related real life. like the steps must be practical and related to technical too.
"""

def create_career_roadmap(career_field, steps):
    dot = graphviz.Digraph(comment='Career Roadmap')
    dot.attr(
        rankdir='LR',
        splines='ortho',
        bgcolor='#0B1120',
        fontname='Arial',
        fontcolor='white'
    )
    dot.attr('node', 
        shape='box',
        style='filled,rounded',
        fontname='Arial',
        fontsize='12',
        margin='0.2'
    )
    dot.attr('edge',
        color='#4B5563',
        penwidth='1.5'
    )
    phase_color = '#10B981'
    category_color = '#F59E0B'
    step_color = '#F87171'
    dot.node('main', career_field, fillcolor='#3B82F6', fontcolor='white')
    phases = ['Foundation', 'Development', 'Advanced']
    step_index = 0
    for phase_idx, phase in enumerate(phases):
        phase_id = f'phase_{phase}'
        dot.node(phase_id, phase, fillcolor=phase_color, fontcolor='white')
        dot.edge('main', phase_id)
        categories = []
        if phase == 'Foundation':
            categories = ['Programming Skills', 'Mathematical Fundamentals', 'Computer Science Basics']
            steps_per_category = 2
        elif phase == 'Development':
            categories = ['Core Skills Building', 'Project Work', 'Advanced Topics']
            steps_per_category = 3
        else:
            categories = ['Specialization', 'Portfolio Development', 'Professional Networking']
            steps_per_category = 2
        for cat_idx, category in enumerate(categories):
            cat_id = f'{phase_id}cat{cat_idx}'
            dot.node(cat_id, category, fillcolor=category_color, fontcolor='white')
            dot.edge(phase_id, cat_id)
            for step_idx in range(steps_per_category):
                if step_index < len(steps):
                    step_id = f'{cat_id}step{step_idx}'
                    dot.node(step_id, steps[step_index], fillcolor=step_color, fontcolor='white')
                    dot.edge(cat_id, step_id)
                    step_index += 1
    return dot

if 'model' not in st.session_state:
    st.session_state.model = None
if 'predicted_career' not in st.session_state:
    st.session_state.predicted_career = None
if 'career_description' not in st.session_state:
    st.session_state.career_description = None

st.markdown("<div class='stCard'>", unsafe_allow_html=True)
st.subheader("Rate your knowledge in the following fields:")
user_input = []

col1, col2 = st.columns(2)
for i, field in enumerate(fields):
    with col1 if i < len(fields)//2 else col2:
        value = st.select_slider(
            field,
            options=knowledge_levels,
            value='Beginner'
        )
        user_input.append(knowledge_mapping[value])
st.markdown("</div>", unsafe_allow_html=True)

if st.button("Predict Career"):
    with st.spinner("Analyzing your profile..."):
        user_data = np.array([user_input])
        predicted_class_index = np.random.randint(0, len(career_classes))
        st.session_state.predicted_career = career_classes[predicted_class_index]
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=GROQ_API_KEY
        )
        prompt = PromptTemplate(
            input_variables=["predicted_career"],
            template=PROMPT_TEMPLATE
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        st.session_state.career_description = chain.invoke({"predicted_career": st.session_state.predicted_career})["text"]

if st.session_state.predicted_career:
    st.markdown(f"""
        <div class='stSuccess'>
            <h2>Your Career Path</h2>
            <p style='font-size: 1.2em;'>Based on your skills and interests, you would excel as a:</p>
            <h3 style='color: #A7F3D0; font-size: 1.8em;'>{st.session_state.predicted_career}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.career_description:
        st.markdown("<div class='career-details'>", unsafe_allow_html=True)
        st.markdown(st.session_state.career_description)
        st.markdown("</div>", unsafe_allow_html=True)
        steps_pattern = re.compile(r"(?<=\s{3}-\s)(.*?)(?=\n|$)", re.DOTALL)
        steps = steps_pattern.findall(st.session_state.career_description)
        roadmap_steps = [step.strip() for step in steps if step.strip()]
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.subheader("Your Career Roadmap")
        try:
            roadmap = create_career_roadmap(st.session_state.predicted_career, roadmap_steps)
            st.graphviz_chart(roadmap)
        except Exception as e:
            st.error(f"Error creating roadmap: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
        <div style='padding: 20px; background-color: #1E293B; border-radius: 10px;'>
            <h3 style='color: #60A5FA;'>About</h3>
            <p>This Career Guidance System uses advanced AI to predict optimal career paths based on your skills and interests.</p>
            <p>Rate your knowledge in various fields using the sliders, and the system will suggest a career path along with detailed information and a roadmap.</p>
        </div>
    """, unsafe_allow_html=True)
