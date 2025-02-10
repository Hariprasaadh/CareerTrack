from click import prompt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time
GROQ_API_KEY=st.secrets['GROQ_API_KEY']

# Set page config
st.set_page_config(
    page_title="Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS with animations and glowing effects
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

    /* Main container styling */
    .main {
        font-family: 'Poppins', sans-serif;
        font-weight: bold;
        animation: fadeIn 0.5s ease-in;
    }

    .title {
        text-align: center;
        color: purple;
        font-weight: bold;

    }

    .stRadio > label {
        background: #f0f0f0;
        padding: 10px 15px;
        border-radius: 10px;
        transition: all 0.3s ease;
        color:purple;
    }

    .stRadio > label:hover {
        transform: translateY(-2px);
        box-shadow: 6px 6px 12px #d9d9d9, -6px -6px 12px #ffffff;
        color:blue;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 15px 0;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        animation: pulse 2s infinite;
    }

    # .stButton>button:hover {
    #     transform: translateY(-2px);
    #     box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
    # }

    /* Card styling */
    .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }

    

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_llm():
    """Initialize the Groq LLM"""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )


def create_prompt_template():
    """Create the prompt template for mental health assessment"""
    prompt = '''
    You are a mental health advisor. Your task is to assess the overall mental health of a person based on their responses to 14 questions and provide actionable suggestions to improve their well-being. Additionally, you will offer recommendations for five different aspects of mental health: emotional stability, stress management, social connections, productivity, and self-care.

    ---

    Survey Questions:

    1. Emotional Well-being:  
       How often do you feel emotionally stable and balanced?  
       {emotional_wellbeing}

    2. Stress Level:  
       How often do you feel overwhelmed or under pressure?  
       {stress_level}

    3. Sleep Quality:  
       How often do you experience restful sleep?  
       {sleep_quality}

    4. Energy Levels:  
       How often do you feel energetic and motivated throughout the day?  
       {energy_levels}

    5. Appetite Consistency:  
       How often is your appetite consistent and unaffected by emotions?  
       {appetite_consistency}

    6. Self-Worth:  
       How often do you feel confident and positive about yourself?  
       {self_worth}

    7. Focus and Productivity:  
       How often can you focus and complete tasks efficiently?  
       {focus_productivity}

    8. Social Interactions:  
       How often do you feel connected and supported by others?  
       {social_interactions}

    9. Negative Thoughts:  
       How often do you experience persistent negative thoughts?  
       {negative_thoughts}

    10. Peer Support:  
        Do you actively engage in support groups or social activities?  
        {peer_support}

    11. Journaling Habit:  
        Do you maintain a daily journaling habit?  
        {journaling_habit}

    12. Inspirational Content:  
        How often do you engage with inspirational or uplifting content?  
        {inspirational_content}

    13. Work-Life Balance:  
        How balanced do you feel between your work/studies and personal life?  
        {work_life_balance}

    14. Professional Help:  
        Have you sought professional help for mental health concerns?  
        {professional_help}

    ---

    Based on the responses, please provide:

    1. A detailed mental health assessment covering:
       - Emotional Stability (considering emotional_wellbeing, self_worth, negative_thoughts)
       - Stress Management (considering stress_level, work_life_balance)
       - Social Connections (considering social_interactions, peer_support)
       - Productivity (considering focus_productivity, energy_levels)
       - Self-Care (considering sleep_quality, appetite_consistency, journaling_habit)

    2. Specific recommendations for improvement in each area
    3. A set of actionable, daily practices to maintain mental well-being
    4. Any areas that might benefit from professional support

    Please format your response in markdown with clear headers and bullet points.
    '''

    return PromptTemplate(
        input_variables=[
            "emotional_wellbeing", "stress_level", "sleep_quality", "energy_levels",
            "appetite_consistency", "self_worth", "focus_productivity", "social_interactions",
            "negative_thoughts", "peer_support", "journaling_habit", "inspirational_content",
            "work_life_balance", "professional_help"
        ],
        template=prompt
    )


def create_radar_chart(responses):
    """Create a radar chart for numerical responses"""
    categories = [
        'Emotional Wellbeing', 'Stress Level', 'Sleep Quality',
        'Energy Levels', 'Appetite Consistency', 'Self Worth',
        'Focus & Productivity', 'Social Interactions', 'Negative Thoughts',
        'Work-Life Balance'
    ]

    values = [
        responses['emotional_wellbeing'], responses['stress_level'],
        responses['sleep_quality'], responses['energy_levels'],
        responses['appetite_consistency'], responses['self_worth'],
        responses['focus_productivity'], responses['social_interactions'],
        responses['negative_thoughts'], responses['work_life_balance']
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line_color='#4CAF50',
        fillcolor='rgba(76, 175, 80, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        showlegend=False,
        title="Mental Health Assessment Radar Chart",
        title_x=0.5
    )

    return fig


def create_yes_no_chart(responses):
    """Create a bar chart for yes/no responses"""
    yes_no_questions = {
        'Peer Support': responses['peer_support'],
        'Journaling Habit': responses['journaling_habit'],
        'Professional Help': responses['professional_help']
    }

    df = pd.DataFrame({
        'Question': list(yes_no_questions.keys()),
        'Response': [1 if v == 'Yes' else 0 for v in yes_no_questions.values()]
    })

    fig = px.bar(
        df,
        x='Question',
        y='Response',
        color='Response',
        color_continuous_scale=['#ff6b6b', '#4CAF50'],
        title="Yes/No Responses"
    )

    fig.update_layout(
        showlegend=False,
        yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No', 'Yes'])
    )

    return fig


def main():
    # Header with animation class
    st.markdown('<h1 class="title">🧠 Mental Health Assessment Tool</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; animation: fadeIn 1s ease-in;'>
    This tool provides a personalized mental health assessment based on your responses. 
    Please answer all questions honestly for the most accurate evaluation.
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    if 'progress' not in st.session_state:
        st.session_state.progress = 0

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Assessment Questions", "Visualization", "Results"])

    with tab1:
        st.header("Assessment Questions")

        # Numerical scale questions with radio buttons
        numerical_questions = {
            "emotional_wellbeing": "How often do you feel emotionally stable and balanced?",
            "stress_level": "How often do you feel overwhelmed or under pressure?",
            "sleep_quality": "How often do you experience restful sleep?",
            "energy_levels": "How often do you feel energetic and motivated throughout the day?",
            "appetite_consistency": "How often is your appetite consistent and unaffected by emotions?",
            "self_worth": "How often do you feel confident and positive about yourself?",
            "focus_productivity": "How often can you focus and complete tasks efficiently?",
            "social_interactions": "How often do you feel connected and supported by others?",
            "negative_thoughts": "How often do you experience persistent negative thoughts?",
            "work_life_balance": "How balanced do you feel between your work/studies and personal life?"
        }

        # Create columns for better layout
        col1, col2 = st.columns(2)

        # Distribute questions between columns with radio buttons
        for i, (key, question) in enumerate(numerical_questions.items()):
            with col1 if i < len(numerical_questions) // 2 else col2:
                st.session_state.responses[key] = st.radio(
                    f"{i + 1}. {question}",
                    options=[1, 2, 3, 4, 5],
                    horizontal=True,
                    help="1 = Never, 5 = Always"
                )

        # Yes/No questions with radio buttons
        st.subheader("Additional Questions")
        yes_no_questions = {
            "peer_support": "Do you actively engage in support groups or social activities?",
            "journaling_habit": "Do you maintain a daily journaling habit?",
            "professional_help": "Have you sought professional help for mental health concerns?"
        }

        cols = st.columns(3)
        for i, (key, question) in enumerate(yes_no_questions.items()):
            with cols[i]:
                st.session_state.responses[key] = st.radio(
                    question,
                    options=["Yes", "No"],
                    horizontal=True
                )

        # Inspirational content radio
        st.session_state.responses["inspirational_content"] = st.radio(
            "How often do you engage with inspirational or uplifting content?",
            options=["Low", "Medium", "High"],
            horizontal=True
        )

    with tab2:
        if st.session_state.responses:
            # Create and display visualizations
            st.subheader("Assessment Visualizations")

            # Display radar chart
            radar_chart = create_radar_chart(st.session_state.responses)
            st.plotly_chart(radar_chart, use_container_width=True)

            # Display yes/no responses chart
            yes_no_chart = create_yes_no_chart(st.session_state.responses)
            st.plotly_chart(yes_no_chart, use_container_width=True)

    with tab3:
        if st.button("Generate Assessment", key="generate"):
            try:
                # Create progress bar
                progress_bar = st.progress(0)

                # Simulate progress
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                with st.spinner("Generating your mental health assessment..."):
                    # Initialize LLM and chain
                    llm = initialize_llm()
                    chain = LLMChain(llm=llm, prompt=create_prompt_template())

                    # Generate assessment
                    output = chain.run(st.session_state.responses)

                    # Display results with animation
                    st.markdown(f"""
                    <div style='animation: fadeIn 1s ease-in;'>
                        {output}
                    </div>
                    """, unsafe_allow_html=True)

                    # Add download button
                    st.download_button(
                        label="📥 Download Report",
                        data=output,
                        file_name="mental_health_assessment.txt",
                        mime="text/plain"
                    )

                    # Success message
                    st.success("Assessment generated successfully!")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()