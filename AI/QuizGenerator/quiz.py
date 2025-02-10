import streamlit as st
import os
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
GROQ_API_KEY=st.secrets['GROQ_API_KEY']

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.5,
    groq_api_key=GROQ_API_KEY
)


def fetch_questions(text_content, quiz_level, num_questions):
    RESPONSE_JSON_CONTENT = {
        "mcqs": [
            {
                "mcq": "multiple choice question1",
                "options": {
                    "a": "choice here1",
                    "b": "choice here2",
                    "c": "choice here3",
                    "d": "choice here4",
                },
                "correct": "correct choice option in the form of a, b, c or d",
            }
        ]
    }

    RESPONSE_JSON_PYQ = {
        "mcqs": [
            {
                "mcq": "multiple choice question1 (GATE CSE 2024)",
                "options": {
                    "a": "choice here1",
                    "b": "choice here2",
                    "c": "choice here3",
                    "d": "choice here4",
                },
                "correct": "correct choice option in the form of a, b, c or d",
            }
        ]
    }

    if len(text_content) > 100:
        RESPONSE_JSON = RESPONSE_JSON_CONTENT
        prompt_ques = PromptTemplate.from_template(
            """
            Text: {text_content}
            You are an expert in generating MCQ type quiz on the basis of provided content to help students excel in their studies. 
            Given the above text, create a quiz of {num_questions} multiple choice questions keeping difficulty level as {quiz_level}. 
            Make sure the questions are not repeated and check all the questions to be conforming the text as well.
            Make sure to format your response like RESPONSE_JSON below and use it as a guide.
            Return the JSON response only as double quotes not as single quotes.
            Here is the RESPONSE_JSON: 
            {RESPONSE_JSON}   
            """
        )
    else:
        RESPONSE_JSON = RESPONSE_JSON_PYQ
        prompt_ques = PromptTemplate.from_template(
            """
            You are an expert in helping students prepare for the GATE exam by providing high-quality previous year questions based on the topic provided in the {text_content} variable.
            Your task is to create a quiz consisting of {num_questions} unique multiple-choice questions derived from GATE previous year questions, ensuring the following:
                The difficulty level of the questions must match the {quiz_level} specified.
                Include the year of the GATE question in brackets at the end of each question (e.g., [GATE 2022]).
                Ensure all questions align closely with the topic mentioned in {text_content}.
                Verify that questions are not repeated and conform to the topic and difficulty level specified.

            Make sure to format your response like RESPONSE_JSON below and use it as a guide.
            Return the JSON response only as double quotes not as single quotes.
            Here is the RESPONSE_JSON: 
            {RESPONSE_JSON}

            Instructions:
            Provide only the JSON response, ensuring it uses double quotes for all keys and values.
            Do not include any additional explanations or preambles in your output.
            """
        )

    chain_ques = prompt_ques | llm
    response = chain_ques.invoke({
        "text_content": text_content,
        "quiz_level": quiz_level,
        "RESPONSE_JSON": RESPONSE_JSON,
        "num_questions": num_questions
    })

    json_parser = JsonOutputParser()
    json_res = json_parser.parse(response.content)
    return json_res


def create_performance_charts(score, num_questions):
    # Donut chart for score percentage
    fig_donut = go.Figure(data=[go.Pie(
        labels=['Correct', 'Incorrect'],
        values=[score, num_questions - score],
        hole=.7,
        marker_colors=['#4CAF50', '#ff6b6b']
    )])

    fig_donut.update_layout(
        showlegend=False,
        annotations=[dict(text=f'{int((score / num_questions) * 100)}%', x=0.5, y=0.5, font_size=40, showarrow=False)],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=0, b=0, l=0, r=0),
        height=300
    )

    # Bar chart for question-wise analysis
    categories = [f'Q{i + 1}' for i in range(num_questions)]
    values = [1 if i < score else 0 for i in range(num_questions)]

    fig_bar = go.Figure(data=[go.Bar(
        x=categories,
        y=values,
        marker_color=['#4CAF50' if v == 1 else '#ff6b6b' for v in values]
    )])

    fig_bar.update_layout(
        title="Question-wise Performance",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=300,
        margin=dict(t=50, b=50, l=50, r=50)
    )

    return fig_donut, fig_bar


def main():
    st.set_page_config(page_title="Quiz Generator", page_icon="🎓", layout="wide")

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');


        .stApp {
                background: rgb(0,71,171);
                background: linear-gradient(159deg, rgba(0,71,171,1) 0%, rgba(28,169,201,1) 100%);
                #background-image: linear-gradient(to top, #00c6fb 0%, #005bea 100%);
                #background-image: linear-gradient(to top, #30cfd0 0%, #330867 100%);            

                font-family: 'Poppins', sans-serif;
        }

        .stTextArea label {
            font-size: 28px !important;
            font-weight: 600 !important;
            color: #FFFFFF !important;
            margin-bottom: 15px !important;
            font-family: 'Poppins', sans-serif !important;
            text-transform: uppercase !important;  /* Make it all caps */
            letter-spacing: 1px !important;  /* Space out the letters slightly */
            background: linear-gradient(90deg, #FFFFFF, #E0E0E0) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }   

        .main-header {
            background: rgba(255, 255, 255, 0.1);
            padding: 2rem;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }



        .question-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }

        .question-card:hover {
            transform: translateY(-5px);
        }


         .stMarkdown p {
            font-size: 26px !important;
            color: white !important;
            margin-top: 20px !important;
        }
         .stRadio > div {
            gap: 2rem !important;
        }

        .stRadio > div > label {
            font-size: 22px !important;
            color: white !important;
        }

        .question-card div p {
            font-size: 24px !important;
            margin: 15px 0 !important;
            color: white !important;
        }

        .stButton>button {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            border: none;
            padding: 0.8rem 1.5rem;
            color: white;
            font-weight: 600;
            border-radius: 25px;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 1rem;
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            color: #ebe70e;
            box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
        }

        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 18px;
            border-radius: 10px;
        }

        .stTextArea label, .stSelectbox label {
            font-size: 28px !important;
            font-weight: 600 !important;
            color: #FFFFFF !important;
            margin-bottom: 15px !important;
            font-family: 'Poppins', sans-serif !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            background: linear-gradient(90deg, #FFFFFF, #E0E0E0) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }


        .stSelectbox>div>div>div {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }

        .stRadio>label {
            color: #E0E0E0;
            font-size: 1rem;
        }

        .results-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 2rem;
            border-radius: 15px;
            margin-top: 2rem;
            backdrop-filter: blur(10px);
        }

        .correct-answer {
            color: #00ff00 !important;  /* Brighter green */
            font-weight: 700 !important;
            font-size: 24px !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5) !important;
            background: rgba(0, 128, 0, 0.3) !important;  /* Darker green background */
            padding: 10px 15px !important;
            border-radius: 5px !important;
            background-color: white;


            display: block !important;
        }

        .incorrect-answer {
            color: #ff0066 !important;  /* Brighter red */
            background-color: #f0c9c2;
            font-weight: 700 !important;
            font-size: 24px !important;
            padding: 10px 15px !important;
            border-radius: 5px !important;
            display: block !important;
        }

        .results-card {
            background: rgba(0, 0, 0, 0.4) !important;  /* Much darker background */
            padding: 2rem !important;
            border-radius: 15px !important;
            margin-top: 2rem !important;
            backdrop-filter: blur(10px) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        }

        .stInfo{
            font-size: 42px !important;
            font-weight: 800 !important;
            padding: 20px !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
        }


        .stMarkdown {
            color: #E0E0E0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='main-header'>
            <h1 style='text-align: center; color: white;'>🎓 AI-Powered Interactive Quiz Generator</h1>
            <h3 style='text-align: center; color: #f5edeb;'>Elevate Your Learning Experience</h3>
        </div>
    """, unsafe_allow_html=True)

    if 'quiz_data' not in st.session_state:
        st.session_state.quiz_data = None
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'text_content' not in st.session_state:
        st.session_state.text_content = ""

    col1, col2 = st.columns([2, 1])

    with col1:
        text_content = st.text_area(
            "Enter Content or Topic",
            value=st.session_state.text_content,
            height=200,
            placeholder="Paste educational content or enter a topic/subject name (like Operating Systems) for GATE PYQs..."
        )

    with col2:
        num_questions = st.number_input(
            "Number of Questions",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )

        quiz_level = st.selectbox(
            "Difficulty Level",
            options=["Easy", "Medium", "Hard"],
            index=1
        )

        if st.button("Generate Quiz 🎯"):
            if not text_content:
                st.warning("Please enter some content first.", icon="⚠️")
                return

            with st.spinner("Creating your quiz... 🤖"):
                st.session_state.quiz_data = fetch_questions(text_content, quiz_level, num_questions)
                st.session_state.quiz_submitted = False
                st.session_state.user_answers = {}
                st.session_state.text_content = text_content

    if st.session_state.quiz_data:
        for i, mcq in enumerate(st.session_state.quiz_data['mcqs'], 1):
            st.markdown(f"""
                <div class='question-card'>
                    <h3>Question {i}: {mcq['mcq']}</h3>
                </div>
            """, unsafe_allow_html=True)

            answer_key = f"q_{i}"
            if answer_key not in st.session_state.user_answers:
                st.session_state.user_answers[answer_key] = None

            selected_answer = st.radio(
                "Choose your answer:",
                options=['a', 'b', 'c', 'd'],
                key=f"radio_{i}",
                index=None if st.session_state.user_answers[answer_key] is None
                else ['a', 'b', 'c', 'd'].index(st.session_state.user_answers[answer_key]),
                horizontal=True
            )

            if selected_answer is not None:
                st.session_state.user_answers[answer_key] = selected_answer

            for opt in ['a', 'b', 'c', 'd']:
                st.markdown(f"**{opt.upper()})** {mcq['options'][opt]}")

            st.markdown("---")

        if not st.session_state.quiz_submitted:
            if st.button("Submit Quiz ✨", disabled=st.session_state.quiz_submitted):
                st.session_state.quiz_submitted = True

        if st.session_state.quiz_submitted:
            st.markdown("<div class='results-card'>", unsafe_allow_html=True)
            score = sum(1 for i, mcq in enumerate(st.session_state.quiz_data['mcqs'], 1)
                        if st.session_state.user_answers.get(f"q_{i}") == mcq['correct'])

            fig_donut, fig_bar = create_performance_charts(score, num_questions)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_donut, use_container_width=True)
            with col2:
                st.plotly_chart(fig_bar, use_container_width=True)

            for i, mcq in enumerate(st.session_state.quiz_data['mcqs'], 1):
                user_answer = st.session_state.user_answers.get(f"q_{i}")
                correct_answer = mcq['correct']

                if user_answer == correct_answer:
                    st.markdown(f"""
                        <div style='padding: 10px; background: rgba(76, 175, 80, 0.1); border-radius: 5px; margin: 5px 0;'>
                            <span class='correct-answer'>✅ Question {i}: Correct!</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style='padding: 10px; background: rgba(255, 107, 107, 0.1); border-radius: 5px; margin: 5px 0;'>
                            <span class='incorrect-answer'>❌ Question {i}: Incorrect (Correct answer: {correct_answer})</span>
                        </div>
                    """, unsafe_allow_html=True)

            final_score = int((score / num_questions) * 100)
            if final_score >= 80:
                st.balloons()
                st.success(f"🎉 Excellent! Your score: {final_score}%")
            elif final_score >= 60:
                st.success(f"👏 Good job! Your score: {final_score}%")
            else:
                st.info(f"📚 Keep practicing! Your score: {final_score}%")

            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()