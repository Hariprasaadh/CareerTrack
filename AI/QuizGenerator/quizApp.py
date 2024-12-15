import streamlit as st
import json
import os
import re
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
GROQ_API_KEY = os.getenv("GROQCLOUD_API")

llm = ChatGroq(
    model_name="llama-3.1-70b-versatile",
    temperature=0.3,
    groq_api_key=GROQ_API_KEY
)

def fetch_questions(text_content, quiz_level):
    RESPONSE_JSON = {
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
            },
            # Example other MCQs
        ]
    }

    prompt_ques = PromptTemplate.from_template(
        """
        Text: {text_content}
        You are an expert in generating MCQ type quiz on the basis of provided content to help students excel in their studies. 
        Given the above text, create a quiz of 5 multiple choice questions keeping difficulty level as {quiz_level}. 
        Make sure the questions are not repeated and check all the questions to be conforming the text as well.
        Make sure to format your response like RESPONSE_JSON below and use it as a guide.
        Return the JSON response only as double quotes not as single quotes. Keep the response in double quotes.
        Here is the RESPONSE_JSON: 
        {RESPONSE_JSON}   
        """
    )

    chain_ques = prompt_ques | llm

    response = chain_ques.invoke({
        "text_content": text_content,
        "quiz_level": quiz_level,
        "RESPONSE_JSON": RESPONSE_JSON
    })
    json_parser = JsonOutputParser()
    json_res = json_parser.parse(response.content)
    return json_res

def main():
    st.set_page_config(page_title="Quiz Generator", page_icon="🎓", layout="wide")

    # Title and Header Styling
    st.markdown(
        """
        <h1 style='text-align: center; color: white;'>🎓 Interactive Quiz Generator</h1>
        <h2 style='text-align: center; color: white;'>Generate MCQs for Better Learning!</h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        .stApp {
            color: white; 
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>label {
            font-size: 18px;
            color: white;
        }
        .stSelectbox>label {
            font-size: 18px;
            color: white;
        }
        .stRadio>label {
            font-size: 16px;
            color: white;
        }
        .stSubheader {
            font-size: 24px;
            color: white; /* White text for subheaders */
        }
        .stMarkdown {
            font-size: 18px;
            color: white; /* White text for markdown */
        }
        .stTextInput>div>div>input {
            color: white; /* Input text color */
        }
        .stSelectbox>div>div>input {
            color: white; /* Selectbox input text color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if 'quiz_data' not in st.session_state:
        st.session_state.quiz_data = None
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'text_content' not in st.session_state:
        st.session_state.text_content = ""

    text_content = st.text_area("Enter the content for quiz generation",
                                value=st.session_state.text_content,
                                height=200,
                                placeholder="Paste educational content here...")

    quiz_level = st.select_slider(
        "Select difficulty level",
        options=["Easy", "Medium", "Hard"],
        value="Medium",
        help="Choose the difficulty level for the generated questions."
    )

    def generate_quiz():
        st.session_state.quiz_data = fetch_questions(text_content, quiz_level)
        st.session_state.quiz_submitted = False
        st.session_state.user_answers = {}
        st.session_state.text_content = text_content

    # Button to generate quiz with a visual spinner
    if st.button("Generate Quiz"):
        if not text_content:
            st.warning("Please enter some content first.", icon="⚠️")
            return

        with st.spinner("Generating quiz... Please wait!"):
            generate_quiz()

    if st.session_state.quiz_data:
        st.subheader("Generated Quiz")
        for i, mcq in enumerate(st.session_state.quiz_data['mcqs'], 1):
            st.markdown(f"### Question {i}: {mcq['mcq']}")

            # Create unique key for each question
            answer_key = f"q_{i}"

            # Initialize the answer in session state if not exists
            if answer_key not in st.session_state.user_answers:
                st.session_state.user_answers[answer_key] = None

            # Radio button for options
            selected_answer = st.radio(
                "Choose your answer:",
                options=['a', 'b', 'c', 'd'],
                key=f"radio_{i}",
                index=None if st.session_state.user_answers[answer_key] is None
                else ['a', 'b', 'c', 'd'].index(st.session_state.user_answers[answer_key]),
                horizontal=True
            )

            # Update session state when answer changes
            if selected_answer is not None:
                st.session_state.user_answers[answer_key] = selected_answer

            # Display options in a more engaging way
            for opt in ['a', 'b', 'c', 'd']:
                st.write(f"**{opt.upper()})** {mcq['options'][opt]}")

            st.write("---")

        # Submit button with visual feedback
        if st.button("Submit Quiz", disabled=st.session_state.quiz_submitted):
            st.session_state.quiz_submitted = True
            score = 0
            st.subheader("Results")

            for i, mcq in enumerate(st.session_state.quiz_data['mcqs'], 1):
                answer_key = f"q_{i}"
                user_answer = st.session_state.user_answers.get(answer_key)
                correct_answer = mcq['correct']

                if user_answer == correct_answer:
                    score += 1
                    st.success(f"Question {i}: Correct! ✅", icon="✅")
                else:
                    st.error(f"Question {i}: Incorrect ❌ (Correct answer: {correct_answer})", icon="❌")

            # Display final score with positive reinforcement
            st.subheader(f"Final Score: {score}/5 ({score * 20}%)")
            if score == 5:
                st.balloons()
                st.success("Perfect score! Excellent work! 🎉", icon="🎉")
            elif score >= 3:
                st.success("Good job! Keep practicing! 👍", icon="👍")
            else:
                st.info("You might want to review the material again. Keep learning! 📚", icon="📚")

if __name__ == "__main__":
    main()
