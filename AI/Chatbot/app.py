import os
import streamlit as st
from dotenv import load_dotenv

import google.generativeai as genai

load_dotenv()

#Configure Streamlit Page Settings
st.set_page_config(
    page_title="CareerBuddy",
    page_icon="🧑‍💻",
    layout="centered"
)

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

#Set up Google Gemini AI Model
genai.configure(api_key=GOOGLE_API_KEY)
model=genai.GenerativeModel('gemini-2.0-flash-exp')

#Function to translate roles between Gemini and Streamlit Technology
# (Streamlit differentiates between these with the help of icons)
def translate_role(user_role):
    if user_role=="model":
        return "assistant"
    else:
        return user_role

#To maintain the Session State
#(Save the Session State)
if "chat_session" not in st.session_state:
    st.session_state.chat_session=model.start_chat(history=[])


st.title("CareerBuddy - Your personal guide to career excellence.")


#Display the Chat History
for message in st.session_state.chat_session.history:
    with st.chat_message(translate_role(message.role)):
        st.markdown(message.parts[0].text)


user_prompt=st.chat_input("Ask CareerBuddy")
if user_prompt:
    st.chat_message("user").markdown(user_prompt)

    #Send User message to Gemini and get the response
    ai_response=st.session_state.chat_session.send_message(user_prompt)

    with st.chat_message("assistant"):

        st.markdown(ai_response.text)