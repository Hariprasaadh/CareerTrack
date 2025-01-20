from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.prompts import PromptTemplate
import streamlit as st

# Load environment variables
load_dotenv()

# Define the prompt template
prompt = PromptTemplate.from_template(
    """
    INPUT:
    # Transcript: {transcript}

    ### Enhanced YouTube Video Summariser and Analysis for Academic and Career Growth
    - Extract the technical keywords from the transcript provided and summarise only that. Don't say what speaker is doing.
    - Don't include the word "Speaker" (IMPORTANT)
    - Dont include the word "Transcript" (IMPORTANT)
    - Dont make summary very small. Break down the content into points and summarise it. Highlight each point. Size of summarised content should 25 percent of original content
    - Only summarise the technical contents add some more points to it if needed only for technical concepts.
    - Dont add any unrelated content.
    - Accurately summarise the spoken content of the given YouTube video, preserving context, technical terms, and key points.
    - Summarize the transcription into concise sections, highlighting critical takeaways.

    OUTPUT STRUCTURE:
    ##Highlight these Headings properly. Clearly distinguish between heading and its content
    #Topic  (content should not be bold)
    # Keywords (3-5 Keywords only)
    #Summary (in points)  (30% of video content)  
    #Key Takeaways (3-5 points)
    ##Next Steps
    """
)

# Function to extract transcript from the YouTube link
def extract_transcript(video_url):
    video_id = video_url.split("v=")[1].split("&")[0]
    transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = " "
    for i in transcript_text:
        transcript += " " + i["text"]
    return transcript

# Function to summarize the transcript using the LLM
def summarise(transcript, prompt):
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.5,
        api_key=os.getenv("GROQCLOUD_API")
    )
    model = prompt | llm
    res = model.invoke(input={'transcript': transcript})
    return res.content

# Streamlit UI
st.set_page_config(page_title="YouTube Transcript to Detailed Notes", layout="wide")
st.title("📹 YouTube in a Nutshell: Quick Summaries Made Easy!")

# Sidebar with instructions
st.sidebar.header("📋 **Instructions**")
st.sidebar.markdown("""
1. 🌐 **Enter a YouTube Video URL** in the text box below to start.
2. 📑 **Press 'Get Detailed Notes'** to extract and summarize the technical content.
3. 💡 **View the summary** with key takeaways, technical insights, and more.
4. 🔍 **Refine** your learning and enhance your understanding!
""")

# Input for YouTube video link
youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    video_id = youtube_link.split("v=")[1].split("&")[0]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=False, width=500)
# Button to trigger the detailed notes extraction
if st.button("Get Detailed Notes"):
    with st.spinner("Extracting and summarizing transcript..."):
        try:
            transcript_text = extract_transcript(youtube_link)

            if transcript_text:
                summary = summarise(transcript_text, prompt)
                st.markdown("## Detailed Notes:")
                st.write(summary)
            else:
                st.error("No transcript available for this video. Please try another.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Add some styling after the process
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

        .stApp {
            background: linear-gradient(120deg, #4e21cc 25%, #367b9c 50%, #8231d4 75%);
            background-size: 200% 100%;
            font-family: 'Poppins', sans-serif;
        }

        div.stButton > button {
            background: linear-gradient(90deg, #ff7eb3, #ff758c, #ff5964);
            border: none;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
        }

        div.stButton > button:hover {
            transform: scale(1.1);
            color:white;
            background: linear-gradient(90deg, #ff5964, #ff758c, #ff7eb3);
            box-shadow: 0px 5px 15px rgba(255, 89, 100, 0.5);
        }

        .stSidebar {
            background-color: #003300;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 10px;
            width: 250px;
            height: 200px;
        }

        .stSidebar .stMarkdown {
            font-size: 12px;
            color: #fff;
        }

    </style>
""", unsafe_allow_html=True)