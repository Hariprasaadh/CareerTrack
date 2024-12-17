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
    #Summary (in points)
    #Additional Points (3-5 points)
    #Key Takeaways (3-5 points)
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
st.title("📹 YouTube Transcript to Detailed Notes Converter")

# Sidebar with instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. **Enter a YouTube Video URL** in the text box below.
2. **Press 'Get Detailed Notes'** to extract and summarize the technical content.
3. **View the summary** with key takeaways and technical insights.
""")

# Input for YouTube video link
youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    video_id = youtube_link.split("v=")[1].split("&")[0]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

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

# Add some styling
st.markdown("""
<style>
    .css-ffhzg2 {
        background-color: #f7f7f7;
    }
    .css-1v3fvcr {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #00A300;
        color: white;
        font-size: 18px;
        padding: 12px 24px;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #007500;
    }
    .stTextInput>div>div>input {
        font-size: 16px;
        padding: 12px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)
