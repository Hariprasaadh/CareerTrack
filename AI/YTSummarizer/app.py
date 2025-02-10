import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY=st.secrets['GROQ_API_KEY']
GOOGLE_API_KEY=st.secrets['GEMINI_API_KEY']

genai.configure(api_key=GOOGLE_API_KEY)

summary_prompt = PromptTemplate.from_template(
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


def extract_transcript(video_url):
    try:
        video_id = video_url.split("v=")[1].split("&")[0]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join(item["text"] for item in transcript_text)
        return transcript
    except Exception as e:
        st.error(f"Error extracting transcript: {str(e)}")
        return None


def summarise(transcript):
    try:
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.5,
            api_key=GROQ_API_KEY
        )
        model = summary_prompt | llm
        response = model.invoke(input={'transcript': transcript})
        return response.content
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None


def create_mindmap_markdown(text):
    try:
        model = genai.GenerativeModel('gemini-pro')

        max_chars = 30000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            st.warning(f"Text was truncated to {max_chars} characters due to length limitations.")

        prompt = """
        Create a hierarchical markdown mindmap from the following text. 
        Use proper markdown heading syntax (# for main topics, ## for subtopics, ### for details).
        Focus on the main concepts and their relationships.
        Include relevant details and connections between ideas.
        Keep the structure clean and organized.

        Format the output exactly like this example:
        # Main Topic
        ## Subtopic 1
        ### Detail 1
        - Key point 1
        - Key point 2
        ### Detail 2
        ## Subtopic 2
        ### Detail 3
        ### Detail 4

        Text to analyze: {text}

        Respond only with the markdown mindmap, no additional text.
        """

        response = model.generate_content(prompt.format(text=text))
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating mindmap: {str(e)}")
        return None


def create_markmap_html(markdown_content):
    if not markdown_content:
        return None

    markdown_content = json.dumps(markdown_content)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            #mindmap {{
                width: 100%;
                height: 700px;
                background: #e1f0ee;
                border-radius: 15px;
                padding: 20px;
            }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/d3@6"></script>
        <script src="https://cdn.jsdelivr.net/npm/markmap-view"></script>
        <script src="https://cdn.jsdelivr.net/npm/markmap-lib@0.14.3/dist/browser/index.min.js"></script>
    </head>
    <body>
        <svg id="mindmap"></svg>
        <script>
            window.onload = async () => {{
                try {{
                    const markdown = {markdown_content};
                    const transformer = new markmap.Transformer();
                    const {{root}} = transformer.transform(markdown);
                    const mm = new markmap.Markmap(document.querySelector('#mindmap'), {{
                        maxWidth: 500,
                        color: (node) => {{
                            const colors = [
                                '#00ff87', '#00bfff', '#ff3366', '#ff6b6b', 
                                '#4facfe', '#00f2fe', '#cd9cf2', '#6713d2'
                            ];
                            return colors[node.depth % colors.length];
                        }},
                        paddingX: 20,
                        autoFit: true,
                        initialExpandLevel: 3,
                        duration: 750,
                        spacingHorizontal: 80,
                        spacingVertical: 30,
                        linkShape: 'diagonal',
                        linkWidth: (node) => 2 + (4 - node.depth) * 0.5,
                        nodeMinHeight: 16,
                    }});
                    mm.setData(root);
                    mm.fit();
                }} catch (error) {{
                    console.error('Error rendering mindmap:', error);
                    document.body.innerHTML = '<p style="color: red;">Error rendering mindmap. Please check the console for details.</p>';
                }}
            }};
        </script>
    </body>
    </html>
    """
    return html_content


def analyze_transcript_data(transcript_list):
    df = pd.DataFrame(transcript_list)

    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

    df['start_time'] = df['start'].apply(lambda x: str(timedelta(seconds=int(x))))
    df['duration'] = df['duration'].round(2)

    df['words_per_minute'] = (df['word_count'] / df['duration']) * 60

    return df


def create_visualizations(df):
    fig_timeline = px.line(
        df,
        x='start_time',
        y='word_count',
        title='Word Count Distribution Over Time',
        labels={'start_time': 'Video Timeline', 'word_count': 'Number of Words'}
    )
    fig_timeline.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, l=50, r=30)
    )

    fig_wpm = px.line(
        df,
        x='start_time',
        y='words_per_minute',
        title='Speaking Speed (Words per Minute) Over Time',
        labels={'start_time': 'Video Timeline', 'words_per_minute': 'Words per Minute'}
    )
    fig_wpm.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, l=50, r=30)
    )

    fig_distribution = px.histogram(
        df,
        x='word_count',
        title='Distribution of Words per Segment',
        labels={'word_count': 'Words per Segment', 'count': 'Frequency'}
    )
    fig_distribution.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, l=50, r=30)
    )

    return fig_timeline, fig_wpm, fig_distribution


def main():
    st.set_page_config(page_title="YouTube Video Analyzer", layout="wide")

    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

            .stApp {
                background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
                font-family: 'Poppins', sans-serif;
            }

            .stButton > button {
                background: linear-gradient(45deg, #FF512F 0%, #F09819 100%);
                border: none;
                color: white;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 10px;
                transition: all 0.3s ease;
            }

            .stButton > button:hover {
                box-shadow: 0 5px 20px rgba(0,0,0,0.3);
                background: linear-gradient(45deg, #F09819 0%, #FF512F 100%);
            }

            h1, h2, h3 {
                color: white;
                font-weight: 700;
            }

            .stSidebar {
                background: rgba(0, 0, 0, 0.3);
                border-right: 1px solid rgba(255, 255, 255, 0.1);
            }

            .stSidebar .stMarkdown {
                color: white;
            }

            .content-box {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                padding: 20px;
                margin: 10px 0;
            }

            .metric-container {
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎥 YouTube Video Analyzer")

    with st.sidebar:
        st.header("📋 How to Use")
        st.markdown("""
        ### 🎯 Quick Start
        1. 📝 Copy any YouTube video URL
        2. 🔗 Paste it in the input box
        3. 🚀 Click 'Analyze Video'

        ### 📊 Available Features
        * 📑 **Summary**: Get concise video content
        * 🧠 **Mind Map**: Visual concept representation
        * 📈 **Analytics**: View detailed metrics

        """)

    youtube_link = st.text_input("🔗 Enter YouTube Video Link:",
                                 placeholder="https://youtube.com/watch?v=...")

    if youtube_link:
        try:
            video_id = youtube_link.split("v=")[1].split("&")[0]
            st.image(f"http://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
                     use_container_width=True)
        except:
            st.error("Invalid YouTube URL")

    if st.button("🎯 Analyze Video"):
        with st.spinner("Processing video..."):
            try:
                video_id = youtube_link.split("v=")[1].split("&")[0]
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

                transcript_text = " ".join(item["text"] for item in transcript_list)

                if transcript_text:
                    summary = summarise(transcript_text)

                    if summary:
                        tab1, tab2, tab3 = st.tabs(["📝 Summary", "🧠 Mind Map", "📊 Analytics"])

                        with tab1:
                            st.markdown(summary)

                        with tab2:
                            mindmap_markdown = create_mindmap_markdown(summary)
                            if mindmap_markdown:
                                html_content = create_markmap_html(mindmap_markdown)
                                if html_content:
                                    st.components.v1.html(html_content, height=800)

                        with tab3:
                            df = analyze_transcript_data(transcript_list)
                            fig_timeline, fig_wpm, fig_distribution = create_visualizations(df)

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.markdown("""
                                    <div class="metric-container">
                                        <h3>Total Words</h3>
                                        <h2>{}</h2>
                                    </div>
                                """.format(df['word_count'].sum()), unsafe_allow_html=True)

                            with col2:
                                avg_wpm = df['words_per_minute'].mean()
                                st.markdown("""
                                    <div class="metric-container">
                                        <h3>Average Speed</h3>
                                        <h2>{:.1f} WPM</h2>
                                    </div>
                                """.format(avg_wpm), unsafe_allow_html=True)

                            with col3:
                                duration = str(timedelta(seconds=int(df['start'].iloc[-1] + df['duration'].iloc[-1])))
                                st.markdown("""
                                    <div class="metric-container">
                                        <h3>Duration</h3>
                                        <h2>{}</h2>
                                    </div>
                                """.format(duration), unsafe_allow_html=True)

                            st.plotly_chart(fig_timeline, use_container_width=True)
                            st.plotly_chart(fig_wpm, use_container_width=True)
                            st.plotly_chart(fig_distribution, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing video: {str(e)}")


if __name__ == "__main__":
    main()
