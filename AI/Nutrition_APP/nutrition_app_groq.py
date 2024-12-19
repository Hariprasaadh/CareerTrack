import os
import base64

from dotenv import load_dotenv
from groq import Groq
from PIL import Image
import streamlit as st

# Load environment variables
load_dotenv()

# Get API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Input prompt for the nutrition analysis
input_prompt = """

Analyze the image and identify all food items present.
For each food item, estimate the approximate calorie content. 
Please respond in the following format (Should be in points):

1. Item 1: [Food Name] - [Calories]
2. Item 2: [Food Name] - [Calories]
...
...
Total Calories: XXX

Please note that if the image includes items such as packed foods, sugary drinks, or oily dishes, you should clearly state that these are not ideal for a healthier diet.

Finally, as an expert nutritionist, provide a summary comment on the food in the image. 
This comment should be  10 sentences long , focused on the overall nutritional quality, 
highlight areas for improvement, and suggest healthier alternatives if needed keeping in mind that this suggest is for students.

**Important:** Only include food items present in the image
"""


# Function to encode the image
def encode_image(image):
    return base64.b64encode(image).decode("utf-8")


# Initialize Groq client
client = Groq()

# Set up the Streamlit app
st.set_page_config(page_title="Nutrition Checker", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #555;
        margin-bottom: 30px;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🥗 Nutrition Checker App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle" style="color:white;">🍎 Analyze your food and get personalized nutrition tips! 🥦</div>',
    unsafe_allow_html=True,
)

# Upload section
st.markdown('<div class="section-header">📤 Upload Your Food Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)
else:
    st.info("📂 Please upload an image to proceed.")

# Analyze button
if st.button("🔍 Analyze Nutrition"):
    if uploaded_file:
        # Get the base64 encoded image
        image_bytes = uploaded_file.getvalue()
        base64_image = encode_image(image_bytes)

        try:
            # Create Groq completion
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": input_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                model="llama-3.2-11b-vision-preview",
            )
            st.success("✅ Nutrition analysis complete!")
            st.markdown('<div class="section-header">📋 Analysis Results</div>', unsafe_allow_html=True)
            st.write(chat_completion.choices[0].message.content)

            # Display nutrition tips
            st.markdown('<div class="section-header">💡 Nutrition Tips</div>', unsafe_allow_html=True)
            st.markdown(
                """
                - 🥤 **Stay hydrated:** Drink plenty of water throughout the day.
                - 🍎 **Include fruits and vegetables:** Aim for at least 5 servings a day.
                - 🌾 **Choose whole grains:** Opt for brown rice, whole wheat bread, and oatmeal.
                - 🍬 **Limit sugar and processed foods:** Keep snacks healthy and low in added sugar.
                - 🥗 **Balance your meals:** Include protein, carbs, and healthy fats in every meal.
                - 🏃 **Exercise regularly:** Incorporate at least 30 minutes of physical activity daily.
                """
            )
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
    else:
        st.error("🚨 Please upload an image to analyze.")
