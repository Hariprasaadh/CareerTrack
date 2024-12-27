'''pip install langchain
pip install langchain-groq
pip install python-dotenv'''

from dotenv import load_dotenv
import os

load_dotenv(r"C:\Users\prann\OneDrive\เอกสาร\Projects\Essentials\API_KEYS\GROQ_API.env")

api_key = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key=api_key
)

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    You are a helpful assistant that can check grammar and style. Please analyze the following text for grammar and style issues and provide a corrected version:
    Keep the output of yours short and crisp.
    Highlight the error.
    Also along with the corrected grammar, also show alternative options.
    {text}
    """
)

chain = prompt | llm

def generate_output(text):
    response = chain.invoke({"text": text})
    return response.content

def main():
    text = input("Enter the text to check for grammar and style: ")
    response = generate_output(text)
    print(response)

main()