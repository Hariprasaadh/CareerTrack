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
    input_variables=["text", "style"],
    template="""
    You are a helpful assistant that can paraphrase texts. Please paraphrase the following text in a {style} style:

    {text}

    Paraphrase the text while maintaining its original meaning and context.
    """
)

chain = prompt | llm

def generate_output(text, style=None):
    response = chain.invoke({"text": text, "style": style})
    return response.content

def main():
    text = input("Enter the text to paraphrase: ")
    style = input("Enter the style of the paraphrase (optional): ")
    response = generate_output(text, style)
    print(response)

main()