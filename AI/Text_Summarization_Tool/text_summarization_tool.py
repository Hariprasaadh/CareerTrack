'''pip install langchain
pip install langchain-groq
pip install python-dotenv
'''

from dotenv import load_dotenv
import os

load_dotenv("/content/API.env")

api_key = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq

llm = ChatGroq(
    model = "llama-3.1-70b-versatile",
    temperature = 0,
    groq_api_key = api_key
)

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["text", "no_of_words", "style"],
    template="""
    You are a helpful assistant that can summarize texts. Please summarize the following text in a {style} style:

    {text}

    If the 'no_of_words' variable is provided, summarize the text in at most {no_of_words} words.
    Otherwise, summarize the text in a concise and clear manner, capturing the main points.
    """
)

from langchain.chains import LLMChain

chain = LLMChain(
    llm = llm,
    prompt = prompt
)

def generate_output(text, no_of_words=None, style=None):
  response = chain.invoke({"text": text, "no_of_words": no_of_words, "style": style})
  return response["text"]

def main():

  text = input("Enter the text to summarize: ")
  no_of_words = int(input("Enter the number of words in the summary (optional): "))
  style = input("Enter the style of the summary (optional): ")

  response = generate_output(text, no_of_words, style)