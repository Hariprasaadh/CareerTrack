from dotenv import load_dotenv
import os

load_dotenv(r"C:\Users\prann\OneDrive\เอกสาร\Projects\Essentials\API_KEYS\GROQ_API.env")

api_key = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq

llm = ChatGroq(
    model = "llama-3.1-70b-versatile",
    temperature = 0.5,
    groq_api_key = api_key
)

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=[
        "sender_name",
        "sender_details",
        "receiver_name",
        "receiver_details",
        "subject",
        "email_description",
        "additional_features",
        "word_limit",
        "style"
    ],
    template="""
    You are a highly skilled professional email writer. Your task is to compose a concise, impactful, and polished email based on the details provided below. The email must be tailored to the context, adhere to the word limit, and maintain a professional tone. The format of the email should change according to the specified style.

    1. *Sender Information:*
       - Name: {sender_name}
       - Details: {sender_details}

    2. *Recipient Information:*
       - Name: {receiver_name}
       - Details: {receiver_details}

    3. *Subject of the Email:*
       - {subject}

    4. *Purpose of the Email:*
       - {email_description}

    5. *Additional Requirements:*
       - {additional_features}

    6. *Word Limit:*
       - Ensure the email does not exceed {word_limit} words.

    7. *Style:*
       - The style of the email should be {style}.
       - If the style is formal: Use a professional tone, structured sentences, and a clear, concise format.
       - If the style is casual: Use a more relaxed tone with a friendly approach, but still keep it professional.
       - If the style is conversational: Use an approachable tone and language, maintaining politeness and clarity while being less formal.
       - If the style is persuasive: Focus on motivating the reader to take action, using a confident tone with a call to action.
       - If the style is informative: Focus on providing all the relevant details in a clear, direct manner without unnecessary fluff.

    *Important Guidelines for Writing the Email:*
    - *Tone:* Ensure a professional and respectful tone that matches the context (e.g., formal for business, slightly conversational for partnerships).
    - *Clarity:* Use clear and precise language to convey the key message effectively.
    - *Structure:* 
       - *Subject Line:* Craft a subject line that is engaging and reflects the email’s purpose.
       - *Opening Sentence:* Start with a warm and relevant greeting to set the tone.
       - *Body Paragraph(s):* Present the main message concisely, focusing on key points.
       - *Closing Statement:* Provide a thoughtful and clear call-to-action or next steps.
    - *Personalization:* Incorporate sender and recipient details meaningfully to add a personal touch.
    - *Word Economy:* Avoid redundant language and stick to the word limit while ensuring the email is impactful.

    Generate a final output that is error-free, persuasive, and context-aware, ensuring it achieves its intended purpose, while adapting to the specified style.
    """
)

chain = prompt | llm

def generate_email(inputs):
    try:
        if not inputs.get("sender_name") or not inputs.get("receiver_name"):
            raise ValueError("Both sender_name and receiver_name are required.")
        if not inputs.get("subject") or not inputs.get("email_description"):
            raise ValueError("Both subject and email_description are required.")
        
        output = chain.invoke(inputs)
        return output.content
    except Exception as e:
        print(f"Error: {e}")

def main():
    sender_name = input("Enter sender's name: ")
    receiver_name = input("Enter receiver's name: ")
    subject = input("Enter the subject of the email: ")
    email_description = input("Enter the brief descption of the email: ")

    sender_details = input("Enter sender's details (optional): ") or None
    receiver_details = input("Enter receiver's details (optional): ") or None
    additional_features = input("Enter additional requirements (optional): ") or None
    word_limit = input("Enter the word limit for the email (optional): ") or None
    style = input("Enter the style of the email (e.g., formal, casual, etc.): ")

    inputs = {
        "sender_name": sender_name,
        "sender_details": sender_details,
        "receiver_name": receiver_name,
        "receiver_details": receiver_details,
        "subject": subject,
        "email_description": email_description,
        "additional_features": additional_features,
        "word_limit": word_limit,
        "style": style
    }

    response = generate_email(inputs)
    print(response)

if __name__ == "__main__":
    main()
