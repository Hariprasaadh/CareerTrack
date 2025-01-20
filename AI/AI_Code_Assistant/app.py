import os
import logging
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

def setup_logging():
    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s - %(levelname)s - %(message)s", 
                        handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])

def validate_input(value, prompt, required=True, options=None):
    if required and not value:
        st.warning(prompt)
        return None
    if options and value not in options:
        st.warning(f"Invalid input. Choose from {options}.")
        return None
    return value

def process_code_task(programming_language, task_type, code=None, description=None, output_format=None):
    if not programming_language or not task_type:
        raise ValueError("Programming language and task type are required")

    if task_type == "Generate Code" and not code and not description:
        raise ValueError("For Generate Code task, either code or description must be provided")

    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0.5,
        groq_api_key= "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",
    )

    prompt = PromptTemplate(
        input_variables=["programming_language", "code", "description", "output_format", "task_type"],
        template="""
        You are a highly skilled programming expert. Based on the provided details, your task is to perform the requested action in {programming_language}:

        1. *Programming Language:*
           - {programming_language}

        2. *Task Type:*
           - {task_type}

        3. *Code (if applicable):*
           - {code}

        4. *User Request (Description):*
           - {description}

        5. *Desired Output Format:*
           - {output_format}

        *Guidelines:*
        - If the task type is *"Generate Code"*:
          - Generate a code snippet based on the description provided by the user. 
          - Ensure that the generated code follows best practices, is functional, and meets the described requirement.

        - If the task type is *"Explain Code"*:
          - Provide an explanation for the given code, breaking it down into its components and functionality.
          
        - If the task type is *"Debug Code"*:
          - Identify issues in the provided code, explain the problems, and suggest fixes.

        - If the task type is *"Optimize Code"*:
          - Suggest improvements for the provided code in terms of performance, readability, or efficiency.

        - Tailor the output based on the desired format:
          - Plain Text: Provide a continuous explanation or code in natural language.
          - Step-by-Step: Break down the task or explanation in a sequential manner.
          - Bullet Points: Summarize the key points concisely.

        *Important:*
        - If the user is generating code and the code field is empty, generate the code based on the description.
        - Ensure that the output is relevant, clear, and meets the user's requirements.
        """
    )

    chain = prompt | llm

    result = chain.invoke({
        "programming_language": programming_language,
        "code": code if code else "No code provided",
        "description": description if description else "No description provided",
        "output_format": output_format if output_format else "Plain Text",
        "task_type": task_type
    })

    return result.content

def main():
    setup_logging()
    st.title("Programming Assistant with LLM Integration")
    st.write("Provide the details of your programming task, and the assistant will generate, explain, debug, or optimize code for you.")

    programming_language = st.text_input("Enter the programming language:")
    task_type = st.selectbox(
        "Select Task Type:",
        ["Generate Code", "Explain Code", "Debug Code", "Optimize Code"]
    )
    code = st.text_area("Enter code (optional):", height=200)
    description = st.text_area("Describe the task (optional):", height=200)
    output_format = st.selectbox(
        "Select Output Format:",
        ["Plain Text", "Step-by-Step", "Bullet Points"]
    )

    if st.button("Process Task"):
        try:
            programming_language = validate_input(
                programming_language, "Programming language is required.", required=True
            )
            task_type = validate_input(
                task_type, "Task type is required.", required=True
            )
            output_format = validate_input(
                output_format, "Output format is required.", required=True
            )

            if programming_language and task_type and output_format:
                result = process_code_task(
                    programming_language, task_type, code or None, description or None, output_format
                )
                st.success("Task Processed Successfully!")
                st.text_area("Result:", result, height=400)

        except ValueError as e:
            st.error(f"Input Error: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            logging.exception("Unexpected Error")

if __name__ == "__main__":
    main()
