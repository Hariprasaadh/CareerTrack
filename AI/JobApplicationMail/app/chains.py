import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
load_dotenv()

class Chain():
    def __init__ (self):
        self.llm=ChatGroq(model_name="llama-3.1-70b-versatile",temperature=0,groq_api_key=os.getenv("GROQCLOUD_API"))
    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]
    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
         ### JOB DESCRIPTION:
    {job_description}
    
    ### INSTRUCTION:
    You are K. Hariprasaadh, a highly motivated and dedicated candidate pursuing B.Tech in CSE with a specialization in AI and ML at VIT University, Chennai. 
    With an exceptional academic record (CGPA: 9.61) and a proven track record of success in competitive programming, machine learning, and computer vision, 
    you possess a versatile technical skill set that includes Python, C++, and Java. Your accomplishments include winning multiple hackathons and delivering 
    impactful projects such as a crop management web application and innovative educational technology tools.
    Introduce yourself properly at the start who you are and where you are from.
    Your task is to craft a professional and persuasive job application email expressing your genuine enthusiasm for joining Nike. 
    Highlight how your technical expertise, project experience, and achievements align with Nike’s mission, vision, and the specific role requirements. 

    Provide links to your professional portfolio, GitHub, LinkedIn, or other relevant platforms that showcase your work, achievements, and technical expertise. 
    Customize the tone of the email to reflect Nike's values of innovation, performance, and inclusivity. 
    Also add the most relevant ones from the following links to showcase portfolio and add my profile link also: {link_list}
    Finally add the my profile link to showcase my work and contact from {link_list}.
    Make it sound very professional and it should be very formal.
    ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content
    
    