import requests
from bs4 import BeautifulSoup
import pandas as pd
import re



job_name = input("Enter job name: ")

job_name_split = job_name.split()

job_name_title = job_name.title()

if(len(job_name_split) == 1):
    url = f'https://www.timesjobs.com/candidate/job-search.html?searchType=personalizedSearch&from=submit&searchTextSrc=&searchTextText="{job_name_title}"&txtKeywords={job_name_split[0]}&txtLocation='

elif(len(job_name_split) == 2):
    url = f'https://www.timesjobs.com/candidate/job-search.html?searchType=personalizedSearch&from=submit&searchTextSrc=&searchTextText="{job_name_title}"&txtKeywords={job_name_split[0]}+{job_name_split[1]}&txtLocation='

elif(len(job_name_split) == 3):
    url = f'https://www.timesjobs.com/candidate/job-search.html?searchType=personalizedSearch&from=submit&searchTextSrc=&searchTextText="{job_name_title}"&txtKeywords={job_name_split[0]}+{job_name_split[1]}+{job_name_split[2]}&txtLocation='

elif(len(job_name_split) == 4):
    url = f'https://www.timesjobs.com/candidate/job-search.html?searchType=personalizedSearch&from=submit&searchTextSrc=&searchTextText="{job_name_title}"&txtKeywords={job_name_split[0]}+{job_name_split[1]}+{job_name_split[2]}+{job_name_split[3]}&txtLocation='

elif(len(job_name_split) == 4):
    url = f'https://www.timesjobs.com/candidate/job-search.html?searchType=personalizedSearch&from=submit&searchTextSrc=&searchTextText="{job_name_title}"&txtKeywords={job_name_split[0]}+{job_name_split[1]}+{job_name_split[2]}+{job_name_split[3]}+{job_name_split[4]}&txtLocation='

response = requests.get(url)

soup = BeautifulSoup(response.text, "lxml")

job_titles_list = []
job_titles_list_final = []
job_titles = soup.find_all("h2", class_ = "heading-trun")
for job_title in job_titles[:5]:
    job_titles_list.append(job_title["title"])

company_names_list = []
company_names = soup.find_all("h3", class_ = "joblist-comp-name")
for company_name in company_names[: 5]:
    company_name = company_name.text
    cleaned_company_name = re.sub(r'\s+', ' ', company_name).strip()
    company_names_list.append(cleaned_company_name)

hyperlinks_list = []
hyperlinks = soup.find_all("a", class_ = "posoverlay_srp")
for hyperlink in hyperlinks[:5]:
    hyperlinks_list.append(hyperlink["href"])

print(hyperlinks_list)

'''skills_req_list = []
skills_req = soup.find_all("")
'''

job_recommendations_dataframe = pd.DataFrame({"Job Titles": job_titles_list, "Company Names": company_names_list, "HyperLinks": hyperlinks_list})

print(job_recommendations_dataframe)