'''pip install requests 
pip install beautifulsoup4 
pip install pandas 
pip install lxml'''

import requests
from bs4 import BeautifulSoup
import pandas as pd

course_name = input("Enter the course name: ")

course_split = course_name.split()

if len(course_split) == 1:
    url = f"https://www.coursera.org/search?query={course_split[0]}"
elif len(course_split) == 2:
    url = f"https://www.coursera.org/search?query={course_split[0]}%20{course_split[1]}"
elif len(course_split) == 3:
    url = f"https://www.coursera.org/search?query={course_split[0]}%20{course_split[1]}%20{course_split[2]}"
elif len(course_split) == 4:
    url = f"https://www.coursera.org/search?query={course_split[0]}%20{course_split[1]}%20{course_split[2]}%20{course_split[3]}"
elif len(course_split) == 5:
    url = f"https://www.coursera.org/search?query={course_split[0]}%20{course_split[1]}%20{course_split[2]}%20{course_split[3]}%20{course_split[4]}"
elif len(course_split) == 6:
    url = f"https://www.coursera.org/search?query={course_split[0]}%20{course_split[1]}%20{course_split[2]}%20{course_split[3]}%20{course_split[4]}%20{course_split[5]}"
else:
    url = f"https://www.coursera.org/search?query={course_split[0]}%20{course_split[1]}%20{course_split[2]}%20{course_split[3]}%20{course_split[4]}%20{course_split[5]}%20{course_split[6]}"

response = requests.get(url)
soup = BeautifulSoup(response.text, "lxml")

educators_list = [educator.text for educator in soup.find_all("p", class_="cds-ProductCard-partnerNames css-vac8rf")[:5]]
course_titles_list = [title.text for title in soup.find_all("h3", class_="cds-CommonCard-title css-6ecy9b")[:5]]
skills_list = [skill.text for skill in soup.find_all("div", class_="cds-CommonCard-bodyContent")[:5]]
ratings = soup.find_all("div", class_ = "cds-RatingStat-meter")
ratings_list = [rate.text.split("Rating")[0].strip() for rate in ratings[:5]]
hyperlinks_list = ["https://www.coursera.org" + hyperlink["href"] for hyperlink in soup.find_all("a", class_="cds-119 cds-113 cds-115 cds-CommonCard-titleLink css-vflzcf cds-142")[:5]]
levels_list = [level.text.split()[0] for level in soup.find_all("div", class_="cds-CommonCard-metadata")[:5]]
images_list = []
images_div = soup.find_all("img")
for i in images_div[5:10]:
    images_list.append(i["src"])


course_data = pd.DataFrame({
    "Course Title": course_titles_list,
    "Educator": educators_list,
    "Skills": skills_list,
    "Rating": ratings_list,
    "Link": hyperlinks_list,
    "Level": levels_list,
    "Images Link": images_list
})

print(course_data)




