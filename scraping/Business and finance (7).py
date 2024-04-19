import requests
from bs4 import BeautifulSoup
import csv

url = "https://www.brecorder.com/news/40298495/multiplicity-of-taxes-heavily-burdening-industrial-sector"
req = requests.get(url)
soup = BeautifulSoup(req.content, "html.parser")

title_element = soup.find("h1", attrs={"data-id": "40298495"})
title = title_element.text.strip() if title_element else "Title not found"

content_container = soup.find("div", class_="story__content overflow-hidden pt-2 mt-2")
paragraphs = content_container.find_all("p")
content = "\n".join(paragraph.text.strip() for paragraph in paragraphs)

print("Title: ", title)
print("Content: ", content)


file_path = "E:\\Semester 6\\GT\\article_data.csv"

with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)    
    if csvfile.tell() == 0:
        csv_writer.writerow(['Title', 'Content'])
    csv_writer.writerow([title, content])

print("Data appended to article_data.csv at", file_path)