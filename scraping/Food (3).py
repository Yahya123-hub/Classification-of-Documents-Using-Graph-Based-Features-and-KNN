import requests
from bs4 import BeautifulSoup
import csv

url = "https://foodtank.com/news/2024/03/cultivate-food-sovereignty-in-your-home-garden-with-these-resources/"
req = requests.get(url)
soup = BeautifulSoup(req.content, "html.parser")


title_element = soup.find("h1", itemprop="name headline")
title = title_element.text.strip() if title_element else "Title not found"

content_container = soup.find("div", class_="post-content entry-content cf")
paragraphs = content_container.find_all("p")
paragraphs = paragraphs[:-1]
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