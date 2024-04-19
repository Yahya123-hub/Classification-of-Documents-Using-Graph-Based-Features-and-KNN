import requests
from bs4 import BeautifulSoup
import csv

url = "https://thestoryexchange.org/meet-mfk-fisher-who-changed-the-way-we-write-about-food/?fbclid=IwAR1G-uISTngA_UrOsfPmlPrtD9iJG5oNK71vFMt-nXxLfhKXx2fYHMm3JH4_aem_ARPOxTGASW_JQAtIqUO904fAq_F7WiHzaa_txBxP-CwUPLhUKBD0aLRlkdHkQARCll-k08RSpEUOp4GETXjub162_Av5sQSn6OD1VLop_IT8gV0XlEgfalOzTlgstbPGPIE-a31Q11z0-8sNYuIoZUu8"
req = requests.get(url)
soup = BeautifulSoup(req.content, "html.parser")


title_element = soup.find("h1", attrs={"class": "entry-title"})
title = title_element.text.strip() if title_element else "Title not found"

content_container = soup.find("div", class_="container-fluid")
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
