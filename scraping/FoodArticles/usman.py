import requests
from bs4 import BeautifulSoup

# Read article URLs from the file
with open("article_urls.txt", "r", encoding="utf-8") as file:
    article_urls = file.readlines()

# Remove any leading/trailing whitespaces and newline characters
article_urls = [url.strip() for url in article_urls]

# Function to extract paragraphs from an article URL


def extract_paragraphs(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")
        # Find the element containing the article content
        # Update class if necessary
        article_content = soup.find("article", class_="single")
        if article_content:
            # Find all paragraphs within the article content
            paragraphs = article_content.find_all("p")
            # Combine paragraphs into a single string
            article_text = "\n".join([p.get_text() for p in paragraphs])
            return article_text
        else:
            print(f"No article content found in {url}")
            return None
    except Exception as e:
        print(f"Error occurred while processing {url}: {e}")
        return None


# Save paragraphs into a text file
with open("article_paragraphs.txt", "w", encoding="utf-8") as file:
    for i, url in enumerate(article_urls, start=1):
        # Extract paragraphs from the article URL
        article_text = extract_paragraphs(url)
        if article_text:
            # Write article number and paragraphs to the file
            file.write(f"Article {i}:\n")
            file.write(article_text)
            file.write("\n.....................................\n")

print("Paragraphs from the articles have been saved to article_paragraphs.txt")
