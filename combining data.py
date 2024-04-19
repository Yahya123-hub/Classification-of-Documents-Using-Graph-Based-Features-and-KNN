import csv

# Absolute path to the input text file
input_file_path = r"E:\Semester 6\GT\food_article_paragraphs.txt"

# Absolute path to the existing CSV file
existing_csv_path = r"E:\Semester 6\GT\scraped_article_data.csv"

# Read content from the input text file
with open(input_file_path, "r", encoding="utf-8") as file:
    content = file.read()

# Split content into articles based on the "Article" separator
articles = content.split("Article ")

# Remove the empty first element
articles = articles[1:]

# Append paragraphs to the existing CSV file
with open(existing_csv_path, "a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    for i, article in enumerate(articles, start=1):
        # Split each article into title and content
        title, content = article.split(":\n", 1)
        # Write article number and content to the CSV file
        writer.writerow([f"Article {i}", content.strip()])

print("Content from the text file has been appended to existing csv")
