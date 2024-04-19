import csv

# Absolute path to the existing CSV file
existing_csv_path = r"E:\Semester 6\GT\scraped_article_data.csv"

# Absolute path to save the testing data CSV file
testing_csv_path = r"E:\Semester 6\GT\testing_data.csv"

# Number of articles to select for training
num_training_articles = 36

# Number of articles to select for testing
num_testing_articles = 9

# Read the articles from the existing CSV file
articles = []
with open(existing_csv_path, "r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    header = next(reader)  # Read the header row
    for row in reader:
        if len(row) >= 2:  # Ensure row has at least 2 values
            article_number, content = row[0], row[1]
            articles.append([article_number, content])

# Split the articles into training and testing sets
training_articles = articles[:num_training_articles]
testing_articles = articles[num_training_articles:num_training_articles + num_testing_articles]

# Write the selected articles to the testing data CSV file
with open(testing_csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header row
    writer.writerows(testing_articles)

print(f"{len(testing_articles)} articles have been saved to testing_data.csv")
