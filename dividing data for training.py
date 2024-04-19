import csv

# Absolute path to the existing CSV file
existing_csv_path = r"E:\Semester 6\GT\scraped_article_data.csv"

# Absolute path to save the training data CSV file
training_csv_path = r"E:\Semester 6\GT\training_data.csv"

# Number of articles to select for training
num_training_articles = 36

# Read the first 36 articles from the existing CSV file
training_articles = []
with open(existing_csv_path, "r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    header = next(reader)  # Read the header row
    for row in reader:
        if len(row) >= 2:  # Ensure row has at least 2 values
            article_number, content = row[0], row[1]
            training_articles.append([article_number, content])
            if len(training_articles) >= num_training_articles:
                break

# Write the selected articles to the training data CSV file
with open(training_csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header row
    writer.writerows(training_articles)

print(f"{len(training_articles)} articles have been saved to training_data.csv")
