import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Absolute path to the existing CSV file
existing_csv_path = r"E:\Github Repos\Classification-of-Documents-Using-Graph-Based-Features-and-KNN\scraped_article_data.csv"

# Absolute path to save the preprocessed data CSV file
preprocessed_csv_path = r"E:\Github Repos\Classification-of-Documents-Using-Graph-Based-Features-and-KNN\preprocessed_data.csv"

# Number of articles to preprocess
num_articles = 45

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocess the text data
preprocessed_data = []
with open(existing_csv_path, "r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    header = next(reader)  # Read the header row
    for i, row in enumerate(reader, start=1):
        if i > num_articles:
            break
        if len(row) >= 2:
            article_number, content = row[0], row[1]
            # Tokenization
            tokens = word_tokenize(content)
            # Stop-word removal
            filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
            # Stemming
            stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
            preprocessed_data.append([article_number, ' '.join(stemmed_tokens)])

# Write the preprocessed data to the new CSV file
with open(preprocessed_csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header row
    writer.writerows(preprocessed_data)

print(f"{len(preprocessed_data)} articles have been preprocessed and saved to preprocessed_data.csv")
