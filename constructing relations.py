import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Absolute path to the preprocessed CSV file
preprocessed_csv_path = r"E:\Github Repos\Classification-of-Documents-Using-Graph-Based-Features-and-KNN\preprocessed_data.csv"

# Define window size for co-occurrence
window_size = 5

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Create a defaultdict to store co-occurrence counts
co_occurrences = defaultdict(int)

# Read preprocessed data from the CSV file
with open(preprocessed_csv_path, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        article_title = row['Title']  # Get the article title
        # Tokenize the preprocessed content
        tokens = word_tokenize(row['Content'])  # Assuming preprocessed content is in the 'Content' column
        # Iterate through the tokens and update co-occurrence counts
        for i, token in enumerate(tokens):
            # Stem the token
            stemmed_token = stemmer.stem(token)
            # Check if the token is not a stop word
            if stemmed_token not in stop_words:
                # Update co-occurrence counts within the window
                for j in range(max(i - window_size, 0), min(i + window_size + 1, len(tokens))):
                    if i != j:  # Avoid self-loops
                        # Stem the co-occurring token
                        co_occ_token = stemmer.stem(tokens[j])
                        if co_occ_token not in stop_words:
                            # Update co-occurrence count
                            co_occurrences[(stemmed_token, co_occ_token)] += 1
        # Save the co-occurrence relationships to a CSV file
        output_csv_path = r"E:\Github Repos\Classification-of-Documents-Using-Graph-Based-Features-and-KNN\word_relationships.csv"
        with open(output_csv_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for (term1, term2), weight in co_occurrences.items():
                writer.writerow([article_title, f"{term1} - {term2}", weight])
        print(f"Co-occurrence relationships for {article_title} have been saved.")


#We read the preprocessed data from the CSV file.
#For each preprocessed document, we tokenize the content and iterate through the tokens.
#For each token, we stem it, check if it's not a stop word, and then update co-occurrence counts within the specified window size.
#We use a defaultdict to store the co-occurrence counts, where the keys are pairs of terms and the values are their co-occurrence frequencies.
#Finally, we save the co-occurrence counts for verification to csv.
#the co-occurrence count refers to the number of times two terms (words) appear within a certain context or window size in the preprocessed document.
