import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def read_data_from_csv(file_path):
    """
    Read data from CSV file.

    Args:
    - file_path: Path to the CSV file

    Returns:
    - List of tuples (Title, Content)
    """
    data = pd.read_csv(file_path)
    return list(zip(data['Title'], data['Content']))

def preprocess_text(text):
    """
    Preprocess text data by tokenizing and removing punctuation.

    Args:
    - text: Input text

    Returns:
    - List of tokens
    """
    # Tokenize and remove punctuation
    tokens = text.split()
    tokens = [token.strip(",.?!") for token in tokens]
    return tokens

def generate_graph(text):
    """
    Generate a graph representation from text.

    Args:
    - text: Input text

    Returns:
    - NetworkX graph
    """
    tokens = preprocess_text(text)
    graph = nx.Graph()
    for i, token in enumerate(tokens):
        if i < len(tokens) - 1:
            next_token = tokens[i + 1]
            graph.add_edge(token, next_token)
    return graph

def mine_frequent_subgraphs(documents, min_support=0.5):
    """
    Mine frequent subgraphs from a list of documents.

    Args:
    - documents: List of tuples (Title, Content)
    - min_support: Minimum support threshold for frequent subgraph mining

    Returns:
    - List of dictionaries where each dictionary contains frequent subgraphs for one article
    """
    frequent_subgraphs_all_articles = []

    for title, content in documents:
        graph = generate_graph(content)
        frequent_subgraphs = []

        # Mine frequent subgraphs
        for component in nx.connected_components(graph):
            subgraph = graph.subgraph(component)
            frequent_subgraphs.append(subgraph)

        frequent_subgraphs_all_articles.append({title: frequent_subgraphs})

    return frequent_subgraphs_all_articles

# Example usage
if __name__ == "__main__":
    # Read data from CSV
    file_path = "training_data.csv"
    documents = read_data_from_csv(file_path)

    # Mine frequent subgraphs
    min_support = 0.5
    frequent_subgraphs_all_articles = mine_frequent_subgraphs(documents, min_support)

    # Plot and save frequent subgraphs as images for each article
    for i, article_data in enumerate(frequent_subgraphs_all_articles, start=1):
        title, frequent_subgraphs = list(article_data.items())[0]
        for j, subgraph in enumerate(frequent_subgraphs, start=1):
            plt.figure()
            nx.draw(subgraph, with_labels=True)
            plt.title(f"Article {i}: {title} - Subgraph {j}")
            plt.savefig(f"subgraph_article_{i}_{j}.png")
            plt.close()
