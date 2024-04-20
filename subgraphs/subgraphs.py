
import csv
import networkx as nx
import matplotlib.pyplot as plt

def create_graph_from_csv(csv_path):
    G = nx.Graph()
    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        for row in reader:
            if len(row) >= 2:  # Ensure row has at least 2 values
                article_number, content = row[0], row[1]
    
                words = content.split()  # Split content into words
                for i in range(len(words) - 1):
                    word1 = words[i]
                    word2 = words[i + 1]
                    # Add nodes if not already in the graph
                    G.add_node(word1)
                    G.add_node(word2)
                    # Add edge between words
                    G.add_edge(word1, word2)
    return G

# Load training data and create graph
training_csv_path = "training_data.csv"
training_graph = create_graph_from_csv(training_csv_path)

def frequent_subgraphs(graph, min_support=0.2, max_size=3):
    frequent_subs = []
    for sub_size in range(2, max_size + 1):
        for subgraph in nx.find_cliques(graph):
            if len(subgraph) == sub_size:
                count = 0
                for node in graph.nodes():
                    if all(n in subgraph for n in graph.neighbors(node)):
                        count += 1
                support = count / len(graph)
                if support >= min_support:
                    frequent_subs.append(graph.subgraph(subgraph))
    return frequent_subs


frequent_subgraphs_list = frequent_subgraphs(training_graph, min_support=0.2, max_size=3)  # Adjust min_support and max_size as needed

# Plot the frequent subgraphs
for i, subgraph in enumerate(frequent_subgraphs_list):
    plt.figure(figsize=(6, 4))
    nx.draw(subgraph, with_labels=True, font_weight='bold')
    plt.title(f"Frequent Subgraph {i+1}")
    plt.show()
