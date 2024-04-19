import csv
import networkx as nx
import matplotlib.pyplot as plt

# Absolute path to the CSV file containing co-occurrence relationships
co_occurrence_csv_path = r"E:\Github Repos\Classification-of-Documents-Using-Graph-Based-Features-and-KNN\word_relationships.csv"

# Read co-occurrence relationships from the CSV file and group them by article title
article_relationships = {}
with open(co_occurrence_csv_path, "r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    header = next(reader)  # Read the header row
    for row in reader:
        article_title, terms, weight = row
        if article_title not in article_relationships:
            article_relationships[article_title] = []
        article_relationships[article_title].append((terms, int(weight)))

# Create and save individual graphs for each article
for article_title, relationships in article_relationships.items():
    # Create a directed graph
    graph = nx.DiGraph()
    # Add edges to the graph based on the relationships
    for terms, weight in relationships:
        term1, term2 = terms.split(" - ")
        graph.add_edge(term1, term2, weight=weight)
    # Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)  # Positions for all nodes
    nx.draw(graph, pos, with_labels=True, node_size=500, font_size=10, edge_color='b', alpha=0.7, arrows=False)
    nx.draw_networkx_edge_labels(graph, pos, font_color='red', edge_labels=nx.get_edge_attributes(graph, 'weight'))
    plt.title(f"Co-occurrence Graph - {article_title}")
    plt.axis('off')
    # Save the graph
    output_graph_path = f"E:\Github Repos\Classification-of-Documents-Using-Graph-Based-Features-and-KNN\graph_{article_title}.png"
    plt.savefig(output_graph_path)
    plt.close()
    print(f"Graph for article '{article_title}' saved to: {output_graph_path}")
