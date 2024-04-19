import csv
import networkx as nx
import matplotlib.pyplot as plt

# Absolute path to the CSV file containing co-occurrence relationships
co_occurrence_csv_path = "E:\Github Repos\Classification-of-Documents-Using-Graph-Based-Features-and-KNN\word_relationships.csv"

# Create a directed graph
graph = nx.DiGraph()

# Read co-occurrence relationships from the CSV file and add edges to the graph
with open(co_occurrence_csv_path, "r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    header = next(reader)  # Read the header row
    for row in reader:
        terms, weight = row[0].split(" - "), int(row[1])
        term1, term2 = terms
        graph.add_edge(term1, term2, weight=weight)

# Draw the graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(graph)  # Positions for all nodes
nx.draw(graph, pos, with_labels=True, node_size=500, font_size=10, edge_color='b', alpha=0.7, arrows=False)
nx.draw_networkx_edge_labels(graph, pos, font_color='red', edge_labels=nx.get_edge_attributes(graph, 'weight'))
plt.title("Co-occurrence Graph")
plt.axis('off')
plt.show()
