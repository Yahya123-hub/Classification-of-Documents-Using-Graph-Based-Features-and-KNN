import cv2
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from PIL import Image

MIN_NODE_AREA = 50  # Adjust the value according to your specific requirements
EDGE_THRESHOLD = 50  # Adjust the value according to your specific requirements


# Function to perform image processing and detect nodes and edges
def image_processing(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform edge detection
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    
    # Check OpenCV version
    (major, minor) = cv2.__version__.split('.')[:2]
    if int(major) < 4:  # For OpenCV version 3.x
        _, contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:  # For OpenCV version 4.x or later
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Identify nodes based on contours
    nodes = []
    for contour in contours:
        # Check if contour has at least 3 points
        if len(contour) >= 3:
            # Calculate area of contour
            area = cv2.contourArea(contour)
            # Check if contour area is large enough to be considered a node
            if area > MIN_NODE_AREA:
                # Calculate centroid of contour as node
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    nodes.append((cX, cY))
    
    # Identify edges based on proximity of nodes
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1 = nodes[i]
            node2 = nodes[j]
            # Calculate Euclidean distance between nodes
            distance = np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
            # If distance is below a threshold, consider nodes connected by an edge
            if distance < EDGE_THRESHOLD:
                edges.append((node1, node2))
    
    return nodes, edges

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform edge detection
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    
    # Check OpenCV version
    (major, minor) = cv2.__version__.split('.')[:2]
    if int(major) < 4:  # For OpenCV version 3.x
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:  # For OpenCV version 4.x or later
        _, contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Identify nodes based on contours
    nodes = []
    for contour in contours:
        # Check if contour is valid and has enough points
        if len(contour) >= 3:
            # Calculate area of contour
            area = cv2.contourArea(contour)
            # Check if contour area is large enough to be considered a node
            if area > MIN_NODE_AREA:
                # Calculate centroid of contour as node
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    nodes.append((cX, cY))
    
    # Identify edges based on proximity of nodes
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1 = nodes[i]
            node2 = nodes[j]
            # Calculate Euclidean distance between nodes
            distance = np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
            # If distance is below a threshold, consider nodes connected by an edge
            if distance < EDGE_THRESHOLD:
                edges.append((node1, node2))
    
    return nodes, edges

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform edge detection
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    
    # Check OpenCV version
    (major, minor) = cv2.__version__.split('.')[:2]
    if int(major) < 4:  # For OpenCV version 3.x
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:  # For OpenCV version 4.x or later
        _, contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Identify nodes based on contours
    nodes = []
    for contour in contours:
        # Check if contour is valid
        if len(contour) >= 3:  # Ensure contour has at least 3 points
            # Calculate area of contour
            area = cv2.contourArea(contour)
            # Check if contour area is large enough to be considered a node
            if area > MIN_NODE_AREA:
                # Calculate centroid of contour as node
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    nodes.append((cX, cY))
    
    # Identify edges based on proximity of nodes
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1 = nodes[i]
            node2 = nodes[j]
            # Calculate Euclidean distance between nodes
            distance = np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
            # If distance is below a threshold, consider nodes connected by an edge
            if distance < EDGE_THRESHOLD:
                edges.append((node1, node2))
    
    return nodes, edges

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform edge detection
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    
    # Check OpenCV version
    (major, minor) = cv2.__version__.split('.')[:2]
    if int(major) < 4:  # For OpenCV version 3.x
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:  # For OpenCV version 4.x or later
        _, contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Identify nodes based on contours
    nodes = []
    for contour in contours:
        # Calculate area of contour
        area = cv2.contourArea(contour)
        # Check if contour area is large enough to be considered a node
        if area > MIN_NODE_AREA:
            # Calculate centroid of contour as node
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                nodes.append((cX, cY))
    
    # Identify edges based on proximity of nodes
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1 = nodes[i]
            node2 = nodes[j]
            # Calculate Euclidean distance between nodes
            distance = np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
            # If distance is below a threshold, consider nodes connected by an edge
            if distance < EDGE_THRESHOLD:
                edges.append((node1, node2))
    
    return nodes, edges


def reconstruct_graph(nodes, edges):
    # Create an empty graph
    graph = nx.Graph()
    
    # Add nodes to the graph
    for node in nodes:
        graph.add_node(node)
    
    # Add edges to the graph
    for edge in edges:
        node1, node2 = edge  # Assuming edge is a tuple of nodes
        graph.add_edge(node1, node2)
    
    return graph



def frequent_subgraph_mining(graphs, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    num_subgraphs = len(graphs)
    for i, subgraph in enumerate(graphs, start=1):
        plt.figure()
        nx.draw(subgraph, with_labels=True)
        plt.title(f"Subgraph {i}")
        subgraph_name = f"subgraph_{i}.png"
        subgraph_path = os.path.join(output_folder, subgraph_name)
        plt.savefig(subgraph_path)
        plt.close()
        print(f"Created subgraph {i}/{num_subgraphs}")
    
    combined_graph = nx.compose_all(graphs)
    
    common_subgraphs = []
    for subgraph_nodes in nx.enumerate_all_cliques(combined_graph):
        if len(subgraph_nodes) == 3:
            subgraph = combined_graph.subgraph(subgraph_nodes)
            common_subgraphs.append(subgraph)
    
    num_common_subgraphs = len(common_subgraphs)
    for i, subgraph in enumerate(common_subgraphs, start=num_subgraphs + 1):
        plt.figure()
        nx.draw(subgraph, with_labels=True)
        plt.title(f"Subgraph {i}")
        subgraph_name = f"subgraph_{i}.png"
        subgraph_path = os.path.join(output_folder, subgraph_name)
        plt.savefig(subgraph_path)
        plt.close()
        print(f"Created subgraph {i}/{num_common_subgraphs}")
    
    return common_subgraphs

    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save each subgraph as a separate PNG image
    num_subgraphs = len(graphs)
    for i, subgraph in enumerate(graphs, start=1):
        # Create a new figure for each subgraph
        plt.figure()
        nx.draw(subgraph, with_labels=True)
        plt.title(f"Subgraph {i}")
        # Save the figure as a PNG image
        subgraph_name = f"subgraph_{i}.png"
        subgraph_path = os.path.join(output_folder, subgraph_name)
        plt.savefig(subgraph_path)
        # Close the figure to release memory
        plt.close()
        
        # Print indicator
        print(f"Created subgraph {i}/{num_subgraphs}")
    
    # Combine all graphs into a single graph
    combined_graph = nx.compose_all(graphs)
    
    # Apply frequent subgraph mining algorithm (e.g., subgraph isomorphism)
    # For simplicity, let's assume we are looking for all subgraphs of size 3
    common_subgraphs = []
    for subgraph in nx.enumerate_all_cliques(combined_graph):
        if len(subgraph) == 3:  # Assuming we are interested in subgraphs of size 3
            common_subgraphs.append(subgraph)
    
    # Save each subgraph as a separate PNG image
    num_common_subgraphs = len(common_subgraphs)
    for i, subgraph in enumerate(common_subgraphs, start=num_subgraphs + 1):
        # Create a new figure for each subgraph
        plt.figure()
        nx.draw(subgraph, with_labels=True)
        plt.title(f"Subgraph {i}")
        # Save the figure as a PNG image
        subgraph_name = f"subgraph_{i}.png"
        subgraph_path = os.path.join(output_folder, subgraph_name)
        plt.savefig(subgraph_path)
        # Close the figure to release memory
        plt.close()
        
        # Print indicator
        print(f"Created subgraph {i}/{num_common_subgraphs}")
    
    return common_subgraphs
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save each subgraph as a separate PNG image
    for i, subgraph in enumerate(graphs, start=1):
        # Create a new figure for each subgraph
        plt.figure()
        nx.draw(subgraph, with_labels=True)
        plt.title(f"Subgraph {i}")
        # Save the figure as a PNG image
        subgraph_name = f"subgraph_{i}.png"
        subgraph_path = os.path.join(output_folder, subgraph_name)
        plt.savefig(subgraph_path)
        # Close the figure to release memory
        plt.close()
    
    return graphs
    # Combine all graphs into a single graph
    combined_graph = nx.compose_all(graphs)
    
    # Apply frequent subgraph mining algorithm (e.g., subgraph isomorphism)
    # For simplicity, let's assume we are looking for all subgraphs of size 3
    common_subgraphs = []
    for subgraph in nx.enumerate_all_cliques(combined_graph):
        if len(subgraph) == 3:  # Assuming we are interested in subgraphs of size 3
            common_subgraphs.append(subgraph)
    
    # Save each subgraph as a separate PNG image
    for i, subgraph in enumerate(common_subgraphs, start=1):
        # Create a new figure for each subgraph
        plt.figure()
        nx.draw(subgraph, with_labels=True)
        plt.title(f"Subgraph {i}")
        # Save the figure as a PNG image
        subgraph_name = f"subgraph_{i}.png"
        subgraph_path = os.path.join(output_folder, subgraph_name)
        plt.savefig(subgraph_path)
        # Close the figure to release memory
        plt.close()
    
    return common_subgraphs


def extract_features_to_csv(common_subgraphs, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['subgraph_id', 'node_count', 'edge_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i, subgraph in enumerate(common_subgraphs, start=1):
            node_count = len(subgraph.nodes())
            edge_count = len(subgraph.edges())
            writer.writerow({
                'subgraph_id': f'Subgraph {i}',
                'node_count': node_count,
                'edge_count': edge_count
            })




# Load PNG images of the graphs
graph_images = []
num_graphs = 30  # Update with the total number of graph images
for i in range(1, num_graphs + 1):
    image_path = f"E:/Github Repos/Classification-of-Documents-Using-Graph-Based-Features-and-KNN/prepped graphs/graph ({i}).png"
    graph_images.append(cv2.imread(image_path))

# Process each graph image
for image in graph_images:
    # Perform image processing and detect nodes and edges
    nodes, edges = image_processing(image)
    # Reconstruct graph from node and edge data
    graph = reconstruct_graph(nodes, edges)
    # Apply frequent subgraph mining on the reconstructed graph
    output_folder = r"E:/Github Repos/Classification-of-Documents-Using-Graph-Based-Features-and-KNN/generated_subgraphs"
    common_subgraphs = frequent_subgraph_mining([graph], output_folder)
    # Extract features from identified common subgraphs
    output_file = os.path.abspath("E:/Github Repos/Classification-of-Documents-Using-Graph-Based-Features-and-KNN/extracted_features.csv")
    extract_features_to_csv(common_subgraphs, output_file)
    print(f"Features have been saved to {output_file}")
