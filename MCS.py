import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

def load_graph(image_path):
    # Load graph from image
    print(f"Loading graph from image: {image_path}")
    image = plt.imread(image_path)
    # Check if image has valid dimensions for a graph (e.g., 3 channels for RGB)
    if len(image.shape) != 3:
        raise ValueError("Invalid image dimensions. Expected RGB image.")
    # Convert image to grayscale
    gray_image = np.mean(image, axis=2)
    # Convert grayscale image to binary (0 or 1) based on thresholding
    binary_image = (gray_image > np.mean(gray_image)) * 1
    # Create a square image by cropping or resizing
    min_dim = min(binary_image.shape)
    square_image = binary_image[:min_dim, :min_dim]
    # Convert binary image to graph
    return nx.from_numpy_array(square_image)

def compute_mcs(graph1, graph2):
    # Compute MCS between two graphs
    mcs_size = 0
    
    # Convert graphs to adjacency matrices
    adjacency_matrix1 = nx.to_numpy_array(graph1)
    adjacency_matrix2 = nx.to_numpy_array(graph2)
    
    # Iterate over all pairs of nodes in graph1 and graph2
    for node1 in graph1.nodes():
        for node2 in graph2.nodes():
            if adjacency_matrix1[node1, node1] == 1 and adjacency_matrix2[node2, node2] == 1:
                if adjacency_matrix1[node1, node2] == 1 and adjacency_matrix2[node2, node1] == 1:
                    # Found a common node, increment MCS size
                    mcs_size += 1
    
    return mcs_size

def distance_metric(graph1, graph2):
    # Compute distance metric based on MCS between two graphs
    return compute_mcs(graph1, graph2)

def knn(train_graphs, test_graph, train_labels, k):
    # Compute distances between test graph and all training graphs
    distances = [distance_metric(train_graph, test_graph) for train_graph in train_graphs]

    # Get indices of k-nearest neighbors
    nearest_indices = np.argsort(distances)[:k]

    # Get class labels of k-nearest neighbors
    nearest_labels = [train_labels[i] for i in nearest_indices]

    # Predict class label based on majority vote
    predicted_label = max(set(nearest_labels), key=nearest_labels.count)

    return predicted_label

# Example usage
graph_folder = "E:\Github Repos\Classification-of-Documents-Using-Graph-Based-Features-and-KNN\prepped graphs"
graph_filenames = os.listdir(graph_folder)
num_graphs = len(graph_filenames)

# Load graphs from image files
train_graphs = []
train_labels = []  # Assuming labels are available for training graphs
for i, filename in enumerate(graph_filenames):
    graph_path = os.path.join(graph_folder, filename)
    print(f"Loading graph {i+1}/{num_graphs} from image: {graph_path}")
    graph = load_graph(graph_path)
    train_graphs.append(graph)
    # Assuming labels are stored in the filename or any other way
    # You need to implement this part to extract labels from filenames or any other source
    # For now, let's assume labels are integers corresponding to different classes
    train_labels.append(int(filename.split('.')[0].split('(')[-1]))

# Choose value of k (number of neighbors)
k = 5

# Load test graph (You need to implement this part)
test_graph = None  # You need to load the test graph from file or any other source

# Perform classification for the test graph
predicted_label = knn(train_graphs, test_graph, train_labels, k)
print("Predicted label:", predicted_label)


#--- Load training graphs from image files
train_graphs = []
train_labels = []  # Assuming labels are available for training graphs
train_graph_folder = "E:\Github Repos\Classification-of-Documents-Using-Graph-Based-Features-and-KNN\training_graphs"
train_graph_filenames = os.listdir(train_graph_folder)
num_train_graphs = len(train_graph_filenames)

for i, filename in enumerate(train_graph_filenames):
    graph_path = os.path.join(train_graph_folder, filename)
    print(f"Loading training graph {i+1}/{num_train_graphs} from image: {graph_path}")
    graph = load_graph(graph_path)
    train_graphs.append(graph)
    # Assuming labels are stored in the filename or any other way
    # You need to implement this part to extract labels from filenames or any other source
    # For now, let's assume labels are integers corresponding to different classes
    train_labels.append(int(filename.split('.')[0].split('(')[-1]))

# Load test graphs from image files
test_graphs = []
test_graph_folder = "E:\Github Repos\Classification-of-Documents-Using-Graph-Based-Features-and-KNN\test_graphs"
test_graph_filenames = os.listdir(test_graph_folder)
num_test_graphs = len(test_graph_filenames)

for i, filename in enumerate(test_graph_filenames):
    graph_path = os.path.join(test_graph_folder, filename)
    print(f"Loading test graph {i+1}/{num_test_graphs} from image: {graph_path}")
    graph = load_graph(graph_path)
    test_graphs.append(graph)

# Choose value of k (number of neighbors)
k = 5

# Perform classification for each test graph
for i, test_graph in enumerate(test_graphs):
    predicted_label = knn(train_graphs, test_graph, train_labels, k)
    print(f"Predicted label for test graph {i+1}: {predicted_label}")
