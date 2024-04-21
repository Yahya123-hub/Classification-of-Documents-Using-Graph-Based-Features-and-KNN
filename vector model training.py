import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import networkx as nx

# Step 1: Load Training Graphs
def load_graph(image_path):
    # Load graph from image
    image = plt.imread(image_path)
    gray_image = np.mean(image, axis=2)
    binary_image = (gray_image > np.mean(gray_image)) * 1
    min_dim = min(binary_image.shape)
    square_image = binary_image[:min_dim, :min_dim]
    return nx.from_numpy_array(square_image)

train_graph_folder = r'E:\Github Repos\Classification-of-Documents-Using-Graph-Based-Features-and-KNN\training_graphs'
train_graph_filenames = os.listdir(train_graph_folder)
train_graphs = []

for filename in train_graph_filenames:
    graph_path = os.path.join(train_graph_folder, filename)
    graph = load_graph(graph_path)
    train_graphs.append(graph)

# Step 2: Extract Features
# For demonstration, let's extract node degree as features
def extract_features(graphs):
    features = []
    for graph in graphs:
        degrees = dict(nx.degree(graph))
        # Append node degree values as features
        features.append(list(degrees.values()))
    return np.array(features)

X = extract_features(train_graphs)

# Step 3: Prepare Labels
# Assuming you have a list of labels corresponding to each training graph
# For demonstration, let's assume labels are stored in the filenames
# Extract labels from filenames
label_encoder = LabelEncoder()
train_labels = [int(filename.split('(')[-1].split(')')[0]) for filename in train_graph_filenames]
# Encode labels
y = label_encoder.fit_transform(train_labels)

# Step 4: Train Machine Learning Model
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Support Vector Classifier (SVC)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Step 5: Evaluate Model
y_pred = svm_model.predict(X_test)

# Load Predicted and True Labels from the file
predicted_labels_file = r"E:\Github Repos\Classification-of-Documents-Using-Graph-Based-Features-and-KNN\vector_labels.txt"
with open(predicted_labels_file, "r") as file:
    lines = file.readlines()

true_labels = []
predicted_labels = []
for line in lines:
    # Extracting true and predicted labels from each line
    true_label = int(line.split("True label:")[1].strip())
    predicted_label = int(line.split("Predicted label:")[1].split(",")[0].strip())
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Save vector results to a text file
vector_results_file = r"E:\Github Repos\Classification-of-Documents-Using-Graph-Based-Features-and-KNN\vector_results.txt"
with open(vector_results_file, "w") as file:
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Precision: {precision}\n")
    file.write(f"Recall: {recall}\n")
    file.write(f"F1-score: {f1}\n")
