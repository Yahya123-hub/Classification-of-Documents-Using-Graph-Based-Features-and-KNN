from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load Predicted and True Labels
predicted_labels_file = r"E:\Github Repos\Classification-of-Documents-Using-Graph-Based-Features-and-KNN\labels.txt"
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

# Print true labels and predicted labels
print("True Labels:", true_labels)
print("Predicted Labels:", predicted_labels)

# Check if predicted labels cover the same classes as the true labels
true_classes = set(true_labels)
predicted_classes = set(predicted_labels)

print("True Classes:", true_classes)
print("Predicted Classes:", predicted_classes)

# Compare true and predicted classes
missing_classes = true_classes - predicted_classes
extra_classes = predicted_classes - true_classes

print("Classes in True Labels but not in Predicted Labels:", missing_classes)
print("Classes in Predicted Labels but not in True Labels:", extra_classes)

# Step 2: Compute Metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print("Graph Based Results")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Step 3: Generate Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Step 4: Compare with Vector-based Methods
vector_results_file = r"E:\Github Repos\Classification-of-Documents-Using-Graph-Based-Features-and-KNN\vector_results.txt"
with open(vector_results_file, "r") as file:
    vector_results = file.read()

# Print the vector results
print("Vector Results")
print(vector_results)

