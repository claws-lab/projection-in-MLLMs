import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from tqdm import tqdm

# Check if CUDA (GPU support) is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the dataset
dataset_path = 'test_dataset.csv'  # Replace with your CSV file path
data = pd.read_csv(dataset_path)

# Convert labels in the dataset to lowercase
data['label'] = data['label'].str.lower()

# Extract unique labels from the dataset to use as classes
classes = data['label'].unique().tolist()

# Initialize the CLIP model for zero-shot classification
model_name = "openai/clip-vit-large-patch14-336"
classifier = pipeline("zero-shot-image-classification", model=model_name, device=device)

# Function to classify a single image
def classify_image(image_path, classifier, classes):
    try:
        scores = classifier(image_path, candidate_labels=classes)
        # Get the highest scoring label
        top_result = max(scores, key=lambda x: x['score'])
        return top_result['label'].lower()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Classify each image in the dataset and store the results
predicted_labels = []
true_labels = data['label'].tolist()

# Using tqdm for progress display
for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Classifying"):
    predicted_label = classify_image(row['imagefilename'], classifier, classes)
    predicted_labels.append(predicted_label)

# Compute metrics
accuracy = accuracy_score(true_labels, predicted_labels)
macro_f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
macro_precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
macro_recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)

# Print the metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")