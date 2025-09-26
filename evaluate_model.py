import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score)
import matplotlib.pyplot as plt

# Import custom Transformer layers required for loading the model
from transformer_layers import PositionalEncoding, EncoderBlock

# --- 1. Setup and Artifact Loading ---

# Suppress less important TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define paths for the saved model and data
MODEL_PATH = 'models/sentiment_transformer.keras'
VECTORIZER_PATH = 'models/vectorizer.pkl'
TEST_DATA_PATH = 'models/test_data.pkl'
BATCH_SIZE = 32

print("Loading artifacts...")

# Define custom objects for loading the model with custom layers
custom_objects = {
    'PositionalEncoding': PositionalEncoding,
    'EncoderBlock': EncoderBlock
}

# Load the trained model
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at '{MODEL_PATH}'.")
    exit()
transformer_classifier = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects=custom_objects,
    compile=False # We don't need to compile for inference
)

# Load and reconstruct the vectorizer
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer_data = pickle.load(f)
vectorize_layer = TextVectorization.from_config(vectorizer_data['config'])
vectorize_layer.set_weights(vectorizer_data['weights'])

# Load the test data
with open(TEST_DATA_PATH, 'rb') as f:
    test_data = pickle.load(f)
test_texts = test_data['texts']
test_labels = test_data['labels']
print("Artifacts loaded successfully.")

# --- 2. Data Preparation ---
print("Preparing test data...")

# Filter out any test samples that might be empty after vectorization
# (e.g., reviews containing only out-of-vocabulary words)
filtered_test_texts = []
filtered_test_labels = []
for text, label in zip(test_texts, test_labels):
    vectorized_text = vectorize_layer([text])
    # tf.math.count_nonzero counts how many tokens are not '0' (padding)
    # If there is at least one real token, the sample is valid.
    if tf.math.count_nonzero(vectorized_text) > 0:
        filtered_test_texts.append(text)
        filtered_test_labels.append(label)

if len(test_texts) != len(filtered_test_texts):
    print(
        f"Removed {len(test_texts) - len(filtered_test_texts)} invalid samples (empty or only OOV words).")

# Vectorization function for the tf.data pipeline
def vectorize_text(text, label):
    return vectorize_layer(text), label

# Create a tf.data.Dataset from the validated test data
test_dataset = tf.data.Dataset.from_tensor_slices((filtered_test_texts, filtered_test_labels))
test_dataset = test_dataset.map(vectorize_text).batch(BATCH_SIZE)

# Prediction and Evaluation
print("Making predictions...")
# Predict probabilities for the test set
y_pred_probs = transformer_classifier.predict(test_dataset).flatten()
# Convert probabilities to class labels (0 or 1) based on a 0.5 threshold
y_pred_classes = (y_pred_probs > 0.5).astype(int)
y_true = np.array(filtered_test_labels)

# Calculate and print evaluation metrics
print("\n--- Evaluation Metrics on the Test Set ---")
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes)
recall = recall_score(y_true, y_pred_classes)
f1 = f1_score(y_true, y_pred_classes)
auc_val = roc_auc_score(y_true, y_pred_probs)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:   {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC:      {auc_val:.4f}")

# Visualizations
print("\nGenerating visualizations...")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_val:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Dashed line for random chance
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()