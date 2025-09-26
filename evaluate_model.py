import os
import pickle
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score)
import matplotlib.pyplot as plt

from transformer_layers import PositionalEncoding, EncoderBlock

# Suppress less important TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. Setup and Artifact Loading ---
MODEL_PATH = 'models/sentiment_transformer.keras'
TEST_DATASET_PATH = 'models/test_dataset_tf'
TEST_LABELS_PATH = 'models/test_labels.pkl'
BATCH_SIZE = 32

print("Loading artifacts...")

# Define custom objects for loading the model with custom layers
custom_objects = {
    'PositionalEncoding': PositionalEncoding,
    'EncoderBlock': EncoderBlock
}

# Load the trained model
transformer_classifier = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects=custom_objects,
    compile=False
)

# Load the pre-processed test dataset and the true labels
test_dataset = tf.data.Dataset.load(TEST_DATASET_PATH)
with open(TEST_LABELS_PATH, 'rb') as f:
    y_true = pickle.load(f)

print("Artifacts loaded successfully.")

# --- 2. Prediction and Evaluation ---
print("Making predictions...")
# The dataset is already batched and vectorized, ready for prediction
y_pred_probs = transformer_classifier.predict(test_dataset).flatten()
y_pred_classes = (y_pred_probs > 0.5).astype(int)

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

# --- 3. Visualizations ---
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
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()