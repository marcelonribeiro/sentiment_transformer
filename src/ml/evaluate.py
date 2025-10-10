import argparse
import pickle
import json
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score)
import matplotlib.pyplot as plt

# Project imports
from src.ml.transformer_layers import PositionalEncoding, EncoderBlock


def run_evaluation(model_path: Path, test_dataset_path: Path, test_labels_path: Path,
                   metrics_path: Path, cm_path: Path, roc_path: Path):
    """
    Evaluates a trained model, calculates metrics, and saves visualizations.
    """
    print("Loading artifacts for evaluation...")
    custom_objects = {'PositionalEncoding': PositionalEncoding, 'EncoderBlock': EncoderBlock}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    test_dataset = tf.data.Dataset.load(str(test_dataset_path))
    with open(test_labels_path, 'rb') as f:
        y_true = pickle.load(f)

    print("Making predictions on the test set...")
    y_pred_probs = model.predict(test_dataset).flatten()
    y_pred_classes = (y_pred_probs > 0.5).astype(int)

    # Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes)
    recall = recall_score(y_true, y_pred_classes)
    f1 = f1_score(y_true, y_pred_classes)
    auc = roc_auc_score(y_true, y_pred_probs)

    metrics = {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1_score': f1, 'auc': auc
    }
    print(f"Evaluation Metrics: {metrics}")

    # Save Metrics to JSON
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    # Generate and Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion Matrix saved to {cm_path}")

    # Generate and Save ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC Curve saved to {roc_path}")


if __name__ == "__main__":
    """Main function to parse arguments and run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained sentiment analysis model.")

    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained .keras model file.")
    parser.add_argument("--test-dataset-path", type=str, required=True,
                        help="Path to the saved test tf.data.Dataset directory.")
    parser.add_argument("--test-labels-path", type=str, required=True, help="Path to the pickled test labels file.")
    parser.add_argument("--metrics-path", type=str, required=True, help="Path to save the output metrics JSON file.")
    parser.add_argument("--cm-path", type=str, required=True, help="Path to save the confusion matrix PNG image.")
    parser.add_argument("--roc-path", type=str, required=True, help="Path to save the ROC curve PNG image.")

    args = parser.parse_args()

    run_evaluation(
        model_path=Path(args.model_path),
        test_dataset_path=Path(args.test_dataset_path),
        test_labels_path=Path(args.test_labels_path),
        metrics_path=Path(args.metrics_path),
        cm_path=Path(args.cm_path),
        roc_path=Path(args.roc_path)
    )
