import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import os

# 1. Força o desligamento do XLA via código (sobrescreve o ambiente)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import tensorflow as tf
from tensorflow.keras import mixed_precision

# 2. Desabilita explicitamente o compilador JIT (que usa XLA)
tf.config.optimizer.set_jit(False)

# 3. Ativa Mixed Precision (Economiza MUITA VRAM e acelera o treino na GTX 1660)
# Isso faz o modelo usar float16 onde possível, reduzindo o uso de memória pela metade.
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print(">>> Configurações de Memória: XLA Desativado | Mixed Precision Ativado")

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from gensim.models import KeyedVectors
import mlflow

# Project imports
from src.ml.transformer_layers import PositionalEncoding, EncoderBlock
from src.ml.model_wrapper import SentimentTransformerModel
from src.config import settings

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)


def create_transformer_classifier(vocab_size, num_layers, d_model, num_heads, dff,
                                  max_seq_len, embedding_matrix, rate=0.1):
    """Defines and creates the Transformer-based classifier model."""
    inputs = tf.keras.layers.Input(shape=(max_seq_len,))

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=d_model, weights=[embedding_matrix],
        trainable=False, mask_zero=True
    )
    x = embedding_layer(inputs)
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    x = PositionalEncoding(max_seq_len, d_model)(x)
    x = tf.keras.layers.Dropout(rate)(x)

    for _ in range(num_layers):
        x = EncoderBlock(d_model, num_heads, dff, rate)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(rate)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def run_training(train_csv: Path, val_csv: Path, test_csv: Path, embeddings_bin: Path, artifacts_dir: Path):
    """
    Trains the Transformer model and logs it to MLflow.
    """
    # --- 1. Load pre-processed data ---
    print("Loading pre-processed datasets...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_texts, train_labels = train_df['text'].tolist(), train_df['sentiment'].tolist()
    val_texts, val_labels = val_df['text'].tolist(), val_df['sentiment'].tolist()
    test_texts, test_labels = test_df['text'].tolist(), test_df['sentiment'].tolist()

    # --- 2. Vectorization and Embedding Matrix ---
    print("Loading embeddings and creating matrix...")
    word_vectors = KeyedVectors.load(str(embeddings_bin))
    embedding_dim = word_vectors.vector_size

    MAX_VOCAB_SIZE = 20000
    MAX_SEQUENCE_LENGTH = 200
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_VOCAB_SIZE, output_mode='int', output_sequence_length=MAX_SEQUENCE_LENGTH
    )
    vectorize_layer.adapt(train_texts)

    vocab = vectorize_layer.get_vocabulary()
    word_index = dict(zip(vocab, range(len(vocab))))
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, i in word_index.items():
        if word in word_vectors:
            embedding_matrix[i] = word_vectors[word]

    # --- 3. Model Building and Training ---
    print("Building and training the model...")
    D_MODEL, NUM_LAYERS, NUM_HEADS, DFF, DROPOUT_RATE = 300, 2, 6, 512, 0.1

    transformer_classifier = create_transformer_classifier(
        vocab_size=MAX_VOCAB_SIZE, num_layers=NUM_LAYERS, d_model=D_MODEL,
        num_heads=NUM_HEADS, dff=DFF, max_seq_len=MAX_SEQUENCE_LENGTH,
        embedding_matrix=embedding_matrix, rate=DROPOUT_RATE
    )
    transformer_classifier.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    def vectorize_text_and_label(text, label):
        return vectorize_layer(text), label

    train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels)).map(vectorize_text_and_label).batch(
        32)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_texts, val_labels)).map(vectorize_text_and_label).batch(32)

    history = transformer_classifier.fit(train_dataset, epochs=20, validation_data=val_dataset,
                                         callbacks=[early_stopping])

    print("Training finished.")

    print("\nSaving all artifacts to disk before logging to MLflow...")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Salvar modelo, vocabulário e dataset de teste
    model_path = artifacts_dir / "sentiment_transformer.keras"
    vocab_path = artifacts_dir / "vectorizer_vocab.pkl"
    test_dataset_path = artifacts_dir / "test_dataset_tf"
    test_labels_path = artifacts_dir / "test_labels.pkl"

    transformer_classifier.save(model_path)

    with open(vocab_path, "wb") as f:
        pickle.dump(vectorize_layer.get_vocabulary(), f)

    vectorized_test_texts = vectorize_layer(test_texts)
    test_dataset_vectorized = tf.data.Dataset.from_tensor_slices(
        (vectorized_test_texts, np.array(test_labels))
    )
    test_dataset_vectorized = test_dataset_vectorized.batch(32)
    test_dataset_vectorized.save(str(test_dataset_path))

    with open(test_labels_path, 'wb') as f:
        pickle.dump(np.array(test_labels), f)

    print("\nLogging to MLflow...")

    mlflow.set_experiment("sentiment-analysis-transformer")

    run = mlflow.start_run()
    try:
        print("Logging parameters...")
        mlflow.log_params({
            "vocab_size": MAX_VOCAB_SIZE, "max_seq_len": MAX_SEQUENCE_LENGTH,
            "embedding_dim": D_MODEL, "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS, "dff": DFF, "dropout": DROPOUT_RATE
        })

        print("Logging metrics...")
        best_epoch_index = np.argmin(history.history['val_loss'])
        final_metrics = {
            "val_loss": history.history['val_loss'][best_epoch_index],
            "val_accuracy": history.history['val_accuracy'][best_epoch_index],
            "val_auc": history.history['val_auc'][best_epoch_index]
        }
        mlflow.log_metrics(final_metrics)

        artifacts_for_mlflow = {
            "transformer_model": str(model_path),
            "vectorizer_vocab": str(vocab_path)
        }

        print("Logging model to MLflow Registry...")
        mlflow.pyfunc.log_model(
            name="sentiment_transformer",
            python_model=SentimentTransformerModel(),
            artifacts=artifacts_for_mlflow,
            code_paths=["src/ml/model_wrapper.py", "src/ml/transformer_layers.py"],
            registered_model_name=settings.MODEL_NAME_REGISTRY
        )
        print(f"Model '{settings.MODEL_NAME_REGISTRY}' registered successfully in run {run.info.run_id}")

    finally:
        print("Finalizing MLflow run...")
        mlflow.end_run()

    print("Clearing Keras session...")
    tf.keras.backend.clear_session()

if __name__ == "__main__":
    """Main function to parse arguments and run the training pipeline."""
    parser = argparse.ArgumentParser(description="Train a Transformer model for sentiment analysis.")

    parser.add_argument("--train-csv", type=str, required=True, help="Path to the training CSV file.")
    parser.add_argument("--val-csv", type=str, required=True, help="Path to the validation CSV file.")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to the test CSV file.")
    parser.add_argument("--embeddings-bin", type=str, required=True, help="Path to the binary embeddings file.")
    parser.add_argument("--artifacts-dir", type=str, required=True, help="Directory to save output artifacts.")

    args = parser.parse_args()

    run_training(
        train_csv=Path(args.train_csv),
        val_csv=Path(args.val_csv),
        test_csv=Path(args.test_csv),
        embeddings_bin=Path(args.embeddings_bin),
        artifacts_dir=Path(args.artifacts_dir)
    )