import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from gensim.models import KeyedVectors

# Import custom Transformer layers
from transformer_layers import PositionalEncoding, EncoderBlock

# Data Loading and Preparation
print("Loading and preparing data...")
try:
    # Load the dataset
    df = pd.read_csv('data/B2W-Reviews01.csv', dtype={'product_id': str, 'user_id': str})
except FileNotFoundError:
    print("Error: B2W-Reviews01.csv not found.")
    exit()

# Select relevant columns and drop rows with missing values
df = df[['review_text', 'overall_rating']].dropna()
df.columns = ['text', 'rating']

# Create binary sentiment labels: 0 for negative (rating < 3), 1 for positive (rating == 5)
df_negative = df.loc[df['rating'] < 3].copy()
df_negative['sentiment'] = 0
df_positive = df.loc[df['rating'] == 5].copy()
df_positive['sentiment'] = 1

# Combine positive and negative reviews
df_filtered = pd.concat([df_negative, df_positive])

# Balance the dataset to have an equal number of positive and negative reviews
n_samples = df_filtered.groupby('sentiment')['rating'].count().min()
df_neg_balanced = df_negative.sample(n=n_samples, random_state=42)
df_pos_balanced = df_positive.sample(n=n_samples, random_state=42)
df_balanced = pd.concat([df_neg_balanced, df_pos_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)
df_balanced = df_balanced[['text', 'sentiment']]

# --- Data Cleaning ---
# Drop rows where 'text' is null and convert text to string type
df_balanced.dropna(subset=['text'], inplace=True)
df_balanced['text'] = df_balanced['text'].astype(str)
# Clean the text: convert to lowercase and remove non-alphanumeric characters
df_balanced['text_clean'] = df_balanced['text'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
# Remove samples that are empty after cleaning
df_balanced_final = df_balanced[df_balanced['text_clean'].str.strip().astype(bool)].copy()
df_balanced_final.drop(columns=['text_clean'], inplace=True)
print(f"Final dataset has {len(df_balanced_final)} samples.")

# --- 2. Data Splitting (Train/Validation/Test) ---
print("Splitting the data...")
# Split the data into training (70%), validation (15%), and test (15%) sets
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df_balanced_final['text'].tolist(),
    df_balanced_final['sentiment'].tolist(),
    test_size=0.3,
    random_state=42,
    stratify=df_balanced_final['sentiment'].tolist()
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_labels,
    test_size=0.5,
    random_state=42,
    stratify=temp_labels
)
print(f"Training: {len(train_texts)}, Validation: {len(val_texts)}, Test: {len(test_texts)}")

# --- 3. Loading Pre-trained Embeddings ---
print("Loading pre-trained embeddings (fastText)...")
embedding_file = 'data/cc.pt.300.vec'
try:
    # Load pre-trained Portuguese word vectors
    word_vectors = KeyedVectors.load_word2vec_format(embedding_file)
    embedding_dim = word_vectors.vector_size
except FileNotFoundError:
    print(f"Error: Embedding file not found at '{embedding_file}'.")
    exit()

# Vectorization and Embedding Matrix Creation
print("Creating and adapting the vectorization layer...")
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 200
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH
)
# Adapt the layer to the training text data to build the vocabulary
vectorize_layer.adapt(train_texts)

# Create an embedding matrix to use the pre-trained vectors
vocab = vectorize_layer.get_vocabulary()
word_index = dict(zip(vocab, range(len(vocab))))
embedding_matrix = np.zeros((len(vocab), embedding_dim))
for word, i in word_index.items():
    if word in word_vectors:
        embedding_matrix[i] = word_vectors[word]
print("Embedding matrix created.")

# Transformer Model Construction and Compilation
print("Building the Transformer model...")
def create_transformer_classifier(vocab_size, num_layers, d_model, num_heads, dff,
                                  max_seq_len, embedding_matrix, rate=0.1):
    """Defines and creates the Transformer-based classifier model."""
    inputs = layers.Input(shape=(max_seq_len,))

    # Embedding layer with pre-trained weights (not trainable)
    embedding_layer = layers.Embedding(
        input_dim=vocab_size, output_dim=d_model, weights=[embedding_matrix],
        trainable=False, mask_zero=True
    )
    x = embedding_layer(inputs)
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32)) # Scale the embeddings
    x = PositionalEncoding(max_seq_len, d_model)(x)
    x = layers.Dropout(rate)(x)

    # Add Transformer encoder blocks
    for _ in range(num_layers):
        x = EncoderBlock(d_model, num_heads, dff, rate)(x)

    # Global pooling to get a fixed-size vector for classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(rate)(x)
    # Output layer for binary classification
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Hyperparameters for the model
D_MODEL, NUM_LAYERS, NUM_HEADS, DFF, DROPOUT_RATE = 300, 2, 6, 512, 0.1
transformer_classifier = create_transformer_classifier(
    vocab_size=MAX_VOCAB_SIZE, num_layers=NUM_LAYERS, d_model=D_MODEL,
    num_heads=NUM_HEADS, dff=DFF, max_seq_len=MAX_SEQUENCE_LENGTH,
    embedding_matrix=embedding_matrix, rate=DROPOUT_RATE
)
# Compile the model with loss, optimizer, and metrics
transformer_classifier.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
transformer_classifier.summary()

# Model Training
print("\nStarting training...")
# Use early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
BATCH_SIZE = 32

# Create a text vectorization function for the tf.data pipeline
def vectorize_text(text, label):
    return vectorize_layer(text), label

# Create tf.data.Dataset objects for efficient training
train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels)).map(vectorize_text).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_texts, val_labels)).map(vectorize_text).batch(BATCH_SIZE)

# Train the model
history = transformer_classifier.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset,
    callbacks=[early_stopping]
)
print("Training finished.")

# Saving Artifacts
print("Saving artifacts for evaluation...")
# Create the 'models' directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the trained model
transformer_classifier.save('models/sentiment_transformer.keras')

# Save the vectorization layer's configuration and weights
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump({'config': vectorize_layer.get_config(), 'weights': vectorize_layer.get_weights()}, f)

# Save the test set for consistent evaluation
with open('models/test_data.pkl', 'wb') as f:
    pickle.dump({'texts': test_texts, 'labels': test_labels}, f)

print("Model, vectorizer, and test data saved successfully in the 'models/' folder.")