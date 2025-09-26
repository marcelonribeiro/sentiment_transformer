import os
from gensim.models import KeyedVectors

# Path to the files
# Change if your data folder has a different name
DATA_FOLDER = 'data'
TEXT_EMBEDDING_FILE = os.path.join(DATA_FOLDER, 'cc.pt.300.vec')
BINARY_EMBEDDING_FILE = os.path.join(DATA_FOLDER, 'cc.pt.300.bin')

print("Starting conversion (this may take several minutes)...")

# 1. Load the text file (the slow part)
word_vectors = KeyedVectors.load_word2vec_format(TEXT_EMBEDDING_FILE)

# 2. Save the vectors in Gensim's native binary format (fast)
word_vectors.save(BINARY_EMBEDDING_FILE)

print(f"Conversion complete! Binary file saved at: '{BINARY_EMBEDDING_FILE}'")