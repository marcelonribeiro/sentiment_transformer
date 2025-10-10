import mlflow
import pickle
import tensorflow as tf
import pandas as pd

from src.ml.transformer_layers import PositionalEncoding, EncoderBlock

MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 200

class SentimentTransformerModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        custom_objects = {
            'PositionalEncoding': PositionalEncoding,
            'EncoderBlock': EncoderBlock
        }

        model_path = context.artifacts["transformer_model"]
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        vocab_path = context.artifacts["vectorizer_vocab"]
        with open(vocab_path, 'rb') as f:
            vocabulary_list = pickle.load(f)

        self.vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=MAX_VOCAB_SIZE,
            output_mode='int',
            output_sequence_length=MAX_SEQUENCE_LENGTH
        )

        self.vectorize_layer.set_vocabulary(vocabulary_list)

        print("Model and vectorizer loaded successfully using the robust method.")

    def predict(self, context, model_input):
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("A entrada deve ser um DataFrame do pandas.")

        texts = model_input.iloc[:, 0].astype(str).tolist()

        vectorized_texts = self.vectorize_layer(texts)

        predictions = self.model.predict(vectorized_texts).flatten()

        return predictions