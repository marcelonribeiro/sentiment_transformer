import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class PositionalEncoding(layers.Layer):
    """
    Injects positional information into the input embeddings.
    Since the model contains no recurrence or convolution, it needs a way to understand
    the order of the sequence. This layer adds a unique positional encoding vector
    to each embedding.
    """
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'position': self.position,
            'd_model': self.d_model,
        })
        return config

    def get_angles(self, position, i, d_model):
        """Calculates the angle rates for the positional encoding formula."""
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        """
        Generates the positional encoding matrix.
        Uses a formula with sine and cosine functions of different frequencies.
        """
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        # Apply sine to even indices in the array
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cosine to odd indices in the array
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        """Adds the positional encoding to the input tensor."""
        if isinstance(inputs, tf.SparseTensor):
            dense_inputs = tf.sparse.to_dense(inputs)
        elif isinstance(inputs, tf.RaggedTensor):
            dense_inputs = inputs.to_tensor()
        else:
            dense_inputs = inputs

        seq_len = tf.shape(dense_inputs)[1]
        return dense_inputs + self.pos_encoding[:, :seq_len, :]


def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculates the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type (padding or look-ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k).

    Returns:
      output, attention_weights
    """
    # Matrix multiplication between query and key
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # Scale matmul_qk by the square root of the depth
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Add the mask to the scaled tensor (if a mask is provided)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9) # -1e9 is a large negative number

    # Softmax is applied on the last axis (seq_len_k) to get the attention weights
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiply the attention weights by the value vector
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(layers.Layer):
    """
    Performs multi-head attention.
    This layer splits the query, key, and value into multiple heads, applies scaled
    dot-product attention independently on each head, and then concatenates the results.
    """

    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

    def build(self, input_shape):
        self.wq = layers.Dense(self.d_model)
        self.wk = layers.Dense(self.d_model)
        self.wv = layers.Dense(self.d_model)
        self.dense = layers.Dense(self.d_model)
        super(MultiHeadAttention, self).build(input_shape)

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
        })
        return config

    def split_heads(self, x, batch_size):
        """
        Splits the last dimension into (num_heads, depth).
        Transposes the result to be of shape (batch_size, num_heads, seq_len, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        # Pass inputs through their respective dense layers
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Split into multiple heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Apply scaled dot-product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # Concatenate the heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # Pass through the final dense layer
        output = self.dense(concat_attention)
        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    """
    Creates a point-wise feed-forward network.
    This consists of two dense layers with a ReLU activation in between.
    """
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        layers.Dense(d_model)                  # (batch_size, seq_len, d_model)
    ])

class EncoderBlock(layers.Layer):
    """
    Represents one block of the Transformer encoder.
    It consists of a multi-head attention sub-layer and a point-wise feed-forward
    network sub-layer. Each sub-layer has a residual connection followed by layer normalization.
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

    def build(self, input_shape):
        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
        self.ffn = point_wise_feed_forward_network(self.d_model, self.dff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)
        super(EncoderBlock, self).build(input_shape)

    def get_config(self):
        config = super(EncoderBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate,
        })
        return config

    def call(self, x, training=None, mask=None):
        # --- First Sub-layer: Multi-Head Attention ---
        # Self-attention: Q, K, and V are all from the same input 'x'
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # Residual connection and layer normalization
        out1 = self.layernorm1(x + attn_output)

        # --- Second Sub-layer: Point-wise Feed-Forward Network ---
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Residual connection and layer normalization
        out2 = self.layernorm2(out1 + ffn_output)

        return out2