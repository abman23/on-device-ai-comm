import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

'''https://keras.io/examples/generative/vq_vae/'''

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings # the number of codebooks.
        self.embedding_dim = embedding_dim # the dimension of a codebook.

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        input_shape = tf.shape(x)

        # Quantization.
        encoding_indices = self.get_code_indices(x)
        encoding_indices = tf.cast(encoding_indices, tf.int64)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, input):
        # get codebook indices for each semantic encoder output

        # flatten the inputs keeping `embedding_dim` intact.
        num_of_input_elems = np.prod(input.shape)
        assert num_of_input_elems % self.embedding_dim == 0, f"Argument 'embedding_dim' should be a factor of total input data, {num_of_input_elems}."
        flattened_inputs = tf.reshape(input, [-1, self.embedding_dim])

        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        encoding_indices = tf.cast(tf.convert_to_tensor(encoding_indices), tf.float32)

        return encoding_indices
    
    def reconstruct_with_indices(self, indices):
        indices = tf.cast(indices, tf.int64)

        encodings = tf.one_hot(indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        return quantized
    
    def handle_invalid_values(self, output):
        ### If invalid outputs exist, convert them into closest valid outputs.
        # change all negative values to 0
        output = tf.maximum(output, 0)
        # change bigger than num_embeddings-1 values to num_embeddings-1 (They are indices, so it should be -1)
        output = tf.minimum(output, self.num_embeddings-1)
        # round values
        output = tf.round(output)

        return output