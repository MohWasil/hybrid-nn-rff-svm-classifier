import tensorflow as tf
import numpy as np

class RFFLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, gamma=0.1, trainable=False, name="rff_layer"):
        super(RFFLayer, self).__init__(name=name)
        self.output_dim = output_dim
        self.gamma = gamma
        self._trainable = trainable  # Save externally passed value

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.W = self.add_weight(
            name="W",
            shape=(input_dim, self.output_dim),
            initializer=tf.random_normal_initializer(stddev=np.sqrt(2 * self.gamma)),
            trainable=self._trainable  # Use self._trainable here
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.output_dim,),
            initializer=tf.random_uniform_initializer(0, 2 * np.pi),
            trainable=self._trainable
        )

    def call(self, inputs):
        projection = tf.matmul(inputs, self.W) + self.b
        return tf.sqrt(2.0 / self.output_dim) * tf.cos(projection)
