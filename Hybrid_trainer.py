import numpy as np
import tensorflow as tf
from Hybrid_optimizer import train_hybrid_model
class HybridOVRClassifier:
    def __init__(self, build_model_fn, learning_rate=0.001, epochs=20, batch_size=32):
        self.build_model_fn = build_model_fn
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.models = []
        self.class_labels = []

    def fit(self, x_train, y_train):
        self.class_labels = np.unique(y_train)
        self.models = []

        for label in self.class_labels:
            print(f"\n[Training for class {label} vs rest]")
            y_binary = np.where(y_train == label, 1.0, -1.0)

            model = self.build_model_fn(input_dim=x_train.shape[1])
            train_hybrid_model(model, x_train, y_binary, epochs=self.epochs, batch_size=self.batch_size,
                               first_learning_rate=self.learning_rate)

            # optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            #
            # for epoch in range(self.epochs):
            #     indices = np.random.permutation(len(x_train))
            #     x_train_shuffled = x_train[indices]
            #     y_train_shuffled = y_binary[indices]
            #
            #     for i in range(0, len(x_train), self.batch_size):
            #         x_batch = x_train_shuffled[i:i + self.batch_size]
            #         y_batch = y_train_shuffled[i:i + self.batch_size]
            #
            #         with tf.GradientTape() as tape:
            #             logits = model(x_batch, training=True)
            #             logits = tf.squeeze(logits, axis=-1)
            #             loss = tf.reduce_mean(tf.maximum(0., 1. - y_batch * logits))
            #
            #         grads = tape.gradient(loss, model.trainable_variables)
            #         optimizer.apply_gradients(zip(grads, model.trainable_variables))

            self.models.append(model)

    def predict(self, x):
        predictions = []

        for model in self.models:
            logits = model(x, training=False)
            predictions.append(tf.squeeze(logits, axis=-1))

        # Stack all classifier scores and choose the class with highest score
        predictions = tf.stack(predictions, axis=1)
        return tf.argmax(predictions, axis=1).numpy()
