import tensorflow as tf
from NNTransformer import RFFLayer
from RFFTransform import RFFTransformer
class HybridSVMTrainer(tf.keras.Model):
    def __init__(self, input_dim, rff_dim=500, gamma=0.1):
        super(HybridSVMTrainer, self).__init__()
        # self.dense = tf.keras.layers.Dense(128)
        # self.prelu = tf.keras.layers.PReLU()
        self.feature_net = tf.keras.Sequential([
            tf.keras.layers.Dense(256),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(128),
            tf.keras.layers.ReLU()
        ])
        self.rff = RFFLayer(output_dim=rff_dim, gamma=gamma, trainable=True)
        self.svm = tf.keras.layers.Dense(1, activation='linear')  # Linear output

    def call(self, inputs):
        # x = self.dense(inputs)
        # x = self.prelu(x)
        x = self.feature_net(inputs)
        x = self.rff(x)
        return self.svm(x)

    def compute_svm_loss(self, y_true, y_pred):
        # Ensure labels are -1 or 1
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.where(y_true <= 0, -1.0, 1.0)
        hinge = tf.maximum(0.0, 1 - y_true * tf.squeeze(y_pred))
        return tf.reduce_mean(hinge)

def train_hybrid_model(model, x_train, y_train, epochs=20, batch_size=32, first_learning_rate=0.001, second_learning_rate=0.01):
    optimizer_nn = tf.keras.optimizers.Adam(first_learning_rate)
    optimizer_svm = tf.keras.optimizers.SGD(second_learning_rate)  # Can also customize manually
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataset:
            with tf.GradientTape(persistent=True) as tape:
                predictions = model(x_batch, training=True)
                loss = model.compute_svm_loss(y_batch, predictions)

            # Gradients
            grads = tape.gradient(loss, model.trainable_weights)

            # Apply gradients separately (optional: split for NN and SVM)
            optimizer_nn.apply_gradients(zip(grads[:-2], model.trainable_weights[:-2]))  # For NN layers
            optimizer_svm.apply_gradients(zip(grads[-2:], model.trainable_weights[-2:]))  # For SVM weights

            total_loss += loss.numpy()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
