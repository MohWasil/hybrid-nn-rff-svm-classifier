# # hybrid_model.py (Updated Version with Improvements)
#
# import numpy as np
# import tensorflow as tf
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import TruncatedSVD
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import fetch_20newsgroups
# from RFFTransform import RFFTransformer
#
# # -----------------------------
# # Data Preprocessing Pipeline
# # -----------------------------
# def preprocess_data():
#     newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
#     x_raw, y = newsgroups.data, newsgroups.target
#
#     vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
#     svd = TruncatedSVD(n_components=300, random_state=42)
#     scaler = StandardScaler()
#
#     pipeline = make_pipeline(vectorizer, svd, scaler)
#     x_processed = pipeline.fit_transform(x_raw)
#
#     return train_test_split(x_processed, y, test_size=0.2, random_state=42)
#
# # -----------------------------
# # RFF Layer (Trainable)
# # -----------------------------
# class RFFLayer(tf.keras.layers.Layer):
#     def __init__(self, output_dim, gamma=0.1):
#         super(RFFLayer, self).__init__()
#         self.output_dim = output_dim
#         self.gamma = gamma
#
#     def build(self, input_shape):
#         self.w = self.add_weight(shape=(input_shape[-1], self.output_dim),
#                                  initializer=tf.keras.initializers.RandomNormal(stddev=np.sqrt(2 * self.gamma)),
#                                  trainable=True)
#         self.b = self.add_weight(shape=(self.output_dim,),
#                                  initializer='uniform',
#                                  trainable=True)
#
#     def call(self, x):
#         projection = tf.matmul(x, self.w) + self.b
#         return tf.sqrt(2.0 / self.output_dim) * tf.math.cos(projection)
#
# # -----------------------------
# # Hybrid Model without advance method
# # -----------------------------
# class HybridModel(tf.keras.Model):
#     def __init__(self, input_dim, rff_dim=1000, gamma=0.1):
#         super(HybridModel, self).__init__()
#         self.feature_net = tf.keras.Sequential([
#             tf.keras.layers.Dense(256),
#             # tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.LayerNormalization(),
#             tf.keras.layers.ReLU(),
#             tf.keras.layers.Dropout(0.1),
#             tf.keras.layers.Dense(128),
#             tf.keras.layers.ReLU()
#         ])
#         self.rff = RFFLayer(output_dim=rff_dim, gamma=gamma)
#         self.output_layer = tf.keras.layers.Dense(20, activation='softmax')
#         self.rfft = RFFTransformer(random_state=42)
#
#     def call(self, inputs):
#         x = self.feature_net(inputs)
#         x = self.rff(x)
#         # x = self.rfft.fit_transform(x)
#         return self.output_layer(x)
#
#
# # -----------------------------
# # Training Function
# # -----------------------------
# def train_model():
#     x_train, x_test, y_train, y_test = preprocess_data()
#
#     model = HybridModel(input_dim=x_train.shape[1])
#     model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))
#
#     # from Hybrid_optimizer import HybridSVMTrainer
#     # from Hybrid_trainer import HybridOVRClassifier
#     #
#     # def build_model_fn(input_dim):
#     #     return HybridSVMTrainer(input_dim=input_dim, rff_dim=1000, gamma=0.05)
#     #
#     # # Import your HybridOVRClassifier and model builder
#     # classifier = HybridOVRClassifier(build_model_fn, epochs=15, batch_size=32)
#     # classifier.fit(x_train[:1000], y_train[:1000])
#     #
#     # # Prediction and evaluation
#     # y_pred = classifier.predict(x_test)
#     #
#     # from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
#     # print("Accuracy:", accuracy_score(y_test, y_pred))
#     # print("Classification Report:\n", classification_report(y_test, y_pred))
#     # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#
#
#
# if __name__ == "__main__":
#     train_model()








# import numpy as np
# import tensorflow as tf
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from tensorflow.keras.layers import Dense, PReLU, Layer
# from tensorflow.keras.models import Model
#
# # --------------------------
# # RFF Layer Definition
# # --------------------------
# class RFFLayer(Layer):
#     def __init__(self, output_dim, gamma=0.1):
#         super().__init__()
#         self.output_dim = output_dim
#         self.gamma = gamma
#
#     def build(self, input_shape):
#         self.W = self.add_weight(
#             name='W',
#             shape=(input_shape[-1], self.output_dim),
#             initializer=tf.random_normal_initializer(stddev=np.sqrt(2 * self.gamma)),
#             trainable=False
#         )
#         self.b = self.add_weight(
#             name='b',
#             shape=(self.output_dim,),
#             initializer=tf.random_uniform_initializer(0, 2 * np.pi),
#             trainable=False
#         )
#
#     def call(self, inputs):
#         projection = tf.linalg.matmul(inputs, self.W) + self.b
#         return tf.sqrt(2.0 / self.output_dim) * tf.cos(projection)
#
# # --------------------------
# # Hybrid Model Definition
# # --------------------------
# class HybridModel(Model):
#     def __init__(self, input_dim, rff_dim=500, gamma=0.1, l2_reg=1e-4):
#         super().__init__()
#         self.dense = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
#         self.prelu = PReLU()
#         self.rff = RFFLayer(rff_dim, gamma=gamma)
#         self.svm = Dense(1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
#
#     def call(self, inputs):
#         x = self.dense(inputs)
#         x = self.prelu(x)
#         x = self.rff(x)
#         return self.svm(x)
#
#     def compute_svm_loss(self, y_true, y_pred, regularization_loss):
#         y_true = tf.cast(tf.where(y_true <= 0, -1.0, 1.0), tf.float32)
#         hinge = tf.maximum(0.0, 1 - y_true * tf.squeeze(y_pred))
#         return tf.reduce_mean(hinge) + regularization_loss
#
# # --------------------------
# # Training Function
# # --------------------------
# def train_hybrid_model(model, x_train, y_train, epochs=20, batch_size=32, lr_nn=0.001, lr_svm=0.01):
#     optimizer_nn = tf.keras.optimizers.Adam(lr_nn)
#     optimizer_svm = tf.keras.optimizers.Adam(lr_svm)
#     dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
#
#     for epoch in range(epochs):
#         total_loss = 0
#         for x_batch, y_batch in dataset:
#             with tf.GradientTape(persistent=True) as tape:
#                 predictions = model(x_batch, training=True)
#                 loss = model.compute_svm_loss(y_batch, predictions, sum(model.losses))
#
#             grads = tape.gradient(loss, model.trainable_weights)
#             optimizer_nn.apply_gradients(zip(grads[:-2], model.trainable_weights[:-2]))
#             optimizer_svm.apply_gradients(zip(grads[-2:], model.trainable_weights[-2:]))
#
#             total_loss += loss.numpy()
#
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
#
# # --------------------------
# # Preprocessing Function
# # --------------------------
# def preprocess_20newsgroups(n_components=300):
#     newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
#     x_raw = newsgroups.data
#     y = newsgroups.target
#
#     vectorizer = TfidfVectorizer(max_features=20000, stop_words='english')
#     x_tfidf = vectorizer.fit_transform(x_raw)
#
#     svd = TruncatedSVD(n_components=n_components, random_state=42)
#     x_svd = svd.fit_transform(x_tfidf)
#
#     scaler = StandardScaler()
#     x_scaled = scaler.fit_transform(x_svd)
#
#     return train_test_split(x_scaled, y, test_size=0.2, random_state=42)
#
# # --------------------------
# # Main Execution
# # --------------------------
# if __name__ == "__main__":
#     x_train, x_test, y_train, y_test = preprocess_20newsgroups()
#
#     # One-vs-Rest Training
#     classes = np.unique(y_train)
#     models = []
#     for cls in classes:
#         print(f"\nTraining for class {cls} vs rest")
#         y_train_bin = np.where(y_train == cls, 1.0, -1.0)
#         y_test_bin = np.where(y_test == cls, 1.0, -1.0)
#
#         model = HybridModel(input_dim=x_train.shape[1], rff_dim=500, gamma=0.2)
#         train_hybrid_model(model, x_train, y_train_bin, epochs=20, batch_size=32)
#
#         models.append(model)
#
#     # Prediction
#     print("\nEvaluating on test set...")
#     test_preds = []
#     for model in models:
#         pred = model(x_test, training=False)
#         test_preds.append(tf.squeeze(pred, axis=-1).numpy())
#
#     test_preds = np.stack(test_preds, axis=1)
#     y_pred = np.argmax(test_preds, axis=1)
#
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print("Classification Report:\n", classification_report(y_test, y_pred))

















import re
import numpy as np
import tensorflow as tf

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1) DATA LOADING & CLEANING
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)            # remove URLs
    text = re.sub(r'[^a-z0-9\s]', ' ', text)        # remove non-alphanumeric
    text = re.sub(r'\s+', ' ', text).strip()        # collapse spaces
    return text

raw = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
texts = [clean_text(t) for t in raw.data]
labels = raw.target
n_classes = len(raw.target_names)

# 2) TF-IDF → SVD → SCALING PIPELINE
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_tfidf = vectorizer.fit_transform(texts)

svd    = TruncatedSVD(n_components=300, random_state=42)
X_svd  = svd.fit_transform(X_tfidf)

scaler = StandardScaler()
X_proc = scaler.fit_transform(X_svd)

# train/val/test split
X_temp, X_test,  y_temp, y_test  = train_test_split(
    X_proc, labels, test_size=0.2, random_state=42, stratify=labels
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

print(f"Shapes → train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

# 3) DEFINE & TRAIN FEATURE EMBEDDING NETWORK
class FeatureNet(tf.keras.Model):
    def __init__(self, embed_dim=128, dropout_rate=0.5):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        self.do1 = tf.keras.layers.Dropout(dropout_rate)
        self.fc2 = tf.keras.layers.Dense(embed_dim, activation=None)  # embeddings

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.do1(x, training=training)
        return self.fc2(x)

# build classifier on top of embeddings
inputs = tf.keras.Input(shape=(X_train.shape[1],), name="tfidf_input")
emb    = FeatureNet(embed_dim=128)(inputs)
logits = tf.keras.layers.Dense(n_classes, activation='softmax')(emb)
nn_model = tf.keras.Model(inputs, logits, name="feature_classifier")

nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train
from sklearn.utils import class_weight
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
nn_model.fit(X_train, y_train, class_weight=dict(enumerate(weights)), epochs=10)

# 4) EXTRACT EMBEDDINGS
feature_extractor = tf.keras.Model(inputs, emb, name="feature_extractor")
# … (earlier code that trains nn_model and builds feature_extractor)

# 1) Extract embeddings
X_emb_train = feature_extractor.predict(X_train, batch_size=64)
X_emb_test  = feature_extractor.predict(X_test,  batch_size=64)

# 2) Balance via SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_emb_res, y_res = sm.fit_resample(X_emb_train, y_train)

# 3) Train an RBF SVM
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_emb_res, y_res)

# 4) Evaluate
y_pred = svm.predict(X_emb_test)
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy (balanced):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=raw.target_names))
