import numpy as np
import SVM_Model
from RFFTransform import RFFTransformer
from NNTransformer import RFFLayer
import tensorflow as tf



np.random.seed()

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
# X, y = load_iris(return_X_y=True)
# X = StandardScaler().fit_transform(X)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# def build_hybrid_model(input_dim, rff_dim=500, gamma=0.1):
#     inputs = tf.keras.Input(shape=(input_dim,))
#
#     x = tf.keras.layers.Dense(128)(inputs)
#     x = tf.keras.layers.PReLU()(x)
#
#     x = RFFLayer(output_dim=rff_dim, gamma=gamma)(x)
#
#     outputs = tf.keras.layers.Dense(1, activation='linear')(x)  # Linear SVM layer
#
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model
# model = build_hybrid_model(input_dim=x_train.shape[1], rff_dim=300)
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     loss=tf.keras.losses.CategoricalHinge(),
#     metrics=['accuracy']
# )
# # model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test))
#
#
# # Apply RFF
# rff = RFFTransformer(gamma=0.5, n_components=500, random_state=42)
# X_rff = rff.fit_transform(x_train)
# # x_train, x_test, y_train, y_test = train_test_split(X_rff, y, test_size=0.2)
# model = SVM_Model.SVM()
# model._one_vs_rest(X_rff, y_train)
# # res = model._one_vs_rest_predict(x_test)
# res = model._one_vs_rest_predict(X_rff)
# print(y_train)
# print(res)









# Load and reduce features
# iris = datasets.load_iris()
# X = iris.data[:, :2]  # Use first two features
# y = iris.target
#
# # Standardize
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # Train your model
# svm = SVM_Model.SVM(lr=0.01, n_iter=10, kernel='sigmoid')
# svm._one_vs_rest(X_scaled, y)
#
# # Plot decision boundaries
# x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
# y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
# xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
#                      np.linspace(y_min, y_max, 500))
#
# grid = np.c_[xx.ravel(), yy.ravel()]
# Z = svm._one_vs_rest_predict(grid)
# Z = Z.reshape(xx.shape)
#
# plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, s=40, cmap=plt.cm.coolwarm, edgecolors='k')
# plt.title("Decision Boundaries (Iris - Custom SVM)")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()






# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from Hybrid_trainer import HybridOVRClassifier
# from Hybrid_optimizer import HybridSVMTrainer
#
# # 1. Load and preprocess the dataset
# X, y = load_iris(return_X_y=True)
# X = StandardScaler().fit_transform(X)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 2. Define the model builder function
# def build_model_fn(input_dim):
#     return HybridSVMTrainer(input_dim=input_dim, rff_dim=300, gamma=0.5)
#
# # 3. Initialize and train the Hybrid OVR classifier
# hybrid_model = HybridOVRClassifier(
#     build_model_fn=build_model_fn,
#     learning_rate=0.001,
#     epochs=20,
#     batch_size=16
# )
#
# print("\n🚀 Starting training...")
# hybrid_model.fit(x_train, y_train)
#
# # 4. Perform predictions
# print("\n🔍 Testing on unseen test data:")
# y_pred = hybrid_model.predict(x_test)
#
# # 5. Evaluation
# accuracy = np.mean(y_pred == y_test)
# print(f"\n✅ Test Accuracy: {accuracy:.4f}")
# print("\n📌 True Labels:", y_test)
# print("📌 Predicted Labels:", y_pred)




from test import preprocess_sms_dataset
from Hybrid_optimizer import HybridSVMTrainer
from Hybrid_trainer import HybridOVRClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train, X_test, y_train, y_test, vectorizer = preprocess_sms_dataset('./SMSSpamCollection')

def build_model_fn(input_dim):
    return HybridSVMTrainer(input_dim=input_dim, rff_dim=500, gamma=0.2)


# Create hybrid classifier
hybrid_clf = HybridOVRClassifier(build_model_fn, learning_rate=0.001, epochs=20, batch_size=32)

# Train
hybrid_clf.fit(X_train, y_train)


# Predict
y_pred = hybrid_clf.predict(X_test)

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['ham', 'spam']))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
