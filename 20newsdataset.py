from Hybrid_optimizer import HybridSVMTrainer
from Hybrid_trainer import HybridOVRClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.preprocessing import StandardScaler

# Load dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
texts = newsgroups.data
labels = newsgroups.target

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text

texts_clean = [clean_text(t) for t in texts]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(texts_clean).toarray()

# Label encoding
le = LabelEncoder()
y = le.fit_transform(labels)

# Normalization, using standard scaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

X, y = X[:1000], y[:1000]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
print("Train shape:", x_train.shape, "Test shape:", x_test.shape)


def build_model_fn(input_dim):
    return HybridSVMTrainer(input_dim=input_dim, rff_dim=1000, gamma=0.05)

# Import your HybridOVRClassifier and model builder
classifier = HybridOVRClassifier(build_model_fn, epochs=20, batch_size=32)
classifier.fit(x_train, y_train)

# Prediction and evaluation
y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
