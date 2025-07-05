import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def preprocess_sms_dataset(filepath='SMSSpamCollection'):
    # Step 1: Read as tab-separated file
    df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'text'])

    # Step 2: Clean the text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()

    df['text'] = df['text'].apply(clean_text)

    # Step 3: Encode labels: ham=0, spam=1
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    # Step 4: Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
    )

    # Step 5: Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    return X_train_vec, X_test_vec, y_train.values, y_test.values, vectorizer
