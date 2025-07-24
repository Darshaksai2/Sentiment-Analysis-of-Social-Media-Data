import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys

warnings.filterwarnings('ignore')

required_nltk_data = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']

print("Checking/Downloading NLTK resources...")
for resource in required_nltk_data:
    try:
        nltk.data.find(f'corpora/{resource}')
        print(f"Resource '{resource}' already present.")
    except LookupError:
        print(f"Resource '{resource}' not found. Downloading...")
        try:
            nltk.download(resource)
            print(f"Resource '{resource}' downloaded successfully.")
        except Exception as e:
            print(f"Error downloading '{resource}': {e}")
            print(f"Please try running 'python -m nltk.downloader {resource}' in your terminal.")
            sys.exit(1)

print("NLTK resources check/download complete.\n")

dataset_path = 'training.1600000.processed.noemoticon.csv'

try:
    df = pd.read_csv(dataset_path, encoding='ISO-8859-1', engine='python', header=None)
    df = df[[0, 5]]
    df.columns = ['sentiment', 'text']

    sentiment_mapping = {0: 'negative', 4: 'positive'}
    df['sentiment'] = df['sentiment'].map(sentiment_mapping)

    df = df.dropna(subset=['sentiment'])

    df = df.sample(n=10000, random_state=42).reset_index(drop=True)
    print(f"Loaded a sample of {len(df)} rows from Sentiment140 dataset.")

except FileNotFoundError:
    print(f"'{dataset_path}' not found. Using a smaller dummy dataset for demonstration.")
    print("Please download 'training.1600000.processed.noemoticon.csv' from Kaggle if you want to use the full dataset.")
    data = {
        'text': [
            "I love this product! It's absolutely amazing and works perfectly.",
            "The customer service was terrible, very disappointed with the response.",
            "This is an okay movie, nothing special, just average.",
            "Absolutely fantastic experience. Highly recommend to everyone!",
            "Never buying from them again, total waste of money and time.",
            "The delivery was on time, no issues, just standard.",
            "So happy with my new phone! It's sleek and fast.",
            "Worst day ever. Everything went wrong, truly awful.",
            "Neutral feedback here. No strong feelings either way.",
            "Great support, fixed my issue quickly. Thank you!",
            "Broken item, very frustrating experience.",
            "The weather is just fine today, nothing exciting.",
            "Thrilled with the results, exceeded expectations!",
            "What a disaster, everything failed."
        ],
        'sentiment': [
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative"
        ]
    }
    df = pd.DataFrame(data)

print("\n--- Initial Data Exploration ---")
print("First 5 rows of the dataset:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())
print("\nMissing values:")
print(df.isnull().sum())

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

print("\n--- Text Preprocessing ---")
df['cleaned_text'] = df['text'].apply(preprocess_text)

print("\nOriginal vs. Cleaned Text Samples:")
for i in range(min(5, len(df))):
    print(f"Original: {df['text'].iloc[i]}")
    print(f"Cleaned:  {df['cleaned_text'].iloc[i]}\n")

df = df[df['cleaned_text'].str.strip() != '']
if len(df) == 0:
    print("No valid text data remaining after preprocessing. Exiting.")
    sys.exit(1)

print("\n--- Feature Extraction (TF-IDF) ---")
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf_vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

print(f"Shape of TF-IDF matrix (features): {X.shape}")
print(f"Number of target labels: {len(y)}")

print("\n--- Model Training (Logistic Regression) ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
model.fit(X_train, y_train)

print("\nModel training complete.")

print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

print(f"Accuracy on test set: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("\n--- Further Improvements & Next Steps ---")
print("1. Experiment with different `max_features` and `ngram_range` for `TfidfVectorizer`.")
print("2. Try other traditional ML models: Multinomial Naive Bayes, Linear SVC (SVM).")
print("3. Implement GridSearchCV or RandomizedSearchCV for hyperparameter tuning.")
print("4. For imbalanced datasets, consider `class_weight='balanced'` in LogisticRegression or using techniques like SMOTE.")
print("5. Explore Deep Learning models (LSTMs/GRUs with Word Embeddings) for larger datasets and potentially higher accuracy.")
print("6. Integrate emoji handling or negation phrase handling for more nuanced sentiment.")

print("\n--- Predicting Sentiment for New Text ---")

def predict_sentiment(new_text):
    cleaned_text = preprocess_text(new_text)
    if not cleaned_text:
        return "Cannot classify empty text after preprocessing."
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

test_phrases = [
    "I'm so incredibly happy with this service! It's beyond expectation.",
    "This movie was a total disappointment. Absolutely boring.",
    "The restaurant was okay, nothing to complain about but nothing special either.",
    "What a horrible day, everything went wrong and I'm very upset.",
    "Received the package quickly and in perfect condition.",
    "The product broke after one use."
]

for phrase in test_phrases:
    sentiment = predict_sentiment(phrase)
    print(f"Text: '{phrase}'\nPredicted Sentiment: {sentiment}\n")