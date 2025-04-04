import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
df = pd.read_csv("c.csv")  # Use your cleaned file name
print(df.columns)

# Fill missing values
df.dropna(subset=['Cleaned_Text', 'Category'], inplace=True)

# Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\s{2,}', ' ', text)  # Remove extra whitespace
    return text.strip()

# Apply cleaning
df['Cleaned_Text'] = df['Cleaned_Text'].apply(clean_text)

# Split the dataset
X = df['Cleaned_Text']
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model Training
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Prediction & Evaluation
y_pred = clf.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(clf, 'resume_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
