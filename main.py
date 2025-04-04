import pickle
import pandas as pd

# Load the trained model and vectorizer
with open("resume_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load resumes from CSV
df = pd.read_csv("resume.csv")  # Ensure the CSV has a column named 'Text'

if "Text" not in df.columns:
    raise ValueError("CSV file must contain a column named 'Text' with resume content.")

# Transform the text data
text_vectors = vectorizer.transform(df["Text"])  

# Predict categories
df["Predicted_Category"] = model.predict(text_vectors)

# Save the results
df.to_csv("resume_predictions.csv", index=False)

print("Predictions saved to resume_predictions.csv")
