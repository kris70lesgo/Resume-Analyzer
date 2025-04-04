import pandas as pd
import re

# Step 1: Load the CSV file (Set low_memory=False to avoid dtype warning)
df = pd.read_csv("Dataset.csv", low_memory=False)

# Step 2: Define a text cleaning function
def clean_text(text):
    text = str(text).lower()  # convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r'\W+', ' ', text)  # remove special characters
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text.strip()

# Step 3: Apply cleaning to the 'Text' column
df['Cleaned_Text'] = df['Text'].apply(clean_text)

# Step 4: Save the cleaned data to a new CSV
df[['Category', 'Cleaned_Text']].to_csv("Cleaned_Resumes.csv", index=False)

print("✅ Resume text cleaned and saved to Cleaned_Resumes.csv")
