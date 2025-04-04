### 📄 Resume Classifier

This project classifies resumes into different job categories using machine learning. The model is trained using TF-IDF vectorization and a classification algorithm.  

---

## 🚀 Setup

### 1️⃣ Install Dependencies
```bash
pip install pandas scikit-learn nltk
```

### 2️⃣ Prepare Data
- Place your `resume.csv` file in the project directory.  
- Ensure it has the following columns:  
  - `Category` → The job category  
  - `Text` → Resume text  

### 3️⃣ Train the Model
Run the training script to generate model files:  
```bash
python model.py
```
This will create:  
✅ `resume_classifier.pkl` → Trained classifier  
✅ `vectorizer.pkl` → TF-IDF vectorizer  

### 4️⃣ Test the Model
After training, test the model with `main.py`:  
```bash
python main.py
```
It will predict categories for given resume inputs.  

---

## 📂 Project Structure

```
/test
️│── model.py               # Training script  
️│── main.py                # Testing script  
️│── resume.csv             # Resume dataset  
️│── vectorizer.pkl         # Saved TF-IDF vectorizer  
️│── resume_classifier.pkl  # Trained model  
️│── README.md              # Project documentation  
```

---

## ⚡ Example Usage
Modify `main.py` to test a single resume:
```python
import pickle

# Load model
with open("vectorizer.pkl", "rb") as v_file, open("resume_classifier.pkl", "rb") as m_file:
    vectorizer = pickle.load(v_file)
    model = pickle.load(m_file)

resume_text = "Experienced Python developer with knowledge of AI and ML."
resume_vectorized = vectorizer.transform([resume_text])
prediction = model.predict(resume_vectorized)

print("Predicted Category:", prediction[0])
```

---

## 🛠 Troubleshooting

### ❌ `UnpicklingError: invalid load key`
🔹 Ensure `vectorizer.pkl` and `resume_classifier.pkl` were saved properly.  
🔹 Re-run `model.py` to regenerate the files.  

---

