import os
from PyPDF2 import PdfReader
import joblib

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# Function to preprocess text
def preprocess_text(text):
    # Lowercase the text and remove unnecessary characters
    try:
        text = text.lower()  # Convert to lowercase
        text = ''.join(char for char in text if char.isalnum() or char.isspace())  # Remove special characters
        return text.strip()
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""

# Function to load labeled data from a directory
def load_data_and_labels(data_dir):
    papers = []
    labels = []
    # Expecting subdirectories like "Non-Publishable" (label 0) and "Publishable" (label 1)
    categories = ["Non-Publishable", "Publishable"]
    
    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        print(f"Checking category: {category_path}")
        
        if not os.path.exists(category_path):
            print(f"Warning: Directory not found - {category_path}")
            continue
        
        for file in os.listdir(category_path):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(category_path, file)
                print(f"Processing file: {file}")
                text = extract_text_from_pdf(pdf_path)
                if text:
                    papers.append(preprocess_text(text))
                    labels.append(label)

    if not papers:
        print("Warning: No valid data found. Please check the data directory.")
    return papers, labels

# Function to save model and vectorizer
def save_model_and_vectorizer(model, vectorizer, model_path, vectorizer_path):
    try:
        joblib.dump(model, model_path)
        print(f"Model saved at {model_path}")
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Vectorizer saved at {vectorizer_path}")
    except Exception as e:
        print(f"Error saving model or vectorizer: {e}")

# Function to load model and vectorizer
def load_model_and_vectorizer(model_path, vectorizer_path):
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("Model and vectorizer loaded successfully.")
        return model, vectorizer
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        return None, None