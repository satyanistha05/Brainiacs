import os
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from utils import (
    load_data_and_labels,
    save_model_and_vectorizer,
    load_model_and_vectorizer,
    extract_text_from_pdf,
    preprocess_text,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Base directory
DATA_DIR = os.path.join(BASE_DIR, "data")  # Data directory
PAPERS_DIR = os.path.join(BASE_DIR, "papers")  # Papers directory
MODEL_PATH = os.path.join(BASE_DIR, "model", "publishability_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

# TRAINING PHASE
def train_model():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found at {DATA_DIR}. Please ensure the path is correct.")

    logging.info("Loading and preprocessing data...")
    papers, labels = load_data_and_labels(DATA_DIR)

    if not papers:
        raise ValueError("No data loaded. Please ensure the data directory contains valid text files.")

    logging.info("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(papers).toarray()

    logging.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    logging.info("Training the model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    logging.info("Evaluating the model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    logging.info("Saving the model and vectorizer...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    save_model_and_vectorizer(model, vectorizer, MODEL_PATH, VECTORIZER_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")
    logging.info(f"Vectorizer saved to {VECTORIZER_PATH}")

# TESTING PHASE
def classify_papers():
    if not os.path.exists(PAPERS_DIR):
        raise FileNotFoundError(f"Papers directory not found at {PAPERS_DIR}. Please ensure the path is correct.")

    logging.info("Loading the trained model and vectorizer...")
    model, vectorizer = load_model_and_vectorizer(MODEL_PATH, VECTORIZER_PATH)

    if not model or not vectorizer:
        raise ValueError("Failed to load the model or vectorizer. Please ensure the model and vectorizer files exist.")

    logging.info("Classifying papers...")
    results = []
    for file in os.listdir(PAPERS_DIR):
        if file.endswith(".pdf"):
            paper_path = os.path.join(PAPERS_DIR, file)
            logging.info(f"Processing: {file}")

            paper_text = extract_text_from_pdf(paper_path)
            if not paper_text:
                logging.warning(f"No text extracted from {file}. Skipping...")
                continue

            preprocessed_text = preprocess_text(paper_text)
            prediction = model.predict(vectorizer.transform([preprocessed_text]))[0]
            results.append((file, "Publishable" if prediction == 1 else "Non-Publishable"))

    logging.info("\nClassification Results:")
    for file, classification in results:
        print(f"{file}: {classification}")

# Main workflow
if __name__ == "__main__":
    logging.info("=== Training Phase ===")
    train_model()

    logging.info("\n=== Testing Phase ===")
    classify_papers()
