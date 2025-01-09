import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Extract text from PDF
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

# Preprocess text
def preprocess_text(text):
    return text.lower()

# Generate embeddings
def generate_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings

# Load reference papers
def load_reference_papers(reference_dir):
    reference_texts = {"Publishable": {}, "Non-Publishable": []}
    for category in ["Publishable", "Non-Publishable"]:
        category_path = os.path.join(reference_dir, category)
        if category == "Publishable":
            for conference in os.listdir(category_path):
                conference_path = os.path.join(category_path, conference)
                if os.path.isdir(conference_path):
                    reference_texts["Publishable"][conference] = []
                    for file in os.listdir(conference_path):
                        if file.endswith(".pdf"):
                            file_path = os.path.join(conference_path, file)
                            text = extract_text_from_pdf(file_path)
                            reference_texts["Publishable"][conference].append(text)
        else:
            for file in os.listdir(category_path):
                if file.endswith(".pdf"):
                    file_path = os.path.join(category_path, file)
                    text = extract_text_from_pdf(file_path)
                    reference_texts["Non-Publishable"].append(text)
    return reference_texts

# Calculate cosine similarity
def calculate_similarity(paper_embedding, reference_embeddings):
    return cosine_similarity([paper_embedding], reference_embeddings)[0]
