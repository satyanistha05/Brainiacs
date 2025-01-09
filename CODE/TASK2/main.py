import os
import pandas as pd
import joblib  # For loading the model and vectorizer
from utils import load_reference_papers, extract_text_from_pdf, preprocess_text, generate_embeddings, calculate_similarity
from sentence_transformers import SentenceTransformer

# Paths
REFERENCE_DIR = "/app/data/reference_papers"
TESTING_DIR = "/app/data/testing_papers"
RESULTS_FILE = "/app/results/results.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
TASK1_MODEL_PATH = "/app/model/publishability_model.pkl"  # Task 1 model path
VECTORIZER_PATH = "/app/model/vectorizer.pkl"             # Task 1 vectorizer path

# Load Task 1 model and vectorizer
print("Loading Task 1 model and vectorizer...")
task1_model = joblib.load(TASK1_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Load reference papers for Task 2
print("Loading reference papers for Task 2...")
references = load_reference_papers(REFERENCE_DIR)

# Generate embeddings for reference papers
print("Generating embeddings for reference papers...")
model = SentenceTransformer(MODEL_NAME)

publishable_embeddings = {}
non_publishable_embeddings = model.encode(references["Non-Publishable"])

for conference, papers in references["Publishable"].items():
    publishable_embeddings[conference] = model.encode(papers)

# Process testing papers
print("Processing testing papers...")
results = []
for file in os.listdir(TESTING_DIR):
    if file.endswith(".pdf"):
        file_path = os.path.join(TESTING_DIR, file)
        text = preprocess_text(extract_text_from_pdf(file_path))

        # Task 1: Use vectorizer and model to classify paper
        task1_features = vectorizer.transform([text])  # Use the vectorizer for feature extraction
        publishable = task1_model.predict(task1_features)[0]  # Output: 1 or 0

        if publishable == 0:  # Non-Publishable paper
            results.append([file, 0, "na", "na"])
            continue

        # Task 2: Recommend conference for Publishable papers
        paper_embedding = model.encode([text])[0]

        # Check similarity with Non-Publishable references
        non_publishable_similarities = calculate_similarity(paper_embedding, non_publishable_embeddings)
        if max(non_publishable_similarities) > 0.8:  # Threshold for Non-Publishable
            results.append([file, 0, "na", "na"])
            continue

        # Check similarity with Publishable references
        best_match = None
        best_similarity = 0
        for conference, conf_embeddings in publishable_embeddings.items():
            similarities = calculate_similarity(paper_embedding, conf_embeddings)
            max_similarity = max(similarities)
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_match = conference

        # Generate recommendation and rationale
        if best_match:
            rationale = (
                f"The paper aligns with {best_match} due to its similarity with reference papers in that conference. "
                f"Similarity score: {best_similarity:.2f}."
            )
            results.append([file, 1, best_match, rationale])
        else:
            results.append([file, 0, "na", "The paper does not align with any reference conference."])

# Save results
print("Saving results...")
os.makedirs("/app/results", exist_ok=True)
results_df = pd.DataFrame(results, columns=["Paper ID", "Publishable", "Conference", "Rationale"])
results_df.to_csv(RESULTS_FILE, index=False)
print(f"Results saved to {RESULTS_FILE}")
