from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

STOP_WORDS = {'the', 'a', 'is', 'in', 'of', 'and', 'to', 'it', 'for', 'that'}

def basic_preprocess_text(text: str) -> str:
    # Lowercase, remove non-alphabetic chars, tokenize, and filter stopwords.
    text = re.sub(r'[^a-z\s]', '', text.lower())
    return ' '.join(word for word in text.split() if word not in STOP_WORDS)

def read_and_preprocess_documents(directory_path: str) -> pd.DataFrame:
    # Reads all .txt files, assigns IDs, and applies preprocessing.
    documents_list = []
    for idx, filename in enumerate(os.listdir(directory_path), start=1):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as f:
                content = f.read()
            documents_list.append({
                'doc_id': idx,
                'filename': filename,
                'text': content,
                'processed_text': basic_preprocess_text(content)
            })

    return pd.DataFrame(documents_list)

def build_vectorizer_and_matrix(processed_texts: list[str]):
    # Builds vectorizer and TD matrix (binary TF, no IDF).
    vectorizer = TfidfVectorizer(binary=True, use_idf=False)
    td_matrix = vectorizer.fit_transform(processed_texts)
    return vectorizer, td_matrix


def calculate_similarity_vsm_query(query_text: str, td_matrix, vectorizer, documents_df: pd.DataFrame) -> pd.DataFrame:
    # Calculate cosine similarity of query against documents.
    query_vector = vectorizer.transform([basic_preprocess_text(query_text)])
    cosine_scores = cosine_similarity(query_vector, td_matrix).flatten()
    results_df = documents_df[['doc_id', 'text']].copy()
    results_df['similarity_score'] = cosine_scores
    return results_df.sort_values(by='similarity_score', ascending=False)

def calculate_precision_at_k(results_df: pd.DataFrame, relevant_doc_ids: set, k: int) -> float:
    # Precision@K: fraction of relevant docs in top-k.
    if k <= 0:
        return 0.0
    top_k_docs = results_df.head(k)['doc_id']
    return sum(doc_id in relevant_doc_ids for doc_id in top_k_docs) / k


def calculate_pr_curve_points(results_df: pd.DataFrame, relevant_doc_ids: set) -> list[tuple[float, float]]:
    # Generate Precision-Recall curve points.
    total_relevant = len(relevant_doc_ids)
    if total_relevant == 0:
        return [(0.0, 1.0)]

    retrieved, relevant_retrieved = 0, 0
    pr_points = [(0.0, 1.0)]

    for doc_id in results_df['doc_id']:
        retrieved += 1
        if doc_id in relevant_doc_ids:
            relevant_retrieved += 1
        precision = relevant_retrieved / retrieved
        recall = relevant_retrieved / total_relevant
        pr_points.append((recall, precision))

    # Ensure recall goes to 1
    if pr_points[-1][0] < 1.0:
        pr_points.append((1.0, pr_points[-1][1]))
    return pr_points


def plot_pr_curve(pr_points, title="Precision-Recall Curve"):
    # Plot PR curve with Matplotlib.
    plt.figure()
    recalls, precisions = zip(*pr_points)
    plt.plot(recalls, precisions, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    documents_directory = 'documents'
    documents_df = read_and_preprocess_documents(documents_directory)

    print("\nFinal DataFrame with Processed Text:")
    print(documents_df)

    # Build vectorizer and TD matrix once (avoid multiple fits)
    vectorizer, td_matrix = build_vectorizer_and_matrix(documents_df['processed_text'])

    print("\nTerm-Document Matrix:")
    print(pd.DataFrame(td_matrix.toarray(), 
                       index=documents_df['doc_id'], 
                       columns=vectorizer.get_feature_names_out()))

    # Query handling
    user_query = input("\nEnter your search query for VSM: ")
    vsm_results = calculate_similarity_vsm_query(user_query, td_matrix, vectorizer, documents_df)
    print(f"\nVSM Results for query: '{user_query}'")
    print(vsm_results)

    # Evaluation
    relevant_docs_for_query = {2, 3}  # ground truth
    if not vsm_results.empty:
        print("\n--- Evaluating VSM Results ---")
        for k in [1, 2, 3]:
            print(f"Precision @ {k} (VSM): {calculate_precision_at_k(vsm_results, relevant_docs_for_query, k):.2f}")

        pr_points_vsm = calculate_pr_curve_points(vsm_results, relevant_docs_for_query)
        print("\nPR Curve Points (VSM):", pr_points_vsm)
        plot_pr_curve(pr_points_vsm)
