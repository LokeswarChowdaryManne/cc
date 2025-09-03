import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

def basic_preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = {'the', 'a', 'is', 'in', 'of', 'and', 'to', 'it', 'for', 'that'}
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

def read_and_preprocess_documents(directory_path):
    documents_list = []
    doc_id_counter = 1

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents_list.append({
                    'doc_id': doc_id_counter,
                    'filename': filename,
                    'text': content,
                    'processed_text': basic_preprocess_text(content)
                })
                doc_id_counter += 1

    return pd.DataFrame(documents_list)

def build_and_query_bim(documents_df, query_text):
    original_docs = documents_df['text'].tolist()
    tokenized_docs = [doc.split() for doc in documents_df['processed_text']]
    processed_query = basic_preprocess_text(query_text)
    query_tokens = set(processed_query.split())

    # Vocabulary
    vocabulary = set(token for doc in tokenized_docs for token in doc)
    N = len(tokenized_docs)
    bim_scores = np.zeros(N)

    # Score Calculation
    for term in query_tokens:
        if term not in vocabulary:
            continue
        n_t = sum(term in doc for doc in tokenized_docs)
        weight = math.log((N - n_t + 0.5) / (n_t + 0.5)) if n_t < N else 0
        for i, doc in enumerate(tokenized_docs):
            if term in doc:
                bim_scores[i] += weight

    results = sorted(
        zip(bim_scores, documents_df['doc_id'], original_docs),
        key=lambda x: x[0],
        reverse=True
    )
    return pd.DataFrame(results, columns=['bim_score', 'doc_id', 'text'])

def calculate_precision_at_k(results_df, relevant_doc_ids, k):
    if k <= 0:
        return 0.0
    top_k_docs = results_df.head(k)['doc_id']
    retrieved_relevant = sum(doc_id in relevant_doc_ids for doc_id in top_k_docs)
    return retrieved_relevant / k

def calculate_pr_curve_points(results_df, relevant_doc_ids):
    total_relevant = len(relevant_doc_ids)
    if total_relevant == 0:
        return [(0.0, 1.0)]

    retrieved_count = 0
    relevant_retrieved = 0
    pr_points = [(0.0, 1.0)]

    for _, row in results_df.iterrows():
        retrieved_count += 1
        if row['doc_id'] in relevant_doc_ids:
            relevant_retrieved += 1
        precision = relevant_retrieved / retrieved_count
        recall = relevant_retrieved / total_relevant
        pr_points.append((recall, precision))

    if pr_points[-1][0] < 1.0:
        pr_points.append((1.0, pr_points[-1][1]))

    return pr_points

if __name__ == "__main__":
    documents_directory = 'documents'  # Path to your directory with txt files
    documents_df = read_and_preprocess_documents(documents_directory)

    print("\nFinal DataFrame with Processed Text:")
    print(documents_df)

    user_query_bim = input("Enter your search query for BIM: ")
    bim_results = build_and_query_bim(documents_df, user_query_bim)

    print(f"\nBIM Results for query: '{user_query_bim}'")
    print(bim_results)

    relevant_docs_for_query = {2, 3}
    print(f"\nAssuming relevant documents for a query are: {relevant_docs_for_query}\n")

    if not bim_results.empty:
        print("--- Evaluating BIM Results ---")
        for k in [1, 2, 3]:
            precision_at_k_bim = calculate_precision_at_k(bim_results, relevant_docs_for_query, k)
            print(f"Precision @ {k} (BIM): {precision_at_k_bim:.2f}")

        pr_points_bim = calculate_pr_curve_points(bim_results, relevant_docs_for_query)
        print("\nPR Curve Points (BIM):", pr_points_bim)

        plt.figure()
        plt.plot([p[0] for p in pr_points_bim], [p[1] for p in pr_points_bim], marker='o')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (BIM)')
        plt.grid(True)
        plt.show()
    else:
        print("No BIM results found. Please check the query/documents.")
