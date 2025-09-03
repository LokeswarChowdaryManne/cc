import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# --- Preprocessing ---
def preprocess(text: str) -> str:
    return re.sub(r'[^a-z\s]', '', text.lower())

def shingles(text: str, k=3):
    return {text[i:i+k] for i in range(len(text) - k + 1)}

# --- Read Documents ---
def read_documents(directory: str) -> pd.DataFrame:
    docs = []
    for idx, filename in enumerate(os.listdir(directory), start=1):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                content = f.read()
            docs.append({
                "doc_id": idx,
                "filename": filename,
                "text": content,
                "shingles": shingles(preprocess(content))
            })
    return pd.DataFrame(docs)

# --- Jaccard Similarity ---
def jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)

def rank_jaccard(query, docs_df, k=3):
    query_shingles = shingles(preprocess(query), k)
    scores = [
        jaccard(query_shingles, shingles_set) for shingles_set in docs_df["shingles"]
    ]
    results = pd.DataFrame({
        "doc_id": docs_df["doc_id"],
        "text": docs_df["text"],
        "jaccard_score": scores
    }).sort_values(by="jaccard_score", ascending=False)
    return results

# --- Evaluation ---
def precision_at_k(results_df, relevant, k):
    if k <= 0: return 0.0
    top_k = results_df.head(k)["doc_id"]
    return sum(doc in relevant for doc in top_k) / k

def pr_curve_points(results_df, relevant):
    total_rel = len(relevant)
    if total_rel == 0: return [(0.0, 1.0)]
    retrieved, rel_retrieved = 0, 0
    points = [(0.0, 1.0)]
    for _, row in results_df.iterrows():
        retrieved += 1
        if row["doc_id"] in relevant:
            rel_retrieved += 1
        precision = rel_retrieved / retrieved
        recall = rel_retrieved / total_rel
        points.append((recall, precision))
    if points[-1][0] < 1.0:
        points.append((1.0, points[-1][1]))
    return points

def plot_pr_curve(points, title="Precision-Recall Curve (Jaccard)"):
    plt.figure()
    recalls, precisions = zip(*points)
    plt.plot(recalls, precisions, marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.show()

# --- Main ---
if __name__ == "__main__":
    docs_df = read_documents("documents")
    print("\nDocuments Loaded:\n", docs_df)

    query = input("\nEnter query for Jaccard: ")
    results = rank_jaccard(query, docs_df)

    print(f"\nJaccard Results for query: '{query}'")
    print(results)

    relevant = {2, 3}  # Example relevance set
    for k in [1, 2, 3]:
        print(f"Precision @ {k}: {precision_at_k(results, relevant, k):.2f}")

    pr_points = pr_curve_points(results, relevant)
    print("\nPR Curve Points:", pr_points)
    plot_pr_curve(pr_points)
