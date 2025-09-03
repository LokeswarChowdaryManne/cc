import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Preprocessing ---
STOP_WORDS = {'the', 'a', 'is', 'in', 'of', 'and', 'to', 'it', 'for', 'that'}

def preprocess(text: str) -> list[str]:
    text = re.sub(r'[^a-z\s]', '', text.lower())
    return [w for w in text.split() if w not in STOP_WORDS]

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
                "tokens": preprocess(content)
            })
    return pd.DataFrame(docs)

# --- BM25 Implementation ---
class BM25:
    def __init__(self, tokenized_docs, k1=1.5, b=0.75):
        self.docs = tokenized_docs
        self.N = len(tokenized_docs)
        self.avgdl = sum(len(doc) for doc in tokenized_docs) / self.N
        self.k1 = k1
        self.b = b
        self.df = {}
        self.idf = {}
        self.build()

    def build(self):
        for doc in self.docs:
            for word in set(doc):
                self.df[word] = self.df.get(word, 0) + 1
        for word, freq in self.df.items():
            self.idf[word] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

    def score(self, query, index):
        doc = self.docs[index]
        score = 0.0
        doc_len = len(doc)
        freq = {}
        for word in doc:
            freq[word] = freq.get(word, 0) + 1

        for word in query:
            if word in freq:
                tf = freq[word]
                idf = self.idf.get(word, 0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * num / den
        return score

    def rank(self, query):
        query_tokens = preprocess(query)
        scores = [self.score(query_tokens, idx) for idx in range(self.N)]
        return scores

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

def plot_pr_curve(points, title="Precision-Recall Curve (BM25)"):
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

    bm25 = BM25(docs_df["tokens"].tolist())

    query = input("\nEnter query for BM25: ")
    scores = bm25.rank(query)

    results = pd.DataFrame({
        "doc_id": docs_df["doc_id"],
        "text": docs_df["text"],
        "bm25_score": scores
    }).sort_values(by="bm25_score", ascending=False)

    print(f"\nBM25 Results for query: '{query}'")
    print(results)

    relevant = {2, 3}  # Example relevance set
    for k in [1, 2, 3]:
        print(f"Precision @ {k}: {precision_at_k(results, relevant, k):.2f}")

    pr_points = pr_curve_points(results, relevant)
    print("\nPR Curve Points:", pr_points)
    plot_pr_curve(pr_points)
