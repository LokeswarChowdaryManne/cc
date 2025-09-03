"""
Plagiarism Checker Assignment
-----------------------------

To maintain the integrity of your data, it’s a good idea to reduce duplicate documents 
in the system. Plagiarism Checker helps us to detect duplicates. The database has N documents. 
The common structure of each document is title, author names, affiliation, abstract, keywords, 
Paper content, Conclusion. A Plagiarism Checker reads a new document and adds it to the database 
if it is not a duplicate of any document in the database. 

Design a Plagiarism Checker with the following features:

A) Verify if the titles are exactly same (Apply BinaryDistance(u,v), which gives the
   binary distance between vectors u and v, equal to 0 if they are identical and 1 otherwise.).
   If same, label the document as duplicate and discard it else proceed to second part of the Checker.

B) Represent documents (Paper content) as term document vectors with weight of a
   term in a document computed as:
        w_ik = (tf_ik / len_i) * log( (N+1) / (0.5 + df_k) )
   where tf_ik = term frequency, len_i = document length, df_k = document frequency, N = total docs.

C) Identify a document as duplicate if the cosine similarity of the document 
   with any existing document is more than the threshold α (α = 0.85).

D) Identify k-Shingles (N grams) in the documents and apply Jaccard similarity to
   identify duplicates.

E) Apply probability retrieval model (Okapi BM25) to identify duplicates.

Dataset:
--------
D1: Information requirement: query considers the user feedback as information requirement to search.
D2: Information retrieval: query depends on the model of information retrieval used.
D3: Prediction problem: Many problems in information retrieval can be viewed as prediction problems
D4: Search: A search engine is one of applications of information retrieval models.

New documents:
D5: Feedback: feedback is typically used by the system to modify the query and improve prediction
D6: Information retrieval: ranking in information retrieval algorithms depends on user query
"""

import re
import math
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- Utility Functions -------------------

def tokenize(text):
    """Lowercase and tokenize a document into words."""
    return re.findall(r'\w+', text.lower())

def binary_distance(u, v):
    """Return 0 if two titles are identical, else 1."""
    return 0 if u.strip().lower() == v.strip().lower() else 1

# ------------------- Weight Calculation -------------------

def compute_term_weights(documents):
    """
    Compute term weights using the given formula:
    w_ik = (tf_ik / len_i) * log( (N+1) / (0.5 + df_k) )
    """
    tokenized_docs = [tokenize(doc) for doc in documents]
    N = len(tokenized_docs)  # total number of docs
    df = Counter()  # document frequency

    # Count document frequency
    for doc in tokenized_docs:
        for term in set(doc):
            df[term] += 1

    # Build term weights for each document
    weights = []
    vocab = sorted(set(term for doc in tokenized_docs for term in doc))

    for doc in tokenized_docs:
        tf = Counter(doc)  # term frequency
        len_i = len(doc)   # document length
        vector = []
        for term in vocab:
            tf_ik = tf[term]
            if tf_ik > 0:
                weight = (tf_ik / len_i) * math.log((N + 1) / (0.5 + df[term]))
            else:
                weight = 0.0
            vector.append(weight)
        weights.append(vector)
    return np.array(weights), vocab

# ------------------- Similarity Measures -------------------

def cosine_sim_matrix(weights):
    """Compute cosine similarity between documents."""
    return cosine_similarity(weights)

def jaccard_similarity(doc1, doc2, k=3):
    """Compute Jaccard similarity between two docs using k-shingles."""
    shingles1 = set([doc1[i:i+k] for i in range(len(doc1)-k+1)])
    shingles2 = set([doc2[i:i+k] for i in range(len(doc2)-k+1)])
    return len(shingles1 & shingles2) / len(shingles1 | shingles2)

def bm25_score(query_tokens, doc_tokens, avgdl, N, df, k1=1.5, b=0.75):
    """Compute BM25 score of a document for a given query."""
    score = 0.0
    len_d = len(doc_tokens)
    tf = Counter(doc_tokens)
    for term in query_tokens:
        if term not in df: 
            continue
        idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
        numerator = tf[term] * (k1 + 1)
        denominator = tf[term] + k1 * (1 - b + b * len_d / avgdl)
        score += idf * numerator / denominator
    return score

# ------------------- Main Program -------------------

if __name__ == "__main__":
    # Sample dataset
    existing_docs = {
        "D1": "Information requirement: query considers the user feedback as information requirement to search.",
        "D2": "Information retrieval: query depends on the model of information retrieval used.",
        "D3": "Prediction problem: Many problems in information retrieval can be viewed as prediction problems",
        "D4": "Search: A search engine is one of applications of information retrieval models."
    }

    new_docs = {
        "D5": "Feedback: feedback is typically used by the system to modify the query and improve prediction",
        "D6": "Information retrieval: ranking in information retrieval algorithms depends on user query"
    }

    # Step A: Title check (Binary Distance)
    print("\nStep A: Title Check (Binary Distance)")
    for new_id, new_text in new_docs.items():
        for old_id, old_text in existing_docs.items():
            dist = binary_distance(new_text.split(":")[0], old_text.split(":")[0])
            if dist == 0:
                print(f"{new_id} is duplicate of {old_id} (Title Match)")
    
    # Step B + C: Vector model and Cosine Similarity
    print("\nStep B + C: Vector Space Model with Cosine Similarity")
    all_docs = list(existing_docs.values()) + list(new_docs.values())
    weights, vocab = compute_term_weights(all_docs)
    sim_matrix = cosine_sim_matrix(weights)

    threshold = 0.85
    doc_ids = list(existing_docs.keys()) + list(new_docs.keys())
    for i in range(len(existing_docs), len(all_docs)):
        for j in range(len(existing_docs)):
            if sim_matrix[i][j] > threshold:
                print(f"{doc_ids[i]} is duplicate of {doc_ids[j]} (Cosine={sim_matrix[i][j]:.2f})")

    # Step D: k-Shingles + Jaccard
    print("\nStep D: k-Shingles + Jaccard Similarity")
    for new_id, new_text in new_docs.items():
        for old_id, old_text in existing_docs.items():
            jacc = jaccard_similarity(new_text.lower(), old_text.lower(), k=3)
            if jacc > 0.5:
                print(f"{new_id} is duplicate of {old_id} (Jaccard={jacc:.2f})")

    # Step E: BM25
    print("\nStep E: BM25 Scoring")
    tokenized_docs = [tokenize(doc) for doc in all_docs]
    N = len(tokenized_docs)
    avgdl = sum(len(doc) for doc in tokenized_docs) / N
    df = Counter(term for doc in tokenized_docs for term in set(doc))

    for new_id, new_text in new_docs.items():
        new_tokens = tokenize(new_text)
        scores = {}
        for old_id, old_text in existing_docs.items():
            old_tokens = tokenize(old_text)
            scores[old_id] = bm25_score(new_tokens, old_tokens, avgdl, N, df)
        best_match = max(scores, key=scores.get)
        print(f"{new_id} most similar to {best_match} (BM25={scores[best_match]:.2f})")
