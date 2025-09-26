import pandas as pd
import numpy as np

# ---------- Term-Document Matrix ----------
def create_tdm(sentences):
    all_words = []
    for sent in sentences:
        words = sent.lower().split()
        all_words.extend(words)

    terms = sorted(list(set(all_words)))
    num_terms = len(terms)
    num_docs = len(sentences)

    tdm = pd.DataFrame(0, index=terms, columns=range(num_docs))

    for j, sent in enumerate(sentences):
        words = sent.lower().split()
        for word in words:
            tdm.loc[word, j] += 1

    tdm_matrix = tdm.to_numpy()
    return tdm_matrix, terms

# ---------- Simple SVD ----------
def simple_svd(matrix, tolerance=1e-10):
    A = np.array(matrix, dtype=float)
    m, n = A.shape

    AtA = np.dot(A.T, A)
    eigenvalues, eigenvectors = np.linalg.eigh(AtA)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = eigenvectors[:, idx]

    singular_values = np.sqrt(eigenvalues)

    Sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        if singular_values[i] > tolerance:
            Sigma[i, i] = singular_values[i]

    if np.any(singular_values > tolerance):
        Sigma_inv = np.diag(1.0 / singular_values[singular_values > tolerance])
        U = np.dot(A, np.dot(V[:, :len(Sigma_inv)], Sigma_inv))
        U, _ = np.linalg.qr(U)
    else:
        U = np.eye(m)

    Vh = V.T

    return U, Sigma, Vh

# ---------- Cosine Similarity ----------
def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# ---------- Document Ranking ----------
def rank_documents(query, terms, U, Sigma, Vh, sentences):
    # Create query vector in term space
    query_vec = np.zeros(len(terms))
    for word in query.lower().split():
        if word in terms:
            idx = terms.index(word)
            query_vec[idx] += 1

    # Project query into LSA space
    singular_values = np.diag(Sigma)
    inv_singular = np.array([1/s if s > 1e-10 else 0 for s in singular_values])
    Sigma_inv = np.diag(inv_singular)

    query_lsa = np.dot(np.dot(query_vec.T, U), Sigma_inv)

    # Document vectors in LSA space (columns of Vh)
    doc_vectors = Vh.T  # each row is a document in concept space

    # Compute cosine similarity
    similarities = [cosine_similarity(query_lsa, doc_vec) for doc_vec in doc_vectors]

    # Rank documents
    ranked_idx = np.argsort(similarities)[::-1]
    ranked_docs = [(sentences[i], similarities[i]) for i in ranked_idx]

    return ranked_docs

# ---------- Main ----------
if __name__ == "__main__":
    sentences = [
        "Data science is fun",
        "Machine learning is part of data science",
        "Deep learning is a subset of machine learning"
    ]

    tdm_matrix, terms = create_tdm(sentences)
    print("Term-Document Matrix:")
    print(pd.DataFrame(tdm_matrix, index=terms))

    U, Sigma, Vh = simple_svd(tdm_matrix)
    print(U)
    print(Sigma)
    print(Vh)

    A_reconstructed = np.dot(U, np.dot(np.diag(Sigma.diagonal()), Vh))
    print("Reconstructed Matrix:")
    print(np.round(A_reconstructed, 2))

    error = np.linalg.norm(tdm_matrix - A_reconstructed)
    print(f"Reconstruction error: {error:.6f}")

    # ---------- Document Ranking ----------
    query = "machine learning data"
    ranked_docs = rank_documents(query, terms, U, Sigma, Vh, sentences)

    print(f"Ranking of documents for query: '{query}'\n")
    for doc, score in ranked_docs:
        print(f"Score: {score:.4f} => {doc}")

#______________________________________________________________________________________________________________________________________________________________________________

import numpy as np
import random

def get_ngrams(sequence, n):
    words = sequence.split()  # Split into words
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

def create_shingle_doc_matrix(sequences, n):
    all_shingles = set()
    for seq in sequences:
        shingles = get_ngrams(seq, n)
        all_shingles.update(shingles)
    all_shingles = sorted(list(all_shingles))

    n_shingles = len(all_shingles)
    n_docs = len(sequences)
    shingle_doc_matrix = np.zeros((n_shingles, n_docs), dtype=int)

    for j, doc in enumerate(sequences):
        doc_shingles = get_ngrams(doc, n)
        for i, shingle in enumerate(all_shingles):
            if shingle in doc_shingles:
                shingle_doc_matrix[i, j] = 1

    return shingle_doc_matrix, all_shingles

def get_example_hash_functions(modulus):
    h1 = lambda r: (1 * r + 1) % modulus
    h2 = lambda r: (3 * r + 1) % modulus
    return [h1, h2]

def generate_hash_functions(num_hashes, modulus):
    hash_functions = []
    for _ in range(num_hashes):
        a = random.randint(1, modulus - 1)
        b = random.randint(0, modulus - 1)
        hash_functions.append(lambda r, a=a, b=b: (a * r + b) % modulus)
    return hash_functions

def compute_signature_matrix(shingle_doc_matrix, num_hashes, modulus=None):
    n_shingles, n_docs = shingle_doc_matrix.shape
    if modulus is None:
        modulus = n_shingles

    hash_functions = get_example_hash_functions(modulus)
    # hash_functions = generate_hash_functions(num_hashes, modulus)  # RANDOM

    INF = np.inf
    sig_matrix = np.full((num_hashes, n_docs), INF)

    for r in range(n_shingles):
        hash_values = [h(r) for h in hash_functions]
        for c in range(n_docs):
            if shingle_doc_matrix[r, c] == 1:
                for i in range(num_hashes):
                    sig_matrix[i, c] = min(sig_matrix[i, c], hash_values[i])

    return sig_matrix

def compute_signature_matrix_permutations(shingle_doc_matrix, num_hashes):
    n_shingles, n_docs = shingle_doc_matrix.shape

    # Initialize signature matrix with infinity
    INF = np.inf
    sig_matrix = np.full((num_hashes, n_docs), INF)

    # Generate random permutations of row indices
    rows = list(range(n_shingles))
    permutations = [random.sample(rows, len(rows)) for _ in range(num_hashes)]

    # Compute MinHash signature using permutations
    for i, perm in enumerate(permutations):
        for col in range(n_docs):
            for row in perm:
                if shingle_doc_matrix[row, col] == 1:
                    sig_matrix[i, col] = row
                    break  # Take the first row index where a 1 is found

    return sig_matrix

def minhash_similarity(sig_vec1, sig_vec2):
    matches = np.sum(sig_vec1 == sig_vec2)
    return matches / len(sig_vec1)

def jaccard_similarity(shingles1, shingles2):
    set1 = set(shingles1)
    set2 = set(shingles2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

def print_pairwise_similarity_matrices(sequences, sig_matrix, n):
    n_docs = sig_matrix.shape[1]
    minhash_sim_matrix = np.zeros((n_docs, n_docs))
    jaccard_sim_matrix = np.zeros((n_docs, n_docs))

    # Compute shingles for each document
    doc_shingles = [get_ngrams(seq, n) for seq in sequences]

    for i in range(n_docs):
        for j in range(n_docs):
            if i == j:
                minhash_sim_matrix[i, j] = 1.0
                jaccard_sim_matrix[i, j] = 1.0
            else:
                minhash_sim_matrix[i, j] = minhash_similarity(sig_matrix[:, i], sig_matrix[:, j])
                jaccard_sim_matrix[i, j] = jaccard_similarity(doc_shingles[i], doc_shingles[j])

    # Print MinHash similarity matrix
    print("MinHash Pairwise Similarity Matrix:")
    for i in range(n_docs):
        for j in range(n_docs):
            if i == j:
                print("*", end="\t")
            elif i < j:
                print(f"{minhash_sim_matrix[i, j]:.2f}", end="\t")
            else:
                print("-", end="\t")
        print()

    # Print Jaccard similarity matrix
    print("\nJaccard Pairwise Similarity Matrix:")
    for i in range(n_docs):
        for j in range(n_docs):
            if i == j:
                print("*", end="\t")
            elif i < j:
                print(f"{jaccard_sim_matrix[i, j]:.2f}", end="\t")
            else:
                print("-", end="\t")
        print()

    return minhash_sim_matrix, jaccard_sim_matrix

def detect_duplicates(minhash_sim_matrix, jaccard_sim_matrix, threshold=0.9):
    n_docs = minhash_sim_matrix.shape[1]

    print(f"\n[DUPLICATES - MinHash] Detecting duplicates with similarity >= {threshold}:")
    found_minhash = False
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            similarity = minhash_sim_matrix[i, j]
            if similarity >= threshold:
                print(f"Docs {i+1} and {j+1}: MinHash Similarity = {similarity:.2f}")
                found_minhash = True
    if not found_minhash:
        print("No duplicates found.")

    print(f"\n[DUPLICATES - Jaccard] Detecting duplicates with similarity >= {threshold}:")
    found_jaccard = False
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            similarity = jaccard_sim_matrix[i, j]
            if similarity >= threshold:
                print(f"Docs {i+1} and {j+1}: Jaccard Similarity = {similarity:.2f}")
                found_jaccard = True
    if not found_jaccard:
        print("No duplicates found.")

# --- Example usage with simplified output ---
sequences = ["the cat sat", "the dog ran", "cat and dog", "the cat ran"]
n = 2
shingle_doc_matrix, shingle_vocab = create_shingle_doc_matrix(sequences, n)
# print("Shingles:", shingle_vocab)
# print("Shingle-Doc Matrix:\n", shingle_doc_matrix)

# Using hash functions
num_hashes = 2
modulus = 5
print("\n=== MinHash with Hash Functions ===")
sig_matrix_hash = compute_signature_matrix(shingle_doc_matrix, num_hashes, modulus)
# print("Signature Matrix (Hash Functions):")
# print(sig_matrix_hash)
minhash_sim_matrix_hash, jaccard_sim_matrix = print_pairwise_similarity_matrices(sequences, sig_matrix_hash, n)
detect_duplicates(minhash_sim_matrix_hash, jaccard_sim_matrix, threshold=0.9)

# Using permutations
num_hashes = 2
print("\n=== MinHash with Permutations ===")
sig_matrix_perm = compute_signature_matrix_permutations(shingle_doc_matrix, num_hashes)
# print("Signature Matrix (Permutations):")
# print(sig_matrix_perm)
minhash_sim_matrix_perm, jaccard_sim_matrix = print_pairwise_similarity_matrices(sequences, sig_matrix_perm, n)
detect_duplicates(minhash_sim_matrix_perm, jaccard_sim_matrix, threshold=0.9)

#______________________________________________________________________________________________________________________________________________________________________________

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
n_nodes = 3

H = np.zeros((n_nodes, n_nodes), dtype=float)

# Add edges (directed) based on the graph
edges = [(0, 1), (2, 1), (1, 0), (1, 2)]  # (0,1): 1→2, (2,1): 3→2, (1,0): 2→1, (1,2): 2→3

for src, dest in edges:
    H[src][dest] = 1

print("Adjacency Matrix ")
print(H)

# Step 1: If a row of H has no 1's, then replace each element by 1/N
H_step1 = H.copy()
for i in range(n_nodes):
    if np.sum(H_step1[i]) == 0:
        H_step1[i] = np.ones(n_nodes, dtype=float) / n_nodes
print("\nAfter Step 1 (H after handling rows with no 1's):")
print(H_step1)

# Step 2: Divide each 1 in H by the number of 1's in its row (based on original H)
H_step2 = H_step1.copy()
for i in range(n_nodes):
    row_sum = np.sum(H[i])  # Number of 1's in the original row
    if row_sum > 0:
        # # Create a mask for positions with 1's in the original H
        # mask = (H[i] == 1)
        # H_step2[i] = H_step1[i]  # Start with the Step 1 matrix
        # H_step2[i, mask] = H_step2[i, mask] / row_sum  # Divide only the 1's
        H_step2[i] = H_step2[i] / row_sum
print("\nAfter Step 2 (H after dividing by row sum):")
print(H_step2)

# Step 3: Multiply the resulting matrix by (1 - α)
alpha = 0.5
S = H_step2 * (1 - alpha)
print("\nAfter Step 3 (S = H * (1 - α)):")
print(S)

# Step 4: Add α/N to every entry of the resulting matrix to obtain G
G = S + (alpha / n_nodes)
print("\nAfter Step 4 (Final TPM G):")
print(G)

# Initial PageRank vector
# PR = np.array([1/3, 1/3, 1/3], dtype=float)
PR = np.ones(n_nodes, dtype=float) / n_nodes
PR = np.array([1/n_nodes] * n_nodes, dtype=float)

max_iterations = 100
epsilon = 0.01  # Convergence threshold

for iteration in range(max_iterations):
    PR_new = np.dot(PR, G)  # PR^(k+1) = PR^(k) * G
    # Normalize to ensure sum is 1
    print(f"\nIteration {iteration}: {PR_new}")

    if np.sum(np.abs(PR_new - PR)) < epsilon:
        print(f"Converged at iteration {iteration}")
        break
    PR = PR_new

print(f"\nFinal PageRank: {PR_new}")

def visualize_page_rank(page_rank, n_nodes):
    nodes = [f"Node {i+1}" for i in range(n_nodes)]

    plt.figure(figsize=(10, 6))
    plt.bar(nodes, page_rank, color=plt.cm.Paired(np.linspace(0, 1, n_nodes)))
    plt.ylabel('PageRank Value')
    plt.title('PageRank Distribution Across Nodes')
    plt.ylim(0, max(page_rank) + 0.1)
    for i, v in enumerate(page_rank):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')  # Add value labels on top of bars
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- VISUALIZE GRAPH ---
G_nx = nx.from_numpy_array(H, create_using=nx.DiGraph)
plt.figure(figsize=(6, 4))
pos = nx.spring_layout(G_nx)
nx.draw(G_nx, pos, with_labels=True, node_color='skyblue', node_size=1500, arrowsize=20)
plt.title("Document Graph (from adjacency matrix)")
plt.show()

visualize_page_rank(PR_new, n_nodes)

#______________________________________________________________________________________________________________________________________________________________________________

