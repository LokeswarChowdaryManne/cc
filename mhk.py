import random

def h1(x, n_shingles):
    return (x + 2) % n_shingles

def h2(x, n_shingles):
    return (3 * x + 1) % n_shingles

def h3(x, n_shingles):
    return (x + 4) % n_shingles

def minhash_signature_matrix(shingle_doc_matrix, hash_functions):
    n_shingles = len(shingle_doc_matrix)
    n_docs = len(shingle_doc_matrix[0])
    n_hashes = len(hash_functions)
    signature_matrix = [[float('inf')] * n_docs for _ in range(n_hashes)]

    for i in range(n_shingles):
        for j in range(n_docs):
            if shingle_doc_matrix[i][j] == 1:
                for k in range(n_hashes):
                    hash_val = hash_functions[k](i, n_shingles)
                    if hash_val < signature_matrix[k][j]:
                        signature_matrix[k][j] = hash_val
    return signature_matrix


def get_shingles(text, k=3):
    words = text.lower().split()
    shingles = set()
    for i in range(len(words) - k + 1):
        shingle = ' '.join(words[i:i+k])
        shingles.add(shingle)
    return list(shingles)

def build_shingle_vocab(docs, k=3):
    vocab = set()
    for doc in docs:
        vocab.update(get_shingles(doc, k))
    return sorted(list(vocab))

def create_shingle_doc_matrix(docs, k=3):
    vocab = build_shingle_vocab(docs, k)
    n_shingles = len(vocab)
    n_docs = len(docs)
    matrix = [[0] * n_docs for _ in range(n_shingles)]

    for i, shingle in enumerate(vocab):
        for j, doc in enumerate(docs):
            if shingle in get_shingles(doc, k):
                matrix[i][j] = 1
    return matrix, vocab

def jaccard(s1, s2):
    inter = len(set(s1).intersection(s2))
    union = len(set(s1).union(s2))
    if union == 0:
        return 0.0
    return inter / union

def estimate_jaccard(sig1, sig2):
    matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
    return matches / len(sig1)


documents = [
    "This is document one",
    "This is document two",
    "Another document here",
    "This is document one and two"
]

hash_functions = [h1, h2, h3]

shingle_doc_matrix_example, shingle_vocab = create_shingle_doc_matrix(documents, k=2)

print("Shingle Vocabulary:", shingle_vocab)
print("\nShingle Document Matrix:")
for row in shingle_doc_matrix_example:
    print(row)

signature_matrix_example = minhash_signature_matrix(shingle_doc_matrix_example, hash_functions)
print("\nMinHash Signature Matrix for Example Documents:")
for row in signature_matrix_example:
    print(row)

print("\nEstimated Jaccard Similarity from MinHash Signatures:")
signatures_per_doc = [[signature_matrix_example[j][i] for j in range(len(signature_matrix_example))] for i in range(len(signature_matrix_example[0]))]

for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        est_sim = estimate_jaccard(signatures_per_doc[i], signatures_per_doc[j])
        print(f"Estimated Jaccard(Doc{i+1}, Doc{j+1}): {est_sim:.4f}")

print("\nActual Jaccard Similarity:")
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        shingles1 = set(get_shingles(documents[i], k=2))
        shingles2 = set(get_shingles(documents[j], k=2))
        actual_sim = jaccard(shingles1, shingles2)
        print(f"Actual Jaccard(Doc{i+1}, Doc{j+1}): {actual_sim:.4f}")
