import random
import numpy as np

docs = [
    "the cat sat on the mat",
    "the dog sat on the mat",
    "the cat chased the dog"
]

def get_shingles(doc, k=2):
    words = doc.split()
    return {" ".join(words[i:i+k]) for i in range(len(words)-k+1)}

k = 2
shingle_set = set()
doc_shingles = []

for d in docs:
    sh = get_shingles(d, k)
    print(sh)
    doc_shingles.append(sh)
    shingle_set |= sh

print("Doc :: ", doc_shingles)
shingles = list(shingle_set)
print("Shin : ", shingles)

# Step 2: Build binary shingle–document matrix
matrix = []
for sh in shingles:
    row = [1 if sh in doc_shingles[j] else 0 for j in range(len(docs))]
    matrix.append(row)

sd_matrix = np.array(matrix)

print("Shingle–Document Matrix:")
print(sd_matrix)



# Step 3: MinHash Implementation
num_shingles, num_docs = sd_matrix.shape
num_hashes = 5  # number of permutations
signature = np.full((num_hashes, num_docs), np.inf)

rows = list(range(num_shingles))
permutations = [random.sample(rows, len(rows)) for _ in range(num_hashes)]
print("Per :: ", permutations)

for i, perm in enumerate(permutations):
    for col in range(num_docs):
        counter = 1
        for row in perm:
            if sd_matrix[row, col] == 1:
                signature[i, col] = counter
                break
            counter += 1

print("\nSignature Matrix:")
print(signature)

# Step 4: Similarity from signatures
def minhash_sim(col1, col2):
    return np.mean(signature[:, col1] == signature[:, col2])

print("\nSimilarity between Doc1 & Doc2:", minhash_sim(0, 1))
print("Similarity between Doc1 & Doc3:", minhash_sim(0, 2))
print("Similarity between Doc2 & Doc3:", minhash_sim(1, 2))

def jaccard_sim(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

print("\nJaccard Similarities (Exact):")
print("Doc1 & Doc2:", jaccard_sim(doc_shingles[0], doc_shingles[1]))
print("Doc1 & Doc3:", jaccard_sim(doc_shingles[0], doc_shingles[2]))
print("Doc2 & Doc3:", jaccard_sim(doc_shingles[1], doc_shingles[2]))


import random
import numpy as np

docs = [
    "the cat sat on the mat",
    "the dog sat on the mat",
    "the cat chased the dog"
]

def get_shingles(doc, k=2):
    words = doc.split()
    return {" ".join(words[i:i+k]) for i in range(len(words)-k+1)}

k = 2
shingle_set = set()
doc_shingles = []

for d in docs:
    sh = get_shingles(d, k)
    doc_shingles.append(sh)
    shingle_set |= sh

shingles = list(shingle_set)
num_shingles = len(shingles)

# Step 2: Build binary shingle–document matrix
matrix = []
for sh in shingles:
    row = [1 if sh in doc_shingles[j] else 0 for j in range(len(docs))]
    matrix.append(row)

sd_matrix = np.array(matrix)

print("Shingle–Document Matrix:")
print(sd_matrix)

# Step 3: MinHash using hash functions
num_shingles, num_docs = sd_matrix.shape
num_hashes = 5  # number of hash functions
signature = np.full((num_hashes, num_docs), np.inf)

# Create random hash functions of form: h(x) = (a*x + b) % p
p = num_shingles  # prime > num_shingles
hash_funcs = [(random.randint(1, p-1), random.randint(0, p-1)) for _ in range(num_hashes)]

for i, (a, b) in enumerate(hash_funcs):
    for row in range(num_shingles):
        hash_val = (a * row + b) % p
        for col in range(num_docs):
            if sd_matrix[row, col] == 1:
                if hash_val < signature[i, col]:
                    signature[i, col] = hash_val

print("\nSignature Matrix:")
print(signature.astype(int))

# Step 4: Similarity from signatures
def minhash_sim(col1, col2):
    return np.mean(signature[:, col1] == signature[:, col2])

print("\nSimilarity between Doc1 & Doc2:", minhash_sim(0, 1))
print("Similarity between Doc1 & Doc3:", minhash_sim(0, 2))
print("Similarity between Doc2 & Doc3:", minhash_sim(1, 2))

def jaccard_sim(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

print("\nJaccard Similarities (Exact):")
print("Doc1 & Doc2:", jaccard_sim(doc_shingles[0], doc_shingles[1]))
print("Doc1 & Doc3:", jaccard_sim(doc_shingles[0], doc_shingles[2]))
print("Doc2 & Doc3:", jaccard_sim(doc_shingles[1], doc_shingles[2]))