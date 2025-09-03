import os
import re
import nltk
import pandas as pd
from collections import defaultdict

# Download English stopwords from NLTK
nltk.download("stopwords", quiet=True)
stopwords = set(nltk.corpus.stopwords.words("english"))

def load_documents(folder="documents"):
    # Load all text files from the given folder into a dictionary (doc_id → text)
    docs = {}
    for idx, filename in enumerate(os.listdir(folder), 1):
        path = os.path.join(folder, filename)  # Build file path
        if filename.endswith(".txt"):  # Process only .txt files
            with open(path, "r", encoding="utf-8") as f:
                docs[idx] = f.read().lower()  # Read and store lowercase text
    return docs

def preprocess(text):
    # Tokenize text, lowercase, and remove stopwords
    words = re.findall(r"\w+", text.lower())  # Extract words
    return [w for w in words if w not in stopwords]  # Remove stopwords

def build_inverted_index(docs):
    # Build inverted index mapping each word → set of doc_ids containing it
    inverted_index = defaultdict(set)
    for doc_id, text in docs.items():
        for word in preprocess(text):  # Preprocess text into words
            inverted_index[word].add(doc_id)  # Add doc_id for each word
    return inverted_index

def boolean_retrieval(query, inverted_index, all_docs):
    # Perform Boolean retrieval based on AND, OR, NOT operators
    query = query.upper()  # Normalize query to uppercase for operators
    tokens = query.split()  # Split query into tokens

    result = set(all_docs)  # Start with all documents
    operator = None  # Track the last seen operator

    for token in tokens:
        if token in ["AND", "OR", "NOT"]:  # If token is an operator
            operator = token
        else:
            docs_with_term = inverted_index.get(token.lower(), set())  # Docs containing the term
            if operator is None:  # First term in query
                result = docs_with_term
            elif operator == "AND":  # Intersection
                result = result & docs_with_term
            elif operator == "OR":  # Union
                result = result | docs_with_term
            elif operator == "NOT":  # Difference
                result = result - docs_with_term
    return result

if __name__ == "__main__":
    # Step 1: Load documents into memory
    docs = load_documents("documents")
    
    # Step 2: Store set of all document IDs
    all_docs = set(docs.keys())
    
    # Step 3: Build inverted index for Boolean retrieval
    inverted_index = build_inverted_index(docs)

    print("\nBoolean Retrieval System Ready!")
    print("Enter queries using AND, OR, NOT (example: information AND retrieval).\n")

    # Step 4: Take query input from user
    query = input("Enter query: ")

    # Step 5: Run Boolean retrieval
    result = boolean_retrieval(query, inverted_index, all_docs)

    # Step 6: Print matching documents or no match
    if result:
        print(f"\nMatching Documents: {sorted(result)}")
    else:
        print("\nNo documents matched your query.")
