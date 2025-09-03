import os
import re
import nltk
import pandas as pd
from collections import defaultdict

# Download English stopwords
nltk.download("stopwords", quiet=True)
stopwords = set(nltk.corpus.stopwords.words("english"))

def load_documents(folder="documents"):
    # Load all text files from the folder into a dictionary
    docs = {}
    for idx, filename in enumerate(os.listdir(folder), 1):
        path = os.path.join(folder, filename)
        if filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs[idx] = f.read().lower()
    return docs

def preprocess(text):
    # Tokenize, lowercase, and remove stopwords
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if w not in stopwords]

def build_inverted_index(docs):
    # Build an inverted index mapping terms → set of doc_ids
    inverted_index = defaultdict(set)
    for doc_id, text in docs.items():
        for word in preprocess(text):
            inverted_index[word].add(doc_id)
    return inverted_index

def boolean_retrieval(query, inverted_index, all_docs):
    # Perform Boolean retrieval using AND, OR, NOT operators
    query = query.upper()
    tokens = query.split()

    result = set(all_docs)  # Start with all documents
    operator = None         # Keep track of last operator

    for token in tokens:
        if token in ["AND", "OR", "NOT"]:
            operator = token  # Update operator
        else:
            docs_with_term = inverted_index.get(token.lower(), set())  # Docs containing term
            if operator is None:
                result = docs_with_term
            elif operator == "AND":
                result = result & docs_with_term
            elif operator == "OR":
                result = result | docs_with_term
            elif operator == "NOT":
                result = result - docs_with_term
    return result

if __name__ == "__main__":
    # Load documents and build inverted index
    docs = load_documents("documents")
    all_docs = set(docs.keys())
    inverted_index = build_inverted_index(docs)

    print("\nBoolean Retrieval System Ready!")
    print("Enter queries using AND, OR, NOT (example: information AND retrieval).\n")

    # Ask only once for query
    query = input("Enter query: ")

    # Run Boolean retrieval and print results
    result = boolean_retrieval(query, inverted_index, all_docs)
    if result:
        print(f"\nMatching Documents: {sorted(result)}")
    else:
        print("\nNo documents matched your query.")
