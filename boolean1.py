import os
import re
import nltk
import pandas as pd
from collections import defaultdict

nltk.download("stopwords", quiet=True)
stopwords = set(nltk.corpus.stopwords.words("english"))

def load_documents(folder="documents"):
    docs = {}
    for idx, filename in enumerate(os.listdir(folder), 1):
        path = os.path.join(folder, filename)
        if filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs[idx] = f.read().lower()
    return docs

def preprocess(text):
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if w not in stopwords]

def build_inverted_index(docs):
    inverted_index = defaultdict(set)
    for doc_id, text in docs.items():
        for word in preprocess(text):
            inverted_index[word].add(doc_id)
    return inverted_index

def boolean_retrieval(query, inverted_index, all_docs):
    query = query.upper()
    tokens = query.split()

    result = set(all_docs)
    operator = None

    for token in tokens:
        if token in ["AND", "OR", "NOT"]:
            operator = token
        else:
            docs_with_term = inverted_index.get(token.lower(), set())
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
    docs = load_documents("documents")
    all_docs = set(docs.keys())
    inverted_index = build_inverted_index(docs)

    print("\n Boolean Retrieval System Ready!")
    print("Enter queries using AND, OR, NOT (example: information AND retrieval).\n")

    # Ask only ONCE
    query = input("Enter query: ")

    result = boolean_retrieval(query, inverted_index, all_docs)
    if result:
        print(f"\nMatching Documents: {sorted(result)}")
    else:
        print("\nNo documents matched your query.")
