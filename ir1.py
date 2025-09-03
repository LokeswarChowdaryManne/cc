"""
Programming Assignment - Simple Index and Retrieval System
-----------------------------------------------------------

The aim of this programming assignment is to create a simple index and design a retrieval system.

Tasks:
1. Create an Inverted Index
   - Implement a program that reads through the list of sorted terms and creates an in-memory inverted index.

2. Treat the 10 most prominent words as stop words
   - Let the program compute the index size after removing these frequent stopwords.

3. Boolean Retrieval
   - Process simple Boolean queries using the index created above.
   - Provide support for:
       a) Conjunctive queries (AND queries)
       b) Mixed operator queries (AND, OR, NOT)

References (Reading Assignment):
- http://lucene.apache.org/core/
- http://tartarus.org/~martin/PorterStemmer/
- http://www.crummy.com/software/BeautifulSoup/
- http://nltk.org/

Data sets (you can test with your own .txt files):
- http://ir.dcs.gla.ac.uk/resources/test_collections/
- https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases.html
"""
import os
import re
import collections
import nltk
from collections import defaultdict

# Download stopwords (only once)
nltk.download("stopwords", quiet=True)
stop_words = set(nltk.corpus.stopwords.words("english"))

# ------------------------------
# Function: Preprocess Text
# ------------------------------
def preprocess(text):
    """
    Preprocess text by:
    - Lowercasing
    - Tokenizing
    - Removing stopwords
    - Keeping only alphabetic words
    """
    text = text.lower()
    words = re.findall(r"\w+", text)
    return [w for w in words if w not in stop_words]

# ------------------------------
# Function: Build Inverted Index
# ------------------------------
def build_inverted_index(documents):
    """
    Build inverted index mapping term -> set of document IDs
    """
    inverted_index = defaultdict(set)
    for doc_id, text in documents.items():
        for word in preprocess(text):
            inverted_index[word].add(doc_id)
    return inverted_index

# ------------------------------
# Function: Reduce Index Size (remove top frequent terms)
# ------------------------------
def reduce_index(inverted_index):
    """
    Remove the most frequent terms (like stopwords).
    - Dynamically chooses how many terms to remove based on dataset size
    """
    term_frequencies = {term: len(docs) for term, docs in inverted_index.items()}
    sorted_terms = sorted(term_frequencies.items(), key=lambda x: x[1], reverse=True)

    # Decide how many frequent terms to remove
    remove_count = min(10, max(1, len(sorted_terms) // 5))  # 20% or up to 10
    top_terms = [t for t, _ in sorted_terms[:remove_count]]

    for term in top_terms:
        del inverted_index[term]

    return inverted_index, set(top_terms)

# ------------------------------
# Function: Boolean Retrieval
# ------------------------------
def boolean_retrieval(query, inverted_index, all_docs):
    """
    Process Boolean queries with operators: AND, OR, NOT
    """
    query = query.upper().split()
    result = set(all_docs)  # start with all docs
    operator = None

    for token in query:
        if token in ["AND", "OR", "NOT"]:
            operator = token
        else:
            docs_with_term = inverted_index.get(token.lower(), set())
            if operator is None:  # first term
                result = docs_with_term
            elif operator == "AND":
                result &= docs_with_term
            elif operator == "OR":
                result |= docs_with_term
            elif operator == "NOT":
                result -= docs_with_term
    return result

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    # Load documents from folder
    folder = "documents"  # ensure you have a 'documents' folder with .txt files
    documents = {}
    for idx, filename in enumerate(os.listdir(folder), 1):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                documents[idx] = f.read()

    # Build inverted index
    inverted_index = build_inverted_index(documents)
    print(f"\nOriginal Inverted Index Size: {len(inverted_index)} terms")

    # Reduce index (remove frequent terms)
    reduced_index, removed_terms = reduce_index(inverted_index)
    print(f"Reduced Inverted Index Size: {len(reduced_index)} terms")
    print(f"Removed frequent terms: {removed_terms}\n")

    # Show available terms
    print("Terms in Index:", list(reduced_index.keys()))

    # Run Boolean Query
    all_docs = set(documents.keys())
    query = input("\nEnter Boolean query (use AND, OR, NOT): ")
    result = boolean_retrieval(query, reduced_index, all_docs)

    if result:
        print(f"\nMatching Documents: {sorted(result)}")
    else:
        print("\nNo documents matched your query.")
