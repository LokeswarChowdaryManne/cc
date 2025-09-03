"""
Assignment: Build a small domain Search Engine (lyrics example)

Features included:
- Preprocessing (tokenization, stopword removal, optional stemming/lemmatization)
- Inverted index creation
- Simple index compression: gap encoding + variable-byte encoding
- Retrieval models: VSM (TF-IDF + cosine) and Okapi BM25
- Simple query mapping using WordNet synonyms (optional)
- Display top-k results
- Explainability: term contribution to scores for top results

Usage:
- Place text files (one document per .txt file) into 'lyrics/' folder
- Run: python lyrics_search.py
- Provide a query and choose model 'vsm' or 'bm25'
"""

import os
import re
import math
import json
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

# optional imports
try:
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    nltk_available = True
except Exception:
    nltk_available = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Configuration
# -------------------------------
DOCS_FOLDER = "lyrics"       # folder of .txt docs (one doc = one file)
TOP_K = 5                    # results to show by default
BM25_K1 = 1.5
BM25_B = 0.75

# -------------------------------
# Utilities and Preprocessing
# -------------------------------
if nltk_available:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    STOPWORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
else:
    # small fallback stopword set if nltk is not installed
    STOPWORDS = {"the", "a", "an", "and", "is", "in", "of", "to", "it", "for"}

def normalize_text(text):
    """Lowercase and remove non-alphabetic characters (keep spaces)."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(text, do_lemmatize=True):
    """Tokenize, remove stopwords, optionally lemmatize."""
    text = normalize_text(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 0]
    if nltk_available and do_lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens

# -------------------------------
# Loading Documents
# -------------------------------
def load_documents(folder=DOCS_FOLDER):
    """
    Load .txt files from folder. Returns list of (doc_id, filename, raw_text).
    """
    docs = []
    filenames = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
    for i, fn in enumerate(filenames, start=1):
        path = os.path.join(folder, fn)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        docs.append((i, fn, raw))
    return docs

# -------------------------------
# Inverted Index (uncompressed)
# -------------------------------
def build_inverted_index(docs):
    """
    Build inverted index: term -> list of doc_ids
    Also return term frequencies per doc and doc lengths.
    """
    index = defaultdict(list)            # postings lists (unsorted initially)
    tf = defaultdict(lambda: defaultdict(int))   # tf[doc_id][term]
    doc_len = {}                         # document lengths (#tokens)

    for doc_id, filename, raw in docs:
        tokens = preprocess(raw)
        doc_len[doc_id] = len(tokens)
        counts = Counter(tokens)
        for term, cnt in counts.items():
            index[term].append(doc_id)
            tf[doc_id][term] = cnt

    # ensure postings are sorted and unique
    for term in list(index.keys()):
        plist = sorted(set(index[term]))
        index[term] = plist
    return index, tf, doc_len

# -------------------------------
# Simple Compression: gap + variable-byte
# -------------------------------
def gap_encode(postings):
    """Convert sorted postings list to gaps (first element as-is)."""
    if not postings:
        return []
    gaps = [postings[0]]
    for i in range(1, len(postings)):
        gaps.append(postings[i] - postings[i-1])
    return gaps

def vb_encode_number(n):
    """Variable-byte encode a single integer, return list of bytes (ints 0-255)."""
    bytes_list = []
    while True:
        bytes_list.insert(0, n % 128)  # 7 bits per byte
        if n < 128:
            break
        n = n // 128
    bytes_list[-1] += 128  # mark last byte by setting high bit
    return bytes_list

def vb_encode_list(numbers):
    """Encode list of integers using variable-byte; returns list of ints (0-255)."""
    bytelist = []
    for n in numbers:
        bytelist.extend(vb_encode_number(n))
    return bytelist

def compress_index(index):
    """
    Compress index: for each term store variable-byte encoding of gap-encoded postings.
    Returns dict term -> bytes (list of ints).
    """
    compressed = {}
    for term, postings in index.items():
        gaps = gap_encode(postings)
        encoded = vb_encode_list(gaps)
        compressed[term] = bytes(encoded)  # store as bytes object
    return compressed

def decompress_vb_stream(bs):
    """Decode a bytes object produced by vb_encode_list into list of integers."""
    numbers = []
    n = 0
    for byte in bs:
        b = byte
        if b >= 128:
            n = 128 * n + (b - 128)
            numbers.append(n)
            n = 0
        else:
            n = 128 * n + b
    return numbers

def gap_decode(gaps):
    """Decode gap list back to postings."""
    if not gaps:
        return []
    postings = [gaps[0]]
    for g in gaps[1:]:
        postings.append(postings[-1] + g)
    return postings

# -------------------------------
# TF-IDF (VSM) Preparation
# -------------------------------
def compute_tfidf_matrix(docs, tf, index):
    """
    Builds TF-IDF vectors (dense) using our vocabulary (terms from index).
    Returns: vectorizer-like structures: vocab list, tfidf_matrix (ndarray doc x term), idf dict
    """
    vocab = sorted(index.keys())
    N = len(docs)
    df = {term: len(index[term]) for term in vocab}
    idf = {term: math.log((N + 1) / (df[term] + 0.5)) for term in vocab}  # smoothing similar to assignment
    # Build matrix
    doc_ids = [doc_id for doc_id, _, _ in docs]
    mat = np.zeros((len(doc_ids), len(vocab)), dtype=float)
    for i, doc_id in enumerate(doc_ids):
        doc_tf = tf.get(doc_id, {})
        len_i = sum(doc_tf.values()) if doc_tf else 0
        for j, term in enumerate(vocab):
            tf_ik = doc_tf.get(term, 0)
            if tf_ik > 0 and len_i > 0:
                mat[i, j] = (tf_ik / len_i) * idf[term]
    # normalize rows to unit length for cosine calculation
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat_norm = mat / norms
    return vocab, mat_norm, idf, doc_ids

# -------------------------------
# BM25 Implementation
# -------------------------------
def build_bm25_structures(docs, tf, index):
    """
    Build DF, avgdl, N for BM25 scoring
    """
    N = len(docs)
    df = {term: len(index[term]) for term in index}
    avgdl = np.mean([sum(tf.get(doc_id, {}).values()) for doc_id, _, _ in docs]) if docs else 0.0
    return df, avgdl, N

def bm25_score_for_query(query_terms, doc_id, tf, df, N, avgdl, k1=BM25_K1, b=BM25_B):
    """Compute BM25 score for one document and query (list of terms)."""
    score = 0.0
    doc_tf = tf.get(doc_id, {})
    len_d = sum(doc_tf.values()) if doc_tf else 0.0
    for term in query_terms:
        if term not in df:
            continue
        idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)  # idf smoothing
        f = doc_tf.get(term, 0)
        denom = f + k1 * (1 - b + b * (len_d / avgdl)) if avgdl > 0 else f + k1
        if denom > 0:
            score += idf * (f * (k1 + 1)) / denom
    return score

# -------------------------------
# Query mapping (simple synonym expansion)
# -------------------------------
def expand_query_with_synonyms(query_terms, max_synonyms=1):
    """Expand query terms with one synonym from WordNet each (if available)."""
    if not nltk_available:
        return query_terms
    expanded = list(query_terms)
    for term in query_terms:
        synsets = wordnet.synsets(term)
        if synsets:
            lemmas = synsets[0].lemmas()
            for l in lemmas:
                name = l.name().replace('_', ' ').lower()
                if name != term:
                    expanded.append(name)
                    break
    return expanded

# -------------------------------
# Search Functions
# -------------------------------
def search_vsm(query, docs, vocab, mat_norm, idf, doc_ids, top_k=TOP_K):
    """Search with VSM using precomputed normalized TF-IDF matrix."""
    qtokens = [t for t in preprocess(query)]
    if not qtokens:
        return []
    # build query vector in the same vocab order
    qvec = np.zeros((len(vocab),), dtype=float)
    qcounts = Counter(qtokens)
    qlen = sum(qcounts.values())
    for j, term in enumerate(vocab):
        tf_q = qcounts.get(term, 0)
        if tf_q > 0:
            qvec[j] = (tf_q / qlen) * idf.get(term, 0.0)
    # normalize qvec
    qnorm = np.linalg.norm(qvec)
    if qnorm == 0:
        return []
    qvec /= qnorm
    # cosine: mat_norm dot qvec
    scores = mat_norm.dot(qvec)
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_idx:
        score = float(scores[idx])
        if score <= 0:
            continue
        doc_id = doc_ids[idx]
        # explainability: show term contributions
        term_contribs = {}
        for j, term in enumerate(vocab):
            if qvec[j] != 0 and mat_norm[idx, j] != 0:
                term_contribs[term] = float(qvec[j] * mat_norm[idx, j])
        # sort contributions
        top_terms = sorted(term_contribs.items(), key=lambda x: x[1], reverse=True)[:5]
        results.append((doc_id, score, top_terms))
    return results

def search_bm25(query, docs, tf, df, N, avgdl, top_k=TOP_K):
    qtokens = [t for t in preprocess(query)]
    if not qtokens:
        return []
    # optional expansion
    # qtokens = expand_query_with_synonyms(qtokens)
    scores = {}
    for doc_id, _, _ in docs:
        score = bm25_score_for_query(qtokens, doc_id, tf, df, N, avgdl)
        scores[doc_id] = score
    # pick top
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for doc_id, score in sorted_docs:
        if score <= 0:
            continue
        # explainability: per-term BM25 contribution
        contribs = []
        for term in set(qtokens):
            c = 0.0
            if term in df:
                # compute contribution of this term to score by recomputing component
                f = tf.get(doc_id, {}).get(term, 0)
                if f > 0:
                    idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
                    denom = f + BM25_K1 * (1 - BM25_B + BM25_B * (sum(tf.get(doc_id, {}).values())/avgdl)) if avgdl>0 else f + BM25_K1
                    c = idf * (f * (BM25_K1 + 1)) / denom
            if c>0:
                contribs.append((term, float(c)))
        contribs = sorted(contribs, key=lambda x: x[1], reverse=True)
        results.append((doc_id, float(score), contribs[:5]))
    return results

# -------------------------------
# Display utilities
# -------------------------------
def doc_summary(docs_map, doc_id, max_chars=240):
    """Return filename and snippet for a doc id."""
    fn, text = docs_map[doc_id]['filename'], docs_map[doc_id]['text']
    snippet = text.strip().replace('\n', ' ')[:max_chars]
    return fn, snippet

# -------------------------------
# Main interactive routine
# -------------------------------
def main():
    print("Loading documents from folder:", DOCS_FOLDER)
    docs = load_documents(DOCS_FOLDER)   # list of (doc_id, filename, raw_text)
    if not docs:
        print("No documents found in folder", DOCS_FOLDER, ". Please add .txt files and retry.")
        return

    # small docs map for quick lookup
    docs_map = {doc_id: {'filename': filename, 'text': raw} for doc_id, filename, raw in docs}

    index, tf, doc_len = build_inverted_index(docs)
    print(f"Built inverted index with {len(index)} terms over {len(docs)} docs")

    # compress index (optional)
    compressed = compress_index(index)
    print("Index compressed (variable-byte + gap). Example bytes length for a term (first 5 terms):")
    for i, term in enumerate(list(compressed.keys())[:5]):
        print(f"  {term}: {len(compressed[term])} bytes")

    # prepare VSM structures
    vocab, mat_norm, idf, doc_ids = compute_tfidf_matrix(docs, tf, index)

    # prepare BM25 structures
    df_bm25, avgdl, N = build_bm25_structures(docs, tf, index)
    # interactive queries
    while True:
        print("\nEnter a query (or `quit` to exit). Example: 'love heartbreak' ")
        q = input("Query> ").strip()
        if q.lower() in ('q', 'quit', 'exit'):
            print("Goodbye.")
            break
        model = input("Choose model (vsm / bm25) [vsm]: ").strip().lower() or "vsm"
        try:
            k = int(input(f"Top-k to show [{TOP_K}]: ").strip() or TOP_K)
        except ValueError:
            k = TOP_K

        if model == 'vsm':
            results = search_vsm(q, docs, vocab, mat_norm, idf, doc_ids, top_k=k)
        else:
            results = search_bm25(q, docs, tf, df_bm25, N, avgdl, top_k=k)

        if not results:
            print("No results found.")
            continue

        print(f"\nTop-{len(results)} results (model={model}):")
        for rank, (doc_id, score, contribs) in enumerate(results, start=1):
            fn, snippet = doc_summary(docs_map, doc_id)
            print(f"\nRank {rank}. DocID {doc_id} | score: {score:.4f} | file: {fn}")
            print(f"  Snippet: {snippet}...")
            if contribs:
                print("  Top term contributions:")
                for term, c in contribs:
                    print(f"    {term}: {c:.4f}")

if __name__ == "__main__":
    main()
