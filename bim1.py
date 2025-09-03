import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# Function for basic preprocessing: lowercasing, removing non-alphabetic chars, removing stopwords
def basic_preprocess_text(text):
    text = text.lower()  # convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # remove non-letters
    stop_words = {'the', 'a', 'is', 'in', 'of', 'and', 'to', 'it', 'for', 'that'}  # simple stopword list
    tokens = [word for word in text.split() if word not in stop_words]  # filter stopwords
    return ' '.join(tokens)  # return cleaned text

# Reads all .txt files from a directory, preprocesses, and builds a dataframe
def read_and_preprocess_documents(directory_path):
    documents_list = []  # list to hold document info
    doc_id_counter = 1  # doc ID counter

    for filename in os.listdir(directory_path):  # iterate over files in directory
        if filename.endswith(".txt"):  # only process text files
            file_path = os.path.join(directory_path, filename)  # full path
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()  # read file content
                documents_list.append({
                    'doc_id': doc_id_counter,  # assign doc ID
                    'filename': filename,  # store filename
                    'text': content,  # original text
                    'processed_text': basic_preprocess_text(content)  # cleaned text
                })
                doc_id_counter += 1  # increment doc ID

    return pd.DataFrame(documents_list)  # return as DataFrame

# Build BIM model and return ranked results for a query
def build_and_query_bim(documents_df, query_text):
    original_docs = documents_df['text'].tolist()  # store original texts
    tokenized_docs = [doc.split() for doc in documents_df['processed_text']]  # tokenize processed docs
    processed_query = basic_preprocess_text(query_text)  # clean query
    query_tokens = set(processed_query.split())  # unique query terms

    # Create vocabulary from all documents
    vocabulary = set(token for doc in tokenized_docs for token in doc)
    N = len(tokenized_docs)  # total number of documents
    bim_scores = np.zeros(N)  # initialize BIM scores

    # Score calculation for each query term
    for term in query_tokens:
        if term not in vocabulary:  # skip if term not in corpus
            continue
        n_t = sum(term in doc for doc in tokenized_docs)  # number of docs containing term
        weight = math.log((N - n_t + 0.5) / (n_t + 0.5)) if n_t < N else 0  # BIM weight
        for i, doc in enumerate(tokenized_docs):  # check each document
            if term in doc:  # add weight if term present
                bim_scores[i] += weight

    # Rank results by score (descending)
    results = sorted(
        zip(bim_scores, documents_df['doc_id'], original_docs),
        key=lambda x: x[0],
        reverse=True
    )
    return pd.DataFrame(results, columns=['bim_score', 'doc_id', 'text'])  # return results

# Precision@K: fraction of top-K docs that are relevant
def calculate_precision_at_k(results_df, relevant_doc_ids, k):
    if k <= 0:
        return 0.0
    top_k_docs = results_df.head(k)['doc_id']  # top K retrieved docs
    retrieved_relevant = sum(doc_id in relevant_doc_ids for doc_id in top_k_docs)  # relevant retrieved
    return retrieved_relevant / k  # precision value

# Generate Precision-Recall curve points
def calculate_pr_curve_points(results_df, relevant_doc_ids):
    total_relevant = len(relevant_doc_ids)  # total relevant documents
    if total_relevant == 0:  # edge case
        return [(0.0, 1.0)]

    retrieved_count = 0  # number of retrieved docs
    relevant_retrieved = 0  # number of relevant docs retrieved
    pr_points = [(0.0, 1.0)]  # start at recall=0, precision=1

    for _, row in results_df.iterrows():
        retrieved_count += 1  # increment retrieved docs
        if row['doc_id'] in relevant_doc_ids:  # check relevance
            relevant_retrieved += 1
        precision = relevant_retrieved / retrieved_count  # calculate precision
        recall = relevant_retrieved / total_relevant  # calculate recall
        pr_points.append((recall, precision))  # add point

    if pr_points[-1][0] < 1.0:  # ensure curve reaches recall=1
        pr_points.append((1.0, pr_points[-1][1]))

    return pr_points

# Main execution block
if __name__ == "__main__":
    documents_directory = 'documents'  # folder with txt files
    documents_df = read_and_preprocess_documents(documents_directory)  # load and preprocess

    print("\nFinal DataFrame with Processed Text:")
    print(documents_df)  # show dataframe

    user_query_bim = input("Enter your search query for BIM: ")  # take query
    bim_results = build_and_query_bim(documents_df, user_query_bim)  # get results

    print(f"\nBIM Results for query: '{user_query_bim}'")
    print(bim_results)  # show ranked results

    relevant_docs_for_query = {2, 3}  # define relevant docs manually
    print(f"\nAssuming relevant documents for a query are: {relevant_docs_for_query}\n")

    if not bim_results.empty:
        print("--- Evaluating BIM Results ---")
        for k in [1, 2, 3]:  # evaluate precision at different cutoffs
            precision_at_k_bim = calculate_precision_at_k(bim_results, relevant_docs_for_query, k)
            print(f"Precision @ {k} (BIM): {precision_at_k_bim:.2f}")

        # Compute PR curve
        pr_points_bim = calculate_pr_curve_points(bim_results, relevant_docs_for_query)
        print("\nPR Curve Points (BIM):", pr_points_bim)

        # Plot PR curve
        plt.figure()
        plt.plot([p[0] for p in pr_points_bim], [p[1] for p in pr_points_bim], marker='o')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (BIM)')
        plt.grid(True)
        plt.show()
    else:
        print("No BIM results found. Please check the query/documents.")
