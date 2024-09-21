#-------------------------------------------------------------------------
# AUTHOR: Jeffrey Ni
# FILENAME: indexing.py
# SPECIFICATION: This script processes collection.csv, removes stopwords,
# applies stemming, and constructs a TF-IDF document-term matrix.
# FOR: CS 4250- Assignment #1
# TIME SPENT: Approximately 3 hours
#-----------------------------------------------------------*/

import csv
import math  # Required for IDF calculations

# Path to the CSV file
csv_file_path = r"C:\Users\Jeffrey\OneDrive\Desktop\CS4250\Assignment 1\collection.csv"

# Function to read the CSV and return documents and labels
def read_csv(file_path):
    documents, labels = [], []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if row:
                documents.append(row[0])
                labels.append(row[1] if len(row) > 1 else 'I')  # Default to 'I'
    return documents, labels

# Function to remove stopwords
def remove_stopwords(docs, stopwords):
    return [' '.join(word for word in doc.split() if word not in stopwords) for doc in docs]

# Function to apply stemming
def stem_documents(docs, stem_map):
    return [' '.join(stem_map.get(word, word) for word in doc.split()) for doc in docs]

# Function to calculate TF-IDF matrix
def calculate_tfidf(docs):
    term_frequencies = []
    for doc in docs:
        term_count = {}
        for word in doc.split():
            term_count[word] = term_count.get(word, 0) + 1
        term_frequencies.append(term_count)

    idf_values = {}
    total_docs = len(docs)

    for term in set(word for doc in docs for word in doc.split()):
        doc_freq = sum(1 for doc in docs if term in doc.split())
        idf_values[term] = math.log10(total_docs / doc_freq) if doc_freq > 0 else 0

    tfidf_matrix = []
    for i, term_count in enumerate(term_frequencies):
        tfidf_row = []
        for term in idf_values:
            tf = term_count.get(term, 0) / len(docs[i].split())
            tfidf_row.append(tf * idf_values[term])
        tfidf_matrix.append(tfidf_row)

    return tfidf_matrix, list(idf_values.keys())

# Function to score documents based on a query
def score_documents(tfidf_matrix, query, stopwords, unique_terms):
    filtered_query = ' '.join(word for word in query.split() if word not in stopwords)
    query_vector = [1 if term in filtered_query.split() else 0 for term in unique_terms]
    
    scores = []
    for doc_vector in tfidf_matrix:
        score = sum(q * d for q, d in zip(query_vector, doc_vector))
        scores.append(score)
    return scores

# Load documents and labels
doc_texts, doc_labels = read_csv(csv_file_path)

# Define stopwords and stemming mapping
stopwords = {'I', 'and', 'She', 'They', 'her', 'their'}
stem_mapping = {
    "cats": "cat",
    "dogs": "dog",
    "loves": "love",
}

# Process documents
cleaned_docs = remove_stopwords(doc_texts, stopwords)
stemmed_docs = stem_documents(cleaned_docs, stem_mapping)

# Calculate TF-IDF matrix
tfidf_matrix, unique_terms = calculate_tfidf(stemmed_docs)

# Print the TF-IDF matrix
print("\nTF-IDF Document-Term Matrix:")
print("\t" + "\t".join(unique_terms))
for i, row in enumerate(tfidf_matrix):
    print(f"D{i + 1}\t" + "\t".join(f"{value:.4f}" for value in row))

# Score documents based on a query
search_query = "cat and dogs"
document_scores = score_documents(tfidf_matrix, search_query, stopwords, unique_terms)

# Output document scores
print("\nDocument Scores based on Query:")
for i, score in enumerate(document_scores):
    print(f"Document D{i + 1} - Score: {score:.4f}")

# Calculate precision and recall
threshold = 0.1
true_pos = false_pos = false_neg = 0

# Evaluate the scores against the labels
for score, label in zip(document_scores, doc_labels):
    if score >= threshold:
        true_pos += int(label == 'R')  
        false_pos += int(label != 'R')  
    else:
        false_neg += int(label == 'R')

# Calculate precision
precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0

# Calculate recall
recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

# Present precision and recall results differently
print("\nEvaluation Metrics:")
print(f"Precision: {precision:.4f} | Recall: {recall:.4f}")
