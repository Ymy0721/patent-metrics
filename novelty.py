import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba  # Added for jieba tokenization

def load_stopwords(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        return [word.strip() for word in f if word.strip()]

def jieba_tokenizer(text):
    return jieba.lcut(text)  # Use jieba to perform tokenization

def compute_novelty_scores(corpus, stopwords):
    vectorizer = TfidfVectorizer(stop_words=stopwords, tokenizer=jieba_tokenizer, norm=None)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    scores = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(i)
        sum_tfidf = row.sum()
        count_terms = (row != 0).sum()
        score = sum_tfidf / count_terms if count_terms else 0
        scores.append(score)
    return scores

def calculate_novelty(csv_path, invention_col, abstract_col, stopwords_path):
    # Load stop words
    stopwords = load_stopwords(stopwords_path)
    
    # Read CSV data
    df = pd.read_csv(csv_path)
    
    # Ensure the required columns exist
    for col in [invention_col, abstract_col]:
        if col not in df.columns:
            print(f"Column '{col}' not found in CSV.")
            sys.exit(1)
    
    # Fill missing values and prepare corpora
    invention_corpus = df[invention_col].fillna("").tolist()
    abstract_corpus = df[abstract_col].fillna("").tolist()
    
    # Compute novelty scores for each column separately
    df['novelty_invention'] = compute_novelty_scores(invention_corpus, stopwords)
    df['novelty_abstract'] = compute_novelty_scores(abstract_corpus, stopwords)
    
    return df

if __name__ == "__main__":
    # Parameters (modify as necessary)
    csv_input_path = './data/raw/patents.csv'
    invention_col = '发明名称'
    abstract_col = '摘要'
    stopwords_file = './data/stop_words/third-part.txt'
    output_path = './data/output/novelty_output.csv'
    
    result_df = calculate_novelty(csv_input_path, invention_col, abstract_col, stopwords_file)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Novelty scores saved to {output_path}")