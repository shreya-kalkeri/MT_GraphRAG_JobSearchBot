import json
import gzip
import random
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer

# Define the paths to your JSON files
train_path = r'C:\Users\shrey\Downloads\train.jsonl.gz'
validation_path = r'C:\Users\shrey\Downloads\validation.jsonl.gz'
test_path = r'C:\Users\shrey\Downloads\test.jsonl.gz'

# Function to load JSON data from a gzip file
def load_json_data(filepath):
    with gzip.open(filepath, 'rt', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

# Load the datasets
train_data = load_json_data(train_path)
validation_data = load_json_data(validation_path)
test_data = load_json_data(test_path)

def extract_fields(data):
    sentence_pairs = [(item['sentence1'], item['sentence2']) for item in data]
    scores = [item['score'] for item in data]
    return sentence_pairs, scores

# Extract fields from the datasets
train_sentence_pairs, train_scores = extract_fields(train_data)
validation_sentence_pairs, validation_scores = extract_fields(validation_data)
test_sentence_pairs, test_scores = extract_fields(test_data)

# Print some of the extracted data to verify
print(train_sentence_pairs[:5], train_scores[:5])
print(validation_sentence_pairs[:5], validation_scores[:5])
print(test_sentence_pairs[:5], test_scores[:5])

def sample_subset(sentence_pairs, scores, num_samples):
    sampled_indices = random.sample(range(len(sentence_pairs)), num_samples)
    sampled_sentence_pairs = [sentence_pairs[i] for i in sampled_indices]
    sampled_scores = [scores[i] for i in sampled_indices]
    return sampled_sentence_pairs, sampled_scores

# Sample subsets
train_sentence_pairs, train_scores = sample_subset(train_sentence_pairs, train_scores, 500)
validation_sentence_pairs, validation_scores = sample_subset(validation_sentence_pairs, validation_scores, 200)
test_sentence_pairs, test_scores = sample_subset(test_sentence_pairs, test_scores, 200)

#Generate Embeddings
# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embeddings(sentence_pairs):
    sentences = [sentence for pair in sentence_pairs for sentence in pair]
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the embeddings from the [CLS] token
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings

# Generate embeddings for each subset
train_embeddings = get_embeddings(train_sentence_pairs)
validation_embeddings = get_embeddings(validation_sentence_pairs)
test_embeddings = get_embeddings(test_sentence_pairs)

# Calculate cosine similarity
def calculate_similarity(embeddings, num_pairs):
    similarities = []
    for i in range(num_pairs):
        emb1 = embeddings[i]
        emb2 = embeddings[i + num_pairs]
        sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))
        similarities.append(sim[0][0])
    return np.array(similarities)

train_similarities = calculate_similarity(train_embeddings, len(train_sentence_pairs))
validation_similarities = calculate_similarity(validation_embeddings, len(validation_sentence_pairs))
test_similarities = calculate_similarity(test_embeddings, len(test_sentence_pairs))

# Evaluate the performance
def evaluate_similarity(similarities, ground_truth):
    pearson_corr = pearsonr(similarities, ground_truth)[0]
    spearman_corr = spearmanr(similarities, ground_truth)[0]
    return pearson_corr, spearman_corr

train_pearson, train_spearman = evaluate_similarity(train_similarities, train_scores)
validation_pearson, validation_spearman = evaluate_similarity(validation_similarities, validation_scores)
test_pearson, test_spearman = evaluate_similarity(test_similarities, test_scores)

print(f"Train - Pearson Correlation: {train_pearson}, Spearman Correlation: {train_spearman}")
print(f"Validation - Pearson Correlation: {validation_pearson}, Spearman Correlation: {validation_spearman}")
print(f"Test - Pearson Correlation: {test_pearson}, Spearman Correlation: {test_spearman}")

#SentenceBERT

# Load Sentence-BERT model
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

def get_sbert_embeddings(sentence_pairs):
    sentences = [sentence for pair in sentence_pairs for sentence in pair]
    embeddings = sbert_model.encode(sentences)
    return embeddings

# Generate embeddings for each subset using SBERT
train_sbert_embeddings = get_sbert_embeddings(train_sentence_pairs)
validation_sbert_embeddings = get_sbert_embeddings(validation_sentence_pairs)
test_sbert_embeddings = get_sbert_embeddings(test_sentence_pairs)

# Calculate cosine similarity using SBERT embeddings
train_sbert_similarities = calculate_similarity(train_sbert_embeddings, len(train_sentence_pairs))
validation_sbert_similarities = calculate_similarity(validation_sbert_embeddings, len(validation_sentence_pairs))
test_sbert_similarities = calculate_similarity(test_sbert_embeddings, len(test_sentence_pairs))

# Evaluate the performance using SBERT embeddings
train_sbert_pearson, train_sbert_spearman = evaluate_similarity(train_sbert_similarities, train_scores)
validation_sbert_pearson, validation_sbert_spearman = evaluate_similarity(validation_sbert_similarities, validation_scores)
test_sbert_pearson, test_sbert_spearman = evaluate_similarity(test_sbert_similarities, test_scores)

print(f"SBERT Train - Pearson Correlation: {train_sbert_pearson}, Spearman Correlation: {train_sbert_spearman}")
print(f"SBERT Validation - Pearson Correlation: {validation_sbert_pearson}, Spearman Correlation: {validation_sbert_spearman}")
print(f"SBERT Test - Pearson Correlation: {test_sbert_pearson}, Spearman Correlation: {test_sbert_spearman}")

print('Success')