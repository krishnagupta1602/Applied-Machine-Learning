def tokenize(text):
    """
    Tokenizes the input text into a set of words.
    Converts to lowercase and splits by whitespace.
    """
    return set(text.lower().split())

def jaccard_similarity(set1, set2):
    """
    Computes the Jaccard similarity between two sets.
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(intersection) / len(union)

def calculate_jaccard_similarity(sentences, summary):
    """
    Calculate the Jaccard similarity between each sentence and the summary.

    Parameters:
    - sentences: A list of sentence strings.
    - summary: A summary string.

    Returns:
    - A list of Jaccard similarity scores, where each score is the similarity of a sentence to the summary.
    """
    summary_tokens = tokenize(summary)
    similarities = []

    for sentence in sentences:
        sentence_tokens = tokenize(sentence)
        similarity = jaccard_similarity(sentence_tokens, summary_tokens)
        similarities.append(similarity)

    return similarities

# Example usage:
sentences = [
    "This is the first sentence.",
    "Here is another sentence.",
    "This sentence is similar to the summary.",
    "An unrelated sentence.",
    "Completely different from the summary.",
    "The summary and this sentence have some words in common.",
    "Totally irrelevant sentence.",
    "This sentence might be somewhat related.",
    "A completely unrelated statement.",
    "Here we have a different statement altogether."
]

summary = "This is a summary that somewhat relates to the first few sentences."

scores = calculate_jaccard_similarity(sentences, summary)
print(scores)





import re

text = "Find 123 in this, also look for numbers like 456,789, or even in-text 101dalmatians."
numbers = re.findall(r'\b\d+\b', text)
print(numbers)




from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

# Sample corpus
corpus = [
    "This is the first sentence.",
    "This sentence is about natural language processing.",
    "We are learning how to rank sentences using TF-IDF.",
    "TF-IDF stands for Term Frequency-Inverse Document Frequency."
]

# Custom TF-IDF Vectorizer using augmented frequency
class AugmentedTfidfVectorizer(TfidfVectorizer):
    def _count_terms(self, raw_documents):
        """Custom count method to apply augmented frequency."""
        X = super().fit_transform(raw_documents)
        X = X.toarray()
        
        # Apply augmented frequency (tf / max_tf in each document)
        for i in range(X.shape[0]):
            max_freq = X[i].max()
            if max_freq > 0:  # Avoid division by zero
                X[i] = X[i] / max_freq
        
        return X

# Initialize custom augmented TF-IDF vectorizer
vectorizer = AugmentedTfidfVectorizer()

# Fit and transform the corpus to compute TF-IDF scores using augmented frequency
tfidf_matrix = vectorizer.fit_transform(corpus)

# Get feature names (terms)
feature_names = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a DataFrame for a better view
tfidf_df = pd.DataFrame(tfidf_matrix, columns=feature_names)

# Calculate sentence importance by summing TF-IDF scores for each sentence
sentence_scores = tfidf_df.sum(axis=1)

# Rank sentences by their importance scores
ranked_indices = sentence_scores.argsort()[::-1]
ranked_sentences = [corpus[i] for i in ranked_indices]

# Display the TF-IDF matrix and ranked sentences
print("TF-IDF Matrix (using augmented frequency):")
print(tfidf_df)
print("\nRanked Sentences by Importance:")
for rank, sentence in enumerate(ranked_sentences, 1):
    print(f"{rank}: {sentence}")
    








from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample corpus
corpus = [
    "This is the first sentence.",
    "This sentence is about natural language processing.",
    "We are learning how to rank sentences using TF-IDF.",
    "TF-IDF stands for Term Frequency-Inverse Document Frequency."
]

# Initialize the TF-IDF vectorizer with sublinear_tf to reduce the impact of sentence length
vectorizer = TfidfVectorizer(sublinear_tf=True, norm='l2')

# Fit and transform the corpus to compute TF-IDF scores
tfidf_matrix = vectorizer.fit_transform(corpus)

# Get feature names (terms)
feature_names = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a DataFrame for a better view
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Calculate sentence importance by summing TF-IDF scores for each sentence
sentence_scores = tfidf_df.sum(axis=1)

# Rank sentences by their importance scores
ranked_indices = sentence_scores.argsort()[::-1]
ranked_sentences = [corpus[i] for i in ranked_indices]

# Display the TF-IDF matrix and ranked sentences
print("TF-IDF Matrix:")
print(tfidf_df)
print("\nRanked Sentences by Importance:")
for rank, sentence in enumerate(ranked_sentences, 1):
    print(f"{rank}: {sentence}")
    








from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Sample corpus with multiple sentences
corpus = [
    "This is the first sentence.",
    "This sentence is about natural language processing.",
    "We are learning how to rank sentences using TF-IDF.",
    "TF-IDF stands for Term Frequency-Inverse Document Frequency."
]

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the corpus to compute TF-IDF scores
tfidf_matrix = vectorizer.fit_transform(corpus)

# Sum the TF-IDF scores for each sentence
sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)

# Rank sentences by their scores
ranked_sentences = [corpus[i] for i in np.argsort(-sentence_scores)]

# Display ranked sentences
print("Ranked sentences:")
for rank, sentence in enumerate(ranked_sentences, 1):
    print(f"{rank}: {sentence}")
    






from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample corpus
corpus = [
    "This is the first sentence.",
    "This sentence is about natural language processing.",
    "We are learning how to rank sentences using BERT embeddings.",
    "BERT embeddings capture the contextual meaning of sentences."
]

# Function to get BERT sentence embeddings
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    # Get the embeddings of the [CLS] token
    cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return cls_embedding

# Get embeddings for all sentences in the corpus
sentence_embeddings = np.vstack([get_sentence_embedding(sentence) for sentence in corpus])

# Compute the similarity of each sentence to the mean embedding (representing the document's overall context)
mean_embedding = np.mean(sentence_embeddings, axis=0)
similarity_scores = cosine_similarity(sentence_embeddings, mean_embedding.reshape(1, -1)).flatten()

# Rank sentences by similarity score
ranked_indices = np.argsort(-similarity_scores)
ranked_sentences = [corpus[i] for i in ranked_indices]

# Display ranked sentences
print("Ranked sentences:")
for rank, sentence in enumerate(ranked_sentences, 1):
    print(f"{rank}: {sentence}")
    








import spacy
import pytextrank

# Load a spaCy model
nlp = spacy.load("en_core_web_sm")

# Add PyTextRank to the pipeline
tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

# Sample text
text = "Your text goes here."

# Process the document
doc = nlp(text)

# Get ranked sentences
for sent in doc._.textrank.summary(limit_phrases=10, limit_sentences=5):
    print(sent)
    







pos_weights = {
    'NN': 1.0,   # Noun, singular or mass
    'NNS': 1.0,  # Noun, plural
    'NNP': 1.0,  # Proper noun, singular
    'NNPS': 1.0, # Proper noun, plural
    'JJ': 1.0,   # Adjective
    'VB': 1.0,   # Verb, base form
    'VBD': 1.0,  # Verb, past tense
    'VBG': 1.0,  # Verb, gerund or present participle
    'VBN': 1.0,  # Verb, past participle
    'VBP': 1.0,  # Verb, non-3rd person singular present
    'VBZ': 1.0   # Verb, 3rd person singular present
}


# Calculate POS tag weighted score
def calculate_pos_tag_score(sentence):
    tokens = word_tokenize(sentence)
    tagged_tokens = pos_tag(tokens)
    pos_score = sum(pos_weights.get(tag, 0) for word, tag in tagged_tokens)
    return pos_score

# Calculate stopword penalty score
def calculate_stopword_score(sentence, stopwords):
    tokens = word_tokenize(sentence)
    stopword_score = -sum(1 for word in tokens if word.lower() in stopwords)
    return stopword_score

# Process each sentence in the call
for sentence in call_data:
    sentence_embedding = get_cls_embedding(sentence)

    # Score 1: Cosine similarity with summary embeddings
    cosine_score = calculate_cosine_similarity(sentence_embedding, summary_embeddings)

    # Score 2: POS tag weighted score
    pos_score = calculate_pos_tag_score(sentence)

    # Score 3: Stopword penalty score
    stopword_score = calculate_stopword_score(sentence, stopwords)

    # Print the scores for the sentence
    print(f"Sentence: {sentence}")
    print(f"Cosine Similarity Score: {cosine_score:.4f}")
    print(f"POS Tag Score: {pos_score:.4f}")
    print(f"Stopword Penalty Score: {stopword_score}")
    print()
