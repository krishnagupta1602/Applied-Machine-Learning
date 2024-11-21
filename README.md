def remove_stopwords(corpus):
    """
    Remove stopwords from a list of sentences.

    Parameters:
    - corpus: list of str, input sentences.

    Returns:
    - list of str, sentences without stopwords.
    """
    stop_words = set(stopwords.words('english'))  # Get the list of English stopwords
    cleaned_corpus = []
    
    for sentence in corpus:
        words = sentence.split()  # Split sentence into words
        filtered_words = [word for word in words if word.lower() not in stop_words]
        cleaned_corpus.append(' '.join(filtered_words))  # Reconstruct the sentence
    
    return cleaned_corpus
    
    
    from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def get_top_ngrams(corpus, n=1, top_k=10):
    """
    Extract top n-grams from a list of text.
    
    Parameters:
    - corpus: list of str, input text.
    - n: int, n-gram size (1 for unigram, 2 for bigram, 3 for trigram).
    - top_k: int, number of top n-grams to return.
    
    Returns:
    - DataFrame with n-grams and their frequencies.
    """
    # Initialize CountVectorizer for n-grams
    vectorizer = CountVectorizer(ngram_range=(n, n))
    X = vectorizer.fit_transform(corpus)
    
    # Summing up the counts of each n-gram
    ngram_counts = X.toarray().sum(axis=0)
    ngram_features = vectorizer.get_feature_names_out()
    
    # Create a DataFrame with n-grams and their frequencies
    ngram_freq = pd.DataFrame({'ngram': ngram_features, 'frequency': ngram_counts})
    
    # Sort and get top k
    ngram_freq = ngram_freq.sort_values(by='frequency', ascending=False).head(top_k)
    
    return ngram_freq

# Example usage
corpus = [
    "This is a sample text with sample words.",
    "Text analysis is an interesting field.",
    "We are exploring n-grams from this text data."
]

# Top 10 unigrams
print("Top Unigrams:")
print(get_top_ngrams(corpus, n=1, top_k=10))

# Top 10 bigrams
print("\nTop Bigrams:")
print(get_top_ngrams(corpus, n=2, top_k=10))

# Top 10 trigrams
print("\nTop Trigrams:")
print(get_top_ngrams(corpus, n=3, top_k=10))







\bcds?\b(?:\s+accounts?)?





\b(?:brokerage(?:\s+(?:account|section|tab|accounts))?|brokers?|broker)\b




import torch

# Assume you have your trained model
# model = ClassifierNN()
# (Train your model here)

# Save the model
torch.save(model.state_dict(), "pretrained_model.pth")

# To load the model later
# 1. Create a new instance of the model
loaded_model = ClassifierNN()

# 2. Load the saved state dictionary into the model
loaded_model.load_state_dict(torch.load("pretrained_model.pth"))

# 3. Set the model to evaluation mode
loaded_model.eval()

# Now you can use loaded_model for predictions




import torch
import numpy as np

# Assume the model is already trained and loaded here
# model = ClassifierNN()
# model.load_state_dict(torch.load("model.pth"))  # Load if needed

# Example new entry (f3, f4)
new_entry = [2.5, 3.5]  # Replace with the actual new values of f3 and f4

# Preprocess the new entry (use the same scaler that was used during training)
new_entry_scaled = scaler.transform([new_entry])  # Scaling it just like during training

# Convert to PyTorch tensor
new_entry_tensor = torch.tensor(new_entry_scaled, dtype=torch.float32)

# Make the model evaluation ready
model.eval()

# Forward pass to get the prediction
with torch.no_grad():  # No need to compute gradients for prediction
    output = model(new_entry_tensor)
    predicted_label = torch.round(output)  # Round the output (0 or 1 for binary classification)

# Print the predicted label
print(f"Predicted label for the new entry: {predicted_label.item()}")






import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming data and labels are stored in X and y
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Define the neural network
class ClassifierNN(nn.Module):
    def __init__(self):
        super(ClassifierNN, self).__init__()
        self.fc1 = nn.Linear(4, 32)  # input layer
        self.fc2 = nn.Linear(32, 16)  # hidden layer
        self.fc3 = nn.Linear(16, 1)   # output layer
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Initialize the model, loss function, and optimizer
model = ClassifierNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(torch.tensor(X_train, dtype=torch.float32))
    loss = criterion(outputs.squeeze(), torch.tensor(y_train, dtype=torch.float32))
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(torch.tensor(X_val, dtype=torch.float32))
        val_loss = criterion(val_outputs.squeeze(), torch.tensor(y_val, dtype=torch.float32))
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')






import pandas as pd

# Example function to get score (replace this with your actual function)
def get_score(row):
    # Placeholder for score logic based on the row
    return [row['column1'] * 2]  # Example return, modify as needed

# Function to process the dataframe in chunks
def process_in_batches(df, batch_size):
    final_scores = []
    
    # Iterate through the dataframe in chunks
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i:i + batch_size]
        
        # Process each row in the chunk
        for index, row in chunk.iterrows():
            score = get_score(row)
            final_scores.append(score)
    
    return final_scores

# Example dataframe (replace with your actual dataframe)
data = {'column1': range(1586)}  # Replace with actual data
df = pd.DataFrame(data)

# Process the dataframe in 20 batches
batch_size = len(df) // 20  # Calculate the batch size (around 79 per batch)
final_scores = process_in_batches(df, batch_size)

print(len(final_scores))  # To check the total number of scores processed



from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# List of sentences (all form one document)
sentences = [
    "This is the first sentence.",
    "This is the second sentence.",
    "And this is the third one."


# Join all sentences into a single document
document = " ".join(sentences)

# Initialize TfidfVectorizer and fit on the whole document
vectorizer = TfidfVectorizer()

# Fit on the combined document (treating it as one large document)
vectorizer.fit([document])

# Get the IDF values from the fitted vectorizer
idf_values = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

# Initialize an empty list to store the TF-IDF matrix (each row will be a sentence)
tfidf_matrix = []

# For each sentence, calculate its term frequencies and multiply by the IDF values
for sentence in sentences:
    # Get the term frequencies (TF) for the sentence
    word_counts = vectorizer.transform([sentence]).toarray().flatten()
    
    # Calculate the TF-IDF by multiplying TF with IDF
    tfidf_scores = word_counts * np.array([idf_values.get(word, 0) for word in vectorizer.get_feature_names_out()])
    
    # Append the sentence's TF-IDF scores to the matrix
    tfidf_matrix.append(tfidf_scores)

# Convert to a NumPy array for better printing
tfidf_matrix = np.array(tfidf_matrix)

# Print the final TF-IDF matrix (one array per sentence)
print("TF-IDF Matrix (each row corresponds to a sentence):")
print(tfidf_matrix)

# Get feature names (words) for reference (optional)
feature_names = vectorizer.get_feature_names_out()
print("\nFeature Names (columns in the matrix):", feature_names)











from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# List of sentences (all form one document)
sentences = [
    "This is the first sentence.",
    "This is the second sentence.",
    "And this is the third one."
]

# Join all sentences into a single document
document = " ".join(sentences)

# Initialize TfidfVectorizer and fit on the whole document
vectorizer = TfidfVectorizer()

# Fit on the combined document
vectorizer.fit([document])

# Get the IDF values from the fitted vectorizer
idf_values = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

# Initialize an empty list to store the TF-IDF for each sentence
tfidf_per_sentence = []

# For each sentence, calculate its term frequencies and then multiply by the IDF values
for sentence in sentences:
    word_counts = vectorizer.transform([sentence]).toarray().flatten()
    tfidf_scores = word_counts * np.array([idf_values.get(word, 0) for word in vectorizer.get_feature_names_out()])
    tfidf_per_sentence.append(tfidf_scores)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Display the TF-IDF results for each sentence
for i, sentence in enumerate(sentences):
    print(f"Sentence {i + 1}: {sentence}")
    for word, score in zip(feature_names, tfidf_per_sentence[i]):
        if score > 0:
            print(f"  {word}: {score}")
            







from sklearn.feature_extraction.text import TfidfVectorizer

# List of sentences (all form one document)
sentences = [
    "This is the first sentence.",
    "This is the second sentence.",
    "And this is the third one."
]

# Join all sentences into a single document
document = " ".join(sentences)

# Initialize the TfidfVectorizer and fit it on the entire document (as one unit)
vectorizer = TfidfVectorizer()

# Fit on the combined document but do not transform yet
vectorizer.fit([document])

# Now transform each sentence individually using the fitted vectorizer
tfidf_matrix = vectorizer.transform(sentences)

# Get the feature names and the TF-IDF scores for each sentence
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray()

# Display the TF-IDF results for each sentence
for i, sentence in enumerate(sentences):
    print(f"Sentence {i + 1}: {sentence}")
    for word, score in zip(feature_names, tfidf_scores[i]):
        if score > 0:
            print(f"  {word}: {score}")
            



import numpy as np

def otsu_threshold(L):
    # Sort the data
    L = np.sort(L)
    
    # Total number of data points
    total = len(L)
    
    # Compute cumulative sums and means
    sumT = np.sum(L)
    weightB, sumB = 0, 0
    max_var, threshold = 0, 0
    
    for i in range(total):
        weightB += 1          # Number of elements in class B
        weightF = total - weightB  # Number of elements in class F
        if weightF == 0:
            break
        
        sumB += L[i]              # Sum of elements in class B
        meanB = sumB / weightB     # Mean of class B
        meanF = (sumT - sumB) / weightF  # Mean of class F
        
        # Between-class variance
        between_var = weightB * weightF * (meanB - meanF) ** 2
        
        # Maximize between-class variance
        if between_var > max_var:
            max_var = between_var
            threshold = L[i]       # The current value as the threshold
    
    return threshold

# Example usage:
L = [3, 5, 10, 20, 50, 100, 200, 300]
threshold = otsu_threshold(L)
print(f"Otsu's Threshold: {threshold}")






import pandas as pd
from itertools import combinations

def find_zero_sum_numbers(df):
    final_numbers = []
    indices_to_exclude = set()  # Keep track of rows that have already been used

    # Iterate over all possible combination sizes (from 1 to len(df))
    for r in range(1, len(df) + 1):
        # Get combinations of row indices that are not in the exclusion list
        available_indices = [i for i in range(len(df)) if i not in indices_to_exclude]
        for combo in combinations(available_indices, r):
            # Calculate the sum of 'debit' and 'amount' for this combination
            debit_sum = df.iloc[list(combo)]['debit'].sum()
            amount_sum = df.iloc[list(combo)]['amount'].sum()
            
            # Check if both sums are zero
            if debit_sum == 0 and amount_sum == 0:
                # Add the 'number' values to the final list
                final_numbers.extend(df.iloc[list(combo)]['number'].tolist())
                
                # Add these indices to the exclusion list
                indices_to_exclude.update(combo)
                
                # Break to move to the next iteration since we've found a valid combination
                break
    
    return final_numbers

# Sample DataFrame
data = {
    'number': [1, 2, 3, 4, 5],
    'debit': [100, -100, 50, -50, 0],
    'amount': [200, -200, 50, -50, 0]
}
df = pd.DataFrame(data)

# Get the final numbers that meet the condition
final_numbers = find_zero_sum_numbers(df)
print(final_numbers)









df.apply(lambda row: row['dr'] if abs(row['dr']) >= abs(row['cr']) else row['cr'], axis=1)


import pandas as pd
from itertools import combinations

def find_zero_sum_numbers(df):
    final_numbers = []
    indices_to_exclude = set()  # Keep track of rows that have already been used

    # Iterate over all possible combination sizes (from 1 to len(df))
    for r in range(1, len(df) + 1):
        # Get combinations of row indices that are not in the exclusion list
        available_indices = [i for i in range(len(df)) if i not in indices_to_exclude]
        for combo in combinations(available_indices, r):
            # Calculate the sum of 'debit' and 'amount' for this combination
            debit_sum = df.loc[list(combo), 'debit'].sum()
            amount_sum = df.loc[list(combo), 'amount'].sum()
            
            # Check if both sums are zero
            if debit_sum == 0 and amount_sum == 0:
                # Add the 'number' values to the final list
                final_numbers.extend(df.loc[list(combo), 'number'].tolist())
                
                # Add these indices to the exclusion list
                indices_to_exclude.update(combo)
                
                # Break to move to the next iteration since we've found a valid combination
                break
    
    return final_numbers

# Sample DataFrame
data = {
    'number': [1, 2, 3, 4, 5],
    'debit': [100, -100, 50, -50, 0],
    'amount': [200, -200, 50, -50, 0]
}
df = pd.DataFrame(data)

# Get the final numbers that meet the condition
final_numbers = find_zero_sum_numbers(df)
print(final_numbers)


############################################









import pandas as pd

# Sample DataFrame
data = {
    'ID': [1, 1, 1, 2, 2, 3, 3, 3, 3],
    'Side': ['B', 'S', 'B', 'B', 'B', 'S', 'S', 'B', 'B'],
    'Value': [100, 200, 150, 110, 120, 300, 400, 250, 260]
}
df = pd.DataFrame(data)

# Function to apply conditions
def apply_wash_label(group):
    sides_present = group['Side'].unique()
    if 'B' in sides_present and 'S' in sides_present:
        # Further checks for each subgroup if needed
        # For simplicity, adding 'wash' to all rows in this group
        group['Label'] = 'wash'
    else:
        group['Label'] = ''
    return group

# Apply the function to each group based on 'ID'
df = df.groupby('ID').apply(apply_wash_label)

# Display the resulting DataFrame
print(df)





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
tfidf_matrix = vectorizer.fit_transform(corpus).toarray()

# Calculate the length-normalized TF-IDF score for each sentence
sentence_scores = np.sum(tfidf_matrix, axis=1) / np.count_nonzero(tfidf_matrix, axis=1)

# Rank sentences by their normalized scores
ranked_sentences = [corpus[i] for i in np.argsort(-sentence_scores)]

# Display ranked sentences
print("Ranked sentences:")
for rank, sentence in enumerate(ranked_sentences, 1):
    print(f"{rank}: {sentence}")
    







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
