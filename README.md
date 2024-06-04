import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import numpy as np

# Download stopwords if not already done
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to preprocess the text
def preprocess(text):
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Assuming df is your DataFrame with a column 'text'
df = pd.DataFrame({
    'text': [
        'The cat in the hat disabled the hat.',
        'The hat is a good cat.',
        'Dogs are friendly animals.',
        'Cats and dogs are not friends.',
        'The quick brown fox jumps over the lazy dog.'
    ]
})
df['processed_text'] = df['text'].apply(preprocess)

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(df['processed_text'])

# Initialize LDA with evaluate_every parameter
lda = LatentDirichletAllocation(n_components=2, random_state=0, evaluate_every=1, max_iter=10)

# List to store perplexity values
perplexities = []

# Custom function to fit LDA and store perplexity
def fit_lda_and_record_perplexity(lda, dtm, perplexities):
    for i in range(1, lda.max_iter + 1):
        lda.partial_fit(dtm)
        perplexity = lda.perplexity(dtm)
        perplexities.append(perplexity)
        print(f"Iteration {i}, Perplexity: {perplexity}")

# Fit the LDA model and record perplexities
fit_lda_and_record_perplexity(lda, dtm, perplexities)

# Plot perplexity values
plt.plot(range(1, lda.max_iter + 1), perplexities)
plt.xlabel('Iteration')
plt.ylabel('Perplexity')
plt.title('LDA Perplexity over Iterations using TF-IDF')
plt.show()

# Function to display topics and their top words with frequencies
def display_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_words_freq = [topic[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics[topic_idx] = list(zip(topic_words, topic_words_freq))
    return topics

no_top_words = 10
tf_feature_names = vectorizer.get_feature_names_out()
topics = display_topics(lda, tf_feature_names, no_top_words)

# Print the topics with top words and their frequencies
for topic, words in topics.items():
    print(f"Topic {topic}:")
    for word, freq in words:
        print(f"{word}: {freq}")
    print("\n")

# Transform the data to get the topic distribution for each document
topic_distribution = lda.transform(dtm)


# Assign each document to the topic with the highest probability
df['topic'] = topic_distribution.argmax(axis=1)

# Print the topic assignments
print(df[['text', 'processed_text', 'topic']])



def plot_topic_word_frequencies(topics):
    for topic_idx, words in topics.items():
        words_list, frequencies = zip(*words)
        plt.figure(figsize=(10, 6))
        plt.bar(words_list, frequencies, color='blue')
        plt.xlabel('Words')
        plt.ylabel('Frequencies')
        plt.title(f'Topic {topic_idx} Top Words')
        plt.xticks(rotation=45)
        plt.show()

plot_topic_word_frequencies(topics)


https://github.com/sethns/Latent-Dirichlet-Allocation-LDA-/blob/main/Topic%20Modeling%20_%20Extracting%20Topics_%20Using%20Sklearn.ipynb 

def get_word_frequencies(dtm, topic_assignments, feature_names, n_topics):
    word_freq = {i: {} for i in range(n_topics)}
    for doc_idx, topic in enumerate(topic_assignments):
        feature_index = dtm[doc_idx, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [dtm[doc_idx, x] for x in feature_index])
        for word_idx, score in tfidf_scores:
            word = feature_names[word_idx]
            if word in word_freq[topic]:
                word_freq[topic][word] += score
            else:
                word_freq[topic][word] = score
    return word_freq

# Calculate word frequencies for each topic
word_frequencies = get_word_frequencies(dtm, df['topic'], tf_feature_names, lda.n_components)

# Plot bar graphs for each topic based on word frequencies
def plot_topic_word_frequencies(word_frequencies, no_top_words):
    for topic_idx, freqs in word_frequencies.items():
        sorted_words = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:no_top_words]
        words, frequencies = zip(*sorted_words)
        plt.figure(figsize=(10, 6))
        plt.bar(words, frequencies, color='blue')
        plt.xlabel('Words')
        plt.ylabel('Frequencies')
        plt.title(f'Topic {topic_idx} Top Words')
        plt.xticks(rotation=45)
        plt.show()

plot_topic_word_frequencies(word_frequencies, no_top_words)


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS

# Download stopwords if not already done
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to preprocess the text
def preprocess(text):
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Sample data
df = pd.DataFrame({
    'text': [
        'The cat in the hat disabled the hat.',
        'The hat is a good cat.',
        'Dogs are friendly animals.',
        'Cats and dogs are not friends.',
        'The quick brown fox jumps over the lazy dog.'
    ]
})
df['processed_text'] = df['text'].apply(preprocess)

# Initialize the CountVectorizer
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(df['processed_text'])

# Initialize LDA
lda = LatentDirichletAllocation(n_components=2, random_state=0, max_iter=10)

# Fit the LDA model
lda.fit(dtm)

# Get the topic-word distributions
topic_word_distributions = lda.components_

# Calculate cosine similarity between topics
topic_similarity = cosine_similarity(topic_word_distributions)

print("Cosine Similarity Between Topics:")
print(topic_similarity)

# Use MDS to visualize the distance between topics
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
topic_distances = 1 - topic_similarity
pos = mds.fit_transform(topic_distances)

# Plot the topics in 2D space
plt.figure(figsize=(10, 6))
plt.scatter(pos[:, 0], pos[:, 1], marker='o')

for i in range(len(pos)):
    plt.text(pos[i, 0], pos[i, 1], f'Topic {i}', fontsize=12)

plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.title('Topic Distance Visualization using MDS')
plt.show()

