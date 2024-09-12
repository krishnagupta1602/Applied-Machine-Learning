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
