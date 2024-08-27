import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate an example DataFrame with 5 lakh positive amounts
np.random.seed(42)
data = {'amounts': np.random.lognormal(mean=5, sigma=1.5, size=500000)}
df = pd.DataFrame(data)

# Plotting the Probability Density Function using KDE
plt.figure(figsize=(10, 6))
sns.kdeplot(df['amounts'], bw_adjust=0.5)  # Adjust `bw_adjust` for smoother or rougher curves
plt.xlabel('Amount')
plt.ylabel('Density')
plt.title('Probability Density Function of Amounts')
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Example data
amounts = np.array([100, 200, 300, 400, 500, 1000])

# Sort the data
sorted_data = np.sort(amounts)
rank = np.arange(1, len(sorted_data) + 1)

# Log-log plot
plt.figure(figsize=(8, 6))
plt.loglog(sorted_data, rank, marker='o', linestyle='none')
plt.xlabel('Amount')
plt.ylabel('Rank')
plt.title('Log-Log Plot')
plt.show()



import pandas as pd

# Assuming df is your DataFrame and 'category_column' is the categorical column
category_counts = df['category_column'].value_counts(normalize=True) * 100

# Sort the percentages in descending order
category_distribution = category_counts.sort_values(ascending=False)

print(category_distribution)




import pandas as pd
import os

# Sample DataFrame
data = {
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40],
    'C': [100, 200, 300, 400]
}

df = pd.DataFrame(data)

# Define the output folder and file name
output_folder = 'output'  # Specify the folder where you want to save the pickle file
output_filename = 'my_dataframe.pkl'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Construct the full path for the pickle file
output_path = os.path.join(output_folder, output_filename)

# Save the DataFrame to a pickle file
df.to_pickle(output_path)

print(f"DataFrame saved to {output_path}")




from bs4 import BeautifulSoup

# Load the HTML file
with open('your_html_file.html', 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML content with BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Find all tables in the HTML
tables = soup.find_all('table')

# Select the third table (index 2 since it's 0-based)
if len(tables) >= 3:
    third_table = tables[2]

    # Extract rows and columns from the table
    rows = third_table.find_all('tr')
    for row in rows:
        columns = row.find_all(['td', 'th'])  # Find all columns (both td and th)
        column_data = [column.text.strip() for column in columns]  # Extract text from each column
        print(column_data)  # Print or store the data
else:
    print("The HTML file does not contain 3 tables.")






import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Step 1: Read the local HTML file
file_path = 'path/to/your/file.html'  # Replace with the path to your HTML file

# Ensure the file exists
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
else:
    print(f"File not found: {file_path}")
    exit()

# Step 2: Parse the HTML content
soup = BeautifulSoup(content, 'html.parser')

# Step 3: Find the table
table = soup.find('table')  # Adjust the selector if the table has a specific id or class

# Step 4: Extract the headers (optional)
headers = []
for th in table.find_all('th'):
    headers.append(th.text.strip())

# Step 5: Extract the rows
rows = []
for tr in table.find_all('tr'):
    cells = tr.find_all(['td', 'th'])
    row = [cell.text.strip() for cell in cells]
    rows.append(row)

# Step 6: Create a DataFrame
if headers:
    df = pd.DataFrame(rows[1:], columns=headers)  # Assuming first row is header
else:
    df = pd.DataFrame(rows)

# Step 7: Display or save the DataFrame
print(df)
df.to_csv('output.csv', index=False)  # Optionally save to a CSV file



import requests
from bs4 import BeautifulSoup
import pandas as pd

# Step 1: Fetch the webpage
url = 'https://example.com/page-with-table'  # Replace with the URL of the webpage
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    print("Successfully fetched the webpage!")
    # Step 2: Parse the webpage content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Step 3: Find the table
    table = soup.find('table')  # Adjust the selector if the table has a specific id or class

    # Step 4: Extract the headers (optional)
    headers = []
    for th in table.find_all('th'):
        headers.append(th.text.strip())

    # Step 5: Extract the rows
    rows = []
    for tr in table.find_all('tr'):
        cells = tr.find_all(['td', 'th'])
        row = [cell.text.strip() for cell in cells]
        rows.append(row)

    # Step 6: Create a DataFrame
    if headers:
        df = pd.DataFrame(rows[1:], columns=headers)  # Assuming first row is header
    else:
        df = pd.DataFrame(rows)

    # Step 7: Display or save the DataFrame
    print(df)
    df.to_csv('output.csv', index=False)  # Optionally save to a CSV file
else:
    print(f"Failed to fetch the webpage. Status code: {response.status_code}")
    



import pandas as pd

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importances)



# Dictionary to store recall scores for each feature
feature_recall = {}

# Perform cross-validation for the model with all features
baseline_scores = cross_val_score(model, X, y, cv=5, scoring='recall')
baseline_recall = np.mean(baseline_scores)
print(f"Baseline Recall with all features: {baseline_recall}")

# Iterate over each feature to drop it and evaluate recall
for feature in X.columns:
    # Drop the feature
    X_temp = X.drop(columns=[feature])

    # Perform cross-validation
    scores = cross_val_score(model, X_temp, y, cv=5, scoring='recall')

    # Store the average recall score
    feature_recall[feature] = np.mean(scores)
    print(f"Recall without {feature}: {feature_recall[feature]}")

# Convert the dictionary to a DataFrame for easier analysis
recall_drop_df = pd.DataFrame.from_dict(feature_recall, orient='index', columns=['Recall'])
recall_drop_df['Drop in Recall'] = baseline_recall - recall_drop_df['Recall']
recall_drop_df.sort_values(by='Drop in Recall', ascending=False, inplace=True)

print("\nImpact of Dropping Each Feature on Recall:")
print(recall_drop_df)



import pandas as pd

# Sample DataFrame
data = {
    'amounts': [50, 150, 200, 600, 350, 80, 10, 500, 700, 90]
}
df = pd.DataFrame(data)

# Calculate the thresholds
top_25_threshold = df['amounts'].quantile(0.75)
bottom_2_threshold = df['amounts'].quantile(0.02)

# Define the conditions for labeling
conditions = [
    (df['amounts'] >= top_25_threshold),    # Top 25
    (df['amounts'] <= bottom_2_threshold),  # Bottom 2%
]

# Define the corresponding labels
labels = ['high', 'low']

# Apply the conditions and create the new column
df['amount_category'] = pd.cut(df['amounts'],
                               bins=[-float('inf'), bottom_2_threshold, top_25_threshold, float('inf')],
                               labels=['low', 'normal', 'high'])

# Display the updated DataFrame
print(df)







# Define the bin edges
bins = [0, 100, 500, float('inf')]  # float('inf') means "above 500"

# Define the corresponding labels
labels = ['low', 'high', 'very high']

# Create a new column with bucketed data
df['amount_category'] = pd.cut(df['amounts'], bins=bins, labels=labels, right=False)

# Display the updated DataFrame
print(df)







import pandas as pd
import matplotlib.pyplot as plt

# Sample DataFrame
data = {
    'amounts': [100, 150, 200, 150, 100, 200, 300, 100, 200, 300, 400, 500]
}
df = pd.DataFrame(data)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(df['amounts'], bins=10, edgecolor='black')
plt.xlabel('Amounts')
plt.ylabel('Frequency')
plt.title('Histogram of Amounts')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()







# Filter df5 for rows where 'days' column is greater than 100
filtered_df5 = df5[df5["days"] > 100]

# Concatenate the filtered df5 to the final_result DataFrame
final_result = pd.concat([final_result, filtered_df5]).reset_index(drop=True)

# final_result now contains the updated DataFrame with the additional rows from df5



import pandas as pd

# Assuming df1, df2, df3, df4, df5 are your DataFrames
dfs = [df1, df2, df3, df4, df5]

# Start with an empty list to hold the results
result_dfs = []

# Iterate through pairs of consecutive DataFrames
for i in range(len(dfs) - 1):
    # Perform a set difference based on "Item Id"
    diff_df = dfs[i][~dfs[i]["Item Id"].isin(dfs[i + 1]["Item Id"])]
    result_dfs.append(diff_df)

# Combine the results into one DataFrame
final_result = pd.concat(result_dfs).reset_index(drop=True)

# final_result now contains the rows which are in 1 but not 2, 2 but not 3, and so on






set PATH=C:\Python310;%PATH:OldPythonPath;=%


set PATH=%PATH%;C:\Python310


from pathlib import Path
import json

# Get the directory of the current script (main.py)
current_directory = Path(__file__).resolve().parent

# Find the parent directory (where config.json is located)
parent_directory = current_directory.parent

# Construct the path to config.json
config_path = parent_directory / "config.json"

# Load the JSON configuration file
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Example usage of the config
print(config)





import logging
import one
import two
import three

# Configure the logger
logging.basicConfig(
    level=logging.ERROR,  # Set logging level to ERROR
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Logs to a file named "app.log"
        logging.StreamHandler()  # Also logs to the console
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        one.function_one()
    except Exception as e:
        logger.error("Error in module one", exc_info=True)

    try:
        two.function_two()
    except Exception as e:
        logger.error("Error in module two", exc_info=True)

    try:
        three.function_three()
    except Exception as e:
        logger.error("Error in module three", exc_info=True)

if __name__ == "__main__":
    main()


one.py****

import logging

logger = logging.getLogger(__name__)

def function_one():
    try:
        # Your code here
        raise ValueError("An error occurred in one.py")  # Example error
    except Exception as e:
        logger.error("Error in function_one", exc_info=True)
        raise

    



import fitz  # PyMuPDF
from PIL import Image
import pytesseract

def detect_tables_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Use get_pixmap() if you have a newer version of PyMuPDF, or get_image() if it's older.
        # Adjust the code according to the version.
        try:
            # For newer versions of PyMuPDF
            pix = page.get_pixmap()
        except AttributeError:
            # For older versions of PyMuPDF
            pix = page.get_image()  # `get_image()` was the older method

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        text = pytesseract.image_to_string(img)
        if "table" in text.lower():
            print(f"Potential table detected on page {page_num + 1}")
            return True
    return False

pdf_path = "your_pdf_file.pdf"  # Replace with your PDF file path
if detect_tables_with_ocr(pdf_path):
    print("The PDF contains tables (detected via OCR).")
else:
    print("No tables found in the PDF (via OCR).")




timport pandas as pd

lfrom sklearn.feature_extraction.text import TfidfVectorizer
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

def calculate_coherence_score(model, texts, dictionary):
    coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()
topic_numbers = range(2, 11)
coherence_scores = []
perplexities = []

for n_topics in topic_numbers:
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0, max_iter=10)
    lda.fit(dtm)
    
    # Calculate coherence score
    coherence_scores.append(calculate_coherence_score(lda, processed_texts, dictionary))
    
    # Calculate perplexity
    perplexities.append(lda.perplexity(dtm))

# Plot coherence score and perplexity
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(topic_numbers, coherence_scores, marker='o')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence Score')
plt.title('Coherence Score by Number of Topics')

plt.subplot(1, 2, 2)
plt.plot(topic_numbers, perplexities, marker='o')
plt.xlabel('Number of Topics')
plt.ylabel('Perplexity')
plt.title('Perplexity by Number of Topics')

plt.tight_layout()
plt.show()

plt.show()

topic_word_distributions = lda.components_

# Normalize the topic-word distributions to get probabilities
topic_word_distributions /= topic_word_distributions.sum(axis=1)[:, np.newaxis]

# Calculate pairwise distances between topics
distances = pairwise_distances(topic_word_distributions, metric='euclidean')

# Calculate the stress
# Stress is typically calculated using the formula:
# stress = sqrt(sum((d_ij - d_ij')^2) / sum(d_ij^2))
# where d_ij are the original distances and d_ij' are the distances in the lower-dimensional space

# For simplicity, we use the distances directly without transforming to lower-dimensional space
stress = np.sqrt(np.sum((distances - np.mean(distances))**2) / np.sum(distances**2))

print("Distances between topics:")
print(distances)

print(f"Stress: {stress}")

# Input list
input_list = ['a\nb', 'c', '\nd']

# Remove newline characters and join with spaces
combined_string = ' '.join(item.replace('\n', '') for item in input_list)

# Wrap the result in a list
result = [combined_string]

print(result)  # Output: ['ab c d']

import re

text = 'i menas one'
pattern = r'\bmenas\b'

result = re.sub(pattern, '', text).strip()
print(result)

import re

text = 'The term "payment" means the amount due.'

# Define the regex pattern
pattern = r'"payment" means (.+)'

# Use re.search to find the pattern in the text
match = re.search(pattern, text)
if match:
    print(match.group(1))

import re

text_list = ['The term "payment" means the amount due.', 'Hi this is us']

# Define the regex pattern
pattern = r'"payment" means (.+)'

# Use list comprehension to apply the regex and capture the desired text
matches = [re.search(pattern, text).group(1) for text in text_list if re.search(pattern, text)]

print(matches)

import re

text_list = ['int is 2.99%', 'tax rate is 5.5%', 'growth was 12%']

# Define the regex pattern to match a percentage value
pattern = r'\b\d+(\.\d+)?%'

# Use list comprehension to apply the regex and capture the desired text
matches = [re.search(pattern, text).group() for text in text_list if re.search(pattern, text)]

print(matches)
import re

text_list = ['The term "payment" means the amount due.', 'Hi this us']

# Define the regex pattern
pattern = r'.*?"payment" means .+'

# Use list comprehension to find matches
matches = [match.group() for text in text_list for match in [re.search(pattern, text)] if match]

print(matches)


import re

# List of strings
l = ['abc', 'abc $200']

# Regular expression pattern to match dollar amounts
pattern = r'\$\d+'

# Find strings containing a dollar amount
matching_strings = [s for s in l if re.search(pattern, s)]

# Output the result
print(matching_strings)


import pdfplumber

def split_text_at_bold(pdf_path):
    split_texts = []
    current_text = ""
    last_bold = False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract characters with their font info
            chars = page.chars
            
            for char in chars:
                is_bold = 'Bold' in char['fontname']
                if is_bold:
                    if not last_bold and current_text:
                        split_texts.append(current_text.strip())
                        current_text = ""
                    current_text += char['text']
                    last_bold = True
                else:
                    if last_bold and current_text:
                        split_texts.append(current_text.strip())
                        current_text = ""
                    current_text += char['text']
                    last_bold = False

            if current_text:  # Append any remaining text after the last character
                split_texts.append(current_text.strip())
                current_text = ""

    return split_texts

# Example usage
pdf_path = "your_pdf_file.pdf"
split_texts = split_text_at_bold(pdf_path)
for part in split_texts:
    print(part)
    print("---")


from google.cloud import speech_v1p1beta1 as speech
import io
import azure.cognitiveservices.speech as speechsdk
import json
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator



import requests

def assemblyai_wav_to_text(wav_file_path):
    api_key = "YourAssemblyAIKey"
    headers = {"authorization": api_key}

    upload_url = "https://api.assemblyai.com/v2/upload"
    with open(wav_file_path, "rb") as f:
        response = requests.post(upload_url, headers=headers, files={"file": f})
    audio_url = response.json()["upload_url"]

    transcript_request = {
        "audio_url": audio_url
    }
    transcript_url = "https://api.assemblyai.com/v2/transcript"
    transcript_response = requests.post(transcript_url, json=transcript_request, headers=headers)
    transcript_id = transcript_response.json()["id"]

    transcript_result_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    while True:
        result_response = requests.get(transcript_result_url, headers=headers)
        result = result_response.json()
        if result["status"] == "completed":
            print(f"Recognized Text: {result['text']}")
            break
        elif result["status"] == "failed":
            print("Transcription failed")
            break

wav_file_path = "path/to/your/file.wav"
assemblyai_wav_to_text(wav_file_path)







import deepspeech
import wave
import numpy as np

def deepspeech_wav_to_text(wav_file_path):
    model_path = 'path/to/deepspeech/model.pbmm'
    scorer_path = 'path/to/deepspeech/scorer.scorer'
    
    model = deepspeech.Model(model_path)
    model.enableExternalScorer(scorer_path)

    with wave.open(wav_file_path, 'rb') as wf:
        frames = wf.getnframes()
        buffer = wf.readframes(frames)
        data16 = np.frombuffer(buffer, dtype=np.int16)

    text = model.stt(data16)
    print(f"Recognized Text: {text}")

wav_file_path = "path/to/your/file.wav"
deepspeech_wav_to_text(wav_file_path)

SpeechRecognition
Google Cloud Speech-to-Text
Microsoft Azure Speech Service
IBM Watson Speech to Text
AssemblyAI
DeepSpeech
Wav2Vec2 (via Hugging Face's transformers)
PyDub
SpeechBrain
Coqui STT
Kaldi (via PyKaldi)
Vosk (via Kaldi)
These libraries offer a range of options for integrating speech-to-text functionality into Python applications.




Moree 
ChatGPT
Certainly! Here are additional libraries and APIs for speech-to-text conversion in Python:

Houndify API
Sphinx (CMU Sphinx)
Vosk API
Picovoice Leopard
Rev AI
Kaldi (Python interface through PyKaldi)
Whisper (OpenAI)
Amazon Transcribe
Speechmatics
Deepgram
Ailia SDK
AssemblyAI
Nuance Dragon SDK
iSpeech
Vivox
Phonexia Speech Platform
These libraries and APIs provide a wide range of features and capabilities for converting speech to text, accommodating various use cases and requirements.

More
ChatGPT
Here are even more libraries and APIs for speech-to-text conversion in Python:

Librispeech
Baidu PaddlePaddle
Microsoft Cognitive Toolkit (CNTK)
Kaldi ASR (via py-kaldi-asr)
Spacy + pyAudioAnalysis
Snowboy
Sonic SDK
Nuance Mix
PhonicMind
Wit.ai
Aeneas
Kaldi GStreamer Server
OpenSeq2Seq (NVIDIA)
TranscribeMe
Google Speech Engine (via PyDub)
These libraries and APIs span a broad spectrum of capabilities, from highly specialized to general-purpose, ensuring that developers have a rich set of tools to choose from for their speech-to-text needs.

More
ChatGPT
Certainly! Here are even more options for speech-to-text libraries and APIs in Python:

Speechmatics Real-Time ASR
Soniox
Clarifai Speech Recognition
Audeme MOVI Speech Recognizer
Dragonfly (for speech recognition scripting)
Otter.ai API
Speechmatics ASR Cloud
Speechnotes API
Vocapia Research VoxSigma
Kaldi-based STT with py-kaldi-asr
Sensory TrulyHandsfree
Bing Speech API (via Microsoft Cognitive Services)
Nuance Mix
Kaldi Online GMM and NNet3 Chain models
Eesen End-to-End Speech Recognition
Voxpopuli


import numpy as np
import tensorflow as tf
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer

# Preprocess the text data
def preprocess_text(texts):
    # Tokenize the text
    tokenized_texts = [gensim.utils.simple_preprocess(text) for text in texts]
    return tokenized_texts

# Train word2vec embeddings
def train_word2vec(tokenized_texts):
    model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Create a vocabulary and document-term matrix
def create_vocabulary(tokenized_texts):
    vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    dt_matrix = vectorizer.fit_transform(tokenized_texts)
    vocab = vectorizer.get_feature_names_out()
    return dt_matrix, vocab

# Initialize LDA model
def initialize_lda(dt_matrix, num_topics):
    lda = gensim.models.LdaModel(corpus=[[(i, int(count)) for i, count in enumerate(doc)] for doc in dt_matrix.toarray()],
                                 num_topics=num_topics,
                                 id2word=dict(enumerate(vocab)),
                                 passes=10)
    return lda

# Combine word2vec and LDA
class Lda2Vec:
    def __init__(self, word2vec_model, lda_model, num_topics):
        self.word2vec_model = word2vec_model
        self.lda_model = lda_model
        self.num_topics = num_topics

    def fit(self, tokenized_texts, epochs=10):
        for epoch in range(epochs):
            # Optimize the model parameters
            # Update the topic and word vectors
            pass

    def get_topics(self):
        topics = []
        for i in range(self.num_topics):
            topics.append(self.lda_model.show_topic(i))
        return topics

# Example usage
texts = [
    "Natural language processing is a field of artificial intelligence",
    "Machine learning can be applied to natural language processing",
    "Deep learning models are used in many AI applications",
    "AI and machine learning are closely related fields"
]

tokenized_texts = preprocess_text(texts)
word2vec_model = train_word2vec(tokenized_texts)
dt_matrix, vocab = create_vocabulary(tokenized_texts)
lda_model = initialize_lda(dt_matrix, num_topics=2)

lda2vec_model = Lda2Vec(word2vec_model, lda_model, num_topics=2)
lda2vec_model.fit(tokenized_texts)

topics = lda2vec_model.get_topics()
for i, topic in enumerate(topics):
    print(f"Topic {i}: {topic}")


import pandas as pd
from openpyxl import load_workbook

# Load the workbook and the specific sheet
file_path = 'your_file.xlsx'
sheet_name = 'xyz'
workbook = load_workbook(filename=file_path, data_only=True)
sheet = workbook[sheet_name]

# Initialize an empty dictionary to store the data
data = {}

# Iterate over the columns
for col in sheet.iter_cols(values_only=True):
    # The first value in the column is the header
    header = col[0]
    # The rest are the data values
    data[header] = list(col[1:])

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df)

@echo off
REM Optional: Activate the Anaconda environment
call C:\Users\YourUsername\Anaconda3\Scripts\activate.bat base

REM Change directory to where your Python script is located
cd C:\path\to\your\script

REM Run the Python script
python your_script.py


sorted_items = sorted(count_dict.items(), key=lambda item: item[1], reverse=True)
top_2_words = [item[0] for item in sorted_items[:2]]

# Input string
text = "I am an aspiring analyst currently studying"

# List of words to keep
words_to_keep = ["analyst", "studying"]

# Split the string into words
words = text.split()

# Filter words that are in the words_to_keep list
filtered_words = [word for word in words if word in words_to_keep]



pip install PyMuPDF pytesseract Pillow

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

def detect_tables_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        text = pytesseract.image_to_string(img)
        if "table" in text.lower():
            print(f"Potential table detected on page {page_num + 1}")
            return True
    return False

pdf_path = "your_pdf_file.pdf"  # Replace with your PDF file path
if detect_tables_with_ocr(pdf_path):
    print("The PDF contains tables (detected via OCR).")
else:
    print("No tables found in the PDF (via OCR).")
    


# Join the filtered words back into a string
result = ' '.join(filtered_words)

print(result)







import fitz  # PyMuPDF
from PIL import Image
import pytesseract

def detect_tables_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]  # Access the page directly by its index
        pix = page.get_pixmap(







      import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io

def detect_tables_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    if len(doc) > 0:  # Ensure the document has at least one page
        page = doc[0]  # Get the first page
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))  # Convert the pixmap to an image

        text = pytesseract.image_to_string(img)
        if "table" in text.lower():
            print("Potential table detected on the first page.")
            return True
        else:
            print("No table detected on the first page.")
            return False
    else:
        print("The PDF has no pages.")
        return False

pdf_path = "your_pdf_file.pdf"  # Replace with your PDF file path
if detect_tables_with_ocr(pdf_path):
    print("The PDF contains tables (detected via OCR).")
else:
    print("No tables found in the PDF (via OCR).")
    
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        text = pytesseract.image_to_string(img)
        if "table" in text.lower():
            print(f"Potential table detected on page {page_num + 1}")
            return True
    return False

pdf_path = "your_pdf_file.pdf"  # Replace with your PDF file path
if detect_tables_with_ocr(pdf_path):
    print("The PDF contains tables (detected via OCR).")
else:
    print("No tables found in the PDF (via OCR).")



    
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

def detect_tables_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Use get_pixmap() if you have a newer version of PyMuPDF, or get_image() if it's older.
        # Adjust the code according to the version.
        try:
            # For newer versions of PyMuPDF
            pix = page.get_pixmap()
        except AttributeError:
            # For older versions of PyMuPDF
            pix = page.get_image()  # `get_image()` was the older method

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        text = pytesseract.image_to_string(img)
        if "table" in text.lower():
            print(f"Potential table detected on page {page_num + 1}")
            return True
    return False

pdf_path = "your_pdf_file.pdf"  # Replace with your PDF file path
if detect_tables_with_ocr(pdf_path):
    print("The PDF contains tables (detected via OCR).")
else:
    print("No tables found in the PDF (via OCR).")






import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io

def detect_tables_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    if len(doc) > 0:  # Ensure the document has at least one page
        page = doc[0]  # Get the first page
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))  # Convert the pixmap to an image

        text = pytesseract.image_to_string(img)
        if "table" in text.lower():
            print("Potential table detected on the first page.")
            return True
        else:
            print("No table detected on the first page.")
            return False
    else:
        print("The PDF has no pages.")
        return False

pdf_path = "your_pdf_file.pdf"  # Replace with your PDF file path
if detect_tables_with_ocr(pdf_path):
    print("The PDF contains tables (detected via OCR).")
else:
    print("No tables found in the PDF (via OCR).")


