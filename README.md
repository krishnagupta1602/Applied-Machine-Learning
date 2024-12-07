List:
Ordered: Elements maintain insertion order.
Mutable: Can modify (add, remove, change elements).
Duplicates: Allows duplicate elements.
Search Order: O(n) for searching.
Memory: More memory overhead due to dynamic resizing.
When to use: When you need ordered, mutable, and potentially duplicated items.

Tuple:
Ordered: Elements maintain insertion order.
Immutable: Cannot modify once created.
Duplicates: Allows duplicate elements.
Search Order: O(n) for searching.
Memory: More memory efficient than lists due to immutability.
When to use: When you need an ordered, immutable collection, typically for fixed data.

Set:
Unordered: No guarantee of element order.
Mutable: Can add/remove elements, but cannot modify elements directly.
No Duplicates: Automatically removes duplicates.
Search Order: O(1) average for membership testing (using hashing).
Memory: Efficient for uniqueness and fast lookups.
When to use: When you need unique items and fast membership testing.

Dictionary:
Ordered (from Python 3.7): Maintains insertion order.
Mutable: Can modify, add, or remove key-value pairs.
No Duplicates in Keys: Keys must be unique.
Search Order: O(1) average for key lookup.
Memory: More memory overhead for key-value pairs compared to sets.
When to use: When you need key-value pairs with fast lookups by key.

Summary of Selection:
Memory Efficiency: Tuple > Set > List > Dictionary.
Speed: Set, Dictionary (O(1) lookups) > List, Tuple (O(n) lookups).
Use Case:
List: Ordered, mutable sequence with duplicates.
Tuple: Ordered, immutable sequence.
Set: Unordered, unique items, fast membership test.
Dictionary: Key-value pairs with fast key lookup.






Features of Lists in Python
Ordered and Mutable: Lists maintain the order of elements and allow modification (add, remove, or change elements).
Heterogeneous Elements: Lists can store a mix of data types (e.g., integers, strings, and even other lists).
Indexed Access: Elements can be accessed via zero-based indexing and slicing.
Dynamic Size: Lists can grow or shrink as needed; there is no fixed size.
Wide Range of Methods: Python lists support a variety of built-in methods for common operations like sorting, appending, and searching.

# 1. Creating and Appending to a List
my_list = [1, 2, 3]
my_list.append(4)  # Adds 4 to the end of the list
print("After append:", my_list)  # Output: [1, 2, 3, 4]

# 2. Extending a List
my_list.extend([5, 6])  # Adds multiple elements
print("After extend:", my_list)  # Output: [1, 2, 3, 4, 5, 6]

# 3. Inserting Elements
my_list.insert(2, 'a')  # Inserts 'a' at index 2
print("After insert:", my_list)  # Output: [1, 2, 'a', 3, 4, 5, 6]

# 4. Removing Elements
my_list.remove('a')  # Removes the first occurrence of 'a'
print("After remove:", my_list)  # Output: [1, 2, 3, 4, 5, 6]

# 5. Popping Elements
last_element = my_list.pop()  # Removes and returns the last element
print("Popped element:", last_element)  # Output: 6
print("After pop:", my_list)  # Output: [1, 2, 3, 4, 5]

# 6. Reversing a List
my_list.reverse()  # Reverses the list in place
print("After reverse:", my_list)  # Output: [5, 4, 3, 2, 1]

# 7. Sorting a List
unsorted_list = [3, 1, 4, 2]
unsorted_list.sort()  # Sorts the list in ascending order
print("After sort:", unsorted_list)  # Output: [1, 2, 3, 4]

# 8. Finding Index of an Element
index = my_list.index(4)  # Finds the index of the first occurrence of 4
print("Index of 4:", index)  # Output: 1

# 9. Counting Occurrences
my_list = [1, 2, 2, 3]
count = my_list.count(2)  # Counts how many times 2 appears
print("Count of 2:", count)  # Output: 2

# 10. Copying a List
copy_list = my_list.copy()  # Creates a shallow copy of the list
print("Copied list:", copy_list)  # Output: [1, 2, 2, 3]

# 11. Clearing a List
my_list.clear()  # Removes all elements from the list
print("After clear:", my_list)  # Output: []


Features of Tuples in Python
Immutable: Tuples cannot be modified after creation, ensuring data integrity.
Ordered: Tuples maintain the order of elements, similar to lists.
Heterogeneous Elements: Tuples can store a mix of data types.
Faster than Lists: Accessing and iterating through tuples is faster due to their immutability.
Supports Indexing and Slicing: Elements can be accessed by index or sliced like lists.


# 1. Creating a Tuple
my_tuple = (1, 2, 3, 4)
print("Tuple:", my_tuple)  # Output: (1, 2, 3, 4)

# 2. Accessing Elements by Index
print("Element at index 2:", my_tuple[2])  # Output: 3

# 3. Slicing a Tuple
print("Sliced tuple (1:3):", my_tuple[1:3])  # Output: (2, 3)

# 4. Finding Length of a Tuple
print("Length of tuple:", len(my_tuple))  # Output: 4

# 5. Checking Membership
print("Is 3 in tuple?", 3 in my_tuple)  # Output: True

# 6. Concatenating Tuples
new_tuple = my_tuple + (5, 6)
print("After concatenation:", new_tuple)  # Output: (1, 2, 3, 4, 5, 6)

# 7. Repeating Tuples
repeated_tuple = my_tuple * 2
print("After repetition:", repeated_tuple)  # Output: (1, 2, 3, 4, 1, 2, 3, 4)

# 8. Finding Index of an Element
index = my_tuple.index(3)
print("Index of 3:", index)  # Output: 2

# 9. Counting Occurrences of an Element
count = my_tuple.count(2)
print("Count of 2:", count)  # Output: 1

# 10. Nesting Tuples
nested_tuple = (my_tuple, (5, 6))
print("Nested tuple:", nested_tuple)  # Output: ((1, 2, 3, 4), (5, 6))

# 11. Unpacking a Tuple
a, b, c, d = my_tuple
print("Unpacked values:", a, b, c, d)  # Output: 1 2 3 4


Features of Sets in Python
Unordered and Unindexed: Sets do not maintain order, and elements cannot be accessed using an index.
Unique Elements: Sets automatically remove duplicates, ensuring all elements are unique.
Mutable: You can add or remove elements, but the set itself cannot contain mutable elements like lists.
Optimized for Membership Testing: Sets provide fast checks for whether an element exists.
Supports Mathematical Set Operations: Union, intersection, difference, and symmetric difference are supported.


# 1. Creating a Set
my_set = {1, 2, 3, 4}
print("Set:", my_set)  # Output: {1, 2, 3, 4}

# 2. Adding Elements
my_set.add(5)  # Adds 5 to the set
print("After add:", my_set)  # Output: {1, 2, 3, 4, 5}

# 3. Removing Elements
my_set.remove(3)  # Removes 3 from the set (raises error if not found)
print("After remove:", my_set)  # Output: {1, 2, 4, 5}

# 4. Checking Membership
print("Is 2 in set?", 2 in my_set)  # Output: True

# 5. Union of Two Sets
another_set = {3, 4, 5, 6}
union_set = my_set.union(another_set)
print("Union:", union_set)  # Output: {1, 2, 3, 4, 5, 6}

# 6. Intersection of Two Sets
intersection_set = my_set.intersection(another_set)
print("Intersection:", intersection_set)  # Output: {4, 5}

# 7. Difference of Two Sets
difference_set = my_set.difference(another_set)
print("Difference:", difference_set)  # Output: {1, 2}

# 8. Clearing a Set
my_set.clear()  # Removes all elements
print("After clear:", my_set)  # Output: set()


Features of Dictionaries in Python
Key-Value Pairs: Dictionaries store data as key-value pairs, where each key is unique.
Unordered: Dictionaries do not maintain the order of elements (Python 3.7+ maintains insertion order).
Mutable: You can add, update, or remove key-value pairs.
Fast Lookup: Dictionaries provide efficient key-based access to values.
Heterogeneous Keys and Values: Both keys and values can be of any data type.


# 1. Creating a Dictionary
my_dict = {'a': 1, 'b': 2, 'c': 3}
print("Dictionary:", my_dict)  # Output: {'a': 1, 'b': 2, 'c': 3}

# 2. Accessing Values by Key
print("Value for 'b':", my_dict['b'])  # Output: 2

# 3. Adding or Updating Key-Value Pairs
my_dict['d'] = 4  # Adds 'd': 4
my_dict['a'] = 10  # Updates value of 'a'
print("After add/update:", my_dict)  # Output: {'a': 10, 'b': 2, 'c': 3, 'd': 4}

# 4. Removing a Key-Value Pair
del my_dict['b']  # Removes the key 'b'
print("After delete:", my_dict)  # Output: {'a': 10, 'c': 3, 'd': 4}

# 5. Checking Key Existence
print("Is 'c' a key?", 'c' in my_dict)  # Output: True

# 6. Getting Keys, Values, and Items
keys = my_dict.keys()  # Returns dictionary keys
values = my_dict.values()  # Returns dictionary values
items = my_dict.items()  # Returns key-value pairs
print("Keys:", keys)  # Output: dict_keys(['a', 'c', 'd'])
print("Values:", values)  # Output: dict_values([10, 3, 4])
print("Items:", items)  # Output: dict_items([('a', 10), ('c', 3), ('d', 4)])

# 7. Popping an Item
popped_item = my_dict.pop('d')  # Removes and returns the item for 'd'
print("Popped item:", popped_item)  # Output: 4
print("After pop:", my_dict)  # Output: {'a': 10, 'c': 3}

# 8. Clearing a Dictionary
my_dict.clear()  # Removes all key-value pairs
print("After clear:", my_dict)  # Output: {}




Features of Strings in Python
Immutable: Strings cannot be changed after creation. Any modification results in a new string.
Ordered: Strings maintain the order of characters, and indexing is supported.
Heterogeneous: Strings can contain letters, digits, punctuation, and special characters.
Supports Indexing and Slicing: Characters can be accessed by their index or sliced into substrings.
Built-in Methods: Strings have many methods for manipulation, searching, and formatting.



# 1. Creating a String
my_str = "Hello, World!"
print("String:", my_str)  # Output: Hello, World!

# 2. Accessing Characters by Index
print("Character at index 0:", my_str[0])  # Output: H

# 3. Slicing a String
print("Sliced string (0:5):", my_str[0:5])  # Output: Hello

# 4. Changing Case
print("Uppercase:", my_str.upper())  # Output: HELLO, WORLD!
print("Lowercase:", my_str.lower())  # Output: hello, world!

# 5. Finding a Substring
print("Index of 'World':", my_str.find('World'))  # Output: 7

# 6. Replacing Substring
new_str = my_str.replace('World', 'Python')
print("After replace:", new_str)  # Output: Hello, Python!

# 7. Checking if a Substring Exists
print("Contains 'Hello'?", 'Hello' in my_str)  # Output: True

# 8. Counting Occurrences of a Substring
print("Count of 'o':", my_str.count('o'))  # Output: 2

# 9. Splitting a String
split_str = my_str.split(',')  # Splits into a list at the comma
print("After split:", split_str)  # Output: ['Hello', ' World!']

# 10. Stripping Whitespace
stripped_str = "  Hello!  ".strip()  # Removes leading and trailing whitespace
print("After strip:", stripped_str)  # Output: Hello!


List vs Tuple
Immutability:

Tuple is immutable, meaning it can't be modified after creation, which makes it more suitable for read-only data or when you need to ensure data integrity.
List is mutable, so it can be changed (added to, removed from, or modified), which makes it more flexible for dynamic data.
Memory Efficiency:

Tuple consumes less memory than a list, making it more space-efficient when dealing with large data that doesn’t require modification.
List takes up more memory due to its mutability.
Performance:

Tuple is generally faster for iteration and accessing elements due to its immutability.
List has slightly higher overhead because of the need to support modifications.
Tuple vs Set
Uniqueness of Elements:
Set automatically removes duplicates and ensures all elements are unique, whereas Tuple can contain duplicate elements.
Search and Lookup:
Set provides faster membership testing (O(1) average time complexity) compared to Tuple, which has O(n) for membership testing.
Order:
Tuple maintains the order of elements, while Set is unordered. This makes Tuple suitable for ordered data and Set for ensuring uniqueness without caring about the order.
Set vs Dictionary
Key-Value Association:

Dictionary stores key-value pairs, making it ideal when you need to map one element to another (e.g., for lookups, counting, etc.).
Set only stores unique values without any key-value association, making it simpler when only uniqueness and membership testing are needed.
Memory and Speed:

Dictionary generally requires more memory due to storing both keys and values, while Set stores only the elements.
Both Set and Dictionary offer O(1) average time complexity for search operations, but Dictionary adds overhead for handling both keys and values.
List vs Set
Uniqueness:

Set automatically removes duplicates, whereas List allows duplicates, making Set the go-to choice when you want only unique elements.
Order:

List maintains the order of elements, making it suitable for situations where the order matters.
Set is unordered, meaning there is no guarantee about the order of elements.
Search Efficiency:

Set offers faster membership testing (O(1) average time complexity) compared to List, which is O(n) for membership checks.
List vs Dictionary
Key-Value Pairing:

Dictionary is preferred when you need to associate a key with a value (e.g., a phone book or database).
List is a simple sequence of elements and is not suited for key-value pairs.
Performance for Search:

Dictionary offers O(1) average time complexity for searching by keys, while searching through a List requires O(n) time.
Flexibility:

List is more flexible in terms of order and can be indexed using integers, while Dictionary provides key-based indexing, which may be more useful for mapping real-world data.
Summary of When to Use Each:
Use a List when:
You need ordered, mutable collections.
You may need to modify the collection (add, remove, or update elements).
You need indexed access to elements.
Use a Tuple when:
You need immutable data (i.e., the data shouldn't change).
You want a more memory-efficient collection than a list.
You need to store a fixed sequence of elements.
Use a Set when:
You need to store a collection of unique elements.
You care only about membership (e.g., checking if an element is in the set).
You don't care about the order of elements.
Use a Dictionary when:
You need to store data as key-value pairs.
You need fast lookups by key.
You need to associate values with specific keys for easy access and modification.
Choosing the appropriate data structure depends on the specific requirements of your program, such as performance (speed of search), memory constraints, and the type of operations you need to perform on the data.

















Apple - Plape
Google - Gogeol
Amazon - Zanoma
Microsoft - Crosotfim
Facebook - Koofceab
Tesla - Stale
Samsung - Usmangs
Netflix - Feltnix
Adobe - Dobea
Intel - Letin
IBM - BMI
Coca-Cola - Ocac-Calo
Nike - Kien
Pepsi - Speip
Spotify - Tiysofp
Disney - Ysdein
Uber - Rube
Oracle - Lacreor
Sony - Ynos
Twitter - Ttwiter


python.CoInitialize():

This initializes the COM library for use in the current thread.
It's required for interacting with COM objects like Outlook when you're using libraries like pywin32.
Without this, you might encounter issues when trying to dispatch and use COM objects.
Outlook=win32.client.Dispatch("Outlook.Application").GetNameSpace("MAPI"):

This connects to the Outlook application via the Dispatch method from pywin32, creating a COM object that allows Python to interact with Outlook.
GetNameSpace("MAPI") accesses the Messaging Application Programming Interface (MAPI), which is used to manage Outlook data (emails, calendars, contacts, etc.).
The Outlook object now represents a session with Outlook and provides access to its features.
folders = outlook.Folders:

This retrieves the collection of top-level folders in the Outlook application.
From here, you can iterate through or access specific folders (like "Inbox" or "Sent Items") to extract emails or perf

Santa - Natas
Snowflake - Fakeslown
Reindeer - Deerrein
Mistletoe - Eetostilm
Ornament - Monanert
Carolers - Losrecar
Stocking - Gitskonc
Wreath - Thwear
Sleigh - Hgiles
Elf - Fle
Office Words
Manager - Ragamen
Colleague - Gelueocal
Printer - Nertipr
Meeting - Gteniem
Schedule - Eldusche
Desk - Sdek
Laptop - Ptalpo
Paperclip - Reppacpli
Chair - Ahirc
Email - Iamle
Keyboard - Bradykoe
Mouse - Esuom
Document - Cemdonut
Stationery - Aitnoryets
Whiteboard - Oathibedwr
Presentation - Sentationpre
Planner - Lerpnna
Files - Sfiel
Task - Ksat
Coffee - Efofec
Christmas Words (Easy)
Tree - Eetr
Star - Tsar
Gift - Fgit
Bell - Lelb
Snow - Nows
Elf - Fle
Holly - Loyhl
Candy - Dycan
Santa - Natas
Frost - Sroft


orm other operations.


r'\b(?:security|privacy|fraud practice)\b'




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
