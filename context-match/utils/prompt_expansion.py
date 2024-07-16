import os
import platform
import nltk
from nltk.corpus import wordnet

# Function to get NLTK data directory based on the operating system
def get_nltk_data_dir():
    system = platform.system()
    if system == 'Windows':
        nltk_data_dir = os.path.join(os.environ['APPDATA'], 'nltk_data')
    elif system == 'Darwin':  # macOS
        nltk_data_dir = os.path.expanduser('~/Library/Application Support/nltk_data')
    else:  # Linux and other OS
        nltk_data_dir = os.path.expanduser('~/nltk_data')
    return nltk_data_dir

# Ensure the NLTK data directory is set
nltk_data_dir = get_nltk_data_dir()
nltk.data.path.append(nltk_data_dir)

# Ensure punkt tokenizer is available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print(f"Downloading wordnet to {nltk_data_dir}")
    nltk.download('wordnet', download_dir=nltk_data_dir)

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

prompt = "happy"
expanded_prompt = get_synonyms(prompt)
print(expanded_prompt)