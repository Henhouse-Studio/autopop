import os
import nltk
import string
import platform
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from utils.prompt_to_openai import *


# Function to get NLTK data directory based on the operating system
def get_nltk_data_dir():
    system = platform.system()
    if system == "Windows":
        nltk_data_dir = os.path.join(os.environ["APPDATA"], "nltk_data")
    elif system == "Darwin":  # macOS
        nltk_data_dir = os.path.expanduser("~/Library/Application Support/nltk_data")
    else:  # Linux and other OS
        nltk_data_dir = os.path.expanduser("~/nltk_data")
    return nltk_data_dir


# Ensure the NLTK data directory is set
nltk_data_dir = get_nltk_data_dir()
nltk.data.path.append(nltk_data_dir)

# Ensure necessary NLTK data packages are available
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    # print(f"Downloading wordnet to {nltk_data_dir}")
    nltk.download("wordnet", download_dir=nltk_data_dir, quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    # print(f"Downloading stopwords to {nltk_data_dir}")
    nltk.download("stopwords", download_dir=nltk_data_dir, quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    # print(f"Downloading punkt tokenizer to {nltk_data_dir}")
    nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)


# Function to get synonyms for a given word
def get_synonyms(word, max_synonyms=3):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if len(synonyms) >= max_synonyms:
                break
            synonyms.add(lemma.name())
    return synonyms


# Function to expand prompt with synonyms and append as keywords
def expand_prompt_with_synonyms(prompt, max_synonyms_per_word=2):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(prompt)
    words = [
        word
        for word in words
        if word.lower() not in stop_words and word not in string.punctuation
    ]

    added_words = set(words)
    keywords = set()

    for word in words:
        synonyms = get_synonyms(word, max_synonyms=max_synonyms_per_word)
        for synonym in synonyms:
            if synonym not in added_words:
                keywords.add(synonym)
                added_words.add(synonym)

    keywords_str = "\nkeywords: " + ", ".join(keywords)
    return prompt + ", " + keywords_str

def get_enriched_prompt(original_prompt: str, api_key: str):
    """
    Get the enriched prompt from OpenAI's API.

    :param prompt: The prompt to enrich (str).
    :param api_key: The OpenAI API key (str).
    :return: The enriched prompt (str).
    """
    prompt = original_prompt + "\n\nBased on the prompt above, what columns should be present in the database? Return it as a Python list and nothing else."

    response = prompt_openai(prompt=prompt, api_key=api_key, max_tokens=50)
    response = sub("```python", "", response)
    response = sub("```", "", response)

    # make response into a string
    # response = json.loads(response)
    # response = ', '.join(response)

    enriched_prompt = original_prompt + ". The table should contain column names similar to: " + response

    return enriched_prompt