import os
import nltk
import json
import string
import platform
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from utils.prompt_to_openai import prompt_openai, clean_output


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


def get_synonyms(word, max_synonyms=3):
    """
    Function to get synonyms for a given word using NLTK

    :param word: The word to find synonyms for (str).
    :param max_synonyms: The number of synonyms to return (int).
    :return: The set of synonyms (set).
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if len(synonyms) >= max_synonyms:
                break
            synonyms.add(lemma.name())

    return synonyms


def handle_prompt(
    prompt,
    api_key: str,
    expand_with_syn: bool = False,
    expand_with_openAI: bool = True,
    print_prompt: bool = False,
):
    """
    Function to handle the prompt enrichment process

    :param prompt: The prompt to enrich (str).
    :param api_key: The OpenAI API key (str).
    :param expand_with_syn: Expand the prompt with synonyms (bool).
    :param expand_with_openAI: Expand the prompt with OpenAI (bool).
    :param print_prompt: Print the prompt (bool).
    :return: The enriched prompt (str).
    """
    if expand_with_openAI:
        prompt = get_enriched_prompt(prompt, api_key=api_key)

    if expand_with_syn:
        prompt = expand_prompt_with_synonyms(prompt)

    if print_prompt:
        print(f"The input prompt is: \n{prompt}")

    return prompt


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

    keywords_str = "\n- These are keywords extracted from the prompt: " + ", ".join(
        keywords
    )

    return prompt + ", " + keywords_str


def get_enriched_prompt(original_prompt: str, api_key: str, max_tokens: int = 250):
    """
    Get the enriched prompt from OpenAI's API.

    :param prompt: The prompt to enrich (str).
    :param api_key: The OpenAI API key (str).
    :return: The enriched prompt (str).
    """

    prompt = (
        original_prompt
        + """\n\nBased on the prompt above, return 5 columns that should be present in the database and 5 keywords to help search for the relevant tables. 
        Return these 10 items into one single Python list and nothing else."""
    )

    response = prompt_openai(prompt=prompt, api_key=api_key, max_tokens=max_tokens)
    response = clean_output(response)

    # Check for empty response
    if not response:
        raise ValueError("Received empty response from API")

    # Try to decode the JSON
    try:
        response_json = json.loads(response)

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        raise

    # Ensure response_json is a list or dict
    if not isinstance(response_json, (list, dict)):
        raise ValueError("Decoded JSON is not a list or dict")

    # Flatten and convert to strings if necessary
    if isinstance(response_json, list):
        flattened_response = []
        for item in response_json:
            if isinstance(item, list):
                flattened_response.extend(map(str, item))
            else:
                flattened_response.append(str(item))
        response = ", ".join(flattened_response)
    else:
        response = ", ".join(str(value) for value in response_json.values())

    enriched_prompt = (
        original_prompt
        + ".\n- The table should contain column names similar to: "
        + response
        + "\n"
    )

    return enriched_prompt
