import os
import json
import nltk
from nltk.corpus import words
from typing import List, Optional

import nltk
from nltk.corpus import brown
from collections import Counter

def generate_frequent_word_list(
    output_path: str = 'frequent_word_list.json',
    limit: int = 20000,
    min_length: int = 3,
    max_length: int = 15,
    exclude_proper_nouns: bool = True,
    exclude_punctuation: bool = True,
    lowercase: bool = True,
    verbose: bool = True
) -> List[str]:
    """
    Generates a word list based on word frequency using NLTK's Brown corpus.

    Parameters:
        All parameters are similar to generate_word_list().

    Returns:
        List[str]: A list of filtered frequent words.
    """
    # Check if the word list already exists
    if os.path.isfile(output_path):
        if verbose:
            print(f"Loading existing word list from {output_path}...")
        with open(output_path, 'r') as f:
            word_list = json.load(f)
        if verbose:
            print(f"Loaded {len(word_list)} words.")
        return word_list

    # Download required NLTK corpora if not already present
    try:
        nltk.data.find('corpora/brown')
    except LookupError:
        if verbose:
            print("Downloading NLTK 'brown' corpus for frequency analysis...")
        nltk.download('brown')

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        if verbose:
            print("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')

    # Retrieve words from the Brown corpus
    brown_words = brown.words()
    if verbose:
        print(f"Total words in Brown corpus: {len(brown_words)}")

    # Compute word frequencies
    word_freq = Counter([word.lower() for word in brown_words])

    # Sort words by frequency
    sorted_words = [word for word, freq in word_freq.most_common()]

    # Initialize an empty set to avoid duplicates
    filtered_words = set()

    for word in sorted_words:
        # Apply length filters
        if len(word) < min_length or len(word) > max_length:
            continue

        # Exclude proper nouns by checking capitalization
        if exclude_proper_nouns and word[0].isupper():
            continue

        # Exclude words with punctuation
        if exclude_punctuation and not word.isalpha():
            continue

        # Convert to lowercase if specified
        if lowercase:
            word = word.lower()

        filtered_words.add(word)

        # Stop if the limit is reached
        if len(filtered_words) >= limit:
            break

    # Convert the set to a sorted list
    word_list = sorted(list(filtered_words))
    if verbose:
        print(f"Filtered frequent words count: {len(word_list)}")

    # Save the word list to the specified JSON file
    with open(output_path, 'w') as f:
        json.dump(word_list, f)
    if verbose:
        print(f"Word list saved to {output_path}")

    return word_list
