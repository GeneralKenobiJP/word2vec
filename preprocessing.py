import re
from typing import Counter

import numpy as np

def read_txt_file(path: str) -> str:
    """
    Read text file into a string
    :param path: Path to the text file
    :return: A string with data
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def tokenize(text: str) -> list[str]:
    """
    Tokenize text into lowercase words consisting of letters and digits
    :param text: Text to tokenize
    :return: List of tokens
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", " ", text)

    # Cross out empty strings
    return [token for token in text.split() if token]

def build_vocab(tokens: list[str], min_count: int) \
        -> tuple[list[str], dict[str, int], dict[int, str], list[int], np.ndarray]:
    """
    Build a vocabulary of the tokens, and a map the corpus accordingly
    so that word2vec training can be performed.
    :param tokens: List of tokens
    :param min_count: Minimum number of occurrences for the tokens to be considered.
            The tokens below the threshold will be discarded.
    :return: A tuple consisting of:
        - Vocabulary: a list of tokens above the frequency threshold,
            sorted by decreasing frequency and alphabetical order.
        - word_to_id: dictionary mapping words to their indices in the vocabulary.
        - id_to_word: dictionary mapping indices in vocabulary to the words.
        - corpus_ids: corpus text transformed into a list of indices by mapping each word into its
            index in the vocabulary (if such an index exists). The list preserves the sequence in which
            the words appear in the corpus text.
        - counts: an array storing frequency of tokens as they appear in the vocabulary.
    """
    # Count token frequencies
    counter = Counter(tokens)
    # Keep only tokens that appear at least min_count times
    vocab = [word for word, count in counter.items() if count >= min_count]

    # Sort by decreasing frequency, then alphabetically for deterministic behaviour
    vocab.sort(key=lambda w: (-counter[w], w))

    # Assign integer ids
    word_to_id = {word: idx for idx, word in enumerate(vocab)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    corpus_ids = [word_to_id[word] for word in tokens if word in word_to_id]

    counts = np.array([counter[word] for word in vocab], dtype=np.float64)

    return vocab, word_to_id, id_to_word, corpus_ids, counts
