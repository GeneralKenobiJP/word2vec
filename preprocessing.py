import re
from typing import Counter

import numpy as np

def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Cross out empty strings
    return [token for token in text.split() if token]

def build_vocab(tokens: list[str], min_count: int) -> tuple[list[str], dict, dict, list, np.ndarray]:
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
