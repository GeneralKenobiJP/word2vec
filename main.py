import numpy as np

from preprocessing import tokenize, build_vocab
from word2vec import Word2Vec

if __name__ == '__main__':
    print('Hello World')

    text = "Did you ever hear the tragedy of Darth Plagueis the Wise? I thought not. It's not a story the Jedi would tell you. It's a Sith legend. Darth Plagueis was a Dark Lord of the Sith, so powerful and so wise he could use the Force to influence the midichlorians to create life... He had such a knowledge of the dark side that he could even keep the ones he cared about from dying. The dark side of the Force is a pathway to many abilities some consider to be unnatural. He became so powerful... the only thing he was afraid of was losing his power, which eventually, of course, he did. Unfortunately, he taught his apprentice everything he knew, then his apprentice killed him in his sleep. It's ironic he could save others from death, but not himself."

    tokens = tokenize(text)

    vocab, word_to_id, id_to_word, corpus_ids, counts = build_vocab(tokens, min_count=1)

    model = Word2Vec(
        vocab_size=len(word_to_id),
        embedding_dim=20,
        lr=0.05,
        negative_samples=4,
        seed=42
    )

    model.train(
        corpus_ids=corpus_ids,
        window_size=3,
        epochs=10,
    )

    for word in ["you", "lord", "tragedy", "wise"]:
        print(f"\nNearest neighbors of '{word}':")
        for neighbor, score in model.most_similar(word, word_to_id, id_to_word, top_k=5):
            print(f"  {neighbor:>10s}  cosine={score:.4f}")

    for word in vocab:
        print(f"\nWord '{word}': {model.compute_embedding(word, word_to_id)}")

    print('Goodbye World!')