import numpy as np
from sklearn.decomposition import PCA

from preprocessing import tokenize, build_vocab, read_txt_file
from word2vec import Word2Vec

import matplotlib.pyplot as plt

def analogy(
    a: str,
    b: str,
    c: str,
    W: np.ndarray,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Solve analogy of the form: a - b + c ~= d

    Returns the top_k nearest words to v(a) - v(b) + v(c),
    excluding a, b, and c themselves.

    Parameters
    ----------
    a, b, c : str
        Words in the analogy expression.
    W : np.ndarray
        Embedding matrix of shape (vocab_size, embedding_dim).
    word_to_id : dict[str, int]
        Word -> integer id mapping.
    id_to_word : dict[int, str]
        Integer id -> word mapping.
    top_k : int
        Number of candidates to return.

    Returns
    -------
    list[tuple[str, float]]
        List of (word, cosine_similarity).
    """
    # Check vocabulary membership explicitly.
    for word in (a, b, c):
        if word not in word_to_id:
            raise KeyError(f"Word '{word}' is not in the vocabulary.")

    # Retrieve embeddings.
    v_a = W[word_to_id[a]]
    v_b = W[word_to_id[b]]
    v_c = W[word_to_id[c]]

    # Compute target vector in the original embedding space.
    target = v_a - v_b + v_c

    # Compute cosine similarity between every vocabulary embedding and target.
    scores = W @ target
    W_norms = np.linalg.norm(W, axis=1)
    target_norm = np.linalg.norm(target)

    sims = scores / np.maximum(W_norms * target_norm, 1e-12)

    # Sort from largest cosine similarity to smallest.
    ranked_ids = np.argsort(-sims)

    # Exclude the input words.
    banned = {word_to_id[a], word_to_id[b], word_to_id[c]}

    results = []
    for idx in ranked_ids:
        if idx in banned:
            continue
        results.append((id_to_word[idx], float(sims[idx])))
        if len(results) == top_k:
            break

    return results

def simple_experiment():
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
        counts=counts,
        window_size=3,
        epochs=10,
    )

    model.save("model/test_model_crime_and_punishment.npz", word_to_id)

    for word in ["you", "lord", "tragedy", "wise"]:
        print(f"\nNearest neighbors of '{word}':")
        for neighbor, score in model.most_similar(word, word_to_id, id_to_word, top_k=5):
            print(f"  {neighbor:>10s}  cosine={score:.4f}")

    for word in vocab:
        print(f"\nWord '{word}': {model.compute_embedding(word, word_to_id)}")

    print("\n\n ### \n\n")

    model, word_to_id, id_to_word = Word2Vec.load("model/test_model.npz")

    for word in ["you", "lord", "tragedy", "wise"]:
        print(f"\nNearest neighbors of '{word}':")
        for neighbor, score in model.most_similar(word, word_to_id, id_to_word, top_k=5):
            print(f"  {neighbor:>10}  cosine={score:.4f}")

    for word in vocab:
        print(f"\nWord '{word}': {model.compute_embedding(word, word_to_id)}")

def proper_experiment(dataset: str, model_name: str,
                      query_a: str, query_b: str, query_c: str, query_d: str,
                      retrain: bool = True):
    text = read_txt_file(dataset)

    tokens = tokenize(text)
    vocab, word_to_id, id_to_word, corpus_ids, counts = build_vocab(tokens, min_count=50)

    print(vocab)

    print(f"Size of the vocab: {len(vocab)}")
    print(f"Size of the corpus: {len(corpus_ids)}")

    print(f'counts - {query_a}: {counts[word_to_id[query_a]]}')
    print(f'counts - {query_b}: {counts[word_to_id[query_b]]}')
    print(f'counts - {query_c}: {counts[word_to_id[query_c]]}')
    print(f'counts - {query_d}: {counts[word_to_id[query_d]]}')

    print("\n\n ### \n\n")
    print("Vocab...")
    print(vocab)
    print("\n\n ### \n\n")

    if retrain:

        model = Word2Vec(
            vocab_size=len(word_to_id),
            embedding_dim=100,
            lr=0.01,
            negative_samples=10,
            seed=42
        )

        model.train(
            corpus_ids=corpus_ids,
            word_to_id=word_to_id,
            counts=counts,
            window_size=3,
            epochs=10,
        )

        model.save(model_name, word_to_id)

    else:
        model, word_to_id, _ = Word2Vec.load(model_name)

    emb_a = model.compute_embedding(query_a, word_to_id)
    emb_b = model.compute_embedding(query_b, word_to_id)
    emb_c = model.compute_embedding(query_c, word_to_id)
    emb_d = model.compute_embedding(query_d, word_to_id)

    diff = emb_b - emb_a
    estimated_emb_d = emb_c + diff

    print(f'{query_a}_emb: {emb_a}')
    print(f'{query_b}_emb: {emb_b}')
    print(f'{query_c}_emb: {emb_c}')
    print(f'{query_d}_emb: {emb_d}')
    print(f'diff: {diff}')
    print(f'estimated_{query_d}_emb: {estimated_emb_d}')

    embeddings = np.array([emb_a, emb_b, emb_c, emb_d, estimated_emb_d])
    pca = PCA(n_components=2)
    embeddings = pca.fit_transform(embeddings)

    print(embeddings)

    labels = [f'{query_a}_emb', f'{query_b}_emb', f'{query_c}_emb', f'{query_d}_emb', f'estimated_{query_d}_emb']
    plt.scatter(embeddings[:, 0], embeddings[:, 1])
    for i, txt in enumerate(labels):
        plt.annotate(txt, (embeddings[i, 0], embeddings[i, 1]))
    plt.show()

    results = analogy(query_a, query_b, query_c, model.W_in, word_to_id, id_to_word, 20)
    print(results)

if __name__ == '__main__':
    print('Hello World')

    # proper_experiment("dataset/wiki.train.tokens", "model/wiki.npz",
    #                   "british", "french", "london", "paris",
    #                   retrain=False)
    proper_experiment("dataset/wiki.train.tokens", "model/wiki.npz",
                      "quick", "slow", "quickly", "slowly",
                      retrain=False)

    print('Goodbye World!')