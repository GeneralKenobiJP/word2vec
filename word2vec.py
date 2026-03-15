import json
import numpy as np
from math_utils import sigmoid

TEMP_MODEL_PATH = "model/temp/temp_model.npz"

class Word2Vec:
    """
    word2vec with Continuous Skip-gram Model
    It tries to maximize classification of a word based on another word in the same sentence.
    We use each current word as an input to a log-linear classifier with continuous
    projection layer, and predict words within a certain range before and after the current word

    https://arxiv.org/abs/1301.3781

    To avoid computing softmax over the entire vocabulary, we perform negative sampling.
    Namely, we consider the true word-context pair to be a "positive sample", and then
    choose to sample k random words that we will treat as "negative samples".
    It is a surrogate objective: instead of maximizing the exact conditional probability of the context word
    under a normalized model,
    it learns embeddings that are good at discriminating true word-context co-occurrences from noise

    We use subsampling to limit the consideration of the most frequent words, as they
    provide less information than rare words (think "the" vs. "Warsaw", the pair ("Warsaw", "Poland") is
    more semantically informative than the pair ("Warsaw", "the").

    https://proceedings.neurips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 50,
                 lr: float = 1e-3,
                 negative_samples: int = 10,
                 seed: int = 42):
        """
        word2vec constructor
        :param vocab_size: Size of the vocabulary (words which we want to embed in the vector space)
        :param embedding_dim: Dimensionality of the embedding space
        :param lr: Learning rate for the training
        :param negative_samples: Number of negative samples to consider
        :param seed: Seed of the random number generator (for reproducibility)
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.negative_samples = negative_samples
        self.distribution = self._negative_sampling_distribution()
        self.subsampling_probs = None

        self.rng = np.random.default_rng(seed)

        # Input embedding for the center (currently considered) word
        self.W_in = self.rng.uniform(-0.5/embedding_dim, 0.5/embedding_dim, (vocab_size, embedding_dim)).astype(np.float64)
        # Output embedding for the context
        self.W_out = self.rng.uniform(-0.5/embedding_dim, 0.5/embedding_dim, (vocab_size, embedding_dim)).astype(np.float64)

    def sample_negatives(self, positive_idx: int) -> np.ndarray:
        """
        Sample random words (negative samples) to be treated as a noise to distinguish the true context from.
        Omit the positive sample (actual context)
        :param positive_idx: Index of the positive sample (actual context)
        :return: An ndarray of negative samples (len = self.negative_samples)
        """
        negatives = []

        while len(negatives) < self.negative_samples:
            candidate = self.rng.choice(self.vocab_size, p=self.distribution)

            # Reject the candidate sample if it is an actual context
            if candidate != positive_idx:
                negatives.append(candidate)

        return np.array(negatives, dtype=np.int64)

    def _negative_sampling_distribution(self, counts: np.ndarray = None) -> np.ndarray:
        """
        Construct a distribution for negative sampling.
        We will sample from the unigram distribution raised to the power of 3/4,
        which is empirically found to be overperforming
        (https://proceedings.neurips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)

        :param counts: Frequencies of the words in vocabulary
        :return: An ndarray of probability distribution for negative sampling
        """
        if counts is None:
            return np.ones(self.vocab_size, dtype=np.float64) / self.negative_samples

        dist = counts ** 0.75
        return dist / dist.sum()

    def _compute_subsampling_probs(self, counts: np.ndarray, subsampling_constant: float = 1e-5) -> np.ndarray:
        """
        Compute subsampling probabilities for the corpus.
        We want most frequent words to be subsamples, as they provide limited semantic information.

        The new sampling probability becomes:
            P(w_i) = sqrt(t/frequency(w_i)),
            where t is an empirically chosen hyperparameter threshold, typically around 1e-5

        :param counts: Frequencies of the words in vocabulary
        :param subsampling_constant: the hyperparameter t
        :return: An ndarray of subsampling probabilities
        """
        freqs = counts / counts.sum()
        probs = np.sqrt(subsampling_constant / (freqs + 1e-9))

        return np.minimum(1.0, probs)

    def _subsample_corpus(self, corpus_ids: list[int] | np.ndarray) -> np.ndarray[int]:
        """
        Subsample corpus according to the subsampling distribution.
        :param corpus_ids: List of indices corresponding to the corpus words indexed by the vocabulary.
        :return: A new, subsampled corpus, with limited occurrence of the most frequent words.
        """
        corpus_ids = np.asarray(corpus_ids, dtype=np.int64)

        probs = self.subsampling_probs[corpus_ids]
        mask = self.rng.random(len(corpus_ids)) < probs
        return corpus_ids[mask]

    def step(self, center_idx: int, context_idx: int) -> float:
        """
        Performs one step of gradient descent.

        Our goal is to maximize log(sigm(u_o^T * v_c) + sum_i=1_k log(sigm(-u_n_i^T * v_c))
        Therefore, the loss function is: L = - objective

        Let us substitute s^+ := u_o^T * v_c, s_i^- := u_n_i^T * v_c
        Since:
            -log(sigm(x)) = log(1+e^(-x))
            -log(sigm(-x)) = log(1+e^x)
        We obtain:
            L = log(1+e^(-s^+)) + sum_i=1_k log(1+e^(s_i^-))

        Since:
            d/dx (-log(sigm(x))) = sigm(x) - 1
            d/dx (-log(sigm(-x))) = sigm(x)
        We obtain:
            dL/ds^+ = sigm(s^+) - 1
            dL/ds^- = sigm(-s_i^-)
        Therefore, knowing that ds/dv_c = u:
            dL/dv_i = (sigm(s^+) - 1) * u_o + sum_i=1_k sigm(s_i^-) * u_n_i
            dL/du_o = (sigm(s^+) - 1) * v_c
            dL/du_n_i = sigm(s_i^-) * v_c
        Thus, we obtain our gradient.
        The update rule is:
            v_c <- v_c - lr * dL/dv_c
            u_o <- u_o - lr * dL/du_o
            u_n_i <- u_n_i - lr * dL/du_n_i

        :param center_idx: Currently considered word (as an index in the vocabulary)
        :param context_idx: Currently considered context (as an index in the vocabulary)
        :return: Pairwise loss
        """
        v_c = self.W_in[center_idx].copy()
        u_o = self.W_out[context_idx].copy()

        neg_indices = self.sample_negatives(context_idx)
        u_neg = self.W_out[neg_indices].copy()

        # FORWARD PASS

        pos_score = np.dot(u_o, v_c)
        neg_scores = np.dot(u_neg, v_c)

        pos_prob = sigmoid(pos_score)
        neg_probs = sigmoid(neg_scores)

        # L = log(1+e^(-s^+)) + sum_i=1_k log(1+e^(s_i^-))
        # np.logaddexp(x1, x2) = log(exp(x1) + exp(x2))
        loss = float(
            np.logaddexp(0.0, -pos_score) + np.sum(np.logaddexp(0.0, neg_scores))
        )

        # BACKWARD PASS

        grad_v_c = ((pos_prob - 1.0) * u_o + np.sum(neg_probs[:, None] * u_neg, axis=0))
        grad_u_o = (pos_prob - 1.0) * v_c
        grad_u_neg = neg_probs[:, None] * v_c[None, :]

        # UPDATE

        self.W_in[center_idx] -= self.lr * grad_v_c
        self.W_out[context_idx] -= self.lr * grad_u_o
        np.add.at(self.W_out, neg_indices, -self.lr * grad_u_neg)

        return loss

    def train(self,
              corpus_ids: list[int],
              counts: np.ndarray,
              word_to_id: dict[str, int] = None,
              window_size: int = 5,
              subsampling_constant: float = 1e-5,
              epochs: int = 10) -> list[float]:
        """
        Train a word2vec model.
        :param corpus_ids: corpus text transformed into a list of indices by mapping each word into its
            index in the vocabulary (if such an index exists). The list preserves the sequence in which
            the words appear in the corpus text.
        :param counts: an array storing frequency of tokens as they appear in the vocabulary.
        :param word_to_id: dictionary mapping words to their indices in the vocabulary.
        :param window_size: Size of the context window.
        :param subsampling_constant: The hyperparameter constant for the subsampling frequency
            (the "t" in P(w_i) = sqrt(t/frequency(w_i)) )
        :param epochs: The number of epochs to train the model for.
        :return: the list of losses for each epoch
        """
        self.distribution = self._negative_sampling_distribution(counts)
        self.subsampling_probs = self._compute_subsampling_probs(counts, subsampling_constant)

        losses = []

        for epoch in range(epochs):
            # Subsample the corpus
            cur_corpus_ids = self._subsample_corpus(corpus_ids)
            cur_corpus_len = len(cur_corpus_ids)

            total_loss = 0.0
            pair_count = 0

            for t, center_idx in enumerate(cur_corpus_ids):
                context_window_left = max(0, t - window_size)
                context_window_right = min(cur_corpus_len, t + window_size + 1)

                for j in range(context_window_left, context_window_right):
                    # Context of the word should be different from the word itself
                    if j == t:
                        continue

                    context_idx = cur_corpus_ids[j]

                    pair_loss = self.step(center_idx, context_idx)

                    total_loss += pair_loss
                    pair_count += 1

            avg_loss = total_loss / max(pair_count, 1)

            losses.append(avg_loss)

            print(f"Epoch {epoch}: avg loss per pair = {avg_loss}")

            # Model snapshot
            if word_to_id:
                self.save(TEMP_MODEL_PATH, word_to_id)

        return losses

    def most_similar(self,
                     query: str | np.ndarray[float],
                     word_to_id: dict[str, int],
                     id_to_word: dict[int, str],
                     top_k: int = 5) -> list[tuple[str, float]]:
        """
        Find the most similar words to the query.
        That is, given a word, find k words that lie closest in the embedding space.
        :param query: Word or embedding to find similar words for.
        :param word_to_id: dictionary mapping words to their indices in the vocabulary.
        :param id_to_word: dictionary mapping indices in vocabulary to the words.
        :param top_k: top k similar words to return
        :return: top k most similar words to the query.
        """
        if isinstance(query, str):
            # Compute embedding of the query
            q = word_to_id[query]
            q_vec = self.W_in[q]
        else:
            q = None
            q_vec = query

        # Compute cosine similarity between the query embedding and other embeddings
        score = np.dot(self.W_in, q_vec)
        norms = np.linalg.norm(self.W_in, axis=1)
        q_norm = np.linalg.norm(q_vec)
        cos_similarity = score / np.maximum(norms * q_norm, 1e-9)
        # Sort according to the cosine similarity
        best_ids = np.argsort(-cos_similarity)

        results = []

        for idx in best_ids:
            if isinstance(query, str) and idx == q:
                continue
            results.append((id_to_word[idx], cos_similarity[idx]))
            if len(results) >= top_k:
                break

        return results

    def compute_embedding(self,
                          query: str,
                          word_to_id: dict[str, int]):
        """
        Compute the embedding for a query in the vector (embedding) space.
        :param query: Word to embed
        :param word_to_id: dictionary mapping words to their indices in the vocabulary.
        :return: Vector embedding of the query.
        """
        id = word_to_id[query]
        return self.W_in[id].copy()

    def save(self, path: str, word_to_id: dict[str, int]) -> None:
        """
        Save the model into a compressed file.
        :param path: Path to save the model into
        :param word_to_id: dictionary mapping words to their indices in the vocabulary.
        """
        vocab_json = json.dumps(word_to_id, ensure_ascii=False)
        np.savez_compressed(
            path,
            W_in=self.W_in,
            W_out=self.W_out,
            vocab_json=vocab_json,
            vocab_size=self.W_in.shape[0],
            embedding_dim=self.W_in.shape[1],
            lr=self.lr,
            negative_samples=self.negative_samples,
        )

    @staticmethod
    def load(path: str) -> tuple["Word2Vec", dict[str, int], dict[int, str]]:
        """
        Load the model from a compressed file.
        :param path: Path to the saved model.
        :return: A tuple consisting of:
            - model: a loaded Word2Vec model
            - word_to_id: dictionary mapping words to their indices in the vocabulary.
            - id_to_word: dictionary mapping indices in vocabulary to the words.
        """
        data = np.load(path, allow_pickle=False)

        W_in = data["W_in"]
        W_out = data["W_out"]
        vocab_json = str(data["vocab_json"])
        word_to_id = json.loads(vocab_json)
        word_to_id = {str(word): int(idx) for word, idx in word_to_id.items()}
        id_to_word = {idx: word for word, idx in word_to_id.items()}

        lr = float(data["lr"])
        negative_samples = int(data["negative_samples"])

        model = Word2Vec(
            vocab_size=W_in.shape[0],
            embedding_dim=W_in.shape[1],
            lr=lr,
            negative_samples=negative_samples,
        )

        model.W_in = W_in.astype(np.float64, copy=True)
        model.W_out = W_out.astype(np.float64, copy=True)

        return model, word_to_id, id_to_word
