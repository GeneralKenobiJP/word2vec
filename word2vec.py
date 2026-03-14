import json

import numpy as np
from math_utils import sigmoid

class Word2Vec:
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 50,
                 lr: float = 1e-3,
                 negative_samples: int = 10,
                 seed: int = 42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.negative_samples = negative_samples

        self.rng = np.random.default_rng(seed)

        self.W_in = self.rng.uniform(-0.5/embedding_dim, 0.5/embedding_dim, (vocab_size, embedding_dim)).astype(np.float64)
        self.W_out = self.rng.uniform(-0.5/embedding_dim, 0.5/embedding_dim, (vocab_size, embedding_dim)).astype(np.float64)

    def sample_negatives(self, positive_idx: int) -> np.ndarray:
        negatives = []

        while len(negatives) < self.negative_samples:
            candidate = self.rng.choice(self.vocab_size)

            if candidate != positive_idx:
                negatives.append(candidate)

        return np.array(negatives, dtype=np.int64)

    def step(self, center_idx: int, context_idx: int) -> float:
        v_c = self.W_in[center_idx].copy()
        u_o = self.W_out[context_idx].copy()

        neg_indices = self.sample_negatives(context_idx)
        u_neg = self.W_out[neg_indices].copy()

        # FORWARD PASS

        pos_score = np.dot(u_o, v_c)
        neg_scores = np.dot(u_neg, v_c)

        pos_prob = sigmoid(pos_score)
        neg_probs = sigmoid(neg_scores)

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
              window_size: int = 5,
              epochs: int = 10):
        corpus_len = len(corpus_ids)

        losses = []

        for epoch in range(epochs):

            total_loss = 0.0
            pair_count = 0

            for t, center_idx in enumerate(corpus_ids):
                context_window_left = max(0, t - window_size)
                context_window_right = min(corpus_len, t + window_size - 1)

                for j in range(context_window_left, context_window_right):
                    if j == t:
                        continue

                    context_idx = corpus_ids[j]

                    pair_loss = self.step(center_idx, context_idx)

                    total_loss += pair_loss
                    pair_count += 1

            avg_loss = total_loss / max(pair_count, 1)

            losses.append(avg_loss)

            print(f"Epoch {epoch}: avg loss per pair = {avg_loss}")

        return losses

    def most_similar(self,
                     query: str,
                     word_to_id: dict[str, int],
                     id_to_word: dict[int, str],
                     top_k: int = 5) -> list[tuple[str, float]]:
        q = word_to_id[query]
        q_vec = self.W_in[q]

        score = np.dot(self.W_in, q_vec)

        norms = np.linalg.norm(self.W_in, axis=1)
        q_norm = np.linalg.norm(q_vec)

        cos_similarity = score / np.maximum(norms * q_norm, 1e-9)
        best_ids = np.argsort(-cos_similarity)

        results = []

        for idx in best_ids:
            if idx == q:
                continue
            results.append((id_to_word[idx], cos_similarity[idx]))
            if len(results) >= top_k:
                break

        return results

    def compute_embedding(self,
                          query: str,
                          word_to_id: dict[str, int]):
        id = word_to_id[query]
        return self.W_in[id]

    def save(self, path: str, word_to_id: dict[str, int]) -> None:
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
