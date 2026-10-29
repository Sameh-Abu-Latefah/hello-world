import numpy as np


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    norm_a = np.sqrt(np.dot(a, a))
    norm_b = np.sqrt(np.dot(b, b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")

    return float(np.dot(a, b) / (norm_a * norm_b))