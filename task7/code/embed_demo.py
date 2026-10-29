import numpy as np
from sentence_transformers import SentenceTransformer
from similarity import cosine_similarity

SENTENCES = [
    "The cat sat on the sofa.",
    "A kitten is resting on the couch.",
    "The stock market fell sharply today.",
    "Shares dropped after the bad news.",
    "I love eating fresh pizza.",
    "Pizza is my favorite food.",
    "The bank is closed on Fridays.",
    "We walked along the river bank.",
    "The delivery was very slow.",
    "My package arrived late.",
]

PAIRS = [(1, 2), (3, 4), (5, 6), (9, 10), (7, 8), (1, 3), (5, 9)]


def similarity_matrix(embeddings):
    size = len(embeddings)
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])
    return matrix


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(SENTENCES)

    print("shape:", embeddings.shape)
    print("first 5 numbers of the first embedding:", embeddings[0][:5])

    print("\nPair similarities (sentence ids start at 1):")
    for a, b in PAIRS:
        score = cosine_similarity(embeddings[a - 1], embeddings[b - 1])
        print(f"  ({a}, {b}): {score:.3f}")

    print("\nFull 10 x 10 similarity matrix:")
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    print(similarity_matrix(embeddings))


if __name__ == "__main__":
    main()