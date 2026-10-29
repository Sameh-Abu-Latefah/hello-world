from pathlib import Path

import matplotlib.pyplot as plt
from embed_demo import SENTENCES
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(SENTENCES)

    points = PCA(n_components=2).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(points[:, 0], points[:, 1])

    for sentence_id, (x, y) in enumerate(points, start=1):
        ax.annotate(
            str(sentence_id),
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=12,
        )

    ax.set_title("Sentence embeddings reduced to 2D with PCA")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")

    output = Path(__file__).resolve().parent.parent / "results" / "plot_sentences.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved plot_sentences.png")


if __name__ == "__main__":
    main()