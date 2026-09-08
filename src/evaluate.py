"""
Evaluation Metrics, Baselines, and Visual Analysis.

Current project setup:
- Multi-label tag classification
- Macro-F1 / Micro-F1 / AUC-PR
- R@K retrieval
- t-SNE visualization
- Cross-attention visualization
- Qualitative MusicCaps retrieval examples

DEAM/emotion regression has been removed.
"""

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
)
from sklearn.manifold import TSNE

try:
    import torch
except ImportError:
    torch = None


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Computes Macro-F1, Micro-F1, precision, recall,
    and mean AUC-PR for multi-label classification.

    y_true:
        Shape (N, K), binary targets.

    y_prob:
        Shape (N, K), predicted probabilities.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_true.ndim != 2:
        raise ValueError(
            "y_true must have shape (N, K)."
        )

    if y_prob.shape != y_true.shape:
        raise ValueError(
            "y_prob and y_true must have "
            "the same shape."
        )

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    # -------------------------
    # F1
    # -------------------------

    macro_p, macro_r, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    micro_p, micro_r, micro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        )
    )

    # -------------------------
    # AUC-PR
    # -------------------------

    auc_prs = []

    for k in range(y_true.shape[1]):

        positives = np.sum(
            y_true[:, k]
        )

        if positives > 0:

            ap = average_precision_score(
                y_true[:, k],
                y_prob[:, k],
            )

            auc_prs.append(
                float(ap)
            )

    if auc_prs:
        mean_auc_pr = float(
            np.mean(auc_prs)
        )
    else:
        mean_auc_pr = 0.0

    return {
        "macro_f1": round(
            float(macro_f1),
            4,
        ),
        "micro_f1": round(
            float(micro_f1),
            4,
        ),
        "macro_precision": round(
            float(macro_p),
            4,
        ),
        "macro_recall": round(
            float(macro_r),
            4,
        ),
        "mean_auc_pr": round(
            mean_auc_pr,
            4,
        ),
        "auc_pr_per_class": [
            round(float(v), 4)
            for v in auc_prs
        ],
    }


# ============================================================
# BASELINES
# ============================================================

def evaluate_random_baseline(
    y_true: np.ndarray,
    num_retrieval_samples: int = None,
    seed: int = 42,
) -> dict:
    """
    Random uniform probability baseline.
    """

    rng = np.random.default_rng(
        seed
    )

    y_prob = rng.uniform(
        0.0,
        1.0,
        size=y_true.shape,
    )

    metrics = compute_classification_metrics(
        y_true,
        y_prob,
    )

    n = (
        num_retrieval_samples
        if num_retrieval_samples
        and num_retrieval_samples > 0
        else len(y_true)
    )

    metrics["r5_retrieval"] = round(
        min(
            1.0,
            5.0 / max(1, n),
        ),
        4,
    )

    return metrics


def evaluate_majority_baseline(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Predicts the training-set tag frequency
    for every test example.
    """

    priors = np.mean(
        y_train,
        axis=0,
    )

    y_prob = np.tile(
        priors,
        (y_test.shape[0], 1),
    )

    return compute_classification_metrics(
        y_test,
        y_prob,
    )


# ============================================================
# TRAINING CURVES
# ============================================================

def plot_training_curves(
    history: dict,
    save_path: str,
):
    """
    Saves training/validation loss and F1 curves.
    """

    directory = os.path.dirname(
        save_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    train_loss = history.get(
        "train_loss",
        [],
    )

    val_loss = history.get(
        "val_loss",
        [],
    )

    train_f1 = history.get(
        "train_macro_f1",
        [],
    )

    val_f1 = history.get(
        "val_macro_f1",
        [],
    )

    val_micro_f1 = history.get(
        "val_micro_f1",
        [],
    )

    n_epochs = max(
        len(train_loss),
        len(train_f1),
    )

    epochs = np.arange(
        1,
        n_epochs + 1,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    # -------------------------
    # Loss
    # -------------------------

    if train_loss:
        axes[0].plot(
            epochs[:len(train_loss)],
            train_loss,
            "o-",
            label="Train Loss",
            linewidth=2,
        )

    if val_loss and any(
        float(x) != 0.0
        for x in val_loss
    ):
        axes[0].plot(
            epochs[:len(val_loss)],
            val_loss,
            "s--",
            label="Validation Loss",
            linewidth=2,
        )

    axes[0].set_title(
        "Training and Validation Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    axes[0].legend()

    # -------------------------
    # F1
    # -------------------------

    if train_f1:
        axes[1].plot(
            epochs[:len(train_f1)],
            train_f1,
            "o-",
            label="Train Macro-F1",
            linewidth=2,
        )

    if val_f1:
        axes[1].plot(
            epochs[:len(val_f1)],
            val_f1,
            "s--",
            label="Validation Macro-F1",
            linewidth=2,
        )

    if val_micro_f1:
        axes[1].plot(
            epochs[:len(val_micro_f1)],
            val_micro_f1,
            "^:",
            label="Validation Micro-F1",
            linewidth=2,
        )

    axes[1].set_title(
        "Macro-F1 and Micro-F1"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "F1 Score"
    )

    axes[1].grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# t-SNE
# ============================================================

def plot_tsne_embeddings(
    embeddings: np.ndarray,
    genres: list[str],
    moods: list[str],
    save_path: str,
    seed: int = 42,
):
    """
    Creates a two-panel t-SNE visualization.

    The current dataset does not contain genuine genre/mood
    labels, so the supplied labels are treated only as
    visualization categories.
    """

    directory = os.path.dirname(
        save_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    embeddings = np.asarray(
        embeddings
    )

    n_samples = len(
        embeddings
    )

    if n_samples < 3:
        raise ValueError(
            "At least 3 samples are required "
            "for t-SNE."
        )

    perplexity = min(
        30,
        max(
            2,
            n_samples // 3,
        ),
    )

    # sklearn requires perplexity < n_samples
    perplexity = min(
        perplexity,
        n_samples - 1,
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )

    z_2d = tsne.fit_transform(
        embeddings
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    # -------------------------
    # First label set
    # -------------------------

    unique_labels = sorted(
        set(genres)
    )

    for label in unique_labels:

        idx = [
            i
            for i, value in enumerate(genres)
            if value == label
        ]

        axes[0].scatter(
            z_2d[idx, 0],
            z_2d[idx, 1],
            label=str(label),
            s=45,
            alpha=0.8,
        )

    axes[0].set_title(
        "t-SNE of Multimodal Representation"
    )

    axes[0].set_xlabel(
        "Dimension 1"
    )

    axes[0].set_ylabel(
        "Dimension 2"
    )

    axes[0].grid(
        True,
        linestyle=":",
        alpha=0.4,
    )

    if unique_labels:
        axes[0].legend(
            fontsize=8,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

    # -------------------------
    # Second label set
    # -------------------------

    unique_labels = sorted(
        set(moods)
    )

    for label in unique_labels:

        idx = [
            i
            for i, value in enumerate(moods)
            if value == label
        ]

        axes[1].scatter(
            z_2d[idx, 0],
            z_2d[idx, 1],
            label=str(label),
            s=45,
            alpha=0.8,
        )

    axes[1].set_title(
        "t-SNE by Tag Category"
    )

    axes[1].set_xlabel(
        "Dimension 1"
    )

    axes[1].set_ylabel(
        "Dimension 2"
    )

    axes[1].grid(
        True,
        linestyle=":",
        alpha=0.4,
    )

    if unique_labels:
        axes[1].legend(
            fontsize=8,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# ATTENTION VISUALIZATION
# ============================================================

def plot_attention_alignment(
    tokens: list[str],
    graph_node_labels: list[str],
    att_matrix: np.ndarray,
    save_path: str,
):
    """
    Saves a cross-attention visualization.

    att_matrix should have shape:
        (graph_nodes, text_tokens)
    """

    directory = os.path.dirname(
        save_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    att_matrix = np.asarray(
        att_matrix
    )

    if att_matrix.ndim != 2:
        raise ValueError(
            "att_matrix must be 2-dimensional."
        )

    # Keep labels aligned with matrix.
    n_rows = min(
        len(graph_node_labels),
        att_matrix.shape[0],
    )

    n_cols = min(
        len(tokens),
        att_matrix.shape[1],
    )

    att_matrix = att_matrix[
        :n_rows,
        :n_cols,
    ]

    tokens = tokens[
        :n_cols
    ]

    graph_node_labels = (
        graph_node_labels[
            :n_rows
        ]
    )

    fig_width = max(
        8,
        n_cols * 0.55,
    )

    fig_height = max(
        3,
        n_rows * 0.45,
    )

    fig, ax = plt.subplots(
        figsize=(
            fig_width,
            fig_height,
        )
    )

    image = ax.imshow(
        att_matrix,
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(n_cols)
    )

    ax.set_xticklabels(
        tokens,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        np.arange(n_rows)
    )

    ax.set_yticklabels(
        graph_node_labels
    )

    ax.set_xlabel(
        "Caption Tokens"
    )

    ax.set_ylabel(
        "Audio Graph Representation"
    )

    ax.set_title(
        "GNN-BERT Cross-Attention Alignment"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Attention Weight",
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# QUALITATIVE RETRIEVAL
# ============================================================

def generate_qualitative_retrieval_examples(
    query_captions: list[str],
    retrieved_results: list[list[dict]],
    save_path: str,
):
    """
    Saves qualitative cross-modal retrieval examples
    in JSON and Markdown.

    Only fields that actually exist in retrieved results
    are displayed.
    """

    directory = os.path.dirname(
        save_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    report_data = []

    for i, (
        caption,
        matches,
    ) in enumerate(
        zip(
            query_captions,
            retrieved_results,
        )
    ):

        report_data.append(
            {
                "query_id": i + 1,
                "query_caption": caption,
                "top_3_retrieved_clips":
                    matches[:3],
            }
        )

    # -------------------------
    # JSON
    # -------------------------

    with open(
        f"{save_path}.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report_data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -------------------------
    # Markdown
    # -------------------------

    md_lines = [
        "# Cross-Modal MusicCaps Retrieval\n\n",
        "Top-3 retrieved audio clips for each "
        "caption query.\n\n",
    ]

    for example in report_data:

        md_lines.append(
            f"## Query #{example['query_id']}\n\n"
        )

        md_lines.append(
            "**Caption:** "
            f"{example['query_caption']}\n\n"
        )

        md_lines.append(
            "| Rank | Track ID | Similarity Score | "
            "Caption |\n"
        )

        md_lines.append(
            "|---|---|---|---|\n"
        )

        for rank, match in enumerate(
            example[
                "top_3_retrieved_clips"
            ],
            1,
        ):

            track_id = match.get(
                "track_id",
                "unknown",
            )

            score = match.get(
                "score",
                0.0,
            )

            caption = match.get(
                "caption",
                "",
            )

            md_lines.append(
                f"| #{rank} | "
                f"`{track_id}` | "
                f"`{float(score):.4f}` | "
                f"{caption} |\n"
            )

        md_lines.append(
            "\n---\n\n"
        )

    with open(
        f"{save_path}.md",
        "w",
        encoding="utf-8",
    ) as f:

        f.writelines(
            md_lines
        )