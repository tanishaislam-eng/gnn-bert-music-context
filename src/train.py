"""
Real-data training orchestrator for GNN-BERT Music Context.

Pipeline:
- Random baseline
- Majority baseline
- CNN mel-spectrogram baseline
- Task 1: BERT-only
- Task 2: GNN-only
- Task 3: Early Concat
- Task 3: GNN-BERT Cross-Attention
- Task 4: MusicCaps InfoNCE + retrieval

Uses:
- MagnaTagATune for tag classification
- Genuine MusicCaps audio-caption pairs for contrastive learning

DEAM/emotion regression is disabled.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.prepare_real_dataset import TOP_TAGS as TAGS_VOCAB
from src.dataset import MusicContextDataset, collate_music_batch
from src.bert_encoder import MusicBertEncoder, BertTagClassifier
from src.gnn_model import MusicGNNClassifier
from src.cnn_baseline import MelSpectrogramCNN
from src.fusion_model import GNNBertFusionModel
from src.contrastive import (
    DualEncoderGNNBERT,
    compute_retrieval_metrics,
    zero_shot_tag_prediction,
)
from src.evaluate import (
    compute_classification_metrics,
    evaluate_random_baseline,
    evaluate_majority_baseline,
    plot_training_curves,
    plot_tsne_embeddings,
    plot_attention_alignment,
    generate_qualitative_retrieval_examples,
)


# ============================================================
# UTILITIES
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def make_loader(
    dataset,
    bert_encoder,
    batch_size=4,
    shuffle=False,
    drop_last=False,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=lambda batch: collate_music_batch(
            batch,
            bert_encoder=bert_encoder,
        ),
    )


# ============================================================
# GENERIC EVALUATION
# ============================================================

def evaluate_model(
    model,
    data_loader,
    device,
    model_type,
):
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():

        for batch in data_loader:

            targets = batch["tag_targets"]
            all_targets.append(
                targets.cpu().numpy()
            )

            if model_type == "bert":

                output = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )

            elif model_type == "cnn":

                output = model(
                    batch["mel_spec"].to(device)
                )

            elif model_type == "gnn":

                output = model(
                    batch["x"].to(device),
                    batch["edge_index"].to(device),
                    batch=batch["batch"].to(device),
                    edge_weight=batch[
                        "edge_weight"
                    ].to(device),
                )

            else:
                raise ValueError(
                    f"Unknown model type: {model_type}"
                )

            all_probs.append(
                output["probs"]
                .cpu()
                .numpy()
            )

    probabilities = np.vstack(all_probs)
    targets = np.vstack(all_targets)

    return compute_classification_metrics(
        targets,
        probabilities,
    )


# ============================================================
# TASK 1 — BERT
# ============================================================

def train_bert(
    train_loader,
    val_loader,
    device,
    epochs=4,
    lr=2e-5,
):

    print("\n" + "=" * 60)
    print("TASK 1 — BERT-ONLY")
    print("=" * 60)

    model = BertTagClassifier(
        num_tags=len(TAGS_VOCAB),
        freeze_backbone=False,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
        "val_micro_f1": [],
    }

    for epoch in range(1, epochs + 1):

        model.train()

        train_losses = []
        train_probs = []
        train_targets = []

        for batch in train_loader:

            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch[
                "attention_mask"
            ].to(device)

            targets = batch[
                "tag_targets"
            ].to(device)

            output = model(
                input_ids,
                attention_mask,
            )

            loss = model.compute_loss(
                output["logits"],
                targets,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            train_losses.append(
                loss.item()
            )

            train_probs.append(
                output["probs"]
                .detach()
                .cpu()
                .numpy()
            )

            train_targets.append(
                targets.cpu().numpy()
            )

        train_probs = np.vstack(
            train_probs
        )

        train_targets = np.vstack(
            train_targets
        )

        train_metrics = compute_classification_metrics(
            train_targets,
            train_probs,
        )

        # ---------------- VALIDATION ----------------

        model.eval()

        val_losses = []
        val_probs = []
        val_targets = []

        with torch.no_grad():

            for batch in val_loader:

                targets = batch[
                    "tag_targets"
                ].to(device)

                output = model(
                    batch[
                        "input_ids"
                    ].to(device),
                    batch[
                        "attention_mask"
                    ].to(device),
                )

                loss = model.compute_loss(
                    output["logits"],
                    targets,
                )

                val_losses.append(
                    loss.item()
                )

                val_probs.append(
                    output["probs"]
                    .cpu()
                    .numpy()
                )

                val_targets.append(
                    targets.cpu()
                    .numpy()
                )

        val_probs = np.vstack(
            val_probs
        )

        val_targets = np.vstack(
            val_targets
        )

        val_metrics = compute_classification_metrics(
            val_targets,
            val_probs,
        )

        train_loss = float(
            np.mean(train_losses)
        )

        val_loss = float(
            np.mean(val_losses)
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_macro_f1"].append(
            train_metrics["macro_f1"]
        )

        history["val_macro_f1"].append(
            val_metrics["macro_f1"]
        )

        history["val_micro_f1"].append(
            val_metrics["micro_f1"]
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro-F1: "
            f"{val_metrics['macro_f1']:.4f}"
        )

    return model, val_metrics, history


# ============================================================
# CNN BASELINE
# ============================================================

def train_cnn(
    train_loader,
    val_loader,
    device,
    epochs=4,
    lr=5e-4,
):

    print("\n" + "=" * 60)
    print("BASELINE — MEL-SPECTROGRAM CNN")
    print("=" * 60)

    model = MelSpectrogramCNN(
        num_tags=len(TAGS_VOCAB)
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    for epoch in range(1, epochs + 1):

        model.train()

        losses = []

        for batch in train_loader:

            optimizer.zero_grad()

            output = model(
                batch["mel_spec"].to(device)
            )

            targets = batch[
                "tag_targets"
            ].to(device)

            loss = model.compute_loss(
                output["logits"],
                targets,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            losses.append(
                loss.item()
            )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {np.mean(losses):.4f}"
        )

    metrics = evaluate_model(
        model,
        val_loader,
        device,
        "cnn",
    )

    print(
        f"CNN Macro-F1: "
        f"{metrics['macro_f1']:.4f}"
    )

    return model, metrics


# ============================================================
# TASK 2 — GNN
# ============================================================

def train_gnn(
    train_loader,
    val_loader,
    device,
    epochs=4,
    lr=5e-4,
):

    print("\n" + "=" * 60)
    print("TASK 2 — GRAPH SAGE GNN")
    print("=" * 60)

    model = MusicGNNClassifier(
        in_dim=32,
        hidden_dim=128,
        out_dim=128,
        num_tags=len(TAGS_VOCAB),
        gnn_type="graphsage",
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    for epoch in range(1, epochs + 1):

        model.train()

        losses = []

        for batch in train_loader:

            optimizer.zero_grad()

            output = model(
                batch["x"].to(device),
                batch["edge_index"].to(device),
                batch=batch["batch"].to(device),
                edge_weight=batch[
                    "edge_weight"
                ].to(device),
            )

            targets = batch[
                "tag_targets"
            ].to(device)

            loss = model.compute_loss(
                output["logits"],
                targets,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            losses.append(
                loss.item()
            )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {np.mean(losses):.4f}"
        )

    metrics = evaluate_model(
        model,
        val_loader,
        device,
        "gnn",
    )

    print(
        f"GNN Macro-F1: "
        f"{metrics['macro_f1']:.4f}"
    )

    return model, metrics


# ============================================================
# TASK 3 — FUSION
# ============================================================

def evaluate_fusion(
    model,
    data_loader,
    device,
):

    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():

        for batch in data_loader:

            output = model(
                batch["x"].to(device),
                batch["edge_index"].to(device),
                batch["input_ids"].to(device),
                attention_mask=batch[
                    "attention_mask"
                ].to(device),
                batch=batch["batch"].to(device),
                edge_weight=batch[
                    "edge_weight"
                ].to(device),
            )

            all_probs.append(
                output["tag_probs"]
                .cpu()
                .numpy()
            )

            all_targets.append(
                batch["tag_targets"]
                .cpu()
                .numpy()
            )

    return compute_classification_metrics(
        np.vstack(all_targets),
        np.vstack(all_probs),
    )


def train_fusion(
    train_loader,
    val_loader,
    device,
    fusion_type,
    epochs=4,
    lr=2e-5,
):

    print("\n" + "=" * 60)
    print(
        f"TASK 3 — GNN-BERT "
        f"{fusion_type.upper()}"
    )
    print("=" * 60)

    model = GNNBertFusionModel(
        num_tags=len(TAGS_VOCAB),
        in_node_dim=32,
        gnn_hidden_dim=128,
        gnn_out_dim=128,
        gnn_type="graphsage",
        fusion_type=fusion_type,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
        "val_micro_f1": [],
    }

    for epoch in range(1, epochs + 1):

        model.train()

        train_losses = []
        train_probs = []
        train_targets = []

        for batch in train_loader:

            optimizer.zero_grad()

            output = model(
                batch["x"].to(device),
                batch["edge_index"].to(device),
                batch["input_ids"].to(device),
                attention_mask=batch[
                    "attention_mask"
                ].to(device),
                batch=batch["batch"].to(device),
                edge_weight=batch[
                    "edge_weight"
                ].to(device),
            )

            targets = batch[
                "tag_targets"
            ].to(device)

            loss = model.compute_loss(
                output["tag_logits"],
                targets,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            train_losses.append(
                loss.item()
            )

            train_probs.append(
                output["tag_probs"]
                .detach()
                .cpu()
                .numpy()
            )

            train_targets.append(
                targets.cpu()
                .numpy()
            )

        train_probs = np.vstack(
            train_probs
        )

        train_targets = np.vstack(
            train_targets
        )

        train_metrics = compute_classification_metrics(
            train_targets,
            train_probs,
        )

        val_metrics = evaluate_fusion(
            model,
            val_loader,
            device,
        )

        train_loss = float(
            np.mean(train_losses)
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            0.0
        )

        history["train_macro_f1"].append(
            train_metrics["macro_f1"]
        )

        history["val_macro_f1"].append(
            val_metrics["macro_f1"]
        )

        history["val_micro_f1"].append(
            val_metrics["micro_f1"]
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Macro-F1: "
            f"{val_metrics['macro_f1']:.4f}"
        )

    return model, val_metrics, history


# ============================================================
# TASK 4 — MUSICCAPS
# ============================================================

def split_musiccaps(
    items,
    seed=42,
):

    items = list(items)

    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)

    # Keep at least 20 genuine MusicCaps pairs
    # for the held-out retrieval test set.
    test_size = max(20, int(round(n * 0.10)))

    # Keep approximately 10% validation data.
    val_size = max(1, int(round(n * 0.10)))

    train_size = n - val_size - test_size

    if train_size < 2:
        raise ValueError(
            "Not enough MusicCaps pairs for "
            "train/validation/test splitting."
        )

    train_end = train_size
    val_end = train_size + val_size

    return (
        items[:train_end],
        items[train_end:val_end],
        items[val_end:],
    )


def train_contrastive(
    train_loader,
    test_loader,
    zero_shot_loader,
    device,
    epochs=4,
    lr=2e-5,
):

    print("\n" + "=" * 60)
    print("TASK 4 — MUSICCAPS CONTRASTIVE LEARNING")
    print("=" * 60)

    model = DualEncoderGNNBERT(
        in_node_dim=32,
        gnn_hidden_dim=128,
        proj_dim=128,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    # --------------------------------------------------------
    # CONTRASTIVE TRAINING
    # --------------------------------------------------------

    for epoch in range(1, epochs + 1):

        model.train()

        losses = []

        for batch in train_loader:

            optimizer.zero_grad()

            z_audio, z_text, _ = model(
                batch["x"].to(device),
                batch["edge_index"].to(device),
                batch["input_ids"].to(device),
                attention_mask=batch[
                    "attention_mask"
                ].to(device),
                batch=batch["batch"].to(device),
                edge_weight=batch[
                    "edge_weight"
                ].to(device),
            )

            loss, _ = model.compute_infonce_loss(
                z_audio,
                z_text,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            losses.append(
                loss.item()
            )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"InfoNCE Loss: "
            f"{np.mean(losses):.4f}"
        )

        # --------------------------------------------------------
    # MUSICCAPS TEST EMBEDDINGS
    # --------------------------------------------------------

    model.eval()

    audio_embeddings = []
    text_embeddings = []

    with torch.no_grad():

        for batch in test_loader:

            z_audio, z_text, _ = model(
                batch["x"].to(device),
                batch["edge_index"].to(device),
                batch["input_ids"].to(device),
                attention_mask=batch[
                    "attention_mask"
                ].to(device),
                batch=batch["batch"].to(device),
                edge_weight=batch[
                    "edge_weight"
                ].to(device),
            )

            audio_embeddings.append(
                z_audio.cpu()
            )

            text_embeddings.append(
                z_text.cpu()
            )

    audio_embeddings = torch.cat(
        audio_embeddings
    )

    text_embeddings = torch.cat(
        text_embeddings
    )

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    retrieval_metrics = compute_retrieval_metrics(
        audio_embeddings,
        text_embeddings,
        top_k_list=[1, 5, 10],
    )

    r1 = retrieval_metrics.get(
        "t2a_r@1",
        0.0,
    )

    r5 = retrieval_metrics.get(
        "t2a_r@5",
        0.0,
    )

    r10 = retrieval_metrics.get(
        "t2a_r@10",
        0.0,
    )

    print(
        f"R@1: {r1:.4f} | "
        f"R@5: {r5:.4f} | "
        f"R@10: {r10:.4f}"
    )
    # --------------------------------------------------------
    # ZERO-SHOT TAG PREDICTION
    #
    # IMPORTANT:
    # MusicCaps provides genuine captions but does not
    # provide the same 50-tag MTT ground-truth vocabulary.
    #
    # Therefore zero-shot tag prediction is evaluated on
    # the real MagnaTagATune test set instead.
    # --------------------------------------------------------

    zero_shot_probs = []
    zero_shot_targets = []

    with torch.no_grad():

        for batch in zero_shot_loader:

            probs = zero_shot_tag_prediction(
                model,
                batch,
                TAGS_VOCAB,
                device=device,
            )

            zero_shot_probs.append(
                probs
            )

            zero_shot_targets.append(
                batch["tag_targets"]
                .cpu()
                .numpy()
            )

    zero_shot_probs = np.vstack(
        zero_shot_probs
    )

    zero_shot_targets = np.vstack(
        zero_shot_targets
    )

    classification_metrics = (
        compute_classification_metrics(
            zero_shot_targets,
            zero_shot_probs,
        )
    )
    

    classification_metrics[
        "r1_retrieval"
    ] = r1

    classification_metrics[
        "r5_retrieval"
    ] = r5

    classification_metrics[
        "r10_retrieval"
    ] = r10

    return model, classification_metrics


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_full_pipeline(
    output_dir="data",
    results_dir="results",
    epochs=4,
    batch_size=4,
):

    set_seed(42)

    device = get_device()

    print("\n")
    print("=" * 70)
    print("GNN-BERT MUSIC CONTEXT PIPELINE")
    print("=" * 70)

    print(f"Device: {device}")

    output_dir = Path(output_dir)
    results_dir = Path(results_dir)

    split_dir = output_dir / "splits"
    processed_dir = output_dir / "processed"

    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    train_path = split_dir / "train.json"
    val_path = split_dir / "val.json"
    test_path = split_dir / "test.json"

    for path in (
        train_path,
        val_path,
        test_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. "
                "Run prepare_real_dataset.py first."
            )

    train_split = load_json(
        train_path
    )

    val_split = load_json(
        val_path
    )

    test_split = load_json(
        test_path
    )

    # --------------------------------------------------------
    # VOCABULARY
    # --------------------------------------------------------

    vocab_path = (
        processed_dir /
        "vocab.json"
    )

    if vocab_path.exists():

        active_vocab = load_json(
            vocab_path
        )

        TAGS_VOCAB[:] = active_vocab

    print(
        f"MTT train: {len(train_split)}"
    )

    print(
        f"MTT validation: {len(val_split)}"
    )

    print(
        f"MTT test: {len(test_split)}"
    )

    print(
        f"Number of tags: "
        f"{len(TAGS_VOCAB)}"
    )

    # --------------------------------------------------------
    # BERT TOKENIZER
    # --------------------------------------------------------

    bert_encoder = MusicBertEncoder()

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = MusicContextDataset(
        train_split,
        processed_dir=str(
            processed_dir
        ),
        num_tags=len(TAGS_VOCAB),
    )

    val_dataset = MusicContextDataset(
        val_split,
        processed_dir=str(
            processed_dir
        ),
        num_tags=len(TAGS_VOCAB),
    )

    test_dataset = MusicContextDataset(
        test_split,
        processed_dir=str(
            processed_dir
        ),
        num_tags=len(TAGS_VOCAB),
    )

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------

    train_loader = make_loader(
        train_dataset,
        bert_encoder,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = make_loader(
        val_dataset,
        bert_encoder,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = make_loader(
        test_dataset,
        bert_encoder,
        batch_size=batch_size,
        shuffle=False,
    )

    # --------------------------------------------------------
    # BASELINE TARGETS
    # --------------------------------------------------------

    train_targets = []

    for batch in train_loader:
        train_targets.append(
            batch["tag_targets"]
            .numpy()
        )

    y_train = np.vstack(
        train_targets
    )

    test_targets = []

    for batch in test_loader:
        test_targets.append(
            batch["tag_targets"]
            .numpy()
        )

    y_test = np.vstack(
        test_targets
    )

    # --------------------------------------------------------
    # RANDOM BASELINE
    # --------------------------------------------------------

    print("\n--- Random baseline ---")

    random_metrics = (
        evaluate_random_baseline(
            y_test
        )
    )

    # --------------------------------------------------------
    # MAJORITY BASELINE
    # --------------------------------------------------------

    print("\n--- Majority baseline ---")

    majority_metrics = (
        evaluate_majority_baseline(
            y_train,
            y_test,
        )
    )

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    cnn_model, cnn_metrics = train_cnn(
        train_loader,
        val_loader,
        device,
        epochs=epochs,
    )

    # --------------------------------------------------------
    # BERT
    # --------------------------------------------------------

    bert_model, bert_metrics, bert_history = train_bert(
        train_loader,
        val_loader,
        device,
        epochs=epochs,
    )

    # --------------------------------------------------------
    # GNN
    # --------------------------------------------------------

    gnn_model, gnn_metrics = train_gnn(
        train_loader,
        val_loader,
        device,
        epochs=epochs,
    )

    # --------------------------------------------------------
    # EARLY CONCAT
    # --------------------------------------------------------

    concat_model, concat_metrics, concat_history = (
        train_fusion(
            train_loader,
            val_loader,
            device,
            "concat",
            epochs=epochs,
        )
    )

    # --------------------------------------------------------
    # CROSS ATTENTION
    # --------------------------------------------------------

    cross_model, cross_metrics, cross_history = (
        train_fusion(
            train_loader,
            val_loader,
            device,
            "cross_attention",
            epochs=epochs,
        )
    )

    # --------------------------------------------------------
    # MUSICCAPS
    # --------------------------------------------------------

    musiccaps_path = (
        split_dir /
        "musiccaps_test.json"
    )

    if not musiccaps_path.exists():
        raise FileNotFoundError(
            "musiccaps_test.json not found."
        )

    musiccaps = load_json(
        musiccaps_path
    )

    if len(musiccaps) < 20:
        raise ValueError(
            "At least 20 MusicCaps pairs "
            "are required."
        )

    mc_train, _, mc_test = split_musiccaps(
        musiccaps
    )

    mc_train_dataset = MusicContextDataset(
        mc_train,
        processed_dir=str(
            processed_dir
        ),
        num_tags=len(TAGS_VOCAB),
    )

    mc_test_dataset = MusicContextDataset(
        mc_test,
        processed_dir=str(
            processed_dir
        ),
        num_tags=len(TAGS_VOCAB),
    )

    mc_train_loader = make_loader(
        mc_train_dataset,
        bert_encoder,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    mc_test_loader = make_loader(
        mc_test_dataset,
        bert_encoder,
        batch_size=batch_size,
        shuffle=False,
    )

    contrastive_model, contrastive_metrics = (
        train_contrastive(
            mc_train_loader,
            mc_test_loader,
            test_loader,
            device,
            epochs=epochs,
        )
    )

    
    # ========================================================
    # RESULTS TABLE
    # ========================================================

    table3 = {
        "Random tags": {
            "macro_f1": random_metrics[
                "macro_f1"
            ],
            "auc_pr": random_metrics[
                "mean_auc_pr"
            ],
            "mae_emotion": "-",
            "r5_retrieval": "-",
        },

        "Majority tags": {
            "macro_f1": majority_metrics[
                "macro_f1"
            ],
            "auc_pr": majority_metrics[
                "mean_auc_pr"
            ],
            "mae_emotion": "-",
            "r5_retrieval": "-",
        },

        "CNN mel-spec": {
            "macro_f1": cnn_metrics[
                "macro_f1"
            ],
            "auc_pr": cnn_metrics[
                "mean_auc_pr"
            ],
            "mae_emotion": "-",
            "r5_retrieval": "-",
        },

        "Task 1: BERT-only": {
            "macro_f1": bert_metrics[
                "macro_f1"
            ],
            "auc_pr": bert_metrics[
                "mean_auc_pr"
            ],
            "mae_emotion": "-",
            "r5_retrieval": "-",
        },

        "Task 2: GNN-only": {
            "macro_f1": gnn_metrics[
                "macro_f1"
            ],
            "auc_pr": gnn_metrics[
                "mean_auc_pr"
            ],
            "mae_emotion": "-",
            "r5_retrieval": "-",
        },

        "Task 3: Early Concat": {
            "macro_f1": concat_metrics[
                "macro_f1"
            ],
            "auc_pr": concat_metrics[
                "mean_auc_pr"
            ],
            "mae_emotion": "-",
            "r5_retrieval": "-",
        },

        "Task 3: GNN-BERT (Cross-Attn)": {
            "macro_f1": cross_metrics[
                "macro_f1"
            ],
            "auc_pr": cross_metrics[
                "mean_auc_pr"
            ],
            "mae_emotion": "-",
            "r5_retrieval": "-",
        },

        "Task 4: Contrastive (Zero-Shot)": {
            "macro_f1": contrastive_metrics[
                "macro_f1"
            ],
            "auc_pr": contrastive_metrics[
                "mean_auc_pr"
            ],
            "mae_emotion": "-",
            "r5_retrieval": contrastive_metrics[
                "r5_retrieval"
            ],
            "r1_retrieval": contrastive_metrics[
                "r1_retrieval"
            ],
            "r10_retrieval": contrastive_metrics[
                "r10_retrieval"
            ],
        },
    }

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        results_dir /
        "metrics.json"
    )

    with open(
        metrics_path,
        "w",
    ) as f:
        json.dump(
            table3,
            f,
            indent=2,
        )

    print(
        f"\nSaved results to: "
        f"{metrics_path}"
    )

    # ========================================================
    # PLOTS
    # ========================================================

    plots_dir = (
        results_dir /
        "plots"
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        plot_training_curves(
            cross_history,
            str(
                plots_dir /
                "f1_loss_curves.png"
            ),
        )

    except Exception as e:

        print(
            f"Training curve skipped: {e}"
        )

    # --------------------------------------------------------
    # t-SNE
    # --------------------------------------------------------

    try:

        cross_model.eval()

        embeddings = []
        labels_1 = []
        labels_2 = []

        with torch.no_grad():

            for item in test_dataset:

                batch = collate_music_batch(
                    [item],
                    bert_encoder=bert_encoder,
                )

                output = cross_model(
                    batch["x"].to(device),
                    batch["edge_index"].to(device),
                    batch["input_ids"].to(device),
                    attention_mask=batch[
                        "attention_mask"
                    ].to(device),
                    batch=batch[
                        "batch"
                    ].to(device),
                    edge_weight=batch[
                        "edge_weight"
                    ].to(device),
                )

                embeddings.append(
                    output["z"]
                    .cpu()
                    .numpy()
                )

                labels_1.append(
                    "MagnaTagATune"
                )

                tags = item.get(
                    "tags",
                    []
                )

                labels_2.append(
                    tags[0]
                    if tags
                    else "other"
                )

        if len(embeddings) >= 3:

            plot_tsne_embeddings(
                np.vstack(
                    embeddings
                ),
                labels_1,
                labels_2,
                str(
                    plots_dir /
                    "tsne_multimodal.png"
                ),
            )

    except Exception as e:

        print(
            f"t-SNE plot skipped: {e}"
        )

    # --------------------------------------------------------
    # ATTENTION HEATMAP
    # --------------------------------------------------------

    try:

        sample = test_dataset[0]

        batch = collate_music_batch(
            [sample],
            bert_encoder=bert_encoder,
        )

        cross_model.eval()

        with torch.no_grad():

            output = cross_model(
                batch["x"].to(device),
                batch["edge_index"].to(device),
                batch["input_ids"].to(device),
                attention_mask=batch[
                    "attention_mask"
                ].to(device),
                batch=batch[
                    "batch"
                ].to(device),
                edge_weight=batch[
                    "edge_weight"
                ].to(device),
            )

        if output.get(
            "att_weights"
        ) is not None:

            attention = (
                output["att_weights"]
                .squeeze()
                .cpu()
                .numpy()
            )

            words = sample.get(
                "caption",
                ""
            ).split()

            n = min(
                len(words),
                len(attention),
                20,
            )

            if n > 0:

                plot_attention_alignment(
                    words[:n],
                    ["Graph Readout"],
                    attention[:n].reshape(
                        1,
                        -1,
                    ),
                    str(
                        plots_dir /
                        "cross_attention_heatmap.png"
                    ),
                )

    except Exception as e:

        print(
            f"Attention heatmap skipped: {e}"
        )

    # ========================================================
    # QUALITATIVE RETRIEVAL
    # ========================================================

    retrieval_dir = (
        results_dir /
        "retrieval_examples"
    )

    retrieval_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        contrastive_model.eval()

        query_items = mc_test[:10]

        query_captions = [
            item.get(
                "caption",
                "",
            )
            for item in query_items
        ]

        retrieved_results = []

        with torch.no_grad():

            for caption in query_captions:

                tokens = bert_encoder.tokenize(
                    [caption],
                    device=device,
                )

                z_text = (
                    contrastive_model
                    .encode_text(
                        tokens["input_ids"],
                        tokens.get(
                            "attention_mask"
                        ),
                    )
                )

                matches = []

                for track in mc_test_dataset:

                    track_batch = (
                        collate_music_batch(
                            [track],
                            bert_encoder=bert_encoder,
                        )
                    )

                    z_audio = (
                        contrastive_model
                        .encode_audio_graph(
                            track_batch[
                                "x"
                            ].to(device),
                            track_batch[
                                "edge_index"
                            ].to(device),
                            edge_weight=
                            track_batch[
                                "edge_weight"
                            ].to(device),
                        )
                    )

                    similarity = float(
                        torch.matmul(
                            z_text,
                            z_audio.t(),
                        ).item()
                    )

                    matches.append(
                        {
                            "track_id":
                                track.get(
                                    "track_id",
                                    "",
                                ),
                            "caption":
                                track.get(
                                    "caption",
                                    "",
                                ),
                            "score":
                                round(
                                    similarity,
                                    4,
                                ),
                        }
                    )

                matches.sort(
                    key=lambda x:
                        x["score"],
                    reverse=True,
                )

                retrieved_results.append(
                    matches[:10]
                )

        generate_qualitative_retrieval_examples(
            query_captions,
            retrieved_results,
            save_path=str(
                retrieval_dir /
                "retrieval_samples"
            ),
        )

    except Exception as e:

        print(
            f"Retrieval examples skipped: {e}"
        )

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n" + "=" * 70)
    print("PIPELINE FINISHED")
    print("=" * 70)

    print(
        f"Metrics: {metrics_path}"
    )

    print(
        f"Plots: {plots_dir}"
    )

    return table3


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Real-data GNN-BERT "
            "Music Context Pipeline"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Training epochs",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Processed data directory",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )

    args = parser.parse_args()

    run_full_pipeline(
        output_dir=args.data_dir,
        results_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )