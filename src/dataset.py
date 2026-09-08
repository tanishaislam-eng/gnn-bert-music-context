"""
PyTorch Dataset and Collation Pipeline for Music Multi-Modal Graphs and Text.

Supports:
- Real preprocessed audio graphs
- Real log-mel spectrograms
- 50-tag multi-label classification
- Real MusicCaps captions
- BERT tokenization
- Batched disjoint graphs

Emotion/DEAM targets are intentionally removed because DEAM
is not part of the selected project pipeline.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MusicContextDataset(Dataset):
    """
    Dataset representing:

        T = (X_audio, X_text, G, y)

    where:
        X_audio  = log-mel spectrogram
        X_text   = title/artist context or MusicCaps caption
        G        = acoustic segment graph
        y        = multi-label music tags
    """

    def __init__(
        self,
        metadata_list: list[dict],
        processed_dir: str = "data/processed",
        num_tags: int = 50,
    ):
        self.metadata = metadata_list
        self.processed_dir = processed_dir
        self.num_tags = num_tags

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        item = self.metadata[idx]
        track_id = item["track_id"]

        # ---------------------------------------------------------
        # 1. Load real preprocessed graph
        # ---------------------------------------------------------
        pt_path = os.path.join(
            self.processed_dir,
            f"{track_id}_graph.pt",
        )

        json_path = os.path.join(
            self.processed_dir,
            f"{track_id}_graph.json",
        )

        if os.path.exists(pt_path):
            graph_data = torch.load(
                pt_path,
                weights_only=False,
            )

            x = graph_data["x"].float()
            edge_index = graph_data["edge_index"].long()

            if "edge_weight" in graph_data:
                edge_weight = graph_data["edge_weight"].float()
            else:
                edge_weight = torch.ones(
                    edge_index.size(1),
                    dtype=torch.float32,
                )

        elif os.path.exists(json_path):
            with open(json_path, "r") as f:
                graph_data = json.load(f)

            x = torch.tensor(
                graph_data["x"],
                dtype=torch.float32,
            )

            edge_index = torch.tensor(
                graph_data["edge_index"],
                dtype=torch.long,
            )

            edge_weights = graph_data.get(
                "edge_weight",
                [1.0] * edge_index.shape[1],
            )

            edge_weight = torch.tensor(
                edge_weights,
                dtype=torch.float32,
            )

        else:
            raise FileNotFoundError(
                f"Real graph file not found for track "
                f"{track_id}."
            )

        # ---------------------------------------------------------
        # 2. Load real log-mel spectrogram
        # ---------------------------------------------------------
        mel_path = os.path.join(
            self.processed_dir,
            f"{track_id}_mel.npy",
        )

        if not os.path.exists(mel_path):
            raise FileNotFoundError(
                f"Real log-mel spectrogram not found for "
                f"track {track_id}: {mel_path}"
            )

        mel_spec = torch.from_numpy(
            np.load(mel_path)
        ).float()

        # Expected format:
        # (128, time)

        # ---------------------------------------------------------
        # 3. Create 50-dimensional multi-label tag vector
        # ---------------------------------------------------------
        tag_vector = torch.zeros(
            self.num_tags,
            dtype=torch.float32,
        )

        for tag_idx in item.get("tag_indices", []):
            try:
                tag_idx = int(tag_idx)
            except (TypeError, ValueError):
                continue

            if 0 <= tag_idx < self.num_tags:
                tag_vector[tag_idx] = 1.0

        # ---------------------------------------------------------
        # 4. Real text context
        # ---------------------------------------------------------
        title = str(
            item.get("title", "")
        ).strip()

        artist = str(
            item.get("artist", "")
        ).strip()

        caption = str(
            item.get("caption", "")
        ).strip()

        # MusicCaps has genuine captions.
        # MTT uses title/artist context prepared by the
        # real-data preprocessing pipeline.
        if not caption:
            if title and artist:
                caption = f"{title} by {artist}"
            elif title:
                caption = title
            elif artist:
                caption = artist
            else:
                caption = "music"

        return {
            "track_id": track_id,
            "title": title,
            "artist": artist,
            "genre": item.get("genre", "Unknown"),
            "caption": caption,
            "tags": item.get("tags", []),

            # Graph
            "x": x,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "num_nodes": x.size(0),

            # Audio
            "mel_spec": mel_spec,

            # Labels
            "tag_targets": tag_vector,
        }


def collate_music_batch(
    batch: list[dict],
    bert_encoder=None,
    max_length: int = 128,
) -> dict:
    """
    Collate function for multimodal batches.

    Creates:
    - One disjoint batched graph
    - Batched mel spectrograms
    - Batched tag labels
    - BERT input IDs
    - BERT attention masks
    """

    batch_size = len(batch)

    node_features = []
    edge_indices = []
    edge_weights = []
    batch_vectors = []

    mel_specs = []
    tag_targets = []

    captions = []
    track_ids = []

    node_offset = 0

    # ---------------------------------------------------------
    # Process each sample
    # ---------------------------------------------------------
    for batch_idx, item in enumerate(batch):

        # -------------------------
        # Graph
        # -------------------------
        x = item["x"]
        edge_index = item["edge_index"].clone()
        edge_weight = item["edge_weight"]

        node_features.append(x)

        # Shift node indices so graphs form one
        # disjoint graph.
        edge_index = edge_index + node_offset

        edge_indices.append(edge_index)
        edge_weights.append(edge_weight)

        batch_vectors.append(
            torch.full(
                (item["num_nodes"],),
                batch_idx,
                dtype=torch.long,
            )
        )

        node_offset += item["num_nodes"]

        # -------------------------
        # Mel spectrogram
        # -------------------------
        mel = item["mel_spec"]

        # Ensure exactly 128 mel bins.
        if mel.shape[0] != 128:

            if mel.shape[0] > 128:
                mel = mel[:128, :]

            else:
                pad = torch.zeros(
                    128 - mel.shape[0],
                    mel.shape[1],
                    dtype=mel.dtype,
                )

                mel = torch.cat(
                    [mel, pad],
                    dim=0,
                )

        # Standardize time dimension to 128.
        if mel.shape[1] != 128:

            if mel.shape[1] > 128:
                mel = mel[:, :128]

            else:
                pad = torch.zeros(
                    mel.shape[0],
                    128 - mel.shape[1],
                    dtype=mel.dtype,
                )

                mel = torch.cat(
                    [mel, pad],
                    dim=1,
                )

        mel_specs.append(mel)

        # -------------------------
        # Labels / text
        # -------------------------
        tag_targets.append(
            item["tag_targets"]
        )

        captions.append(
            item["caption"]
        )

        track_ids.append(
            item["track_id"]
        )

    # ---------------------------------------------------------
    # Batched graph
    # ---------------------------------------------------------
    batched_x = torch.cat(
        node_features,
        dim=0,
    )

    batched_edge_index = torch.cat(
        edge_indices,
        dim=1,
    )

    batched_edge_weight = torch.cat(
        edge_weights,
        dim=0,
    )

    batched_batch_vector = torch.cat(
        batch_vectors,
        dim=0,
    )

    # ---------------------------------------------------------
    # Batched audio
    # ---------------------------------------------------------
    batched_mel = torch.stack(
        mel_specs,
        dim=0,
    )

    # Shape:
    # (batch, 128, 128)

    # ---------------------------------------------------------
    # Batched labels
    # ---------------------------------------------------------
    batched_tag_targets = torch.stack(
        tag_targets,
        dim=0,
    )

    # Shape:
    # (batch, 50)

    # ---------------------------------------------------------
    # BERT tokenization
    # ---------------------------------------------------------
    if bert_encoder is not None:

        tokens = bert_encoder.tokenize(
            captions,
            max_length=max_length,
        )

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

    else:

        # Placeholder tensors only when the BERT encoder
        # is intentionally not being used.
        input_ids = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
        )

        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
        )

    # ---------------------------------------------------------
    # Final batch
    # ---------------------------------------------------------
    return {
        "track_ids": track_ids,
        "captions": captions,

        # Graph
        "x": batched_x,
        "edge_index": batched_edge_index,
        "edge_weight": batched_edge_weight,
        "batch": batched_batch_vector,

        # Audio
        "mel_spec": batched_mel,

        # BERT
        "input_ids": input_ids,
        "attention_mask": attention_mask,

        # Labels
        "tag_targets": batched_tag_targets,
    }