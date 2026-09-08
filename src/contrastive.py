"""
Contrastive Cross-Modal Alignment (Task 4)

Implements:
- GNN audio encoder
- DistilBERT text encoder
- Shared projection space
- Symmetric InfoNCE loss
- Audio-to-text and text-to-audio Recall@K
- Zero-shot tag prediction

Uses genuine MusicCaps audio-caption pairs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bert_encoder import MusicBertEncoder
from src.gnn_model import MusicGNNEncoder


class DualEncoderGNNBERT(nn.Module):
    """
    Dual encoder for cross-modal music-text alignment.

    Audio:
        Graph -> GNN -> projection -> normalized embedding

    Text:
        Caption -> DistilBERT -> projection -> normalized embedding
    """

    def __init__(
        self,
        in_node_dim: int = 32,
        gnn_hidden_dim: int = 128,
        gnn_out_dim: int = 128,
        gnn_type: str = "graphsage",
        bert_model_name: str = "distilbert-base-uncased",
        proj_dim: int = 128,
        temperature: float = 0.07,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.gnn = MusicGNNEncoder(
            in_dim=in_node_dim,
            hidden_dim=gnn_hidden_dim,
            out_dim=gnn_out_dim,
            gnn_type=gnn_type,
            dropout=dropout,
        )

        # Genuine pretrained DistilBERT.
        # Frozen to keep Task 4 computationally manageable.
        self.bert = MusicBertEncoder(
            model_name=bert_model_name,
            freeze_backbone=True,
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(gnn_out_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(self.bert.hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

        temperature = max(float(temperature), 1e-3)

        self.logit_scale = nn.Parameter(
            torch.tensor(
                np.log(1.0 / temperature),
                dtype=torch.float32,
            )
        )

    # =========================================================
    # AUDIO ENCODER
    # =========================================================

    def encode_audio_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:

        _, graph_embedding = self.gnn(
            x,
            edge_index,
            batch=batch,
            edge_weight=edge_weight,
        )

        z_audio = self.audio_proj(graph_embedding)

        return F.normalize(
            z_audio,
            p=2,
            dim=-1,
        )

    # =========================================================
    # TEXT ENCODER
    # =========================================================

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        _, cls_embedding = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )

        z_text = self.text_proj(cls_embedding)

        return F.normalize(
            z_text,
            p=2,
            dim=-1,
        )

    # =========================================================
    # FORWARD
    # =========================================================

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        batch: torch.Tensor = None,
        edge_weight: torch.Tensor = None,
    ):

        z_audio = self.encode_audio_graph(
            x,
            edge_index,
            batch=batch,
            edge_weight=edge_weight,
        )

        z_text = self.encode_text(
            input_ids,
            attention_mask=attention_mask,
        )

        logit_scale = torch.clamp(
            self.logit_scale.exp(),
            min=1.0,
            max=100.0,
        )

        temperature = 1.0 / logit_scale

        return z_audio, z_text, temperature

    # =========================================================
    # INFONCE
    # =========================================================

    def compute_infonce_loss(
        self,
        z_audio: torch.Tensor,
        z_text: torch.Tensor,
    ):

        batch_size = z_audio.size(0)

        if batch_size < 2:
            raise ValueError(
                "InfoNCE requires a batch size of at least 2."
            )

        z_audio = F.normalize(
            z_audio,
            p=2,
            dim=-1,
        )

        z_text = F.normalize(
            z_text,
            p=2,
            dim=-1,
        )

        logit_scale = torch.clamp(
            self.logit_scale.exp(),
            min=1.0,
            max=100.0,
        )

        similarity = (
            torch.matmul(
                z_audio,
                z_text.transpose(0, 1),
            )
            * logit_scale
        )

        targets = torch.arange(
            batch_size,
            device=z_audio.device,
        )

        loss_audio_to_text = F.cross_entropy(
            similarity,
            targets,
        )

        loss_text_to_audio = F.cross_entropy(
            similarity.transpose(0, 1),
            targets,
        )

        loss = (
            loss_audio_to_text
            + loss_text_to_audio
        ) / 2.0

        return loss, {
            "loss_contrastive": float(
                loss.detach().cpu()
            ),
            "loss_a2t": float(
                loss_audio_to_text.detach().cpu()
            ),
            "loss_t2a": float(
                loss_text_to_audio.detach().cpu()
            ),
            "temperature": float(
                (1.0 / logit_scale)
                .detach()
                .cpu()
            ),
        }


# =============================================================
# RETRIEVAL METRICS
# =============================================================

def compute_retrieval_metrics(
    audio_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    top_k_list=None,
) -> dict:

    if top_k_list is None:
        top_k_list = [1, 5, 10]

    if audio_embeddings.size(0) != text_embeddings.size(0):
        raise ValueError(
            "Audio and text embeddings must contain "
            "the same number of paired samples."
        )

    audio_embeddings = F.normalize(
        audio_embeddings,
        p=2,
        dim=-1,
    )

    text_embeddings = F.normalize(
        text_embeddings,
        p=2,
        dim=-1,
    )

    similarity = torch.matmul(
        text_embeddings,
        audio_embeddings.transpose(0, 1),
    )

    similarity = similarity.detach().cpu().numpy()

    num_samples = similarity.shape[0]

    metrics = {}

    if num_samples == 0:
        for k in top_k_list:
            metrics[f"t2a_r@{k}"] = 0.0
            metrics[f"a2t_r@{k}"] = 0.0
        return metrics

    # Text -> Audio
    text_to_audio_ranks = np.argsort(
        -similarity,
        axis=1,
    )

    # Audio -> Text
    audio_to_text_ranks = np.argsort(
        -similarity.T,
        axis=1,
    )

    for k in top_k_list:

        k_eff = min(
            int(k),
            num_samples,
        )

        correct_t2a = sum(
            i in text_to_audio_ranks[i, :k_eff]
            for i in range(num_samples)
        )

        correct_a2t = sum(
            i in audio_to_text_ranks[i, :k_eff]
            for i in range(num_samples)
        )

        metrics[f"t2a_r@{k}"] = round(
            correct_t2a / num_samples,
            4,
        )

        metrics[f"a2t_r@{k}"] = round(
            correct_a2t / num_samples,
            4,
        )

    return metrics


# =============================================================
# ZERO-SHOT TAG PREDICTION
# =============================================================

def zero_shot_tag_prediction(
    model: DualEncoderGNNBERT,
    audio_graph_batch: dict,
    candidate_tags: list[str],
    device: torch.device,
) -> np.ndarray:
    """
    Predict tags without training a dedicated tag classifier.

    Candidate tags are converted to text prompts and embedded
    using the same BERT projection space as the captions.
    """

    if not candidate_tags:
        raise ValueError(
            "candidate_tags cannot be empty."
        )

    model.eval()

    with torch.no_grad():

        # -----------------------------------------------------
        # 1. Encode candidate tag prompts
        # -----------------------------------------------------

        tag_prompts = [
            f"This music track features {tag} style and instrumentation."
            for tag in candidate_tags
        ]

        text_inputs = model.bert.tokenize(
            tag_prompts,
            max_length=64,
            device=device,
        )

        z_tags = model.encode_text(
            text_inputs["input_ids"],
            text_inputs.get("attention_mask"),
        )

        # -----------------------------------------------------
        # 2. Prepare audio graph
        # -----------------------------------------------------

        x = audio_graph_batch["x"].to(device)

        edge_index = audio_graph_batch[
            "edge_index"
        ].to(device)

        batch_vector = audio_graph_batch.get("batch")

        if batch_vector is not None:
            batch_vector = batch_vector.to(device)

        edge_weight = audio_graph_batch.get(
            "edge_weight"
        )

        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        # -----------------------------------------------------
        # 3. Encode audio
        # -----------------------------------------------------

        z_audio = model.encode_audio_graph(
            x,
            edge_index,
            batch=batch_vector,
            edge_weight=edge_weight,
        )

        # -----------------------------------------------------
        # 4. Audio-tag similarity
        # -----------------------------------------------------

        similarities = torch.matmul(
            z_audio,
            z_tags.transpose(0, 1),
        )

        # Cosine similarity -> [0, 1]
        probabilities = (
            similarities + 1.0
        ) / 2.0

        return probabilities.cpu().numpy()