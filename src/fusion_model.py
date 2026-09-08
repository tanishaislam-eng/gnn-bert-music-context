"""
GNN-BERT Multi-Modal Fusion Model (Task 3)

Supports:
- Cross-attention fusion
- Early concatenation fusion
- BERT-only ablation
- GNN-only ablation
- Multi-label tag classification

Emotion/DEAM regression is intentionally removed because the current
real-data pipeline does not use DEAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bert_encoder import MusicBertEncoder
from src.gnn_model import MusicGNNEncoder


class CrossAttentionFusion(nn.Module):
    """
    Graph representation attends to BERT token representations.

    Query  : GNN representation
    Key/Value: BERT token sequence
    """

    def __init__(
        self,
        g_dim: int = 128,
        text_dim: int = 768,
        proj_dim: int = 128,
    ):
        super().__init__()

        self.g_dim = g_dim
        self.text_dim = text_dim
        self.proj_dim = proj_dim

        self.W_Q = nn.Linear(g_dim, proj_dim, bias=False)
        self.W_K = nn.Linear(text_dim, proj_dim, bias=False)
        self.W_V = nn.Linear(text_dim, proj_dim, bias=False)

        self.scale = proj_dim ** -0.5

    def forward(
        self,
        g: torch.Tensor,
        H_text: torch.Tensor,
        text_mask: torch.Tensor = None,
    ):
        # g:       (B, g_dim)
        # H_text:  (B, seq_len, text_dim)

        Q = self.W_Q(g).unsqueeze(1)
        K = self.W_K(H_text)
        V = self.W_V(H_text)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale

        if text_mask is not None:
            mask = text_mask.unsqueeze(1).bool()
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        att_weights = F.softmax(scores, dim=-1)

        context = torch.bmm(att_weights, V).squeeze(1)

        # [GNN representation ; attended text context]
        z = torch.cat([g, context], dim=-1)

        return z, att_weights


class GNNBertFusionModel(nn.Module):
    """
    Task 3 GNN-BERT fusion model.

    fusion_type:
        - "cross_attention"
        - "concat"
        - "bert_only"
        - "gnn_only"

    Main output:
        tag_logits
        tag_probs
        z
        g
        t_cls
        att_weights
    """

    def __init__(
        self,
        num_tags: int = 50,
        in_node_dim: int = 32,
        gnn_hidden_dim: int = 128,
        gnn_out_dim: int = 128,
        gnn_type: str = "graphsage",
        bert_model_name: str = "distilbert-base-uncased",
        fusion_type: str = "cross_attention",
        dropout: float = 0.2,
    ):
        super().__init__()

        self.fusion_type = fusion_type.lower()

        valid_types = {
            "cross_attention",
            "concat",
            "bert_only",
            "gnn_only",
        }

        if self.fusion_type not in valid_types:
            raise ValueError(
                f"Unknown fusion_type='{fusion_type}'. "
                f"Choose from {sorted(valid_types)}."
            )

        # -------------------------
        # GNN encoder
        # -------------------------
        self.gnn = MusicGNNEncoder(
            in_dim=in_node_dim,
            hidden_dim=gnn_hidden_dim,
            out_dim=gnn_out_dim,
            gnn_type=gnn_type,
            dropout=dropout,
        )

        # -------------------------
        # BERT encoder
        # -------------------------
        self.bert = MusicBertEncoder(
            model_name=bert_model_name
        )

        text_dim = self.bert.hidden_dim

        # -------------------------
        # Fusion
        # -------------------------
        if self.fusion_type == "cross_attention":
            self.cross_attn = CrossAttentionFusion(
                g_dim=gnn_out_dim,
                text_dim=text_dim,
                proj_dim=gnn_out_dim,
            )

            fused_dim = gnn_out_dim * 2

        elif self.fusion_type == "concat":
            self.cross_attn = None
            fused_dim = gnn_out_dim + text_dim

        elif self.fusion_type == "bert_only":
            self.cross_attn = None
            fused_dim = text_dim

        else:  # gnn_only
            self.cross_attn = None
            fused_dim = gnn_out_dim

        self.dropout = nn.Dropout(dropout)

        # -------------------------
        # Tag classification head
        # -------------------------
        self.tag_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_tags),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        batch: torch.Tensor = None,
        edge_weight: torch.Tensor = None,
    ):
        """
        Returns a dictionary containing multimodal representations
        and tag predictions.
        """

        # =========================================================
        # GNN
        # =========================================================
        _, g = self.gnn(
            x,
            edge_index,
            batch=batch,
            edge_weight=edge_weight,
        )

        # =========================================================
        # BERT
        # =========================================================
        H_text, t_cls = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )

        # =========================================================
        # Fusion / Ablation
        # =========================================================
        att_weights = None

        if self.fusion_type == "cross_attention":

            z, att_weights = self.cross_attn(
                g,
                H_text,
                text_mask=attention_mask,
            )

        elif self.fusion_type == "concat":

            z = torch.cat([g, t_cls], dim=-1)

        elif self.fusion_type == "bert_only":

            z = t_cls

        elif self.fusion_type == "gnn_only":

            z = g

        else:
            raise RuntimeError(
                f"Unsupported fusion type: {self.fusion_type}"
            )

        z = self.dropout(z)

        # =========================================================
        # Tag prediction
        # =========================================================
        tag_logits = self.tag_head(z)
        tag_probs = torch.sigmoid(tag_logits)

        return {
            "z": z,
            "g": g,
            "t_cls": t_cls,
            "tag_logits": tag_logits,
            "tag_probs": tag_probs,
            "att_weights": att_weights,
        }

    def compute_loss(
        self,
        tag_logits: torch.Tensor,
        tag_targets: torch.Tensor,
    ):
        """
        Multi-label binary cross-entropy loss.
        """

        loss = F.binary_cross_entropy_with_logits(
            tag_logits,
            tag_targets.float(),
        )

        return loss

    def compute_multi_task_loss(
        self,
        tag_logits: torch.Tensor,
        tag_targets: torch.Tensor,
        *args,
        **kwargs,
    ):
        """
        Backward-compatible wrapper.

        Emotion regression is no longer used, so only the tag
        classification loss is calculated.
        """

        loss = self.compute_loss(
            tag_logits,
            tag_targets,
        )

        loss_dict = {
            "loss_tags": loss.item(),
            "total_loss": loss.item(),
        }

        return loss, loss_dict