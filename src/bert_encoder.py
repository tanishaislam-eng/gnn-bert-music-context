"""
BERT Contextual Text Encoder and Task 1 Multi-Label Tag Classifier

Uses a genuine pretrained DistilBERT/BERT model from Hugging Face.
No mock/fallback transformer is used.

Outputs:
- H_text: contextual token representations
- cls_repr: pooled first-token representation
- multi-label tag predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer


class MusicBertEncoder(nn.Module):
    """
    Genuine pretrained BERT/DistilBERT encoder.

    Input:
        Natural-language music metadata or MusicCaps captions.

    Output:
        H_text:
            (batch_size, sequence_length, hidden_dim)

        cls_repr:
            (batch_size, hidden_dim)
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        freeze_backbone: bool = True,
        hidden_dim: int = 768,
    ):
        super().__init__()

        self.model_name = model_name

        print(
            f"Loading pretrained transformer: {model_name}"
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name
            )

            self.backbone = AutoModel.from_pretrained(
                model_name,
                attn_implementation="eager",
        )

        except Exception as e:
            raise RuntimeError(
                "\nFailed to load the genuine pretrained "
                f"transformer '{model_name}'.\n\n"
                "The project requires a real pretrained "
                "Hugging Face model.\n"
                "No mock/fallback encoder is allowed.\n\n"
                f"Original error:\n{e}"
            ) from e

        self.hidden_dim = (
            self.backbone.config.hidden_size
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            print(
                "✓ DistilBERT backbone frozen."
            )
        else:
            print(
                "✓ DistilBERT backbone will be fine-tuned."
            )

        print(
            f"✓ Transformer hidden dimension: "
            f"{self.hidden_dim}"
        )

    def tokenize(
        self,
        texts: list[str],
        max_length: int = 128,
        device: torch.device = None,
    ):
        """
        Tokenize natural-language text using the
        genuine Hugging Face tokenizer.
        """

        if not isinstance(texts, list):
            texts = list(texts)

        texts = [
            "" if text is None else str(text)
            for text in texts
        ]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if device is not None:
            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

        return encoded

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        """
        Forward pass through genuine pretrained BERT.

        Returns:
            H_text:
                (batch, sequence_length, hidden_dim)

            cls_repr:
                (batch, hidden_dim)
        """

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        H_text = outputs.last_hidden_state

        # DistilBERT does not have a separate pooled_output.
        # The first token representation is therefore used
        # as the sequence-level representation.
        cls_repr = H_text[:, 0, :]

        return H_text, cls_repr


class BertTagClassifier(nn.Module):
    """
    Task 1: BERT baseline for multi-label music-tag
    classification.

    t = BERT_CLS(X_text)

    y_hat = sigmoid(Wt + b)
    """

    def __init__(
        self,
        num_tags: int = 50,
        model_name: str = "distilbert-base-uncased",
        freeze_backbone: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = MusicBertEncoder(
            model_name=model_name,
            freeze_backbone=freeze_backbone,
        )

        hidden_dim = self.encoder.hidden_dim

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_tags,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        H_text, cls_repr = self.encoder(
            input_ids,
            attention_mask,
        )

        dropped = self.dropout(
            cls_repr
        )

        logits = self.classifier(
            dropped
        )

        probs = torch.sigmoid(
            logits
        )

        return {
            "logits": logits,
            "probs": probs,
            "cls_repr": cls_repr,
            "H_text": H_text,
        }

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Multi-label binary cross-entropy loss.
        """

        targets = targets.float()

        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
        )