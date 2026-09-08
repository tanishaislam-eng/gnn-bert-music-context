"""
Baseline 2 (B2): 2D-CNN Baseline on Mel-Spectrogram.

Uses real log-mel spectrograms from the prepared dataset.

Input:
    (B, 128, 128)
    or
    (B, 1, 128, 128)

Output:
    50 multi-label music-tag predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MelSpectrogramCNN(nn.Module):
    """
    4-block 2D CNN baseline.

    Architecture:
        Conv -> BN -> ReLU -> MaxPool
        Conv -> BN -> ReLU -> MaxPool
        Conv -> BN -> ReLU -> MaxPool
        Conv -> BN -> ReLU -> MaxPool
        Global Average Pooling
        Fully Connected classifier

    The CNN operates only on the audio mel-spectrogram.
    It does not use graph or text information.
    """

    def __init__(
        self,
        num_tags: int = 50,
        in_channels: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_tags = num_tags

        # -----------------------------------------------------
        # Block 1
        # -----------------------------------------------------
        self.conv1 = nn.Conv2d(
            in_channels,
            32,
            kernel_size=3,
            padding=1,
        )

        self.bn1 = nn.BatchNorm2d(32)

        # -----------------------------------------------------
        # Block 2
        # -----------------------------------------------------
        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1,
        )

        self.bn2 = nn.BatchNorm2d(64)

        # -----------------------------------------------------
        # Block 3
        # -----------------------------------------------------
        self.conv3 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            padding=1,
        )

        self.bn3 = nn.BatchNorm2d(128)

        # -----------------------------------------------------
        # Block 4
        # -----------------------------------------------------
        self.conv4 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            padding=1,
        )

        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # Final multi-label classifier.
        self.fc = nn.Linear(
            128,
            num_tags,
        )

    def forward(
        self,
        mel_spec: torch.Tensor,
    ):
        """
        Forward pass.

        Accepted input:
            (B, 128, T)
            (B, 1, 128, T)

        Returns:
            logits
            probabilities
            embedding
        """

        # -----------------------------------------------------
        # Add channel dimension if necessary
        # -----------------------------------------------------
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)

        if mel_spec.dim() != 4:
            raise ValueError(
                "mel_spec must have shape "
                "(B, 128, T) or (B, 1, 128, T). "
                f"Received shape: {tuple(mel_spec.shape)}"
            )

        # -----------------------------------------------------
        # CNN Block 1
        # -----------------------------------------------------
        x = self.conv1(mel_spec)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # -----------------------------------------------------
        # CNN Block 2
        # -----------------------------------------------------
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # -----------------------------------------------------
        # CNN Block 3
        # -----------------------------------------------------
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # -----------------------------------------------------
        # CNN Block 4
        # -----------------------------------------------------
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)

        # -----------------------------------------------------
        # Global average pooling
        # -----------------------------------------------------
        embedding = self.global_pool(
            x
        ).flatten(1)

        # Shape:
        # (B, 128)

        dropped = self.dropout(
            embedding
        )

        logits = self.fc(
            dropped
        )

        probabilities = torch.sigmoid(
            logits
        )

        return {
            "logits": logits,
            "probs": probabilities,
            "embedding": embedding,
        }

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-label binary cross-entropy loss.
        """

        return F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
        )