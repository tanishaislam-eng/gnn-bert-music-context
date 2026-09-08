# GNN-Based BERT for Understanding Context from Music

**Course:** CSE425 / EEE474 / CSE715 — Neural Networks  
**Prepared By:** Tanisha Islam  
**Institution:** BRAC University  
**Year:** 2026

---

## Project Overview

This project explores how audio structure and textual context can be combined to understand music using deep learning.

The system combines:

- BERT/DistilBERT for textual music context
- Graph Neural Networks (GNNs) for structural audio information
- CNNs for mel-spectrogram audio classification
- Multimodal GNN-BERT fusion
- Contrastive learning for audio-text alignment and retrieval

The experiments use real music data from **MagnaTagATune (MTT)** and **MusicCaps**.

---

## Datasets

### MagnaTagATune (MTT)

MagnaTagATune is used as the main multi-label music-tagging dataset.

For this project, a manageable subset was used:

- Training: 800 clips
- Validation: 100 clips
- Test: 100 clips
- Total: 1,000 clips
- Tag vocabulary: 50 tags

The official split information was used when preparing the subset.

MTT provides the audio and tag supervision used for the main classification experiments.

### MusicCaps

MusicCaps is used for genuine audio-text alignment.

- Total processed audio-caption pairs: 188
- Contrastive training: 150 pairs
- Validation: 18 pairs
- Test: 20 pairs

Only genuine MusicCaps audio-caption pairs are used for the cross-modal retrieval experiment.

---

## Audio Preprocessing

Audio is processed at **22,050 Hz**.

The preprocessing pipeline extracts:

- 128-bin log-mel spectrograms
- 12-dimensional chroma features
- 20-dimensional MFCC features

For graph construction, chroma and MFCC features are combined to create **32-dimensional node features**.

Audio is divided into segments, where each segment becomes a graph node.

Graph edges are constructed using:

1. Temporal adjacency between neighboring segments
2. Acoustic similarity using cosine similarity

This produces a structural representation of each music clip.

---

## Models

### 1. CNN Mel-Spectrogram Baseline

A convolutional neural network processes the 128-bin log-mel spectrogram and predicts the 50 music tags.

### 2. BERT-Only Baseline

A pretrained **DistilBERT (`distilbert-base-uncased`)** model encodes available textual music context.

The BERT backbone is used for semantic representation, followed by a multi-label tag classifier.

### 3. GNN-Only Model

The music structure graph is processed using a Graph Neural Network.

The implementation supports:

- GraphSAGE
- Graph Attention Network (GAT)

Graph-level representations are used for music-tag prediction.

### 4. Early Concatenation Fusion

The GNN audio representation and BERT text representation are concatenated before classification.

### 5. GNN-BERT Cross-Attention Fusion

The multimodal model combines graph-based audio representations with BERT token representations using cross-attention.

This allows structural audio information to interact with textual context before tag prediction.

### 6. Contrastive Audio-Text Model

A dual-encoder architecture is trained using InfoNCE contrastive loss.

- GNN encodes audio graphs
- BERT encodes MusicCaps captions
- Both representations are projected into a shared embedding space

The model is evaluated using audio-text retrieval.

---

## Experimental Results

The following results were obtained from the final experiment.

| Model | Macro-F1 | AUC-PR | Retrieval R@5 |
|---|---:|---:|---:|
| Random Tags | 0.0575 | 0.0957 | - |
| Majority Tags | 0.0000 | 0.0637 | - |
| CNN Mel-Spectrogram | 0.0239 | 0.2337 | - |
| BERT-Only | 0.0000 | 0.1024 | - |
| GNN-Only | 0.0091 | 0.1907 | - |
| Early Concatenation | 0.0000 | 0.0887 | - |
| GNN-BERT Cross-Attention | 0.0000 | 0.0935 | - |
| Contrastive Zero-Shot | 0.0408 | 0.0634 | 0.3000 |

### MusicCaps Retrieval

The contrastive model achieved:

- R@1: **0.0500**
- R@5: **0.3000**
- R@10: **0.4500**

The relatively low Macro-F1 values show the difficulty of balanced prediction across the imbalanced 50-tag label set.

The CNN obtained an AUC-PR of **0.2337**, indicating that it retained useful ranking ability even though its fixed-threshold Macro-F1 was low.

The retrieval experiment also demonstrates partial cross-modal alignment, with the correct audio appearing within the top 10 results for a portion of the MusicCaps queries.

---

## Evaluation Outputs

Experimental outputs are stored in:

```text
results/
├── metrics.json
├── plots/
│   ├── f1_loss_curves.png
│   ├── tsne_multimodal.png
│   └── cross_attention_heatmap.png
└── retrieval_examples/
    ├── retrieval_samples.json
    └── retrieval_samples.md
