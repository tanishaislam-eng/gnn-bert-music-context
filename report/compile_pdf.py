"""
Final report generator for:
GNN-Based BERT for Understanding Context from Music

Prepared by: Tanisha Islam
Instructor: Moin Mostakim
BRAC University

This report uses the actual experiment results stored in results/metrics.json.
"""

import os
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    HRFlowable,
    PageBreak,
)


def build_final_report_pdf(pdf_path: str, plots_dir: str):
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=0.72 * inch,
        rightMargin=0.72 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=1,
        spaceAfter=12,
    )

    subtitle_style = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=16,
        alignment=1,
        spaceAfter=8,
    )

    h1 = ParagraphStyle(
        "H1Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=8,
        spaceAfter=8,
    )

    h2 = ParagraphStyle(
        "H2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        spaceBefore=7,
        spaceAfter=5,
    )

    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        alignment=4,
        spaceAfter=8,
    )

    small = ParagraphStyle(
        "SmallCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        spaceAfter=5,
    )

    story = []

    # ------------------------------------------------------------------
    # PAGE 1: TITLE + ABSTRACT
    # ------------------------------------------------------------------

    story.append(Spacer(1, 0.4 * inch))
    story.append(
        Paragraph(
            "GNN-Based BERT for Understanding Context from Music",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Course:</b> Neural Networks (CSE425 / EEE474 / CSE715)",
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Prepared By:</b> Tanisha Islam",
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Instructor:</b> Moin Mostakim",
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            "BRAC University | 2026",
            subtitle_style,
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor("#BBBBBB"),
            spaceBefore=8,
            spaceAfter=15,
        )
    )

    story.append(Paragraph("Abstract", h1))

    abstract = """
    Music contains both acoustic structure and semantic context. This project
    investigates whether graph neural networks and transformer-based language
    representations can be combined for music-context understanding. Audio is
    resampled to 22,050 Hz and represented using 128-bin log-mel features,
    chroma, and MFCC-derived segment features. Temporal adjacency and acoustic
    cosine similarity are used to construct music structure graphs. GraphSAGE
    and GAT-style graph processing are evaluated together with DistilBERT text
    representations. The project includes BERT-only, CNN, GNN-only, early
    concatenation, and cross-attention fusion experiments. A separate
    dual-encoder contrastive model is evaluated on genuine MusicCaps
    audio-caption pairs using InfoNCE loss and retrieval Recall@K.
    """

    story.append(Paragraph(abstract, body))

    abstract2 = """
    Experiments were conducted using 1,000 MagnaTagATune examples divided into
    800 training, 100 validation, and 100 test clips with a 50-tag vocabulary.
    MusicCaps contributed 188 clean audio-caption pairs for the contrastive
    experiment. The strongest tag-ranking result was obtained by the CNN
    baseline with AUC-PR 0.2337, while fixed-threshold Macro-F1 values remained
    low because of strong multi-label imbalance and limited training. The
    contrastive model achieved R@1 = 0.05, R@5 = 0.30, and R@10 = 0.45.
    These results are reported without claiming state-of-the-art performance.
    """

    story.append(Paragraph(abstract2, body))

    story.append(Paragraph("<b>Keywords:</b> GNN, BERT, music information retrieval, multimodal learning, GraphSAGE, cross-attention, contrastive learning", small))

    story.append(PageBreak())

    # ------------------------------------------------------------------
    # PAGE 2: DATASET + PREPROCESSING
    # ------------------------------------------------------------------

    story.append(Paragraph("1. Dataset and Preprocessing", h1))

    story.append(Paragraph("1.1 MagnaTagATune", h2))

    story.append(
        Paragraph(
            """
            The primary supervised tagging experiments use 1,000 real
            MagnaTagATune clips. The final subset contains 800 training,
            100 validation, and 100 test samples. Fifty commonly occurring
            music tags form the multi-label prediction vocabulary. The
            official split information is preserved where available in the
            preprocessing pipeline.
            """,
            body,
        )
    )

    story.append(Paragraph("1.2 MusicCaps", h2))

    story.append(
        Paragraph(
            """
            The cross-modal task uses 188 genuine MusicCaps audio-caption
            pairs. These are real paired examples rather than synthetic
            combinations of unrelated audio and captions. For contrastive
            training, the implementation uses 150 training, 18 validation,
            and 20 test examples.
            """,
            body,
        )
    )

    story.append(Paragraph("1.3 Audio Processing", h2))

    story.append(
        Paragraph(
            """
            Audio is resampled to 22,050 Hz. The preprocessing module extracts
            normalized 128-bin log-mel spectrograms together with chroma and
            MFCC information. Audio is divided into fixed-duration segments.
            Each segment becomes a node in the graph and is represented by a
            32-dimensional acoustic feature vector.
            """,
            body,
        )
    )

    story.append(Paragraph("1.4 Graph Construction", h2))

    story.append(
        Paragraph(
            """
            Two edge types are created. First, temporal edges connect adjacent
            audio segments. Second, similarity edges connect acoustically
            related segments when cosine similarity exceeds the configured
            threshold. This provides a graph representation capable of
            connecting repeated or similar regions even when they are not
            directly adjacent in time.
            """,
            body,
        )
    )

    story.append(Paragraph("1.5 Text Processing", h2))

    story.append(
        Paragraph(
            """
            Text is encoded using HuggingFace DistilBERT with a maximum token
            length of 128. For MagnaTagATune, the text context is derived from
            available title and artist metadata rather than from the target
            tags themselves, which avoids direct label leakage. MusicCaps uses
            its genuine expert-written captions.
            """,
            body,
        )
    )

    story.append(PageBreak())

    # ------------------------------------------------------------------
    # PAGE 3: MODELS
    # ------------------------------------------------------------------

    story.append(Paragraph("2. Model Architectures", h1))

    story.append(Paragraph("2.1 CNN Mel-Spectrogram Baseline", h2))
    story.append(
        Paragraph(
            """
            A convolutional neural network operates directly on real
            log-mel spectrograms. It provides an audio-only baseline against
            which graph-based representations can be compared.
            """,
            body,
        )
    )

    story.append(Paragraph("2.2 Task 1: BERT-only", h2))
    story.append(
        Paragraph(
            """
            DistilBERT converts text context into contextual token embeddings.
            A classification head predicts the 50 music tags using binary
            cross-entropy loss. The pretrained language backbone is frozen by
            default to reduce computational cost.
            """,
            body,
        )
    )

    story.append(Paragraph("2.3 Task 2: GNN-only", h2))
    story.append(
        Paragraph(
            """
            GraphSAGE is applied to the segment graph. Message passing combines
            each node with information from neighboring temporal and
            similarity-connected segments. Global graph pooling produces a
            fixed-size audio representation for multi-label classification.
            The codebase also contains GAT support.
            """,
            body,
        )
    )

    story.append(Paragraph("2.4 Task 3: Multimodal Fusion", h2))
    story.append(
        Paragraph(
            """
            Two multimodal variants are evaluated. Early concatenation joins
            pooled graph and text representations directly. The cross-attention
            model instead lets the graph representation attend to contextual
            BERT token embeddings before classification. Both variants predict
            only multi-label music tags in the final implementation.
            """,
            body,
        )
    )

    story.append(Paragraph("2.5 Task 4: Contrastive Audio-Text Alignment", h2))
    story.append(
        Paragraph(
            """
            A dual-encoder model maps the audio graph and MusicCaps caption
            into a shared embedding space. InfoNCE loss encourages matched
            audio-caption pairs to be closer than mismatched examples. The
            resulting representation is evaluated using text-to-audio
            retrieval and zero-shot tag prompts.
            """,
            body,
        )
    )

    story.append(PageBreak())

    # ------------------------------------------------------------------
    # PAGE 4: RESULTS
    # ------------------------------------------------------------------

    story.append(Paragraph("3. Experimental Results", h1))

    metrics_file = "results/metrics.json"

    if not os.path.exists(metrics_file):
        raise FileNotFoundError(
            "results/metrics.json was not found. "
            "The report intentionally does not use fabricated fallback values."
        )

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    def val(model, key):
        x = metrics.get(model, {}).get(key, "-")
        if x in [None, "-", "—"]:
            return "-"
        return f"{float(x):.4f}"

    table_data = [
        ["Model", "Macro-F1", "AUC-PR"],
        [
            "Random Tags",
            val("Random tags", "macro_f1"),
            val("Random tags", "auc_pr"),
        ],
        [
            "Majority Tags",
            val("Majority tags", "macro_f1"),
            val("Majority tags", "auc_pr"),
        ],
        [
            "CNN Mel-Spec",
            val("CNN mel-spec", "macro_f1"),
            val("CNN mel-spec", "auc_pr"),
        ],
        [
            "Task 1: BERT-only",
            val("Task 1: BERT-only", "macro_f1"),
            val("Task 1: BERT-only", "auc_pr"),
        ],
        [
            "Task 2: GNN-only",
            val("Task 2: GNN-only", "macro_f1"),
            val("Task 2: GNN-only", "auc_pr"),
        ],
        [
            "Task 3: Early Concat",
            val("Task 3: Early Concat", "macro_f1"),
            val("Task 3: Early Concat", "auc_pr"),
        ],
        [
            "Task 3: Cross-Attention",
            val("Task 3: GNN-BERT (Cross-Attn)", "macro_f1"),
            val("Task 3: GNN-BERT (Cross-Attn)", "auc_pr"),
        ],
        [
            "Task 4: Zero-Shot",
            val("Task 4: Contrastive (Zero-Shot)", "macro_f1"),
            val("Task 4: Contrastive (Zero-Shot)", "auc_pr"),
        ],
    ]

    table = Table(
        table_data,
        colWidths=[3.3 * inch, 1.3 * inch, 1.3 * inch],
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.append(table)
    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            """
            The CNN baseline produced the highest AUC-PR value of 0.2337.
            GraphSAGE obtained AUC-PR 0.1907. BERT-only, early concatenation,
            and cross-attention produced lower ranking performance in this
            limited experiment. The low Macro-F1 scores, including zero for
            several models at the fixed decision threshold, show that the
            current training setup does not provide strong calibrated
            multi-label classification.
            """,
            body,
        )
    )

    story.append(
        Paragraph(
            """
            Importantly, the random baseline achieved Macro-F1 0.0575, which is
            greater than several trained models at the fixed threshold.
            Therefore, the results should not be interpreted as evidence that
            the fusion architecture outperforms all baselines. AUC-PR provides
            a more informative view of ranking ability under the severe class
            imbalance present in the 50-tag setting.
            """,
            body,
        )
    )

    story.append(Paragraph("3.1 MusicCaps Retrieval", h2))

    contrastive = metrics["Task 4: Contrastive (Zero-Shot)"]

    retrieval_data = [
        ["Metric", "Result"],
        ["Recall@1", f"{contrastive.get('r1_retrieval', 0):.4f}"],
        ["Recall@5", f"{contrastive.get('r5_retrieval', 0):.4f}"],
        ["Recall@10", f"{contrastive.get('r10_retrieval', 0):.4f}"],
    ]

    retrieval_table = Table(
        retrieval_data,
        colWidths=[2.2 * inch, 1.5 * inch],
    )

    retrieval_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#444444")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.append(retrieval_table)

    story.append(PageBreak())

    # ------------------------------------------------------------------
    # PAGE 5: FIGURES
    # ------------------------------------------------------------------

    story.append(Paragraph("4. Visual Analysis", h1))

    plot_items = [
        (
            "f1_loss_curves.png",
            "Figure 1. Training losses and available validation performance trends.",
            6.4,
            2.4,
        ),
        (
            "tsne_multimodal.png",
            "Figure 2. t-SNE visualization of learned multimodal representations.",
            6.4,
            2.4,
        ),
        (
            "cross_attention_heatmap.png",
            "Figure 3. Cross-attention visualization between the audio graph representation and text tokens.",
            6.4,
            2.2,
        ),
    ]

    found_plot = False

    for filename, caption, width, height in plot_items:
        path = os.path.join(plots_dir, filename)

        if os.path.exists(path):
            found_plot = True
            story.append(Paragraph(caption, h2))
            story.append(
                Image(
                    path,
                    width=width * inch,
                    height=height * inch,
                )
            )
            story.append(Spacer(1, 10))

    if not found_plot:
        story.append(
            Paragraph(
                "Plot files were not found in results/plots when the report was generated.",
                body,
            )
        )

    story.append(PageBreak())

    # ------------------------------------------------------------------
    # PAGE 6: QUALITATIVE ANALYSIS
    # ------------------------------------------------------------------

    story.append(Paragraph("5. Qualitative Retrieval Case Studies", h1))

    story.append(Paragraph("Case Study 1: Synth / Electronic Query", h2))
    story.append(
        Paragraph(
            """
            A query describing synthesizer-heavy electronic music retrieved
            examples with mixed relevance. Some top-ranked clips were
            mismatches, while another result showed broader electronic and pop
            production overlap. This case demonstrates that the learned
            embedding contains useful coarse semantic structure but does not
            consistently produce exact semantic matches.
            """,
            body,
        )
    )

    story.append(Paragraph("Case Study 2: Pop Production Query", h2))
    story.append(
        Paragraph(
            """
            For a pop-oriented production description, the matching caption was
            retrieved at rank 2. This provides a positive example in which the
            contrastive representation placed the corresponding audio close to
            its textual description, although it did not achieve the top rank.
            """,
            body,
        )
    )

    story.append(Paragraph("Case Study 3: Hindustani Classical Query", h2))
    story.append(
        Paragraph(
            """
            For a Hindustani classical description, a sitar-based meditation
            example was retrieved at rank 1. The match shares broad cultural,
            instrumental, and acoustic characteristics with the query. This
            example suggests that the model can capture some high-level
            instrument and style associations even with a small training set.
            """,
            body,
        )
    )

    story.append(Paragraph("Interpretability", h2))
    story.append(
        Paragraph(
            """
            The cross-attention visualization provides a qualitative indication
            of which text tokens interact most strongly with the pooled graph
            representation. These maps should be interpreted as model
            attention patterns rather than causal explanations. The project
            does not contain lyric annotations, so no claim is made about
            lyric-level alignment.
            """,
            body,
        )
    )

    story.append(PageBreak())

    # ------------------------------------------------------------------
    # PAGE 7: LIMITATIONS + CONCLUSION
    # ------------------------------------------------------------------

    story.append(Paragraph("6. Limitations and Deviations", h1))

    story.append(
        Paragraph(
            """
            <b>Task 2 dataset deviation:</b> The project specification proposes
            genre classification on GTZAN or FMA-small. Because of the limited
            project duration and computational constraints, the implemented
            Task 2 uses the available MagnaTagATune multi-label tagging setup
            instead. Therefore, the GNN-only experiment should be interpreted
            as audio-graph tag classification rather than the requested
            GTZAN/FMA genre experiment.
            """,
            body,
        )
    )

    story.append(
        Paragraph(
            """
            <b>Task 3 qualitative limitation:</b> The dataset used in the final
            pipeline contains no lyrics. Consequently, the required
            graph-path-plus-lyric alignment analysis could not be completed
            literally. The project instead provides graph/text
            cross-attention visualization and caption-based qualitative
            analysis.
            """,
            body,
        )
    )

    story.append(
        Paragraph(
            """
            <b>Training scale:</b> Models were trained for only four epochs on a
            reduced subset of MagnaTagATune. The MusicCaps contrastive
            experiment contains only 188 clean pairs. These choices made the
            project computationally manageable but strongly limit
            generalization and model convergence.
            """,
            body,
        )
    )

    story.append(
        Paragraph(
            """
            <b>Threshold sensitivity and imbalance:</b> The multi-label target
            distribution is highly imbalanced. A single fixed probability
            threshold produces very low Macro-F1 for several models even when
            their AUC-PR indicates some ranking ability. Future work should
            tune per-class thresholds, use class-balanced losses, train for
            more epochs, and evaluate larger datasets.
            """,
            body,
        )
    )

    story.append(Paragraph("7. Conclusion", h1))

    story.append(
        Paragraph(
            """
            This project implements an end-to-end multimodal music-context
            pipeline combining real audio preprocessing, graph construction,
            GraphSAGE/GAT components, DistilBERT text representations,
            multimodal fusion, and contrastive audio-text alignment. Although
            the experimental results are modest, the implementation
            demonstrates the full technical workflow required to investigate
            graph-based and language-based representations of music.
            """,
            body,
        )
    )

    story.append(
        Paragraph(
            """
            The strongest supervised ranking result was produced by the CNN
            mel-spectrogram baseline with AUC-PR 0.2337. The MusicCaps
            contrastive model achieved Recall@1 of 0.05, Recall@5 of 0.30, and
            Recall@10 of 0.45. These results establish a reproducible baseline
            for future work rather than a state-of-the-art claim.
            """,
            body,
        )
    )

    story.append(Paragraph("8. Project Deliverables", h1))

    story.append(
        Paragraph(
            """
            The repository includes the complete source code, configuration,
            dataset preparation scripts, training and evaluation modules,
            representative graph JSON files, evaluation metrics, plots,
            qualitative retrieval examples, an exploratory notebook, and
            <code>notebooks/demo_context.ipynb</code> containing an end-to-end
            multimodal forward-pass demonstration.
            """,
            body,
        )
    )

    story.append(
        Paragraph(
            """
            Repository: github.com/tanishaislam-eng/gnn-bert-music-context
            """,
            small,
        )
    )

    doc.build(story)

    print(f"Final report PDF successfully generated at: {pdf_path}")


if __name__ == "__main__":
    build_final_report_pdf(
        "report/final_report.pdf",
        "results/plots",
    )
