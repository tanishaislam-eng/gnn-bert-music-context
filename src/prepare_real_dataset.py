"""
Real Dataset Ingestion Pipeline for MagnaTagATune and MusicCaps

Datasets:
- MagnaTagATune: real MP3 audio + real annotations + official train/val/test splits
- MusicCaps: real WAV audio + genuine human expert captions

No synthetic audio.
No mock data.
No heuristic emotion scores.
No target-tag leakage into BERT text.

Final setup:
- MagnaTagATune: 800 train / 100 validation / 100 test
- MusicCaps: all available genuine pairs, split later for contrastive learning
- 50-tag MagnaTagATune vocabulary
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.audio_features import (
    load_and_resample_audio,
    extract_log_mel_spectrogram,
    extract_segment_features,
)
from src.graph_builder import build_segment_graph, save_graph_sample


# ============================================================
# PATHS
# ============================================================

BASE_DATA_DIR = Path("/Users/tanishaislam/MusicCaps")

MTT_AUDIO_DIR = BASE_DATA_DIR / "MagnaTagATune" / "audio"
MTT_SPLITS_FILE = BASE_DATA_DIR / "MagnaTagATune" / "splits.csv"
MTT_ANNO_FILE = BASE_DATA_DIR / "annotations_final.csv"
MTT_CLIP_INFO = BASE_DATA_DIR / "clip_info_final.csv"

MUSICCAPS_AUDIO_DIR = BASE_DATA_DIR / "audio"
MUSICCAPS_META_FILE = BASE_DATA_DIR / "musiccaps_clean.csv"


# ============================================================
# TOP 50 MTT TAGS
# These are taken from the MagnaTagATune annotation vocabulary.
# ============================================================

TOP_TAGS = [
    "rock",
    "pop",
    "alternative",
    "indie",
    "electronic",
    "dance",
    "metal",
    "punk",
    "jazz",
    "blues",
    "classical",
    "folk",
    "country",
    "hip-hop",
    "rap",
    "ambient",
    "techno",
    "house",
    "trance",
    "instrumental",
    "guitar",
    "acoustic",
    "piano",
    "strings",
    "violin",
    "cello",
    "drums",
    "beat",
    "bass",
    "synth",
    "vocal",
    "female",
    "male",
    "fast",
    "slow",
    "melodic",
    "happy",
    "sad",
    "dark",
    "calm",
    "energetic",
    "soft",
    "loud",
    "live",
    "remix",
    "soundtrack",
    "movie",
    "christmas",
    "indian",
    "world",
]


# ============================================================
# HELPERS
# ============================================================

def safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clean_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def build_tag_indices(tags):
    return [
        TOP_TAGS.index(tag)
        for tag in tags
        if tag in TOP_TAGS
    ]


def process_audio_to_features(
    audio_path,
    mel_output_path,
    graph_output_path,
    duration,
    segment_duration,
):
    """
    Process one real audio file into:
    - log-mel spectrogram
    - segment-level graph
    """

    y = load_and_resample_audio(
        str(audio_path),
        target_sr=22050,
        duration=duration,
    )

    mel = extract_log_mel_spectrogram(
        y,
        sr=22050,
        n_mels=128,
    )

    np.save(mel_output_path, mel)

    seg_feats = extract_segment_features(
        y,
        sr=22050,
        segment_duration=segment_duration,
    )

    graph_dict = build_segment_graph(
        seg_feats,
        similarity_threshold=0.55,
    )

    save_graph_sample(
        graph_dict,
        str(graph_output_path),
    )

    return graph_dict


# ============================================================
# MAIN
# ============================================================

def process_real_dataset(
    output_dir="data",
    mtt_train_limit=800,
    mtt_val_limit=100,
    mtt_test_limit=100,
    musiccaps_sample_limit=None,
):
    print("=" * 60)
    print("INGESTING REAL DATASETS")
    print("MagnaTagATune + MusicCaps")
    print("=" * 60)

    proc_dir = Path(output_dir) / "processed"
    splits_dir = Path(output_dir) / "splits"

    proc_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 1. READ MAGNATAGATUNE ANNOTATIONS
    # ========================================================

    print("\n[1/4] Reading MagnaTagATune annotations...")

    if not MTT_ANNO_FILE.exists():
        raise FileNotFoundError(
            f"Missing MagnaTagATune annotations: {MTT_ANNO_FILE}"
        )

    anno_df = pd.read_csv(
        MTT_ANNO_FILE,
        sep="\t",
    )

    print(f"  Annotation rows: {len(anno_df)}")

    # Make sure clip_id exists
    if "clip_id" not in anno_df.columns:
        raise ValueError(
            "annotations_final.csv does not contain 'clip_id'."
        )

    # Build annotation dictionary
    anno_dict = {}

    for _, row in anno_df.iterrows():
        clip_id = safe_int(row["clip_id"])

        if clip_id is None:
            continue

        tags_present = []

        for tag in TOP_TAGS:
            if tag in anno_df.columns:
                try:
                    if int(row[tag]) == 1:
                        tags_present.append(tag)
                except (TypeError, ValueError):
                    pass

        anno_dict[clip_id] = tags_present

    print(
        f"  ✓ Loaded annotations for "
        f"{len(anno_dict)} clips."
    )

    # ========================================================
    # 2. READ CLIP INFORMATION
    # ========================================================

    print("\n[2/4] Reading MagnaTagATune clip metadata...")

    clip_info_map = {}

    if MTT_CLIP_INFO.exists():

        # IMPORTANT:
        # clip_info_final.csv is comma-separated.
        info_df = pd.read_csv(
            MTT_CLIP_INFO,
            sep=",",
            on_bad_lines="skip",
        )

        print(
            f"  Clip metadata rows: {len(info_df)}"
        )

        for _, row in info_df.iterrows():

            clip_id = safe_int(row.get("clip_id"))

            if clip_id is None:
                continue

            title = clean_text(
                row.get("title", "")
            )

            artist = clean_text(
                row.get("artist", "")
            )

            original_url = clean_text(
                row.get("original_url", "")
            )

            clip_info_map[clip_id] = {
                "title": title,
                "artist": artist,
                "original_url": original_url,
            }

    else:
        print(
            "  Warning: clip_info_final.csv not found."
        )

    # ========================================================
    # 3. PROCESS MAGNATAGATUNE
    # ========================================================

    print("\n[3/4] Processing MagnaTagATune audio...")

    if not MTT_SPLITS_FILE.exists():
        raise FileNotFoundError(
            f"Missing split file: {MTT_SPLITS_FILE}"
        )

    if not MTT_AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Missing MTT audio directory: {MTT_AUDIO_DIR}"
        )

    splits_df = pd.read_csv(
        MTT_SPLITS_FILE
    )

    required_split_columns = {
        "clip_id",
        "split",
        "mp3_path",
    }

    missing = required_split_columns - set(
        splits_df.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns in splits.csv: {missing}"
        )

    train_rows = splits_df[
        splits_df["split"] == "train"
    ]

    val_rows = splits_df[
        splits_df["split"] == "val"
    ]

    test_rows = splits_df[
        splits_df["split"] == "test"
    ]

    print(f"  Available official train: {len(train_rows)}")
    print(f"  Available official val:   {len(val_rows)}")
    print(f"  Available official test:  {len(test_rows)}")

    # Use exact requested limits.
    train_rows = train_rows.head(
        int(mtt_train_limit)
    )

    val_rows = val_rows.head(
        int(mtt_val_limit)
    )

    test_rows = test_rows.head(
        int(mtt_test_limit)
    )

    selected_mtt = pd.concat(
        [
            train_rows,
            val_rows,
            test_rows,
        ],
        ignore_index=True,
    )

    print(
        f"  Selected: "
        f"{len(train_rows)} train / "
        f"{len(val_rows)} val / "
        f"{len(test_rows)} test"
    )

    mtt_catalog = []

    processed_count = 0

    for _, row in selected_mtt.iterrows():

        clip_id = safe_int(row["clip_id"])

        if clip_id is None:
            continue

        split = clean_text(row["split"])

        mp3_path = clean_text(
            row["mp3_path"]
        )

        mp3_name = os.path.basename(
            mp3_path
        )

        audio_file = MTT_AUDIO_DIR / mp3_name

        if not audio_file.exists():
            print(
                f"  Missing audio for clip {clip_id}: "
                f"{audio_file}"
            )
            continue

        track_id = f"mtt_{clip_id}"

        tags = anno_dict.get(
            clip_id,
            [],
        )

        tag_indices = build_tag_indices(
            tags
        )

        metadata = clip_info_map.get(
            clip_id,
            {},
        )

        title = clean_text(
            metadata.get("title", "")
        )

        artist = clean_text(
            metadata.get("artist", "")
        )

        original_url = clean_text(
            metadata.get("original_url", "")
        )

        # Do NOT put the target tags into the text input.
        # This prevents BERT label leakage.
        if title and artist:
            context_text = (
                f"Track title: {title}. "
                f"Artist: {artist}."
            )
        elif title:
            context_text = (
                f"Track title: {title}."
            )
        elif artist:
            context_text = (
                f"Artist: {artist}."
            )
        else:
            context_text = (
                f"MagnaTagATune music clip {clip_id}."
            )

        try:

            graph_dict = process_audio_to_features(
                audio_file,
                proc_dir / f"{track_id}_mel.npy",
                proc_dir / f"{track_id}_graph",
                duration=15.0,
                segment_duration=3.0,
            )

            mtt_catalog.append(
                {
                    "track_id": track_id,
                    "clip_id": clip_id,

                    "title": title
                    if title
                    else f"MTT Clip {clip_id}",

                    "artist": artist
                    if artist
                    else "Unknown Artist",

                    "artist_id": artist
                    if artist
                    else "unknown",

                    # MTT does not provide a reliable
                    # single genre field here.
                    "genre": "",

                    "tags": tags,
                    "tag_indices": tag_indices,

                    # Text is metadata only, not target tags.
                    "caption": context_text,

                    "split": split,

                    "num_nodes": graph_dict[
                        "num_nodes"
                    ],

                    "num_edges": graph_dict[
                        "num_edges"
                    ],

                    "dataset_source":
                        "MagnaTagATune",

                    "original_url":
                        original_url,
                }
            )

            processed_count += 1

            if processed_count % 25 == 0:
                print(
                    f"  Processed "
                    f"{processed_count}/"
                    f"{len(selected_mtt)} "
                    f"MTT clips..."
                )

        except Exception as e:

            print(
                f"  Warning on "
                f"{audio_file.name}: {e}"
            )

    print(
        f"✓ Completed "
        f"{processed_count} real MagnaTagATune tracks."
    )

    # ========================================================
    # 4. PROCESS MUSICCAPS
    # ========================================================

    print("\n[4/4] Processing MusicCaps...")

    if not MUSICCAPS_META_FILE.exists():
        raise FileNotFoundError(
            f"Missing MusicCaps metadata: "
            f"{MUSICCAPS_META_FILE}"
        )

    if not MUSICCAPS_AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Missing MusicCaps audio directory: "
            f"{MUSICCAPS_AUDIO_DIR}"
        )

    with open(
        MUSICCAPS_META_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)
        mc_rows = list(reader)

    # By default process ALL available genuine pairs.
    if musiccaps_sample_limit is not None:
        mc_rows = mc_rows[
            :int(musiccaps_sample_limit)
        ]

    print(
        f"  MusicCaps metadata rows: "
        f"{len(mc_rows)}"
    )

    musiccaps_catalog = []

    mc_processed = 0

    for i, row in enumerate(mc_rows):

        filename = clean_text(
            row.get("audio_file", "")
        )

        if not filename:
            filename = (
                f"mc_{i + 1:03d}.wav"
            )

        audio_path = (
            MUSICCAPS_AUDIO_DIR /
            filename
        )

        if not audio_path.exists():
            print(
                f"  Missing MusicCaps audio: "
                f"{audio_path}"
            )
            continue

        caption = clean_text(
            row.get("caption", "")
        )

        if not caption:
            print(
                f"  Skipping {filename}: "
                f"empty caption."
            )
            continue

        track_id = f"mc_{i + 1:03d}"

        try:

            graph_dict = process_audio_to_features(
                audio_path,
                proc_dir / f"{track_id}_mel.npy",
                proc_dir / f"{track_id}_graph",
                duration=10.0,
                segment_duration=2.5,
            )

            # MusicCaps is used for genuine
            # audio-caption retrieval.
            #
            # We intentionally DO NOT infer tags,
            # genres, artists, or emotions from
            # caption keywords.
            musiccaps_catalog.append(
                {
                    "track_id": track_id,

                    "title":
                        f"MusicCaps Clip #{i + 1}",

                    "artist": "",

                    "artist_id": "",

                    "genre": "",

                    "tags": [],

                    "tag_indices": [],

                    # Genuine expert caption.
                    "caption": caption,

                    # This is a source pair, not an
                    # artificial label.
                    "split": "musiccaps",

                    "num_nodes": graph_dict[
                        "num_nodes"
                    ],

                    "num_edges": graph_dict[
                        "num_edges"
                    ],

                    "dataset_source":
                        "MusicCaps",

                    "audio_file": filename,
                }
            )

            mc_processed += 1

            if mc_processed % 25 == 0:
                print(
                    f"  Processed "
                    f"{mc_processed}/"
                    f"{len(mc_rows)} "
                    f"MusicCaps clips..."
                )

        except Exception as e:

            print(
                f"  Warning on MusicCaps "
                f"{filename}: {e}"
            )

    print(
        f"✓ Completed "
        f"{mc_processed} genuine MusicCaps pairs."
    )

    # ========================================================
    # 5. SAVE DATA
    # ========================================================

    print("\nWriting processed metadata...")

    full_catalog = (
        mtt_catalog +
        musiccaps_catalog
    )

    with open(
        proc_dir / "metadata_catalog.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            full_catalog,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # MTT official splits
    # --------------------------------------------------------

    train_split = [
        item
        for item in mtt_catalog
        if item["split"] == "train"
    ]

    val_split = [
        item
        for item in mtt_catalog
        if item["split"] == "val"
    ]

    test_split = [
        item
        for item in mtt_catalog
        if item["split"] == "test"
    ]

    # --------------------------------------------------------
    # MusicCaps genuine pairs
    # --------------------------------------------------------

    musiccaps_pairs = musiccaps_catalog

    with open(
        splits_dir / "train.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            train_split,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        splits_dir / "val.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            val_split,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        splits_dir / "test.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            test_split,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Keep this filename because train.py uses it.
    with open(
        splits_dir / "musiccaps_test.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            musiccaps_pairs,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    with open(
        proc_dir / "vocab.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            TOP_TAGS,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("REAL DATASET INGESTION COMPLETED")
    print("=" * 60)

    print(
        f"  • Total processed tracks: "
        f"{len(full_catalog)}"
    )

    print(
        f"  • MTT Train: "
        f"{len(train_split)}"
    )

    print(
        f"  • MTT Validation: "
        f"{len(val_split)}"
    )

    print(
        f"  • MTT Test: "
        f"{len(test_split)}"
    )

    print(
        f"  • MusicCaps pairs: "
        f"{len(musiccaps_pairs)}"
    )

    print(
        f"  • Tag vocabulary: "
        f"{len(TOP_TAGS)} tags"
    )

    print(
        f"  • Processed directory: "
        f"{proc_dir}"
    )

    print(
        f"  • Split directory: "
        f"{splits_dir}"
    )

    print("=" * 60)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    process_real_dataset(
        output_dir="data",
        mtt_train_limit=800,
        mtt_val_limit=100,
        mtt_test_limit=100,
        musiccaps_sample_limit=None,
    )