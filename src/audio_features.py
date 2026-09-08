"""
Audio Feature Extraction and Segmentation Pipeline
Strictly handles real audio files: 22,050 Hz resampling, log-mel spectrograms (128 bins),
chroma features (12 bins), MFCCs (20 bins), and segment-level feature extraction.
Zero synthetic audio or mock data generation.
"""

import os
import math
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

try:
    import soundfile as sf
except ImportError:
    sf = None


def load_and_resample_audio(
    file_path: str,
    target_sr: int = 22050,
    duration: float = 30.0,
) -> np.ndarray:
    """
    Loads real audio file from disk, converts to mono if necessary,
    and resamples to target_sr (default 22,050 Hz).
    Raises FileNotFoundError or RuntimeError if the real file is invalid or missing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Real audio file does not exist: {file_path}")

    if librosa is not None:
        try:
            y, sr = librosa.load(file_path, sr=target_sr, duration=duration, mono=True)
            return y.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {file_path} using librosa: {e}")
    elif sf is not None:
        try:
            y, sr = sf.read(file_path)
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            # Resample if needed
            if sr != target_sr and librosa is not None:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            if duration is not None:
                max_samples = int(duration * target_sr)
                y = y[:max_samples]
            return y.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {file_path} using soundfile: {e}")
    else:
        raise ImportError("librosa or soundfile is strictly required to process real audio files.")


def extract_log_mel_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
) -> np.ndarray:
    """
    Computes normalized 128-bin log-mel spectrogram (n_mels x T) from real audio waveform.
    """
    if librosa is None:
        raise ImportError("librosa is required for log-mel spectrogram extraction.")

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize per track to [0, 1] range
    min_val, max_val = np.min(log_mel), np.max(log_mel)
    if max_val - min_val > 1e-5:
        norm_mel = (log_mel - min_val) / (max_val - min_val)
    else:
        norm_mel = np.zeros_like(log_mel)

    return norm_mel.astype(np.float32)


def extract_chroma_features(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_chroma: int = 12,
) -> np.ndarray:
    """
    Computes 12-dimensional pitch chroma representation over time from real audio.
    """
    if librosa is None:
        raise ImportError("librosa is required for chroma extraction.")

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_chroma=n_chroma)
    col_sum = np.sum(chroma, axis=0, keepdims=True) + 1e-6
    chroma = chroma / col_sum
    return chroma.astype(np.float32)


def extract_mfcc_features(
    y: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 20,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Computes 20-dimensional MFCC features for timbral texture representation from real audio.
    """
    if librosa is None:
        raise ImportError("librosa is required for MFCC extraction.")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfcc.astype(np.float32)


def extract_segment_features(
    y: np.ndarray,
    sr: int = 22050,
    segment_duration: float = 3.0,
    n_chroma: int = 12,
    n_mfcc: int = 20,
) -> list[np.ndarray]:
    """
    Segments real audio into fixed time windows and computes node feature vectors
    h_i^(0) = [mean_chroma (12), mean_mfcc (20)] -> 32 dimensions per node.
    """
    segment_samples = int(segment_duration * sr)
    total_samples = len(y)

    if total_samples < segment_samples:
        segment_duration = max(1.0, total_samples / (sr * 3))
        segment_samples = int(segment_duration * sr)

    num_segments = max(1, math.ceil(total_samples / segment_samples))

    segment_features = []
    for s in range(num_segments):
        start = s * segment_samples
        end = min(start + segment_samples, total_samples)
        chunk = y[start:end]
        if len(chunk) < segment_samples // 4:
            continue

        chroma = extract_chroma_features(chunk, sr=sr, n_chroma=n_chroma)
        mfcc = extract_mfcc_features(chunk, sr=sr, n_mfcc=n_mfcc)

        chroma_mean = np.mean(chroma, axis=1)  # 12
        mfcc_mean = np.mean(mfcc, axis=1)      # 20

        # Feature vector for segment node i: dimension = 12 + 20 = 32
        node_feat = np.concatenate([chroma_mean, mfcc_mean]).astype(np.float32)

        # Normalize node feature vector
        norm = np.linalg.norm(node_feat)
        if norm > 1e-6:
            node_feat = node_feat / norm

        segment_features.append(node_feat)

    if not segment_features:
        raise ValueError(f"Audio signal of length {total_samples} was insufficient to extract segment features.")

    return segment_features
