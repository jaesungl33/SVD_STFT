# src/audioio/audio_io.py
import numpy as np
import soundfile as sf
import librosa

def load_audio(path: str, sr: int) -> np.ndarray:
    """
    Read a WAV file, convert to mono, and resample to target rate.
    Args:
        path (str): File path to audio file.
        sr (int): Target sample rate.
    Returns:
        np.ndarray: Audio signal (mono, shape (n,)).
    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If audio cannot be loaded or resampled.
    """
    audio, file_sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    return audio

def save_audio(audio: np.ndarray, sr: int, out_path: str) -> None:
    """
    Export audio as WAV.
    Args:
        audio (np.ndarray): Audio signal.
        sr (int): Sample rate.
        out_path (str): Output file path.
    Raises:
        IOError: If file cannot be written.
    """
    sf.write(out_path, audio, sr)

def preprocess(audio: np.ndarray) -> np.ndarray:
    """
    Normalize to [-1, 1].
    Args:
        audio (np.ndarray): Input audio signal.
    Returns:
        np.ndarray: Preprocessed audio.
    Raises:
        ValueError: If input is invalid.
    """
    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio 