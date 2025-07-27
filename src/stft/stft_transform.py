# src/stft/stft_transform.py
import numpy as np
import librosa

def compute_stft(audio: np.ndarray, n_fft: int, hop: int, win: str) -> np.ndarray:
    """
    Compute STFT with given window/hop/window type.

    Args:
        audio (np.ndarray): Audio signal.
        n_fft (int): FFT size.
        hop (int): Hop length.
        win (str): Window type.

    Returns:
        np.ndarray: Complex STFT.

    Raises:
        ValueError: If parameters are invalid.
    """
    if audio.ndim != 1:
        raise ValueError("Input audio must be 1-D (mono)")
    return librosa.stft(audio, n_fft=n_fft, hop_length=hop, window=win)

def reconstruct_audio(S_mod: np.ndarray, hop: int, win: str) -> np.ndarray:
    """
    Inverse transform and overlap-add with librosa.istft.

    Args:
        S_mod (np.ndarray): Complex STFT.
        hop (int): Hop length.
        win (str): Window type.

    Returns:
        np.ndarray: Reconstructed audio signal.

    Raises:
        ValueError: If parameters are invalid.
    """
    return librosa.istft(S_mod, hop_length=hop, window=win) 