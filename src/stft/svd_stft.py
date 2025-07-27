import numpy as np
from typing import Sequence, Tuple, Optional, List, Callable
import librosa
from utils.blocks import split_into_blocks, reassemble_blocks, pseudo_permutation
from bitarray import bitarray
from bitarray.util import ba2int, int2ba

def hamming_encode(bits: List[int]) -> List[int]:
    """Encode bits using (7,4) Hamming code."""
    # Pad bits to multiple of 4
    pad = (4 - len(bits) % 4) % 4
    bits = bits + [0] * pad
    encoded = []
    for i in range(0, len(bits), 4):
        d = bits[i:i+4]
        # Generator matrix for (7,4) Hamming code
        G = np.array([[1,1,0,1], [1,0,1,1], [1,0,0,0], [0,1,1,1], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
        c = (G @ np.array(d)) % 2
        encoded.extend(c.tolist())
    return encoded

def hamming_decode(bits: List[int]) -> List[int]:
    """Decode bits using (7,4) Hamming code (single error correction)."""
    decoded = []
    H = np.array([[1,0,1,0,1,0,1], [0,1,1,0,0,1,1], [0,0,0,1,1,1,1]])
    for i in range(0, len(bits), 7):
        c = np.array(bits[i:i+7])
        if len(c) < 7:
            break
        s = (H @ c) % 2
        syndrome = int(''.join(str(x) for x in s), 2)
        if syndrome != 0 and syndrome <= 7:
            c[syndrome-1] ^= 1  # Correct single error
        # Extract data bits (positions 2,4,5,6)
        d = [c[2], c[4], c[5], c[6]]
        decoded.extend(d)
    return decoded

def compute_stft(audio: np.ndarray, sr: int, n_fft: int, hop_length: int, window: str) -> np.ndarray:
    """
    Compute the complex STFT of a 1-D audio signal.
    Args:
        audio: mono audio samples (shape: (n_samples,))
        sr: sampling rate (Hz)
        n_fft: FFT window size
        hop_length: number of samples between successive frames
        window: window type (e.g. 'hann')
    Returns:
        Complex spectrogram (shape: (n_fft//2+1, n_frames))
    Raises:
        ValueError: if audio is not 1-D or parameters are invalid.
    """
    if audio.ndim != 1:
        raise ValueError("Input audio must be 1-D (mono)")
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)
    return S

def embed_svd_stft(
    S_complex: np.ndarray,
    bits: Sequence[int],
    alpha: float,
    block_size: Tuple[int, int],
    key: int
) -> Tuple[np.ndarray, Optional[List[float]]]:
    """
    Embed a binary watermark into the magnitude of an STFT via SVD.
    Args:
        S_complex: input complex STFT (n_bins × n_frames)
        bits: list of 0/1 watermark bits
        alpha: embedding strength coefficient
        block_size: (rows, cols) of each block to SVD
        key: integer seed for pseudorandom block selection
    Returns:
        Modified complex STFT with embedded watermark.
    Raises:
        ValueError: if payload > #blocks or invalid block_size.
    """
    S_mag = np.abs(S_complex)
    S_phase = np.angle(S_complex)
    shape = S_mag.shape
    blocks = split_into_blocks(S_mag, block_size)
    n_blocks = len(blocks)
    if len(bits) > n_blocks:
        raise ValueError("Payload too large for number of blocks.")
    perm = pseudo_permutation(n_blocks, key)
    sigma_ref = []
    for i, bit in enumerate(bits):
        idx = perm[i]
        block = blocks[idx]
        U, Sigma, Vh = np.linalg.svd(block, full_matrices=False)
        sigma_ref.append(Sigma[0])
        # Additive embedding on the largest singular value
        Sigma[0] = Sigma[0] + alpha * (2 * bit - 1)
        block_mod = U @ np.diag(Sigma) @ Vh
        blocks[idx] = block_mod
    S_mag_mod = reassemble_blocks(blocks, shape, block_size)
    S_complex_mod = S_mag_mod * np.exp(1j * S_phase)
    return S_complex_mod, sigma_ref

def extract_svd_stft(
    S_complex_mod: np.ndarray,
    alpha: float,
    block_size: Tuple[int, int],
    key: int,
    num_bits: int,
    threshold: Optional[float] = None,
    sigma_ref: Optional[List[float]] = None
) -> List[int]:
    """
    Extract a watermark bitstream from a watermarked STFT.
    Args:
        S_complex_mod: watermarked complex STFT
        alpha: embedding strength used during embed
        block_size: same block dims as embed
        key: seed for pseudorandom block order
        num_bits: number of bits to recover
        threshold: decision threshold (if None, use blind median-based)
    Returns:
        List of extracted bits (0/1).
    Raises:
        ValueError: if num_bits > #blocks.
    """
    S_mag_mod = np.abs(S_complex_mod)
    shape = S_mag_mod.shape
    blocks_mod = split_into_blocks(S_mag_mod, block_size)
    n_blocks = len(blocks_mod)
    if num_bits > n_blocks:
        raise ValueError("num_bits exceeds number of available blocks.")
    perm = pseudo_permutation(n_blocks, key)
    sigmas = []
    for idx in perm[:num_bits]:
        block = blocks_mod[idx]
        U, Sigma, Vh = np.linalg.svd(block, full_matrices=False)
        sigmas.append(Sigma[0])
    if sigma_ref is not None:
        # Reference (non-blind) extraction
        extracted = [1 if s > r else 0 for s, r in zip(sigmas, sigma_ref)]
    else:
        if threshold is None:
            threshold = np.median(sigmas)
        extracted = [1 if s > threshold else 0 for s in sigmas]
    return extracted

def snr_metric(original: np.ndarray, watermarked: np.ndarray) -> float:
    """Compute SNR in dB."""
    noise = original - watermarked
    return 10 * np.log10(np.sum(original ** 2) / (np.sum(noise ** 2) + 1e-10))

def calibrate_parameters(
    audio: np.ndarray,
    sr: int,
    bit_pattern: Sequence[int],
    alpha_candidates: Sequence[float],
    block_sizes: Sequence[Tuple[int, int]],
    metric_fn: Callable[[float, float], float],
    pilot_len: int = 5 * 16000,
    key: int = 42
) -> Tuple[float, Tuple[int, int], float]:
    """
    Calibrate (alpha, block_size, threshold) for SVD-STFT watermarking.

    Args:
        audio (np.ndarray): Input audio signal.
        sr (int): Sample rate.
        bit_pattern (Sequence[int]): Pilot bits for calibration.
        alpha_candidates (Sequence[float]): List of alpha values to try.
        block_sizes (Sequence[Tuple[int, int]]): List of block sizes to try.
        metric_fn (Callable): Function combining SNR and BER into a score.
        pilot_len (int): Length of pilot segment (samples).
        key (int): Secret key for block permutation.

    Returns:
        Tuple[float, Tuple[int, int], float]: (best_alpha, best_block_size, best_threshold)
    """
    best_score = -np.inf
    best_params = None
    pilot_audio = audio[:pilot_len]
    for alpha in alpha_candidates:
        for block_size in block_sizes:
            try:
                S = compute_stft(pilot_audio, sr, n_fft=1024, hop_length=256, window='hann')
                S_wm, sigma_ref = embed_svd_stft(S, bit_pattern, alpha, block_size, key)
                from stft.stft_transform import reconstruct_audio  # avoid circular import
                # Use positional arguments for reconstruct_audio
                audio_wm = reconstruct_audio(S_wm, 256, 'hann')
                S2 = compute_stft(audio_wm, sr, n_fft=1024, hop_length=256, window='hann')
                # Use reference threshold (median of all Sigma[0])
                S_mag_mod = np.abs(S2)
                blocks_mod = split_into_blocks(S_mag_mod, block_size)
                sigmas = []
                for idx in range(len(bit_pattern)):
                    block = blocks_mod[idx]
                    U, Sigma, Vh = np.linalg.svd(block, full_matrices=False)
                    sigmas.append(Sigma[0])
                threshold = np.median(sigmas)
                bits_rec = extract_svd_stft(S2, alpha, block_size, key, num_bits=len(bit_pattern), threshold=threshold, sigma_ref=sigma_ref)
                # Fix SNR calculation to use overlapping region
                min_len = min(len(pilot_audio), len(audio_wm))
                snr = snr_metric(pilot_audio[:min_len], audio_wm[:min_len])
                ber = sum(b1 != b2 for b1, b2 in zip(bit_pattern, bits_rec)) / len(bit_pattern)
                score = metric_fn(snr, ber)
                if score > best_score:
                    best_score = score
                    best_params = (alpha, block_size, threshold)
            except Exception as e:
                print(f"Calibration failed for alpha={alpha}, block_size={block_size}: {e}")
                continue
    if best_params is None:
        raise RuntimeError("Calibration failed for all parameter combinations.")
    return best_params 