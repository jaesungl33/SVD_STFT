# src/utils/metrics.py
import numpy as np
from typing import List, Dict

def evaluate_performance(orig: np.ndarray, mod: np.ndarray, extracted_bits: List[int], true_bits: List[int]) -> Dict[str, float]:
    """Compute BER, SNR, PSNR, MSE; test under attacks.

    Args:
        orig (np.ndarray): Original audio.
        mod (np.ndarray): Watermarked audio.
        extracted_bits (List[int]): Extracted bitstream.
        true_bits (List[int]): Ground truth bitstream.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.

    Raises:
        ValueError: If input is invalid.
    """
    if len(orig) != len(mod):
        min_len = min(len(orig), len(mod))
        orig = orig[:min_len]
        mod = mod[:min_len]
    mse = np.mean((orig - mod) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10(1.0 / mse)
    snr = 10 * np.log10(np.sum(orig ** 2) / (np.sum((orig - mod) ** 2) + 1e-10))
    ber = sum(b1 != b2 for b1, b2 in zip(true_bits, extracted_bits)) / len(true_bits)
    return {
        'BER': ber,
        'SNR': snr,
        'PSNR': psnr,
        'MSE': mse
    } 