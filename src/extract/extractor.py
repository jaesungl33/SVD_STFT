# src/extract/extractor.py
import numpy as np
from typing import List, Tuple, Optional

def extract_watermark(S_mod: np.ndarray, alpha: float, block_shape: Tuple[int, int], key: int, num_bits: int, threshold: Optional[float] = None) -> List[int]:
    """Recover bits; if blind, use sign of (Σ′–Σ_ref) or threshold on Σ′ changes.

    Args:
        S_mod (np.ndarray): Watermarked complex STFT.
        alpha (float): Embedding strength.
        block_shape (Tuple[int, int]): Block dimensions.
        key (int): Secret key for block permutation.
        num_bits (int): Number of bits to extract.
        threshold (float, optional): Extraction threshold.

    Returns:
        List[int]: Extracted bitstream.

    Raises:
        ValueError: If parameters are invalid.
    """
    S_mag_mod = np.abs(S_mod)
    shape = S_mag_mod.shape
    block_size = block_shape
    rows, cols = shape
    block_rows, block_cols = block_size
    n_blocks = (rows // block_rows) * (cols // block_cols)
    if num_bits > n_blocks:
        raise ValueError("num_bits exceeds number of available blocks.")
    # Split into blocks
    blocks = []
    for i in range(0, rows - rows % block_rows, block_rows):
        for j in range(0, cols - cols % block_cols, block_cols):
            block = S_mag_mod[i:i+block_rows, j:j+block_cols]
            blocks.append(block.copy())
    # Permute blocks
    rng = np.random.RandomState(key)
    perm = rng.permutation(len(blocks))
    sigmas = []
    for idx in perm[:num_bits]:
        block = blocks[idx]
        U, Sigma, Vh = np.linalg.svd(block, full_matrices=False)
        sigmas.append(Sigma[0])
    if threshold is None:
        threshold = np.median(sigmas)
    extracted = [1 if s > threshold else 0 for s in sigmas]
    return extracted 