# src/embed/embedder.py
import numpy as np
from typing import List, Tuple

def embed_watermark(S: np.ndarray, bits: List[int], alpha: float, block_shape: Tuple[int, int], key: int) -> np.ndarray:
    """Embed bits into blocks of S using SVD and secret key.

    Args:
        S (np.ndarray): Complex STFT.
        bits (List[int]): Bitstream to embed.
        alpha (float): Embedding strength.
        block_shape (Tuple[int, int]): Block dimensions.
        key (int): Secret key for block permutation.

    Returns:
        np.ndarray: Watermarked complex STFT.

    Raises:
        ValueError: If parameters are invalid.
    """
    S_mag = np.abs(S)
    S_phase = np.angle(S)
    shape = S_mag.shape
    block_size = block_shape
    rows, cols = shape
    block_rows, block_cols = block_size
    n_blocks = (rows // block_rows) * (cols // block_cols)
    if len(bits) > n_blocks:
        raise ValueError("Payload too large for number of blocks.")
    # Split into blocks
    blocks = []
    for i in range(0, rows - rows % block_rows, block_rows):
        for j in range(0, cols - cols % block_cols, block_cols):
            block = S_mag[i:i+block_rows, j:j+block_cols]
            blocks.append(block.copy())
    # Permute blocks
    rng = np.random.RandomState(key)
    perm = rng.permutation(len(blocks))
    for i, bit in enumerate(bits):
        idx = perm[i]
        block = blocks[idx]
        U, Sigma, Vh = np.linalg.svd(block, full_matrices=False)
        Sigma[0] = Sigma[0] + alpha * (2 * bit - 1)
        block_mod = U @ np.diag(Sigma) @ Vh
        blocks[idx] = block_mod
    # Reassemble
    S_mag_mod = np.zeros_like(S_mag)
    idx = 0
    for i in range(0, rows - rows % block_rows, block_rows):
        for j in range(0, cols - cols % block_cols, block_cols):
            S_mag_mod[i:i+block_rows, j:j+block_cols] = blocks[idx]
            idx += 1
    S_mod = S_mag_mod * np.exp(1j * S_phase)
    return S_mod 