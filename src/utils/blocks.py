# src/utils/blocks.py
import numpy as np
from typing import Tuple, List
import random

def split_into_blocks(S_mag: np.ndarray, block_shape: Tuple[int, int]) -> List[np.ndarray]:
    """Split a 2D array into non-overlapping blocks.

    Args:
        S_mag (np.ndarray): 2D magnitude array.
        block_shape (Tuple[int, int]): Block dimensions.

    Returns:
        List[np.ndarray]: List of blocks.
    """
    rows, cols = S_mag.shape
    block_rows, block_cols = block_shape
    blocks = []
    for i in range(0, rows - rows % block_rows, block_rows):
        for j in range(0, cols - cols % block_cols, block_cols):
            block = S_mag[i:i+block_rows, j:j+block_cols]
            blocks.append(block.copy())
    return blocks

def reassemble_blocks(blocks: List[np.ndarray], shape: Tuple[int, int], block_shape: Tuple[int, int]) -> np.ndarray:
    """Reassemble blocks into a 2D array of given shape.

    Args:
        blocks (List[np.ndarray]): List of blocks.
        shape (Tuple[int, int]): Target array shape.
        block_shape (Tuple[int, int]): Block dimensions.

    Returns:
        np.ndarray: Reassembled 2D array.
    """
    rows, cols = shape
    block_rows, block_cols = block_shape
    S_mag = np.zeros((rows, cols), dtype=blocks[0].dtype)
    idx = 0
    for i in range(0, rows - rows % block_rows, block_rows):
        for j in range(0, cols - cols % block_cols, block_cols):
            S_mag[i:i+block_rows, j:j+block_cols] = blocks[idx]
            idx += 1
    return S_mag

def pseudo_permutation(n: int, key: int) -> List[int]:
    """Generate a pseudorandom permutation of n elements using a key.

    Args:
        n (int): Number of elements.
        key (int): Secret key.

    Returns:
        List[int]: Permuted indices.
    """
    rng = random.Random(key)
    indices = list(range(n))
    rng.shuffle(indices)
    return indices 