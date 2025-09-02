import numpy as np
import pytest
from stft.svd_stft import embed_svd_stft

def test_embed_watermark():
    # Test embedding functionality
    S = np.random.randn(64, 64) + 1j * np.random.randn(64, 64)
    bits = [1, 0, 1, 0]
    alpha = 0.01
    block_size = (8, 8)
    key = 42
    
    S_wm, sigma_ref = embed_svd_stft(S, bits, alpha, block_size, key)
    
    # Check that the watermarked STFT has the same shape
    assert S_wm.shape == S.shape
    # Check that sigma_ref is returned
    assert len(sigma_ref) == len(bits)
    # Check that the watermarked STFT is different from original
    assert not np.allclose(S, S_wm) 