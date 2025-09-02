import numpy as np
import pytest
from stft.svd_stft import embed_svd_stft, extract_svd_stft

def test_extract_watermark():
    # Test extraction functionality
    S = np.random.randn(64, 64) + 1j * np.random.randn(64, 64)
    bits = [1, 0, 1, 0]
    alpha = 0.01
    block_size = (8, 8)
    key = 42
    
    # Embed watermark
    S_wm, sigma_ref = embed_svd_stft(S, bits, alpha, block_size, key)
    
    # Extract watermark
    extracted_bits = extract_svd_stft(S_wm, alpha, block_size, key, len(bits))
    
    # Check that we get the same number of bits
    assert len(extracted_bits) == len(bits)
    # Check that all bits are 0 or 1
    assert all(bit in [0, 1] for bit in extracted_bits) 