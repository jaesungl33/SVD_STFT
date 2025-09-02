import numpy as np
import pytest
from stft.svd_stft import compute_stft, embed_svd_stft, extract_svd_stft
from stft.stft_transform import reconstruct_audio

def test_svd_stft_embed_extract():
    sr = 16000
    n_fft = 256  # Larger FFT for more blocks
    hop_length = 64
    audio = np.random.randn(sr * 2)  # 2 seconds of audio
    S = compute_stft(audio, sr, n_fft, hop_length, 'hann')
    n_blocks = (S.shape[0] // 8) * (S.shape[1] // 8)
    bits = np.random.randint(0, 2, size=min(8, n_blocks)).tolist()
    
    # Use a larger alpha for more reliable embedding
    alpha = 0.05
    S_wm, sigma_ref = embed_svd_stft(S, bits, alpha, (8, 8), key=123)
    S2 = S_wm.copy()
    
    # Use reference extraction for better reliability
    bits_rec = extract_svd_stft(S2, alpha, (8, 8), 123, num_bits=len(bits), sigma_ref=sigma_ref)
    ber = sum(b1 != b2 for b1, b2 in zip(bits, bits_rec)) / len(bits)
    
    # With reference extraction and larger alpha, BER should be very low
    assert ber <= 0.1  # Should be much better than random
    # Check that we get the right number of bits
    assert len(bits_rec) == len(bits) 