import numpy as np
import pytest
from stft.svd_stft import compute_stft, embed_svd_stft, extract_svd_stft
from stft.stft_transform import reconstruct_audio

def test_svd_stft_embed_extract():
    sr = 16000
    n_fft = 128
    hop_length = 32
    audio = np.random.randn(sr * 2)
    S = compute_stft(audio, sr, n_fft, hop_length, 'hann')
    n_blocks = (S.shape[0] // 8) * (S.shape[1] // 8)
    bits = np.random.randint(0, 2, size=min(8, n_blocks)).tolist()
    S_wm = embed_svd_stft(S, bits, 0.01, (8, 8), key=123)
    S2 = S_wm.copy()
    bits_rec = extract_svd_stft(S2, 0.01, (8, 8), 123, num_bits=len(bits))
    ber = sum(b1 != b2 for b1, b2 in zip(bits, bits_rec)) / len(bits)
    assert ber == 0.0 