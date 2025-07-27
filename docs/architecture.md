# Architecture

## Overview
VoiceMarkWatermark implements robust audio watermarking using SVD on STFT blocks. The pipeline consists of:

1. **Audio I/O & Preprocessing**: Load, normalize, and optionally filter audio.
2. **STFT**: Compute time-frequency representation.
3. **Embedding**: Modify singular values of STFT magnitude blocks.
4. **ISTFT**: Reconstruct watermarked audio.
5. **Extraction**: Recover watermark bits from modified STFT.
6. **Evaluation**: Compute BER, SNR, PSNR, MSE, and robustness.

## Module Structure
- `io/`: Audio loading, saving, preprocessing
- `stft/`: STFT and ISTFT transforms
- `embed/`: Watermark embedding
- `extract/`: Watermark extraction
- `utils/`: Metrics, block operations

See `usage.md` for function-level details. 

# SVD-STFT Watermarking

## Function Signatures
- `compute_stft(audio, sr, n_fft, hop_length, window)`
- `embed_svd_stft(S_complex, bits, alpha, block_size, key)`
- `extract_svd_stft(S_complex_mod, alpha, block_size, key, num_bits, threshold=None)`

## High-Level Flow
1. Compute STFT of audio.
2. Embed watermark bits into STFT magnitude using SVD on blocks (with pseudorandom block assignment).
3. Reconstruct watermarked audio from modified STFT.
4. Extract watermark by inspecting singular values of blocks in the watermarked STFT.

See `docs/usage.md` for code examples. 