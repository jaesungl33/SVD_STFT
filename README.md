# VoiceMarkWatermark

A Python 3.10 module for robust audio watermarking using SVD-STFT.

## Purpose
Embed and extract binary watermarks in audio by modifying singular values of the STFT, supporting research in robust, imperceptible watermarking.

## Features
- SVD-based watermark embedding and extraction
- Blockwise STFT manipulation
- Robustness evaluation (BER, SNR, etc.)
- Modular, testable, and extensible codebase

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run example embedding/extraction (see `docs/usage.md`)

## Project Structure
- `src/` — Core modules (io, stft, embed, extract, utils)
- `tests/` — Unit tests (pytest)
- `docs/` — Architecture and usage docs
- `.github/` — CI workflows

See `docs/architecture.md` for details.