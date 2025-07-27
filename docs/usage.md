# Usage

## Embedding a Watermark
```python
from src.io.audio_io import load_audio, save_audio
from src.stft.stft_transform import compute_stft, reconstruct_audio
from src.embed.embedder import embed_watermark

# Load and preprocess audio
x = load_audio('input.wav', sr=16000)

# Compute STFT
S = compute_stft(x, n_fft=1024, hop=256, win='hann')

# Embed watermark
bits = [0, 1, 1, 0, ...]  # your bitstream
S_wm = embed_watermark(S, bits, alpha=0.01, block_shape=(8,8), key=42)

# Reconstruct and save
x_wm = reconstruct_audio(S_wm, hop=256, win='hann')
save_audio(x_wm, sr=16000, out_path='watermarked.wav')
```

## Extracting a Watermark
```python
from src.extract.extractor import extract_watermark

bits_extracted = extract_watermark(S_wm, alpha=0.01, block_shape=(8,8), key=42, num_bits=len(bits))
```

## Running Tests
```bash
pytest tests/
``` 

## SVD-STFT Watermarking Example
```python
import soundfile as sf
from src.stft.svd_stft import compute_stft, embed_svd_stft, extract_svd_stft
from src.stft.stft_transform import reconstruct_audio

audio, sr = sf.read('in.wav')
S = compute_stft(audio, sr, n_fft=1024, hop_length=256, window='hann')
S_wm = embed_svd_stft(S, bits=[1,0,1,1], alpha=0.02, block_size=(8,8), key=42)
audio_wm = reconstruct_audio(S_wm, hop_length=256, window='hann')
sf.write('watermarked.wav', audio_wm, sr)
# … later …
S2 = compute_stft(audio_wm, sr, 1024, 256, 'hann')
bits_rec = extract_svd_stft(S2, 0.02, (8,8), 42, num_bits=4)
``` 