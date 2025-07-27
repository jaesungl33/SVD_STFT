import numpy as np
import soundfile as sf
import librosa
import os
import sys
from audioio.audio_io import load_audio, save_audio, preprocess
from stft.svd_stft import compute_stft, embed_svd_stft, extract_svd_stft
from stft.stft_transform import reconstruct_audio

def run_robustness_test(input_path, alpha, block_size, num_bits, sr=16000, n_fft=1024, hop_length=256, window="hann", key=42):
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_watermarked{ext}"
    # 1. Load and preprocess audio
    audio = load_audio(input_path, sr)
    audio = preprocess(audio)
    # 2. Compute STFT
    S = compute_stft(audio, sr, n_fft, hop_length, window)
    # 3. Embed watermark
    bits = np.random.randint(0, 2, size=num_bits).tolist()
    S_wm = embed_svd_stft(S, bits, alpha, block_size, key)
    # 4. Reconstruct and save watermarked audio
    audio_wm = reconstruct_audio(S_wm, hop_length, window)
    save_audio(audio_wm, sr, output_path)
    # 5. Apply distortion (resample down and up)
    audio_dist = librosa.resample(audio_wm, orig_sr=sr, target_sr=sr//2)
    audio_dist = librosa.resample(audio_dist, orig_sr=sr//2, target_sr=sr)
    # 6. Extract watermark from distorted audio
    S2 = compute_stft(audio_dist, sr, n_fft, hop_length, window)
    bits_rec = extract_svd_stft(S2, alpha, block_size, key, num_bits=num_bits)
    # 7. Compute BER
    ber = sum(b1 != b2 for b1, b2 in zip(bits, bits_rec)) / num_bits
    return ber, bits, bits_rec, output_path

def main():
    """
    Parameter sweep for SVD-STFT watermarking robustness.
    Tries different alpha, block_size, and num_bits, and prints BER for each.
    """
    if len(sys.argv) < 2:
        print("Usage: python robustness_test.py <input_audio>")
        sys.exit(1)
    input_path = sys.argv[1]
    alphas = [0.01, 0.05, 0.1, 0.2]
    block_sizes = [(8, 8), (16, 16), (32, 32)]
    num_bits_list = [4, 8, 16]
    best_ber = 1.0
    best_cfg = None
    print("alpha\tblock_size\tnum_bits\tBER")
    for alpha in alphas:
        for block_size in block_sizes:
            for num_bits in num_bits_list:
                try:
                    ber, bits, bits_rec, output_path = run_robustness_test(input_path, alpha, block_size, num_bits)
                    print(f"{alpha}\t{block_size}\t{num_bits}\t{ber:.3f}")
                    if ber < best_ber:
                        best_ber = ber
                        best_cfg = (alpha, block_size, num_bits, bits, bits_rec, output_path)
                except Exception as e:
                    print(f"{alpha}\t{block_size}\t{num_bits}\tERROR: {e}")
    if best_cfg:
        alpha, block_size, num_bits, bits, bits_rec, output_path = best_cfg
        print("\nBest configuration:")
        print(f"alpha={alpha}, block_size={block_size}, num_bits={num_bits}")
        print(f"Original bits:  {bits}")
        print(f"Extracted bits: {bits_rec}")
        print(f"Bit Error Rate (BER): {best_ber:.3f}")
        print(f"Watermarked audio saved as: {output_path}")
    else:
        print("No successful configuration found.")

if __name__ == "__main__":
    main() 