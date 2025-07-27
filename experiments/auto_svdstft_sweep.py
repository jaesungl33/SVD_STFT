import numpy as np
import soundfile as sf
import librosa
import os
import sys
from glob import glob
from audioio.audio_io import load_audio, save_audio, preprocess
from stft.svd_stft import compute_stft, embed_svd_stft, extract_svd_stft
from stft.stft_transform import reconstruct_audio

def run_robustness_test(input_path, alpha, block_size, num_bits, sr=16000, n_fft=1024, hop_length=256, window="hann", key=42):
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_watermarked{ext}"
    audio = load_audio(input_path, sr)
    audio = preprocess(audio)
    S = compute_stft(audio, sr, n_fft, hop_length, window)
    bits = np.random.randint(0, 2, size=num_bits).tolist()
    S_wm = embed_svd_stft(S, bits, alpha, block_size, key)
    audio_wm = reconstruct_audio(S_wm, hop_length, window)
    save_audio(audio_wm, sr, output_path)
    audio_dist = librosa.resample(audio_wm, orig_sr=sr, target_sr=sr//2)
    audio_dist = librosa.resample(audio_dist, orig_sr=sr//2, target_sr=sr)
    S2 = compute_stft(audio_dist, sr, n_fft, hop_length, window)
    bits_rec = extract_svd_stft(S2, alpha, block_size, key, num_bits=num_bits)
    ber = sum(b1 != b2 for b1, b2 in zip(bits, bits_rec)) / num_bits
    return ber, bits, bits_rec, output_path

def main():
    """
    Batch parameter sweep for SVD-STFT watermarking robustness.
    Sweeps alpha, block_size, num_bits for all FLAC/WAV files in a directory.
    Prints best BER for each file and a summary table at the end.
    """
    if len(sys.argv) < 2:
        print("Usage: python auto_svdstft_sweep.py <audio_dir>")
        sys.exit(1)
    audio_dir = sys.argv[1]
    audio_files = sorted(glob(os.path.join(audio_dir, '**', '*.flac'), recursive=True) +
                         glob(os.path.join(audio_dir, '**', '*.wav'), recursive=True))
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        sys.exit(1)
    alphas = [0.01, 0.05, 0.1, 0.2]
    block_sizes = [(8, 8), (16, 16), (32, 32)]
    num_bits_list = [4, 8, 16]
    results = []
    print("file\tbest_BER\talpha\tblock_size\tnum_bits")
    for audio_path in audio_files:
        best_ber = 1.0
        best_cfg = None
        for alpha in alphas:
            for block_size in block_sizes:
                for num_bits in num_bits_list:
                    try:
                        ber, bits, bits_rec, output_path = run_robustness_test(audio_path, alpha, block_size, num_bits)
                        if ber < best_ber:
                            best_ber = ber
                            best_cfg = (alpha, block_size, num_bits, bits, bits_rec, output_path)
                    except Exception as e:
                        continue
        if best_cfg:
            alpha, block_size, num_bits, bits, bits_rec, output_path = best_cfg
            print(f"{os.path.basename(audio_path)}\t{best_ber:.3f}\t{alpha}\t{block_size}\t{num_bits}")
            results.append((audio_path, best_ber, alpha, block_size, num_bits, output_path))
        else:
            print(f"{os.path.basename(audio_path)}\tERROR\t-\t-\t-")
    print("\nSummary:")
    for audio_path, best_ber, alpha, block_size, num_bits, output_path in results:
        print(f"{audio_path}: best_BER={best_ber:.3f}, alpha={alpha}, block_size={block_size}, num_bits={num_bits}, watermarked={output_path}")

if __name__ == "__main__":
    main() 