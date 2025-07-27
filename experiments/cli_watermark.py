import argparse
import numpy as np
import os
import sys
import soundfile as sf
import json
from glob import glob
from audioio.audio_io import load_audio, save_audio, preprocess
from stft.svd_stft import (
    compute_stft, embed_svd_stft, extract_svd_stft, calibrate_parameters, hamming_encode, hamming_decode
)
from stft.stft_transform import reconstruct_audio
import matplotlib.pyplot as plt

WATERMARKED_DIR = "watermarked_audio"

EXAMPLE_USAGE = """
Examples:
  python cli_watermark.py --input in.wav --auto-tune --ecc --reference-extract
  python cli_watermark.py --input-dir myaudios/ --auto-tune --ecc --reference-extract --output-csv results.csv
  python cli_watermark.py --input in.wav --extract-only --alpha 0.01 --block-size 8x8 --key 42 --num-bits 16 --ecc --reference-extract --sigma-ref-file in_watermarked.wav.sigma_ref.json
"""

def ensure_output_dir():
    if not os.path.exists(WATERMARKED_DIR):
        os.makedirs(WATERMARKED_DIR)

def parse_block_size(s):
    if 'x' in s:
        parts = s.split('x')
    elif ',' in s:
        parts = s.split(',')
    else:
        raise ValueError(f"Invalid block size: {s}")
    return (int(parts[0]), int(parts[1]))

def parse_metric_fn(expr):
    def metric(snr, ber):
        SNR = snr
        BER = ber
        return eval(expr, {"SNR": SNR, "BER": BER})
    return metric

def plot_waveform(audio, sr, title="Waveform"):
    plt.figure(figsize=(10, 3))
    plt.plot(np.arange(len(audio)) / sr, audio)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def plot_spectrogram(audio, sr, title="Spectrogram"):
    import librosa.display
    S = np.abs(librosa.stft(audio))
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def process_file(
    input_path, output_path, mode, pilot_bits_length, alpha_candidates, block_sizes, metric_fn,
    alpha=None, block_size=None, threshold=None, key=42, sr=16000, n_fft=1024, hop_length=256, window='hann',
    log_results=None, ecc=False, reference_extract=False, extract_only=False, sigma_ref_file=None, num_bits=None,
    plot=False
):
    ensure_output_dir()
    try:
        if extract_only:
            print(f"[INFO] Extracting watermark from {input_path}")
            audio = load_audio(input_path, sr)
            audio = preprocess(audio)
            S = compute_stft(audio, sr, n_fft, hop_length, window)
            sigma_ref = None
            if reference_extract and sigma_ref_file:
                with open(sigma_ref_file, 'r') as f:
                    sigma_ref = json.load(f)
            if num_bits is None:
                print("--num-bits must be specified in extract-only mode.")
                sys.exit(1)
            bits_rec = extract_svd_stft(S, alpha, block_size, key, num_bits=num_bits, threshold=threshold, sigma_ref=sigma_ref)
            if ecc:
                bits_dec = hamming_decode(bits_rec)
            else:
                bits_dec = bits_rec
            print(f"Extracted bits: {bits_dec}")
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(''.join(str(b) for b in bits_dec))
            return
        # Normal embed+extract mode
        print(f"[INFO] Loading input: {input_path}")
        audio, file_sr = sf.read(input_path)
        if audio.ndim > 1:
            print("[WARN] Input is stereo; converting to mono.")
            audio = np.mean(audio, axis=1)
        if file_sr != sr:
            import librosa
            print(f"[WARN] Resampling from {file_sr} Hz to {sr} Hz.")
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        audio = preprocess(audio)
        bits = np.random.randint(0, 2, size=pilot_bits_length).tolist()
        if ecc:
            bits_enc = hamming_encode(bits)
        else:
            bits_enc = bits
        sigma_ref = None
        if mode == 'auto':
            pilot_bits = bits_enc
            alpha_opt, block_opt, thresh_opt = calibrate_parameters(
                audio, sr, pilot_bits, alpha_candidates, block_sizes, metric_fn, key=key
            )
            alpha, block_size, threshold = alpha_opt, block_opt, thresh_opt
            print(f"[Auto-tune] Using alpha={alpha}, block_size={block_size}, threshold={threshold:.4f}")
        elif mode == 'manual':
            if alpha is None or block_size is None:
                raise ValueError("Manual mode requires --alpha and --block-size.")
        S = compute_stft(audio, sr, n_fft, hop_length, window)
        S_wm, sigma_ref = embed_svd_stft(S, bits_enc, alpha, block_size, key)
        base_name = os.path.basename(input_path)
        out_base = os.path.splitext(base_name)[0] + '_watermarked.wav'
        output_path = os.path.join(WATERMARKED_DIR, out_base) if not output_path else output_path
        audio_wm = reconstruct_audio(S_wm, hop_length, window)
        save_audio(audio_wm, sr, output_path)
        print(f"[INFO] Watermarked audio saved to: {output_path}")
        if reference_extract:
            sigma_ref_path = output_path + '.sigma_ref.json'
            with open(sigma_ref_path, 'w') as f:
                json.dump([float(x) for x in sigma_ref], f)
        S2 = compute_stft(audio_wm, sr, n_fft, hop_length, window)
        if reference_extract:
            bits_rec = extract_svd_stft(S2, alpha, block_size, key, num_bits=len(bits_enc), sigma_ref=sigma_ref)
        else:
            bits_rec = extract_svd_stft(S2, alpha, block_size, key, num_bits=len(bits_enc), threshold=threshold)
        if ecc:
            bits_dec = hamming_decode(bits_rec)
            bits_dec = bits_dec[:len(bits)]
        else:
            bits_dec = bits_rec[:len(bits)]
        ber = sum(b1 != b2 for b1, b2 in zip(bits, bits_dec)) / len(bits)
        snr = 10 * np.log10(np.sum(audio[:len(audio_wm)] ** 2) / (np.sum((audio[:len(audio_wm)] - audio_wm) ** 2) + 1e-10))
        print(f"[RESULT] BER: {ber:.3f} | SNR: {snr:.2f} dB | Output: {output_path}")
        print(f"Embedded bits:  {bits}")
        print(f"Extracted bits: {bits_dec}")
        if plot:
            plot_waveform(audio, sr, title="Original Audio")
            plot_waveform(audio_wm, sr, title="Watermarked Audio")
            plot_spectrogram(audio, sr, title="Original Spectrogram")
            plot_spectrogram(audio_wm, sr, title="Watermarked Spectrogram")
        if log_results is not None:
            log_results.append({
                'file': input_path,
                'output': output_path,
                'alpha': alpha,
                'block_size': block_size,
                'threshold': threshold,
                'ber': ber,
                'snr': snr,
                'embedded_bits': ''.join(str(b) for b in bits),
                'extracted_bits': ''.join(str(b) for b in bits_dec),
                'ecc': ecc,
                'reference_extract': reference_extract
            })
        print("[INFO] Done!")
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}. Check your input file path.")
    except Exception as e:
        print(f"[ERROR] {e}")

def main_menu():
    while True:
        print("""
SVD-STFT Watermarking Menu:
1. Watermark a file
2. Extract watermark from a file
3. Batch process a directory
4. Visualize audio/watermark
5. Exit
""")
        choice = input("Select an option: ").strip()
        if choice == '1':
            input_path = input("Input audio file: ").strip()
            pilot_bits_length = int(input("Pilot bits length [default 64]: ") or 64)
            auto_tune = input("Auto-tune? (y/n): ").lower().startswith('y')
            ecc = input("Enable ECC? (y/n): ").lower().startswith('y')
            reference_extract = input("Reference extraction? (y/n): ").lower().startswith('y')
            plot = input("Plot waveform/spectrogram? (y/n): ").lower().startswith('y')
            process_file(
                input_path, None, 'auto' if auto_tune else 'manual', pilot_bits_length,
                [0.005, 0.01, 0.02], [(8,8), (16,16)], lambda snr, ber: 0.7*snr+0.3*(1-ber),
                ecc=ecc, reference_extract=reference_extract, plot=plot
            )
        elif choice == '2':
            input_path = input("Watermarked audio file: ").strip()
            alpha = float(input("Alpha: "))
            block_size = parse_block_size(input("Block size (e.g. 8x8): "))
            key = int(input("Secret key: "))
            num_bits = int(input("Number of bits to extract: "))
            ecc = input("ECC decoding? (y/n): ").lower().startswith('y')
            reference_extract = input("Reference extraction? (y/n): ").lower().startswith('y')
            sigma_ref_file = input("Sigma ref file (if reference extraction, else leave blank): ").strip() or None
            process_file(
                input_path, None, 'manual', 64, [0.01], [block_size], lambda snr, ber: 0, # metric unused
                alpha=alpha, block_size=block_size, key=key, num_bits=num_bits, ecc=ecc,
                reference_extract=reference_extract, extract_only=True, sigma_ref_file=sigma_ref_file
            )
        elif choice == '3':
            input_dir = input("Input directory: ").strip()
            pilot_bits_length = int(input("Pilot bits length [default 64]: ") or 64)
            auto_tune = input("Auto-tune? (y/n): ").lower().startswith('y')
            ecc = input("Enable ECC? (y/n): ").lower().startswith('y')
            reference_extract = input("Reference extraction? (y/n): ").lower().startswith('y')
            process_file_args = {
                'pilot_bits_length': pilot_bits_length,
                'alpha_candidates': [0.005, 0.01, 0.02],
                'block_sizes': [(8,8), (16,16)],
                'metric_fn': lambda snr, ber: 0.7*snr+0.3*(1-ber),
                'ecc': ecc, 'reference_extract': reference_extract
            }
            audio_files = sorted(glob(os.path.join(input_dir, '**', '*.flac'), recursive=True) +
                                 glob(os.path.join(input_dir, '**', '*.wav'), recursive=True))
            for audio_path in audio_files:
                process_file(audio_path, None, 'auto' if auto_tune else 'manual', **process_file_args)
        elif choice == '4':
            input_path = input("Audio file to visualize: ").strip()
            audio, sr = sf.read(input_path)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            plot_waveform(audio, sr, title="Waveform")
            plot_spectrogram(audio, sr, title="Spectrogram")
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")

def main():
    parser = argparse.ArgumentParser(
        description="SVD-STFT Watermarking CLI with auto-tune, ECC, reference extraction, extract-only, plotting, and batch support.",
        epilog=EXAMPLE_USAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', type=str, help='Input audio file')
    parser.add_argument('--output', type=str, help='Output audio file or extracted bits file (optional, auto-generated if not set)')
    parser.add_argument('--input-dir', type=str, help='Input directory for batch mode')
    parser.add_argument('--output-csv', type=str, help='CSV file to log results (batch mode)')
    parser.add_argument('--auto-tune', action='store_true', help='Enable auto-tune calibration')
    parser.add_argument('--pilot-bits-length', type=int, default=64, help='Number of pilot bits for calibration')
    parser.add_argument('--alpha-candidates', type=float, nargs='+', default=[0.005, 0.01, 0.02], help='Alpha values to try')
    parser.add_argument('--block-sizes', type=str, nargs='+', default=['8x8', '16x16'], help='Block sizes to try (e.g. 8x8 16x16)')
    parser.add_argument('--calibration-metric', type=str, default='0.7*SNR+0.3*(1-BER)', help='Metric for calibration (Python expr, SNR and BER allowed)')
    parser.add_argument('--alpha', type=float, help='Manual alpha (overrides auto-tune)')
    parser.add_argument('--block-size', type=str, help='Manual block size (e.g. 8x8)')
    parser.add_argument('--threshold', type=float, help='Manual threshold (overrides auto-tune)')
    parser.add_argument('--key', type=int, default=42, help='Secret key for block permutation')
    parser.add_argument('--ecc', action='store_true', help='Enable ECC (Hamming) encoding/decoding')
    parser.add_argument('--reference-extract', action='store_true', help='Use reference (non-blind) extraction')
    parser.add_argument('--extract-only', action='store_true', help='Extract watermark only (no embedding)')
    parser.add_argument('--sigma-ref-file', type=str, help='Path to reference sigma file for extraction')
    parser.add_argument('--num-bits', type=int, help='Number of bits to extract (required in extract-only mode)')
    parser.add_argument('--plot', action='store_true', help='Plot waveform and spectrogram of original and watermarked audio')
    parser.add_argument('--config', type=str, help='YAML/JSON config file for parameters (optional)')
    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) if args.config.endswith('.yaml') or args.config.endswith('.yml') else json.load(f)
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)

    block_sizes = [parse_block_size(s) for s in args.block_sizes]
    metric_fn = parse_metric_fn(args.calibration_metric)

    if len(sys.argv) == 1:
        main_menu()
        return

    if args.input_dir:
        audio_files = sorted(glob(os.path.join(args.input_dir, '**', '*.flac'), recursive=True) +
                             glob(os.path.join(args.input_dir, '**', '*.wav'), recursive=True))
        log_results = []
        for audio_path in audio_files:
            base_name = os.path.basename(audio_path)
            out_base = os.path.splitext(base_name)[0] + '_watermarked.wav'
            output_path = os.path.join(WATERMARKED_DIR, out_base)
            mode = 'auto' if args.auto_tune else 'manual'
            process_file(
                audio_path, output_path, mode, args.pilot_bits_length, args.alpha_candidates, block_sizes, metric_fn,
                alpha=args.alpha, block_size=parse_block_size(args.block_size) if args.block_size else None,
                threshold=args.threshold, key=args.key, log_results=log_results,
                ecc=args.ecc, reference_extract=args.reference_extract, extract_only=args.extract_only,
                sigma_ref_file=args.sigma_ref_file, num_bits=args.num_bits, plot=args.plot
            )
        if args.output_csv:
            import csv
            with open(args.output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_results[0].keys())
                writer.writeheader()
                writer.writerows(log_results)
    else:
        if not args.input:
            print("--input required for single file mode")
            sys.exit(1)
        mode = 'auto' if args.auto_tune else 'manual'
        process_file(
            args.input, args.output, mode, args.pilot_bits_length, args.alpha_candidates, block_sizes, metric_fn,
            alpha=args.alpha, block_size=parse_block_size(args.block_size) if args.block_size else None,
            threshold=args.threshold, key=args.key,
            ecc=args.ecc, reference_extract=args.reference_extract, extract_only=args.extract_only,
            sigma_ref_file=args.sigma_ref_file, num_bits=args.num_bits, plot=args.plot
        )

if __name__ == "__main__":
    main() 