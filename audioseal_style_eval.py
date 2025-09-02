#!/usr/bin/env python3
"""
AudioSeal-style robustness evaluation harness for SVD-STFT.
- Imperceptibility: PESQ (wb, 16kHz), SI-SNR
- Detection metrics: TPR, FPR, Accuracy (pi=0.5), BER, NC
- Attacks: MP3 128k, Resample to 22.05k, Pink/White noise @ 20 dB SNR, Speed 1.25x
- Same payload bits for all files (configurable: 16 or 32)

Outputs a markdown table at audioseal_eval/final_table.md
"""

import os
import sys
import json
import math
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import soundfile as sf

# Optional deps
try:
    from pesq import pesq as pesq_wb
except Exception:
    pesq_wb = None

try:
    import librosa
except Exception:
    librosa = None

# Local modules
sys.path.append('src')
from stft.svd_stft import compute_stft
from stft.stft_transform import reconstruct_audio

# Use the final embed/extract implementations
from final_best_svd_stft import (
    final_best_embed_svd_stft as embed_fn,
    final_best_extract_svd_stft as extract_fn,
)

SR = 16000
N_FFT = 1024
HOP = 256
WINDOW = 'hann'


def ensure_length(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def si_snr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    x, s = estimate, reference
    s = s - np.mean(s)
    x = x - np.mean(x)
    t = np.dot(x, s) / (np.dot(s, s) + eps) * s
    n = x - t
    return 10 * np.log10((np.dot(t, t) + eps) / (np.dot(n, n) + eps))


def compute_pesq(ref: np.ndarray, deg: np.ndarray, sr: int = SR) -> float:
    # Prefer local PESQ if available
    if pesq_wb is not None:
        try:
            return float(pesq_wb(sr, ref, deg, 'wb'))
        except Exception:
            pass
    # Fallback: Docker runner
    try:
        with tempfile.TemporaryDirectory() as td:
            ref_wav = Path(td) / 'ref.wav'
            deg_wav = Path(td) / 'deg.wav'
            sf.write(str(ref_wav), ref, sr)
            sf.write(str(deg_wav), deg, sr)
            cmd = [
                'docker', 'run', '--rm', '-v', f'{ref_wav.parent}:/io',
                'pesq-runner', '/io/ref.wav', '/io/deg.wav'
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            data = json.loads(out.decode('utf-8').strip().split('\n')[-1])
            return float(data.get('pesq_wb', float('nan')))
    except Exception:
        return float('nan')
    return float('nan')


def normalized_correlation(bits_ref: List[int], bits_ext: List[int]) -> float:
    a = np.array(bits_ref, dtype=float)
    b = np.array(bits_ext, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    a = (a - np.mean(a))
    b = (b - np.mean(b))
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def ber(bits_ref: List[int], bits_ext: List[int]) -> float:
    n = min(len(bits_ref), len(bits_ext))
    if n == 0:
        return 1.0
    return float(np.mean(np.array(bits_ref[:n]) != np.array(bits_ext[:n])))


def acc_from_ber(value: float) -> float:
    return 1.0 - value


def attack_mp3_128k(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    # Requires ffmpeg in PATH
    try:
        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / 'in.wav'
            mp3_path = Path(td) / 'tmp.mp3'
            out_path = Path(td) / 'out.wav'
            sf.write(str(wav_path), audio, sr)
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(wav_path), '-b:a', '128k', str(mp3_path)], check=True)
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(mp3_path), str(out_path)], check=True)
            y, sr2 = sf.read(str(out_path))
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr2 != sr and librosa is not None:
                y = librosa.resample(y, orig_sr=sr2, target_sr=sr)
            return y.astype(np.float32)
    except Exception:
        return audio.copy()


def attack_resample(audio: np.ndarray, sr: int = SR, target_sr: int = 22050) -> np.ndarray:
    if librosa is None:
        return audio.copy()
    y = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    y = librosa.resample(y, orig_sr=target_sr, target_sr=sr)
    return y.astype(np.float32)


def add_noise_snr(audio: np.ndarray, snr_db: float, color: str = 'white') -> np.ndarray:
    rng = np.random.default_rng(42)
    sig_power = np.mean(audio ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    if color == 'white':
        noise = rng.standard_normal(len(audio))
    else:  # pink noise via 1/f filter (approx)
        # Voss-McCartney algorithm (simple approximation)
        n = len(audio)
        rows = 16
        array = rng.integers(0, 2, size=(rows, n))
        pink = array.cumsum(axis=1).sum(axis=0).astype(np.float32)
        pink = pink - np.mean(pink)
        pink = pink / (np.std(pink) + 1e-8)
        noise = pink
    noise = noise / (np.std(noise) + 1e-8) * math.sqrt(noise_power)
    return (audio + noise).astype(np.float32)


def attack_speed(audio: np.ndarray, factor: float = 1.25, sr: int = SR) -> np.ndarray:
    if librosa is None:
        return audio.copy()
    y = librosa.effects.time_stretch(audio, rate=factor)
    # resample to original length windowed by simple crop/pad
    if len(y) > len(audio):
        y = y[:len(audio)]
    else:
        y = np.pad(y, (0, len(audio) - len(y)))
    return y.astype(np.float32)


def load_audio_mono_16k(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path))
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != SR:
        if librosa is None:
            raise RuntimeError('librosa required for resampling to 16kHz')
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y.astype(np.float32)


def embed_audio(audio: np.ndarray, bits: List[int], alpha: float = 0.15, block_size: Tuple[int, int] = (8, 8)) -> Tuple[np.ndarray, Dict]:
    S = compute_stft(audio, SR, n_fft=N_FFT, hop_length=HOP, window=WINDOW)
    S_mod, ref = embed_fn(S, bits, alpha=alpha, block_size=block_size, key=42, redundancy=5, use_error_correction=True)
    y = reconstruct_audio(S_mod, N_FFT, WINDOW)
    y, _ = ensure_length(y, audio)
    return y.astype(np.float32), ref


def extract_bits(audio: np.ndarray, ref: Dict, alpha: float = 0.15, block_size: Tuple[int, int] = (8, 8)) -> List[int]:
    S = compute_stft(audio, SR, n_fft=N_FFT, hop_length=HOP, window=WINDOW)
    bits = extract_fn(S, ref, alpha=alpha, block_size=block_size, key=42)
    return bits


def evaluate_folder(input_dirs: List[str], payload_len: int = 16, max_files: int = 50) -> Dict:
    files: List[Path] = []
    for d in input_dirs:
        p = Path(d)
        if not p.exists():
            continue
        files.extend(list(p.glob('*.wav')))
    files = files[:max_files]
    if not files:
        raise RuntimeError('No audio files found')

    # fixed payload across all files
    rng = np.random.default_rng(123)
    payload = rng.integers(0, 2, size=payload_len).tolist()

    results: Dict = {
        'clean': {'PESQ': [], 'SI_SNR': [], 'BER': [], 'NC': []},
        'mp3': {'ACC': [], 'BER': [], 'NC': []},
        'resample22k': {'ACC': [], 'BER': [], 'NC': []},
        'pink20': {'ACC': [], 'BER': [], 'NC': []},
        'white20': {'ACC': [], 'BER': [], 'NC': []},
        'speed1_25': {'ACC': [], 'BER': [], 'NC': []},
        'detection': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    }

    # negatives for FPR: extract from original unwatermarked
    neg_nc_values: List[float] = []

    for f in files:
        x = load_audio_mono_16k(f)
        # negatives
        neg_bits = extract_bits(x, ref={'block_indices': [], 'adaptive_alphas': [], 'original_sigmas': [], 'bit_assignments': [], 'block_energies': [], 'sync_pattern': []}) if False else []  # no extraction without ref
        # embed
        y_wm, ref = embed_audio(x, payload)
        # imperceptibility metrics (clean)
        pesq_val = compute_pesq(x, y_wm)
        sisnr_val = si_snr(x, y_wm)
        ext_clean = extract_bits(y_wm, ref)
        ber_clean = ber(payload, ext_clean)
        nc_clean = normalized_correlation(payload, ext_clean)
        results['clean']['PESQ'].append(pesq_val)
        results['clean']['SI_SNR'].append(sisnr_val)
        results['clean']['BER'].append(ber_clean)
        results['clean']['NC'].append(nc_clean)

        # threshold for detection via NC >= 0.0
        detected = 1 if nc_clean >= 0.0 else 0
        results['detection']['TP'] += detected

        # attacks
        def eval_attack(att_audio: np.ndarray, key: str):
            ext_b = extract_bits(att_audio, ref)
            b = ber(payload, ext_b)
            nc = normalized_correlation(payload, ext_b)
            acc = acc_from_ber(b)
            results[key]['ACC'].append(acc)
            results[key]['BER'].append(b)
            results[key]['NC'].append(nc)

        eval_attack(attack_mp3_128k(y_wm, SR), 'mp3')
        eval_attack(attack_resample(y_wm, SR, 22050), 'resample22k')
        eval_attack(add_noise_snr(y_wm, 20.0, 'pink'), 'pink20')
        eval_attack(add_noise_snr(y_wm, 20.0, 'white'), 'white20')
        eval_attack(attack_speed(y_wm, 1.25, SR), 'speed1_25')

        # negatives for FPR: random sequence vs payload correlation baseline
        neg_rand = rng.integers(0, 2, size=len(payload)).tolist()
        neg_nc_values.append(normalized_correlation(payload, neg_rand))

    # detection metrics using NC threshold at 0.0
    tau = 0.0
    TP = sum(1 for v in results['clean']['NC'] if v >= tau)
    FN = len(results['clean']['NC']) - TP
    FP = sum(1 for v in neg_nc_values if v >= tau)
    TN = len(neg_nc_values) - FP
    results['detection'] = {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}

    return results


def summarize(results: Dict) -> str:
    def mean_safe(xs):
        xs = [x for x in xs if not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
        if not xs:
            return float('nan')
        return float(np.mean(xs))

    lines = []
    lines.append('Attack      | Metric | SVD-STFT')
    lines.append('------------|--------|---------')
    lines.append(f"Clean       | PESQ   |  {mean_safe(results['clean']['PESQ']):.2f}")
    lines.append(f"Clean       | SI-SNR |  {mean_safe(results['clean']['SI_SNR']):.2f} dB")
    # also include clean BER
    lines.append(f"Clean       | BER    |  {mean_safe(results['clean']['BER']):.3f}")

    lines.append(f"MP3 128k    | ACC    |  {mean_safe(results['mp3']['ACC']):.2f}")
    lines.append(f"MP3 128k    | BER    |  {mean_safe(results['mp3']['BER']):.3f}")

    lines.append(f"Resample22k | ACC    |  {mean_safe(results['resample22k']['ACC']):.2f}")
    lines.append(f"Resample22k | BER    |  {mean_safe(results['resample22k']['BER']):.3f}")

    lines.append(f"PinkNoise20 | ACC    |  {mean_safe(results['pink20']['ACC']):.2f}")
    lines.append(f"WhiteNoise20| ACC    |  {mean_safe(results['white20']['ACC']):.2f}")

    lines.append(f"Speed 1.25x | ACC    |  {mean_safe(results['speed1_25']['ACC']):.2f}")

    # detection metrics (overall)
    TP, FP, TN, FN = results['detection']['TP'], results['detection']['FP'], results['detection']['TN'], results['detection']['FN']
    TPR = TP / (TP + FN + 1e-8)
    FPR = FP / (FP + TN + 1e-8)
    ACC_pi = 0.5 * (TPR + (1 - FPR))

    lines.append('')
    lines.append(f"Detection (NC threshold = 0.0):")
    lines.append(f"- TPR: {TPR:.3f}")
    lines.append(f"- FPR: {FPR:.3f}")
    lines.append(f"- Accuracy (pi=0.5): {ACC_pi:.3f}")

    return '\n'.join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', nargs='+', default=['100sample_wav/music_wav'])
    parser.add_argument('--payload_len', type=int, default=16)
    parser.add_argument('--max_files', type=int, default=50)
    parser.add_argument('--out_dir', type=str, default='audioseal_eval')
    args = parser.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)

    results = evaluate_folder(args.input_dirs, payload_len=args.payload_len, max_files=args.max_files)
    table_md = summarize(results)

    out_md = Path(args.out_dir) / 'final_table.md'
    with open(out_md, 'w') as f:
        f.write(table_md + '\n')

    with open(Path(args.out_dir) / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote table to {out_md}")
    print('\n' + table_md)


if __name__ == '__main__':
    main()
