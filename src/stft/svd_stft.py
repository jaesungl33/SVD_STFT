import numpy as np
from typing import List, Tuple, Optional, Sequence, Callable
from utils.blocks import split_into_blocks, reassemble_blocks, pseudo_permutation


def detect_watermark(
    S_complex: np.ndarray,
    block_size: Tuple[int, int],
    key: int,
    num_test_bits: int = 8,
    threshold: float = 50.0
) -> Tuple[bool, float]:
    """
    Detect if audio is already watermarked by analyzing STFT patterns.
    """
    S_mag = np.abs(S_complex)
    blocks = split_into_blocks(S_mag, block_size)
    n_blocks = len(blocks)
    if n_blocks < num_test_bits:
        return False, 0.0
    perm = pseudo_permutation(n_blocks, key)
    sigmas = []
    for idx in perm[:num_test_bits]:
        block = blocks[idx]
        _, Sigma, _ = np.linalg.svd(block, full_matrices=False)
        sigmas.append(Sigma[0])
    # Simple detection metrics
    alternating_score = 0.0
    for i in range(1, len(sigmas)):
        if (sigmas[i] > sigmas[i - 1]) != (sigmas[i - 1] > sigmas[i - 2] if i > 1 else True):
            alternating_score += 1
    alternating_score = alternating_score / max(1, len(sigmas) - 1)
    sigma_variance = np.var(sigmas)
    sigma_mean = np.mean(sigmas)
    relative_variance = sigma_variance / (sigma_mean ** 2 + 1e-10)
    sorted_sigmas = np.sort(sigmas)
    gaps = np.diff(sorted_sigmas)
    gap_variance = np.var(gaps)
    confidence_score = (
        alternating_score * 20 +
        relative_variance * 10 +
        gap_variance * 100
    )
    is_watermarked = confidence_score > threshold
    return is_watermarked, confidence_score


def hamming_encode(bits: List[int]) -> List[int]:
    """Encode bits using (7,4) Hamming code."""
    G = np.array([[1, 1, 1, 0, 0, 0, 0],
                  [1, 0, 0, 1, 1, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0],
                  [1, 1, 0, 1, 0, 0, 1]])
    while len(bits) % 4 != 0:
        bits.append(0)
    encoded = []
    for i in range(0, len(bits), 4):
        data = np.array(bits[i:i + 4])
        codeword = (G.T @ data) % 2
        encoded.extend(codeword.tolist())
    return encoded


def hamming_decode(bits: List[int]) -> List[int]:
    """Decode bits using (7,4) Hamming code."""
    H = np.array([[1, 0, 1, 0, 1, 0, 1],
                  [0, 1, 1, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1]])
    decoded = []
    for i in range(0, len(bits), 7):
        if i + 7 > len(bits):
            break
        codeword = np.array(bits[i:i + 7])
        syndrome = (H @ codeword) % 2
        # Simple pass-through; proper correction can be added
        data_bits = [codeword[2], codeword[4], codeword[5], codeword[6]]
        decoded.extend(data_bits)
    return decoded


def compute_stft(audio: np.ndarray, sr: int, n_fft: int, hop_length: int, window: str) -> np.ndarray:
    """Compute STFT of audio signal."""
    import librosa
    return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)


def _filter_block_indices_by_midband(total_rows: int, block_rows: int, midband_ratio: Optional[Tuple[float, float]]) -> Optional[set]:
    if not midband_ratio:
        return None
    fmin = int(total_rows * max(0.0, min(1.0, midband_ratio[0])))
    fmax = int(total_rows * max(0.0, min(1.0, midband_ratio[1])))
    allowed_block_rows = set()
    r = 0
    while r + block_rows <= total_rows:
        row_center = r + block_rows // 2
        if fmin <= row_center < fmax:
            allowed_block_rows.add(r)
        r += block_rows
    return allowed_block_rows


def embed_svd_stft(
    S_complex: np.ndarray,
    bits: Sequence[int],
    alpha: float,
    block_size: Tuple[int, int],
    key: int,
    multiplicative: bool = False,
    energy_aware: bool = False,
    energy_gamma: float = 0.5,
    midband_ratio: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, Optional[List[float]]]:
    """
    Embed a watermark bitstream into STFT using SVD.
    Defaults preserve additive, full-band behavior.
    """
    S_mag = np.abs(S_complex)
    S_phase = np.angle(S_complex)
    shape = S_mag.shape
    rows, cols = shape
    block_rows, block_cols = block_size
    blocks = split_into_blocks(S_mag, block_size)
    n_blocks = len(blocks)

    # Midband filtering: determine which block row-starts are allowed
    allowed_block_row_starts = _filter_block_indices_by_midband(rows, block_rows, midband_ratio)

    # Build list of candidate block indices respecting midband
    candidate_indices = []
    r = 0
    idx = 0
    while r + block_rows <= rows:
        c = 0
        row_allowed = (allowed_block_row_starts is None) or (r in allowed_block_row_starts)
        while c + block_cols <= cols:
            if row_allowed:
                candidate_indices.append(idx)
            idx += 1
            c += block_cols
        r += block_rows

    if len(bits) > len(candidate_indices):
        raise ValueError("Payload too large for available (midband-filtered) blocks.")

    perm = pseudo_permutation(len(candidate_indices), key)
    global_rms = float(np.sqrt(np.mean(S_mag ** 2)) + 1e-12)
    sigma_ref: List[float] = []

    for i, bit in enumerate(bits):
        idx_in_candidates = perm[i]
        block_idx = candidate_indices[idx_in_candidates]
        block = blocks[block_idx]
        U, Sigma, Vh = np.linalg.svd(block, full_matrices=False)
        sigma_ref.append(float(Sigma[0]))

        # Energy-aware alpha scaling
        alpha_eff = alpha
        if energy_aware:
            block_rms = float(np.sqrt(np.mean(block ** 2)) + 1e-12)
            alpha_eff = alpha * ((block_rms / global_rms) ** energy_gamma)

        if multiplicative:
            Sigma[0] = Sigma[0] * (1.0 + alpha_eff * (2 * bit - 1))
        else:
            Sigma[0] = Sigma[0] + alpha_eff * (2 * bit - 1)

        block_mod = U @ np.diag(Sigma) @ Vh
        blocks[block_idx] = block_mod

    S_mag_mod = reassemble_blocks(blocks, shape, block_size)
    S_complex_mod = S_mag_mod * np.exp(1j * S_phase)
    return S_complex_mod, sigma_ref


def extract_svd_stft(
    S_complex_mod: np.ndarray,
    alpha: float,
    block_size: Tuple[int, int],
    key: int,
    num_bits: int,
    threshold: Optional[float] = None,
    sigma_ref: Optional[List[float]] = None,
    midband_ratio: Optional[Tuple[float, float]] = None
) -> List[int]:
    """
    Extract a watermark bitstream from a watermarked STFT.
    If midband_ratio is provided, the same filtered block set as embedding is used.
    """
    S_mag_mod = np.abs(S_complex_mod)
    rows, cols = S_mag_mod.shape
    block_rows, block_cols = block_size
    blocks_mod = split_into_blocks(S_mag_mod, block_size)

    allowed_block_row_starts = _filter_block_indices_by_midband(rows, block_rows, midband_ratio)
    candidate_indices = []
    r = 0
    idx = 0
    while r + block_rows <= rows:
        c = 0
        row_allowed = (allowed_block_row_starts is None) or (r in allowed_block_row_starts)
        while c + block_cols <= cols:
            if row_allowed:
                candidate_indices.append(idx)
            idx += 1
            c += block_cols
        r += block_rows

    if num_bits > len(candidate_indices):
        raise ValueError("num_bits exceeds number of available (midband-filtered) blocks.")

    perm = pseudo_permutation(len(candidate_indices), key)
    sigmas: List[float] = []
    for idx_in_candidates in perm[:num_bits]:
        block_idx = candidate_indices[idx_in_candidates]
        block = blocks_mod[block_idx]
        _, Sigma, _ = np.linalg.svd(block, full_matrices=False)
        sigmas.append(float(Sigma[0]))

    if sigma_ref is not None:
        extracted = [1 if s > r else 0 for s, r in zip(sigmas, sigma_ref)]
    else:
        if threshold is None:
            threshold = float(np.median(sigmas))
        extracted = [1 if s > threshold else 0 for s in sigmas]
    return extracted


def snr_metric(original: np.ndarray, watermarked: np.ndarray) -> float:
    """Compute SNR in dB."""
    noise = original - watermarked
    return 10 * np.log10(np.sum(original ** 2) / (np.sum(noise ** 2) + 1e-10))


def calibrate_parameters(
    audio: np.ndarray,
    sr: int,
    bit_pattern: Sequence[int],
    alpha_candidates: Sequence[float],
    block_sizes: Sequence[Tuple[int, int]],
    metric_fn: Callable[[float, float], float],
    pilot_len: int = 5 * 16000,
    key: int = 42
) -> Tuple[float, Tuple[int, int], float]:
    """Calibrate (alpha, block_size, threshold) for SVD-STFT watermarking."""
    best_score = -np.inf
    best_params: Optional[Tuple[float, Tuple[int, int], float]] = None
    pilot_audio = audio[:pilot_len]

    for alpha in alpha_candidates:
        for block_size in block_sizes:
            try:
                S = compute_stft(pilot_audio, sr, 256, 64, 'hann')
                S_wm, sigma_ref = embed_svd_stft(S, bit_pattern, alpha, block_size, key)
                from stft.stft_transform import reconstruct_audio
                audio_wm = reconstruct_audio(S_wm, 256, 'hann')
                min_len = min(len(pilot_audio), len(audio_wm))
                snr = snr_metric(pilot_audio[:min_len], audio_wm[:min_len])
                extracted_bits = extract_svd_stft(S_wm, alpha, block_size, key, len(bit_pattern))
                ber = sum(b1 != b2 for b1, b2 in zip(bit_pattern, extracted_bits)) / len(bit_pattern)
                score = metric_fn(snr, ber)
                if score > best_score:
                    best_score = score
                    best_params = (alpha, block_size, float(np.median(sigma_ref)))
            except Exception:
                continue

    if best_params is None:
        raise RuntimeError("Calibration failed for all parameter combinations.")
    return best_params 