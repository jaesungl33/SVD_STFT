"""
Watermarking methods for comparison testing.
Includes SVD_STFT and other established methods.
"""

import numpy as np
import librosa
from typing import List, Tuple, Optional, Dict, Any
from scipy import signal
from scipy.fft import fft, ifft, dct, idct
import random

class WatermarkMethod:
    """Base class for watermarking methods."""
    
    def __init__(self, name: str):
        self.name = name
        self.calibrated = False
        self.alpha = None
        
    def calibrate(self, audio: np.ndarray, sr: int, target_pesq_drop: float = 0.05) -> float:
        """Calibrate method to achieve target PESQ drop."""
        raise NotImplementedError
        
    def embed(self, audio: np.ndarray, sr: int, bits: List[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Embed watermark and return watermarked audio + metadata."""
        raise NotImplementedError
        
    def extract(self, audio: np.ndarray, sr: int, num_bits: int, metadata: Dict[str, Any]) -> List[int]:
        """Extract watermark from audio."""
        raise NotImplementedError


class SVDSTFTMethod(WatermarkMethod):
    """Your SVD_STFT watermarking method."""
    
    def __init__(self):
        super().__init__("SVD_STFT")
        self.block_size = (32, 32)
        self.key = 42
        
    def calibrate(self, audio: np.ndarray, sr: int, target_pesq_drop: float = 0.05) -> float:
        """Calibrate alpha to achieve target PESQ drop."""
        from .metrics import compute_pesq
        
        # Test alpha values
        alpha_candidates = np.logspace(-3, 1, 20)
        test_bits = [random.randint(0, 1) for _ in range(64)]
        
        # Use first 5 seconds for calibration
        test_audio = audio[:min(len(audio), 5 * sr)]
        
        for alpha in alpha_candidates:
            try:
                watermarked, _ = self.embed(test_audio, sr, test_bits)
                pesq_orig = compute_pesq(test_audio, test_audio, sr)
                pesq_wm = compute_pesq(test_audio, watermarked, sr)
                pesq_drop = pesq_orig - pesq_wm
                
                if pesq_drop <= target_pesq_drop:
                    self.alpha = alpha
                    self.calibrated = True
                    return alpha
            except:
                continue
                
        # Fallback to conservative value
        self.alpha = 0.01
        self.calibrated = True
        return self.alpha
        
    def embed(self, audio: np.ndarray, sr: int, bits: List[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Embed watermark using SVD_STFT."""
        if not self.calibrated:
            raise ValueError("Method must be calibrated first")
            
        from ..stft.svd_stft import embed_svd_stft, compute_stft
        from ..stft.stft_transform import reconstruct_audio
        
        # Compute STFT
        S = compute_stft(audio, sr, 256, 64, 'hann')
        
        # Embed watermark
        S_wm, sigma_ref = embed_svd_stft(
            S, bits, self.alpha, self.block_size, self.key
        )
        
        # Reconstruct audio
        audio_wm = reconstruct_audio(S_wm, 64, 'hann')
        
        # Ensure same length
        if len(audio_wm) > len(audio):
            audio_wm = audio_wm[:len(audio)]
        elif len(audio_wm) < len(audio):
            audio_wm = np.pad(audio_wm, (0, len(audio) - len(audio_wm)))
            
        metadata = {
            'sigma_ref': sigma_ref,
            'alpha': self.alpha,
            'block_size': self.block_size,
            'key': self.key
        }
        
        return audio_wm, metadata
        
    def extract(self, audio: np.ndarray, sr: int, num_bits: int, metadata: Dict[str, Any]) -> List[int]:
        """Extract watermark using SVD_STFT."""
        from ..stft.svd_stft import extract_svd_stft, compute_stft
        
        S = compute_stft(audio, sr, 256, 64, 'hann')
        
        extracted = extract_svd_stft(
            S, metadata['alpha'], metadata['block_size'], 
            metadata['key'], num_bits, sigma_ref=metadata['sigma_ref']
        )
        
        return extracted


class SpreadSpectrumMethod(WatermarkMethod):
    """Spread spectrum watermarking method."""
    
    def __init__(self):
        super().__init__("Spread_Spectrum")
        self.freq_range = (1000, 8000)
        
    def calibrate(self, audio: np.ndarray, sr: int, target_pesq_drop: float = 0.05) -> float:
        """Calibrate alpha to achieve target PESQ drop."""
        from .metrics import compute_pesq
        
        alpha_candidates = np.logspace(-4, 0, 20)
        test_bits = [random.randint(0, 1) for _ in range(64)]
        
        test_audio = audio[:min(len(audio), 5 * sr)]
        
        for alpha in alpha_candidates:
            try:
                watermarked, _ = self.embed(test_audio, sr, test_bits)
                pesq_orig = compute_pesq(test_audio, test_audio, sr)
                pesq_wm = compute_pesq(test_audio, watermarked, sr)
                pesq_drop = pesq_orig - pesq_wm
                
                if pesq_drop <= target_pesq_drop:
                    self.alpha = alpha
                    self.calibrated = True
                    return alpha
            except:
                continue
                
        self.alpha = 0.001
        self.calibrated = True
        return self.alpha
        
    def embed(self, audio: np.ndarray, sr: int, bits: List[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Embed watermark using spread spectrum."""
        if not self.calibrated:
            raise ValueError("Method must be calibrated first")
            
        # Generate pseudo-random sequence
        np.random.seed(42)
        seq_length = len(audio)
        pseudo_seq = np.random.choice([-1, 1], seq_length)
        
        # Create watermark signal
        watermark = np.zeros_like(audio)
        for i, bit in enumerate(bits):
            start_idx = i * (seq_length // len(bits))
            end_idx = min((i + 1) * (seq_length // len(bits)), seq_length)
            watermark[start_idx:end_idx] = (2 * bit - 1) * pseudo_seq[start_idx:end_idx]
            
        # Apply frequency masking
        freqs = np.fft.fftfreq(len(audio), 1/sr)
        mask = (np.abs(freqs) >= self.freq_range[0]) & (np.abs(freqs) <= self.freq_range[1])
        
        watermark_freq = np.fft.fft(watermark)
        watermark_freq[~mask] = 0
        watermark = np.real(np.fft.ifft(watermark_freq))
        
        # Add watermark
        audio_wm = audio + self.alpha * watermark
        
        metadata = {
            'alpha': self.alpha,
            'freq_range': self.freq_range,
            'pseudo_seq_seed': 42
        }
        
        return audio_wm, metadata
        
    def extract(self, audio: np.ndarray, sr: int, num_bits: int, metadata: Dict[str, Any]) -> List[int]:
        """Extract watermark using spread spectrum."""
        np.random.seed(metadata['pseudo_seq_seed'])
        seq_length = len(audio)
        pseudo_seq = np.random.choice([-1, 1], seq_length)
        
        # Apply frequency mask
        freqs = np.fft.fftfreq(len(audio), 1/sr)
        mask = (np.abs(freqs) >= metadata['freq_range'][0]) & (np.abs(freqs) <= metadata['freq_range'][1])
        
        audio_freq = np.fft.fft(audio)
        audio_freq[~mask] = 0
        audio_filtered = np.real(np.fft.ifft(audio_freq))
        
        # Extract bits
        extracted = []
        for i in range(num_bits):
            start_idx = i * (seq_length // num_bits)
            end_idx = min((i + 1) * (seq_length // num_bits), seq_length)
            
            correlation = np.corrcoef(audio_filtered[start_idx:end_idx], 
                                   pseudo_seq[start_idx:end_idx])[0, 1]
            bit = 1 if correlation > 0 else 0
            extracted.append(bit)
            
        return extracted


class DCTMethod(WatermarkMethod):
    """DCT-based watermarking method."""
    
    def __init__(self):
        super().__init__("DCT")
        self.freq_range = (1000, 8000)
        
    def calibrate(self, audio: np.ndarray, sr: int, target_pesq_drop: float = 0.05) -> float:
        """Calibrate alpha to achieve target PESQ drop."""
        from .metrics import compute_pesq
        
        alpha_candidates = np.logspace(-4, 0, 20)
        test_bits = [random.randint(0, 1) for _ in range(64)]
        
        test_audio = audio[:min(len(audio), 5 * sr)]
        
        for alpha in alpha_candidates:
            try:
                watermarked, _ = self.embed(test_audio, sr, test_bits)
                pesq_orig = compute_pesq(test_audio, test_audio, sr)
                pesq_wm = compute_pesq(test_audio, watermarked, sr)
                pesq_drop = pesq_orig - pesq_wm
                
                if pesq_drop <= target_pesq_drop:
                    self.alpha = alpha
                    self.calibrated = True
                    return alpha
            except:
                continue
                
        self.alpha = 0.01
        self.calibrated = True
        return self.alpha
        
    def embed(self, audio: np.ndarray, sr: int, bits: List[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Embed watermark using DCT."""
        if not self.calibrated:
            raise ValueError("Method must be calibrated first")
            
        # Apply DCT
        dct_coeffs = dct(audio, type=2)
        
        # Select frequency range
        freqs = np.fft.fftfreq(len(audio), 1/sr)
        mask = (np.abs(freqs) >= self.freq_range[0]) & (np.abs(freqs) <= self.freq_range[1])
        
        # Embed bits
        for i, bit in enumerate(bits):
            if i < np.sum(mask):
                freq_indices = np.where(mask)[0]
                if i < len(freq_indices):
                    idx = freq_indices[i]
                    dct_coeffs[idx] += self.alpha * (2 * bit - 1) * np.abs(dct_coeffs[idx])
        
        # Inverse DCT
        audio_wm = idct(dct_coeffs, type=2)
        
        metadata = {
            'alpha': self.alpha,
            'freq_range': self.freq_range
        }
        
        return audio_wm, metadata
        
    def extract(self, audio: np.ndarray, sr: int, num_bits: int, metadata: Dict[str, Any]) -> List[int]:
        """Extract watermark using DCT."""
        dct_coeffs = dct(audio, type=2)
        
        freqs = np.fft.fftfreq(len(audio), 1/sr)
        mask = (np.abs(freqs) >= metadata['freq_range'][0]) & (np.abs(freqs) <= metadata['freq_range'][1])
        
        extracted = []
        freq_indices = np.where(mask)[0]
        
        for i in range(min(num_bits, len(freq_indices))):
            idx = freq_indices[i]
            bit = 1 if dct_coeffs[idx] > 0 else 0
            extracted.append(bit)
            
        return extracted


def get_all_methods() -> List[WatermarkMethod]:
    """Get all available watermarking methods."""
    return [
        SVDSTFTMethod(),
        SpreadSpectrumMethod(),
        DCTMethod()
    ]
