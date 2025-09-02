#!/usr/bin/env python3
"""
Simple test script to demonstrate robustness evaluation.
"""

import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# Add src to path
sys.path.append('src')

from utils.robustness import RobustnessEvaluator
from stft.svd_stft import embed_svd_stft, extract_svd_stft, compute_stft
from stft.stft_transform import reconstruct_audio


def create_test_audio(duration: float = 10.0, sample_rate: int = 16000) -> np.ndarray:
    """Create a test audio signal."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Create a complex signal with multiple frequencies
    audio = (0.3 * np.sin(2 * np.pi * 440 * t) +  # A4 note
             0.2 * np.sin(2 * np.pi * 880 * t) +  # A5 note
             0.1 * np.sin(2 * np.pi * 1760 * t))  # A6 note
    return audio


def test_robustness():
    """Test robustness evaluation with a synthetic audio signal."""
    print("Creating test audio...")
    
    # Create test audio
    sample_rate = 16000
    audio = create_test_audio(duration=5.0, sample_rate=sample_rate)
    
    # Create watermark bits
    watermark_bits = np.random.randint(0, 2, 32).tolist()
    print(f"Generated watermark: {watermark_bits}")
    
    # Embed watermark
    print("Embedding watermark...")
    S = compute_stft(audio, sample_rate, n_fft=256, hop_length=64, window='hann')
    S_watermarked, sigma_ref = embed_svd_stft(S, watermark_bits, alpha=0.1, block_size=(8, 8), key=42)
    watermarked_audio = reconstruct_audio(S_watermarked, 256, 'hann')
    
    # Ensure same length
    min_len = min(len(audio), len(watermarked_audio))
    watermarked_audio = watermarked_audio[:min_len]
    audio = audio[:min_len]
    
    # Initialize evaluator
    evaluator = RobustnessEvaluator(sample_rate)
    
    # Define extraction parameters
    extract_params = {
        'alpha': 0.1,
        'block_size': (8, 8),
        'key': 42,
        'num_bits': len(watermark_bits),
        'sigma_ref': sigma_ref
    }
    
    def extract_function(audio_signal, **params):
        """Extract watermark from audio."""
        S = compute_stft(audio_signal, sample_rate, n_fft=256, hop_length=64, window='hann')
        return extract_svd_stft(
            S, 
            params['alpha'],
            params['block_size'],
            params['key'],
            params['num_bits'],
            sigma_ref=params['sigma_ref']
        )
    
    # Test clean extraction first
    print("Testing clean extraction...")
    clean_extracted = extract_function(watermarked_audio, **extract_params)
    clean_ber = evaluator._calculate_ber(watermark_bits, clean_extracted)
    print(f"Clean extraction BER: {clean_ber:.4f}")
    print(f"Clean extracted bits: {clean_extracted}")
    
    # Test robustness
    print("\nTesting robustness against attacks...")
    results = evaluator.evaluate_robustness(
        original_audio=audio,
        watermarked_audio=watermarked_audio,
        extract_function=extract_function,
        extract_params=extract_params,
        watermark_bits=watermark_bits,
        test_config="evaluation"
    )
    
    # Print results
    print("\n" + "="*60)
    print("ROBUSTNESS EVALUATION RESULTS")
    print("="*60)
    
    for attack_name, result in results.items():
        ber = result.get('ber', 1.0)
        snr = result.get('snr', -np.inf)
        success = "✓" if result.get('success', False) else "✗"
        
        print(f"{attack_name:<20} BER: {ber:<8.4f} SNR: {snr:<8.2f} dB {success}")
    
    # Generate report
    print("\n" + "="*60)
    report = evaluator.generate_report(results)
    print(report)
    
    # Save test audio files
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    sf.write(output_dir / "original.wav", audio, sample_rate)
    sf.write(output_dir / "watermarked.wav", watermarked_audio, sample_rate)
    
    print(f"\nTest audio files saved to: {output_dir}")


if __name__ == "__main__":
    test_robustness()
