"""
Robustness evaluation module for audio watermarking.
Implements audio augmentations similar to AudioSeal for testing watermark robustness.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional, Union, Dict, Any
from scipy import signal
from scipy.io import wavfile
import tempfile
import os
import subprocess
import json
from pathlib import Path


class AudioAugmenter:
    """Audio augmentation class for robustness testing."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def bandpass_filter(self, audio: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """Apply bandpass filter to audio."""
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, audio)
    
    def highpass_filter(self, audio: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """Apply highpass filter to audio."""
        nyquist = self.sample_rate / 2
        cutoff = cutoff_freq / nyquist
        b, a = signal.butter(4, cutoff, btype='high')
        return signal.filtfilt(b, a, audio)
    
    def lowpass_filter(self, audio: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """Apply lowpass filter to audio."""
        nyquist = self.sample_rate / 2
        cutoff = cutoff_freq / nyquist
        b, a = signal.butter(4, cutoff, btype='low')
        return signal.filtfilt(b, a, audio)
    
    def speed_change(self, audio: np.ndarray, speed_factor: float) -> np.ndarray:
        """Change audio speed by a factor."""
        return librosa.effects.time_stretch(audio, rate=speed_factor)
    
    def resample(self, audio: np.ndarray, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate and back."""
        # Upsample to target rate
        audio_upsampled = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=target_sr)
        # Downsample back to original rate
        audio_downsampled = librosa.resample(audio_upsampled, orig_sr=target_sr, target_sr=self.sample_rate)
        return audio_downsampled
    
    def boost_audio(self, audio: np.ndarray, factor: float) -> np.ndarray:
        """Amplify audio by multiplying by a factor."""
        return audio * factor
    
    def duck_audio(self, audio: np.ndarray, factor: float) -> np.ndarray:
        """Reduce audio volume by multiplying by a factor."""
        return audio * factor
    
    def add_echo(self, audio: np.ndarray, delay: float, volume: float) -> np.ndarray:
        """Add echo effect to audio."""
        delay_samples = int(delay * self.sample_rate)
        echo_signal = np.zeros_like(audio)
        echo_signal[delay_samples:] = audio[:-delay_samples] * volume
        return audio + echo_signal
    
    def add_pink_noise(self, audio: np.ndarray, std_dev: float) -> np.ndarray:
        """Add pink noise to audio."""
        # Generate pink noise
        noise = np.random.normal(0, 1, len(audio))
        # Apply pink noise filter (1/f filter)
        freqs = np.fft.fftfreq(len(noise), 1/self.sample_rate)
        pink_filter = 1 / np.sqrt(np.abs(freqs) + 1e-10)
        pink_filter[0] = 0  # Remove DC component
        noise_fft = np.fft.fft(noise)
        pink_noise_fft = noise_fft * pink_filter
        pink_noise = np.real(np.fft.ifft(pink_noise_fft))
        # Normalize and scale
        pink_noise = pink_noise / np.std(pink_noise) * std_dev
        return audio + pink_noise
    
    def add_white_noise(self, audio: np.ndarray, std_dev: float) -> np.ndarray:
        """Add white (Gaussian) noise to audio."""
        noise = np.random.normal(0, std_dev, len(audio))
        return audio + noise
    
    def smooth_audio(self, audio: np.ndarray, window_size: int) -> np.ndarray:
        """Smooth audio using moving average filter."""
        kernel = np.ones(window_size) / window_size
        return signal.convolve(audio, kernel, mode='same')
    
    def encode_aac(self, audio: np.ndarray, bitrate: str = "128k") -> np.ndarray:
        """Encode audio in AAC format and decode back."""
        try:
            # Save original audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_in:
                sf.write(temp_in.name, audio, self.sample_rate)
            
            # Encode to AAC
            temp_aac = temp_in.name.replace('.wav', '.aac')
            cmd = [
                'ffmpeg', '-y', '-i', temp_in.name,
                '-c:a', 'aac', '-b:a', bitrate,
                temp_aac
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Decode back to WAV
            temp_out = temp_in.name.replace('.wav', '_decoded.wav')
            cmd = [
                'ffmpeg', '-y', '-i', temp_aac,
                '-c:a', 'pcm_s16le', temp_out
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Load decoded audio
            decoded_audio, _ = sf.read(temp_out)
            
            # Cleanup
            os.unlink(temp_in.name)
            os.unlink(temp_aac)
            os.unlink(temp_out)
            
            return decoded_audio
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: return original audio if ffmpeg not available
            return audio
    
    def encode_mp3(self, audio: np.ndarray, bitrate: str = "128k") -> np.ndarray:
        """Encode audio in MP3 format and decode back."""
        try:
            # Save original audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_in:
                sf.write(temp_in.name, audio, self.sample_rate)
            
            # Encode to MP3
            temp_mp3 = temp_in.name.replace('.wav', '.mp3')
            cmd = [
                'ffmpeg', '-y', '-i', temp_in.name,
                '-c:a', 'mp3', '-b:a', bitrate,
                temp_mp3
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Decode back to WAV
            temp_out = temp_in.name.replace('.wav', '_decoded.wav')
            cmd = [
                'ffmpeg', '-y', '-i', temp_mp3,
                '-c:a', 'pcm_s16le', temp_out
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Load decoded audio
            decoded_audio, _ = sf.read(temp_out)
            
            # Cleanup
            os.unlink(temp_in.name)
            os.unlink(temp_mp3)
            os.unlink(temp_out)
            
            return decoded_audio
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: return original audio if ffmpeg not available
            return audio
    
    def encode_encodec(self, audio: np.ndarray) -> np.ndarray:
        """Encode audio with EnCodec and decode back."""
        try:
            # Resample to 24kHz for EnCodec
            audio_24k = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=24000)
            
            # Save 24kHz audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_in:
                sf.write(temp_in.name, audio_24k, 24000)
            
            # Encode with EnCodec (requires encodec package)
            temp_encoded = temp_in.name.replace('.wav', '_encoded.wav')
            cmd = [
                'python', '-c', 
                f'''
import torch
from encodec import EncodecModel
from encodec.utils import convert_audio
import soundfile as sf

model = EnCodecModel.from_pretrained("facebook/encodec_24khz")
model.eval()

audio, sr = sf.read("{temp_in.name}")
audio = torch.tensor(audio).unsqueeze(0)
audio = convert_audio(audio, sr, model.sample_rate, model.channels)

with torch.no_grad():
    encoded_frames = model.encode(audio)
    decoded = model.decode(encoded_frames)

decoded = decoded.squeeze(0).numpy()
sf.write("{temp_encoded}", decoded, model.sample_rate)
'''
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Resample back to original sample rate
            decoded_audio_24k, _ = sf.read(temp_encoded)
            decoded_audio = librosa.resample(decoded_audio_24k, orig_sr=24000, target_sr=self.sample_rate)
            
            # Cleanup
            os.unlink(temp_in.name)
            os.unlink(temp_encoded)
            
            return decoded_audio
            
        except (subprocess.CalledProcessError, FileNotFoundError, ImportError):
            # Fallback: return original audio if EnCodec not available
            return audio


class RobustnessEvaluator:
    """Evaluator for testing watermark robustness against various attacks."""
    
    def __init__(self, sample_rate: int = 16000):
        self.augmenter = AudioAugmenter(sample_rate)
        self.sample_rate = sample_rate
    
    def evaluate_robustness(
        self,
        original_audio: np.ndarray,
        watermarked_audio: np.ndarray,
        extract_function: callable,
        extract_params: Dict[str, Any],
        watermark_bits: list,
        test_config: str = "evaluation"  # "training" or "evaluation"
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate robustness against all attacks.
        
        Args:
            original_audio: Original audio signal
            watermarked_audio: Watermarked audio signal
            extract_function: Function to extract watermark
            extract_params: Parameters for extraction function
            watermark_bits: Original watermark bits
            test_config: "training" or "evaluation" to use different parameters
        
        Returns:
            Dictionary with results for each attack
        """
        results = {}
        
        # Define attack parameters based on config
        if test_config == "training":
            attack_params = {
                "bandpass_filter": {"low_freq": 300, "high_freq": 8000},
                "highpass_filter": {"cutoff_freq": 500},
                "lowpass_filter": {"cutoff_freq": 5000},
                "speed_change": {"speed_factor": np.random.uniform(0.9, 1.1)},
                "resample": {"target_sr": 32000},
                "boost_audio": {"factor": 1.2},
                "duck_audio": {"factor": 0.8},
                "echo": {"delay": np.random.uniform(0.1, 0.5), "volume": np.random.uniform(0.1, 0.5)},
                "pink_noise": {"std_dev": 0.01},
                "white_noise": {"std_dev": 0.001},
                "smooth": {"window_size": np.random.randint(2, 11)},
                "aac": {"bitrate": "128k"},
                "mp3": {"bitrate": "128k"},
                "encodec": {}
            }
        else:  # evaluation
            attack_params = {
                "bandpass_filter": {"low_freq": 500, "high_freq": 5000},
                "highpass_filter": {"cutoff_freq": 1500},
                "lowpass_filter": {"cutoff_freq": 500},
                "speed_change": {"speed_factor": 1.25},
                "resample": {"target_sr": 32000},
                "boost_audio": {"factor": 10.0},
                "duck_audio": {"factor": 0.1},
                "echo": {"delay": 0.5, "volume": 0.5},
                "pink_noise": {"std_dev": 0.1},
                "white_noise": {"std_dev": 0.05},
                "smooth": {"window_size": 40},
                "aac": {"bitrate": "64k"},
                "mp3": {"bitrate": "32k"},
                "encodec": {}
            }
        
        # Test each attack
        attacks = {
            "bandpass_filter": lambda audio: self.augmenter.bandpass_filter(
                audio, **attack_params["bandpass_filter"]
            ),
            "highpass_filter": lambda audio: self.augmenter.highpass_filter(
                audio, **attack_params["highpass_filter"]
            ),
            "lowpass_filter": lambda audio: self.augmenter.lowpass_filter(
                audio, **attack_params["lowpass_filter"]
            ),
            "speed_change": lambda audio: self.augmenter.speed_change(
                audio, **attack_params["speed_change"]
            ),
            "resample": lambda audio: self.augmenter.resample(
                audio, **attack_params["resample"]
            ),
            "boost_audio": lambda audio: self.augmenter.boost_audio(
                audio, **attack_params["boost_audio"]
            ),
            "duck_audio": lambda audio: self.augmenter.duck_audio(
                audio, **attack_params["duck_audio"]
            ),
            "echo": lambda audio: self.augmenter.add_echo(
                audio, **attack_params["echo"]
            ),
            "pink_noise": lambda audio: self.augmenter.add_pink_noise(
                audio, **attack_params["pink_noise"]
            ),
            "white_noise": lambda audio: self.augmenter.add_white_noise(
                audio, **attack_params["white_noise"]
            ),
            "smooth": lambda audio: self.augmenter.smooth_audio(
                audio, **attack_params["smooth"]
            ),
            "aac": lambda audio: self.augmenter.encode_aac(
                audio, **attack_params["aac"]
            ),
            "mp3": lambda audio: self.augmenter.encode_mp3(
                audio, **attack_params["mp3"]
            ),
            "encodec": lambda audio: self.augmenter.encode_encodec(
                audio
            )
        }
        
        for attack_name, attack_func in attacks.items():
            try:
                # Apply attack
                attacked_audio = attack_func(watermarked_audio.copy())
                
                # Ensure same length
                min_len = min(len(attacked_audio), len(watermarked_audio))
                attacked_audio = attacked_audio[:min_len]
                
                # Extract watermark from attacked audio
                extracted_bits = extract_function(attacked_audio, **extract_params)
                
                # Calculate metrics
                ber = self._calculate_ber(watermark_bits, extracted_bits)
                snr = self._calculate_snr(original_audio[:min_len], attacked_audio)
                
                results[attack_name] = {
                    "ber": ber,
                    "snr": snr,
                    "success": ber < 0.1  # Consider successful if BER < 10%
                }
                
            except Exception as e:
                results[attack_name] = {
                    "ber": 1.0,
                    "snr": -np.inf,
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def _calculate_ber(self, original_bits: list, extracted_bits: list) -> float:
        """Calculate Bit Error Rate."""
        if len(extracted_bits) == 0:
            return 1.0
        
        min_len = min(len(original_bits), len(extracted_bits))
        if min_len == 0:
            return 1.0
        
        errors = sum(1 for i in range(min_len) if original_bits[i] != extracted_bits[i])
        return errors / min_len
    
    def _calculate_snr(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio in dB."""
        noise = original - processed
        signal_power = np.sum(original ** 2)
        noise_power = np.sum(noise ** 2)
        
        if noise_power == 0:
            return np.inf
        
        return 10 * np.log10(signal_power / noise_power)
    
    def generate_report(self, results: Dict[str, Dict[str, float]]) -> str:
        """Generate a comprehensive robustness report."""
        report = "=== ROBUSTNESS EVALUATION REPORT ===\n\n"
        
        # Summary statistics
        successful_attacks = sum(1 for r in results.values() if r.get("success", False))
        total_attacks = len(results)
        success_rate = successful_attacks / total_attacks * 100
        
        report += f"Overall Success Rate: {success_rate:.1f}% ({successful_attacks}/{total_attacks})\n\n"
        
        # Detailed results
        report += "Detailed Results:\n"
        report += "-" * 80 + "\n"
        report += f"{'Attack':<20} {'BER':<10} {'SNR (dB)':<12} {'Success':<10}\n"
        report += "-" * 80 + "\n"
        
        for attack_name, result in results.items():
            ber = result.get("ber", 1.0)
            snr = result.get("snr", -np.inf)
            success = "✓" if result.get("success", False) else "✗"
            
            report += f"{attack_name:<20} {ber:<10.4f} {snr:<12.2f} {success:<10}\n"
        
        return report
