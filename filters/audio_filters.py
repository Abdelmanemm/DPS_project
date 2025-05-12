
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def fft_filter(audio_data, sample_rate, cutoff_freq, filter_type='low'):
    """
    Applies a low-pass or high-pass filter to the input audio data using FFT.

    Parameters:
        audio_data (np.array): The raw audio signal
        sample_rate (int): Sampling rate of the audio
        cutoff_freq (float): Cutoff frequency in Hz
        filter_type (str): 'low' for low-pass, 'high' for high-pass

    Returns:
        np.array: Filtered audio signal
    """
    N = len(audio_data)
    freq = np.fft.fftfreq(N, d=1/sample_rate)  # Frequency axis
    fft_audio = np.fft.fft(audio_data)

    if filter_type == 'low':
        fft_audio[np.abs(freq) > cutoff_freq] = 0
    elif filter_type == 'high':
        fft_audio[np.abs(freq) < cutoff_freq] = 0
    else:
        raise ValueError("Invalid filter_type. Use 'low' or 'high'.")

    filtered_audio = np.fft.ifft(fft_audio)
    return np.real(filtered_audio).astype(audio_data.dtype)


def load_audio(file_path):
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data


def save_audio(file_path, sample_rate, data):
    wavfile.write(file_path, sample_rate, data)