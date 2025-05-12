import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from audio_filters import fft_filter, load_audio, save_audio

def plot_frequency_spectrum(data, sample_rate, title):
    N = len(data)
    freq = np.fft.fftfreq(N, d=1/sample_rate)
    fft_data = np.fft.fft(data)
    magnitude = np.abs(fft_data)

    plt.plot(freq[:N // 2], magnitude[:N // 2])
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

def main():
    input_path = "input_audio.wav"      # 🔁 Replace with your actual path
    output_path = "filtered_audio.wav"
    cutoff = 1000                       # Hz
    filter_type = 'low'                # 'low' or 'high'

    # Load audio
    sample_rate, audio_data = load_audio(input_path)

    # If stereo, convert to mono
    if audio_data.ndim == 2:
        audio_data = audio_data.mean(axis=1).astype(audio_data.dtype)

    # Filter
    filtered = fft_filter(audio_data, sample_rate, cutoff, filter_type)

    # Save filtered audio
    save_audio(output_path, sample_rate, filtered)

    # Plot frequency spectrum
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plot_frequency_spectrum(audio_data, sample_rate, "Original Audio Spectrum")

    plt.subplot(2, 1, 2)
    plot_frequency_spectrum(filtered, sample_rate, f"Filtered Audio Spectrum ({filter_type}-pass)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
