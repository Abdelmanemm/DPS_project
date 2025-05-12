import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from pydub import AudioSegment
import ffmpeg
import tempfile
from resize import resize_bilinear
import imageio


'''Image Utils'''
def get_image_input():
    """Gets and validates image input"""
    path = input("Enter image path: ").strip()
    try:
        img = imageio.imread(path)
        print(f"Loaded: {path} ({img.shape[1]}x{img.shape[0]})")
        return img
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None

def resize_interactive(img):
    """Interactive resize with validation"""
    print(f"Current dimensions: {img.shape[1]}x{img.shape[0]}")
    w = int(input("New width: "))
    h = int(input("New height: "))
    return resize_bilinear(img, h, w)

def get_float_input(prompt, min_val, max_val):
    """Validates float input"""
    while True:
        try:
            val = float(input(prompt))
            if min_val <= val <= max_val:
                return val
            print(f"Value must be between {min_val} and {max_val}")
        except ValueError:
            print("Invalid number")

def get_contrast_params():
    """Gets contrast adjustment parameters"""
    min_out = int(input("Minimum output value (0-254): "))
    max_out = int(input("Maximum output value (1-255): "))
    return max(0, min(min_out, 254)), min(255, max(max_out, 1))

def show_comparison(original, processed, title):
    """Displays before/after comparison for images"""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original if original.ndim == 3 else original, cmap='gray')
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(processed if processed.ndim == 3 else processed, cmap='gray')
    plt.title(title)
    plt.show()

def save_output(data, default_path, is_audio=False):
    """Saves processed output"""
    path = input(f"Enter output path [default: {default_path}]: ").strip() or default_path
    try:
        if is_audio:
            save_audio(path, data[0], data[1])
        else:
            imageio.imwrite(path, data)
        print(f"Saved to {path}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")




''''Audio Utils'''
def plot_audio_frequency(signal, sample_rate, title="Frequency Spectrum"):
    """
    Plots the frequency spectrum of an audio signal
    
    Args:
        signal: Audio samples (1D numpy array)
        sample_rate: Sampling rate in Hz
        title: Plot title
    """
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/sample_rate)[:N//2]
    
    plt.figure(figsize=(10, 4))
    plt.plot(xf, 2/N * np.abs(yf[:N//2]))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

def load_audio(file_path):
    """Loads audio file in any format using pydub"""
    try:
        # Convert to WAV if not already
        if not file_path.lower().endswith('.wav'):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                temp_wav = tmp.name
            
                (ffmpeg
                .input(file_path)
                .output(temp_wav, acodec='pcm_s16le', ar='44100')
                .run(quiet=True))
            file_path = temp_wav
            
        audio = AudioSegment.from_wav(file_path)
        samples = np.array(audio.get_array_of_samples())
        
        # Handle stereo -> mono conversion
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            samples = samples.mean(axis=1)
            
        return audio.frame_rate, samples
    
    except Exception as e:
        print(f"Error loading audio: {str(e)}")
        return None, None

def save_audio(file_path, sample_rate, data):
    """Saves audio as WAV file"""
    try:
        # Convert numpy array to AudioSegment
        if data.dtype != np.int16:
            data = (data * 32767).astype(np.int16)
            
        audio = AudioSegment(
            data.tobytes(),
            frame_rate=sample_rate,
            sample_width=data.dtype.itemsize,
            channels=1
        )
        audio.export(file_path, format="wav")
        print(f"Audio saved to {file_path}")
    except Exception as e:
        print(f"Error saving audio: {str(e)}")

def plot_audio_comparison(original, filtered, sample_rate, cutoff, filter_type):
    """Plots time and frequency domains with both original and filtered signals"""
    plt.figure(figsize=(12, 8))
    
    # Time domain comparison
    plt.subplot(2, 1, 1)
    time = np.arange(len(original))/sample_rate
    plt.plot(time[:1000], original[:1000], label='Original')
    plt.plot(time[:1000], filtered[:1000], alpha=0.7, label='Filtered')
    plt.title(f"Time Domain - {filter_type} Filter (Cutoff: {cutoff}Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    # Frequency domain comparison
    plt.subplot(2, 1, 2)
    N = len(original)
    
    # Calculate FFT for both signals
    yf_orig = fft(original)
    yf_filt = fft(filtered)
    xf = fftfreq(N, 1/sample_rate)[:N//2]
    
    plt.plot(xf, 2/N * np.abs(yf_orig[:N//2]), label='Original')
    plt.plot(xf, 2/N * np.abs(yf_filt[:N//2]), alpha=0.7, label='Filtered')
    
    # Add cutoff line
    plt.axvline(x=cutoff, color='r', linestyle='--', 
               label=f'Cutoff: {cutoff}Hz')
    
    plt.title("Frequency Domain Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()