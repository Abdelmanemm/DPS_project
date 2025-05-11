from audio_filters import load_audio, save_audio, fft_filter

# Provide a path to your test WAV file (mono 16-bit PCM recommended)
input_file = "test_input.wav"  # Replace with your actual WAV file
lowpass_output = "output_lowpass.wav"
highpass_output = "output_highpass.wav"

# Parameters
cutoff_frequency = 1000  # 1000 Hz cutoff
print("Loading audio...")
sample_rate, audio_data = load_audio(input_file)

print(f"Sample Rate: {sample_rate} Hz")
print(f"Audio Shape: {audio_data.shape}")

# Apply Low-Pass Filter
print("Applying low-pass filter...")
filtered_low = fft_filter(audio_data, sample_rate, cutoff_frequency, filter_type='low')
save_audio(lowpass_output, sample_rate, filtered_low)
print(f"Low-pass filtered audio saved to: {lowpass_output}")

# Apply High-Pass Filter
print("Applying high-pass filter...")
filtered_high = fft_filter(audio_data, sample_rate, cutoff_frequency, filter_type='high')
save_audio(highpass_output, sample_rate, filtered_high)
print(f"High-pass filtered audio saved to: {highpass_output}")

print("Test completed successfully.")
