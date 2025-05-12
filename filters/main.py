from audio_filters import fft_filter
from utils import (
    load_audio,
    plot_audio_frequency,
    plot_audio_comparison,
    get_image_input,
    resize_interactive,
    get_float_input,
    get_contrast_params,
    show_comparison,
    save_output
)
from grayscale import grayscale
from edge_detection import sobel_edge_detection
from horizontal_flip import horizontal_flip
from sharpen import sharpen
from brightness import adjust_brightness
from contrast import adjust_contrast

def show_menu():
    """Displays the main processing menu"""
    print("\n" + "="*40)
    print("MAIN MENU".center(40))
    print("="*40)
    print("1. Image Processing")
    print("2. Audio Processing")
    print("3. Exit")

def show_image_menu():
    """Displays image filter options"""
    print("\n" + "="*40)
    print("IMAGE FILTERS".center(40))
    print("="*40)
    print("1. Grayscale")
    print("2. Edge Detection")
    print("3. Horizontal Flip")
    print("4. Resize")
    print("5. Sharpen")
    print("6. Brightness")
    print("7. Contrast")
    print("8. Back to Main Menu")

def show_audio_menu():
    """Displays audio filter options"""
    print("\n" + "="*40)
    print("AUDIO FILTERS".center(40))
    print("="*40)
    print("1. Low-Pass Filter")
    print("2. High-Pass Filter")
    print("3. Back to Main Menu")


def process_image():
    """Handles image processing pipeline"""
    img = get_image_input()
    if img is None:
        return
    
    while True:
        show_image_menu()
        choice = input("Select filter (1-8): ").strip()
        
        if choice == '8':
            break
            
        filters = {
            '1': ('Grayscale', lambda: grayscale(img)),
            '2': ('Edge Detection', lambda: sobel_edge_detection(img)),
            '3': ('Horizontal Flip', lambda: horizontal_flip(img)),
            '4': ('Resize', lambda: resize_interactive(img)),
            '5': ('Sharpen', lambda: sharpen(img)),
            '6': ('Brightness', lambda: adjust_brightness(img, get_float_input("Brightness factor (0.1-3.0): ", 0.1, 3.0))),
            '7': ('Contrast', lambda: adjust_contrast(img, *get_contrast_params()))
        }
        
        if choice in filters:
            name, func = filters[choice]
            result = func()
            show_comparison(img, result, name)
            save_output(result, f"filtered_{name.lower().replace(' ', '_')}.png")
        else:
            print("Invalid choice")

def process_audio():
    """Handles audio processing pipeline"""
    try:
        file_path = input("Enter audio file path: ").strip()
        sample_rate, data = load_audio(file_path)
        if sample_rate is None:
            return
            
        print(f"Loaded: {file_path} | Sample Rate: {sample_rate}Hz | Duration: {len(data)/sample_rate:.2f}s")
        
        # Show original frequency spectrum
        plot_audio_frequency(data, sample_rate, "Original Frequency Spectrum")
        
        while True:
            show_audio_menu()
            choice = input("Select filter (1-3): ").strip()
            
            if choice == '3':
                break
                
            cutoff = get_float_input(
                f"Enter cutoff frequency (max {sample_rate//2}Hz): ",
                20, sample_rate//2 - 1
            )
            
            if choice == '1':
                filtered = fft_filter(data, sample_rate, cutoff, 'low')
                save_path = "audio_lowpass.wav"
                filter_type = "Low-Pass"
            elif choice == '2':
                filtered = fft_filter(data, sample_rate, cutoff, 'high')
                save_path = "audio_highpass.wav"
                filter_type = "High-Pass"
            else:
                print("Invalid choice")
                continue
            
            # Show comparison plots
            plot_audio_comparison(data, filtered, sample_rate, cutoff, filter_type)
            
            # Save the result
            save_output((sample_rate, filtered), save_path, is_audio=True)
            
    except Exception as e:
        print(f"Audio processing error: {str(e)}")

def main():
    """Main program loop"""
    while True:
        show_menu()
        choice = input("Select option (1-3): ").strip()
        
        if choice == '1':
            process_image()
        elif choice == '2':
            process_audio()
        elif choice == '3':
            print("Exiting program...")
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()