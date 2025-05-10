# main_menu.py
# Menu system using match-case for image filter selection

def show_menu():
    print("""\n  Select a filter to apply:
1. Grayscale
2. Gaussian Blur
3. Edge Detection
4. Horizontal Flip
5. Resize
6. Contours
7. Sharpen
8. Brightness
9. Contrast
10. Saturation""")

def get_user_filter():
    while True:
        show_menu()
        choice = input("Enter your choice (1–10): ").strip()

        match choice:
            case "1":
                return 'grayscale'
            case "2":
                return 'gaussian_blur'
            case "3":
                return 'edge_detection'
            case "4":
                return 'horizontal_flip'
            case "5":
                return 'resize'
            case "6":
                return 'contours'
            case "7":
                return 'sharpen'
            case "8":
                return 'brightness'
            case "9":
                return 'contrast'
            case "10":
                return 'saturation'
            case _:
                print(" Invalid input. Please choose a number between 1 and 10.\n")
