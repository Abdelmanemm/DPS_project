import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from filters.grayscale import grayscale


# Read img we will test on it
test_img  = imageio.imread("/mnt/d/Workspace/DPS_project/test/test_img.jpg")
print(f"Image loaded Succefully....")


def test_grayscale(img: np.ndarray) -> None:
    # Apply custom grayscale filter
    custom_gray = grayscale(img)  # Should return a 2D grayscale numpy array

    # Convert numpy image to PIL for comparison (PIL expects RGB)
    pil_gray = Image.fromarray(img).convert("L")  
    pil_gray_np = np.array(pil_gray)

    # Compute Mean Absolute Error between outputs
    mae = np.mean(np.abs(custom_gray.astype(np.float32) - pil_gray_np.astype(np.float32)))
    print(f"Mean Absolute Error (custom vs PIL): {mae:.2f}")

    # Display comparison
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(custom_gray, cmap="gray")
    axs[1].set_title("Implmented Grayscale")
    axs[1].axis("off")

    axs[2].imshow(pil_gray_np, cmap="gray")
    axs[2].set_title("PIL Grayscale")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_grayscale(test_img)