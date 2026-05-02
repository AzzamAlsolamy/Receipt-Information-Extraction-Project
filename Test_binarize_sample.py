import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def load_grayscale(image_path: Path):
    """Load image and convert to grayscale using cv2."""
    image = cv2.imread(str(image_path))
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def otsu_threshold(gray):
    """
    Apply Otsu's binarization using cv2.

    cv2.threshold with cv2.THRESH_BINARY + cv2.THRESH_OTSU automatically
    computes the optimal global threshold by maximizing inter-class variance.

    Returns:
        thresh_val : the threshold value Otsu computed
        binary     : binarized image (H, W) with values {0, 255}
    """
    thresh_val, binary = cv2.threshold(
        gray,
        0,                                       # ignored — Otsu computes it
        255,                                     # max value assigned to white pixels
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return thresh_val, binary


def main():
    # ── Sample image ──────────────────────────────────────────────────────────
    sample_path = Path("/teamspace/studios/this_studio/Receipt-Information-Extraction-Project/Datasets/SROIE2019/train/img/X51008164999.jpg")
    print(f"Sample: {sample_path.name}")

    # ── Load & binarize ───────────────────────────────────────────────────────
    original, gray = load_grayscale(sample_path)
    thresh_val, binary = otsu_threshold(gray)
    print(f"Otsu threshold value: {thresh_val:.1f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        f"Otsu Binarization — {sample_path.name}  |  threshold = {thresh_val:.1f}",
        fontsize=13, fontweight="bold",
    )

    # Original (cv2 loads BGR → convert to RGB for matplotlib)
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original (RGB)", fontsize=10)
    axes[0].axis("off")

    # Grayscale
    axes[1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Grayscale", fontsize=10)
    axes[1].axis("off")

    # Otsu binary
    white_pct = (binary == 255).mean() * 100
    axes[2].imshow(binary, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(
        f"Otsu Binarization\nthreshold = {thresh_val:.1f}  |  {white_pct:.1f}% white",
        fontsize=10,
    )
    axes[2].axis("off")

    plt.tight_layout()

    out_path = Path("otsu_binarization_test.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()