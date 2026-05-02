import cv2
from pathlib import Path

def load_grayscale(image_path: Path):
    """Load image and convert to grayscale using cv2."""
    image = cv2.imread(str(image_path))                  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

    return gray


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
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary


def save_image(image, save_path : Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), image)
         



def process_split(src_img_dir: Path, dst_img_dir: Path) -> None:
    """
    Binarize all JPEG images in src_img_dir and save results to dst_img_dir.
 
    Args:
        src_img_dir: Source directory containing .jpg images.
        dst_img_dir: Destination directory for binarized images.
        threshold:   Binarization threshold (default 0.5).
    """
    image_paths = sorted(src_img_dir.glob("*.jpg"))
 
    if not image_paths:
        print(f"  [!] No .jpg images found in: {src_img_dir}")
        return
 
    print(f"  Processing {len(image_paths)} images from: {src_img_dir}")
 
    for img_path in image_paths:
        gray_image = load_grayscale(img_path)
        otsu_image = otsu_threshold(gray_image)
        save_path = dst_img_dir / img_path.name
        save_image(otsu_image, save_path)
 
    print(f"  Saved binarized images to: {dst_img_dir}")

def main():
    # ── Paths ──────────────────────────────────────────────────────────────────
    base_dir = Path("Dataset")          
    src_root = base_dir / "SROIE2019"
    dst_root = base_dir / "SROIE Binarize"

 
    # ── Splits to process ─────────────────────────────────────────────────────
    # Only the 'train' split has an img sub-folder in the cloned repo.
    # Add more splits here if you add images later (e.g. "test").
    splits = ["test"]
 
    print("=" * 55)
    print("  SROIE2019 — Binarization Pipeline  ")
    print("=" * 55)


    for split in splits:
        print(f"\n[{split.upper()}]")
        src_img_dir = src_root / split / "img"
        dst_img_dir = dst_root / split / "img"


        process_split(src_img_dir, dst_img_dir)


    print("\n✅ Binarization complete.")
    print(f"   Output dataset: {dst_root}")
 
 
if __name__ == "__main__":
    main()  



 
