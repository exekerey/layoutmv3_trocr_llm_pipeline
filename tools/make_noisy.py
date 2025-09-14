import argparse
from pathlib import Path
import cv2, numpy as np
from pdf2image import convert_from_path

def add_noise(img):
    gauss = np.random.normal(0, 12, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + gauss, 0, 255).astype(np.uint8)

def add_blur(img, k=5):
    return cv2.GaussianBlur(img, (k, k), 0)

def adjust_contrast(img, alpha=0.85, beta=8):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def rotate_small(img, angle=3):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def make_noisy_file(src_path: Path, dst_dir: Path):
    if src_path.suffix.lower() == ".pdf":
        pages = convert_from_path(str(src_path))
        for i, pil_im in enumerate(pages, 1):
            im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
            im = add_blur(im, 5)
            im = add_noise(im)
            im = adjust_contrast(im, 0.8, 10)
            im = rotate_small(im, np.random.choice([-4, -3, 3, 4]))
            out = dst_dir / f"{src_path.stem}_p{i}.png"
            cv2.imwrite(str(out), im)
    else:
        im = cv2.imread(str(src_path))
        if im is None:
            return
        im = add_blur(im, 5)
        im = add_noise(im)
        im = adjust_contrast(im, 0.8, 10)
        im = rotate_small(im, np.random.choice([-4, -3, 3, 4]))
        out = dst_dir / f"{src_path.stem}.png"
        cv2.imwrite(str(out), im)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        make_noisy_file(p, dst)

if __name__ == "__main__":
    main()
