import argparse
from pathlib import Path
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

LANG_MAP = {"ru": "rus", "kz": "kaz"}

def ocr_image(img, lang):
    return pytesseract.image_to_string(img, lang=lang)

def run_tesseract(images_dir: str, out_text_dir: str, lang_cli="ru"):
    tess_lang = LANG_MAP.get(lang_cli, "rus")
    images_dir, out_text_dir = Path(images_dir), Path(out_text_dir)
    out_text_dir.mkdir(parents=True, exist_ok=True)

    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".pdf":
            pages = convert_from_path(str(p))
            texts = [ocr_image(img, tess_lang) for img in pages]
            (out_text_dir / f"{p.stem}.txt").write_text("\n".join(texts), encoding="utf-8")
            print(f"Tesseract OCR (pdf): {p.name}")
        elif suf in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            text = ocr_image(Image.open(p), tess_lang)
            (out_text_dir / f"{p.stem}.txt").write_text(text, encoding="utf-8")
            print(f"Tesseract OCR (img): {p.name}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--lang", default="ru", choices=["ru","kz"])
    args = ap.parse_args()
    run_tesseract(args.images, args.out_text, args.lang)
