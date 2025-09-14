import cv2
import numpy as np
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from pathlib import Path


class OCREngine:
    def __init__(self, lang='ru'):
        self.lang = lang
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            rec_batch_num=6,
        )

    def preprocess_image(self, image_path):
        if isinstance(image_path, str):
            path = Path(image_path)
            if path.suffix.lower() == ".pdf":
                pages = convert_from_path(str(path))
                if not pages:
                    raise ValueError(f"PDF пустой: {image_path}")
                img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
            else:
                img = cv2.imread(str(path))
                if img is None:
                    raise ValueError(f"Не удалось открыть файл: {image_path}")
        else:
            img = image_path

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        return denoised

    def recognize(self, image_path, preprocess=True):
        if preprocess:
            img = self.preprocess_image(image_path)
        else:
            if isinstance(image_path, str):
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Не удалось открыть файл: {image_path}")
            else:
                img = image_path

        results = self.ocr.ocr(img, cls=True)

        structured_results = []
        for idx, res in enumerate(results):
            if res:
                for line in res:
                    box, (text, confidence) = line
                    structured_results.append({
                        'text': text,
                        'confidence': float(confidence),
                        'box': box,
                        'page': idx
                    })

        return {
            'results': structured_results,
            'raw_text': ' '.join([r['text'] for r in structured_results])
        }
