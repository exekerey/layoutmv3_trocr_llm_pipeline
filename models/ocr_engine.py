import cv2
import numpy as np
from paddleocr import PaddleOCR


class OCREngine:
    def __init__(self, lang='ru'):
        """
        Initialize OCR engine with specific language support

        Args:
            lang: Language code ('ru', 'kz', etc.)
        """
        self.lang = lang
        self.ocr = PaddleOCR(
            use_angle_cls=True,  # Use angle classifier for rotated text
            lang=lang,  # Language model
            det_db_thresh=0.3,  # Lower threshold for detecting text in noisy images
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,  # Larger value for tighter text boxes
            rec_batch_num=6,  # Batch size for recognition
        )

    def preprocess_image(self, image_path):
        """Apply preprocessing to improve OCR quality on noisy documents""" 
        # Read image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        else:
            img = image_path

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding for better contrast
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    def recognize(self, image_path, preprocess=False):
        """
        Perform OCR on an image and return structured results

        Args:
            image_path: Path to image or image array
            preprocess: Whether to apply preprocessing

        Returns:
            Dictionary with OCR results
        """
        if preprocess:
            img = self.preprocess_image(image_path)
        else:
            if isinstance(image_path, str):
                img = cv2.imread(image_path)
            else:
                img = image_path

        # Run OCR
        results = self.ocr.ocr(img)
        
        structured_results = []
        if results and results[0]:
            res_dict = results[0]
            
            boxes = res_dict.get('dt_polys', [])
            texts = res_dict.get('rec_texts', [])
            scores = res_dict.get('rec_scores', [])

            for box, text, score in zip(boxes, texts, scores):
                structured_results.append({
                    'text': text,
                    'confidence': float(score),
                    'box': box.tolist(), # convert numpy array to list
                    'page': 0 # Assuming single page
                })

        return {
            'results': structured_results,
            'raw_text': ' '.join([r['text'] for r in structured_results])
        }
