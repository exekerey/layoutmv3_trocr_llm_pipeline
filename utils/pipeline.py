import time

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from models.document_processor import DocumentProcessor
from models.llm_processor import LLMProcessor
from models.ocr_engine import OCREngine


class DocumentPipeline:
    def __init__(self, lang='ru', llm_api_key=None):
        """
        Initialize the full document processing pipeline

        Args:
            lang: Language for OCR
            use_gpu: Whether to use GPU
            llm_api_key: API key for OpenAI
        """
        self.ocr_engine = OCREngine(lang=lang)
        self.document_processor = DocumentProcessor()
        self.llm_processor = LLMProcessor(api_key=llm_api_key)

    def process(self, image_path):
        """
        Process a document through the entire pipeline

        Args:
            image_path: Path to document image or PDF

        Returns:
            Processed document information as JSON
        """
        start_time = time.time()

        if image_path.lower().endswith('.pdf'):
            results = []
            with fitz.open(image_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap()

                    if pix.n == 1:
                        img_rgb = cv2.cvtColor(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 1),
                                               cv2.COLOR_GRAY2RGB)
                    elif pix.n == 4:
                        img_rgb = cv2.cvtColor(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 4),
                                               cv2.COLOR_RGBA2RGB)
                    else:
                        img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)

                    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    image_for_pil = Image.fromarray(img_rgb)

                    # Step 1: Run OCR
                    ocr_start = time.time()
                    ocr_results = self.ocr_engine.recognize(img)
                    ocr_time = time.time() - ocr_start

                    # Step 2: Process with Vision Transformer
                    vt_start = time.time()
                    document_analysis = self.document_processor.process_document(image_for_pil, ocr_results)
                    print(document_analysis)

                    vt_time = time.time() - vt_start

                    results.append({
                        "page_num": page_num + 1,
                        "ocr_results": ocr_results,
                        "document_analysis": document_analysis,
                        "processing_times": {
                            "ocr": ocr_time,
                            "vision_transformer": vt_time
                        }
                    })
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Image at {image_path} could not be loaded.")
            image_for_pil = Image.open(image_path).convert("RGB")

            # Step 1: Run OCR
            ocr_start = time.time()
            ocr_results = self.ocr_engine.recognize(img)
            ocr_time = time.time() - ocr_start

            # Step 2: Process with Vision Transformer
            vt_start = time.time()
            document_analysis = self.document_processor.process_document(image_for_pil, ocr_results)
            vt_time = time.time() - vt_start
            results = [{
                "page_num": 1,
                "ocr_results": ocr_results,
                "document_analysis": document_analysis,
                "processing_times": {
                    "ocr": ocr_time,
                    "vision_transformer": vt_time
                }
            }]

        # Step 3: Process with LLM
        llm_start = time.time()

        # Combine text from all pages for the LLM
        full_text = "\n".join([res['ocr_results']['raw_text'] for res in results])

        # For simplicity, we'll use the analysis of the first page for document type and fields
        # A more advanced approach could involve a voting mechanism or other heuristics
        first_page_analysis = results[0]['document_analysis']
        document_type = first_page_analysis['document_type']
        fields = first_page_analysis['fields']

        llm_results = self.llm_processor.process_document(
            full_text,
            document_type,
            fields
        )
        llm_time = time.time() - llm_start

        # Combine results
        total_ocr_time = sum(res['processing_times']['ocr'] for res in results)
        total_vt_time = sum(res['processing_times']['vision_transformer'] for res in results)

        result = {
            "document_type": self._get_document_type_name(document_type),
            "extracted_data": llm_results.get("data", {}),
            "confidence": first_page_analysis['confidence'],  # Using first page confidence
            "processing_times": {
                "ocr": total_ocr_time,
                "vision_transformer": total_vt_time,
                "llm": llm_time,
                "total": time.time() - start_time
            },
            "pages": len(results)
        }

        return result

    def _get_document_type_name(self, type_id):
        types = {
            0: "receipt",
            1: "contract",
            2: "statement",
            3: "other"
        }
        return types.get(type_id, "unknown")
