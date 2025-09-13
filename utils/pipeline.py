import time

import cv2
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
            image_path: Path to document image

        Returns:
            Processed document information as JSON
        """
        start_time = time.time()

        # Step 1: Run OCR
        ocr_start = time.time()
        ocr_results = self.ocr_engine.recognize(image_path)
        ocr_time = time.time() - ocr_start

        # Step 2: Process with Vision Transformer
        vt_start = time.time()
        image = Image.open(image_path).convert("RGB")
        document_analysis = self.document_processor.process_document(image, ocr_results)
        vt_time = time.time() - vt_start

        # Step 3: Process with LLM
        llm_start = time.time()
        document_type = document_analysis['document_type']
        fields = document_analysis['fields']
        llm_results = self.llm_processor.process_document(
            ocr_results['raw_text'],
            document_type,
            fields
        )
        llm_time = time.time() - llm_start

        # Combine results
        result = {
            "document_type": self._get_document_type_name(document_type),
            "extracted_data": llm_results.get("data", {}),
            "confidence": document_analysis['confidence'],
            "processing_times": {
                "ocr": ocr_time,
                "vision_transformer": vt_time,
                "llm": llm_time,
                "total": time.time() - start_time
            }
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
