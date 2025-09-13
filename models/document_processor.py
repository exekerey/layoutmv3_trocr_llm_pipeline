from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3ForTokenClassification
)
import torch
from PIL import Image
import numpy as np


class DocumentProcessor:
    def __init__(self, model_name="microsoft/layoutlmv3-base", device=None):
        """
        Initialize LayoutLMv3 for document understanding

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize processor and model for document classification
        self.processor = LayoutLMv3Processor.from_pretrained(model_name)
        self.doc_classifier = LayoutLMv3ForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)

        # Initialize model for token classification (field extraction)
        self.token_classifier = LayoutLMv3ForTokenClassification.from_pretrained(
            model_name
        ).to(self.device)

    def process_document(self, image, ocr_results):
        """
        Process document with LayoutLMv3 to understand structure

        Args:
            image: PIL Image or path to image
            ocr_results: OCR results from PaddleOCR

        Returns:
            Document structure analysis
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Extract words and bounding boxes from OCR results
        words = [item['text'] for item in ocr_results['results']]
        boxes = [item['box'] for item in ocr_results['results']]

        # Normalize boxes to required format (x1, y1, x2, y2)
        normalized_boxes = []
        for box in boxes:
            x_coords = [coord[0] for coord in box]
            y_coords = [coord[1] for coord in box]
            normalized_boxes.append([
                min(x_coords), min(y_coords), max(x_coords), max(y_coords)
            ])

        # Create model inputs
        encoding = self.processor(
            image,
            words,
            boxes=normalized_boxes,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        # Get document structure predictions
        with torch.no_grad():
            outputs = self.doc_classifier(**encoding)

        # Process token classification for field extraction
        with torch.no_grad():
            token_outputs = self.token_classifier(**encoding)
            token_predictions = token_outputs.logits.argmax(-1).squeeze().tolist()

        # Map token predictions to original words
        field_mappings = self._map_tokens_to_fields(words, token_predictions)

        return {
            'document_type': outputs.logits.argmax(-1).item(),
            'confidence': torch.softmax(outputs.logits, dim=-1).max().item(),
            'fields': field_mappings
        }

    def _map_tokens_to_fields(self, words, token_predictions):
        """Map token predictions to document fields"""
        # This is placeholder logic - would need to be customized based on your specific field labels
        field_map = {
            0: "O",  # Outside of any field
            1: "B-HEADER",  # Beginning of header
            2: "I-HEADER",  # Inside of header
            3: "B-DATE",  # Beginning of date
            4: "I-DATE",  # Inside of date
            # Add more field types as needed
        }

        current_field = None
        fields = {}
        current_text = ""

        for word, prediction in zip(words, token_predictions):
            if prediction == 0:  # Outside any field
                if current_field and current_text:
                    if current_field not in fields:
                        fields[current_field] = []
                    fields[current_field].append(current_text.strip())
                    current_text = ""
                    current_field = None
                continue

            label = field_map.get(prediction, "O")
            if label.startswith("B-"):  # Beginning of a new field
                # Save previous field if exists
                if current_field and current_text:
                    if current_field not in fields:
                        fields[current_field] = []
                    fields[current_field].append(current_text.strip())

                # Start new field
                current_field = label[2:]  # Remove 'B-' prefix
                current_text = word
            elif label.startswith("I-"):  # Continuation of a field
                if current_field == label[2:]:  # Make sure it matches current field
                    current_text += " " + word

        # Don't forget the last field
        if current_field and current_text:
            if current_field not in fields:
                fields[current_field] = []
            fields[current_field].append(current_text.strip())

        return fields