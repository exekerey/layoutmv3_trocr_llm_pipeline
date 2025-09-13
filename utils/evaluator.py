import os
import json
import numpy as np
from rapidfuzz.distance import Levenshtein
import pandas as pd


class OCREvaluator:
    def __init__(self):
        """Initialize evaluator for OCR and document extraction"""
        pass

    def calculate_cer(self, reference, hypothesis):
        """
        Calculate Character Error Rate

        Args:
            reference: Ground truth text
            hypothesis: OCR result text

        Returns:
            Character Error Rate
        """
        if len(reference) == 0:
            return 1.0 if len(hypothesis) > 0 else 0.0

        edit_distance = Levenshtein.distance(reference, hypothesis)
        return edit_distance / len(reference)

    def calculate_wer(self, reference, hypothesis):
        """
        Calculate Word Error Rate

        Args:
            reference: Ground truth text
            hypothesis: OCR result text

        Returns:
            Word Error Rate
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        if len(ref_words) == 0:
            return 1.0 if len(hyp_words) > 0 else 0.0

        # Calculate word-level edit distance
        m, n = len(ref_words), len(hyp_words)
        d = np.zeros((m + 1, n + 1), dtype=np.int32)

        for i in range(m + 1):
            d[i, 0] = i
        for j in range(n + 1):
            d[0, j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i, j] = d[i - 1, j - 1]
                else:
                    d[i, j] = min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1]) + 1

        return d[m, n] / m

    def evaluate_field_extraction(self, ground_truth, extracted):
        """
        Evaluate field extraction performance

        Args:
            ground_truth: Dictionary with ground truth fields
            extracted: Dictionary with extracted fields

        Returns:
            Dictionary with precision, recall, f1 for each field
        """
        results = {}

        # Get all field names
        all_fields = set(ground_truth.keys()) | set(extracted.keys())

        for field in all_fields:
            gt_value = ground_truth.get(field, "")
            ex_value = extracted.get(field, "")

            # Calculate field-level metrics
            if gt_value and ex_value:
                # Field is present in both
                if isinstance(gt_value, list) and isinstance(ex_value, list):
                    # Handle list values (like transactions)
                    tp = len(set(gt_value) & set(ex_value))
                    fp = len(set(ex_value) - set(gt_value))
                    fn = len(set(gt_value) - set(ex_value))
                else:
                    # Handle string values
                    cer = self.calculate_cer(str(gt_value), str(ex_value))
                    # Consider a match if CER is below threshold
                    match = cer < 0.2
                    tp = 1 if match else 0
                    fp = 0 if match else 1
                    fn = 0 if match else 1
            elif not gt_value and not ex_value:
                # Field is absent in both
                tp, fp, fn = 0, 0, 0
            elif gt_value and not ex_value:
                # Field is in ground truth but not extracted
                tp, fp, fn = 0, 0, 1
            else:
                # Field is extracted but not in ground truth
                tp, fp, fn = 0, 1, 0

            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results[field] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "match": tp > 0 and fp == 0 and fn == 0
            }

        # Calculate overall metrics
        matches = sum(1 for field in results if results[field]["match"])
        exact_match = matches == len(all_fields)

        return {
            "field_metrics": results,
            "overall": {
                "precision": sum(results[f]["precision"] for f in results) / len(results) if results else 0,
                "recall": sum(results[f]["recall"] for f in results) / len(results) if results else 0,
                "f1": sum(results[f]["f1"] for f in results) / len(results) if results else 0,
                "exact_match": exact_match
            }
        }

    def validate_json(self, json_data):
        """
        Validate JSON structure

        Args:
            json_data: JSON data to validate

        Returns:
            Dictionary with validation results
        """
        # Check if it's a valid JSON
        if not isinstance(json_data, dict):
            return {
                "valid": False,
                "schema_consistency": 0.0,
                "error": "Not a valid JSON object"
            }

        # Define required fields based on document type
        required_fields = {
            "receipt": ["transaction_date", "amount", "recipient", "transaction_id"],
            "contract": ["contract_number", "contract_date", "client_name", "bank_name"],
            "statement": ["statement_period", "account_number", "opening_balance", "closing_balance"]
        }

        # Determine document type
        doc_type = json_data.get("document_type", "unknown").lower()
        if doc_type not in required_fields:
            doc_type = "unknown"

        if doc_type == "unknown":
            return {
                "valid": True,
                "schema_consistency": 1.0,
                "message": "Document type unknown, can't validate schema"
            }

        # Check required fields
        fields_present = 0
        for field in required_fields[doc_type]:
            if field in json_data and json_data[field] is not None:
                fields_present += 1

        schema_consistency = fields_present / len(required_fields[doc_type])

        return {
            "valid": True,
            "schema_consistency": schema_consistency,
            "fields_present": fields_present,
            "fields_required": len(required_fields[doc_type])
        }