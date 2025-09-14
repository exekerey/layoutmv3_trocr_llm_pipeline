import json
import os

from dotenv import load_dotenv
from openai import OpenAI


class LLMProcessor:
    def __init__(self, api_key=None, model="gpt-4o"):
        """
        Initialize LLM processor for OCR post-processing

        Args:
            api_key: OpenAI API key
            model: Model to use
        """
        if api_key is None:
            load_dotenv()
            api_key = os.environ.get("OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def process_document(self, ocr_text, document_type, fields=None):
        """
        Process OCR text with LLM to extract and validate document info

        Args:
            ocr_text: Raw OCR text
            document_type: Type of document
            fields: Pre-extracted fields from Vision Transformer

        Returns:
            Structured JSON with extracted information
        """
        # Create prompt based on document type
        # if document_type == 0:  # Receipt
        #     prompt = self._create_receipt_prompt(ocr_text, fields)
        # elif document_type == 1:  # Contract
        #     prompt = self._create_contract_prompt(ocr_text, fields)
        # elif document_type == 2:  # Statement
        #     prompt = self._create_statement_prompt(ocr_text, fields)
        # else:
        prompt = self._create_general_prompt(ocr_text, fields)

        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system",
                 "content": "You are a document processing assistant specialized in banking documents. Extract key information and format it as JSON."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract JSON from response
        try:
            extracted_data = json.loads(response.choices[0].message.content)
            return {
                "success": True,
                "data": extracted_data
            }
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Failed to parse JSON response",
                "raw_response": response.choices[0].message.content
            }

    def _create_receipt_prompt(self, ocr_text, fields=None):
        return f"""
        Extract the following information from this bank receipt:
        - Transaction date and time
        - Amount
        - Recipient name/account
        - Sender name/account
        - Transaction ID/reference
        - Bank name
        - Any fees

        Here is the OCR text extracted from the receipt:
        {ocr_text}

        Additional fields detected by computer vision:
        {fields if fields else 'None'}

        Format the output as a JSON object with these fields. If a field is not present, set its value to null.
        """

    def _create_contract_prompt(self, ocr_text, fields=None):
        return f"""
        Extract the following information from this bank contract:
        - Contract number
        - Contract date
        - Client name
        - Client ID number
        - Bank name
        - Contract type
        - Key terms (interest rate, duration, etc.)
        - Signatures present (yes/no)

        Here is the OCR text extracted from the contract:
        {ocr_text}

        Additional fields detected by computer vision:
        {fields if fields else 'None'}

        Format the output as a JSON object with these fields. If a field is not present, set its value to null.
        """

    def _create_statement_prompt(self, ocr_text, fields=None):
        return f"""
        Extract the following information from this bank statement:
        - Statement period (from-to dates)
        - Account number
        - Account holder name
        - Opening balance
        - Closing balance
        - Total deposits
        - Total withdrawals
        - List of transactions (if present)

        Here is the OCR text extracted from the statement:
        {ocr_text}

        Additional fields detected by computer vision:
        {fields if fields else 'None'}

        Format the output as a JSON object with these fields. If a field is not present, set its value to null.
        """

    def _create_general_prompt(self, ocr_text, fields=None):
        return f"""
        Extract key information from this document.
        Determine the document type first (receipt, contract, statement, or other).
        Then extract all relevant financial and personal information present.
        Validate the values in the extracted fields and correct as needed.

        Here is the OCR text:
        {ocr_text}

        Additional fields detected by computer vision:
        {fields if fields else 'None'}

        Format the output as a JSON object with relevant fields. If a field is not present, set its value to null.
        """
