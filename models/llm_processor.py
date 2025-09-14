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
        print(ocr_text, "\n\n", fields)
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[  # noqa
                {"role": "system",
                 "content": self._system_prompt()},
                {"role": "user", "content": self._message_prompt(ocr_text, fields)}
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

    def _message_prompt(self, ocr_text, fields=None):
        return f"""
        Extract key information from this document.
        Determine the document type first (receipt, contract, statement, or other).
        Then extract all relevant financial and personal information present.

        Here is the OCR text:
        {ocr_text}

        Additional fields detected by computer vision:
        {fields if fields else 'None'}

        Format the output as a JSON object with relevant fields. If a field is not present, set its value to null.
        """

    def _system_prompt(self):
        return f"""
        You are a document processing assistant specialized in contract documents. 
        Extract key information and format it as JSON.
        
        Determine the following fields in the provided document and provide them as JSON slugs.
        - contract_number - unique identifier of the contract.
        - contract_date - date when contract is signed or executed.
        - contract_expiration_date - date when contract expires.
        - counterparty_name - name of the counterparty company. Provide it as in the document.
        - counterparty_country - composite string out of country code of the counterparty company in ISO 3166-1 alpha-2 format and country full name in Russian separated by dot.
        - contract_sum - sum of the contract(only numbers).
        - contract_sum_currency - currency code of the contract.
        - contract_payment_currency - payment currency code of the contract.
        
        In case when a field is not present in a document and it's not possible to deduct it, set the field's value to null.
        The OCR may make mistakes, validate the values in the extracted fields and correct them as needed.
        
        Example output:
        {{
            "contract_number": "SM-1712/22",
            "contract_date": "2021-09-22",
            "contract_expiration_date": "2022-12-01",
            "counterparty_name": "ТОО Mas Shelby",
            "counterparty_country": "BY.Беларусь",
            "contract_sum": 3209315.71,
            "contract_sum_currency": "RUB",
            "contract_payment_currency": "RUB",
        }}
        
        """


fields_mapping = {
    "№ онтракта": "contract_number",
    "дата заключения (дата заключения контракта)": "contract_initiation_date",
    "дата окончания (срок действия контракта, договора)": "contract_end_date",
    "контрагент (наименование иностранного контрагента)": "counterparty",
    "страна (страна контрагента (инопартнера));": "counterparty_country",
    "сумма контракта;": "contract_sum",
    "валюта контракта;": "contract_sum_currency",
    "валюта платежа.": "payment_currency",
}
