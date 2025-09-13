# Banking Document OCR

Advanced OCR pipeline for banking documents that exceeds Tesseract capabilities, with support for Russian and Kazakh languages.

## Features

- High-accuracy OCR with PaddleOCR
- Document structure understanding with LayoutLMv3
- Advanced field extraction and validation with LLM
- Support for receipts, contracts, and bank statements
- Noise handling for poor-quality scans
- Structured JSON output

## Installation

1. Clone the repository:
```bash
git clone https://github.com/exekerey/banking-ocr.git
cd banking-ocr
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# For OpenAI API
export OPENAI_API_KEY=your_api_key_here
```

## Usage

### Command-line Interface

```bash
python run.py --image path/to/document.jpg --lang ru --output results.json
```

Options:
- `--image`: Path to document image (required)
- `--lang`: Language code ('ru' for Russian, 'kz' for Kazakh)
- `--output`: Output JSON file
- `--no-gpu`: Disable GPU acceleration

### Web Interface

Start the Streamlit app:

```bash
cd app
streamlit run streamlit_app.py
```

Then open your browser at http://localhost:8501

## Evaluation

To evaluate on a test dataset:

```bash
python utils/evaluator.py --test-dir path/to/test/data --gt-file ground_truth.json
```

## Architecture

The pipeline consists of three main components:

1. **OCR Engine (PaddleOCR)**
   - Detects and recognizes text from images
   - Supports Russian and Kazakh languages
   - Includes preprocessing for noisy documents

2. **Document Structure Analysis (LayoutLMv3)**
   - Understands document layout and structure
   - Classifies document type
   - Extracts key fields based on position and context

3. **LLM Post-processor**
   - Validates and corrects OCR output
   - Structures data into standardized JSON
   - Handles edge cases and ambiguities

## Performance Metrics

The system is evaluated based on:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Field-level Accuracy (precision, recall, F1)
- JSON Structure Validity

## License

This project is open source under the MIT license.