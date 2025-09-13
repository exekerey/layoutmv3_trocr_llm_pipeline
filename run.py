import os
import argparse
from utils.pipeline import DocumentPipeline
import json


def main():
    parser = argparse.ArgumentParser(description='Banking Document OCR')
    parser.add_argument('--image', required=True, help='Path to document image')
    parser.add_argument('--lang', default='ru', choices=['ru', 'kz'], help='Language code')
    parser.add_argument('--output', default='output.json', help='Output JSON file')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')

    args = parser.parse_args()

    # Ensure OPENAI_API_KEY is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")

    # Initialize pipeline
    pipeline = DocumentPipeline(
        lang=args.lang,
        llm_api_key=os.environ.get("OPENAI_API_KEY")
    )

    # Process document
    print(f"Processing document: {args.image}")
    result = pipeline.process(args.image)

    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {args.output}")
    print(f"Document type: {result['document_type']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Processing time: {result['processing_times']['total']:.2f}s")


if __name__ == "__main__":
    main()