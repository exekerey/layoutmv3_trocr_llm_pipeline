import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def map_files(data_folder):
    pdf_files = sorted(list(Path(data_folder).glob("*.pdf")))
    json_files = sorted(list(Path(data_folder).glob("*.json")))

    file_map = {}
    for pdf_path in pdf_files:
        pdf_name_latin = pdf_path.stem.replace('A', 'А').replace('C', 'С')  # Handle A/А and C/С differences

        # Try to find a matching JSON file
        matched_json = None
        for json_path in json_files:
            json_stem_normalized = json_path.stem.replace('A', 'А').replace('C', 'С')
            if pdf_name_latin == json_stem_normalized:
                matched_json = json_path
                break

        if matched_json:
            file_map[str(pdf_path)] = str(matched_json)
        else:
            print(f"Warning: No matching JSON found for PDF: {pdf_path.name}")
    return file_map


def evaluate_with_llm(reference_json, generated_json, client):
    prompt = f"""
You are an expert in document processing and evaluation. Your task is to compare two JSON documents: a reference JSON and a generated JSON.
The goal is to determine how many fields match between the two documents and if the generated document is a perfect match to the reference.

Here are the rules for matching:
1.  **Contract Number and Counterparty Name:** Must be an exact character-by-character match.
2.  **Dates:** Should match if they represent the same date, even if their formats are different (e.g., "2023-01-15" matches "15.01.2023"). If formats are the same but dates are different, they do not match.
3.  **Numbers (including sums):** Should match if they represent the same numerical value, even if their formats are different (e.g., "1,000.50" matches "1000.50" or "1 000,50"). If formats are the same but numbers are different, they do not match.
4.  **Other Fields:** Should match if their content is semantically similar or identical. Minor variations in whitespace or punctuation should be ignored if the core meaning is the same.

Provide your response in a JSON format with two keys:
- `matched_fields_count`: An integer representing the number of fields that match according to the rules above.
- `is_perfect_match`: A boolean (true/false) indicating if all fields in the reference JSON have a corresponding match in the generated JSON, and no extra incorrect fields are present in the generated JSON.

Reference JSON:
```json
{reference_json}
```

Generated JSON:
```json
{generated_json}
```
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Or another suitable model like "gpt-3.5-turbo"
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant that evaluates JSON documents."},
                {"role": "user", "content": prompt}
            ]
        )
        evaluation_result = json.loads(response.choices[0].message.content)
        return evaluation_result
    except Exception as e:
        print(f"Error during LLM evaluation: {e}")
        return {"matched_fields_count": 0, "is_perfect_match": False}


def main():
    parser = argparse.ArgumentParser(description='Evaluate document processing pipeline.')
    parser.add_argument('--data_folder', default='./data',
                        help='Path to the data folder containing PDFs and reference JSONs.')
    parser.add_argument('--output_dir', default='./evaluation_output',
                        help='Directory to save generated JSONs and evaluation results.')
    parser.add_argument('--run_script', default='./run.py', help='Path to the run.py script.')

    args = parser.parse_args()

    data_folder = Path(args.data_folder).resolve()
    output_dir = Path(args.output_dir).resolve()
    run_script = Path(args.run_script).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set. Please set it before running the script.")
        return

    client = OpenAI()  # Initialize OpenAI client

    file_map = map_files(data_folder)

    total_documents = len(file_map)
    perfect_matches_count = 0
    all_eval_results = []

    print(f"Starting evaluation for {total_documents} documents...")

    for pdf_path_str, ref_json_path_str in file_map.items():
        pdf_path = Path(pdf_path_str)
        ref_json_path = Path(ref_json_path_str)

        print(f"Processing {pdf_path.name}...")

        # Define output path for the generated JSON
        generated_json_output_path = output_dir / f"generated_{pdf_path.stem}.json"

        # Run the run.py script
        print(f"Running {run_script.name} for {pdf_path.name}...")
        command = f"python3 {run_script} --image {pdf_path} --output {generated_json_output_path}"

        # Execute the command
        # Note: This assumes run.py handles its own output/errors.
        # For a robust solution, you might want to capture stdout/stderr.
        os.system(command)

        if not generated_json_output_path.exists():
            print(
                f"Error: {generated_json_output_path.name} was not generated by run.py. Skipping evaluation for this document.")
            continue

        # Read reference and generated JSONs
        try:
            with open(ref_json_path, 'r', encoding='utf-8') as f:
                reference_data = json.load(f)
            with open(generated_json_output_path, 'r', encoding='utf-8') as f:
                generated_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for {pdf_path.name}: {e}. Skipping evaluation.")
            continue
        except FileNotFoundError:
            print(
                f"Error: Reference JSON {ref_json_path.name} or generated JSON {generated_json_output_path.name} not found. Skipping evaluation.")
            continue

        # Perform LLM evaluation
        print(f"Evaluating output for {pdf_path.name} with LLM...")
        eval_result = evaluate_with_llm(json.dumps(reference_data, ensure_ascii=False, indent=2),
                                        json.dumps(generated_data, ensure_ascii=False, indent=2),
                                        client)

        all_eval_results.append({
            "pdf_file": pdf_path.name,
            "reference_json": ref_json_path.name,
            "generated_json": generated_json_output_path.name,
            "matched_fields_count": eval_result.get("matched_fields_count", 0),
            "is_perfect_match": eval_result.get("is_perfect_match", False)
        })

        if eval_result.get("is_perfect_match"):
            perfect_matches_count += 1

        print(
            f"  Matched fields: {eval_result.get('matched_fields_count', 0)}, Perfect match: {eval_result.get('is_perfect_match', False)}")

    # Save all evaluation results to a summary file
    summary_file_path = output_dir / "evaluation_summary.json"
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_eval_results, f, ensure_ascii=False, indent=2)

    print("--- Evaluation Summary ---")
    print(f"Total documents processed: {total_documents}")
    print(
        f"Documents with perfect matches: {perfect_matches_count}/{total_documents} ({perfect_matches_count / total_documents:.2%})")
    print(f"Detailed results saved to: {summary_file_path}")


if __name__ == "__main__":
    main()
