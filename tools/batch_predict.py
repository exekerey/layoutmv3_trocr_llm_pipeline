import os, sys, json
from pathlib import Path

from utils.pipeline import DocumentPipeline

def run_batch(images_dir, out_text_dir, out_json_dir, lang="ru"):
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    Path(out_text_dir).mkdir(parents=True, exist_ok=True)
    Path(out_json_dir).mkdir(parents=True, exist_ok=True)

    pipe = DocumentPipeline(lang=lang, llm_api_key=os.environ.get("OPENAI_API_KEY"))

    for p in Path(images_dir).glob("*.pdf"):
        res = pipe.process(str(p))
        (Path(out_text_dir)/(p.stem+".txt")).write_text(res.get("raw_text",""), encoding="utf-8")
        (Path(out_json_dir)/(p.stem+".json")).write_text(
            json.dumps(res.get("fields",{}), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Processed:", p.name)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--lang", default="ru")
    args = ap.parse_args()
    run_batch(args.images, args.out_text, args.out_json, args.lang)
