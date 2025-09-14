#!/usr/bin/env bash
set -euo pipefail

set -a
[ -f .env ] && . .env
set +a

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYBIN="${PYBIN:-/opt/homebrew/opt/python@3.11/bin/python3.11}"
VENVDIR="${VENVDIR:-.venv}"

if ! "$PYBIN" -V 2>/dev/null | grep -q "Python 3\.11"; then
  echo "PYBIN must point to Python 3.11"; exit 1
fi

if [[ -d "$VENVDIR" ]]; then
  VENV_PY="$VENVDIR/bin/python"
  if [[ -x "$VENV_PY" ]]; then
    if ! "$VENV_PY" -V | grep -q "Python 3\.11"; then
      rm -rf "$VENVDIR"
    fi
  else
    rm -rf "$VENVDIR"
  fi
fi

if [[ ! -d "$VENVDIR" ]]; then
  "$PYBIN" -m venv "$VENVDIR"
fi
. "$VENVDIR/bin/activate"

python -V

pip install -U pip wheel setuptools
pip uninstall -y paddlex paddlenlp paddlespeech || true
pip uninstall -y paddleocr paddlepaddle || true
pip install --no-cache-dir paddlepaddle==2.6.1 paddleocr==2.6.1.3
pip install --no-cache-dir opencv-python==4.10.0.84 pdf2image==1.17.0 pillow==10.4.0 numpy==1.26.4
pip install --no-cache-dir transformers==4.44.2 huggingface-hub==0.24.6 torch==2.3.1 torchvision==0.18.1
pip install --no-cache-dir jiwer==4.0.0 python-Levenshtein==0.25.1 pytesseract==0.3.13

: "${HF_HOME:=$HOME/.cache/huggingface}"
: "${TRANSFORMERS_CACHE:=$HF_HOME/transformers}"
: "${TOKENIZERS_PARALLELISM:=false}"
: "${OMP_NUM_THREADS:=1}"
export HF_HOME TRANSFORMERS_CACHE TOKENIZERS_PARALLELISM OMP_NUM_THREADS

if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
  pip install -r "${PROJECT_ROOT}/requirements.txt"
fi

: "${PYTHONPATH:=}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

LANG_CODE="${LANG_CODE:-ru}"
IMAGES_DIR="${IMAGES_DIR:-data}"
GT_OCR_DIR="${GT_OCR_DIR:-data/gt/ocr_text}"
GT_FIELDS_DIR="${GT_FIELDS_DIR:-data/gt/fields_json}"
PRED_TEXT_DIR="${PRED_TEXT_DIR:-data/preds/ocr_text}"
PRED_JSON_DIR="${PRED_JSON_DIR:-data/preds/fields_json}"
PRED_NOISY_TEXT_DIR="${PRED_NOISY_TEXT_DIR:-data/preds_noisy/ocr_text}"
PRED_NOISY_JSON_DIR="${PRED_NOISY_JSON_DIR:-data/preds_noisy/fields_json}"
TESS_TEXT_DIR="${TESS_TEXT_DIR:-data/tesseract/ocr_text}"
TESS_NOISY_TEXT_DIR="${TESS_NOISY_TEXT_DIR:-data/tesseract_noisy/ocr_text}"
MAKE_NOISY="${MAKE_NOISY:-false}"
REPORT_JSON="${REPORT_JSON:-evaluation_report.json}"

mkdir -p "${PRED_TEXT_DIR}" "${PRED_JSON_DIR}" \
         "${PRED_NOISY_TEXT_DIR}" "${PRED_NOISY_JSON_DIR}" \
         "${TESS_TEXT_DIR}" "${TESS_NOISY_TEXT_DIR}"

NOISY_DIR="${PROJECT_ROOT}/data/samples_noisy"
if [[ "${MAKE_NOISY}" == "true" ]]; then
  python "${PROJECT_ROOT}/tools/make_noisy.py" --src "${IMAGES_DIR}" --dst "${NOISY_DIR}" || true
fi

python "${PROJECT_ROOT}/tools/batch_predict.py" \
  --images "${IMAGES_DIR}" \
  --out-text "${PRED_TEXT_DIR}" \
  --out-json "${PRED_JSON_DIR}" \
  --lang "${LANG_CODE}"

if [[ "${MAKE_NOISY}" == "true" && -d "${NOISY_DIR}" ]]; then
  python "${PROJECT_ROOT}/tools/batch_predict.py" \
    --images "${NOISY_DIR}" \
    --out-text "${PRED_NOISY_TEXT_DIR}" \
    --out-json "${PRED_NOISY_JSON_DIR}" \
    --lang "${LANG_CODE}"
fi

python "${PROJECT_ROOT}/tools/run_tesseract.py" \
  --images "${IMAGES_DIR}" \
  --out-text "${TESS_TEXT_DIR}" \
  --lang "${LANG_CODE}" || true

if [[ "${MAKE_NOISY}" == "true" && -d "${NOISY_DIR}" ]]; then
  python "${PROJECT_ROOT}/tools/run_tesseract.py" \
    --images "${NOISY_DIR}" \
    --out-text "${TESS_NOISY_TEXT_DIR}" \
    --lang "${LANG_CODE}" || true
fi

python "${PROJECT_ROOT}/tools/metrics.py" \
  --gt-ocr "${GT_OCR_DIR}" \
  --pred-ocr "${PRED_TEXT_DIR}" \
  --gt-fields "${GT_FIELDS_DIR}" \
  --pred-fields "${PRED_JSON_DIR}" | tee normal_metrics.json

if [[ "${MAKE_NOISY}" == "true" && -d "${NOISY_DIR}" ]]; then
  python "${PROJECT_ROOT}/tools/metrics.py" \
    --gt-ocr "${GT_OCR_DIR}" \
    --pred-ocr "${PRED_NOISY_TEXT_DIR}" \
    --gt-fields "${GT_FIELDS_DIR}" \
    --pred-fields "${PRED_NOISY_JSON_DIR}" | tee noisy_metrics.json
fi

python "${PROJECT_ROOT}/tools/compare_with_tesseract.py" | tee compare_ocr.txt

python - "${REPORT_JSON}" <<'PY'
import json, sys, os
def read_json(p):
    try:
        with open(p, 'r', encoding='utf-8') as f: return json.load(f)
    except: return {}
report = {}
if os.path.exists("normal_metrics.json"): report["our_normal"]=read_json("normal_metrics.json")
if os.path.exists("noisy_metrics.json"): report["our_noisy"]=read_json("noisy_metrics.json")
print(json.dumps(report, ensure_ascii=False, indent=2))
open(sys.argv[1],"w",encoding="utf-8").write(json.dumps(report, ensure_ascii=False, indent=2))
PY
