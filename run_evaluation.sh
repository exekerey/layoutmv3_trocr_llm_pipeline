set -euo pipefail

set -a
[ -f .env ] && . .env
set +a

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "${VENVDIR}" ]]; then
  "${PYBIN}" -m venv "${VENVDIR}"
fi
. "${VENVDIR}/bin/activate"

pip install -U pip wheel
pip install paddlepaddle==3.2.0 paddleocr
if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
  pip install -r "${PROJECT_ROOT}/requirements.txt"
fi
pip install jiwer python-Levenshtein pytesseract pdf2image pillow opencv-python

: "${PYTHONPATH:=}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

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
