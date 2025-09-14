# tools/metrics.py
from pathlib import Path
import json
import Levenshtein as lev
from jiwer import wer

# ---- OCR METRICS ----
def cer(ref: str, hyp: str) -> float:
    ref, hyp = ref or "", hyp or ""
    if not ref and not hyp: return 0.0
    if not ref: return 1.0
    return lev.distance(ref, hyp) / len(ref)

def nld(ref: str, hyp: str) -> float:
    ref, hyp = ref or "", hyp or ""
    denom = max(len(ref), len(hyp), 1)
    return lev.distance(ref, hyp) / denom

def eval_ocr_text_dir(gt_dir: str, pred_dir: str) -> dict:
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    cer_sum = wer_sum = nld_sum = 0.0
    n = 0
    for gt in gt_dir.glob("*.txt"):
        pr = pred_dir / gt.name
        if not pr.exists(): continue
        ref = gt.read_text(encoding="utf-8", errors="ignore")
        hyp = pr.read_text(encoding="utf-8", errors="ignore")
        cer_sum += cer(ref, hyp)
        wer_sum += wer(ref, hyp)
        nld_sum += nld(ref, hyp)
        n += 1
    n = max(n, 1)
    return {
        "Documents": n,
        "CER": cer_sum/n,
        "WER": wer_sum/n,
        "NormalizedLevenshtein": nld_sum/n
    }

# ---- FIELD METRICS ----
REQUIRED = ["Дата","Сумма","IBAN"]
ALL_KEYS = ["Дата","Сумма","Валюта","IBAN","Отправитель","Получатель"]

def _load_json(p: Path) -> dict:
    try: return json.loads(p.read_text(encoding="utf-8"))
    except: return {}

def _eq(a,b) -> bool:
    return (str(a or "").strip() == str(b or "").strip())

def eval_fields_dir(gt_dir: str, pred_dir: str,
                    keys=ALL_KEYS, required=REQUIRED) -> dict:
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    counts = {k: {"TP":0,"FP":0,"FN":0,"TOTAL":0,"ACC_OK":0} for k in keys}
    exact_ok = 0
    total = 0

    for g in gt_dir.glob("*.json"):
        p = pred_dir / g.name
        if not p.exists(): continue
        gt = _load_json(g); pr = _load_json(p)
        total += 1
        all_ok = True
        for k in keys:
            counts[k]["TOTAL"] += 1
            gt_has = k in gt and gt[k] is not None
            pr_has = k in pr and pr[k] is not None and str(pr[k]).strip() != ""
            if gt_has:
                if pr_has and _eq(gt[k], pr[k]):
                    counts[k]["TP"] += 1; counts[k]["ACC_OK"] += 1
                elif pr_has and not _eq(gt[k], pr[k]):
                    counts[k]["FP"] += 1
                    if k in required: all_ok = False
                else:
                    counts[k]["FN"] += 1
                    if k in required: all_ok = False
        if all_ok: exact_ok += 1

    def _f1(TP, FP, FN):
        P = TP/(TP+FP) if (TP+FP) else 0.0
        R = TP/(TP+FN) if (TP+FN) else 0.0
        return 2*P*R/(P+R) if (P+R) else 0.0

    field_acc = {k: round((c["ACC_OK"]/c["TOTAL"]*100.0) if c["TOTAL"] else 0.0, 2) for k,c in counts.items()}
    field_f1  = {k: round(_f1(c["TP"],c["FP"],c["FN"])*100.0, 2) for k,c in counts.items()}
    exact = round(exact_ok/max(total,1)*100.0, 2)

    return {
        "Documents": total,
        "FieldAccuracy(%)": field_acc,
        "FieldF1(%)": field_f1,
        "ExactMatch(%)": exact
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-ocr", required=True, help="GT dir with .txt")
    ap.add_argument("--pred-ocr", required=True, help="Pred dir with .txt")
    ap.add_argument("--gt-fields", required=True, help="GT dir with .json")
    ap.add_argument("--pred-fields", required=True, help="Pred dir with .json")
    args = ap.parse_args()

    ocr = eval_ocr_text_dir(args.gt_ocr, args.pred_ocr)
    fields = eval_fields_dir(args.gt_fields, args.pred_fields)

    print(json.dumps({
        "OCR": ocr,
        "FIELDS": fields
    }, ensure_ascii=False, indent=2))
