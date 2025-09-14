from tools.metrics import eval_ocr_text_dir
import json

def compare(gt_dir, ours, tess, title):
    ours_res = eval_ocr_text_dir(gt_dir, ours)
    tess_res = eval_ocr_text_dir(gt_dir, tess)
    print("\n"+title)
    print("Docs:", ours_res["Documents"], "|", tess_res["Documents"])
    print("CER: ours", ours_res["CER"], "| tess", tess_res["CER"])
    print("WER: ours", ours_res["WER"], "| tess", tess_res["WER"])
    print("NLD: ours", ours_res["NormalizedLevenshtein"], "| tess", tess_res["NormalizedLevenshtein"])

if __name__ == "__main__":
    compare("data/gt/ocr_text","data/preds/ocr_text","data/tesseract/ocr_text","NORMAL SET")
    compare("data/gt/ocr_text","data/preds_noisy/ocr_text","data/tesseract_noisy/ocr_text","NOISY SET")
