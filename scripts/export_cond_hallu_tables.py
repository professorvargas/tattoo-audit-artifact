#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import pandas as pd

# =============================
# CONFIG (você mexe aqui)
# =============================
MODELS_DEFAULT = ["gemma3", "llama3_2_vision", "qwen2_5_vl"]
VARIANTS_KEEP = ["baseline", "crops_black", "crops_white"]

# baseline hit: "any" (interseção com GT) ou "all" (conter todo GT)
BASELINE_HIT_MODE = "any"

# =============================
# helpers
# =============================
def parse_pred_labels(s):
    if pd.isna(s):
        return []
    ss = str(s).strip()
    if not ss:
        return []
    ss = ss.replace(";", ",")
    parts = [p.strip() for p in ss.split(",") if p.strip()]
    # unique preserve order
    out, seen = [], set()
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def parse_gt_image(s):
    if pd.isna(s):
        return set()
    ss = str(s).strip()
    if not ss:
        return set()
    ss = ss.replace(",", ";")
    parts = [p.strip() for p in ss.split(";") if p.strip()]
    return set(parts)

def stem_crop(crop_file):
    return os.path.splitext(str(crop_file))[0]

def safe_div(a, b):
    return (a / b) if b else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="DL_all_vlms_baseline_black_white_tables.xlsx")
    ap.add_argument("--outdir", default="tables", help="output folder")
    ap.add_argument("--split", default="", help="optional: test_open or test_closed")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_excel(args.xlsx, sheet_name="per_crop")

    # filtro split (se quiser)
    if args.split:
        df = df[df["split"] == args.split].copy()

    # normaliza
    df["pred_list"] = df["pred_labels"].apply(parse_pred_labels)
    df["pred_n"] = df["pred_list"].apply(len)
    df["n_fp"] = df["n_fp"].fillna(0).astype(int)

    # GT crop (usa coluna, fallback para stem do crop_file)
    df["gt_crop"] = df["crop_gt_label_from_filename"].fillna("").astype(str).str.strip()
    df.loc[df["gt_crop"] == "", "gt_crop"] = df.loc[df["gt_crop"] == "", "crop_file"].apply(stem_crop)

    # baseline GT set
    df["gt_set"] = df["gt_labels_image"].apply(parse_gt_image)

    # --- baseline rows ---
    base = df[(df["crop_file"] == "__full_image__") & (df["variant"].isin(["baseline"]))].copy()

    def baseline_hit(row):
        pred_set = set(row["pred_list"])
        gt_set = row["gt_set"]
        if not gt_set:
            return False
        if BASELINE_HIT_MODE == "all":
            return gt_set.issubset(pred_set)
        # default "any"
        return len(pred_set & gt_set) > 0

    def baseline_perfect(row):
        pred_set = set(row["pred_list"])
        gt_set = row["gt_set"]
        return bool(gt_set) and (pred_set == gt_set)

    base["hit"] = base.apply(baseline_hit, axis=1)
    base["perfect"] = base.apply(baseline_perfect, axis=1)
    base["hit_with_fp"] = base["hit"] & (base["n_fp"] > 0)
    base["fp_pct"] = base.apply(lambda r: safe_div(r["n_fp"], r["pred_n"]), axis=1)

    # agrega baseline por split+model
    base_sum = base.groupby(["split", "model"], as_index=False).agg(
        n_images=("image_id", "nunique"),
        hit=("hit", "sum"),
        hit_with_fp=("hit_with_fp", "sum"),
        perfect=("perfect", "sum"),
        mean_fp_pct_on_hit=("fp_pct", lambda s: s[base.loc[s.index, "hit"]].mean() if (base.loc[s.index, "hit"].any()) else 0.0),
    )
    base_sum["P(FP>0 | hit)"] = base_sum.apply(lambda r: safe_div(r["hit_with_fp"], r["hit"]), axis=1)
    base_sum["hit_rate"] = base_sum.apply(lambda r: safe_div(r["hit"], r["n_images"]), axis=1)
    base_sum["perfect_rate"] = base_sum.apply(lambda r: safe_div(r["perfect"], r["n_images"]), axis=1)

    # --- crop rows ---
    crops = df[(df["crop_file"] != "__full_image__") & (df["variant"].isin(["crops_black", "crops_white"]))].copy()
    crops["hit"] = crops.apply(lambda r: (r["gt_crop"] in r["pred_list"]), axis=1)
    crops["hit_with_fp"] = crops["hit"] & (crops["n_fp"] > 0)
    crops["fp_pct"] = crops.apply(lambda r: safe_div(r["n_fp"], r["pred_n"]), axis=1)

    crop_sum = crops.groupby(["split", "model", "variant"], as_index=False).agg(
        n_crops=("crop_file", "count"),
        hit=("hit", "sum"),
        hit_with_fp=("hit_with_fp", "sum"),
        mean_fp_pct_on_hit=("fp_pct", lambda s: s[crops.loc[s.index, "hit"]].mean() if (crops.loc[s.index, "hit"].any()) else 0.0),
        mean_nfp_on_hit=("n_fp", lambda s: s[crops.loc[s.index, "hit"]].mean() if (crops.loc[s.index, "hit"].any()) else 0.0),
    )
    crop_sum["P(FP>0 | hit)"] = crop_sum.apply(lambda r: safe_div(r["hit_with_fp"], r["hit"]), axis=1)
    crop_sum["hit_rate"] = crop_sum.apply(lambda r: safe_div(r["hit"], r["n_crops"]), axis=1)

    # salva XLSX com abas + CSVs
    out_xlsx = os.path.join(args.outdir, "tables_cond_hallu.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        base_sum.to_excel(w, index=False, sheet_name="baseline_summary")
        crop_sum.to_excel(w, index=False, sheet_name="crops_summary")
        # opcional: export “long” para pivot/depuração
        crops[["split","model","variant","image_id","crop_file","gt_crop","pred_labels","pred_n","n_fp","hit","hit_with_fp","fp_pct"]].to_excel(
            w, index=False, sheet_name="crops_long"
        )

    base_sum.to_csv(os.path.join(args.outdir, "baseline_summary.csv"), index=False)
    crop_sum.to_csv(os.path.join(args.outdir, "crops_summary.csv"), index=False)

    print("[OK] wrote:", out_xlsx)
    print("[OK] CSVs in:", args.outdir)

if __name__ == "__main__":
    main()
