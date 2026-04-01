from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Tattoo Audit MVP", layout="wide")

MODELS = {
    "Gemma3:12b": "runs/gemma3",
    "Qwen2.5-VL-7B": "runs/qwen2_5_vl",
    "LLaMA 3.2 Vision 11B": "runs/llama3_2_vision",
}

VARIANTS = {
    "baseline": "Baseline",
    "crops_black": "Crop Black",
    "crops_white": "Crop White",
}


def existing_path(*candidates: Path) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_jsonl_safe(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


@st.cache_data(show_spinner=False)
def discover_cases(repo_root: str, split: str) -> List[str]:
    root = Path(repo_root)
    cases: set[str] = set()

    crops_gt = root / "datasets" / "crops_gt" / split
    if crops_gt.exists():
        cases.update(p.name for p in crops_gt.iterdir() if p.is_dir())

    panels_assets = root / "panels_custom_with_images" / "assets" / split
    if panels_assets.exists():
        cases.update(p.name for p in panels_assets.iterdir() if p.is_dir())

    return sorted(cases)


def open_image_if_exists(path: Optional[Path]) -> Optional[Image.Image]:
    if not path or not path.exists():
        return None
    try:
        return Image.open(path)
    except Exception:
        return None


def parse_list_like(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass

        # separadores reais do seu projeto: ; e ,
        if ";" in text or "," in text:
            parts = re.split(r"[;,]", text)
            return [x.strip() for x in parts if x.strip()]

        return [text]

    return [str(value).strip()]

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]


def case_image_path(repo_root: str, split: str, case_id: str) -> Optional[Path]:
    root = Path(repo_root)
    images_dir = root / "datasets" / split / "images"
    for ext in IMAGE_EXTS:
        p = images_dir / f"{case_id}{ext}"
        if p.exists():
            return p
    return None


def case_mask_path(repo_root: str, split: str, case_id: str) -> Optional[Path]:
    root = Path(repo_root)

    # prioridade 1: pasta demo/panel
    panel_dir = root / "panels_custom_with_images" / "assets" / split / case_id
    for name in ["mask.jpg", "mask.png"]:
        p = panel_dir / name
        if p.exists():
            return p

    # prioridade 2: máscara do dataset
    mask_dir = root / "datasets" / split / "mask_rgb"
    for ext in IMAGE_EXTS:
        p = mask_dir / f"{case_id}_mask{ext}"
        if p.exists():
            return p

    return None


def gt_crop_paths(repo_root: str, split: str, case_id: str) -> List[Path]:
    root = Path(repo_root)
    crop_dir = root / "datasets" / "crops_gt" / split / case_id
    if not crop_dir.exists():
        return []
    return sorted([p for p in crop_dir.glob("*.png") if p.is_file()])

def gt_black_crop_paths(repo_root: str, split: str, case_id: str) -> List[Path]:
    root = Path(repo_root)
    crop_dir = root / "datasets" / "crops_gt_black" / split / case_id
    if not crop_dir.exists():
        return []
    return sorted([p for p in crop_dir.glob("*.png") if p.is_file()])


def gt_white_crop_paths(repo_root: str, split: str, case_id: str) -> List[Path]:
    root = Path(repo_root)
    crop_dir = root / "datasets" / "crops_gt_white" / split / case_id
    if not crop_dir.exists():
        return []
    return sorted([p for p in crop_dir.glob("*.png") if p.is_file()])


def gt_labels_from_crops(repo_root: str, split: str, case_id: str) -> List[str]:
    labels = []
    for p in gt_crop_paths(repo_root, split, case_id):
        label = p.stem.strip()
        if label and label not in labels:
            labels.append(label)
    return labels


CASE_KEYS = ["image_id", "image", "case_id", "id", "file", "filename", "img"]
LABEL_KEYS = [
    "pred_labels",
    "labels_pred",
    "labels",
    "predicted_labels",
    "pred",
    "prediction",
    "mapped_labels",
    "parsed_labels",
]
GT_KEYS = ["gt_labels", "gt", "ground_truth", "gold_labels"]
UNKNOWN_KEYS = ["unknown", "has_unknown", "unknown_present"]
FP_KEYS = ["fp", "fp_img", "fp_per_image", "false_positives", "fp_count"]
FN_KEYS = ["fn", "fn_img", "fn_per_image", "false_negatives", "fn_count"]


def row_for_case(df: pd.DataFrame, case_id: str) -> Optional[pd.Series]:
    if df.empty:
        return None
    for col in df.columns:
        if col.lower() in CASE_KEYS or any(k in col.lower() for k in CASE_KEYS):
            series = df[col].astype(str)
            mask = series.str.contains(re.escape(case_id), regex=True, na=False)
            if mask.any():
                return df[mask].iloc[0]
    joined = df.astype(str).agg(" | ".join, axis=1)
    mask = joined.str.contains(re.escape(case_id), regex=True, na=False)
    if mask.any():
        return df[mask].iloc[0]
    return None


def extract_from_row(row: Optional[pd.Series], preferred_keys: Iterable[str]) -> Any:
    if row is None:
        return None
    lowered = {str(col).lower(): col for col in row.index}
    for key in preferred_keys:
        for col_lower, col_orig in lowered.items():
            if key == col_lower or key in col_lower:
                return row[col_orig]
    return None


@st.cache_data(show_spinner=False)
def load_variant_tables(repo_root: str, model_dir: str, split: str, variant: str) -> Dict[str, pd.DataFrame]:
    root = Path(repo_root)
    base = root / model_dir / variant / split / "eval"
    return {
        "metrics_per_image": read_csv_safe(base / "metrics_per_image.csv"),
        "metrics_per_crop": read_csv_safe(base / "metrics_per_crop.csv"),
        "pred_crop_labels": read_csv_safe(base / "pred_crop_labels_per_image.csv"),
        "best_images": read_csv_safe(base / "best_images.csv"),
        "worst_images": read_csv_safe(base / "worst_images.csv"),
    }


def preferred_jsonl_filename(variant: str, split: str) -> str:
    if variant == "baseline":
        return f"p1_baseline_{split}.jsonl"
    if variant == "crops_black":
        return f"p1_crop_{split}.jsonl"
    if variant == "crops_white":
        return f"p1_crop_white_{split}.jsonl"
    return ""

@st.cache_data(show_spinner=False)
def load_variant_jsonl(repo_root: str, model_dir: str, split: str, variant: str) -> List[Dict[str, Any]]:
    root = Path(repo_root)
    raw_dir = root / model_dir / variant / split / "raw"
    if not raw_dir.exists():
        return []

    preferred = raw_dir / preferred_jsonl_filename(variant, split)
    if preferred.exists():
        return read_jsonl_safe(preferred)

    jsonls = sorted(raw_dir.glob("*.jsonl"))
    if not jsonls:
        return []

    return read_jsonl_safe(jsonls[0])


def extract_jsonl_case(records: List[Dict[str, Any]], case_id: str) -> Optional[Dict[str, Any]]:
    for record in records:
        haystack = json.dumps(record, ensure_ascii=False)
        if case_id in haystack:
            return record
    return None


def collect_variant_prediction(repo_root: str, model_name: str, split: str, variant: str, case_id: str) -> Dict[str, Any]:
    model_dir = MODELS[model_name]
    tables = load_variant_tables(repo_root, model_dir, split, variant)

    row = None
    for table_name in ["pred_crop_labels", "metrics_per_image", "metrics_per_crop", "best_images", "worst_images"]:
        row = row_for_case(tables[table_name], case_id)
        if row is not None:
            break

    labels = parse_list_like(extract_from_row(row, LABEL_KEYS))
    gt_labels = parse_list_like(extract_from_row(row, GT_KEYS))
    fp = extract_from_row(row, FP_KEYS)
    fn = extract_from_row(row, FN_KEYS)
    unknown = extract_from_row(row, UNKNOWN_KEYS)

    # fallback de GT pelo dataset real
    if not gt_labels:
        gt_labels = gt_labels_from_crops(repo_root, split, case_id)

    # fallback para JSONL real
    if not labels:
        record = extract_jsonl_case(load_variant_jsonl(repo_root, model_dir, split, variant), case_id)
        if record:
            for key in ["json_obj", "parsed", "prediction", "pred"]:
                candidate = record.get(key)
                if isinstance(candidate, dict):
                    labels = parse_list_like(candidate.get("dominant_elements") or candidate.get("labels"))
                    if unknown is None and "unknown" in candidate:
                        unknown = candidate.get("unknown")
                    if labels:
                        break

            # se não vier dict, tenta campos diretos
            if not labels:
                labels = parse_list_like(record.get("labels") or record.get("pred_labels"))

            # último fallback: tentar json_obj explicitamente
            if not labels and isinstance(record.get("json_obj"), dict):
                labels = parse_list_like(record["json_obj"].get("labels"))
                if unknown is None and "unknown" in record["json_obj"]:
                    unknown = record["json_obj"]["unknown"]

    # se unknown não veio explicitamente, infere pelos labels
    if unknown is None:
        unknown = any(str(x).strip().lower() == "unknown" for x in labels)

    return {
        "labels": labels,
        "gt_labels": gt_labels,
        "fp": fp,
        "fn": fn,
        "unknown": unknown,
        "row": row,
    }


def panel_assets(repo_root: str, split: str, case_id: str) -> Dict[str, Optional[Path]]:
    root = Path(repo_root)
    panel_dir = root / "panels_custom_with_images" / "assets" / split / case_id

    baseline = existing_path(
        panel_dir / "baseline.jpg",
        panel_dir / "baseline.png",
        case_image_path(repo_root, split, case_id),
    )

    mask = existing_path(
        panel_dir / "mask.jpg",
        panel_dir / "mask.png",
        case_mask_path(repo_root, split, case_id),
    )

    # prioridade 1: painéis demo
    black_candidates = sorted(panel_dir.glob("crop_black_*"))
    white_candidates = sorted(panel_dir.glob("crop_white_*"))

    # prioridade 2: datasets completos
    if not black_candidates:
        black_candidates = gt_black_crop_paths(repo_root, split, case_id)

    if not white_candidates:
        white_candidates = gt_white_crop_paths(repo_root, split, case_id)

    
    return {
    "baseline": baseline,
    "mask": mask,
    "crop_black": black_candidates[0] if black_candidates else None,
    "crop_white": white_candidates[0] if white_candidates else None,
}


def compute_risk(repo_root: str, split: str, case_id: str) -> Dict[str, Any]:
    all_predictions: Dict[str, Dict[str, Any]] = {}
    flags: List[str] = []
    score = 0

    for model_name in MODELS:
        all_predictions[model_name] = {}
        base = collect_variant_prediction(repo_root, model_name, split, "baseline", case_id)
        black = collect_variant_prediction(repo_root, model_name, split, "crops_black", case_id)
        white = collect_variant_prediction(repo_root, model_name, split, "crops_white", case_id)
        all_predictions[model_name]["baseline"] = base
        all_predictions[model_name]["crops_black"] = black
        all_predictions[model_name]["crops_white"] = white

        base_set = set(base["labels"])
        black_set = set(black["labels"])
        white_set = set(white["labels"])

        if "unknown" in {x.lower() for x in base_set | black_set | white_set}:
            flags.append(f"{model_name}: unknown present")
            score += 2
        if black_set != white_set and (black_set or white_set):
            flags.append(f"{model_name}: flip black/white")
            score += 2
        if base_set != black_set or base_set != white_set:
            flags.append(f"{model_name}: baseline/crops divergence")
            score += 1

        for variant_name, payload in [("baseline", base), ("crops_black", black), ("crops_white", white)]:
            fp = payload.get("fp")
            try:
                if fp is not None and float(fp) >= 3:
                    flags.append(f"{model_name} {variant_name}: high FP ({fp})")
                    score += 2
            except Exception:
                pass

    return {"score": score, "flags": flags, "predictions": all_predictions}


@st.cache_data(show_spinner=False)
def build_case_ranking(repo_root: str, split: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for case_id in discover_cases(repo_root, split):
        risk = compute_risk(repo_root, split, case_id)
        rows.append({
            "case_id": case_id,
            "risk_score": risk["score"],
            "n_flags": len(risk["flags"]),
            "flags": " | ".join(risk["flags"]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["risk_score", "n_flags", "case_id"], ascending=[False, False, True]).reset_index(drop=True)


def audit_log_path(repo_root: str) -> Path:
    path = Path(repo_root) / "audit_logs"
    path.mkdir(parents=True, exist_ok=True)
    return path / "audit_log.csv"


def append_audit_decision(repo_root: str, payload: Dict[str, Any]) -> None:
    out = audit_log_path(repo_root)
    frame = pd.DataFrame([payload])
    if out.exists():
        frame.to_csv(out, mode="a", header=False, index=False)
    else:
        frame.to_csv(out, index=False)


def show_image(title: str, image_path: Optional[Path]) -> None:
    st.markdown(f"**{title}**")
    image = open_image_if_exists(image_path)
    if image is None:
        st.caption("Image not found.")
    else:
        st.image(image, use_container_width=True)
        st.caption(str(image_path))


st.title("Tattoo Audit MVP")
st.caption("Audit dashboard for risk analysis without rerunning VLMs.")

with st.sidebar:
    st.header("Configuration")
    auto_repo_root = str(Path(__file__).resolve().parent)
    use_custom_root = st.checkbox("Use manual path", value=False)
    repo_root = auto_repo_root
    if use_custom_root:
        repo_root = st.text_input("Project root", value=auto_repo_root)
    else:
        st.text_input("Project root", value=auto_repo_root, disabled=True)
    split = st.selectbox("Split", ["test_open", "test_closed"], index=0)

    ranking = build_case_ranking(repo_root, split)
    if ranking.empty:
        st.warning("No cases found. Check the project root.")
        st.stop()

    only_high_risk = st.checkbox("Show only risk >= 3", value=False)
    filtered = ranking[ranking["risk_score"] >= 3] if only_high_risk else ranking
    case_id = st.selectbox("Case", filtered["case_id"].tolist())

risk = compute_risk(repo_root, split, case_id)
assets = panel_assets(repo_root, split, case_id)
gt_labels = gt_labels_from_crops(repo_root, split, case_id)

col_a, col_b, col_c = st.columns([1.1, 1.1, 1.4])
with col_a:
    show_image("Baseline image", assets["baseline"])
with col_b:
    show_image("Mask / reference", assets["mask"])
with col_c:
    sub1, sub2 = st.columns(2)
    with sub1:
        show_image("Crop black", assets["crop_black"])
    with sub2:
        show_image("Crop white", assets["crop_white"])

st.markdown("---")
st.markdown("### Ground-truth labels")
if gt_labels:
    st.write(gt_labels)
else:
    st.caption("No GT labels found for this case.")

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Risk score", risk["score"])
metric2.metric("Flags", len(risk["flags"]))
metric3.metric("Split", split)

if risk["flags"]:
    st.markdown("### Risk indicators")
    for flag in risk["flags"]:
        st.write(f"- {flag}")
else:
    st.success("No risk indicators detected by the current heuristics.")

st.markdown("### Predictions by model")
for model_name, variants in risk["predictions"].items():
    st.subheader(model_name)
    cols = st.columns(3)
    for idx, variant in enumerate(["baseline", "crops_black", "crops_white"]):
        payload = variants[variant]
        with cols[idx]:
            st.markdown(f"**{VARIANTS[variant]}**")
            st.write("Labels:", payload["labels"] or [])
            if payload["gt_labels"]:
                st.write("GT:", payload["gt_labels"])
            if payload["fp"] is not None:
                st.write("FP:", payload["fp"])
            if payload["fn"] is not None:
                st.write("FN:", payload["fn"])
            if payload["unknown"] is not None:
                st.write("Unknown:", payload["unknown"])
            row = payload.get("row")
            if row is not None:
                with st.expander("Row details"):
                    row_df = pd.DataFrame({"field": row.index, "value": [row[c] for c in row.index]})
                    st.dataframe(row_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### Human audit")
with st.form("audit_form"):
    auditor = st.text_input("Auditor")
    overall_decision = st.selectbox("Overall decision", ["confirmed", "rejected", "uncertain"])
    policy_relevance = st.selectbox(
        "Public policy relevance",
        ["high", "medium", "low", "not assessed"],
        index=3,
    )
    comment = st.text_area("Short comment")
    submitted = st.form_submit_button("Save decision")

if submitted:
    append_audit_decision(
        repo_root,
        {
            "case_id": case_id,
            "split": split,
            "auditor": auditor,
            "decision": overall_decision,
            "policy_relevance": policy_relevance,
            "risk_score": risk["score"],
            "flags": " | ".join(risk["flags"]),
            "comment": comment,
        },
    )
    st.success(f"Decision saved to {audit_log_path(repo_root)}")

st.markdown("---")
st.markdown("### Case ranking")
st.dataframe(ranking, use_container_width=True, hide_index=True)

log_file = audit_log_path(repo_root)
if log_file.exists():
    st.download_button(
        label="Download audit log",
        data=log_file.read_bytes(),
        file_name=log_file.name,
        mime="text/csv",
    )
