from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import streamlit as st
from PIL import Image

import streamlit as st

st.set_page_config(page_title="Tattoo Audit MVP", layout="wide")

st.html("""
<style>
/* Fonte geral da aplicação */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-size: 22px !important;
}

/* Sidebar inteira */
[data-testid="stSidebar"] * {
    font-size: 20px !important;
}

/* Títulos */
h1 { font-size: 2.2rem !important; }
h2 { font-size: 3.8rem !important; }
h3 { font-size: 1.5rem !important; }

/* Markdown, textos, captions, listas */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"],
label,
div,
span {
    font-size: 1.05rem !important;
}

/* Inputs e widgets */
input, textarea, select, button {
    font-size: 1rem !important;
}

/* Métricas */
[data-testid="stMetricLabel"] p {
    font-size: 1.2rem !important;
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
}
[data-testid="stMetricDelta"] {
    font-size: 1rem !important;
}

/* Expander */
[data-testid="stExpander"] summary {
    font-size: 1.1rem !important;
}

/* Dataframe */
[data-testid="stDataFrame"] * {
    font-size: 18px !important;
}
</style>
""")


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

def crop_pairs_by_label(repo_root: str, split: str, case_id: str) -> List[Dict[str, Optional[Path]]]:
    black_map = {p.stem: p for p in gt_black_crop_paths(repo_root, split, case_id)}
    white_map = {p.stem: p for p in gt_white_crop_paths(repo_root, split, case_id)}
    labels = sorted(set(black_map.keys()) | set(white_map.keys()))

    pairs = []
    for label in labels:
        pairs.append({
            "label": label,
            "black": black_map.get(label),
            "white": white_map.get(label),
        })
    return pairs


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

def first_case_row(case_id: str, *dfs: pd.DataFrame) -> Optional[pd.Series]:
    for df in dfs:
        row = row_for_case(df, case_id)
        if row is not None:
            return row
    return None


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def normalize_label_set(values: List[str]) -> set[str]:
    return {str(v).strip().lower() for v in values if str(v).strip()}


def collect_variant_prediction(repo_root: str, model_name: str, split: str, variant: str, case_id: str) -> Dict[str, Any]:
    model_dir = MODELS[model_name]
    tables = load_variant_tables(repo_root, model_dir, split, variant)

    row_labels = first_case_row(
        case_id,
        tables["pred_crop_labels"],
        tables["metrics_per_image"],
        tables["metrics_per_crop"],
        tables["best_images"],
        tables["worst_images"],
    )

    row_metrics = first_case_row(
        case_id,
        tables["metrics_per_image"],
        tables["metrics_per_crop"],
        tables["best_images"],
        tables["worst_images"],
        tables["pred_crop_labels"],
    )

    labels = parse_list_like(extract_from_row(row_labels, LABEL_KEYS))

    gt_labels = parse_list_like(extract_from_row(row_metrics, GT_KEYS))
    if not gt_labels:
        gt_labels = parse_list_like(extract_from_row(row_labels, GT_KEYS))

    fp = extract_from_row(row_metrics, FP_KEYS)
    fn = extract_from_row(row_metrics, FN_KEYS)

    unknown = extract_from_row(row_metrics, UNKNOWN_KEYS)
    if is_missing(unknown):
        unknown = extract_from_row(row_labels, UNKNOWN_KEYS)

    if not gt_labels:
        gt_labels = gt_labels_from_crops(repo_root, split, case_id)

    if not labels:
        record = extract_jsonl_case(load_variant_jsonl(repo_root, model_dir, split, variant), case_id)
        if record:
            for key in ["json_obj", "parsed", "prediction", "pred"]:
                candidate = record.get(key)
                if isinstance(candidate, dict):
                    labels = parse_list_like(candidate.get("dominant_elements") or candidate.get("labels"))
                    if is_missing(unknown) and "unknown" in candidate:
                        unknown = candidate.get("unknown")
                    if labels:
                        break

            if not labels:
                labels = parse_list_like(record.get("labels") or record.get("pred_labels"))

            if not labels and isinstance(record.get("json_obj"), dict):
                labels = parse_list_like(record["json_obj"].get("labels"))
                if is_missing(unknown) and "unknown" in record["json_obj"]:
                    unknown = record["json_obj"]["unknown"]

    if is_missing(unknown):
        unknown = any(str(x).strip().lower() == "unknown" for x in labels)

    # fallback: calcula FP/FN se a tabela não trouxer esses valores
    if gt_labels and labels:
        pred_set = normalize_label_set(labels)
        gt_set = normalize_label_set(gt_labels)

        if is_missing(fp):
            fp = len(pred_set - gt_set)

        if is_missing(fn):
            fn = len(gt_set - pred_set)

    row = row_metrics if row_metrics is not None else row_labels

    return {
        "labels": labels,
        "gt_labels": gt_labels,
        "fp": fp,
        "fn": fn,
        "unknown": unknown,
        "row": row,
    }

def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def build_row_details(payload: Dict[str, Any], case_id: str, split: str) -> pd.DataFrame:
    details: Dict[str, Any] = {}

    row = payload.get("row")
    if row is not None:
        details.update({str(col): row[col] for col in row.index})

    pred_labels = payload.get("labels", []) or []
    gt_labels = payload.get("gt_labels", []) or []

    pred_set = normalize_label_set(pred_labels)
    gt_set = normalize_label_set(gt_labels)

    tp = len(pred_set & gt_set)

    fp = payload.get("fp")
    fn = payload.get("fn")

    if is_missing(fp):
        fp = len(pred_set - gt_set)
    if is_missing(fn):
        fn = len(gt_set - pred_set)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    jaccard = safe_div(tp, tp + fp + fn)

    pred_size = len(pred_set)
    gold_size = len(gt_set)
    overprediction = safe_div(pred_size, gold_size)

    # Ajuste este valor se seu benchmark usar outro tamanho de vocabulário
    label_space_size = 35
    hamming_loss = safe_div(fp + fn, label_space_size)

    # Preenche campos faltantes sem sobrescrever os já existentes
    details.setdefault("split", split)
    details.setdefault("image_id", case_id)
    details.setdefault("pred_labels", ";".join(pred_labels))
    details.setdefault("gt_labels", ";".join(gt_labels))
    details.setdefault("tp", tp)
    details.setdefault("fp", fp)
    details.setdefault("fn", fn)
    details.setdefault("precision", precision)
    details.setdefault("recall", recall)
    details.setdefault("f1", f1)
    details.setdefault("jaccard", jaccard)
    details.setdefault("hamming_loss", hamming_loss)
    details.setdefault("overprediction", overprediction)
    details.setdefault("pred_size", pred_size)
    details.setdefault("gold_size", gold_size)
    details.setdefault("unknown", payload.get("unknown"))

    preferred_order = [
        "split",
        "image_id",
        "pred_labels",
        "gt_labels",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "jaccard",
        "hamming_loss",
        "overprediction",
        "pred_size",
        "gold_size",
        "unknown",
    ]

    ordered_keys = [k for k in preferred_order if k in details]
    ordered_keys += [k for k in details.keys() if k not in ordered_keys]

    return pd.DataFrame({
        "field": ordered_keys,
        "value": [details[k] for k in ordered_keys],
    })

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


def compute_priority(repo_root: str, split: str, case_id: str) -> Dict[str, Any]:
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
        priority = compute_priority(repo_root, split, case_id)
        rows.append({
            "case_id": case_id,
            "priority_score": priority["score"],
            "n_flags": len(priority["flags"]),
            "flags": " | ".join(priority["flags"]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["priority_score", "n_flags", "case_id"], ascending=[False, False, True]).reset_index(drop=True)

def theoretical_max_priority() -> int:
    # por modelo: unknown(2) + flip(2) + divergence(1) + high FP em 3 variantes (6)
    per_model_max = 2 + 2 + 1 + (3 * 2)
    return len(MODELS) * per_model_max  # 33 com 3 modelos


def observed_max_priority(ranking: pd.DataFrame) -> int:
    if ranking.empty:
        return 0
    return int(ranking["priority_score"].max())


def priority_band(score: int) -> str:
    # faixas simples e fáceis de explicar para o auditor
    if score <= 3:
        return "Very low"
    if score <= 9:
        return "Low"
    if score <= 16:
        return "Medium"
    if score <= 20:
        return "High"
    return "Critical"

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
        st.image(image, width=320)
        st.caption(str(image_path))


st.html("""
<h1 style="
    font-size: 56px;
    margin-bottom: 0.2rem;
">
    Tattoo Audit
</h1>
""")
st.caption("Audit dashboard for audit-priority analysis.")

with st.sidebar:
    st.header("Parameters")
    auto_repo_root = str(Path(__file__).resolve().parent)
    use_custom_root = st.checkbox("Use manual path", value=False)
    repo_root = auto_repo_root

    if use_custom_root:
        repo_root = st.text_input("Project root", value=auto_repo_root)
    else:
        st.text_input("Project root", value=auto_repo_root, disabled=True)

    split = "test_open"

    ranking = build_case_ranking(repo_root, split)
    max_observed_score = observed_max_priority(ranking)
    max_theoretical_score = theoretical_max_priority()

    if ranking.empty:
        st.warning("No cases found. Check the project root.")
        st.stop()

    min_priority = st.slider(
        "Minimum priority",
        min_value=0,
        max_value=max_observed_score,
        value=0,
        step=1,
    )

    filtered = ranking[ranking["priority_score"] >= min_priority]

    if filtered.empty:
        st.warning("No cases match the selected minimum priority.")
        st.stop()

    case_id = st.selectbox("Case (Select Tattoo)", filtered["case_id"].tolist())

priority = compute_priority(repo_root, split, case_id)
assets = panel_assets(repo_root, split, case_id)
crop_pairs = crop_pairs_by_label(repo_root, split, case_id)
gt_labels = gt_labels_from_crops(repo_root, split, case_id)

col_a, col_b = st.columns([1.2, 1.0])

with col_a:
    show_image("Baseline image", assets["baseline"])

with col_b:
    show_image("Mask / reference", assets["mask"])

if crop_pairs:
    st.markdown("---")
    st.markdown("### All GT-derived crops")

    for pair in crop_pairs:
        st.markdown(f"**Label: {pair['label']}**")
        c1, c2 = st.columns(2)
        with c1:
            show_image(f"Black crop: {pair['label']}", pair["black"])
        with c2:
            show_image(f"White crop: {pair['label']}", pair["white"])

st.markdown("---")
st.markdown("### Ground-truth labels")
if gt_labels:
    st.write(gt_labels)
else:
    st.caption("No GT labels found for this case.")

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Audit-priority score", f"{priority['score']} / {max_observed_score}")
metric2.metric("Band", priority_band(priority["score"]))
metric3.metric("Flags", len(priority["flags"]))
metric4.metric("Dataset", "test_open")

st.caption(
    f"Audit-priority score is a relative audit-priority index. "
    f"Max observed in this split: {max_observed_score}. "
    f"Theoretical max under current heuristics: {max_theoretical_score}."
)

if priority["flags"]:
    st.markdown("### Why this case received this priority")
    for flag in priority["flags"]:
        st.write(f"- {flag}")
else:
    st.success("No priority indicators detected by the current heuristics.")

st.markdown("### Predictions by model")
for model_name, variants in priority["predictions"].items():
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
                    row_df = build_row_details(payload, case_id, split).copy()
                    row_df["field"] = row_df["field"].astype(str)
                    row_df["value"] = row_df["value"].astype(str)
                    st.dataframe(row_df, width="stretch", hide_index=True)

st.markdown("---")
st.markdown("### Human audit")
with st.form("audit_form"):
    auditor = st.text_input("Auditor")
    audit_decision = st.selectbox(
    "Audit decision",
    ["confirmed priority", "rejected priority", "uncertain"]
)
    policy_relevance = st.selectbox(
        "Potential public policy relevance",
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
            "decision": audit_decision,
            "policy_relevance": policy_relevance,
            "priority_score": priority["score"],
            "flags": " | ".join(priority["flags"]),
            "comment": comment,
        },
    )
    st.success(f"Decision saved to {audit_log_path(repo_root)}")

st.markdown("---")
st.markdown("### Case ranking")
ranking_view = ranking.copy()
ranking_view["band"] = ranking_view["priority_score"].apply(priority_band)
ranking_view["priority_score_display"] = ranking_view["priority_score"].apply(
    lambda x: f"{x} / {max_observed_score}"
)

ranking_view = ranking_view.rename(
    columns={
        "case_id": "Case",
        "priority_score": "Audit-priority score (raw)",
        "priority_score_display": "Audit-priority score",
        "n_flags": "Flags",
        "band": "Band",
    }
)

st.dataframe(
    ranking_view[["Case", "Audit-priority score", "Band", "Flags", "flags"]],
    width="stretch",
    hide_index=True,
)

log_file = audit_log_path(repo_root)
if log_file.exists():
    st.download_button(
        label="Download audit log",
        data=log_file.read_bytes(),
        file_name=log_file.name,
        mime="text/csv",
    )
