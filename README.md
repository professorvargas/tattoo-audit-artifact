# tattoo-audit-artifact

Artifact repository for the paper:

**TattooAudit: An Interactive Auditing Layer for Vision–Language Models in Open-Set Tattoo Analysis**

This repository contains the benchmark-grounded artifact used in the study, including:
- evaluation scripts
- parsing rules
- exported tables
- ranking outputs
- the Tattoo Audit dashboard source code

## Dataset

The public dataset used in the study is **TSSD2023**:  
https://github.com/Brilhador/tssd2023

## Objective

This project evaluates Vision–Language Models (VLMs) in an **open-set** tattoo-analysis scenario using oracle segmentation derived from ground-truth masks. The goal is to isolate the **semantic naming** component of the models, rather than evaluating pixel-level segmentation.

The study compares three input conditions:
1. **Baseline**: full image
2. **GT crop (black background)**: crop derived from the GT mask over a black background
3. **GT crop (white background)**: crop derived from the GT mask over a white background

## Evaluated models

- Gemma3
- Qwen2.5-VL
- LLaMA 3.2 Vision

## Main metric

For each image:
- `GT` = set of ground-truth labels
- `Pred` = set of predicted labels after parsing and normalization

Definitions:
- `TP = |GT ∩ Pred|`
- `FP = |Pred \ GT|`
- `FN = |GT \ Pred|`

Dataset-level micro-F1:
- `micro-F1 = 2TP / (2TP + FP + FN)`

## Repository structure

- `run_experiments.py`: general orchestration
- `experiments/`: crop generation, VLM execution, parsing, metrics, aggregations, and figure/table generation
- `data_meta/`: small metadata files
- `environment.yml`: conda/mamba environment
- `mvp_audit_streamlit.py`: main Tattoo Audit dashboard
- `datasets/`: local data required to run the dashboard and reproduce the evaluated views

## How to run the Tattoo Audit dashboard

Run locally with:

```bash
streamlit run mvp_audit_streamlit.py
