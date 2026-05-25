# DepressionEmo-SL

<p align="left">
  <img src="https://img.shields.io/badge/Paper-ICDH%202026-lightgrey?style=flat-square" alt="Paper: ICDH 2026">
  <img src="https://img.shields.io/badge/PyTorch-Transformer%20Benchmarks-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch Transformer Benchmarks">
  <img src="https://img.shields.io/badge/Python-Research%20Code-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python Research Code">
  <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey?style=flat-square" alt="License: CC BY 4.0">
</p>

Official research code for the paper *Context as a Signal: Context-Aware Transformers for Fine-Grained Depression Emotions*.

Accepted at the IEEE International Conference on Digital Health (ICDH 2026). 🎉

This repository contains:

- sentence-level emotion classification utilities
- context-aware data preparation scripts
- Transformer training and evaluation scripts
- generated benchmark reports and figures

## What Is In This Repository 📦

`DepressionEmo-SL` extends depression-related posts with sentence-level, single-label emotion annotations. The benchmark evaluates how different amounts of surrounding context affect sentence classification.

The label set contains:

- `Anger`
- `Cognitive dysfunction`
- `Emptiness`
- `Hopelessness`
- `Loneliness`
- `Sadness`
- `Suicide intent`
- `Worthlessness`
- `No emotion`

The benchmark code supports context settings ranging from target-sentence-only inputs to wider thread-level context.

## Dataset Summary 📊

### Final Gold Dataset

| Field | Value |
| --- | ---: |
| Posts | 5,154 |
| Sentences | 34,864 |
| Labels | 9 |
| Language | English |

### Label Distribution

| Label | Count | Percent |
| --- | ---: | ---: |
| No emotion | 9,728 | 27.90% |
| Sadness | 5,025 | 14.41% |
| Loneliness | 4,274 | 12.26% |
| Hopelessness | 4,176 | 11.98% |
| Anger | 3,398 | 9.75% |
| Worthlessness | 3,021 | 8.67% |
| Suicide intent | 2,275 | 6.53% |
| Emptiness | 1,667 | 4.78% |
| Cognitive dysfunction | 1,300 | 3.73% |

### Split Sizes

| Split | Posts | Sentences |
| --- | ---: | ---: |
| Train | 4,368 | 24,349 |
| Validation | 446 | 5,258 |
| Test | 340 | 5,257 |

### Agreement Snapshot

- Unanimous agreement across the three primary annotators: `63.17%`
- Any disagreement: `36.83%`
- No-majority cases after the first pass: `1,953` sentences (`5.60%`)
- Fleiss' kappa on the first-pass annotations: `0.6895`

## Repository Layout 🗂️

```text
repository-root/
|- codes/
|  |- training and evaluation scripts
|  |- context generation utilities
|  |- report collection utilities
|  `- plotting utilities
|- run_reports_advanced/
|- requirements.txt
|- LICENSE
`- README.md
```

Notes:

- `codes/` contains the main research scripts.
- `run_reports_advanced/` contains generated result artifacts.
- Local data, checkpoints, caches, and logs are kept outside the tracked research scripts.

## Data Files 🧾

The project expects local sentence-level data files in the format used by the training scripts.

Typical records include sentence identifiers, text fields, and a single gold label.

Context-augmented variants add fields for context mode, marked target sentence text, thread information, and sentence position.

## Setup ⚙️

Use a Python environment with the dependencies in `requirements.txt`.

Some scripts require local data paths and suitable compute resources.

## Running Experiments 🚀

The main experiment scripts are in `codes/`.

At a high level, the workflow is:

1. prepare local sentence-level files
2. create context-aware variants
3. train selected Transformer backbones
4. collect evaluation summaries
5. generate plots from the report outputs

The scripts expose command-line arguments for paths, model names, context settings, batch sizes, seeds, and report locations.

## Benchmark Output Artifacts 📈

Benchmark outputs are stored as generated reports, summary tables, and figures. Large model checkpoint weights are not intended to be tracked.

Top benchmark results from the generated summaries:

| Model | Context | Macro F1 | Accuracy |
| --- | --- | ---: | ---: |
| Mental-RoBERTa ensemble | C0toC3 | 0.6870 | 0.7156 |
| DeBERTa-v3 ensemble | C0toC3 | 0.6842 | 0.7072 |
| Mental-RoBERTa | C3 | 0.6811 | 0.7036 |
| DeBERTa-v3 | C2 | 0.6792 | 0.7025 |
| RoBERTa-base ensemble | C0toC3 | 0.6747 | 0.7065 |

## Ethics And Data Use 🛡️

This repository relates to mental-health text. Use it responsibly.

- Do not attempt re-identification or deanonymization.
- Follow applicable platform terms and institutional review requirements.
- Treat generated model outputs as research artifacts, not clinical tools.
- The models and labels in this repository are not for diagnosis or crisis triage.

## Citation ✍️

If you use this repository before the canonical paper citation is available, cite the repository directly:

```bibtex
@misc{depressionemo_sl_official,
  title = {DepressionEmo-SL: Official research code},
  note = {Official repository for "Context as a Signal: Context-Aware Transformers for Fine-Grained Depression Emotions"},
  year = {2026}
}
```

## License 📄

This repository is released under `CC BY 4.0`. See `LICENSE` for the repository-level license notice.
