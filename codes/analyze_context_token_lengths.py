import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from transformers import AutoTokenizer


os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


SPECIAL_TOKENS: List[str] = [
    "[TARGET]",
    "[/TARGET]",
    "[PREV]",
    "[/PREV]",
    "[NEXT]",
    "[/NEXT]",
    "[CTX]",
    "[/CTX]",
]


VALID_CONTEXT_MODES = {"C0", "C1", "C2", "C3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze token lengths for contextualized sentences and report how "
            "many exceed a max length threshold."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(
            Path("datasets")
            / "gold"
            / "splits"
        ),
        help="Directory containing split JSONL files (train.jsonl, val.jsonl, test.jsonl).",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated split names to analyze.",
    )
    parser.add_argument(
        "--model-name",
        default="google-bert/bert-base-uncased",
        help="Hugging Face model name for the tokenizer.",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated list of model names to analyze.",
    )
    parser.add_argument(
        "--context-modes",
        default="C3",
        help="Comma-separated context modes (C0,C1,C2,C3). Default is C3.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Token length threshold to flag as too long.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for tokenizer.",
    )
    parser.add_argument(
        "--summary-scope",
        default="all",
        choices=["all"],
        help="Summary scope to write (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        default="token_length_analysis",
        help="Directory to write summary CSV/JSON files.",
    )
    return parser.parse_args()


def parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_context_modes(value: str) -> List[str]:
    modes = [item.upper() for item in parse_list(value)]
    if not modes:
        raise ValueError("No context modes provided.")
    invalid = sorted(set(modes) - VALID_CONTEXT_MODES)
    if invalid:
        raise ValueError(f"Invalid context modes: {', '.join(invalid)}")
    return modes


def parse_models(models_value: str, fallback: str) -> List[str]:
    models = [item.strip() for item in models_value.split(",") if item.strip()]
    if models:
        return models
    return [fallback]


def sanitize_model_name(name: str) -> str:
    return name.strip().replace("\\", "-").replace("/", "-").replace(" ", "_")


def split_sentence_id(sentence_id: str) -> Tuple[str, int]:
    if not sentence_id:
        return "", -1
    if "_" not in sentence_id:
        return sentence_id, -1
    base, suffix = sentence_id.rsplit("_", 1)
    try:
        idx = int(suffix)
    except ValueError:
        idx = -1
    return base, idx


def load_threads(path: Path, text_column: str, id_column: str) -> Dict[str, List[str]]:
    threads: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    bad_json = 0
    line_order = 0

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line_order += 1
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            sentence_id = str(obj.get(id_column, ""))
            text = str(obj.get(text_column, ""))
            thread_id, idx = split_sentence_id(sentence_id)
            threads[thread_id].append((idx, line_order, text))

    ordered_threads: Dict[str, List[str]] = {}
    for thread_id, items in threads.items():
        if all(idx >= 0 for idx, _, _ in items):
            items.sort(key=lambda item: (item[0], item[1]))
        else:
            items.sort(key=lambda item: item[1])
        ordered_threads[thread_id] = [text for _, _, text in items]

    if bad_json:
        print(f"Warning: skipped {bad_json} bad JSON lines in {path}.")
    return ordered_threads


def build_context_text(
    sentences: List[str],
    target_index: int,
    context_mode: str,
) -> str:
    sents = [s.strip() for s in sentences]
    s_i = sents[target_index]
    n = len(sents)
    mode = context_mode.upper()

    if mode == "C0":
        return f"[TARGET] {s_i} [/TARGET]"

    if mode == "C1":
        parts: List[str] = []
        if target_index - 1 >= 0:
            parts.append(f"[PREV] {sents[target_index - 1]} [/PREV]")
        parts.append(f"[TARGET] {s_i} [/TARGET]")
        if target_index + 1 < n:
            parts.append(f"[NEXT] {sents[target_index + 1]} [/NEXT]")
        return " ".join(parts)

    if mode == "C2":
        parts = []
        start_prev = max(0, target_index - 2)
        prev_block = sents[start_prev:target_index]
        if prev_block:
            parts.append(f"[PREV] {' '.join(prev_block)} [/PREV]")
        parts.append(f"[TARGET] {s_i} [/TARGET]")
        end_next = min(n, target_index + 3)
        next_block = sents[target_index + 1 : end_next]
        if next_block:
            parts.append(f"[NEXT] {' '.join(next_block)} [/NEXT]")
        return " ".join(parts)

    if mode == "C3":
        parts: List[str] = []
        for idx, sent in enumerate(sents):
            if idx == target_index:
                parts.append(f"[TARGET] {sent} [/TARGET]")
            else:
                parts.append(f"[CTX] {sent} [/CTX]")
        return " ".join(parts)

    raise ValueError(f"Unsupported context_mode: {context_mode}")


def iter_context_texts(
    threads: Dict[str, List[str]],
    context_mode: str,
) -> Iterable[str]:
    for sentences in threads.values():
        for idx in range(len(sentences)):
            yield build_context_text(sentences, idx, context_mode)


def batch_lengths(tokenizer, texts: List[str]) -> List[int]:
    enc = tokenizer(
        texts,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return [len(ids) for ids in enc["input_ids"]]


def compute_lengths(
    tokenizer,
    texts: Iterable[str],
    batch_size: int,
) -> List[int]:
    lengths: List[int] = []
    batch: List[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) >= batch_size:
            lengths.extend(batch_lengths(tokenizer, batch))
            batch = []
    if batch:
        lengths.extend(batch_lengths(tokenizer, batch))
    return lengths


def summarize_lengths(lengths: List[int], max_length: int) -> Dict[str, float]:
    if not lengths:
        return {
            "total": 0,
            "over_max": 0,
            "over_max_pct": 0.0,
            "min": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "mean": 0.0,
        }
    arr = np.array(lengths, dtype=np.int32)
    total = int(arr.size)
    over_max = int((arr > max_length).sum())
    return {
        "total": total,
        "over_max": over_max,
        "over_max_pct": (over_max / total * 100.0) if total else 0.0,
        "min": int(arr.min()),
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


def write_summary(
    output_dir: Path,
    rows: List[Dict[str, object]],
    meta: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "token_length_summary_table.txt"
    json_path = output_dir / "token_length_summary.json"

    headers = [
        "split",
        "context_mode",
        "total",
        "over_max",
        "over_max_pct",
        "min",
        "p50",
        "p90",
        "p95",
        "p99",
        "max",
        "mean",
    ]
    string_rows = [
        [str(row.get(h, "")) for h in headers]
        for row in rows
    ]

    widths = [len(header) for header in headers]
    for row in string_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    line = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    header_line = "| " + " | ".join(
        headers[idx].ljust(widths[idx]) for idx in range(len(headers))
    ) + " |"
    with table_path.open("w", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.write(header_line + "\n")
        handle.write(line + "\n")
        for row in string_rows:
            cells = []
            for idx, cell in enumerate(row):
                if idx == 0:
                    cells.append(cell.ljust(widths[idx]))
                else:
                    cells.append(cell.rjust(widths[idx]))
            handle.write("| " + " | ".join(cells) + " |\n")
        handle.write(line + "\n")

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"meta": meta, "rows": rows}, handle, indent=2)

    print(f"Wrote summary table: {table_path}")
    print(f"Wrote summary JSON: {json_path}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    splits = parse_list(args.splits)
    context_modes = parse_context_modes(args.context_modes)
    models = parse_models(args.models, args.model_name)

    print(f"Input dir: {input_dir}")
    print(f"Splits: {', '.join(splits)}")
    print(f"Context modes: {', '.join(context_modes)}")
    print(f"Max length threshold: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Models: {', '.join(models)}")

    for model_name in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        existing_vocab = tokenizer.get_vocab()
        to_add = [tok for tok in SPECIAL_TOKENS if tok not in existing_vocab]
        if to_add:
            tokenizer.add_special_tokens({"additional_special_tokens": to_add})

        print("")
        print(f"Model: {model_name}")
        print(f"Tokenizer model_max_length: {tokenizer.model_max_length}")

        overall_lengths: Dict[str, List[int]] = {
            mode: [] for mode in context_modes
        }

        for split in splits:
            split_path = input_dir / f"{split}.jsonl"
            if not split_path.exists():
                raise SystemExit(f"Missing split file: {split_path}")
            threads = load_threads(
                split_path,
                text_column="sentence",
                id_column="id_sentence_number",
            )

            for mode in context_modes:
                texts_iter = iter_context_texts(threads, mode)
                lengths = compute_lengths(
                    tokenizer, texts_iter, args.batch_size
                )
                overall_lengths[mode].extend(lengths)

        summary_rows: List[Dict[str, object]] = []
        for mode in context_modes:
            stats = summarize_lengths(overall_lengths[mode], args.max_length)
            stats_row = {"split": "all", "context_mode": mode}
            stats_row.update(stats)
            summary_rows.append(stats_row)

            print(
                f"all | {mode} | total={stats['total']} "
                f"over_max={stats['over_max']} ({stats['over_max_pct']:.2f}%) "
                f"p95={stats['p95']} max={stats['max']}"
            )

        meta = {
            "model_name": model_name,
            "input_dir": str(input_dir),
            "context_modes": context_modes,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "tokenizer_model_max_length": tokenizer.model_max_length,
            "summary_scope": args.summary_scope,
        }

        model_dir = Path(args.output_dir) / sanitize_model_name(model_name)
        write_summary(model_dir, summary_rows, meta)


if __name__ == "__main__":
    main()
