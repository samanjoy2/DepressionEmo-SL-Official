import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


VALID_CONTEXT_MODES = {"C0", "C1", "C2", "C3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate context-augmented datasets from sentence-level JSONL splits."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(
            Path("datasets")
            / "gold"
            / "splits"
        ),
        help="Directory containing train.jsonl, val.jsonl, test.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            Path("datasets")
            / "gold"
            / "contexts"
        ),
        help="Output directory for context datasets.",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated split names to process.",
    )
    parser.add_argument(
        "--context-modes",
        default="C0,C1,C2,C3",
        help="Comma-separated context modes (C0,C1,C2,C3).",
    )
    parser.add_argument(
        "--id-column",
        default="id_sentence_number",
        help="Field containing sentence IDs.",
    )
    parser.add_argument(
        "--text-column",
        default="sentence",
        help="Field containing sentence text.",
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


def load_threads(path: Path, id_column: str, text_column: str):
    threads: Dict[str, List[Tuple[int, int, dict]]] = defaultdict(list)
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
            if sentence_id == "":
                continue
            thread_id, idx = split_sentence_id(sentence_id)
            threads[thread_id].append((idx, line_order, obj))

    if bad_json:
        print(f"Warning: skipped {bad_json} bad JSON lines in {path}.")

    thread_order = []
    for thread_id, items in threads.items():
        first_line = min(item[1] for item in items)
        thread_order.append((first_line, thread_id))
    thread_order.sort()

    ordered_threads: List[Tuple[str, List[dict]]] = []
    for _, thread_id in thread_order:
        items = threads[thread_id]
        if all(idx >= 0 for idx, _, _ in items):
            items.sort(key=lambda item: (item[0], item[1]))
        else:
            items.sort(key=lambda item: item[1])
        ordered_threads.append((thread_id, [obj for _, _, obj in items]))

    return ordered_threads


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = parse_list(args.splits)
    context_modes = parse_context_modes(args.context_modes)

    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Splits: {', '.join(splits)}")
    print(f"Context modes: {', '.join(context_modes)}")

    for split in splits:
        split_path = input_dir / f"{split}.jsonl"
        if not split_path.exists():
            raise SystemExit(f"Missing split file: {split_path}")
        ordered_threads = load_threads(
            split_path, id_column=args.id_column, text_column=args.text_column
        )

        for mode in context_modes:
            mode_dir = output_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            out_path = mode_dir / f"{split}.jsonl"

            total_rows = 0
            with out_path.open("w", encoding="utf-8") as handle:
                for thread_id, rows in ordered_threads:
                    sentences = [
                        str(row.get(args.text_column, ""))
                        for row in rows
                    ]
                    thread_size = len(sentences)
                    for idx, row in enumerate(rows):
                        context_text = build_context_text(
                            sentences,
                            target_index=idx,
                            context_mode=mode,
                        )
                        out_row = dict(row)
                        out_row["context_mode"] = mode
                        out_row["context_text"] = context_text
                        out_row["thread_id"] = thread_id
                        out_row["sentence_index"] = idx
                        out_row["thread_size"] = thread_size
                        handle.write(
                            json.dumps(out_row, ensure_ascii=True) + "\n"
                        )
                        total_rows += 1

            print(f"Wrote {total_rows} rows to {out_path}")


if __name__ == "__main__":
    main()
