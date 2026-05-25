import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split a JSONL sentence dataset into train/val/test while "
            "keeping all sentences from the same post together."
        )
    )
    parser.add_argument(
        "--input",
        default=str(
            Path("datasets")
            / "gold"
            / "data.jsonl"
        ),
        help="Path to input JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            Path("datasets")
            / "gold"
            / "splits"
        ),
        help="Directory to write train/val/test JSONL files.",
    )
    parser.add_argument(
        "--ratios",
        default="0.70,0.15,0.15",
        help="Split ratios for train,val,test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for tie-breaking.",
    )
    parser.add_argument(
        "--weight-total",
        type=float,
        default=5.0,
        help="Weight for matching total sentence counts.",
    )
    return parser.parse_args()


def parse_ratios(text):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("ratios must have three comma-separated values")
    values = [float(p) for p in parts]
    total = sum(values)
    if total <= 0:
        raise ValueError("ratios must sum to a positive value")
    return [v / total for v in values]


def group_id_from_sentence_id(sentence_id):
    if not sentence_id:
        return ""
    if "_" not in sentence_id:
        return sentence_id
    return sentence_id.rsplit("_", 1)[0]


def load_groups(path):
    groups = defaultdict(list)
    group_counts = defaultdict(Counter)
    label_counts = Counter()
    total_rows = 0
    bad_json = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            total_rows += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            sentence_id = obj.get("id_sentence_number", "")
            gid = group_id_from_sentence_id(sentence_id)
            label = obj.get("label")
            groups[gid].append(obj)
            group_counts[gid][label] += 1
            label_counts[label] += 1

    return groups, group_counts, label_counts, total_rows, bad_json


def compute_cost(
    split_counts,
    split_total,
    group_counts,
    group_size,
    target_label_counts,
    target_total,
    weight_total,
):
    cost = 0.0
    for label, target in target_label_counts.items():
        if target <= 0:
            continue
        new_count = split_counts.get(label, 0) + group_counts.get(label, 0)
        diff = (new_count - target) / target
        cost += diff * diff
    new_total = split_total + group_size
    total_diff = (new_total - target_total) / target_total
    cost += weight_total * total_diff * total_diff
    return cost


def summarize_split(name, counts, total, targets):
    print(f"{name}: total={total}")
    for label, target in targets.items():
        count = counts.get(label, 0)
        pct = (count / total * 100.0) if total else 0.0
        tgt_pct = (target / sum(targets.values()) * 100.0) if targets else 0.0
        print(f"  {label}: {count} ({pct:.2f}%) target={tgt_pct:.2f}%")


def main():
    args = parse_args()
    ratios = parse_ratios(args.ratios)
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    groups, group_counts, label_counts, total_rows, bad_json = load_groups(
        input_path
    )
    if bad_json:
        print(f"Warning: skipped {bad_json} bad JSON lines.")

    labels = sorted(label_counts.keys())
    total_sentences = sum(label_counts.values())
    raw_targets = [total_sentences * r for r in ratios]
    train_target = int(round(raw_targets[0]))
    val_target = int(round(raw_targets[1]))
    test_target = total_sentences - train_target - val_target
    target_totals = {
        "train": train_target,
        "val": val_target,
        "test": test_target,
    }
    target_label_counts = {
        split: {label: label_counts[label] * ratios[idx] for label in labels}
        for idx, split in enumerate(SPLITS)
    }

    rng = random.Random(args.seed)
    group_items = []
    for gid, counts in group_counts.items():
        size = sum(counts.values())
        group_items.append((gid, size, counts))
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: item[1], reverse=True)
    max_group_size = group_items[0][1] if group_items else 0

    split_state = {
        split: {
            "groups": [],
            "counts": Counter(),
            "total": 0,
        }
        for split in SPLITS
    }

    for gid, size, counts in group_items:
        best_split = None
        best_cost = None
        candidates = [
            split
            for split in SPLITS
            if split_state[split]["total"] + size
            <= target_totals[split] + max_group_size
        ]
        if not candidates:
            candidates = list(SPLITS)
        for split in candidates:
            state = split_state[split]
            cost = compute_cost(
                state["counts"],
                state["total"],
                counts,
                size,
                target_label_counts[split],
                target_totals[split],
                args.weight_total,
            )
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_split = split
        chosen = split_state[best_split]
        chosen["groups"].append(gid)
        chosen["counts"].update(counts)
        chosen["total"] += size

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        out_path = output_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for gid in split_state[split]["groups"]:
                for obj in groups[gid]:
                    handle.write(json.dumps(obj, ensure_ascii=True) + "\n")

    print(f"Input: {input_path}")
    print(f"Total sentences: {total_sentences}")
    print(f"Total posts: {len(groups)}")
    for split in SPLITS:
        summarize_split(
            split,
            split_state[split]["counts"],
            split_state[split]["total"],
            target_label_counts[split],
        )
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
