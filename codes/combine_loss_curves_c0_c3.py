import os
import argparse
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


MODEL_ALIAS: Dict[str, str] = {
    "FacebookAI-roberta-base": "RoBERTa-base",
    "google-bert-bert-base-uncased": "BERT-base",
    "mental-mental-bert-base-uncased": "MentalBERT-base",
    "mental-mental-roberta-base": "MentalRoBERTa-base",
    "microsoft-deberta-v3-base": "DeBERTaV3-base",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each backbone, take loss_curves.png from C0 and C3 runs "
            "and place them side-by-side in a single comparison image."
        )
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="run_reports_advanced",
        help="Directory containing per-run folders (e.g., FacebookAI-roberta-base_C0_...).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("run_reports_advanced", "combined_loss_curves_c0_c3"),
        help="Directory where combined comparison images will be saved.",
    )
    return parser.parse_args()


def discover_c0_c3_pairs(
    report_dir: str,
) -> Dict[str, Tuple[str, str]]:
    if not os.path.isdir(report_dir):
        raise SystemExit(f"Report directory not found: {report_dir}")

    by_backbone: Dict[str, Dict[str, str]] = {}

    for entry in os.listdir(report_dir):
        full = os.path.join(report_dir, entry)
        if not os.path.isdir(full):
            continue

        parts = entry.split("_")
        if len(parts) < 5:
            continue

        context = parts[-4]
        backbone_short = "_".join(parts[:-4])
        backbone_name = MODEL_ALIAS.get(backbone_short, backbone_short)

        if context not in {"C0", "C3"}:
            continue

        by_backbone.setdefault(backbone_name, {})
        by_backbone[backbone_name][context] = full

    pairs: Dict[str, Tuple[str, str]] = {}
    for backbone, ctx_map in by_backbone.items():
        if "C0" in ctx_map and "C3" in ctx_map:
            pairs[backbone] = (ctx_map["C0"], ctx_map["C3"])

    return pairs


def combine_two_images(
    img_left_path: str,
    img_right_path: str,
    out_path: str,
) -> None:
    left = Image.open(img_left_path)
    right = Image.open(img_right_path)


    target_height = min(left.height, right.height)

    def _resize_keep_aspect(im: Image.Image, target_h: int) -> Image.Image:
        if im.height == target_h:
            return im
        ratio = target_h / im.height
        new_w = int(im.width * ratio)
        return im.resize((new_w, target_h), Image.LANCZOS)

    left_r = _resize_keep_aspect(left, target_height)
    right_r = _resize_keep_aspect(right, target_height)

    total_width = left_r.width + right_r.width
    combined = Image.new("RGB", (total_width, target_height), color=(255, 255, 255))
    combined.paste(left_r, (0, 0))
    combined.paste(right_r, (left_r.width, 0))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.save(out_path)


def build_all_models_figure(
    pairs: Dict[str, Tuple[str, str]],
    out_dir: str,
) -> None:
    if not pairs:
        return

    rows: List[Image.Image] = []
    font = ImageFont.load_default()
    title_height = 30
    row_spacing = 10


    preferred_order = [
        "RoBERTa-base",
        "BERT-base",
        "MentalBERT-base",
        "MentalRoBERTa-base",
        "DeBERTaV3-base",
    ]
    backbones = sorted(
        pairs.keys(),
        key=lambda b: preferred_order.index(b)
        if b in preferred_order
        else len(preferred_order),
    )

    for backbone in backbones:
        c0_folder, c3_folder = pairs[backbone]
        img_c0 = os.path.join(c0_folder, "loss_curves.png")
        img_c3 = os.path.join(c3_folder, "loss_curves.png")

        if not os.path.isfile(img_c0) or not os.path.isfile(img_c3):
            continue

        left = Image.open(img_c0)
        right = Image.open(img_c3)

        target_height = min(left.height, right.height)

        def _resize_keep_aspect(im: Image.Image, target_h: int) -> Image.Image:
            if im.height == target_h:
                return im
            ratio = target_h / im.height
            new_w = int(im.width * ratio)
            return im.resize((new_w, target_h), Image.LANCZOS)

        left_r = _resize_keep_aspect(left, target_height)
        right_r = _resize_keep_aspect(right, target_height)

        row_width = left_r.width + right_r.width
        row_height = title_height + target_height

        row_img = Image.new("RGB", (row_width, row_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(row_img)

        title_text = f"{backbone}: C0 (left) vs. C3 (right)"


        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), title_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            text_w, text_h = font.getsize(title_text)
        text_x = (row_width - text_w) // 2
        text_y = (title_height - text_h) // 2
        draw.text((text_x, text_y), title_text, fill=(0, 0, 0), font=font)

        row_img.paste(left_r, (0, title_height))
        row_img.paste(right_r, (left_r.width, title_height))

        rows.append(row_img)

    if not rows:
        return

    total_width = max(r.width for r in rows)
    total_height = sum(r.height for r in rows) + row_spacing * (len(rows) - 1)

    combined = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))

    y_offset = 0
    for row in rows:
        x_offset = (total_width - row.width) // 2
        combined.paste(row, (x_offset, y_offset))
        y_offset += row.height + row_spacing

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "all_models_loss_curves_C0_vs_C3.png")
    combined.save(out_path)
    print(f"Saved combined figure with all models: {out_path}")


def main() -> None:
    args = parse_args()

    pairs = discover_c0_c3_pairs(args.report_dir)
    if not pairs:
        print("No backbones with both C0 and C3 runs found.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    for backbone, (c0_folder, c3_folder) in sorted(pairs.items()):
        img_c0 = os.path.join(c0_folder, "loss_curves.png")
        img_c3 = os.path.join(c3_folder, "loss_curves.png")

        if not os.path.isfile(img_c0) or not os.path.isfile(img_c3):
            print(
                f"[WARN] Skipping {backbone}: missing loss_curves.png for "
                f"{'C0' if not os.path.isfile(img_c0) else 'C3'}"
            )
            continue

        out_name = f"{backbone}_loss_curves_C0_vs_C3.png"
        out_path = os.path.join(args.out_dir, out_name)
        combine_two_images(img_c0, img_c3, out_path)
        print(f"Saved combined loss curves for {backbone}: {out_path}")


    build_all_models_figure(pairs, args.out_dir)


if __name__ == "__main__":
    main()
