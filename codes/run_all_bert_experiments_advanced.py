import argparse
import os
import subprocess
import sys
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Python clone of run_all_bert_experiments_advanced.ps1.\n"
            "Runs bert_finetune_emotion_sentences_advanced.py over multiple "
            "models, context modes, seeds, and a freeze depth, skipping runs "
            "that already have a completed report."
        )
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/gold",
        help="Data directory passed to bert_finetune_emotion_sentences_advanced.py.",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="auto",
        choices=["auto", "csv", "jsonl"],
        help="Input file format passed to bert_finetune_emotion_sentences_advanced.py.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="splits/train.jsonl",
        help="Train filename passed to bert_finetune_emotion_sentences_advanced.py.",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="splits/val.jsonl",
        help="Validation filename passed to bert_finetune_emotion_sentences_advanced.py.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="splits/test.jsonl",
        help="Test filename passed to bert_finetune_emotion_sentences_advanced.py.",
    )
    parser.add_argument(
        "--item_id_column",
        type=str,
        default="id_sentence_number",
        help="Item ID column passed to bert_finetune_emotion_sentences_advanced.py.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="sentence",
        help="Text column passed to bert_finetune_emotion_sentences_advanced.py.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Label column passed to bert_finetune_emotion_sentences_advanced.py.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[67, 68, 69, 70, 71],
        help="List of random seeds to run (default: 67 68 69 70 71).",
    )
    parser.add_argument(
        "--freeze_encoder_layers",
        type=int,
        default=0,
        help="How many lower encoder layers to freeze (default: 0).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader workers to pass to the training script (default: 2).",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Training batch size to pass to bert_finetune_emotion_sentences_advanced.py (default: 16).",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Evaluation batch size to pass to bert_finetune_emotion_sentences_advanced.py (default: 32).",
    )
    parser.add_argument(
        "--python_executable",
        type=str,
        default=sys.executable,
        help="Python executable to use for launching training (default: current interpreter).",
    )
    return parser.parse_args()


def get_models() -> List[str]:
    return [
        "microsoft/deberta-v3-base",
        "google-bert/bert-base-uncased",
        "FacebookAI/roberta-base",
        "mental/mental-bert-base-uncased",
        "mental/mental-roberta-base",


    ]


def get_contexts() -> List[str]:
    return ["C0", "C1", "C2", "C3"]


def main() -> None:
    args = parse_args()


    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    os.chdir(repo_root)

    epochs = args.epochs
    data_dir = args.data_dir
    data_format = args.data_format
    train_file = args.train_file
    val_file = args.val_file
    test_file = args.test_file
    item_id_column = args.item_id_column
    text_column = args.text_column
    label_column = args.label_column
    seeds = args.seeds
    freeze = args.freeze_encoder_layers
    num_workers = args.num_workers
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    py = args.python_executable

    models = get_models()
    contexts = get_contexts()

    report_dir = "run_reports_advanced"

    print("Starting advanced batch experiments...")
    print(
        f"Epochs: {epochs} | Data dir: {data_dir} | Data format: {data_format} | "
        f"Seeds: {', '.join(str(s) for s in seeds)} | "
        f"FreezeEncoderLayers: {freeze} | num_workers: {num_workers} | "
        f"train_batch_size: {train_batch_size} | eval_batch_size: {eval_batch_size}"
    )

    for model in models:
        for ctx in contexts:
            for seed in seeds:
                model_short = (
                    model.strip()
                    .replace("\\", "-")
                    .replace("/", "-")
                    .replace(" ", "_")
                )


                run_folder_name = (
                    f"{model_short}_{ctx}_{epochs}epochs_seed{seed}_freeze{freeze}"
                )
                run_folder_path = os.path.join(report_dir, run_folder_name)


                run_report_name = f"{model_short}_{ctx}_{epochs}epochs_seed{seed}_freeze{freeze}.txt"
                run_report_path = os.path.join(run_folder_path, run_report_name)

                if os.path.isfile(run_report_path):
                    print()
                    print("===============================================")
                    print(
                        f"Skipping model: {model} | context_mode: {ctx} | "
                        f"seed: {seed} | freeze: {freeze}"
                    )
                    print(f"Completed run already found at: {run_report_path}")
                    print("===============================================")
                    continue

                print()
                print("===============================================")
                print(
                    f"Running model: {model} | context_mode: {ctx} | "
                    f"seed: {seed} | freeze: {freeze}"
                )
                print("===============================================")

                cmd = [
                    py,
                    os.path.join("codes", "bert_finetune_emotion_sentences_advanced.py"),
                    "--model_name_or_path",
                    model,
                    "--context_mode",
                    ctx,
                    "--num_epochs",
                    str(epochs),
                    "--data_dir",
                    data_dir,
                    "--data_format",
                    data_format,
                    "--train_file",
                    train_file,
                    "--val_file",
                    val_file,
                    "--test_file",
                    test_file,
                    "--item_id_column",
                    item_id_column,
                    "--text_column",
                    text_column,
                    "--label_column",
                    label_column,
                    "--seed",
                    str(seed),
                    "--freeze_encoder_layers",
                    str(freeze),
                    "--num_workers",
                    str(num_workers),
                    "--train_batch_size",
                    str(train_batch_size),
                    "--eval_batch_size",
                    str(eval_batch_size),
                ]

                result = subprocess.run(cmd)

                if result.returncode != 0:
                    print(
                        f"Run failed for model={model} context_mode={ctx} "
                        f"seed={seed} freeze={freeze} "
                        f"(exit code {result.returncode})"
                    )
                else:
                    print(
                        f"Finished model={model} context_mode={ctx} "
                        f"seed={seed} freeze={freeze}"
                    )

    print()
    print(
        "All advanced experiments completed. "
        "Check the run_reports_advanced folder for per-run seed/freeze-specific folders."
    )


if __name__ == "__main__":
    main()
