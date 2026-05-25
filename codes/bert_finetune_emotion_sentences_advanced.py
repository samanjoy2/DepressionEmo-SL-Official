import os
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    accuracy_score,
)
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


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


@dataclass
class TrainingConfig:
    model_name_or_path: str = "bert-base-uncased"
    data_dir: str = "datasets/gold"
    data_format: str = "auto"

    train_file: str = "splits/train.jsonl"
    val_file: str = "splits/val.jsonl"
    test_file: str = "splits/test.jsonl"

    use_all_file_split: bool = False
    all_file: str = "data.jsonl"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    item_id_column: str = "id_sentence_number"

    text_column: str = "sentence"
    label_column: str = "label"

    context_mode: str = "C0"

    max_length: int = 512
    auto_max_length: bool = False
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 1e-2
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    seed: int = 67
    num_workers: int = 0
    freeze_encoder_layers: int = 0
    report_dir: str = "run_reports_advanced"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_context_text(
    sentences: List[str],
    target_index: int,
    item_id: str,
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
            prev_sent = sents[target_index - 1]
            parts.append(f"[PREV] {prev_sent} [/PREV]")
        parts.append(f"[TARGET] {s_i} [/TARGET]")
        if target_index + 1 < n:
            next_sent = sents[target_index + 1]
            parts.append(f"[NEXT] {next_sent} [/NEXT]")
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

        blocks: List[str] = []
        for idx, sent in enumerate(sents):
            if idx == target_index:
                blocks.append(f"[TARGET] {sent} [/TARGET]")
            else:
                blocks.append(f"[CTX] {sent} [/CTX]")
        return " ".join(blocks)

    raise ValueError(f"Unsupported context_mode: {context_mode}")


class EmotionContextDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        label_encoder: LabelEncoder,
        text_column: str,
        label_column: str,
        item_id_column: str,
        context_mode: str,
        max_length: int,
    ):
        self.tokenizer = tokenizer
        self.context_mode = context_mode
        self.max_length = max_length

        if item_id_column not in df.columns:
            raise ValueError(
                f"Column '{item_id_column}' (item_id_column) not found in DataFrame."
            )

        self.texts: List[str] = []
        labels_int: List[int] = []


        self.example_ids: List[str] = []


        df = df.copy()

        df["_thread_id"] = (
            df[item_id_column].astype(str).str.split("_").str[0]
        )


        grouped = df.groupby("_thread_id", sort=False)
        for thread_id, group in grouped:

            sentences = group[text_column].astype(str).tolist()
            labels_str = group[label_column].astype(str).tolist()
            labels_encoded = label_encoder.transform(labels_str)
            item_ids = group[item_id_column].astype(str).tolist()

            for idx in range(len(sentences)):
                ctx_text = build_context_text(
                    sentences,
                    target_index=idx,
                    item_id=str(thread_id),
                    context_mode=context_mode,
                )
                self.texts.append(ctx_text)
                labels_int.append(int(labels_encoded[idx]))

                self.example_ids.append(item_ids[idx])

        self.labels: np.ndarray = np.array(labels_int, dtype=np.int64)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)


        item["example_id"] = self.example_ids[idx]
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_data_format(path: str, data_format: str) -> str:
    fmt = (data_format or "auto").lower()
    if fmt != "auto":
        return fmt
    ext = os.path.splitext(path)[1].lower()
    if ext in {".jsonl", ".json"}:
        return "jsonl"
    return "csv"


def read_dataframe(path: str, data_format: str) -> pd.DataFrame:
    fmt = detect_data_format(path, data_format)
    if fmt == "jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)

def load_splits(
    cfg: TrainingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    if cfg.use_all_file_split:
        all_path = os.path.join(cfg.data_dir, cfg.all_file)
        df_all = read_dataframe(all_path, cfg.data_format)


        for col in [cfg.item_id_column, cfg.text_column, cfg.label_column]:
            if col not in df_all.columns:
                raise ValueError(f"Column '{col}' not found in all_file CSV.")

        item_ids = df_all[cfg.item_id_column].astype(str).unique()
        rng = np.random.default_rng(cfg.seed)
        rng.shuffle(item_ids)

        n_items = len(item_ids)
        if n_items == 0:
            raise ValueError("No item_ids found in all_file CSV.")

        if cfg.train_ratio <= 0 or cfg.val_ratio < 0 or cfg.test_ratio < 0:
            raise ValueError("Train/val/test ratios must be non-negative.")
        if abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) > 1e-6:
            raise ValueError(
                "Train/val/test ratios must sum to 1.0 "
                f"(got {cfg.train_ratio + cfg.val_ratio + cfg.test_ratio})."
            )

        n_train = int(n_items * cfg.train_ratio)
        n_val = int(n_items * cfg.val_ratio)
        n_test = n_items - n_train - n_val

        train_ids = set(item_ids[:n_train])
        val_ids = set(item_ids[n_train : n_train + n_val])
        test_ids = set(item_ids[n_train + n_val :])

        train_df = df_all[df_all[cfg.item_id_column].astype(str).isin(train_ids)]
        val_df = df_all[df_all[cfg.item_id_column].astype(str).isin(val_ids)]
        test_df = df_all[df_all[cfg.item_id_column].astype(str).isin(test_ids)]

        print(
            f"Split from all_file by {cfg.item_id_column}: "
            f"{len(train_ids)} train IDs, {len(val_ids)} val IDs, {len(test_ids)} test IDs."
        )
    else:
        train_path = os.path.join(cfg.data_dir, cfg.train_file)
        val_path = os.path.join(cfg.data_dir, cfg.val_file)
        test_path = os.path.join(cfg.data_dir, cfg.test_file)

        train_df = read_dataframe(train_path, cfg.data_format)
        val_df = read_dataframe(val_path, cfg.data_format)
        test_df = read_dataframe(test_path, cfg.data_format)


        for col in [cfg.text_column, cfg.label_column, cfg.item_id_column]:
            if col not in train_df.columns:
                raise ValueError(f"Column '{col}' not found in train CSV.")
            if col not in val_df.columns:
                raise ValueError(f"Column '{col}' not found in val CSV.")
            if col not in test_df.columns:
                raise ValueError(f"Column '{col}' not found in test CSV.")


    label_encoder = LabelEncoder()
    train_labels = train_df[cfg.label_column].astype(str)
    label_encoder.fit(train_labels)


    for split_name, df in [("val", val_df), ("test", test_df)]:
        unknown = set(df[cfg.label_column].astype(str)) - set(label_encoder.classes_)
        if unknown:
            raise ValueError(
                f"Found labels in {split_name} not present in train split: {unknown}"
            )

    print("Label classes:", list(label_encoder.classes_))
    return train_df, val_df, test_df, label_encoder


def create_dataloaders(
    cfg: TrainingConfig,
    tokenizer: AutoTokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_encoder: LabelEncoder,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = EmotionContextDataset(
        train_df,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        text_column=cfg.text_column,
        label_column=cfg.label_column,
        item_id_column=cfg.item_id_column,
        context_mode=cfg.context_mode,
        max_length=cfg.max_length,
    )
    val_dataset = EmotionContextDataset(
        val_df,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        text_column=cfg.text_column,
        label_column=cfg.label_column,
        item_id_column=cfg.item_id_column,
        context_mode=cfg.context_mode,
        max_length=cfg.max_length,
    )
    test_dataset = EmotionContextDataset(
        test_df,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        text_column=cfg.text_column,
        label_column=cfg.label_column,
        item_id_column=cfg.item_id_column,
        context_mode=cfg.context_mode,
        max_length=cfg.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def analyze_and_set_max_length(
    cfg: TrainingConfig,
    tokenizer: AutoTokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_encoder: LabelEncoder,
    log_fn,
) -> None:
    log_fn("Analyzing sequence lengths to choose an appropriate max_length...")

    def build_texts(df: pd.DataFrame) -> List[str]:
        ds = EmotionContextDataset(
            df,
            tokenizer=tokenizer,
            label_encoder=label_encoder,
            text_column=cfg.text_column,
            label_column=cfg.label_column,
            item_id_column=cfg.item_id_column,
            context_mode=cfg.context_mode,
            max_length=cfg.max_length,
        )
        return ds.texts

    all_texts: List[str] = []
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        texts = build_texts(df)
        log_fn(f"{name.capitalize()} split: {len(texts)} contextualized sentences.")
        all_texts.extend(texts)

    if not all_texts:
        log_fn("No texts found; keeping existing max_length.")
        return


    lengths: List[int] = []
    batch_size = 512
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = enc["input_ids"]
        lengths.extend(len(ids) for ids in input_ids)

    lengths_arr = np.array(lengths, dtype=np.int32)
    min_len = int(lengths_arr.min())
    max_len = int(lengths_arr.max())
    median_len = int(np.percentile(lengths_arr, 50))
    p90 = int(np.percentile(lengths_arr, 90))
    p95 = int(np.percentile(lengths_arr, 95))
    p99 = int(np.percentile(lengths_arr, 99))

    log_fn(
        "Sequence length stats (all splits, after context formatting): "
        f"min={min_len}, median={median_len}, p90={p90}, p95={p95}, "
        f"p99={p99}, max={max_len}"
    )


    recommended = p95
    model_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max, int) and 0 < model_max < 10000:
        recommended = min(recommended, model_max)


    recommended = max(recommended, 16)

    log_fn(f"Auto-selected max_length={recommended} (was {cfg.max_length}).")
    cfg.max_length = int(recommended)


def build_model_and_tokenizer(
    cfg: TrainingConfig,
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
):
    model_name = cfg.model_name_or_path


    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(
            f"Fast tokenizer error '{e}'; trying a slow tokenizer for {model_name}."
        )

        lower_name = model_name.lower()
        if "deberta-v3" in lower_name:
            try:
                from transformers import DebertaV2Tokenizer

                tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
                print("Loaded DebertaV2Tokenizer (slow) for DeBERTa v3.")
            except Exception as e2:
                print(
                    f"DebertaV2Tokenizer load failed ('{e2}'); "
                    "falling back to AutoTokenizer(use_fast=False)."
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, use_fast=False
                )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=False
            )


    existing_vocab = tokenizer.get_vocab()
    to_add = [tok for tok in SPECIAL_TOKENS if tok not in existing_vocab]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    if to_add:
        model.resize_token_embeddings(len(tokenizer))

    model.to(cfg.device)
    return model, tokenizer


def freeze_encoder_backbone_layers(
    model: AutoModelForSequenceClassification,
    num_layers_to_freeze: int,
    log_fn,
) -> None:
    if num_layers_to_freeze <= 0:
        log_fn("Not freezing any encoder layers (freeze_encoder_layers=0).")
        return

    backbone = None
    backbone_name = None
    for name in [
        "bert",
        "roberta",
        "deberta",
        "deberta_v2",
        "deberta_v3",
        "distilbert",
        "albert",
        "xlm_roberta",
    ]:
        if hasattr(model, name):
            backbone = getattr(model, name)
            backbone_name = name
            break

    if backbone is None:
        log_fn(
            "Could not identify a known encoder backbone on the model; "
            "skipping layer freezing."
        )
        return

    encoder = getattr(backbone, "encoder", None)
    layers = None
    if encoder is not None and hasattr(encoder, "layer"):
        layers = encoder.layer
    elif hasattr(backbone, "transformer") and hasattr(
        backbone.transformer, "layer"
    ):
        layers = backbone.transformer.layer
    elif hasattr(backbone, "layer"):
        layers = backbone.layer

    if layers is None:
        log_fn(
            f"Could not locate encoder layers for backbone '{backbone_name}'; "
            "skipping layer freezing."
        )
        return

    total_layers = len(layers)
    n_to_freeze = min(max(num_layers_to_freeze, 0), total_layers)
    if n_to_freeze <= 0:
        log_fn(
            f"freeze_encoder_layers={num_layers_to_freeze} resulted in 0 "
            "layers to freeze; skipping."
        )
        return


    if hasattr(backbone, "embeddings"):
        for p in backbone.embeddings.parameters():
            p.requires_grad = False
        log_fn(f"Froze embeddings of backbone '{backbone_name}'.")

    for idx, layer in enumerate(layers):
        if idx < n_to_freeze:
            for p in layer.parameters():
                p.requires_grad = False
            log_fn(
                f"Froze encoder layer {idx} / {total_layers - 1} "
                f"of backbone '{backbone_name}'."
            )
        else:
            log_fn(
                f"Encoder layer {idx} of backbone '{backbone_name}' "
                "remains trainable."
            )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    log_fn(
        f"Trainable parameters after freezing: "
        f"{trainable_params:,} / {total_params:,}."
    )


def evaluate(
    model: AutoModelForSequenceClassification,
    data_loader: DataLoader,
    device: str,
    label_names: List[str],
    desc: str = "Eval",
) -> Tuple[float, str, float, float]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc, leave=False):
            inputs = {
                k: v.to(device) for k, v in batch.items() if k != "example_id"
            }
            outputs = model(**inputs)
            loss = outputs.loss

            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            logits = outputs.logits

            batch_size = batch["labels"].size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    avg_loss = total_loss / max(total_examples, 1)
    macro_f1 = f1_score(
        all_labels, all_preds, average="macro", zero_division=0
    )
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=label_names, digits=4
    )
    return avg_loss, report, macro_f1, acc


def collect_logits_and_labels(
    model: AutoModelForSequenceClassification,
    data_loader: DataLoader,
    device: str,
    desc: str = "Logits",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_ids: List[str] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc, leave=False):
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k != "example_id"
            }
            labels = batch["labels"]
            example_ids = batch.get("example_id")
            outputs = model(**inputs)
            logits = outputs.logits

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

            if example_ids is not None:

                if isinstance(example_ids, (list, tuple)):
                    all_ids.extend([str(x) for x in example_ids])
                elif isinstance(example_ids, torch.Tensor):
                    all_ids.extend(
                        [str(x) for x in example_ids.cpu().numpy().tolist()]
                    )
                else:
                    all_ids.append(str(example_ids))

    if not all_logits:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
        )

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    if not all_ids:
        all_ids = [str(i) for i in range(logits_tensor.size(0))]

    return (
        logits_tensor.numpy(),
        labels_tensor.numpy(),
        all_ids,
    )


def train(cfg: TrainingConfig) -> None:

    log_lines: List[str] = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    set_seed(cfg.seed)

    log(f"Using device: {cfg.device}")
    log(f"Loading data from: {cfg.data_dir}")
    log(
        f"Model: {cfg.model_name_or_path} | Context mode: {cfg.context_mode} | "
        f"Epochs: {cfg.num_epochs}"
    )
    log(
        "Hyperparameters: "
        f"max_length={cfg.max_length}, "
        f"train_batch_size={cfg.train_batch_size}, "
        f"eval_batch_size={cfg.eval_batch_size}, "
        f"learning_rate={cfg.learning_rate}, "
        f"weight_decay={cfg.weight_decay}, "
        f"warmup_ratio={cfg.warmup_ratio}, "
        f"gradient_accumulation_steps={cfg.gradient_accumulation_steps}, "
        f"seed={cfg.seed}, "
        f"freeze_encoder_layers={cfg.freeze_encoder_layers}"
    )


    os.makedirs(cfg.report_dir, exist_ok=True)
    train_df, val_df, test_df, label_encoder = load_splits(cfg)
    num_labels = len(label_encoder.classes_)
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in id2label.items()}

    model_short = (
        cfg.model_name_or_path.strip()
        .replace("\\", "-")
        .replace("/", "-")
        .replace(" ", "_")
    )
    run_name = (
        f"{model_short}_{cfg.context_mode}_{cfg.num_epochs}epochs_"
        f"seed{cfg.seed}_freeze{cfg.freeze_encoder_layers}"
    )
    run_dir = os.path.join(cfg.report_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)


    log(f"Loading model and tokenizer: {cfg.model_name_or_path}")
    model, tokenizer = build_model_and_tokenizer(
        cfg, num_labels=num_labels, id2label=id2label, label2id=label2id
    )


    freeze_encoder_backbone_layers(model, cfg.freeze_encoder_layers, log)


    if cfg.device == "cuda" and torch.cuda.device_count() > 1:
        log(
            f"Using torch.nn.DataParallel on {torch.cuda.device_count()} GPUs "
            "to speed up training."
        )
        model = torch.nn.DataParallel(model)


    if cfg.auto_max_length:
        analyze_and_set_max_length(
            cfg, tokenizer, train_df, val_df, test_df, label_encoder, log
        )


    train_loader, val_loader, test_loader = create_dataloaders(
        cfg, tokenizer, train_df, val_df, test_df, label_encoder
    )


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=cfg.learning_rate
    )

    total_steps = (
        len(train_loader)
        * cfg.num_epochs
        // max(cfg.gradient_accumulation_steps, 1)
    )
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


    global_step = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_macro_f1_scores: List[float] = []
    val_macro_f1_scores: List[float] = []
    train_accuracies: List[float] = []
    val_accuracies: List[float] = []
    best_val_metric = -float("inf")
    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = 0
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_examples = 0
        epoch_train_preds: List[int] = []
        epoch_train_labels: List[int] = []

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}", leave=False
        )

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress_bar):
            inputs = {
                k: v.to(cfg.device)
                for k, v in batch.items()
                if k != "example_id"
            }
            outputs = model(**inputs)
            loss = outputs.loss


            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            logits = outputs.logits

            batch_size = batch["labels"].size(0)
            epoch_loss += float(loss.item()) * batch_size
            epoch_examples += batch_size

            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            labels_np = batch["labels"].detach().cpu().numpy()
            epoch_train_preds.extend(preds.tolist())
            epoch_train_labels.extend(labels_np.tolist())

            loss = loss / max(cfg.gradient_accumulation_steps, 1)
            loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        avg_train_loss = epoch_loss / max(epoch_examples, 1)
        train_losses.append(avg_train_loss)

        if epoch_train_labels:
            train_acc = float(
                (np.array(epoch_train_preds) == np.array(epoch_train_labels)).mean()
            )
            train_macro_f1_epoch = f1_score(
                epoch_train_labels,
                epoch_train_preds,
                average="macro",
                zero_division=0,
            )
        else:
            train_acc = 0.0
            train_macro_f1_epoch = 0.0
        train_accuracies.append(train_acc)
        train_macro_f1_scores.append(train_macro_f1_epoch)

        log("")
        log(
            f"Epoch {epoch:02d} | Train loss = {avg_train_loss:.4f} "
            f"| Train accuracy = {train_acc:.4f} "
            f"| Train macro F1 = {train_macro_f1_epoch:.4f}"
        )


        val_loss, val_report, val_macro_f1, val_accuracy = evaluate(
            model, val_loader, cfg.device, label_names=list(label_encoder.classes_), desc="Validation"
        )
        log(
            f"Epoch {epoch:02d} | Val loss = {val_loss:.4f} "
            f"| Val accuracy = {val_accuracy:.4f} "
            f"| Val macro F1 = {val_macro_f1:.4f}"
        )
        log("Validation classification report:")
        log(val_report)
        val_losses.append(val_loss)
        val_macro_f1_scores.append(val_macro_f1)
        val_accuracies.append(val_accuracy)


        if val_macro_f1 > best_val_metric:
            best_val_metric = val_macro_f1
            best_val_loss = val_loss
            best_epoch = epoch

            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            log(
                f"New best model at epoch {epoch:02d} "
                f"(Val loss = {val_loss:.4f})"
            )


    if best_state_dict is not None:
        log("")
        log(
            f"Loading best model from epoch {best_epoch:02d} "
            f"(Val macro F1 = {best_val_metric:.4f}, "
            f"Val loss = {best_val_loss:.4f}) for final test evaluation."
        )
        model.load_state_dict(
            {k: v.to(cfg.device) for k, v in best_state_dict.items()}
        )

    log("")
    log("=== Final evaluation on test set ===")
    test_loss, test_report, test_macro_f1, test_accuracy = evaluate(
        model, test_loader, cfg.device, label_names=list(label_encoder.classes_), desc="Test"
    )
    log(f"Test loss = {test_loss:.4f}")
    log(f"Test macro F1 = {test_macro_f1:.4f}")
    log(f"Test accuracy = {test_accuracy:.4f}")
    log("Test classification report:")
    log(test_report)


    model_to_save = model.module if hasattr(model, "module") else model
    try:

        model_to_save.save_pretrained(run_dir, safe_serialization=True)
    except TypeError:
        log(
            "Warning: 'safe_serialization' not supported by this transformers "
            "version; saving model with standard serialization instead."
        )
        model_to_save.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)

    labels_path = os.path.join(run_dir, "label_classes.txt")
    with open(labels_path, "w", encoding="utf-8") as lf:
        for label in label_encoder.classes_:
            lf.write(str(label) + "\n")
    log(f"Saved best model, tokenizer, and labels to: {run_dir}")


    try:
        epochs_range = list(range(1, len(train_losses) + 1))
        metrics_df = pd.DataFrame(
            {
                "epoch": epochs_range,
                "train_loss": train_losses,
                "val_loss": val_losses,
                "train_accuracy": train_accuracies,
                "val_accuracy": val_accuracies,
                "train_macro_f1": train_macro_f1_scores,
                "val_macro_f1": val_macro_f1_scores,
            }
        )
        metrics_path = os.path.join(run_dir, "metrics_per_epoch.csv")
        metrics_df.to_csv(metrics_path, index=False)
        log(f"Saved per-epoch metrics to: {metrics_path}")
    except Exception as e:
        log(f"Failed to save metrics CSV: {e}")

    try:
        import matplotlib.pyplot as plt

        if train_losses and val_losses:
            epochs_range = list(range(1, len(train_losses) + 1))


            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(epochs_range, train_losses, label="Train loss")
            ax_loss.plot(epochs_range, val_losses, label="Validation loss")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.set_title("Train vs validation loss")
            ax_loss.legend()
            fig_loss.tight_layout()
            loss_plot_path = os.path.join(run_dir, "loss_curves.png")
            fig_loss.savefig(loss_plot_path, dpi=200)
            plt.close(fig_loss)
            log(f"Saved loss curves plot to: {loss_plot_path}")

        if val_macro_f1_scores:
            epochs_range = list(range(1, len(val_macro_f1_scores) + 1))
            fig_f1, ax_f1 = plt.subplots()
            ax_f1.plot(epochs_range, val_macro_f1_scores, label="Validation macro F1")
            ax_f1.axvline(best_epoch, color="red", linestyle="--", label="Best epoch")
            ax_f1.set_xlabel("Epoch")
            ax_f1.set_ylabel("Macro F1")
            ax_f1.set_title("Validation macro F1 by epoch")
            ax_f1.legend()
            fig_f1.tight_layout()
            f1_plot_path = os.path.join(run_dir, "val_macro_f1.png")
            fig_f1.savefig(f1_plot_path, dpi=200)
            plt.close(fig_f1)
            log(f"Saved macro F1 plot to: {f1_plot_path}")


        if (
            train_losses
            and val_losses
            and train_accuracies
            and val_accuracies
        ):
            epochs_range = list(range(1, len(train_losses) + 1))
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(8, 8), sharex=True
            )


            ax1.plot(epochs_range, train_losses, label="Train loss")
            ax1.plot(epochs_range, val_losses, label="Validation loss")
            ax1.set_ylabel("Loss")
            ax1.set_title("Train vs validation loss")
            ax1.legend()


            ax2.plot(epochs_range, train_accuracies, label="Train accuracy")
            ax2.plot(epochs_range, val_accuracies, label="Validation accuracy")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.set_title("Train vs validation accuracy")
            ax2.legend()

            fig.tight_layout()
            combined_path = os.path.join(run_dir, "loss_and_score_curves.png")
            fig.savefig(combined_path, dpi=200)
            plt.close(fig)
            log(f"Saved combined loss/accuracy plot to: {combined_path}")
    except Exception as e:
        log(f"Could not create training/validation plots: {e}")


    try:
        test_logits, test_labels, test_example_ids = collect_logits_and_labels(
            model, test_loader, cfg.device, desc="Test logits"
        )
        if test_logits.size > 0:
            num_labels = test_logits.shape[1]
            label_names = list(label_encoder.classes_)
            pred_ids = test_logits.argmax(axis=-1)

            logits_df = pd.DataFrame()
            logits_df["example_id"] = test_example_ids
            logits_df["true_label_id"] = test_labels
            logits_df["true_label"] = [label_names[i] for i in test_labels]
            logits_df["pred_label_id"] = pred_ids
            logits_df["pred_label"] = [label_names[i] for i in pred_ids]
            for idx, label_name in enumerate(label_names):
                logits_df[f"logit_{label_name}"] = test_logits[:, idx]

            logits_path = os.path.join(run_dir, "test_logits.csv")
            logits_df.to_csv(logits_path, index=False)
            log(f"Saved test logits to: {logits_path}")


            cm = confusion_matrix(
                test_labels,
                pred_ids,
                labels=list(range(num_labels)),
            )
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
                fig.colorbar(im, ax=ax)
                tick_marks = np.arange(num_labels)
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(label_names, rotation=45, ha="right")
                ax.set_yticklabels(label_names)
                ax.set_ylabel("True label")
                ax.set_xlabel("Predicted label")
                ax.set_title("Confusion matrix (test)")
                thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
                for i in range(num_labels):
                    for j in range(num_labels):
                        ax.text(
                            j,
                            i,
                            format(cm[i, j], "d"),
                            ha="center",
                            va="center",
                            color=(
                                "white"
                                if cm[i, j] > thresh
                                else "black"
                            ),
                        )
                fig.tight_layout()
                cm_path = os.path.join(run_dir, "confusion_matrix_test.png")
                fig.savefig(cm_path, dpi=200)
                plt.close(fig)
                log(f"Saved test confusion matrix plot to: {cm_path}")
            except Exception as e:
                log(f"Could not save confusion matrix plot: {e}")
    except Exception as e:
        log(f"Failed to collect/save test logits: {e}")


    run_report_filename = (
        f"{model_short}_{cfg.context_mode}_{cfg.num_epochs}epochs_"
        f"seed{cfg.seed}_freeze{cfg.freeze_encoder_layers}.txt"
    )
    run_report_path = os.path.join(run_dir, run_report_filename)
    with open(run_report_path, "w", encoding="utf-8") as f:
        for line in log_lines:
            if line.endswith("\n"):
                f.write(line)
            else:
                f.write(line + "\n")


    generic_report_path = os.path.join(run_dir, "run_report.txt")
    with open(generic_report_path, "w", encoding="utf-8") as f:
        for line in log_lines:
            if line.endswith("\n"):
                f.write(line)
            else:
                f.write(line + "\n")

    print(f"Saved run report to: {run_report_path}")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune any BERT-like model on project "
            "sentence-level emotion labels."
        )
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help=(
            "Model name or path for Hugging Face AutoModelForSequenceClassification "
            "(e.g., 'bert-base-uncased', 'bert-base-multilingual-cased')."
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/gold",
        help="Directory containing train/val/test CSV or JSONL files.",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="auto",
        choices=["auto", "csv", "jsonl"],
        help="Input file format (auto=detect by file extension).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="splits/train.jsonl",
        help="Train filename (inside data_dir).",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="splits/val.jsonl",
        help="Validation filename (inside data_dir).",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="splits/test.jsonl",
        help="Test filename (inside data_dir).",
    )
    parser.add_argument(
        "--use_all_file_split",
        action="store_true",
        help=(
            "If set, ignore train/val/test files and instead split a single "
            "CSV or JSONL by item_id into train/dev/test."
        ),
    )
    parser.add_argument(
        "--all_file",
        type=str,
        default="data.jsonl",
        help="Combined filename (inside data_dir) when using --use_all_file_split.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Proportion of item_ids to assign to train when using --use_all_file_split.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Proportion of item_ids to assign to validation when using --use_all_file_split.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Proportion of item_ids to assign to test when using --use_all_file_split.",
    )
    parser.add_argument(
        "--item_id_column",
        type=str,
        default="id_sentence_number",
        help="Name of the column containing item/thread IDs.",
    )
    parser.add_argument(
        "--context_mode",
        type=str,
        default="C0",
        choices=["C0", "C1", "C2", "C3"],
        help=(
            "Context condition for building input text: "
            "C0=no context; C1=+/-1; C2=+/-2; C3=full thread with [CTX]/[TARGET]."
        ),
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="sentence",
        help="Name of the text column in the input files.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Name of the label column in the input files.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--auto_max_length",
        action="store_true",
        help=(
            "If set, ignore the static --max_length value and instead "
            "analyze the dataset after context formatting to choose a "
            "max_length that covers most sequences (based on token counts)."
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Train batch size.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Fraction of total steps for learning rate warmup.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Accumulate gradients over this many steps before optimizer step.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=67,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers. Set >0 for speed if supported.",
    )
    parser.add_argument(
        "--freeze_encoder_layers",
        type=int,
        default=0,
        help=(
            "Number of lower encoder layers to freeze in the backbone "
            "(0 = no freezing)."
        ),
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="run_reports_advanced",
        help="Directory where per-run text reports (losses + test metrics) are saved.",
    )

    args = parser.parse_args()

    cfg = TrainingConfig(
        model_name_or_path=args.model_name_or_path,
        data_dir=args.data_dir,
        data_format=args.data_format,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        use_all_file_split=args.use_all_file_split,
        all_file=args.all_file,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        item_id_column=args.item_id_column,
        text_column=args.text_column,
        label_column=args.label_column,
        context_mode=args.context_mode,
        max_length=args.max_length,
        auto_max_length=args.auto_max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        num_workers=args.num_workers,
        freeze_encoder_layers=args.freeze_encoder_layers,
        report_dir=args.report_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return cfg


if __name__ == "__main__":
    config = parse_args()
    train(config)
