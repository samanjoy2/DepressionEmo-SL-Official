from transformers import AutoTokenizer, AutoModel
import torch
import argparse


def compare_special_token_handling(model_name: str, special_token: str, sentence: str) -> None:


    tokenizer_std = AutoTokenizer.from_pretrained(model_name)
    model_std = AutoModel.from_pretrained(model_name)


    tokenizer_sp = AutoTokenizer.from_pretrained(model_name)
    tokenizer_sp.add_special_tokens({"additional_special_tokens": [special_token]})
    model_sp = AutoModel.from_pretrained(model_name)
    model_sp.resize_token_embeddings(len(tokenizer_sp))


    std_sentence_tokens = tokenizer_std.tokenize(sentence)
    std_sentence_ids = tokenizer_std.convert_tokens_to_ids(std_sentence_tokens)

    std_marker_pieces = tokenizer_std.tokenize(special_token)
    std_marker_piece_ids = tokenizer_std.convert_tokens_to_ids(std_marker_pieces)


    sp_sentence_tokens = tokenizer_sp.tokenize(sentence)
    sp_sentence_ids = tokenizer_sp.convert_tokens_to_ids(sp_sentence_tokens)

    sp_marker_id = tokenizer_sp.convert_tokens_to_ids(special_token)


    emb_std = model_std.get_input_embeddings().weight
    emb_sp = model_sp.get_input_embeddings().weight

    std_marker_embs = emb_std[torch.tensor(std_marker_piece_ids)]
    sp_marker_emb = emb_sp[sp_marker_id]

    print("\n=== STANDARD TOKENIZATION (marker is NOT special) ===")
    print("Sentence:", sentence)
    print("Sentence tokens:", std_sentence_tokens)
    print("Sentence ids:   ", std_sentence_ids)
    print(f"\nMarker '{special_token}' splits into sub-tokens: {std_marker_pieces}")
    print("Their ids:      ", std_marker_piece_ids)
    print("Their embedding shape:", tuple(std_marker_embs.shape))

    print("\n=== SPECIAL TOKENIZATION (marker IS a special token) ===")
    print("Sentence:", sentence)
    print("Sentence tokens:", sp_sentence_tokens)
    print("Sentence ids:   ", sp_sentence_ids)
    print(f"\nMarker '{special_token}' is a SINGLE token.")
    print("Its id:              ", sp_marker_id)
    print("Its embedding shape: ", tuple(sp_marker_emb.shape))

    print("\nEmbedding matrix sizes:")
    print("Standard vocab size:", emb_std.shape[0], "hidden size:", emb_std.shape[1])
    print("Special  vocab size:", emb_sp.shape[0], "hidden size:", emb_sp.shape[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare handling of a marker as normal vs special token."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google-bert/bert-base-uncased",
        help="HF model name or local path.",
    )
    parser.add_argument(
        "--special_token",
        type=str,
        default="[NEXT]",
        help="Your marker token, e.g. [NEXT] or [SCENE].",
    )
    parser.add_argument(
        "--sentence",
        type=str,
        default="I feel [NEXT] sad.",
        help="Example sentence that contains the marker.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_special_token_handling(
        model_name=args.model_name,
        special_token=args.special_token,
        sentence=args.sentence,
    )
