from __future__ import annotations

import cProfile
import json
import os
import pickle
import pstats
import time
import tracemalloc
from pathlib import Path
from typing import Any

from cs336_basics import train_bpe


def bytes_to_serializable_str(b: bytes) -> str:
    """
    Store bytes losslessly in JSON using latin-1.
    latin-1 maps byte 0..255 directly to codepoint 0..255.
    """
    return b.decode("latin-1")


def serialize_vocab_json(vocab: dict[int, bytes], output_path: str | os.PathLike) -> None:
    payload = {str(k): bytes_to_serializable_str(v) for k, v in vocab.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def serialize_merges_txt(merges: list[tuple[bytes, bytes]], output_path: str | os.PathLike) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{bytes_to_serializable_str(left)} {bytes_to_serializable_str(right)}\n")


def serialize_pickle(obj: Any, output_path: str | os.PathLike) -> None:
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)


def get_longest_token(vocab: dict[int, bytes]) -> tuple[int, bytes]:
    """
    Longest by raw byte length. Break ties by smaller vocab id.
    """
    best_id = min(vocab.keys())
    best_token = vocab[best_id]
    for token_id, token_bytes in vocab.items():
        if len(token_bytes) > len(best_token):
            best_id = token_id
            best_token = token_bytes
    return best_id, best_token


def format_token_for_display(b: bytes, max_len: int = 200) -> str:
    """
    Human-readable display for possibly weird bytes/newlines.
    """
    s = repr(b)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def profile_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    output_dir: str | os.PathLike,
    top_n_profile_rows: int = 25,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()

    tracemalloc.start()
    start_time = time.perf_counter()

    profiler.enable()
    vocab, merges = train_bpe.train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    profiler.disable()

    end_time = time.perf_counter()
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_sec = end_time - start_time
    peak_mb = peak_bytes / (1024 * 1024)

    longest_token_id, longest_token = get_longest_token(vocab)

    serialize_vocab_json(vocab, output_dir / "vocab.json")
    serialize_merges_txt(merges, output_dir / "merges.txt")

    serialize_pickle(vocab, output_dir / "vocab.pkl")
    serialize_pickle(merges, output_dir / "merges.pkl")

    metrics = {
        "input_path": str(input_path),
        "vocab_size_requested": vocab_size,
        "special_tokens": special_tokens,
        "actual_vocab_size": len(vocab),
        "num_merges": len(merges),
        "elapsed_seconds": elapsed_sec,
        "peak_tracemalloc_mb": peak_mb,
        "longest_token_id": longest_token_id,
        "longest_token_num_bytes": len(longest_token),
        "longest_token_repr": format_token_for_display(longest_token),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    stats_path = output_dir / "profile.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats("cumulative")
        stats.print_stats(top_n_profile_rows)

    print("Training finished.")
    print(f"Input:              {input_path}")
    print(f"Requested vocab:    {vocab_size}")
    print(f"Actual vocab:       {len(vocab)}")
    print(f"Number of merges:   {len(merges)}")
    print(f"Elapsed time:       {elapsed_sec:.3f} seconds")
    print(f"Peak memory:        {peak_mb:.2f} MB (tracemalloc)")
    print(f"Longest token id:   {longest_token_id}")
    print(f"Longest token len:  {len(longest_token)} bytes")
    print(f"Longest token:      {format_token_for_display(longest_token)}")
    print(f"Saved vocab JSON:   {output_dir / 'vocab.json'}")
    print(f"Saved merges TXT:   {output_dir / 'merges.txt'}")
    print(f"Saved metrics JSON: {output_dir / 'metrics.json'}")
    print(f"Saved profile TXT:  {output_dir / 'profile.txt'}")


if __name__ == "__main__":
    INPUT_PATH = "./data/TinyStoriesV2-GPT4-train.txt"
    VOCAB_SIZE = 10_000
    SPECIAL_TOKENS = ["<|endoftext|>"]
    OUTPUT_DIR = "bpe_profile_output"

    profile_train_bpe(
        input_path=INPUT_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        output_dir=OUTPUT_DIR,
        top_n_profile_rows=30,
    )