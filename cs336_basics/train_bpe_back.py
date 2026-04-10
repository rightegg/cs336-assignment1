import os
from collections import defaultdict
from multiprocessing import Pool
from typing import BinaryIO
import sys

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def process_chunk(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    counts: dict[tuple[bytes, ...], int] = defaultdict(int)

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")

    if len(special_tokens) > 0:
        special_pat = "|".join(re.escape(token) for token in special_tokens)
        pieces = re.split(special_pat, chunk)
    else:
        pieces = [chunk]

    for piece in pieces:
        for match in re.finditer(PAT, piece):
            token = match.group()
            key = tuple(bytes([b]) for b in token.encode("utf-8"))
            counts[key] += 1

    return counts


def get_pairs(word: tuple[bytes, ...]) -> list[tuple[bytes, bytes]]:
    return [(word[i], word[i + 1]) for i in range(len(word) - 1)]


def merge_pair_in_word(
    word: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    a, b = pair
    merged = a + b

    out = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def apply_merge(
    pair: tuple[bytes, bytes],
    word_counts: dict[tuple[bytes, ...], int],
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_words: dict[tuple[bytes, bytes], dict[tuple[bytes, ...], None]],
) -> None:
    affected_words = list(pair_to_words.get(pair, {}).keys())
    if not affected_words:
        return

    new_word_additions: dict[tuple[bytes, ...], int] = defaultdict(int)

    for word in affected_words:
        freq = word_counts.get(word, 0)
        if freq == 0:
            continue

        old_pairs = get_pairs(word)

        for p in old_pairs:
            pair_counts[p] -= freq
            pair_to_words[p].pop(word, None)

            if pair_counts[p] == 0:
                del pair_counts[p]
            if not pair_to_words[p]:
                del pair_to_words[p]

        del word_counts[word]

        new_word = merge_pair_in_word(word, pair)
        new_word_additions[new_word] += freq

    for new_word, freq in new_word_additions.items():
        old_freq = word_counts.get(new_word, 0)
        word_counts[new_word] = old_freq + freq

    for new_word, added_freq in new_word_additions.items():
        for p in get_pairs(new_word):
            pair_counts[p] += added_freq
            pair_to_words[p][new_word] = None


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    for i in range(256):
        vocab[i] = bytes([i])

    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    with open(input_path, "rb") as f:
        num_processes = 4
        boundary_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
        boundaries = find_chunk_boundaries(f, num_processes, boundary_token)

    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(processes=4) as pool:
        chunk_dicts = pool.starmap(process_chunk, tasks)

    pretoken_counts: dict[tuple[bytes, ...], int] = defaultdict(int)
    for d in chunk_dicts:
        for k, v in d.items():
            pretoken_counts[k] += v

    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], dict[tuple[bytes, ...], None]] = defaultdict(dict)

    for word, freq in pretoken_counts.items():
        for pair in get_pairs(word):
            pair_counts[pair] += freq
            pair_to_words[pair][word] = None

    num_merges = vocab_size - len(vocab)
    if num_merges < 0:
        raise ValueError("vocab_size is smaller than initial vocab size")

    for _ in range(num_merges):
        if not pair_counts:
            break

        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)

        vocab[next_id] = best_pair[0] + best_pair[1]
        next_id += 1

        apply_merge(best_pair, pretoken_counts, pair_counts, pair_to_words)

    return vocab, merges