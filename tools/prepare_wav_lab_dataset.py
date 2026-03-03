#!/usr/bin/env python3
import argparse
import hashlib
import os
import random
import re
from collections import defaultdict

from tqdm import tqdm


def _sanitize_id(value: str) -> str:
    value = value.strip()
    value = value.replace(os.sep, "_")
    value = re.sub(r"\s+", "_", value)
    return value


def _read_lab_text(lab_path: str, strip_hash: bool) -> str:
    with open(lab_path, "r", encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f.readlines()]
    text = "".join([line for line in lines if line])
    if strip_hash:
        text = text.lstrip()
        if text.startswith("#"):
            text = text[1:].lstrip()
    return text.strip()


def _write_kaldi_mapping(out_dir: str, items: list[dict]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    items = sorted(items, key=lambda x: x["utt"])
    spk2utts: dict[str, list[str]] = defaultdict(list)
    for item in items:
        spk2utts[item["spk"]].append(item["utt"])

    with open(os.path.join(out_dir, "wav.scp"), "w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item['utt']} {item['wav']}\n")
    with open(os.path.join(out_dir, "text"), "w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item['utt']} {item['text']}\n")
    with open(os.path.join(out_dir, "utt2spk"), "w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item['utt']} {item['spk']}\n")
    with open(os.path.join(out_dir, "spk2utt"), "w", encoding="utf-8") as f:
        for spk in sorted(spk2utts.keys()):
            f.write(f"{spk} {' '.join(spk2utts[spk])}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Kaldi-style wav.scp/text/utt2spk/spk2utt from a wav+lab dataset tree."
    )
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--des_dir", type=str, required=True)
    parser.add_argument("--dev_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1986)
    parser.add_argument("--strip_hash", action="store_true", default=True)
    parser.add_argument("--keep_hash", dest="strip_hash", action="store_false")
    parser.add_argument("--skip_braces", action="store_true", default=True)
    parser.add_argument("--keep_braces", dest="skip_braces", action="store_false")
    parser.add_argument("--min_text_chars", type=int, default=1)
    parser.add_argument("--speakers", type=str, default="")
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src_dir)
    des_dir = os.path.abspath(args.des_dir)
    speakers_filter = None
    if args.speakers.strip():
        speakers_filter = {s.strip() for s in args.speakers.split(",") if s.strip()}

    items_by_spk: dict[str, list[dict]] = defaultdict(list)
    seen_utts: set[str] = set()

    skipped_no_lab = 0
    skipped_filtered_spk = 0
    skipped_braces = 0
    skipped_empty_text = 0

    walk_roots = [src_dir]
    if speakers_filter is not None:
        walk_roots = []
        for spk_raw in sorted(speakers_filter):
            spk_dir = os.path.join(src_dir, spk_raw)
            if not os.path.isdir(spk_dir):
                print(f"Warning: speaker dir not found, skip: {spk_dir}")
                continue
            walk_roots.append(spk_dir)

    for walk_root in walk_roots:
        for dirpath, _, filenames in tqdm(os.walk(walk_root), desc=f"Scanning {os.path.basename(walk_root)}"):
            for fn in filenames:
                if not fn.lower().endswith(".wav"):
                    continue
                wav_path = os.path.join(dirpath, fn)
                lab_path = os.path.splitext(wav_path)[0] + ".lab"
                if not os.path.exists(lab_path):
                    skipped_no_lab += 1
                    continue

                rel_wav = os.path.relpath(wav_path, src_dir)
                spk_raw = rel_wav.split(os.sep, 1)[0]
                if speakers_filter is not None and spk_raw not in speakers_filter:
                    skipped_filtered_spk += 1
                    continue

                spk = _sanitize_id(spk_raw)
                utt_base = os.path.splitext(rel_wav)[0]
                utt = _sanitize_id(utt_base.replace(os.sep, "__"))
                if utt in seen_utts:
                    suffix = hashlib.md5(rel_wav.encode("utf-8")).hexdigest()[:8]
                    utt = f"{utt}_{suffix}"
                seen_utts.add(utt)

                text = _read_lab_text(lab_path, strip_hash=args.strip_hash)
                if not text or len(text) < args.min_text_chars:
                    skipped_empty_text += 1
                    continue
                if args.skip_braces and ("{" in text or "}" in text):
                    skipped_braces += 1
                    continue

                items_by_spk[spk].append({"utt": utt, "wav": os.path.abspath(wav_path), "text": text, "spk": spk})

    if not items_by_spk:
        print("No usable wav+lab pairs found.")
        return 2

    rng = random.Random(args.seed)
    train_items: list[dict] = []
    dev_items: list[dict] = []
    for spk, items in items_by_spk.items():
        items = list(items)
        rng.shuffle(items)
        if args.dev_ratio <= 0 or len(items) <= 1:
            train_items.extend(items)
            continue
        dev_n = max(1, int(round(len(items) * args.dev_ratio)))
        dev_n = min(dev_n, len(items) - 1)
        dev_items.extend(items[:dev_n])
        train_items.extend(items[dev_n:])

    print(
        "Prepared dataset:",
        f"speakers={len(items_by_spk)}",
        f"train_utts={len(train_items)}",
        f"dev_utts={len(dev_items)}",
        f"skipped_no_lab={skipped_no_lab}",
        f"skipped_filtered_spk={skipped_filtered_spk}",
        f"skipped_empty_text={skipped_empty_text}",
        f"skipped_braces={skipped_braces}",
    )

    _write_kaldi_mapping(os.path.join(des_dir, "train"), train_items)
    _write_kaldi_mapping(os.path.join(des_dir, "dev"), dev_items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
