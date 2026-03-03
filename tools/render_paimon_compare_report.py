#!/usr/bin/env python3
"""Generate an HTML side-by-side report for baseline vs tuned wav + mel."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchaudio


def safe_id(value: str) -> str:
    value = value.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:120]


def load_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        raise FileNotFoundError(f"summary file not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_mel(wav_path: Path, out_path: Path, device: torch.device) -> float:
    waveform, sample_rate = torchaudio.load(str(wav_path))
    if waveform.shape[0] > 1:
        waveform = waveform[:1]

    target_sr = 22050
    if sample_rate != target_sr:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sr)(waveform)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        power=2.0,
    ).to(device)
    with torch.no_grad():
        mel = mel_transform(waveform.to(device)).clamp(min=1e-10)
        mel_db = 10.0 * torch.log10(mel)
        mel_db = mel_db - mel_db.amax(dim=(-2, -1), keepdim=True)  # peak-normalize to 0 dB
        mel_db = torch.clamp(mel_db, min=-80.0, max=0.0)

    mel_np = mel_db.squeeze(0).cpu().numpy()
    plt.figure(figsize=(10, 3), dpi=160)
    plt.imshow(mel_np, origin="lower", aspect="auto", cmap="magma", vmin=-80.0, vmax=0.0)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
    plt.close()
    return waveform.shape[-1] / target_sr


def row_html(row_id: str, text: str, baseline: dict, tuned: dict, mel_map: dict) -> str:
    prompt = row_id
    baseline_safe = safe_id(f"b-{row_id}")
    tuned_safe = safe_id(f"t-{row_id}")

    baseline_mel = mel_map[(baseline_safe, "baseline")]
    tuned_mel = mel_map[(tuned_safe, "tuned")]
    baseline_audio = baseline["__audio_src"]
    tuned_audio = tuned["__audio_src"]
    baseline_rtf = baseline.get("rtf")
    tuned_rtf = tuned.get("rtf")
    baseline_prompt_sim = baseline.get("spk_sim_to_prompt")
    tuned_prompt_sim = tuned.get("spk_sim_to_prompt")
    baseline_ref_sim = baseline.get("spk_sim_to_ref")
    tuned_ref_sim = tuned.get("spk_sim_to_ref")
    ref_info = ""
    if baseline_ref_sim is not None or tuned_ref_sim is not None:
        ref_info = (
            "<div class=\"stats\">Ref similarity: "
            f"baseline {baseline_ref_sim:.4f} / tuned {tuned_ref_sim:.4f}</div>"
            if baseline_ref_sim is not None and tuned_ref_sim is not None
            else ""
        )

    def _fmt(v: float | None) -> str:
        return "N/A" if v is None else f"{v:.4f}"

    return f"""
    <section class="pair-card">
      <div class="pair-title">Sentence: {row_id}</div>
      <div class="pair-text">{text}</div>
      <div class="model-grid">
        <article class="side before">
          <h3>Before</h3>
          <audio controls preload="metadata" data-spec="{baseline_safe}" src="{baseline_audio}"></audio>
          <div class="spec-wrap">
            <img src="{baseline_mel}" alt="baseline mel" />
            <div class="playhead" id="playhead-{baseline_safe}"></div>
          </div>
          <div class="stats">RTF: {_fmt(baseline_rtf)} | Prompt sim: {_fmt(baseline_prompt_sim)}</div>
        </article>
        <article class="side after">
          <h3>After (Tuned)</h3>
          <audio controls preload="metadata" data-spec="{tuned_safe}" src="{tuned_audio}"></audio>
          <div class="spec-wrap">
            <img src="{tuned_mel}" alt="tuned mel" />
            <div class="playhead" id="playhead-{tuned_safe}"></div>
          </div>
          <div class="stats">RTF: {_fmt(tuned_rtf)} | Prompt sim: {_fmt(tuned_prompt_sim)}</div>
        </article>
      </div>
      {ref_info}
    </section>
    """


def main() -> None:
    parser = argparse.ArgumentParser(description="Render baseline vs tuned audio report")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("/mnt/sda2/cosyvoice_eval/paimon/summary.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("/mnt/sda2/cosyvoice_eval/paimon/paimon_compare.html"))
    parser.add_argument("--force", action="store_true", help="Regenerate mel PNGs even if they already exist")
    args = parser.parse_args()

    payload = load_summary(args.summary)
    baseline_model = payload["models"]["baseline"]
    tuned_model = payload["models"]["tuned_flow_e10"]

    out_dir = args.out.parent
    mel_dir = out_dir / "mels"
    mel_dir.mkdir(parents=True, exist_ok=True)

    baseline_items = {item["id"]: item for item in baseline_model["items"]}
    tuned_items = {item["id"]: item for item in tuned_model["items"]}
    common_ids = sorted(set(baseline_items) & set(tuned_items))

    mel_map = {}
    rows = []
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    for utt_id in common_ids:
        b_item = baseline_items[utt_id]
        t_item = tuned_items[utt_id]
        if not b_item["outputs"] or not t_item["outputs"]:
            continue

        baseline_safe = safe_id(f"b-{utt_id}")
        tuned_safe = safe_id(f"t-{utt_id}")
        baseline_mel_path = mel_dir / f"{baseline_safe}.png"
        tuned_mel_path = mel_dir / f"{tuned_safe}.png"

        baseline_wav = Path(b_item["outputs"][0]["wav"])
        tuned_wav = Path(t_item["outputs"][0]["wav"])
        if args.force or (not baseline_mel_path.exists()):
            build_mel(baseline_wav, baseline_mel_path, device)
        if args.force or (not tuned_mel_path.exists()):
            build_mel(tuned_wav, tuned_mel_path, device)

        mel_map[(baseline_safe, "baseline")] = f"mels/{baseline_mel_path.name}"
        mel_map[(tuned_safe, "tuned")] = f"mels/{tuned_mel_path.name}"
        try:
            b_item["__audio_src"] = str(baseline_wav.relative_to(out_dir))
        except ValueError:
            b_item["__audio_src"] = str(baseline_wav)
        try:
            t_item["__audio_src"] = str(tuned_wav.relative_to(out_dir))
        except ValueError:
            t_item["__audio_src"] = str(tuned_wav)
        rows.append(row_html(utt_id, b_item["tts_text"], b_item, t_item, mel_map))

    rows_html = "\n".join(rows)
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Baseline vs Tuned - CosyVoice 300M</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --card: #ffffff;
      --line: #dde2ea;
      --text: #141b2d;
      --muted: #54617a;
      --before: #2563eb;
      --after: #0f766e;
      --accent: #0f172a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif;
      background: radial-gradient(circle at 10% 20%, #f0f4ff, var(--bg) 45%, #eef4f7 100%);
      color: var(--text);
    }}
    .container {{
      max-width: 1400px;
      margin: 28px auto;
      padding: 0 18px 28px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    .subtitle {{ color: var(--muted); margin-bottom: 20px; }}
    .pair-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      margin-bottom: 16px;
      box-shadow: 0 8px 28px rgba(20, 27, 45, 0.08);
    }}
    .pair-title {{
      font-weight: 700;
      margin-bottom: 6px;
      color: var(--accent);
      font-size: 16px;
    }}
    .pair-text {{
      margin: 6px 0 12px;
      color: var(--muted);
    }}
    .model-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    .side {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      position: relative;
      background: linear-gradient(180deg, #fcfdff, #f6f9ff);
    }}
    .before h3 {{ color: var(--before); margin: 0 0 8px; }}
    .after h3 {{ color: var(--after); margin: 0 0 8px; }}
    audio {{
      width: 100%;
      margin-bottom: 10px;
      background-color: transparent;
    }}
    .spec-wrap {{
      position: relative;
      border-radius: 8px;
      border: 1px solid var(--line);
      overflow: hidden;
      background: #111;
    }}
    .spec-wrap img {{
      width: 100%;
      display: block;
      cursor: pointer;
    }}
    .playhead {{
      position: absolute;
      top: 0;
      left: 0;
      width: 2px;
      height: 100%;
      background: #fbbf24;
      pointer-events: none;
      opacity: 0.95;
      transform: translateX(-1px);
      transition: left 40ms linear;
    }}
    .stats {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 960px) {{
      .model-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>CosyVoice baseline vs tuned comparison</h1>
    <div class="subtitle">Left: baseline (pretrained) | Right: flow-tuned checkpoint (epoch 10)</div>
    {rows_html}
  </div>
  <script>
    const audioItems = document.querySelectorAll('audio[data-spec]');
    audioItems.forEach((audioEl) => {{
      const playhead = document.getElementById('playhead-' + audioEl.dataset.spec);
      const specWrap = playhead.parentElement;
      const sync = () => {{
        if (!audioEl.duration || Number.isNaN(audioEl.duration) || !isFinite(audioEl.duration) || audioEl.duration <= 0) return;
        const ratio = Math.min(1, Math.max(0, audioEl.currentTime / audioEl.duration));
        playhead.style.left = (ratio * 100).toFixed(4) + '%';
      }};
      audioEl.addEventListener('timeupdate', sync);
      audioEl.addEventListener('seeked', sync);
      audioEl.addEventListener('loadedmetadata', () => {{
        playhead.style.left = '0%';
      }});
      audioEl.addEventListener('ended', () => {{
        playhead.style.left = '100%';
      }});
      specWrap.addEventListener('click', (e) => {{
        const rect = specWrap.getBoundingClientRect();
        if (!audioEl.duration) return;
        const clickRatio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
        audioEl.currentTime = clickRatio * audioEl.duration;
      }});
    }});
  </script>
</body>
</html>
"""
    args.out.write_text(html, encoding="utf-8")
    print(f"Generated: {args.out}")
    print(f"Mel images: {mel_dir}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
