# CosyVoice「派蒙」训练复盘（CosyVoice-300M / flow 微调）

日期：2026-03-03

## 1. 目标与结论

- 目标：用本 repo 训练/适配中文角色音色（speaker：派蒙），并能稳定推理输出可对比的 A/B 结果。
- 结论：在 RTX 2080 SUPER 8GB 上，优先选择 `CosyVoice-1 / CosyVoice-300M`，先把 `flow` 微调跑通；`llm` 训练在 8GB 上容易 OOM（本次未成功产出 llm 训练 checkpoint）。
- `flow` 最优点：`epoch=10`（dev loss `0.584658`，文件：`/mnt/sda2/cosyvoice_exp/paimon/flow/torch_ddp/epoch_10_whole.yaml`）。

## 2. 环境与模型选择

- GPU：RTX 2080 SUPER 8GB（显存紧张是所有决策前提）。
- Python：`3.12.12`（conda env：`/home/zsc/miniconda3/envs/py312`）
- PyTorch：`2.10.0+cu128`
- 数据路径：`/mnt/sda2/中文 - Chinese/`（注意包含空格）。
- 预训练模型：`/mnt/sda2/pretrained_models/CosyVoice-300M/`
  - 关键文件：`campplus.onnx`、`speech_tokenizer_v1.onnx`、`llm.pt`、`flow.pt`、`hift.pt`、`cosyvoice.yaml`

### 为什么用 CosyVoice-1（300M）而不是 CosyVoice-2

- 8GB 显存下，训练 `llm` 更容易 OOM；先用 CosyVoice-1 的 `flow` 微调更现实，先拿到“音色更像目标”的可用结果。

## 3. 数据准备（推荐：离线特征 + parquet）

目标是让训练阶段只读 parquet，避免训练时做重特征计算。

- 由 `wav + lab` 生成 Kaldi 风格清单（`wav.scp/text/spk2utt/utt2spk`）：
  - 使用脚本：`tools/prepare_wav_lab_dataset.py`
- 离线特征（train/dev 都做）：
  - `utt2embedding.pt`、`spk2embedding.pt`
  - `utt2speech_token.pt`
- parquet：
  - train：37 shards（200 utt/shard）
  - dev：1 shard

本次生成的数据根目录：

- `/mnt/sda2/cosyvoice_data/paimon/train/`
- `/mnt/sda2/cosyvoice_data/paimon/dev/`
- 训练 list：
  - `/mnt/sda2/cosyvoice_data/paimon/train.data.list`
  - `/mnt/sda2/cosyvoice_data/paimon/dev.data.list`

### 关键教训：带空格路径会把 manifest 读崩

数据目录包含空格（`中文 - Chinese`），所以读取 `wav.scp/text` 时必须用：

- `line.split(maxsplit=1)`

否则会把路径拆成多段，后续全部错位。

## 4. 训练流程与结果

训练日志：

- `/mnt/sda2/cosyvoice_exp/paimon/train_2026-03-02_172805.log`

结果状态（非常重要）：

- `llm`：在 8GB 上 OOM（最终只留下 init，无有效 `epoch_*`）
- `flow`：完整训练完 `0..19` 共 20 epochs，checkpoint 在：
  - `/mnt/sda2/cosyvoice_exp/paimon/flow/torch_ddp/epoch_XX_whole.pt`

### 关键教训：脚本要 `set -euo pipefail`

原 `run.sh` 默认不 `set -e`，导致：

- `llm` OOM 失败后脚本仍继续进入 `flow` stage
- 容易误以为“两个 stage 都训练成功了”

建议所有 stage 脚本一律加 `set -euo pipefail` 并显式检查 stage 成功与否。

## 5. 推理与评测（A/B 对比 + mel 展示）

本次评测对比的是：

- baseline：原始 `CosyVoice-300M`
- tuned：仅替换 `flow` 为 `epoch=10` 的权重（`llm`/`hift` 仍为 pretrain）

评测音频输出目录：

- `/mnt/sda2/cosyvoice_eval/paimon/`（`baseline__*.wav` vs `tuned_flow_e10__*.wav`）

评测汇总（json）：

- `/mnt/sda2/cosyvoice_eval/paimon/summary.json`

客观对比（CampPlus embedding cosine，越高越像）：

- `spk_sim_to_prompt`：`0.644 -> 0.703`
- `spk_sim_to_ref`（dev）：`0.641 -> 0.694`

### HTML side-by-side（wav 可播放，mel 同步）

生成的页面：

- `/mnt/sda2/cosyvoice_eval/paimon/paimon_compare.html`

对应脚本：

- `tools/render_paimon_compare_report.py`

推荐打开方式（避免浏览器跨域/本地文件限制）：

```bash
cd /mnt/sda2/cosyvoice_eval/paimon
python -m http.server 8000
```

然后访问：

- `http://127.0.0.1:8000/paimon_compare.html`

### 关键教训：checkpoint 不能直接当 `flow.pt` 用

训练保存的 `epoch_XX_whole.pt` 里含额外 key（如 `epoch/step`）。
推理侧 `load_state_dict(..., strict=True)` 期望“纯权重 dict”，否则会加载失败。

处理方式：

- 另存一份纯权重（`pop('epoch'); pop('step')`）再当 `flow.pt` 用
- 或用 `cosyvoice/bin/average_model.py` 产出纯权重（它会跳过 `epoch/step`）

## 6. 代码/兼容性改动（为 Py3.12、空格路径、parquet 等）

本次为了“能跑通 + 省显存 + 读数据稳定”，做过的典型改动点：

- 多个工具脚本对 `wav.scp/text` 的解析改为 `split(maxsplit=1)`（支持路径包含空格）
- 新增 `tools/prepare_wav_lab_dataset.py`：从 `wav+lab` 生成 manifests
- parquet 读取不依赖 pandas：改用 `batch.to_pylist()`
- torch 2.10 兼容：DDP join/timeout 相关处理（避免依赖不存在的属性）
- label smoothing loss 做内存友好实现（避免构造巨大 dense 分布，减少 OOM 风险）

## 7. 下一步建议（按性价比排序）

1. 想让“断句/停顿/念词/语气理解”明显提升：必须把 `llm` 训起来
   - 8GB 下需要更激进降显存：更小 batch、更短句、gradient checkpoint、更严格的 frames 限制等
2. `flow` 继续精修：用 dev 最优点或 top-N average（常能提升稳定性）
3. 固化评测集：每次训练完自动生成 `summary.json + paimon_compare.html`，避免主观听感漂移

