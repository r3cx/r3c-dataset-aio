# r3c-dataset-aio

Hey, this is my dataset prep playground. It's basically everything I use to turn a
folder of anime art into something ready for LoRA training — upscaling, autotagging (via a multi-model ensemble), tag cleaning, parsing training logs, and setting up LoRA metadata.

The tagger side got a big refactor. It used to be a couple of fat single-file
scripts; now it's a small package with a clean three-layer design — **tagger models,
ensembles, and autotagger entry points** — all built on top of one shared
sliding-window pipeline. The rest of my random set of tools live under `Utilities/`.

---

## Table of Contents

- [Quick Start](#quick-start)
- [The Autotagger System](#the-autotagger-system)
  - [How the pipeline works](#how-the-pipeline-works)
  - [Layer 1 — Taggers (the models)](#layer-1--taggers-the-models)
  - [Layer 2 — Ensembles](#layer-2--ensembles)
  - [Layer 3 — Autotaggers (what you actually run)](#layer-3--autotaggers-what-you-actually-run)
  - [CLI flags](#cli-flags)
  - [Running from the command line](#running-from-the-command-line)
- [Tagging Launchers](#tagging-launchers)
- [DatasetPreparer](#datasetpreparer)
- [Utilities](#utilities)
  - [Image Utilities](#image-utilities)
  - [Tag File Utilities](#tag-file-utilities)
  - [Dataset Utilities](#dataset-utilities)
  - [LoRA Utilities](#lora-utilities)
- [Upscalers](#upscalers)
- [Project Structure](#project-structure)

---

## Quick Start

```powershell
# 1. Clone and enter the project
git clone <repo-url>
cd r3c-dataset-aio

# 2. One-shot setup: builds a venv and installs requirements
.\Setup.bat

# 3. Tagging and everything else runs from the .\venv (the .bat launchers do this for you)
```

`Setup.bat` creates `.venv`/`venv`, activates it, and pip-installs `requirements.txt`.
Every launcher and script assumes that venv exists — if you ever see
*"Incomplete Setup Detected. Run 'Setup.bat' first!"*, that's why.

### Models

The tagger weights live in `.gitignore`d folders (`TagManagement/Models/` and
`Upscalers/`) so they're **not** in the repo. They get pulled in automatically:

- The WD1.4 ONNX tagger downloads itself from HuggingFace on first load.
- The Danbooru-v4 taggers are **gated** repos, so their first download needs a
  readable HuggingFace token. Set it one of these ways, once:
  - `--hf_token hf_...` on the command line (or in the .bat)
  - `HF_TOKEN=hf_...` environment variable
  - `hf login` inside the venv

  Once the files are in `Models/`, no credential is ever needed again. You can also
  pre-place the weights and pass `--no_download`.

### Danbooru API config file (optional)

Only the tag-file tools in `Utilities/Tag Files` need this — everything else runs
fine without it. They read your credentials from a `config.py` (it's `.gitignore`d,
so keep that file local):

```python
# config.py — Danbooru API credentials
DANBOORU_USERNAME = "your_danbooru_username"
DANBOORU_API_KEY  = "your_api_key_from_danbooru"
```

That's the whole thing — it just needs to be a module that the scraper can
`from config import` from, and it only looks at two module-level string constants:

- **`DANBOORU_USERNAME`** — your Danbooru account username. It's used for the
  **HTTP Basic auth** credentials (`session.auth = (username, api_key)`) and is also
  stitched into the `User-Agent` header (the code sends `"<username>-tag-exporter/1.0"`).
  Keep it the real username Danbooru knows you by.
- **`DANBOORU_API_KEY`** — your Danbooru API key. Grab it from Danbooru under
  **Account → Edit account → API key**; it's shown once when generated, so copy it
  before the popup closes.

**Where the file lives:** `DanbooruTagScraper.py` puts the project root on `sys.path`
(via `Path(__file__).parents[2]`) and then imports by name, so it just needs to be a
`config.py` sitting **at the project root** — the same folder as the repo `.gitignore`.
Drop it there, keep it local, and the import resolves from there no matter where the
scraper is launched.

### Key dependencies

`requirements.txt` targets a CUDA 12.6 box. The important pins: `torch >= 2.7.0`
(plus `torchvision`), `transformers >= 4.52.0`, `onnxruntime-gpu == 1.19.2`
(the ONNX Eva02 tagger), `gradio >= 4.40.0`, `Pillow` + `pillow-avif-plugin`,
`opencv-python`, and the data toolkit (`numpy`, `pandas`, `tqdm`, `openpyxl`,
`matplotlib`).

---

## The Autotagger System

The whole tagging feature lives in `TagManagement/`. It's split into three layers so
each piece stays small and testable. The class hierarchy:

```
Taggers/   (models)
  Tagger (base: paths, download, load/unload)
    ├── Eva02Tagger           ONNX, WD1.4 EVA02-Large  (regression baseline)
    └── Dbv4Tagger (base)     PyTorch, Danbooru-v4 shared pipeline
          ├── EvaGiantTagger
          ├── ViTGiantTagger
          └── ConvNeXtV2Tagger

Ensembles/ (pure combination math, no models)
  └── EnsembleTagger          sparse equal-weight averaging

Autotaggers/ (entry points)
  Autotagger (base: config, discovery, .txt output, frequencies, CLI)
    └── SlidingWindowAutotagger   (crop gen, composition routing, window merge, selection)
          ├── Eva02SlidingWindowAutotagger   single-model baseline
          └── EnsembleSWAutoTagger           multi-model ensemble
```

### How the pipeline works

Every autotagger runs the same sliding-window routine (lives in
`SlidingWindowAutotagger.py`):

1. **Window the image** at its original resolution — a grid of crops sized to the
   image's short axis (or `--window_size_cap`), with ~10% overlap. A full-image
   **global scan** crop is always appended last.
2. **Tag each crop** with the model(s). Big crops are pre-sized inside the
   DataLoaders (`--num_data_loader_workers`), so the resize never blocks the one
   thread doing inference.
3. **Merge the crops.** Normal tags take the **max** across all crops (so a feature
   in *any* window gets its best confidence). **Composition tags** and **character
   tags** always use the **global scan** value, because crops lack the full-image
   context those tags need (a crop can't tell you how many girls are in the scene,
   or cut a twintail into a "side ponytail").
4. **Select and write.** Apply the general / character thresholds, drop the
   kaomoji + your `--undesired_tags` set, put character tags up front, and write a
   `.txt` next to each image (append or overwrite).

That composition/character split is the heart of why sliding windows work here —
fine local detail (a specific earring) still gets caught in a tight crop, while
scene-level facts come from the whole-image pass.

### Layer 1 — Taggers (the models)

`Taggers/Tagger.py` is a thin, model-agnostic base: it owns the local model folder
(pathed to `TagManagement/Models/<name>`), the HuggingFace download, and the
load/unload lifecycle (it's a context manager). Concrete taggers just declare their
repository + files and implement inference + preprocessing. Two families:

| Tagger class | Model | Where it comes from | Inference | Notes |
|--------------|-------|---------------------|-----------|-------|
| `Eva02Tagger` | WD1.4 EVA02-Large | `SmilingWolf/wd-eva02-large-tagger-v3` | ONNX | 448px, the regression baseline, 4 rating outputs |
| `EvaGiantTagger` | EVA-Giant (dbv4-full) | `animetimm/eva_giant_patch14_560.dbv4-full` | PyTorch | 560px input, ImageNet norm |
| `ViTGiantTagger` | ViT-GiantOpt / SigLIP (dbv4-full) | `animetimm/vit_giantopt_patch16_siglip_384.dbv4-full` | PyTorch | 512px input (the "...384" name is misleading), SigLIP 0.5 norm |
| `ConvNeXtV2Tagger` | ConvNeXtV2-Huge (dbv4-full) | `animetimm/convnextv2_huge.dbv4-full` | PyTorch | 512px input, ImageNet norm |

The three Danbooru-v4 taggers share one shared base (`Dbv4Tagger.py`) because the
`animetimm/*.dbv4-full` repos all use the same transformers/timm layout — input size
and normalization come from each model's own `preprocess.json`, never from the repo
name. They're lazy-imported in `Taggers/__init__.py` so loading the light ONNX path
doesn't pull in `torch` for nothing. Taggers never apply thresholds themselves —
that's always the autotagger's/ensemble's job.

### Layer 2 — Ensembles

`Ensembles/EnsembleTagger.py` is pure, model-free math over the sparse
`{tag: confidence}` dicts each tagger produces. It's equal-weight, no learned
weights, exact string matching. Two combine modes:

- **`mean`** — divide every tag's summed confidence by the number of taggers `N`.
  Simple, but a tag a model is *structurally* missing contributes a 0 that dilutes the
  score.
- **`taggers`** (default) — **vocabulary-aware**: each tag is divided only by the
  number of taggers whose vocab actually contains it, so a missing tag is treated as
  an *absence*, not a false low detection. This is the mode you want since the four
  models don't share the same vocab. `--combine taggers`/`mean` picks it at runtime.

There's also an online `prune_accumulator` that drops tags that can no longer reach
threshold no matter what the remaining taggers do, to keep the in-memory accumulators
small.

### Layer 3 — Autotaggers (what you actually run)

Two runnable leaves, both extending the shared sliding-window mechanism:

- **`Eva02SlidingWindowAutotagger`** — the known-good **regression baseline**.
  Runs exactly one tagger (WD1.4 EVA02-Large) and its merge result is thresholded
  directly. It's the reference every change is checked against.
- **`EnsembleSWAutoTagger`** — the **multi-model ensemble**. Runs several taggers
  (`--taggers eva02,eva_giant,vit_giant,convnextv2`, all four by default) and
  averages their confidences via `EnsembleTagger`, thresholds applied *after* the
  ensemble. On a 24 GB card it runs **tagger-major**: one tagger loaded at a time,
  the whole dataset, accumulate sparse sums, free it, next one — then a final in-RAM
  pass divides, selects, and writes each `.txt` once. Ordering is strictly
  window-merge → per-tagger result → ensemble → threshold → filter → output.

### CLI flags

The shared flags (on every autotagger leaf, defined in `Autotagger.py`):

| Flag | What it does |
|------|--------------|
| `--data_dir` | Folder of images to tag (required) |
| `--threshold` | Default confidence threshold |
| `--general_threshold` / `--character_threshold` | Per-category thresholds (override `--threshold`) |
| `--undesired_tags` | Comma list of tags to drop |
| `--mode_append` | Append to existing `.txt` instead of overwriting |
| `--recursive_gather` | Search subdirectories too |
| `--frequency_tags` | Print a tag-frequency report at the end |
| `--num_data_loader_workers` | Parallel image/crop loaders (parallelism) |
| `--window_size_cap` | Hard cap window size in px (0 = image short axis) |
| `--window_stride_ratio` | Window stride (default 0.9 ≈ 10% overlap) |
| `--use_sequential` | Infer crops one at a time (lower peak memory) |
| `--use_gpu` | Run on CUDA |
| `--bf16` | bfloat16 for the PyTorch taggers — ~2-4x faster + half the VRAM on Ampere+ |
| `--hf_token` | Token for the first download of a gated model |
| `--no_download` | Never download, load from local `Models/` only |
| `--force_download` | Re-pull the models even if present |
| `--verbose` / `--debug_print` | Per-image inference breakdown / per-tag confidence dump |

`EnsembleSWAutoTagger` adds:

| Flag | What it does |
|------|--------------|
| `--taggers` | Comma list: `eva02`, `eva_giant`, `vit_giant`, `convnextv2` (default all four) |
| `--combine` | `taggers` (vocabulary-aware, default) or `mean` (plain /N) |
| `--debug_taggers` | Print each tagger's pre-ensemble confidence per image (high memory) |
| `--debug_taggers_top` | With the above, how many top tags to list per image |

### Running from the command line

The easiest path is the **`.bat` launchers** in `TagManagement/` (in the
[Tagging Launchers](#tagging-launchers) section) — you set `DATA_DIR` / `TRIGGER` /
thresholds once at the top and double-click. Reach for the raw CLI below when you
want to flip a single flag (a different threshold, a tagger subset, `--use_sequential`,
`--debug_print`, ...) without editing a file.

```powershell
# from the project root, venv active
# single-model baseline (EVA02-Large)
python TagManagement\Autotaggers\Eva02SlidingWindowAutotagger.py `
    --data_dir "your\images" --use_gpu --character_threshold=1 --general_threshold=0.65

# the full 4-model ensemble, fast bf16 defaults
python TagManagement\Autotaggers\EnsembleSWAutoTagger.py `
    --data_dir "your\images" --use_gpu --bf16 `
    --general_threshold=0.475 --character_threshold=0.65 `
    --num_data_loader_workers=6 --mode_append

# run a subset of taggers
python TagManagement\Autotaggers\EnsembleSWAutoTagger.py `
    --data_dir "your\images" --taggers eva02,convnextv2 --use_gpu
```

(The `\` is just PowerShell line-continuation — collapse to one line if you prefer.)

**VRAM (GPU mode).** Your peak comes from the *single largest* model you load at
once (the ensemble swaps them one at a time, so they don't stack). The single-model
EVA02 baseline (ONNX) is light — **any card works, ~4–6 GB is comfortable**. The four
Danbooru-v4 PyTorch taggers are the heavy ones: with the default batched, `--bf16`
inference the **minimum is roughly 8 GB of VRAM**, and **16 GB is the comfortable
target** (the `.bat` defaults were tuned for a ~24 GB card). Peak also scales with
image size, `--window_size_cap`, and `--num_data_loader_workers`. If you're short of
VRAM, fit smaller cards by adding `--use_sequential`, lowering `--window_size_cap`,
and dropping the worker count.

---

## Tagging Launchers

These `.bat` files in `TagManagement/` set the common config and run the right
scripts — edit the `DATA_DIR` / `TRIGGER` / thresholds at the top, then double-click.

| Launcher | What it does |
|----------|--------------|
| `Run EnsembleAutotagger.bat` | The main one. Runs the 4-model ensemble (`--bf16`, general 0.475 / char 0.65, 6 workers, append, GPU), then hands off to `DatasetPreparer`. |
| `Run Eva02SlidingWindowAutotagger.bat` | Single-model EVA02 baseline (general 0.65 / char 1.0), then `DatasetPreparer`. |
| `Run Add Trigger.bat` | Just `DatasetPreparer` — adds a trigger tag and strips an old one. Handy after a re-tag. |
| `Run UpscaleImages.bat` | Runs `Utilities/Image/ImageUpscaler.py` with the local ONNX upscaler onto a target folder. |

Each launcher auto-detects the project venv and errors out cleanly if `Setup.bat`
wasn't run.

---

## DatasetPreparer

`TagManagement/Utility/DatasetPreparer.py` — the post-processor that reads each
tagged `.txt` and makes it training-friendly. Pipeline, in order:

1. Escape parentheses and convert underscores → spaces (Danbooru → SD prompt form).
2. Drop your `--undesired_tags` (defaults to a built-in negative set: `unknown`,
   costuming/alternate noise, and the `score_*` tags, etc.).
3. **Precedence pruning** — keep the strongest descriptor per anatomy category
   (e.g. `huge breasts` beats `breasts`), skipped when multiple subjects are present.
4. Prepend a `--quality_tags` value if set (e.g. `score_9`).
5. Remove a `--old_trigger_tag`, then prepend the new `--trigger_tag`.
6. Dedupe while preserving order.

```powershell
python TagManagement\Utility\DatasetPreparer.py --data_dir "your\images" --trigger_tag "@artist" --old_trigger_tag "@old"
```
---

## Utilities

General tools in `Utilities/`, grouped by what they touch.

### Image Utilities (`Utilities/Image/`)

| Script | Description |
|--------|-------------|
| `ComputeResRatios.py` | Calculates constant-area resolutions for various aspect ratios from a base dimension |
| `CopyAndRenameImages.py` | Copies and renames images in batch |
| `DatasetMigrater.py` | Migrates images and their tag files between directories |
| `ImageConverter.py` | Converts images between formats |
| `ImageCropper.py` | Crops a percentage from the bottom of images |
| `ImageUpscaler.py` | Upscales images with an ONNX upscaler model (`--onnx_model`, `--input_dir`, `--output_dir`, `--scale`) |
| `RandomFlip.py` | Randomly flips images horizontally/vertically |

### Tag File Utilities (`Utilities/Tag Files/`)

| Script | Description |
|--------|-------------|
| `DanbooruTagScraper.py` | Pulls the tag list from the Danbooru API into a timestamped CSV (credentials from `config.py`) |
| `FilterTagFile.py` | Filters a Danbooru tag CSV by post-count threshold / category into a cleaner tag set |

### Dataset Utilities (`Utilities/Dataset/`)

| Script | Description |
|--------|-------------|
| `PngInfoPromptToTags.py` | Extracts the embedded prompt from PNG metadata and reformats it into tag `.txt` files |
| `RemoveCleanupTextFromName.py` | Strips cleanup/annotation text from image filenames |
| `SortTagsOutputAsIntervals.py` | Sorts and formats tag lists into fixed-size intervals |
| `FilenameRemoveBrackets.bat` | Removes brackets from filenames in batch |

### LoRA Utilities (`Utilities/Lora/`)

| Script | Description |
|--------|-------------|
| `AddMetadataThumbnail.py` | Adds activation tags, a thumbnail, and metadata into `.safetensors` LoRAs (single file or whole folder) |
| `TrainLogParser.py` | Parses SD training logs and exports the metrics to Excel (step or epoch mode) — used with `LaunchTensorboard.bat` |
| `LaunchTensorboard.bat` | Shortcut to fire up TensorBoard for the training curves |
| `PlotGraph.ipynb` | Jupyter notebook for plotting training graphs |

---

## Upscalers

`Upscalers/` holds the upscaler model weights (e.g. the 4x Nomos 8K DAT model).
The files are binary blobs kept out of git (the folder is preserved with a
`.gitkeep`), so drop your weights in here and point
`Utilities/Image/ImageUpscaler.py` — or `Run UpscaleImages.bat` — at them.

---

## Project Structure

```
r3c-dataset-aio/
├── Setup.bat                     # One-shot venv + install
├── requirements.txt
├── README.md
├── TagManagement/                # The autotagger system
│   ├── Autotaggers/              # Entry points (run these)
│   │   ├── Autotagger.py                 # base: config, discovery, .txt, CLI
│   │   ├── SlidingWindowAutotagger.py    # shared sliding-window mechanism
│   │   ├── Eva02SlidingWindowAutotagger.py   # single-model baseline
│   │   └── EnsembleSWAutoTagger.py       # multi-model ensemble
│   ├── Taggers/                  # The models
│   │   ├── Tagger.py                 # base: pathing, download, lifecycle
│   │   ├── Eva02Tagger.py            # ONNX WD1.4 EVA02-Large
│   │   ├── Dbv4Tagger.py             # shared PyTorch Danbooru-v4 pipeline
│   │   ├── EvaGiantTagger.py
│   │   ├── ViTGiantTagger.py
│   │   └── ConvNeXtV2Tagger.py
│   ├── Ensembles/
│   │   └── EnsembleTagger.py       # pure sparse equal-weight combining
│   ├── Utility/
│   │   └── DatasetPreparer.py      # post-process .txt tag files
│   ├── Models/                     # tagger weights (downloaded; not committed)
│   ├── Run EnsembleAutotagger.bat
│   ├── Run Eva02SlidingWindowAutotagger.bat
│   ├── Run Add Trigger.bat
│   └── Run UpscaleImages.bat
├── Utilities/
│   ├── Image/                      # image processing
│   ├── Tag Files/                  # Danbooru tag scraping + filtering
│   ├── Dataset/                    # dataset/filename/tag-list tools
│   └── Lora/                       # LoRA metadata + training log tools
└── Upscalers/                      # upscaler weights (not committed)
```
