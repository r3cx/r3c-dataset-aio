# r3c-dataset-aio

Hey, this is where I keep all my dataset management tools. Everything from tagging and cleaning to upscaling, log parsing, and LoRA metadata stuff — basically the whole pipeline for prepping datasets for SD training.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Core Tools](#core-tools)
  - [AutoTagger](#autotagger)
  - [AutoTaggerV2](#autotaggers)
  - [DatasetPreparer](#datasetpreparer)
- [Utilities](#utilities)
  - [Image Utilities](#image-utilities)
  - [Tag File Utilities](#tag-file-utilities)
  - [Dataset Utilities](#dataset-utilities)
  - [Statistics Utilities](#statistics-utilities)
  - [LoRA Utilities](#lora-utilities)
- [Taggers](#taggers)
- [Upscalers](#upscalers)
- [Modules](#modules)
- [Archived](#archived)

---

## Quick Start

```bash
# 1. Clone and enter the project directory
cd r3c-dataset-aio

# 2. Run the setup script
.\Setup.bat

# 3. Install Python dependencies
pip install -r requirements.txt
```

### Config File

Some tools require API credentials. Create `config.py` in the project root with your credentials:

```python
# config.py (gitignored, create manually)
DANBOORU_USERNAME = "your_username"
DANBOORU_API_KEY = "your_api_key"
```

This file is ignored by Git and will not be committed.

### requirements.txt

The project requires Python 3.10+ with the following key dependencies:
- **torch** >= 2.7.0 (CUDA 12.6)
- **transformers** >= 4.52.0
- **Pillow** >= 10.0.0 (with pillow-avif-plugin for AVIF support)
- **opencv-python** >= 4.8.0
- **onnxruntime-gpu** (for ONNX model inference)
- **gradio** >= 4.40.0
- **safetensors** >= 0.4.3
- **numpy**, **tqdm**, **matplotlib**, **openpyxl**

---

## Core Tools

### AutoTagger (`Tag Tools/AutoTagger.py`)

Latest autotagger script. Supports both V2 and V3 tagger models via ONNX inference with automatic HuggingFace model download.

**Supported taggers:**
- **SwinV2** - Best accuracy, higher GPU/memory usage (`wd-v1-4-swinv2-tagger-v2`, `wd-swinv2-tagger-v3`)
- **ViT** - Fastest inference, slightly lower precision (`wd-v1-4-vit-tagger`, `wd-vit-tagger-v3`, `wd-vit-large-tagger-v3`)
- **ConvNeXt** - Balanced performance and accuracy (`wd-v1-4-convnext-tagger`, `wd-convnext-tagger-v3`)
- **EVA02** - EVA02 ViT Large (`wd-eva02-large-tagger-v3`)
- **Moat** - Alternative architecture (`wd-v1-4-moat-tagger-v2`)
- **ConvNeXtV2** - Updated ConvNeXt variant (`wd-v1-4-convnextv2-tagger-v2`)

**Features:**
- Configurable tagger selection (edit the `TAGGERS` list in the script)
- Supports `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp` formats
- Automatic tag confidence scoring
- Kaomoji filtering to prevent false positives
- Batch processing with progress bars

### AutoTaggerV2 (`Tag Tools/AutoTaggerV2.py`)

Legacy version constrained to V2 taggers only. Use `AutoTagger.py` for V3 model support.

**Convenience scripts:**
- `Run CreateNewTags.bat` - Generate new tag files for images
- `Run AppendToExistingTags.bat` - Append tags to existing `.txt` files
- `Run Add Trigger.bat` - Add trigger tags to images
- `Run TagClean.bat` - General tag cleaning
- `Run TagCleanChar.bat` - Character-specific tag cleaning

### DatasetPreparer (`Tag Tools/DatasetPreparer.py`)

Post-processing tool for prepared tag files. Cleans, prunes, and formats tags for training.

**Features:**
- Duplicate tag removal while preserving order
- Precedence pruning (e.g., keeps "huge breasts" over "breasts")
- Negative tag filtering (removes low-value tags like `unknown`, `transparent background`)
- Trigger tag injection
- Score tag formatting for Pony models
- Old trigger tag replacement

---

## Utilities

### Image Utilities (`Utilities/Image/`)

| Script | Description |
|--------|-------------|
| `ComputeResRatios.py` | Calculates constant-area resolutions for various aspect ratios from a base dimension |
| `CopyAndRenameImages.py` | Copies and renames images in batch |
| `DatasetMigrater.py` | Migrates images and their tag files between directories |
| `ImageConverter.py` | Converts images between formats |
| `ImageCropper.py` | Crops a percentage from the bottom of images |
| `ImageUpscaler.py` | Upscales images using ONNX upscaler models |
| `RandomFlip.py` | Randomly flips images horizontally/vertically |

### Tag File Utilities (`Utilities/Tag Files/`)

| Script | Description |
|--------|-------------|
| `DanbooruTagScraper.py` | Scrapes tag data from Danbooru API (credentials from `config.py`) |
| `FilterTagFile.py` | Filters Danbooru tag CSVs by post count threshold, category, and other criteria |
| `anima_tags.csv` | Pre-filtered animation tag dataset |
| `danbooru_tags_post_count.csv` | Full Danbooru tag database with post counts |

### Dataset Utilities (`Utilities/Dataset/`)

| Script | Description |
|--------|-------------|
| `FilenameRemoveBrackets.bat` | Removes brackets from filenames |
| `RemoveCleanupTextFromName.py` | Strips cleanup text from image filenames |
| `SortTagsOutputAsIntervals.py` | Sorts and formats tag lists into fixed-size intervals |

### Statistics Utilities (`Utilities/Stats/`)

| Script | Description |
|--------|-------------|
| `TrainLogParser.py` | Parses SD training logs and exports metrics to Excel (step or epoch mode) |
| `Run LogParse.bat` | Launcher for the training log parser |
| `PngInfoPromptToTags.py` | Extracts embedded prompts from PNG metadata and converts to tag format |
| `LaunchTensorboard.bat` | Shortcut to launch TensorBoard for training visualization |
| `PlotGraph.ipynb` | Jupyter notebook for plotting training graphs |

### LoRA Utilities (`Utilities/Lora/`)

| Script | Description |
|--------|-------------|
| `AddMetadataThumbnail.py` | Adds activation tags, thumbnails, and metadata to `.safetensors` LoRA files |

---

## Taggers

Pre-downloaded WD14 tagger model directories in `Taggers/`:

| Directory | Architecture | Version |
|-----------|-------------|---------|
| `wd-v1-4-swinv2-tagger-v2/` | SwinV2 | v2 |
| `wd-swinv2-tagger-v3/` | SwinV2 | v3 |
| `wd-v1-4-vit-tagger/` | ViT | v1.4 |
| `wd-v1-4-vit-tagger-v2/` | ViT | v2 |
| `wd-vit-tagger-v3/` | ViT | v3 |
| `wd-v1-4-convnext-tagger/` | ConvNeXt | v1.4 |
| `wd-v1-4-convnext-tagger-v2/` | ConvNeXt | v2 |
| `wd-convnext-tagger-v3/` | ConvNeXt | v3 |
| `wd-v1-4-convnextv2-tagger-v2/` | ConvNeXtV2 | v2 |
| `wd-v1-4-moat-tagger-v2/` | Moat | v2 |
| `eva02-vit-large-448-8046/` | EVA02 ViT Large | - |

---

## Upscalers

ONNX and PyTorch upscaler models in `Upscalers/`:

| File | Format |
|------|--------|
| `4xNomos8kDAT.onnx` | ONNX |
| `4xNomos8kDAT.pth` | PyTorch |

---

## Modules

- **`Modules/llama_src/`** - llama.cpp source repository for local LLM inference

---

## Archived

Deprecated or legacy tools in `Archived/`:

- `AutoTaggerE621.py` - Legacy E621 autotagger
- `WebpToGif.py` - WebP to GIF converter
- `Caption UI/` - Legacy caption UI tools (llama.cpp build and launcher)
- `Quantum Merge/` - SDXL model merging tools

---

## Project Structure

```
r3c-dataset-aio/
├── config.py                # API credentials (gitignored, create manually)
├── Tag Tools/               # Core tagging and dataset preparation tools
├── Taggers/                 # WD14 tagger model directories
├── Upscalers/               # Image upscaler models
├── Utilities/
│   ├── Image/               # Image processing utilities
│   ├── Tag Files/           # Tag management and scraping tools
│   ├── Dataset/             # Dataset manipulation utilities
│   ├── Stats/               # Training log analysis tools
│   └── Lora/                # LoRA metadata tools
├── Modules/                 # External modules (llama.cpp)
├── Archived/                # Deprecated tools
├── requirements.txt         # Python dependencies
├── Setup.bat                # Project setup script
└── PROJECT_FILES.md         # Detailed file inventory
```
