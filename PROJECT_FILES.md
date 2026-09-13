# Project Files Documentation

This document provides an overview of the files in the r3c-dataset-aio project directory.

## Root
- **config.py** - API credentials (gitignored, create manually — see README for template)
- **README.md** - Project documentation and setup instructions
- **requirements.txt** - Python dependencies required for the project
- **Setup.bat** - Setup script for the project
- **.gitignore** - Git ignore rules
- **.gradio/** - Gradio cache directory
- **.kilo/** - Kilo configuration
- **.vscode/** - VS Code configuration
- **venv/** - Python virtual environment

## Tag Tools
- **AutoTagger.py** - Latest autotagger, supports V2 and V3 tagger models
- **AutoTaggerV2.py** - Legacy version, constrained to V2 taggers only
- **DatasetPreparer.py** - Script for preparing datasets
- `Run Add Trigger.bat` - Adds triggers to images
- `Run AppendToExistingTags.bat` - Appends tags to existing image files
- `Run CreateNewTags.bat` - Creates new tag files
- `Run UpscaleImages.bat` - Upscales images
- `Run TagClean.bat` - General tag cleaning script
- `Run TagCleanChar.bat` - Cleans tags from character images

## Taggers
Tagging model directories:
- **eva02-vit-large-448-8046/** - EVA02 ViT Large tagger model
- **wd-convnext-tagger-v3/** - WD ConvNeXt tagger v3
- **wd-eva02-large-tagger-v3/** - WD EVA02 Large tagger v3
- **wd-swinv2-tagger-v3/** - WD SwinV2 tagger v3
- **wd-v1-4-convnextv2-tagger-v2/** - WD v1.4 ConvNeXtV2 tagger v2
- **wd-v1-4-moat-tagger-v2/** - WD v1.4 Moat tagger v2
- **wd-v1-4-swinv2-tagger-v2/** - WD v1.4 SwinV2 tagger v2
- **wd-vit-large-tagger-v3/** - WD ViT Large tagger v3
- **wd-vit-tagger-v3/** - WD ViT tagger v3

## Upscalers
- **4xNomos8kDAT.onnx** - Nomos 8K DAT upscaler ONNX model
- **4xNomos8kDAT.pth** - Nomos 8K DAT upscaler PyTorch model

## Utilities

### Image
Image manipulation and processing:
- **ComputeResRatios.py** - Computes resolution ratios for images
- **CopyAndRenameImages.py** - Copies and renames images
- **DatasetMigrater.py** - Migrates datasets
- **ImageConverter.py** - Converts images between formats
- **ImageCropper.py** - Crops images
- **ImageUpscaler.py** - Upscales images
- **RandomFlip.py** - Randomly flips images

### Stats
Statistics and analysis tools:
- **TrainLogParser.py** - Parser for training logs
- **Run LogParse.bat** - Launcher for log parser
- **PngInfoPromptToTags.py** - Extracts prompts from PNG info and converts to tags
- **LaunchTensorboard.bat** - Shortcut for launching TensorBoard
- **PlotGraph.ipynb** - Jupyter notebook for plotting graphs

### Tag Files
Tag manipulation and management:
- **DanbooruTagScraper.py** - Scrapes tags from Danbooru API (reads credentials from `config.py`)
- **FilterTagFile.py** - Filters and processes tag files
- **anima_tags.csv** - Animation tags dataset
- **danbooru_tags_post_count.csv** - Danbooru tags with post counts
- **2026-06-24_danbooru_tags_post_count.csv** - Danbooru tags snapshot

### Dataset
Dataset manipulation tools:
- **FilenameRemoveBrackets.bat** - Removes brackets from filenames
- **RemoveCleanupTextFromName.py** - Removes cleanup text from filenames
- **SortTagsOutputAsIntervals.py** - Sorts tags output as intervals

### Lora
LoRA-related utilities:
- **AddMetadataThumbnail.py** - Adds metadata and thumbnails to LoRA files

## Modules
- **llama_src/** - llama.cpp source (C/C++ Python bindings for LLM inference)

## Archived
Deprecated or legacy tools:
- **AutoTaggerE621.py** - Legacy E621 autotagger
- **WebpToGif.py** - WebP to GIF converter
- **danbooru_tags_post_count.csv** - Legacy Danbooru tags data
- **2025-03-17_danbooru_tags_post_count.csv** - Old Danbooru tags snapshot
- **Caption UI/** - Legacy caption UI tools
  - `BuildLLama.bat` - Builds llama.cpp
  - `Run CaptionUI.bat` - Launches caption UI
  - `Captioner/` - Captioner module
- **Quantum Merge/** - Model merging tools
  - `Run QuantMerge.bat` - Launches Quantum Merge
  - `QuantumMerge-sdxl-main/` - SDXL Quantum Merge source

## VAE Comparison
- **vae-comparison/** - VAE model comparison tools (separate git repository)
  - `app.py` - Main application for VAE comparison
  - `README.md` - Documentation for VAE comparison
  - `requirements.txt` - Python dependencies for VAE comparison
  - `examples/` - Example comparison outputs
