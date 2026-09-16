"""
Autotagger base class — generic image-tagging workflow.

Deliberately thin: contains only what is genuinely shared by every
``Autotagger`` in the project, regardless of which autotagger
strategy it uses (sliding-window, full-image-only, etc.):

    * common configuration access
    * image discovery
    * output .txt generation (append / overwrite)
    * frequency tracking
    * shared final-selection building blocks (kaomoji, undesired)
    * progress reporting and error strings

It contains **no** sliding-window, composition-routing, or model-specific
logic. Those live in ``SlidingWindowAutotagger``.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
IMAGE_EXTENSIONS = [
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
    ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP",
]

FILETYPE_TXT = ".txt"

# Error prefixes
ERROR = "Error: "
FILE_OPEN_ERROR = "Failed to open file: "

# Kaomoji tags that look like noise (underscore-separated facial expressions).
DEFAULT_KAOMOJIS = (
    "0_0, (o)_(o), +_+, +_-, ._., <o>_<o>, <|>_<|>, =_=, >_<, 3_3, 6_9, >_o, @_@, "
    "^_^, o_o, u_u, x_x, |_|, ||_||"
)


# -----------------------------------------------------------------------------
# Base class
# -----------------------------------------------------------------------------
class Autotagger(ABC):
    """Shared image-tagging workflow.

    Concrete autotaggers (directly or via ``SlidingWindowAutotagger``) set up
    their Tagger(s) and implement ``_process_images``. ``run`` handles
    everything around it: discovery, processing, and the end-of-run frequency
    report.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config
        self.tag_frequencies = {}
        self._kaomoji_set = {t.strip() for t in DEFAULT_KAOMOJIS.split(",")}

    # ------------------------------------------------------------------
    # Configuration access
    # ------------------------------------------------------------------
    def _cfg(self, name: str, default=None):
        if self.config is None:
            return default
        get = getattr(self.config, "get", None)
        if callable(get):
            return self.config.get(name, default)
        return getattr(self.config, name, default)

    # --- core settings ----------------------------------------------------
    @property
    def data_dir(self) -> str:
        return self._cfg("data_dir")

    @property
    def recursive(self) -> bool:
        return bool(self._cfg("recursive_gather", False))

    @property
    def mode_append(self) -> bool:
        return bool(self._cfg("mode_append", False))

    @property
    def frequency_tags(self) -> bool:
        return bool(self._cfg("frequency_tags", False))

    @property
    def use_gpu(self) -> bool:
        return bool(self._cfg("use_gpu", False))

    @property
    def use_bf16(self) -> bool:
        return bool(self._cfg("bf16", False))

    @property
    def force_download(self) -> bool:
        return bool(self._cfg("force_download", False))

    @property
    def num_workers(self) -> Optional[int]:
        return self._cfg("num_data_loader_workers", None)

    @property
    def verbose(self) -> bool:
        return bool(self._cfg("verbose", False))

    @property
    def debug_print(self) -> bool:
        """Per-image inference breakdown: enabled by ``--debug_print`` or
        implied by ``--verbose``."""
        return bool(self._cfg("debug_print", False) or self._cfg("verbose", False))

    # --- thresholds (resolve the shared --threshold default) --------------
    @property
    def default_threshold(self) -> float:
        return float(self._cfg("threshold", 0.5))

    def _resolved_threshold(self, name: str) -> float:
        value = self._cfg(name)
        return float(value) if value is not None else self.default_threshold

    @property
    def general_threshold(self) -> float:
        return self._resolved_threshold("general_threshold")

    @property
    def character_threshold(self) -> float:
        return self._resolved_threshold("character_threshold")

    # --- filtering --------------------------------------------------------
    @property
    def undesired(self) -> set:
        tags = self._cfg("undesired_tags", "")
        return {t for t in str(tags).split(",") if t}

    @property
    def kaomoji_set(self) -> set:
        return self._kaomoji_set

    # ------------------------------------------------------------------
    # Image discovery
    # ------------------------------------------------------------------
    def gather_image_paths(self, dir_path: Path, recursive: bool = False) -> List[Path]:
        """Collect sorted, deduplicated image file paths from a directory."""
        print(f"Accessing {dir_path}")
        image_paths: List[Path] = []
        search_fn = dir_path.rglob if recursive else dir_path.glob
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(search_fn("*" + ext))
        image_paths = sorted(set(image_paths))
        print(f"Found {len(image_paths)} images.")
        return image_paths

    # ------------------------------------------------------------------
    # Output generation
    # ------------------------------------------------------------------
    def write_tags(self, image_path, tag_text: str, mode_append: Optional[bool] = None) -> None:
        """Write the tag string to the .txt file beside the image.

        ``mode_append`` defaults to the configured append mode.
        """
        if mode_append is None:
            mode_append = self.mode_append

        image_path = str(image_path)
        txt_path = Path(image_path).with_suffix(FILETYPE_TXT)

        if not mode_append:
            txt_path.write_text(f"{tag_text}\n", encoding="utf-8")
            return

        prefix = ", " if txt_path.exists() and txt_path.stat().st_size > 0 else ""
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(f"{prefix}{tag_text}\n")

    # ------------------------------------------------------------------
    # Frequency tracking
    # ------------------------------------------------------------------
    def update_frequencies(self, freq_updates: dict) -> None:
        for tag, count in freq_updates.items():
            self.tag_frequencies[tag] = self.tag_frequencies.get(tag, 0) + count

    def report_frequencies(self) -> None:
        print("\nTag frequencies:")
        for tag, freq in sorted(self.tag_frequencies.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tag}: {freq}")

    # ------------------------------------------------------------------
    # Shared tag-selection building blocks
    # ------------------------------------------------------------------
    def is_kaomoji(self, tag: str) -> bool:
        """True if the underscore-form tag name is a known kaomoji."""
        return tag in self._kaomoji_set

    def is_excluded(self, tag: str) -> bool:
        """True if a human-readable tag name should be dropped because it is in
        the user-specified undesired set."""
        return tag in self.undesired

    # ------------------------------------------------------------------
    # Progress / error reporting
    # ------------------------------------------------------------------
    def progress(self, iterable, desc: str = "Processing"):
        return tqdm(iterable, desc=desc, smoothing=0.0)

    # ------------------------------------------------------------------
    # CLI (shared) — the reference flags verbatim, plus the gated-model flags;
    # each concrete leaf gets an ``if __name__ == "__main__":`` block that calls
    # ``cls.from_cli().run()`` (see the per-leaf ``__main__`` blocks).
    # ------------------------------------------------------------------
    @classmethod
    def from_cli(cls, argv: Optional[List[str]] = None) -> "Autotagger":
        """Parse ``argv`` (defaults to ``sys.argv[1:]``), build a config dict,
        and return an instance of ``cls`` configured by it."""
        args = cls._build_parser().parse_args(argv)
        return cls(vars(args))

    @classmethod
    def _build_parser(cls) -> argparse.ArgumentParser:
        """Standard parser: the shared reference flags (this class) plus any the
        leaf contributes via :meth:`_add_cli_args`."""
        parser = argparse.ArgumentParser(
            prog=cls.__name__,
            description=f"{cls.__name__} image autotagger.",
        )
        cls._base_cli_args(parser)
        cls._add_cli_args(parser)
        return parser

    @staticmethod
    def _base_cli_args(parser: argparse.ArgumentParser) -> None:
        """Add the shared flags mirroring the reference, plus the new
        gated-model flags (``--hf_token`` / ``--no_download``)."""
        parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images to tag")
        parser.add_argument("--num_data_loader_workers", type=int, default=None, help="Parallel workers for image loading (None = single-threaded)")
        parser.add_argument("--threshold", type=float, default=0.5, help="Default confidence threshold (overridden by category-specific flags)")
        parser.add_argument("--general_threshold", type=float, default=None, help="Confidence threshold for general tags")
        parser.add_argument("--character_threshold", type=float, default=None, help="Confidence threshold for character tags")
        parser.add_argument("--undesired_tags", type=str, default="", help="Comma-separated tags to exclude from output")
        parser.add_argument("--recursive_gather", action="store_true", help="Recursively search subdirectories for images")
        parser.add_argument("--frequency_tags", action="store_true", help="Print tag frequency report after processing")
        parser.add_argument("--force_download", action="store_true", help="Force re-download of tagger models from HuggingFace")
        parser.add_argument("--mode_append", action="store_true", help="Append tags to existing .txt instead of overwriting")
        parser.add_argument("--verbose", action="store_true", help="Print all debug output: model/metadata dumps, per-phase details, and the per-image inference breakdown (implies --debug_print)")
        parser.add_argument("--debug_print", action="store_true", help="Print per-image tag breakdown")
        parser.add_argument("--window_size_cap", type=int, default=0, help="Maximum window size cap in pixels (0 = use image short axis)")
        parser.add_argument("--window_stride_ratio", type=float, default=0.9, help="Window stride ratio (10%% overlap at 0.9)")
        parser.add_argument("--use_sequential", action="store_true", help="Use sequential inference instead of batched (lower memory)")
        parser.add_argument("--use_gpu", action="store_true", help="Use CUDA GPU for inference")
        parser.add_argument("--bf16", action="store_true", help="Run the PyTorch (dbv4) taggers in bfloat16 — ~2-4x faster forwards and half the VRAM on Ampere+ GPUs (e.g. RTX 3090); the ONNX WD1.4 tagger is unaffected")
        parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace read token for gated model download (else HF_TOKEN / hf login)")
        parser.add_argument("--no_download", action="store_true", help="Never download; load models only from the local Models/ dir (error if missing)")

    @classmethod
    def _add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Leaf hook for extra flags. No-op by default; leaves override to add
        their own (e.g. the ensemble leaf's ``--taggers``)."""

    # ------------------------------------------------------------------
    # Core hook + top-level run
    # ------------------------------------------------------------------
    @abstractmethod
    def _process_images(self, image_paths) -> None:
        """Concrete autotaggers implement the full tagging loop here, iterating
        over ``image_paths`` and writing outputs."""
        raise NotImplementedError

    def run(self) -> None:
        """Top-level entry point: discover images, process them, report."""
        image_paths = self.gather_image_paths(Path(self.data_dir), self.recursive)
        self._process_images(image_paths)
        if self.frequency_tags:
            self.report_frequencies()
        print("\nAutotagging Completed!")