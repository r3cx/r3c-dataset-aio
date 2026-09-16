"""
Tagger — abstract, model-agnostic base for a single tagger model.

The base is deliberately a thin piece of *infrastructure*. It knows nothing
about any particular model family, weight format, or tag vocabulary. Its only
job is to give every concrete tagger a shared foundation so each tagger can be
self-sufficient:

    * **Model directory** — where a tagger's files live locally. The models
      folder is relatively pathed from this file (``TagManagement/Models``) and
      each tagger occupies ``Models/<name>``.
    * **Downloading** — :meth:`Tagger.ensure_files` fetches a tagger's declared
      files (``model_files``) from HuggingFace into that folder, downloading
      only what is missing (or all of it under ``force_download``).
    * **Lifecycle** — :meth:`load` / :meth:`unload` (also a context manager)
      wrap the tagger-specific :meth:`_load` / :meth:`_unload` hooks, so
      construction is side-effect-free and a tagger becomes ready on first use.
    * **Public interface** — :meth:`tag` (sparse) and :meth:`tag_crops` (dense)
      plus the vocabulary accessors. These are the hooks the Autotagger /
      Ensemble layers call; each concrete tagger supplies its own behaviour.

The base does **not** open, parse, or interpret any model file, does **not**
apply any threshold or tag filtering, and does **not** infer which files a
tagger needs. All of that is the concrete tagger's responsibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from huggingface_hub import get_token, hf_hub_download
from PIL import Image


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Default location of local model weights. This file lives in
# TagManagement/Taggers/, so the project's Models/ folder is one level up.
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "Models"


class Tagger(ABC):
    """Abstract, model-agnostic base for a single tagger.

    Concrete taggers set the class-level configuration (``model_id``,
    ``model_files``, and their own settings), implement the load hooks
    (``_load`` / ``_unload``) and the public interface (``tag`` / ``tag_crops``
    plus the vocabulary accessors). Everything else — pathing, downloading, and
    the load/unload lifecycle — is provided here.
    """

    # ------------------------------------------------------------------
    # Configuration — set by each concrete tagger
    # ------------------------------------------------------------------
    #: HuggingFace repository id the model files are downloaded from.
    model_id: str = ""
    #: Local model folder name. Defaults to the repository basename when unset.
    model_name: str = ""
    #: Files this tagger needs, by name, relative to the repository root / the
    #: local model folder. The base downloads them as-is and never infers this
    #: list — every concrete tagger declares what it requires itself.
    model_files: List[str] = []
    #: Number of leading non-tag ("rating") outputs a model emits before its
    #: real tags. A neutral default; each tagger sets what its model emits.
    num_rating_tags: int = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        models_dir: Optional[object] = None,
        use_gpu: bool = False,
        use_bf16: bool = False,
        force_download: bool = False,
        download: bool = True,
        hf_token: Optional[str] = None,
        verbose: bool = False,
    ):
        self.models_dir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
        self.use_gpu = use_gpu
        # PyTorch taggers: run the model in bfloat16 (Ampere+ GPUs, e.g. RTX
        # 3090) for ~2-4x faster forwards and half the VRAM. Onnx taggers
        # ignore this (their graphs are fixed-precision).
        self.use_bf16 = use_bf16
        self.force_download = force_download
        self.download = download
        # Explicit per-run HuggingFace read credential for gated repos. When
        # unset, huggingface_hub falls back to HF_TOKEN / the hf-login token.
        self.hf_token = hf_token
        # Debug logging: verbose-only lines (model provenance, metadata dumps,
        # device/dtype confirmation) print only when this is set.
        self.verbose = verbose

        self._loaded = False

    def _log(self, message: str) -> None:
        """Print ``message`` only in verbose mode (debug diagnostics)."""
        if self.verbose:
            print(message)

    # ------------------------------------------------------------------
    # Model directory / pathing
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Folder name for this tagger's local model files."""
        return self.model_name or self.model_id.rsplit("/", 1)[-1]

    @property
    def model_dir(self) -> Path:
        """Local folder where this tagger's files are stored / downloaded to."""
        return self.models_dir / self.name

    def file_path(self, filename: str) -> Path:
        """Path to ``filename`` within this tagger's model folder."""
        return self.model_dir / filename

    # ------------------------------------------------------------------
    # File download infrastructure (HuggingFace -> local model folder)
    # ------------------------------------------------------------------
    def ensure_files(self, files: Optional[List[str]] = None) -> Path:
        """Ensure the tagger's files exist locally, downloading any that don't.

        Files are fetched from :attr:`model_id` into :attr:`model_dir`. When
        :attr:`force_download` is set the whole set is re-fetched; otherwise
        only the missing files are downloaded. Returns :attr:`model_dir`.

        This is pure transfer plumbing — it never inspects file contents. A
        tagger whose weights come from elsewhere can simply leave
        :attr:`model_files` empty so this is a no-op.
        """
        files = self.model_files if files is None else list(files)
        to_download = (
            files
            if self.force_download
            else [fname for fname in files if not self.file_path(fname).is_file()]
        )
        if to_download:
            if not (self.hf_token or get_token()):
                # No resolvable credential, but a download is required (gated
                # repo). Warn with actionable remediation, then attempt anyway —
                # the attempt fails cleanly (no retry loop) if it is truly gated.
                print(
                    "\nWARNING: no HuggingFace credential found, but a download "
                    f"is required for '{self.model_id}'."
                    "\n    Set one and retry, choosing any of:"
                    "\n      • HF_TOKEN=hf_... environment variable"
                    '\n      • run "hf login" in the venv'
                    "      • --hf_token hf_... on the command line"
                    f"      • pre-place the files under {self.model_dir} and use --no_download"
                    "\n    Attempting the download anyway; it will fail without a token.\n"
                )
            self.model_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading tagger model from HF: {self.model_id} ({', '.join(to_download)})")
            for fname in to_download:
                # local_dir places each file flat at model_dir/<fname> (matching
                # file_path). The old cache_dir + force_filename pattern makes
                # modern huggingface_hub build a nested models--<repo>/snapshots/
                # tree under model_dir instead, so the file would not resolve.
                hf_hub_download(
                    self.model_id,
                    fname,
                    local_dir=str(self.model_dir),
                    force_download=self.force_download,
                    token=self.hf_token,
                )
        else:
            self._log(f"Using existing tagger model: {self.model_id}")
        return self.model_dir

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> "Tagger":
        """Ensure files are present, then load the model. Idempotent.

        Shared template: download the missing files (when :attr:`download`),
        then delegate to the tagger-specific :meth:`_load`. Override entirely
        only if a tagger has needs beyond this flow.
        """
        if not self._loaded:
            if self.download:
                self.ensure_files()
            self._load()
            self._loaded = True
        return self

    def unload(self) -> None:
        """Release the model via :meth:`_unload`, if loaded."""
        if self._loaded:
            self._unload()
            self._loaded = False

    def __enter__(self) -> "Tagger":
        return self.load()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unload()

    def _require_loaded(self) -> None:
        """Lazily trigger :meth:`load` so the public methods work without an
        explicit call to it."""
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------
    # Worker-side crop presize (optional, declared by concrete taggers)
    # ------------------------------------------------------------------
    @property
    def crop_resize_spec(self):
        """Optional ``(input_size_px, resizer)`` hint for the sliding-window
        autotagger's DataLoader workers.

        When returned, each original-resolution crop is white-padded to a square
        and resized to ``input_size_px`` **inside the worker** (where it runs
        in parallel with the other workers) before being shipped back to the
        main process. Original-resolution crops can be ~60 MB each (a 4500 px
        window of a 4K+ image); at model input size they are ~1 MB — a >50x
        smaller pickle, and the expensive resize no longer occupies the single
        main thread that also drives inference.

        ``resizer`` names the tagger's own resize path so worker output stays
        bit-identical to the tagger's ``_preprocess_image``: ``"cv2"`` (WD1.4
        ONNX: INTER_AREA downscale / INTER_LANCZOS4 upscale) or ``"pil_bicubic"``
        (dbv4 PyTorch). The cheap remainder (float conversion + normalization)
        still runs in the tagger's ``_preprocess_image`` on the main thread.

        ``None`` (default) keeps the original behaviour: crops travel at
        original resolution and the tagger resizes on the main thread.
        """
        return None

    # ------------------------------------------------------------------
    # Tagger-specific load/unload hooks — override in each concrete tagger
    # ------------------------------------------------------------------
    @abstractmethod
    def _load(self) -> None:
        """Open/read this tagger's model. Called by :meth:`load` once its files
        are guaranteed present. The tagger is responsible for its own I/O here."""
        raise NotImplementedError

    @abstractmethod
    def _unload(self) -> None:
        """Release whatever :meth:`_load` opened."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public tagging interface — override in each concrete tagger
    # ------------------------------------------------------------------
    @abstractmethod
    def tag(self, image: Image.Image) -> Dict[str, float]:
        """Tag a single original-resolution image; returns a sparse
        ``{tag_name: confidence}`` mapping. No thresholds are applied."""
        raise NotImplementedError

    @abstractmethod
    def tag_crops(self, crops: List[Image.Image]) -> np.ndarray:
        """Tag a batch of original-resolution crops; returns a dense
        ``(num_crops, num_outputs)`` probability matrix including the leading
        rating outputs."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Vocabulary accessors — each tagger exposes its own tag layout
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def general_tags(self) -> List[str]:
        """General (non-character) tag names, in model-output order."""

    @property
    @abstractmethod
    def character_tags(self) -> List[str]:
        """Character tag names, in model-output order."""

    @property
    @abstractmethod
    def vocab_names(self) -> List[str]:
        """All tag names in model-output order
        (``general_tags + character_tags``)."""
