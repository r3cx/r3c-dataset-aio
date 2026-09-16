"""
Dbv4Tagger — shared implementation for the animetimm Danbooru-v4 taggers.

The three ``animetimm/*.dbv4-full`` repos share one layout (transformers
``config.json`` + ``model.safetensors`` + a hand-rolled ``preprocess.json`` +
``selected_tags.csv`` + ``categories.json`` + ``thresholds.csv``) and there is
no ONNX file, so all of them load through
``transformers.AutoModelForImageClassification`` (the backbones are timm
models, wrapped by transformers' TimmWrapper). This base implements, once,
the load / preprocessing / vocabulary / decode pipeline that would otherwise be
triplicated verbatim.

A deliberate design choice: :meth:`tag_crops` returns a dense matrix whose
columns are *the reordered ``[general...] [character...]`` columns only*, with
``num_rating_tags = 0``. That matches the sliding-window autotagger's positional
contract (``vocab_names[i]`` ↔ output column ``num_rating_tags + i``) regardless
of where general/character tags actually sit inside the raw model head, and it
drops rating / genre / other columns at the tagger so the Autotagger never sees
them.

The base assumes the standard animetimm/dbv4 layout, which holds for all three
repos:

  * column order: ``selected_tags.csv`` rows are in model-output order with NO
    extra unlisted leading columns, i.e. ``len(csv) == config.num_labels`` —
    ratings are a *category* (9), not extra output columns.
    ``_log_metadata`` raises a clear error on any mismatch.
  * category names: "character" and "general" resolved *by name* —
    ``categories.json`` is a list of ``{"category": id, "name": name}`` entries
    and the CSV's ``category`` column carries the raw ids (0=general,
    4=character, 9=rating), mapped by :meth:`_normalize_categories`.
  * preprocessing: ``image_size`` + ``mean``/``std`` from the model's own
    ``preprocess.json`` (the ``test`` transform list: dghs-imgutils-style
    ``{"pre"/"test"/"val"}`` where each value is a list of transform step
    dicts). The pass is white pad-to-square (aspect preserved, nothing
    cropped) + bicubic resize to the input size + normalize; RGB; /255; CHW.
  * arch: ``AutoModelForImageClassification`` resolves each repo's
    ``config.json`` arch class — all three load as
    ``TimmWrapperForImageClassification`` over the timm backbone.

The model input size is sourced from each repo's ``preprocess.json`` — never
from the repo name (e.g. the ``...siglip_384`` repo has a 512 px input); a repo
that omits its size fails loudly in :meth:`_ensure_metadata` rather than being
silently mis-sized. Every assumption is printed by :meth:`_log_metadata` on
first load.
"""

from __future__ import annotations

import csv
import json
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from .Tagger import Tagger

# Category names (selected *by name*, not hardcoded ids).
_CHARACTER_CATEGORY = "character"
_GENERAL_CATEGORY = "general"

# Standard ImageNet normalization (fallback when preprocess.json omits them).
_DEFAULT_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_STD = [0.229, 0.224, 0.225]


class Dbv4Tagger(Tagger):
    """Shared base for the animetimm Danbooru-v4 PyTorch taggers.

    Concrete subclasses are thin: they set ``model_id`` and ``model_files``. All
    load / preprocess / vocab / decode behaviour lives here — including the
    model input size, which is sourced from each repo's ``preprocess.json``
    (the ``test`` ``resize`` step) in :meth:`_ensure_metadata`. There is no
    per-tagger fallback: a repo that omits its input size fails loudly in
    :meth:`_ensure_metadata` rather than being silently mis-sized.
    """

    # Files this tagger requires (skip the redundant ``pytorch_model.bin``
    # twin of ``model.safetensors``). ``thresholds.csv`` / ``categories.json``
    # are informational; the tagger never thresholds.
    _CONFIG_FILE = "config.json"
    _WEIGHTS_FILE = "model.safetensors"
    _PREPROCESS_FILE = "preprocess.json"
    _VOCAB_FILE = "selected_tags.csv"
    _CATEGORIES_FILE = "categories.json"
    _THRESHOLDS_FILE = "thresholds.csv"
    model_files = [
        _CONFIG_FILE,
        _WEIGHTS_FILE,
        _PREPROCESS_FILE,
        _VOCAB_FILE,
        _CATEGORIES_FILE,
        _THRESHOLDS_FILE,
    ]

    # The tagger emits only the reordered [general][character] columns (no
    # leading rating block) — see the module docstring.
    num_rating_tags = 0

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
        super().__init__(
            models_dir=models_dir,
            use_gpu=use_gpu,
            use_bf16=use_bf16,
            force_download=force_download,
            download=download,
            hf_token=hf_token,
            verbose=verbose,
        )
        self._model = None
        self._device: Optional[torch.device] = None
        # Computed dtype: bf16 (fast on Ampere+) when requested on the GPU, else fp32.
        self._dtype: torch.dtype = (
            torch.bfloat16 if (use_gpu and use_bf16) else torch.float32
        )

        self._pre: Dict = {}
        self._categories: Dict = {}
        self._names: List[str] = []
        self._cats: List[str] = []
        self._general_tags: Optional[List[str]] = None
        self._character_tags: Optional[List[str]] = None
        self._gen_idx: List[int] = []
        self._char_idx: List[int] = []
        self._col_order: List[int] = []
        # Input size resolved from preprocess.json in _ensure_metadata (there is no
        # per-tagger default — a repo that omits it is an error, see _ensure_metadata).
        self._image_size: Optional[int] = None
        self._mean: List[float] = list(_DEFAULT_MEAN)
        self._std: List[float] = list(_DEFAULT_STD)
        self._meta_ready = False

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    @property
    def config_path(self):
        return self.file_path(self._CONFIG_FILE)

    @property
    def preprocess_path(self):
        return self.file_path(self._PREPROCESS_FILE)

    @property
    def vocab_path(self):
        return self.file_path(self._VOCAB_FILE)

    @property
    def categories_path(self):
        return self.file_path(self._CATEGORIES_FILE)

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------
    @property
    def general_tags(self) -> List[str]:
        self._require_loaded()
        return self._general_tags

    @property
    def character_tags(self) -> List[str]:
        self._require_loaded()
        return self._character_tags

    @property
    def vocab_names(self) -> List[str]:
        """Tag names aligned to the dense columns emitted by :meth:`tag_crops`
        (``general_tags + character_tags``)."""
        return self.general_tags + self.character_tags

    # ------------------------------------------------------------------
    # Metadata / vocabulary parsing
    # ------------------------------------------------------------------
    def _ensure_metadata(self) -> None:
        """Parse ``preprocess.json`` / ``categories.json`` /
        ``selected_tags.csv`` and compute the [general][character] column
        permutation. Called by :meth:`_load` (and cheaply re-checkable)."""
        if not self._meta_ready:
            # Preprocessing parameters. ``preprocess.json`` may be a dghs-imgutils
            # dict {"pre": [...], "test": [...], "val": [...]} where each value is a
            # LIST of transform step-dicts, or a flat dict ({"size":..., "mean":...}).
            # We always resolve from the *test* config when present.
            with open(self.preprocess_path, "r", encoding="utf-8") as f:
                self._pre = json.load(f) or {}
            test_cfg = (
                self._pre.get("test", self._pre)
                if isinstance(self._pre, dict)
                else self._pre
            )
            self._image_size = self._resolve_image_size(test_cfg)
            if self._image_size is None:
                raise RuntimeError(
                    f"Could not resolve a model input size for '{self.name}' from "
                    f"{self.preprocess_path}. The v4 repos declare it via the test "
                    "'resize' step (or a flat 'size'/'image_size' key); add it there "
                    "— the tagger does not guess an input size."
                )
            self._mean, self._std = self._pick_normalize(test_cfg)

            # Category id → name mapping (may be a list of {category, name} objects,
            # a flat {id: name} dict, or absent). Normalised to a {str-key: name} map.
            try:
                with open(self.categories_path, "r", encoding="utf-8") as f:
                    self._categories = self._normalize_categories(json.load(f) or {})
            except FileNotFoundError:
                self._categories = {}

            # Vocabulary (assumed in model-output order; checked by _log_metadata).
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            header = [h.strip().lower() for h in rows[0]]
            if "name" not in header or "category" not in header:
                raise AssertionError(
                    f"Unexpected selected_tags.csv header (need 'name' and "
                    f"'category'): {rows[0]}"
                )
            nsi, ci = header.index("name"), header.index("category")
            self._names, self._cats = [], []
            for r in rows[1:]:
                if len(r) <= max(nsi, ci) or not r[nsi].strip():
                    continue
                self._names.append(r[nsi].strip())
                self._cats.append(r[ci].strip())

            self._split_columns()
            self._meta_ready = True

    def _split_columns(self) -> None:
        """Classify each csv row by (name-resolved) category and record the
        [general][character] column order used to reorder the dense output."""
        general: List[str] = []
        character: List[str] = []
        self._gen_idx, self._char_idx = [], []
        for idx, (name, raw) in enumerate(zip(self._names, self._cats)):
            cat = str(self._categories.get(raw, raw)).strip().lower()
            if cat == _CHARACTER_CATEGORY:
                character.append(name)
                self._char_idx.append(idx)
            elif cat == _GENERAL_CATEGORY:
                general.append(name)
                self._gen_idx.append(idx)
            # rating / genre / other categories are dropped from the output.
        self._general_tags = general
        self._character_tags = character
        self._col_order = self._gen_idx + self._char_idx

    @staticmethod
    def _normalize_categories(raw) -> Dict[str, str]:
        """Build a `{str-key: name}` lookup from ``categories.json``.

        The animetimm repos ship ``categories.json`` as a LIST of
        ``{"category": <id>, "name": <name>}`` objects (e.g.
        ``[{"category":0,"name":"general"}, {"category":4,"name":"character"},
        {"category":9,"name":"rating"}]``); a ``{id: name}`` dict is handled too.
        The CSV's ``category`` column carries the *id* as a string, so the lookup is
        keyed by ``str(id)``; a name→name entry is added as well so a CSV that
        already carries names also resolves.
        """
        mapping: Dict[str, str] = {}
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                cid = item.get("category")
                name = str(item.get("name", "")).strip().lower()
                if cid is not None:
                    mapping[str(cid)] = name
                if name:
                    mapping[name] = name
        elif isinstance(raw, dict):
            for key, value in raw.items():
                name = str(value).strip().lower()
                mapping[str(key)] = name
                if name:
                    mapping[name] = name
        return mapping

    def _resolve_image_size(self, pre) -> Optional[int]:
        """Input side. Returns the model input side from a dghs-imgutils ``test``
        transform *list* (the ``resize`` step) or a flat ``dict``
        (``image_size``/``size``/``img_size``); returns ``None`` if it can't be
        found (the caller raises — there is no per-tagger fallback). A leading
        ``pad_to_size`` step is ignored (it pads, not the model's input side)."""
        if isinstance(pre, list):
            sizes = []
            for step in pre:
                if not isinstance(step, dict):
                    continue
                s = step.get("size")
                if isinstance(s, (list, tuple)) and s:
                    sizes.append((step.get("type"), s))
                elif isinstance(s, int):
                    sizes.append((step.get("type"), s))
            for step_type, s in reversed(sizes):
                if step_type == "resize":
                    return int(s[0]) if isinstance(s, (list, tuple)) else int(s)
            if sizes:
                s = sizes[-1][1]
                return int(s[0]) if isinstance(s, (list, tuple)) else int(s)
            return None
        if isinstance(pre, dict):
            for key in ("image_size", "size", "img_size"):
                v = pre.get(key)
                if v is None:
                    continue
                if isinstance(v, (list, tuple)):
                    v = v[0]
                if v:
                    return int(v)
        return None

    def _pick_normalize(self, pre):
        """(mean, std) from the ``normalize`` step of a dghs-imgutils ``test``
        transform list, or from a flat ``dict``. Falls back to ImageNet defaults."""
        steps = pre if isinstance(pre, list) else [pre] if isinstance(pre, dict) else []
        mean, std = list(_DEFAULT_MEAN), list(_DEFAULT_STD)
        for step in reversed(steps):
            if isinstance(step, dict):
                if step.get("type") == "normalize" and step.get("mean") and step.get("std"):
                    return list(map(float, step["mean"])), list(map(float, step["std"]))
                if "mean" in step and "std" in step:
                    mean, std = list(map(float, step["mean"])), list(map(float, step["std"]))
        return mean, std

    def _log_metadata(self) -> None:
        """Hard-fail with actionable guidance if the column-order assumption is
        wrong; print the metadata block only in verbose mode."""
        cfg = self._model.config
        num_labels = getattr(cfg, "num_labels", None)
        if num_labels is None:
            num_labels = getattr(cfg, "num_classes", None)
        if self.verbose:
            print(f"Dbv4 metadata for {self.name}:")
            print(f"  arch class      : {type(self._model).__name__}")
            print(f"  input           : image_size={self._image_size}  mean={self._mean}  std={self._std}")
            print(f"  preprocess.json : {self._pre}")
            print(f"  config.num_labels: {num_labels}   selected_tags.csv rows: {len(self._names)}")
            print(f"  categories.json : {self._categories}")
            print(f"  general tags    : {len(self._general_tags)}")
            print(f"  character tags  : {len(self._character_tags)}")
        if num_labels is not None and num_labels != len(self._names):
            raise RuntimeError(
                f"selected_tags.csv rows ({len(self._names)}) != "
                f"config.num_labels ({num_labels}). The dbv4 output-column "
                "convention assumed by Dbv4Tagger is wrong for this model. "
                "Inspect selected_tags.csv + config.json and fix "
                "Dbv4Tagger._ensure_metadata / _split_columns."
            )

    @property
    def crop_resize_spec(self):
        """DataLoader workers resize crops with the PIL bicubic path (matching
        this tagger's ``_preprocess_image``) before shipping them back."""
        if not self._meta_ready:
            self._ensure_metadata()
        return (self._image_size, "pil_bicubic")

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------
    def _load(self) -> None:
        self._ensure_metadata()
        from transformers import AutoModelForImageClassification  # deferred import

        self._device = torch.device("cuda" if self.use_gpu else "cpu")
        self._model = AutoModelForImageClassification.from_pretrained(
            str(self.model_dir), use_safetensors=True
        )
        if self.use_gpu:
            self._log(f"Tagger: using GPU for {self.name} (device {self._device})")
            self._model = self._model.to(self._device)
            if self._dtype is torch.bfloat16:
                self._model = self._model.to(dtype=torch.bfloat16)
        self._model.eval()
        self._log_metadata()
        self._log(f"Tagger: inference dtype = {self._dtype}")

    def _unload(self) -> None:
        self._model = None
        if self.use_gpu and torch is not None:
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Preprocessing (the model's own transform from preprocess.json: white
    # pad-to-square, bicubic resize to the input size, ImageNet mean/std)
    # ------------------------------------------------------------------
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert one original-resolution image/crop to a model-ready tensor.

        Implements each repo's real ``preprocess.json`` ``test`` transform: the
        whole image is padded onto a white square (aspect preserved, nothing is
        cropped), bicubically resized to the input size, then normalised with
        the repo's mean/std. This is the preprocessing the model requires, so
        :meth:`tag` / :meth:`tag_crops` use exactly this geometry — there is no
        separate center-crop pass.
        """
        size = self._image_size
        image = image.convert("RGB")

        # Pad to a square (side = max side) with white, centred. The model's
        # transform pads white — it does not crop — so the entire image is kept
        # (matters for tall/wide portraits, where center-cropping discards the
        # head/feet).
        w, h = image.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(image, ((side - w) // 2, (side - h) // 2))

        # Already at model input size (pre-sized by a DataLoader worker via
        # ``crop_resize_spec``) — the resize is a no-op; skip it.
        if canvas.size != (size, size):
            # Bicubic resize to the model input size (antialias per the JSON).
            image = canvas.resize((size, size), Image.BICUBIC)
        else:
            image = canvas

        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        return torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW

    # ------------------------------------------------------------------
    # Inference + public tagging interface
    # ------------------------------------------------------------------
    def _infer_batch(self, batch: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            out = self._model(pixel_values=batch)
            logits = out.logits if hasattr(out, "logits") else out[0]
            # .float() is a no-op in fp32; required before .numpy() in bf16
            # (torch has no direct bfloat16 -> numpy path).
            probs = torch.sigmoid(logits).float().cpu().numpy()  # (N, num_labels)
        # Reorder to the [general][character] dense-matrix contract.
        return np.ascontiguousarray(np.take(probs, self._col_order, axis=1))

    def tag_crops(self, crops: List[Image.Image]) -> np.ndarray:
        """Preprocess and infer over a list of original-resolution crops.

        Returns a dense ``(num_crops, num_general + num_character)`` matrix of
        sigmoid probabilities, columns ordered as ``vocab_names`` (no leading
        rating columns).
        """
        self._require_loaded()
        batch = torch.stack(
            [self._preprocess_image(c) for c in crops], axis=0
        ).to(self._device, dtype=self._dtype)
        return self._infer_batch(batch)

    def tag(self, image: Image.Image) -> Dict[str, float]:
        """Tag a single image; returns a sparse ``{tag_name: confidence}`` for
        every tag above zero (no thresholds)."""
        self._require_loaded()
        probs = self.tag_crops([image])[0]
        names = self.vocab_names
        return {
            names[i].replace("_", " "): float(probs[i])
            for i in range(len(names))
            if probs[i] > 0.0
        }
