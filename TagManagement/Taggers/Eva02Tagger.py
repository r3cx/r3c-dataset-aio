"""
Eva02Tagger — WD1.4 EVA02-Large tagger model (ONNX).

Concrete :class:`Tagger` for the existing WD1.4 ``wd-eva02-large-tagger-v3``
model (SmilingWolf). It is the regression-baseline model, so its preprocessing,
input resolution, and inference behaviour must match the reference
``AutoTaggerExp.py`` exactly.

This tagger owns all of its model-specific halves that the abstract base leaves
open: the files it requires, its ONNX inference path, the WD1.4
``selected_tags.csv`` vocabulary layout, and its preprocessing. The base
supplies the model-directory pathing, the HuggingFace download, and the
load/unload lifecycle.
"""

from __future__ import annotations

import csv
from typing import Dict, List, Optional

import cv2
import numpy as np
import onnxruntime
from PIL import Image

from .Tagger import Tagger


class Eva02Tagger(Tagger):
    """WD1.4 EVA02-Large tagger (SmilingWolf ``wd-eva02-large-tagger-v3``).

    Weights live in ``Models/wd-eva02-large-tagger-v3`` (``model.onnx`` +
    ``selected_tags.csv``) and are pulled from HuggingFace on first load if any
    are missing.
    """

    # ------------------------------------------------------------------
    # Tagger configuration
    # ------------------------------------------------------------------
    model_id = "SmilingWolf/wd-eva02-large-tagger-v3"
    model_name = "wd-eva02-large-tagger-v3"

    #: WD1.4 EVA02 input resolution (pixels).
    input_size = 448

    # Files this tagger requires, as stored in the HuggingFace repo / model dir.
    _WEIGHTS_FILE = "model.onnx"
    _VOCAB_FILE = "selected_tags.csv"
    model_files = [_WEIGHTS_FILE, _VOCAB_FILE]

    # selected_tags.csv layout (WD1.4): category "0" = general, "4" = character.
    _CSV_HEADER = ("tag_id", "name", "category")
    _CSV_HEADER_ERROR = "Unexpected .csv header format detected: "
    _GENERAL_CATEGORY = "0"
    _CHARACTER_CATEGORY = "4"

    # WD1.4 emits 4 leading rating outputs before the tagged outputs.
    num_rating_tags = 4

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
        # ``use_bf16`` is accepted for a uniform tagger interface but has no
        # effect here: the ONNX graph is fixed-precision (runs fp16 internally
        # on the CUDA EP when the provider is enabled).
        super().__init__(
            models_dir=models_dir,
            use_gpu=use_gpu,
            use_bf16=use_bf16,
            force_download=force_download,
            download=download,
            hf_token=hf_token,
            verbose=verbose,
        )
        self._session: Optional[onnxruntime.InferenceSession] = None
        self._general_tags: Optional[List[str]] = None
        self._character_tags: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    @property
    def weights_path(self):
        """Local path of the ONNX weights file."""
        return self.file_path(self._WEIGHTS_FILE)

    @property
    def vocab_path(self):
        """Local path of the ``selected_tags.csv`` vocabulary."""
        return self.file_path(self._VOCAB_FILE)

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
        """Tag names (underscore form) aligned to the tagged model outputs, in
        the order [general tags..., character tags...]."""
        return self.general_tags + self.character_tags

    # ------------------------------------------------------------------
    # Load / unload (model-specific I/O)
    # ------------------------------------------------------------------
    def _load(self) -> None:
        self._session = onnxruntime.InferenceSession(
            str(self.weights_path), providers=self.providers
        )
        self._load_vocab()

    def _unload(self) -> None:
        self._session = None
        self._general_tags = None
        self._character_tags = None

    @property
    def providers(self) -> List[str]:
        return (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_gpu
            else ["CPUExecutionProvider"]
        )

    @property
    def crop_resize_spec(self):
        """DataLoader workers resize crops with the cv2 path (matching this
        tagger's ``_preprocess_image``) before shipping them back."""
        return (self.input_size, "cv2")

    def _load_vocab(self) -> None:
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        header, data = rows[0], rows[1:]
        assert header[0] == self._CSV_HEADER[0] and header[1] == self._CSV_HEADER[1] \
            and header[2] == self._CSV_HEADER[2], f"{self._CSV_HEADER_ERROR}{header}"
        self._general_tags = [row[1] for row in data if row[2] == self._GENERAL_CATEGORY]
        self._character_tags = [row[1] for row in data if row[2] == self._CHARACTER_CATEGORY]

    # ------------------------------------------------------------------
    # Preprocessing (WD1.4 — must match the reference verbatim)
    # ------------------------------------------------------------------
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Convert a single original-resolution crop to a model-ready tensor.

        Verbatim port of the reference ``preprocess_image``. Returns
        ``(input_size, input_size, 3)`` float32; :meth:`tag_crops` stacks these
        into the ``(num_crops, input_size, input_size, 3)`` batch.
        """
        image = image.convert("RGB")
        arr = np.array(image)[:, :, ::-1]  # RGB -> BGR

        # Pad to a square with white borders (aspect ratio preserved).
        size = max(arr.shape[0:2])
        pad_x, pad_y = size - arr.shape[1], size - arr.shape[0]
        pad_l, pad_t = pad_x // 2, pad_y // 2
        arr = np.pad(
            arr,
            ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        # Already at model input size (pre-sized by a DataLoader worker via
        # ``crop_resize_spec``) — the resize is a no-op; skip it. Otherwise
        # INTER_AREA for shrinking, INTER_LANCZOS4 for expanding.
        if arr.shape[0:2] != (self.input_size, self.input_size):
            interp = cv2.INTER_AREA if size > self.input_size else cv2.INTER_LANCZOS4
            arr = cv2.resize(arr, (self.input_size, self.input_size), interpolation=interp)
        return arr.astype(np.float32)

    # ------------------------------------------------------------------
    # Inference + public tagging interface
    # ------------------------------------------------------------------
    def _infer_batch(self, batch: np.ndarray) -> np.ndarray:
        """Run the ONNX session on a preprocessed batch, returning a dense
        ``(num_images, num_outputs)`` array (raw model outputs)."""
        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name
        return self._session.run([output_name], {input_name: batch})[0]

    def tag_crops(self, crops: List[Image.Image]) -> np.ndarray:
        """Preprocess and infer over a list of original-resolution crops.

        Returns a dense array of shape ``(num_crops, num_outputs)`` — raw model
        outputs including the leading rating entries, which the caller (the
        Autotagger) strips via :attr:`num_rating_tags`.
        """
        self._require_loaded()
        batch = np.stack([self._preprocess_image(c) for c in crops], axis=0)
        return self._infer_batch(batch)

    def tag(self, image: Image.Image) -> Dict[str, float]:
        """Tag a single image and return a sparse ``{tag_name: confidence}``
        for every tag whose confidence is above zero (ratings excluded, no
        thresholds applied)."""
        probs = self.tag_crops([image])
        return self._decode_row(probs[0])

    def _decode_row(self, probs: np.ndarray) -> Dict[str, float]:
        probs = probs[self.num_rating_tags:]
        names = self.vocab_names
        return {
            names[i].replace("_", " "): float(probs[i])
            for i in range(len(names))
            if probs[i] > 0.0
        }
