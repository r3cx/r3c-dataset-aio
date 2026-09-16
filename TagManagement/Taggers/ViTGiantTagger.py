"""
ViTGiantTagger — Danbooru-v4 ViT-GiantOpt/SigLIP tagger (animetimm).

Thin concrete :class:`Dbv4Tagger` for
``animetimm/vit_giantopt_patch16_siglip_384.dbv4-full``. All behaviour comes
from :class:`Dbv4Tagger`; only the identity and files are declared here (the
input size — 512, NOT the 384 implied by the repo name — is sourced from
preprocess.json).
"""

from __future__ import annotations

from .Dbv4Tagger import Dbv4Tagger


class ViTGiantTagger(Dbv4Tagger):
    """ViT-GiantOpt / SigLIP Danbooru-v4 tagger (animetimm).

    Binds one HuggingFace model to the shared :class:`Dbv4Tagger`. This class
    declares **only** :attr:`model_id`; from it the base derives the local
    folder (``Models/vit_giantopt_patch16_siglip_384.dbv4-full``), the file
    paths, and the download source. It overrides **no** stage below
    ``model_id`` -- every step of tagging is inherited from :class:`Dbv4Tagger`.

    Per-model parameters -- read from the model's own ``preprocess.json`` by
    :meth:`Dbv4Tagger._ensure_metadata` at load, **not** hardcoded here:

    * input side    : 512 (the ``test`` ``resize`` step) -- the repo name says
      "384", but the *input* is 512; it is never taken from the name
    * normalization : SigLIP mean/std, all 0.5 ([0.5, 0.5, 0.5] / [0.5, 0.5, 0.5])
      -- this is the one v4 model that does **not** use ImageNet values

    That is why there is no ``default_input_size`` and no preprocessing in this
    class: those values are the model's own, sourced from its metadata file.

    Pipeline (each step is the **base** doing the work; this class does none):

    1. **Files** -- :meth:`Tagger.ensure_files` fetches the base's
       ``model_files`` from ``model_id`` into ``Models/<name>`` only if
       missing, then :meth:`Dbv4Tagger._load` runs.
    2. **Load + metadata** -- :meth:`Dbv4Tagger._load` builds a transformers
       ``AutoModelForImageClassification`` from the local dir (it wraps the
       timm ``vit_giantopt_patch16_siglip_384`` backbone) and
       :meth:`Dbv4Tagger._ensure_metadata` parses ``preprocess.json`` (512 +
       SigLIP 0.5), ``categories.json`` (id->name), and ``selected_tags.csv``
       (12476 rows -> 9225 general / 3247 character, ratings dropped).
    3. **Preprocess** -- :meth:`Dbv4Tagger._preprocess_image`: a *full padded
       pass* (the model's own ``test`` transform): white pad-to-square (nothing
       cropped) -> bicubic resize to 512 -> /255 + SigLIP(0.5) -> CHW float32.
    4. **Inference** -- :meth:`Dbv4Tagger._infer_batch`: forward -> sigmoid ->
       reorder to the dense ``[general..., character...]`` contract
       (``num_rating_tags = 0``).
    5. **Decode** -- :meth:`Dbv4Tagger.tag_crops` returns that dense matrix;
       :meth:`Dbv4Tagger.tag` returns ``{tag: confidence}`` for every value > 0.
       **No thresholds here** -- thresholds are applied downstream (Autotagger
       / Ensemble).

    A model that needs a *different* pipeline gets an override on its **own**
    thin class (preprocess / decode / files), not on the base.
    """

    model_id = "animetimm/vit_giantopt_patch16_siglip_384.dbv4-full"
