"""
EnsembleSWAutoTagger — multi-model sliding-window ensemble autotagger.

Runs several :class:`Taggers` (a single WD14 and the three Danbooru-v4 models)
over the same sliding windows of each image and combines their per-image results
by **equal-weight sparse average** (the ensemble is a :class:`Ensembles.
EnsembleTagger`, *not* a ``Tagger``).

Execution strategy for a 24 GB RTX 3090 is **tagger-major**: one
tagger is loaded at a time, it processes the *entire* dataset (all images; per
image a single batch of all crops + the global scan, with ``--use_sequential``
as the low-memory fallback), its per-image sparse results are accumulated into a
sparse sum per image, the tagger is released, and the next tagger loads. A final
in-RAM pass divides by the number of taggers, selects tags (thresholds are
applied **after** the ensemble), and writes each ``.txt`` once.

Crucial ordering: window merge → per-tagger result → ensemble →
threshold → filter → output — never threshold the raw crops before the ensemble.

Runs directly or as a module (from ``TagManagement/``):
    python Autotaggers/EnsembleSWAutoTagger.py --data_dir <dir> --use_gpu
    python Autotaggers/EnsembleSWAutoTagger.py --data_dir <dir> --taggers eva02,eva_giant
    python -m Autotaggers.EnsembleSWAutoTagger --data_dir <dir> --taggers eva02
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Script bootstrap: running this file as a plain script leaves it unattached to
# a package, so make the ``TagManagement`` root (this file's grandparent)
# importable before the package imports below. No-op in module mode.
# ---------------------------------------------------------------------------
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Autotaggers import SlidingWindowAutotagger  # noqa: E402
from Ensembles import EnsembleTagger  # noqa: E402
from Taggers import (  # noqa: E402
    ConvNeXtV2Tagger,
    Eva02Tagger,
    EvaGiantTagger,
    ViTGiantTagger,
)


class EnsembleSWAutoTagger(SlidingWindowAutotagger):
    """Multi-model sliding-window ensemble autotagger.

    Drives several taggers one at a time (tagger-major) over every image's
    crops, accumulates per-image sparse sums, and, in a final pass, averages by
    the number of taggers, selects tags (thresholds post-ensemble), and writes
    the ``.txt`` once per image.
    """

    #: Short name -> tagger class. The ensemble instantiates only the names in
    #: :attr:`tagger_names`.
    REGISTRY = {
        "eva02": Eva02Tagger,
        "eva_giant": EvaGiantTagger,
        "vit_giant": ViTGiantTagger,
        "convnextv2": ConvNeXtV2Tagger,
    }

    #: Default tagger list (all four) when ``--taggers`` is not given.
    DEFAULT_TAGGERS = ["eva02", "eva_giant", "vit_giant", "convnextv2"]

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------
    @classmethod
    def _add_cli_args(cls, parser):
        parser.add_argument(
            "--taggers",
            type=str,
            default=None,
            help=(
                "Comma list of taggers to run: "
                + ",".join(cls.REGISTRY)
                + " (default: all four)"
            ),
        )
        parser.add_argument(
            "--combine",
            type=str,
            default="taggers",
            choices=["mean", "taggers"],
            help=(
                "How to average the per-tagger confidences. 'mean' divides every tag "
                "by the number of taggers (N). 'taggers' (default) divides each tag by "
                "the number of taggers whose vocabulary contains it, so a structurally "
                "absent tag is an absence, not a 0 that dilutes the score."
            ),
        )
        parser.add_argument(
            "--debug_taggers",
            action="store_true",
            help=(
                "Print each tagger's merged per-image confidence (pre-ensemble) "
                "in a per-tagger column table, so you can see which model "
                "contributed each tag. High-memory (keeps one sparse dict per "
                "image per tagger); intended for inspecting a few images."
            ),
        )
        parser.add_argument(
            "--debug_taggers_top",
            type=int,
            default=30,
            help="With --debug_taggers: how many top-by-confidence tags to list "
            "per image (every tag selected into the output is always listed too).",
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @property
    def tagger_names(self):
        """Ordered short-names of the taggers to run (``--taggers`` or
        ``ensemble_taggers``; default all four, order preserved)."""
        names = self._cfg("taggers")
        if names is None:
            names = self._cfg("ensemble_taggers")
        if names is None:
            return list(self.DEFAULT_TAGGERS)
        if isinstance(names, str):
            names = [n.strip() for n in names.split(",") if n.strip()]
        return list(names)

    @property
    def combine(self) -> str:
        """Ensemble combine mode: ``"mean"`` (plain /N) or ``"taggers"`` (vocabulary-
        aware, default). Resolved from ``--combine`` (falls back to ``"taggers"``)."""
        return str(self._cfg("combine", "taggers") or "taggers")

    def _make_tagger(self, name: str):
        """Instantiate one tagger by short name with the run configuration."""
        cls = self.REGISTRY[name]
        return cls(
            use_gpu=self.use_gpu,
            use_bf16=self.use_bf16,
            force_download=self.force_download,
            hf_token=self._cfg("hf_token"),
            download=not self._cfg("no_download", False),
            verbose=self.verbose,
        )

    # ------------------------------------------------------------------
    # Vocabulary-aware (tagger-aware) combine support
    # ------------------------------------------------------------------
    def _build_tagger_map(self, names):
        """Map each space-form tag to the set of tagger indices (positions in
        ``names``) whose ``selected_tags.csv`` lists it — the union of the members'
        vocabularies. This is the precompute that lets a tag divide only by the
        taggers that can actually emit it (vocabulary-aware normalization).

        Reading every ``name`` column (not just the general/character rows) is a
        harmless superset of what ``to_sparse`` can emit for a tagger: the extra
        non-emit-able rows (e.g. rating) are never referenced in the per-image sums,
        and every emit-able tag is guaranteed present in its tagger's set. No tagger
        is loaded (side-effect-free construction + reading a text file only), so this
        runs before any inference.
        """
        tag_taggers = {}
        for idx, name in enumerate(names):
            csv_path = self.REGISTRY[name]().model_dir / "selected_tags.csv"
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                try:
                    name_col = header.index("name")
                except ValueError:
                    raise AssertionError(
                        f"{csv_path} has no 'name' column: {header}"
                    ) from None
                for row in reader:
                    if not row or name_col >= len(row) or not row[name_col].strip():
                        continue
                    tag = row[name_col].strip().replace("_", " ")
                    tag_taggers.setdefault(tag, set()).add(idx)
        return tag_taggers

    @staticmethod
    def _average(sums, tag_taggers, n):
        """Divide a per-image running *sums* dict into an average: plain ``/ n`` in
        mean mode, or ``len(tag_taggers[tag])`` per tag (never 0) in tagger-aware
        mode. Mirrors :meth:`Ensembles.EnsembleTagger.combine` for the accumulated
        (already summed) form."""
        if tag_taggers is None:
            return {tag: s / n for tag, s in sums.items()}
        return {tag: s / max(1, len(tag_taggers.get(tag, range(n)))) for tag, s in sums.items()}

    # ------------------------------------------------------------------
    # Core loop (tagger-major)
    # ------------------------------------------------------------------
    def _process_images(self, image_paths) -> None:
        names = self.tagger_names
        unknown = [n for n in names if n not in self.REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown tagger name(s) {unknown}. Valid names: {list(self.REGISTRY)}."
            )

        n = len(names)
        t_min = min(self.general_threshold, self.character_threshold)
        debug_taggers = bool(self._cfg("debug_taggers", False))

        # Vocabulary-aware combine: precompute, from each member's CSV, the tag ->
        # set-of-tagger-indices map. None in mean mode (plain /N, unchanged path).
        combine_mode = self.combine
        tag_taggers = self._build_tagger_map(names) if combine_mode == "taggers" else None
        if combine_mode == "taggers":
            if self.verbose:
                print(
                    f"[ensemble] combine=taggers: {len(tag_taggers)} tags across "
                    f"{n} tagger(s) (a tag divides by the taggers whose vocab has it)."
                )

        # path(str) -> {tag: running sum}; insertion order == first-encounter
        # (original) image order.
        accumulators = {}
        union_chars = set()  # character tag names (space form), across taggers
        # path(str) -> tagger_name -> merged sparse dict (debug_taggers only).
        per_tagger = {} if debug_taggers else None

        for name in names:
            tagger = self._make_tagger(name)

            # Load out-of-band so the load cost (weights from disk + GPU copy)
            # is visible and distinguishable from the image pass.
            t_load = time.time()
            tagger.load()
            load_s = time.time() - t_load

            with tagger:
                t_pass = time.time()
                # The tagger declares how workers should presize crops (resize to
                # model input size inside the workers) so the inference thread
                # never spends its core on multi-megapixel resizes.
                resize_spec = tagger.crop_resize_spec
                n_images = 0
                for crops, path in self.progress(
                    self.iter_image_crops(image_paths, resize_spec), desc=name
                ):
                    n_images += 1
                    merged = self.tag_and_merge(tagger, crops)
                    sp = self.to_sparse(tagger, merged)
                    key = str(path)
                    acc = accumulators.setdefault(key, {})
                    for tag, conf in sp.items():
                        acc[tag] = acc.get(tag, 0.0) + conf
                    if per_tagger is not None:
                        per_tagger.setdefault(key, {})[name] = sp
                # Characters are global per model; their union is the
                # character set for final selection.
                union_chars |= {t.replace("_", " ") for t in tagger.character_tags}
                pass_s = time.time() - t_pass

            # Drop tags that can no longer reach the (lowest) threshold even if
            # every remaining (member) tagger added a perfect 1.0. In
            # tagger-aware mode the bound uses each tag's own member count.
            done = names.index(name) + 1
            for acc in accumulators.values():
                EnsembleTagger.prune_accumulator(acc, done, n, t_min, tag_taggers)
            print(
                f"[ensemble] tagger '{name}' done ({done}/{n}): "
                f"load {load_s:.1f}s, {n_images} image(s) in {pass_s:.1f}s."
            )

        # Final pass: divide (by N in mean mode, or by each tag's tagger count in
        # tagger-aware mode), select (thresholds post-ensemble), write once.
        for path, sums in accumulators.items():
            avg = self._average(sums, tag_taggers, n)
            combined, gen_text, char_text, freq = self.select_sparse_tags(avg, union_chars)

            if self.debug_print:
                print(f"\n[{Path(path).name}]")
                self.debug_ensemble(avg, union_chars)
                print(
                    f"\n{path}:\n"
                    f"  Character tags: {char_text}\n"
                    f"  General tags: {gen_text}"
                )

            if debug_taggers:
                self._debug_tagger_table(
                    path,
                    names,
                    per_tagger.get(str(path), {}),
                    avg,
                    set(combined),
                    top=int(self._cfg("debug_taggers_top", 30)),
                )

            self.write_tags(path, ", ".join(combined))
            self.update_frequencies(freq)

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------
    def debug_ensemble(self, avg, character_names, top: int = 30) -> None:
        """Print the top ensemble tags (post-merge, pre-threshold) with their
        averaged confidence."""
        n = len(self.tagger_names)
        rows = sorted(avg.items(), key=lambda kv: kv[1], reverse=True)[:top]
        denom = "N model(s)" if self.combine == "mean" else "tagger(s) that emit each tag"
        print(f"  Top {len(rows)} ensemble tags (avg over {denom}):")
        for tag, conf in rows:
            kind = "character" if tag in character_names else "general"
            print(f"    {tag:40s} {conf:.4f}  [{kind}]")

    def _debug_tagger_table(self, path, names, per, avg, selected, top: int = 30) -> None:
        """Print each tagger's merged (pre-ensemble) confidence for one image.

        ``per`` maps tagger_name -> {tag: merged_conf}. The table lists the
        top-``top`` tags by any-tagger confidence plus every tag selected into
        the output; each row shows the per-tagger confidence and the ensemble
        average (sum/N, missing = 0.0). ``*`` marks a tag selected into the output.
        """
        # Strongest single-tagger confidence per tag, used for ordering + focus.
        ranked = {}
        for sp in per.values():
            for tag, conf in sp.items():
                if conf > ranked.get(tag, 0.0):
                    ranked[tag] = conf
        focus = {t for t, _ in sorted(ranked.items(), key=lambda kv: kv[1], reverse=True)[:top]}
        focus |= selected
        order = sorted(focus, key=lambda t: (ranked.get(t, 0.0), avg.get(t, 0.0)), reverse=True)

        col = 11
        print(f"\n[per-tagger confidences] {Path(path).name}  (N={len(names)})")
        print(
            "    "
            + f"{'tag':<36} "
            + " ".join(f"{nm:>{col}}" for nm in names)
            + f"  {'avg':>5}"
        )
        for tag in order:
            cells = " ".join(f"{per.get(nm, {}).get(tag, 0.0):>{col}.3f}" for nm in names)
            mark = "*" if tag in selected else " "
            print(f"    {tag[:35]:<36} {cells}  {avg.get(tag, 0.0):5.2f} {mark}")
        print("    (* = selected into output; conf = window-merged, pre-ensemble)")


if __name__ == "__main__":
    EnsembleSWAutoTagger.from_cli().run()
