"""
Eva02SlidingWindowAutotagger — concrete single-model sliding-window autotagger.

This is the known-good regression baseline. It runs exactly ONE tagger — the
WD1.4 ``wd-eva02-large-tagger-v3`` (EVA02-Large) — over the sliding windows of
each image, merges the per-crop probabilities, selects tags, and writes the .txt
output beside every image.

It inherits the whole sliding-window mechanism (crop generation, composition
routing, per-crop max/global merge, final selection) from
:class:`SlidingWindowAutotagger`. A concrete leaf only decides *which taggers*
run and *how their per-tagger results are combined*. Here there is a single
tagger whose merged vector is thresholded directly, so there is no cross-tagger
combination step — the per-image pipeline is simply::

    crops ──► tagger.tag_crops ──► merge ──► select ──► .txt

This keeps the baseline an isolated leaf, fully reusing the shared
sliding-window machinery.

Runs directly or as a module (from ``TagManagement/``):
    python Autotaggers/Eva02SlidingWindowAutotagger.py --data_dir <dir> --use_gpu
    python -m Autotaggers.Eva02SlidingWindowAutotagger --data_dir <dir>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Script bootstrap: running this file as a plain script (not a package module)
# leaves it unattached to a package, so make the ``TagManagement`` root (this
# file's grandparent) importable before the package imports below. When run as
# a module (``python -m Autotaggers.Eva02SlidingWindowAutotagger``) ``__package__``
# is set and nothing here runs.
# ---------------------------------------------------------------------------
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Autotaggers import SlidingWindowAutotagger  # noqa: E402
from Taggers import Eva02Tagger  # noqa: E402


class Eva02SlidingWindowAutotagger(SlidingWindowAutotagger):
    """Single-model (EVA02-Large WD1.4) sliding-window autotagger.

    The regression baseline. Drives one :class:`Eva02Tagger`; for every image
    the tagger is run over the image's crops, the per-crop probabilities are
    merged (max for normal tags, global scan for composition/character tags),
    tags are selected, and the result is written to the .txt file beside the
    image.
    """

    def _make_tagger(self) -> Eva02Tagger:
        """Create the one tagger this leaf runs (WD1.4 EVA02-Large)."""
        return Eva02Tagger(
            use_gpu=self.use_gpu,
            use_bf16=self.use_bf16,
            force_download=self.force_download,
            hf_token=self._cfg("hf_token"),
            download=not self._cfg("no_download", False),
            verbose=self.verbose,
        )

    def _process_images(self, image_paths) -> None:
        tagger = self._make_tagger()

        # Load once for the whole pass (visible load cost), process all, done:
        # exactly the reference's load-once, process-all, done lifecycle.
        t_start = time.time()
        tagger.load()
        print(f"Model loaded in {time.time() - t_start:.1f}s.")

        with tagger:
            for crops, path in self.progress(
                self.iter_image_crops(image_paths, tagger.crop_resize_spec),
                desc="tagging",
            ):
                merged = self.tag_and_merge(tagger, crops)
                combined, gen_text, char_text, freq = self.select_tags(tagger, merged)

                if self.debug_print:
                    print(f"\n[{Path(path).name}]")
                    self.debug_selection(tagger, merged)
                    print(
                        f"\n{path}:\n"
                        f"  Character tags: {char_text}\n"
                        f"  General tags: {gen_text}"
                    )

                tag_text = ", ".join(combined)
                self.write_tags(path, tag_text)
                self.update_frequencies(freq)


if __name__ == "__main__":
    Eva02SlidingWindowAutotagger.from_cli().run()
