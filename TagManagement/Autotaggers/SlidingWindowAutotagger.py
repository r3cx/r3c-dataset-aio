"""
SlidingWindowAutotagger — shared sliding-window autotagger mechanism.

Sits between the generic ``Autotagger`` (workflow) and the concrete
leaves (a single-model regression baseline and the multi-model ensemble):

    Autotagger
      └── SlidingWindowAutotagger
            ├── Eva02SlidingWindowAutotagger   (concrete, single-model)
            └── EnsembleSWAutoTagger           (concrete, multi-model)

It owns everything that is common to every sliding-window autotagger, in part
because it depends on having sliding windows:

    * sliding-window geometry and original-resolution crop generation
    * parallel image / crop loading (DataLoader when workers are configured)
    * the COMPOSITION_TAGS set and their routing through the global scan
    * per-tagger window merging (max across crops; global scan for composition
      and character tags)
    * final tag selection (threshold, kaomoji / undesired filtering,
      character-first ordering)

Concrete leaves only decide *which taggers* are run and *how their per-tagger
results are combined* (a single tagger vs. a sparse sum over many taggers).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from Taggers import Tagger
from .Autotagger import Autotagger, ERROR, FILE_OPEN_ERROR

# -----------------------------------------------------------------------------
# Sliding-window configuration defaults (config can override each value)
# -----------------------------------------------------------------------------
DEFAULT_WINDOW_SIZE_CAP = 0        # 0 = use the image short axis, >0 = hard cap in pixels
DEFAULT_WINDOW_STRIDE_RATIO = 0.9  # 90% stride -> 10% overlap between adjacent windows

# -----------------------------------------------------------------------------
# Composition Tags — Full-image context tags that must use global scan only
# -----------------------------------------------------------------------------
# These tags describe overall image composition, framing, subject count,
# or spatial relationships. Dynamic window crops still lack the full-image
# context needed to detect them reliably, so their probability is taken
# exclusively from the global scan crop (the last crop produced).
#
# Categories:
#   A. Composition / format / perspective
#   B. Subject count — must come from global scan
#   C. Spatial position — full-image relationship
#   D. Focus / framing tags — crops create artificial frames
#   E. Relationship / implied tags — full scene context required
#   F. Censorship — requires full image
#   G. Frame-edge artifacts
#   H. Length-variant features — crops truncate features
#   I. Symmetry / count features — crops break bilateral symmetry
#   J. Clothing / appearance — crops change garment classification
#   K. Character tags — character ID requires full-face + context

COMPOSITION_TAGS = {
    # ========================================================================
    # A. VIEW ANGLE / PERSPECTIVE / COMPOSITION / FORMAT
    # ========================================================================
    "dutch angle", "from above", "from behind", "from below", "from side",
    "multiple views", "sideways", "straight-on", "upside-down",
    "fisheye", "perspective", "vanishing point",
    "afterimage", "border", "inset border", "ornate border",
    "outside border", "rounded corners", "viewfinder",
    "chart", "reference sheet", "stats", "collage", 
    "from outside", "glitch", "isometric", "letterboxed", "pillarboxed",
    "lineup", "column lineup", "faux figurine", "negative space",
    "out of frame", "out-of-frame censoring",
    "partially underwater shot", "pov",
    "symmetry", "rotational symmetry",
    "tachi-e", "zoom layer", "projected inset", "card",
    "1koma", "2koma", "3koma", "4koma", "multiple 4koma", "5koma", 
    "comic", "left-to-right manga", "silent comic","segmented comic",
    "cover", "album cover", "cover page", "doujin cover", "dvd cover", 
    "fake cover", "magazine cover", "manga cover",
    "fake screenshot", "fake phone screenshot",

    # ========================================================================
    # B. SUBJECT COUNT — must come from global scan, not window crops
    # ========================================================================
    # Girls
    "1girl", "multiple girls", "2girls", "3girls", "4girls", "5girls", "6+girls",
    # Boys
    "1boy", "multiple boys", "2boys", "3boys", "4boys", "5boys", "6+boys",
    # Others
    "1other", "multiple others", "2others", "3others", "4others", "6+others",
    # Related
    "solo", "people", "crowd", "ambiguous gender",

    # ========================================================================
    # C. SPATIAL POSITION — full-image relationship, not body-relative pose
    #    Window crops lack full-image context and may misidentify spatial 
    #    relationships between the subject and their environment.
    # ========================================================================
    # On surfaces
    "on bed", "on floor", "on couch", "on chair", "on desk", "on table",
    "on bench", "on stool", "on grass", "on roof", "on ground",
    "on lap", "on person", "on vehicle", "on motorcycle", "on scooter",
    "on railing", "on shoulder", "on head",
    # Under
    "under table", "under tree", "under covers", "under kotatsu",
    # Against
    "against wall", "against glass", "against tree", "against railing",
    "against fourth wall",
    # Behind/in front
    "behind another",
    # In/through/containment
    "in box", "in container", "in tree", "in water", "in bucket", "in food",
    "in cup", "in palm", "in the face",
    "through wall", "through screen", "through medium",
    # Above
    "above clouds",
    # Subject matter
    "everyone", "absolutely everyone",
    "landscape", "nature", "no humans",
    "scenery", "still life",
        
    # ========================================================================
    # D. FOCUS / FRAMING TAGS — window crops create artificial focus or the
    #    lack thereof certain features of the subject.
    # ========================================================================
    "ass focus", "back focus", "breast focus", "eye focus",
    "foot focus", "hand focus", "hip focus",
    "solo focus", "male focus", "pectoral focus", "animal focus", 
    "other focus", "food focus", "vehicle focus", "text focus",
    # Body framing
    "head only", "headless",
    "portrait", "upper body", "cowboy shot", "feet out of frame",
    "full body", "wide shot", "very wide shot",
    "lower body", "head out of frame",
    # Out of frame
    "foot out of frame",
    # Other framing
    "close-up", "cropped legs", "cropped torso", "cropped arms",
    "cropped shoulders", "profile",

    # ========================================================================
    # E. RELATIONSHIP / IMPLIED — full scene context required
    #    Implied actions can be falsely detected from crops of the subjects.
    # ========================================================================
    "yuri", "yaoi",
    "implied sex", "implied yuri", "implied yaoi",
    "implied fellatio", "implied fingering", "implied masturbation",
    "implied kiss", "implied futanari", "implied extra ears",

    # ========================================================================
    # F. CENSORSHIP — requires full image; window crops of uncensored areas
    #    produce false "censored" from local patterns
    #    Consider if only the "censored" tag should be filtered as the other 
    #    tags can potentially be specifically identified from within crops.
    # ========================================================================
    "censored", "censored nipples",
    "mosaic censoring", "bar censor", "blank censor", "blur censor",
    "heart censor", "steam censor", "soap censor", "hair censor",
    "tail censor", "transparent censoring", "character censor",
    "convenient censoring", "pointless censoring",
    "censored text", "novelty censor", "light censor",

    # ========================================================================
    # G. FRAME-EDGE ARTIFACTS — window crop boundaries create false detections
    #    A crop edge cutting through a subject looks like they're "peeking" or
    #    "peeking out" from behind something. Edge-relative tags belong here.
    # ========================================================================
    "peeking", "peeking out", "hiding", "taking cover", "around corner"

    # ========================================================================
    # H. LENGTH-VARIANT FEATURES — crops truncate features, changing apparent
    #    size. Such features may be misidentified leading to multiple lengths
    #    being detected from a single subject if they are long enough to be
    #    cropped into multiple windows. 
    #    - Long hair detected in global scan, Short hair detected in window
    # ========================================================================
    # Hair length
    "long hair", "very long hair", "absurdly long hair", "short hair", "medium hair",
    # Clothing
    "long sleeves", "short sleeves",
    "long skirt","long coat",
    "long dress", "short dress",
    # Body Features
    "huge breasts", "gigantic breasts", "large breasts",
    "medium breasts", "small breasts", "flat chest",
    "huge nipples", "small nipples", "large areolae",
    "huge ass", "thick thighs", "muscular",
    "large penis", "huge penis", "small penis", "large testicles", "huge testicles",
    
    # ========================================================================
    # I. SYMMETRY / COUNT FEATURES — crops break bilateral symmetry detection
    #    A crop showing only one side of the face/body will misidentify
    #    symmetric features as asymmetric, and vice versa.
    # ========================================================================
    # --- BODY: Head/Face ---
     # Hair style — one side cropped → false "side ponytail" from "twintails" etc.
    "side ponytail", "hair bun", "asymmetrical hair",
    "one eye closed", "closed eyes", "one-eyed", "cyclops",
    "single blank eye", "single empty eye", "single eyebrow",
    "single tear", "single blush sticker",
    "single horn", "single antler", "single floppy ear", "single animal ear",
    "single bang", "single sidelock", "single braid", "single drill",
    "single hair bun", "single side bun", "single hair intake",
    "single hair streak", "single hair ring", "single hair tube",
    "single ear cover", "single earring", "single tooth",
    "single head wing", "single feather", "single flame",
    # --- BODY: Torso ---
    "single breast", "single inverted nipple", "single nipple piercing","single pasty",
    "single wing", "single mechanical wing",
    # --- BODY: Arms/Hands ---
    "single extra arm", "single mechanical arm", "single mechanical hand",
    "single bare arm", "single hand",
    # --- BODY: Legs/Feet ---
    "single mechanical leg", "single bare leg", "single bare foot",
    # --- CLOTHING: Arms/Hands ---
    "single bare shoulder", "single off shoulder",
    "asymmetrical sleeves",
    "single sleeve", "single detached sleeve",
    "single wide sleeve", "single sleeve cuff",
    "single sleeve past fingers", "single sleeve past wrist",
    "single glove", "single mechanical glove",
    "single fingerless glove", "single half glove", "single elbow glove",
    "single bridal gauntlet", "single gauntlet", "single mitten",
    "single unworn glove", "single handcuff", "single arm armor", 
    "single arm warmer", "single arm guard", "single arm cuff",
    "single wrist cuff", "single wrist guard", "single elbow pad",
    # --- CLOTHING: Torso ---
    "single pauldron", "single shoulder pad",
    "single epaulette", "single sode", "single vambrace",
    "single bracer", "single couter",
    "single strap", "single garter strap",
    "single suspender", "single stripe",
    "single vertical stripe", "single horizontal stripe",
    # --- CLOTHING: Legs/Feet ---
    "single thighhigh", "single over-kneehigh", "single kneehigh",
    "single hiphigh", "single leg pantyhose", "single fishnet legwear",
    "single leg warmer", "single legwear garter", "single leg bodysuit",
    "single pantsleg", "single detached legging", "single fishnet armwear",
    "single boot", "single armored boot", "single knee boot", "single ankle boot",
    "single thigh boot", "single shoe", "single sandal", "single slipper",
    "single sock", "single loose sock", "single sock removed",
    "single ankle cuff", "single knee pad"

    # ========================================================================
    # J. CLOTHING / APPEARANCE — crops change garment classification
    #    Tags that require full-body context to disambiguate, or whose defining
    #    features may be split across crops (garment extent, connected pieces,
    #    full-outfit classification).
    # ========================================================================
    # Garment classification ambiguity — crop can't tell dress from top+skirt,
    # or crop top from dress, without seeing where the garment ends.
    "dress", "crop top",
    # Bottom garments — crop of upper portion may confuse skirt vs dress bottom,
    # or pants vs shorts (distinction is purely hem length, which crops truncate).
    "skirt", "pants", "shorts",
    # Full-body connected garments — need to see torso+bottom connection to
    # distinguish from separate pieces.
    "swimsuit",
    "bikini", "one-piece swimsuit",
    "leotard", "bodysuit",
    "bodystocking",
    # Outfit classification — requires seeing the full ensemble to identify.
    "uniform", "school uniform",
    # Clothing state — crop may show a torn area that's a design detail,
    # or miss the tear entirely; global view confirms intent.
    "torn clothes",
    "partially unbuttoned", "partially unzipped",
    "asymmetrical clothes", "asymmetrical dress", "asymmetrical skirt",
    # Body coverage — requires seeing the full body to confirm what's absent.
    "nude", "partially undressed",
    "topless female", "topless male", "topless other",
    "bottomless",
    "no shirt", "no pants",
    # Bilateral coverage — crop may only show one side.
    "bare shoulders", "barefoot",

    # ========================================================================
    # K. CHARACTER TAGS — character identification requires full-face + context
    # ========================================================================
}


# -----------------------------------------------------------------------------
# Sliding-window geometry — operates on the original-resolution PIL image.
# -----------------------------------------------------------------------------
# The windowing itself never resizes: windows are cut at original resolution so
# every tagger decides its own input size. When a tagger declares a
# ``crop_resize_spec``, the *resize to model input size* additionally happens
# inside the DataLoader workers (see :func:`resize_crop_to_input`) so the main
# process — the one that also runs inference — never spends its single core on
# resizing multi-megapixel crops; the tagger's own ``_preprocess_image`` then
# only does the cheap float conversion / normalization.


def compute_window_size(h: int, w: int, cap: int) -> int:
    """Window size from the image short axis, optionally hard-capped."""
    short_axis = min(h, w)
    if cap and cap > 0:
        return min(short_axis, cap)
    return short_axis


def generate_window_coords(h: int, w: int, window_size: int, stride_ratio: float):
    """Return a list of (y, x) top-left coordinates for a sliding-window grid.

    A final row/column is always added to guarantee full edge coverage.
    """
    stride = max(1, int(window_size * stride_ratio))

    def _axis_positions(length: int):
        positions = []
        pos = 0
        while pos + window_size <= length:
            positions.append(pos)
            pos += stride
        if not positions or positions[-1] + window_size < length:
            positions.append(max(0, length - window_size))
        return positions

    rows = _axis_positions(h)
    cols = _axis_positions(w)
    return [(y, x) for y in rows for x in cols]


def extract_window_crop(image: Image.Image, y: int, x: int, window_size: int) -> Image.Image:
    """Crop an original-resolution region from ``image`` (clamped to bounds).

    Returns a PIL sub-image at original resolution. Resizing to a model input
    is the responsibility of the Tagger.
    """
    w, h = image.size
    y1, y2 = y, min(y + window_size, h)
    x1, x2 = x, min(x + window_size, w)
    return image.crop((x1, y1, x2, y2))


def generate_crops(
    image: Image.Image,
    window_size_cap: int,
    window_stride_ratio: float,
) -> List[Image.Image]:
    """Generate a list of original-resolution crops for a single image.

    Window crops are produced only when the image is larger than the computed
    window size. In every case a full-image global scan is appended as the last
    crop, so the caller can always find the global scan at index
    ``len(crops) - 1``.
    """
    w, h = image.size
    window_size = compute_window_size(h, w, window_size_cap)

    if h > window_size or w > window_size:
        coords = generate_window_coords(h, w, window_size, window_stride_ratio)
        crops = [extract_window_crop(image, y, x, window_size) for (y, x) in coords]
    else:
        # Image fits within one window — behave like the reference (single full
        # crop) so the regression baseline stays equivalent.
        crops = [image]

    # Global scan is always the last crop.
    crops.append(image)
    return crops


# -----------------------------------------------------------------------------
# Worker-side crop presize (mirrors each tagger's _preprocess_image geometry)
# -----------------------------------------------------------------------------
def resize_crop_to_input(crop: Image.Image, input_size: int, resizer: str) -> Image.Image:
    """White pad-to-square + resize one crop to the model input size, using the
    tagger's own resize path so the result is bit-identical to running the
    tagger's full ``_preprocess_image`` on the original-resolution crop:

    * ``"cv2"`` — WD1.4 ONNX path: INTER_AREA when shrinking, INTER_LANCZOS4
      when expanding (resize is per-channel, so the RGB/BGR ordering is
      irrelevant here — the tagger re-applies its own channel handling).
    * ``"pil_bicubic"`` — dbv4 PyTorch path: ``Image.BICUBIC``.

    Runs inside the DataLoader workers (parallel, off the inference thread).
    """
    side = max(crop.size)
    if crop.size != (side, side):
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(crop, ((side - crop.size[0]) // 2, (side - crop.size[1]) // 2))
        crop = canvas
    if resizer == "cv2":
        arr = np.array(crop)[:, :, ::-1]  # RGB -> BGR
        interp = cv2.INTER_AREA if side > input_size else cv2.INTER_LANCZOS4
        arr = cv2.resize(arr, (input_size, input_size), interpolation=interp)
        return Image.fromarray(arr[:, :, ::-1])
    return crop.resize((input_size, input_size), Image.BICUBIC)


def _maybe_presize_crops(
    crops: List[Image.Image], resize_spec: Optional[Tuple[int, str]]
) -> List[Image.Image]:
    """Apply ``crop_resize_spec`` to a crop list (no-op when the spec is None)."""
    if resize_spec is None:
        return crops
    input_size, resizer = resize_spec
    return [resize_crop_to_input(c, input_size, resizer) for c in crops]


# -----------------------------------------------------------------------------
# Data loader — multi-worker crop generation (+ optional worker-side presize)
# -----------------------------------------------------------------------------
class _CropGenerationDataset(torch.utils.data.Dataset):
    """Worker-safe dataset that loads an image, produces its window crops, and
    (when a ``resize_spec`` is given) presizes each crop to the model input
    size — all inside the worker. Only holds picklable primitives so it can be
    fanned out across DataLoader workers."""

    def __init__(
        self,
        image_paths,
        window_size_cap: int,
        window_stride_ratio: float,
        resize_spec: Optional[Tuple[int, str]] = None,
    ):
        self.image_paths = image_paths
        self.window_size_cap = window_size_cap
        self.window_stride_ratio = window_stride_ratio
        self.resize_spec = resize_spec

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = str(self.image_paths[idx])
        try:
            image = Image.open(path).convert("RGB")
            crops = generate_crops(image, self.window_size_cap, self.window_stride_ratio)
            crops = _maybe_presize_crops(crops, self.resize_spec)
        except Exception as e:
            print(f"{FILE_OPEN_ERROR}{path}, {ERROR}{e}")
            return None
        return (crops, path)


def _discard_none_in_batch(batch):
    """Collate function: drop entries whose image failed to load."""
    return [item for item in batch if item is not None]


# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class SlidingWindowAutotagger(Autotagger):
    """Shared sliding-window autotagger mechanism.

    Provides crop generation, composition routing, per-tagger window merging,
    and final tag selection. Concrete leaves implement ``_process_images`` to
    decide which taggers run and how their per-tagger results are combined.
    Subclasses may override ``composition_tags`` to change which tags are
    routed through the global scan.
    """

    #: Composition tags are routed through the global scan only. Subclasses may
    #: override this set.
    composition_tags = COMPOSITION_TAGS

    # ------------------------------------------------------------------
    # Sliding-window configuration
    # ------------------------------------------------------------------
    @property
    def window_size_cap(self) -> int:
        return int(self._cfg("window_size_cap", DEFAULT_WINDOW_SIZE_CAP))

    @property
    def window_stride_ratio(self) -> float:
        return float(self._cfg("window_stride_ratio", DEFAULT_WINDOW_STRIDE_RATIO))

    @property
    def use_sequential(self) -> bool:
        """Infer crops one at a time (lower peak memory) instead of as a single
        batch of all crops. Batched is the default; it matches the reference's
        single-batch ``run_inference_batched``."""
        return bool(self._cfg("use_sequential", False))

    # ------------------------------------------------------------------
    # Crop generation / loading
    # ------------------------------------------------------------------
    def generate_crops(self, image: Image.Image) -> List[Image.Image]:
        """Original-resolution crops for a loaded image (global scan last)."""
        return generate_crops(image, self.window_size_cap, self.window_stride_ratio)

    def iter_image_crops(self, image_paths, resize_spec: Optional[Tuple[int, str]] = None):
        """Yield ``(crops, image_path)`` pairs for every image.

        Uses a multi-worker DataLoader when ``num_workers`` is set; otherwise
        falls back to a single-threaded pass (the same crop presize still
        applies, on the main thread). Corrupt / unreadable images are skipped.

        ``resize_spec`` (a tagger's ``crop_resize_spec``) moves the resize to
        model input size into the workers, so the main process receives small
        crops instead of multi-megapixel originals.
        """
        if self.num_workers:
            dataset = _CropGenerationDataset(
                image_paths,
                self.window_size_cap,
                self.window_stride_ratio,
                resize_spec=resize_spec,
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=_discard_none_in_batch,
                drop_last=False,
            )
            for batch in loader:
                for item in batch:
                    yield item
            return

        for path in image_paths:
            path = str(path)
            try:
                image = Image.open(path).convert("RGB")
                crops = _maybe_presize_crops(self.generate_crops(image), resize_spec)
            except Exception as e:
                print(f"{FILE_OPEN_ERROR}{path}, {ERROR}{e}")
                continue
            yield (crops, path)

    # ------------------------------------------------------------------
    # Composition routing
    # ------------------------------------------------------------------
    def composition_tag_names(self) -> set:
        """composition_tags expressed as underscore names, matching names as
        stored in a tagger vocabulary."""
        return {t.replace(" ", "_") for t in self.composition_tags}

    def _use_global_mask(self, tagger: Tagger) -> np.ndarray:
        """Boolean mask, aligned with ``tagger.vocab_names`` order (general tags
        then character tags), indicating which tag indices must use the global
        scan probability instead of the max-across-crops value:

        * every character tag (by definition), and
        * any general tag listed in ``composition_tags``.
        """
        general = tagger.general_tags
        character = tagger.character_tags
        mask = np.zeros(len(general) + len(character), dtype=bool)
        comp = self.composition_tag_names()
        for i, tag in enumerate(general):
            if tag in comp:
                mask[i] = True
        mask[len(general):] = True
        return mask

    # ------------------------------------------------------------------
    # Per-tagger window merging
    # ------------------------------------------------------------------
    def merge_crop_probabilities(self, tagger: Tagger, prob_matrix: np.ndarray) -> np.ndarray:
        """Merge a tagger's dense per-crop probabilities into a single per-tag
        vector for one image.

        ``prob_matrix`` is the raw output of ``tagger.tag_crops`` with shape
        ``(num_crops, num_outputs)``, where ``num_outputs`` includes the leading
        rating rows. The global scan is the last crop.

        Returns a 1-D vector of shape ``(num_tags,)`` aligned with
        ``tagger.vocab_names`` where:
            * normal (general) tags use the max over all crops, and
            * composition and character tags use the global scan value.

        Rating outputs are stripped.
        """
        probs = np.asarray(prob_matrix)[:, tagger.num_rating_tags:]
        global_idx = probs.shape[0] - 1

        merged = np.max(probs, axis=0)
        global_probs = probs[global_idx]

        mask = self._use_global_mask(tagger)
        merged = merged.copy()
        merged[mask] = global_probs[mask]
        return merged

    def _tag_crops(self, tagger: Tagger, crops: List[Image.Image]) -> np.ndarray:
        """Run ``tagger`` over an image's crops and return the dense per-crop
        probability matrix of shape ``(num_crops, num_outputs)``.

        By default the crops are inferred in a single batch. Under
        :attr:`use_sequential` each crop is inferred on its own (lower memory)
        and the results are stacked. Either way the result is the same
        per-crop matrix, so the merge step applies identically afterwards.
        """
        if self.use_sequential:
            return np.vstack([tagger.tag_crops([c]) for c in crops])
        return tagger.tag_crops(list(crops))

    def tag_and_merge(self, tagger: Tagger, crops: List[Image.Image]) -> np.ndarray:
        """Run ``tagger`` over an image's crops and merge the results into a
        single per-tag vector (see :meth:`merge_crop_probabilities`)."""
        prob_matrix = self._tag_crops(tagger, crops)
        return self.merge_crop_probabilities(tagger, prob_matrix)

    # ------------------------------------------------------------------
    # Sparse results (for cross-tagger accumulation in the ensemble)
    # ------------------------------------------------------------------
    def to_sparse(self, tagger: Tagger, merged: np.ndarray):
        """Convert a merged per-tag vector into a sparse human-readable dict.

        Returns ``{tag_name: confidence}`` for every tag with confidence > 0,
        where ``tag_name`` is the space-separated form.
        """
        names = tagger.vocab_names
        return {
            names[i].replace("_", " "): float(merged[i])
            for i in range(len(names))
            if merged[i] > 0.0
        }

    # ------------------------------------------------------------------
    # Final tag selection
    # ------------------------------------------------------------------
    def select_tags(self, tagger: Tagger, merged: np.ndarray):
        """Select tags from a merged per-tag vector aligned with
        ``tagger.vocab_names``.

        General tags are selected at :attr:`Autotagger.general_threshold` and
        character tags at :attr:`Autotagger.character_threshold`. Kaomoji only
        affects general tags; the user-specified ``undesired`` set affects
        both. Character tags are placed at the front of the combined list.

        Returns ``(combined, general_text, character_text, freq_updates)`` where
        ``combined`` is the ordered, human-readable tag list to write to the
        .txt file.
        """
        general = tagger.general_tags
        character = tagger.character_tags
        num_general = len(general)
        combined = []
        gen_parts = []
        char_parts = []
        freq_updates = {}

        for i, p in enumerate(merged):
            if i < num_general:
                if p >= self.general_threshold:
                    raw_name = general[i]
                    if self.is_kaomoji(raw_name):
                        continue
                    name = raw_name.replace("_", " ")
                    if self.is_excluded(name):
                        continue
                    freq_updates[name] = freq_updates.get(name, 0) + 1
                    gen_parts.append(name)
                    combined.append(name)
            else:
                if p >= self.character_threshold:
                    name = character[i - num_general].replace("_", " ")
                    if self.is_excluded(name):
                        continue
                    freq_updates[name] = freq_updates.get(name, 0) + 1
                    char_parts.append(name)
                    combined.insert(0, name)

        return combined, ", ".join(gen_parts), ", ".join(char_parts), freq_updates

    def select_sparse_tags(self, sparse, character_names):
        """Select tags from an already-averaged sparse ``{tag_name: confidence}``
        mapping (the ensemble's final selection).

        Mirrors :meth:`select_tags` exactly, but on *space-form* names:

        * a tag is a **character** iff its name is in ``character_names`` (the
          union of the member taggers' character sets) — thresholded at
          :attr:`Autotagger.character_threshold`;
        * every other tag is **general** — thresholded at
          :attr:`Autotagger.general_threshold`, with the kaomoji filter applied
          through the underscore form (``DEFAULT_KAOMOJIS`` is underscore-form
          while the sparse names are space-form);
        * the ``undesired`` set applies to both;
        * character tags are placed at the front via ``insert(0, name)``,
          reproducing the baseline's reverse-encounter character order and the
          characters-before-general ordering — so for a single tagger whose
          sparse dict is in vocabulary order the output is byte-identical to
          :meth:`select_tags`.

        Returns ``(combined, gen_text, char_text, freq_updates)``.
        """
        combined = []
        gen_parts = []
        char_parts = []
        freq_updates = {}

        for name, conf in sparse.items():
            if name in character_names:
                if conf < self.character_threshold:
                    continue
                if self.is_excluded(name):
                    continue
                freq_updates[name] = freq_updates.get(name, 0) + 1
                char_parts.append(name)
                combined.insert(0, name)
            else:
                if conf < self.general_threshold:
                    continue
                if self.is_kaomoji(name.replace(" ", "_")):
                    continue
                if self.is_excluded(name):
                    continue
                freq_updates[name] = freq_updates.get(name, 0) + 1
                gen_parts.append(name)
                combined.append(name)

        return combined, ", ".join(gen_parts), ", ".join(char_parts), freq_updates

    def debug_selection(self, tagger: Tagger, merged: np.ndarray) -> None:
        """Print a per-tag breakdown of the tags meeting threshold, marking
        whether each came from the global scan or the max-across-crops merge."""
        general = tagger.general_tags
        character = tagger.character_tags
        num_general = len(general)
        use_global = self._use_global_mask(tagger)

        rows = []
        for i, p in enumerate(merged):
            use_g = bool(use_global[i])
            is_char = i >= num_general
            threshold = self.character_threshold if is_char else self.general_threshold
            if p < threshold:
                continue
            raw = character[i - num_general] if is_char else general[i]
            src = "global" if use_g else "max"
            rows.append((raw.replace("_", " "), p, threshold, src))

        rows.sort(key=lambda x: x[1], reverse=True)
        print(f"  Tags meeting threshold:")
        for tag, prob, threshold, src in rows:
            print(f"    {tag:40s} {prob:.4f} (threshold: {threshold}, source: {src})")
