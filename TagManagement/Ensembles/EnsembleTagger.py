"""
EnsembleTagger — pure, model-free ensemble combination logic.

Deliberately NOT a :class:`Taggers.Tagger` (lives in ``Ensembles/`` per the
project layout): no model loading, no preprocessing, no inference. It operates
only on the sparse ``{tag_name: confidence}`` dicts that each individual tagger
produces (after its own sliding-window merge — the window merge always
happens *before* the ensemble).

All tags are already decoded to their space-separated names by every tagger, so
"matching by normalized tag string" reduces to exact matching on space-form
names. Weights are equal and a tag absent from a tagger contributes ``0.0``.
No learned weights or fuzzy matching.

The plain mode divides every tag by the number of taggers ``N``. Because the
member models have **different vocabularies** (a tag structurally absent from a
model is an *absence*, not a low detection), a tag may instead divide only by
the number of taggers whose vocabulary contains it — pass a ``tag_taggers`` map
(tag -> those tagger indices) to ``combine`` / ``prune_accumulator`` for that
vocabulary-aware behaviour. With omitted ``tag_taggers`` the plain ``/ N`` path
runs unchanged (and with a single tagger both paths are identical).
"""

from __future__ import annotations

from typing import Dict, List, MutableMapping, Optional, Sequence


class EnsembleTagger:
    """Sparse, equal-weight confidence averaging over multiple taggers.

    Methods are stateless (static): they take plain dicts and return plain
    dicts, so the class is trivially unit-testable without any model.
    """

    @staticmethod
    def combine(
        list_of_sparse: Sequence[Dict[str, float]],
        tag_taggers: Optional[MutableMapping[str, Sequence[int]]] = None,
    ) -> Dict[str, float]:
        """Average the sparse per-tagger results.

        Plain mode (``tag_taggers is None``): for each unique tag,
        ``ensemble(tag) = sum(conf over every tagger) / N`` where ``N`` is the
        number of taggers provided. A tag missing from a tagger contributes
        ``0.0`` (it simply is not added). All weights are equal. An empty input
        list yields an empty result.

        Tagger-aware mode (``tag_taggers`` given): a tag divides by the number of
        member taggers whose vocabulary contains it (the length of ``tag_taggers``
        entry), because a tagger that structurally lacks the tag's ``0.0`` is an
        *absence*, not a low detection, and must not dilute the score. ``tag_taggers``
        maps a tag name to the cont. of that tag's contributing tagger indices; a
        tag not present falls back to ``N`` (and never divides by zero).
        """
        if not list_of_sparse:
            return {}
        n = len(list_of_sparse)
        sums: Dict[str, float] = {}
        for part in list_of_sparse:
            for tag, conf in part.items():
                sums[tag] = sums.get(tag, 0.0) + conf
        if tag_taggers is None:
            return {tag: sums[tag] / n for tag in sums}
        return {
            tag: sums[tag] / max(1, len(tag_taggers.get(tag, range(n))))
            for tag in sums
        }

    @staticmethod
    def prune_accumulator(
        accum: MutableMapping[str, float],
        passes_done: int,
        passes_total: int,
        min_threshold: float,
        tag_taggers: Optional[MutableMapping[str, Sequence[int]]] = None,
    ) -> MutableMapping[str, float]:
        """Drop tags in-place that can no longer reach ``min_threshold``.

        ``accum`` holds running *sums* (not averages) of confidence across the
        ``passes_done`` taggers processed so far. A tag is kept only while the
        optimistic bound — assuming every remaining tagger adds a perfect
        ``1.0`` — still reaches the threshold::

            (accum[tag] + (passes_total - passes_done) * 1.0) / passes_total
                >= min_threshold

        Tags strictly below that bound are permanently unreachable and are
        removed to keep the per-image accumulators small. A tag at exactly
        the bound is kept. ``accum`` is modified in place and returned.

        With ``tag_taggers`` given (tagger-aware mode, mirroring :meth:`combine`),
        the bound is computed per tag against the number of *that tag's* member
        taggers rather than ``passes_total``: the denominator is ``len`` of the
        tag's member set, and the optimistic margin is only the *upcoming* members
        (indices ``>= passes_done``), each of which can add at most ``1.0``::

            (accum[tag] + remaining_voters * 1.0) / len(members)

        Without it the classical ``/ passes_total`` bound is used unchanged, so the
        plain (``mean``) path is unaffected.
        """
        if passes_total <= 0:
            return accum
        dropped: List[str] = []
        if tag_taggers is None:
            margin = (passes_total - passes_done) * 1.0
            dropped = [
                tag
                for tag, s in accum.items()
                if (s + margin) / passes_total < min_threshold
            ]
        else:
            for tag, s in accum.items():
                members = tag_taggers.get(tag, range(passes_total))
                denom = max(1, len(members))
                remaining = sum(1 for i in members if i >= passes_done)
                if (s + remaining * 1.0) / denom < min_threshold:
                    dropped.append(tag)
        for tag in dropped:
            del accum[tag]
        return accum
