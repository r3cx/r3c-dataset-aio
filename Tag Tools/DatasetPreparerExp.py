"""
DatasetPreparerExp.py — Post-process .txt tag files after autotagging.

Reads each .txt file alongside the tagged images and applies:
    1. Parenthesis escaping for SD prompt compatibility
    2. Underscore → space normalization
    3. Negative tag removal (low-value, redundant tags)
    4. Precedence pruning (keeps strongest tag per category, e.g. "huge breasts" over "breasts")
    5. Optional trigger tag insertion/removal
    6. Optional quality/score tag insertion
    7. Deduplication while preserving order

Usage:
    python DatasetPreparerExp.py --data_dir "path/to/images" --trigger_tag "artist_name"
"""

import argparse
import os

# -----------------------------------------------------------------------------
# Default Configuration
# -----------------------------------------------------------------------------
SCORE = ""          # Quality/score tags to prepend (e.g. "score_9")
TRIGGER = ""        # Activation/trigger tag to prepend
OLD_TRIGGER = ""    # Legacy trigger tag to remove

# Tags that add no training value and should be stripped
NEGATIVES = (
    "transparent background, unknown, official alternate costume, alternate costume, "
    "alternate hairstyle, alternate breast size, official alternate hairstyle, "
    "alternate hair length, alternate eye color, alternate hair color, "
    "virtual youtuber, borrowed character, score 9, score 8 up, score 7 up, score 6 up, "
    "score 5 up, score 4 up, parody, style parody, "
    "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up"
)

# -----------------------------------------------------------------------------
# Tag Manipulation
# -----------------------------------------------------------------------------

def remove_duplicate_tags(tag_list):
    """Remove duplicate tags while preserving original order.

    Re-splits the tag list by comma first to handle cases where prior steps
    introduced comma-delimited sub-tags. Filters out empty entries.
    """
    tag_list = [t.strip() for t in ", ".join(tag_list).split(",") if t.strip()]
    seen = set()
    unique = []
    for tag in tag_list:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)
    return unique


def prune_precedence(tags, precedence_tags, exception_tags):
    """Keep only the strongest tag from a precedence chain.

    precedence_tags is ordered strongest-first (e.g. ["huge breasts", "large breasts", "breasts"]).
    If the strongest match is found in `tags`, all weaker tags in the chain are removed.
    Skipped entirely if any exception tag is present (e.g. "multiple girls").

    NOTE: precedence_tags is mutated (items removed via .remove()). Pass a fresh list each call.
    """
    if any(tag in exception_tags for tag in tags):
        return tags

    for ptag in precedence_tags:
        if ptag in tags:
            precedence_tags.remove(ptag)
            break

    return [tag for tag in tags if tag not in precedence_tags]


def prune_preceding_tags(tags):
    """Apply all precedence pruning rules to a tag list.

    For each anatomical category, keeps only the strongest descriptor.
    Skipped when multiple subjects are present (exception tags).
    """
    # Male anatomy
    tags = prune_precedence(tags,
        ["gigantic testicles", "huge testicles", "large testicles", "small testicles", "testicles"],
        ["multiple penises", "multiple boys"])
    tags = prune_precedence(tags,
        ["gigantic penis", "huge penis", "large penis", "small penis", "penis"],
        ["multiple penises", "multiple boys"])

    # Female anatomy
    tags = prune_precedence(tags,
        ["gigantic breasts", "huge breasts", "large breasts", "medium breasts",
         "small breasts", "flat chest", "breasts"],
        ["multiple girls"])
    tags = prune_precedence(tags,
        ["huge nipples", "small nipples", "nipples"],
        ["multiple girls"])
    tags = prune_precedence(tags,
        ["large areolae", "areolae"],
        ["multiple girls"])
    tags = prune_precedence(tags,
        ["huge ass", "ass"],
        ["multiple girls"])
    tags = prune_precedence(tags,
        ["thick thighs", "thighs"],
        ["multiple girls"])

    # Hair
    tags = prune_precedence(tags,
        ["absurdly long hair", "very long hair", "long hair"],
        ["multiple girls"])
    tags = prune_precedence(tags,
        ["long bangs", "short bangs", "bangs"],
        ["multiple girls"])

    return tags


def remove_negative_tags(tags, negative_tags):
    """Remove tags that are in the negative set."""
    return [tag for tag in tags if tag not in negative_tags]


def escape_parenthesis_and_replace_underscore(tags):
    """Escape parentheses and normalize underscores for SD prompt format.

    Parentheses are escaped as \\( and \\) for Stable Diffusion weight syntax.
    Underscores are replaced with spaces (Danbooru convention → prompt convention).

    NOTE: Only escapes the first occurrence of each parenthesis character per tag.
    Tags with multiple parenthesis pairs may need additional handling.
    """
    result = []
    for tag in tags:
        if "(" in tag and r"\(" not in tag:
            tag = tag.replace("(", r"\(")
        if ")" in tag and r"\)" not in tag:
            tag = tag.replace(")", r"\)")
        tag = tag.replace("_", " ")
        result.append(tag)
    return result


# -----------------------------------------------------------------------------
# Main Processing
# -----------------------------------------------------------------------------

def process_dataset(dataset_path, trigger, old_trigger, quality, undesired):
    """Process all .txt files in a directory through the tag pipeline."""
    print(f"Start Processing For: {dataset_path}")

    txt_files = [f for f in os.listdir(dataset_path) if f.endswith(".txt")]
    tag_stats = []

    undesired_set = set(t.strip() for t in undesired.split(",") if t.strip())

    for txt in txt_files:
        filepath = os.path.join(dataset_path, txt)

        with open(filepath, "r", encoding="utf-8") as f:
            tags = [t.strip() for t in f.read().split(",") if t.strip()]

        # Pipeline
        tags = escape_parenthesis_and_replace_underscore(tags)
        tags = remove_negative_tags(tags, undesired_set)
        tags = prune_preceding_tags(tags)

        if quality:
            tags.insert(0, quality)
        if old_trigger:
            tags = [t for t in tags if t != old_trigger]
        if trigger:
            tags.insert(0, trigger)

        tags = remove_duplicate_tags(tags)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(", ".join(tags))

        print(f"  Tag Count: {len(tags)} | Processed: {txt}")
        tag_stats.append(len(tags))

    if tag_stats:
        print(f"\nUpdated: {len(txt_files)} files")
        print(f"Average Tag Count: {sum(tag_stats) / len(tag_stats):.1f}")
        print(f"Highest Tag Count: {max(tag_stats)}")
        print(f"Lowest Tag Count: {min(tag_stats)}")
    print()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def setup_argument_parser():
    """Build the argparse.ArgumentParser for this script."""
    parser = argparse.ArgumentParser(
        description="Post-process autotagged .txt files: clean, prune, and format tags.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing image/.txt pairs to process")
    parser.add_argument("--trigger_tag", type=str, default=TRIGGER,
                        help="Activation tag to prepend to each file")
    parser.add_argument("--old_trigger_tag", type=str, default=OLD_TRIGGER,
                        help="Legacy trigger tag to remove from each file")
    parser.add_argument("--quality_tags", type=str, default=SCORE,
                        help="Quality/score tag(s) to prepend (e.g. 'score_9')")
    parser.add_argument("--undesired_tags", type=str, default=NEGATIVES,
                        help="Comma-separated tags to remove from output")
    return parser


if __name__ == "__main__":
    parser = setup_argument_parser()
    args = parser.parse_args()
    process_dataset(args.data_dir, args.trigger_tag, args.old_trigger_tag,
                    args.quality_tags, args.undesired_tags)