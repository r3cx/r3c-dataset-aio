import pandas as pd

INPUT_FILE = "2026-06-24_danbooru_tags_post_count.csv"
OUTPUT_CSV_FILE = "anima_tags.csv"

TAG_THRESHOLD = 25

columns = [
    "id",
    "name",
    "category",
    "post_count",
    "created_at",
    "is_deprecated",
    "words",
]

# ----------------------------------------
# Load
# ----------------------------------------

df = pd.read_csv(
    INPUT_FILE,
    header=None,
    names=columns,
)

# ----------------------------------------
# Type conversion
# ----------------------------------------

df["post_count"] = pd.to_numeric(
    df["post_count"],
    errors="coerce"
)

df["category"] = pd.to_numeric(
    df["category"],
    errors="coerce"
)

df["created_at"] = pd.to_datetime(
    df["created_at"],
    errors="coerce",
    utc=True
)

df["is_deprecated"] = (
    df["is_deprecated"]
    .astype(str)
    .str.lower()
    .map({"true": True, "false": False})
)

# ----------------------------------------
# Global filters
# ----------------------------------------

df = df[
    (~df["is_deprecated"])
    & (df["created_at"] < pd.Timestamp("2026-01-01", tz="UTC"))
]

# ----------------------------------------
# Category-specific thresholds
# ----------------------------------------

artist_mask = (
    (df["category"] == 1)
    & (df["post_count"] >= TAG_THRESHOLD)
)

character_mask = (
    (df["category"] == 4)
    & (df["post_count"] >= TAG_THRESHOLD)
)

general_mask = (
    (df["category"] == 0)
    & (df["post_count"] >= TAG_THRESHOLD)
)

copyright_mask = (
    (df["category"] == 3)
    & (df["post_count"] >= TAG_THRESHOLD)
)

df = df[
    artist_mask
    | character_mask
    | general_mask
    | copyright_mask
]

# ----------------------------------------
# Prefix artists
# ----------------------------------------

artist_rows = df["category"] == 1

df.loc[artist_rows, "name"] = (
    "@"
    + df.loc[artist_rows, "name"].astype(str)
)

# ----------------------------------------
# Remove unused columns
# ----------------------------------------

df = df.drop(
    columns=[
        "id",
        "created_at",
        "words",
        "is_deprecated"
    ]
)

# ----------------------------------------
# Sort by popularity
# ----------------------------------------

df = df.sort_values(
    "post_count",
    ascending=False
)

# ----------------------------------------
# Manual tokens
# ----------------------------------------

manual_tags = [

    # Human quality
    # "masterpiece",
    # "best_quality",
    # "good_quality",
    # "normal_quality",
    # "low_quality",
    # "worst_quality",

    # Pony score
    "score_9",
    "score_8",
    "score_7",
    "score_6",
    "score_5",
    "score_4",
    "score_3",
    "score_2",
    "score_1",

    # Safety
    "safe",
    "sensitive",
    "nsfw",
    "explicit",

    # Time period
    "newest",
    "recent",
    "mid",
    "early",
    "old",

    # Dataset tags
    "ye-pop",
    "deviantart",
]

# Add years

manual_tags.extend(
    [f"year_{year}" for year in range(2000, 2025)]
)

# ----------------------------------------
# Manual tags dataframe
# ----------------------------------------

manual_df = pd.DataFrame({
    "name": manual_tags,
    "category": 5,      # Meta category
    "post_count": 999999999,  # High post count to ensure they appear above other tags
})

# ----------------------------------------
# Append manual tags
# ----------------------------------------

df = pd.concat(
    [manual_df, df],
    ignore_index=True
)

# ----------------------------------------
# Remove duplicates
# Prefer existing Danbooru tags over manual tags
# ----------------------------------------

df = df.drop_duplicates(
    subset=["name"],
    keep="last"
)

# ----------------------------------------
# Save filtered CSV
# ----------------------------------------

df.to_csv(
    OUTPUT_CSV_FILE,
    index=False,
    header=False
)