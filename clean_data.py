import polars as pl
from pathlib import Path
import re
from utils import create_id

# Read in data
print("Reading data...")
df = pl.read_parquet("data/detentions_raw.parquet")

print("Cleaning data...")
# Replace all empty strings with null
df = df.with_columns(
    pl.when(pl.col(pl.String).str.len_chars() == 0)
    .then(None)
    .otherwise(pl.col(pl.String))
    .name.keep()
)

# Drop all bond columns
cols_to_drop = [c for c in df.columns if any(term in c.lower() for term in ["bond", "eid_"])]

# Find and drop all redacted columns
redacted_pattern = r'\(b\)|\(B\)|b\([0-9]\)|B\([0-9]\)'

# Get string columns
string_cols = [name for name, dtype in zip(df.columns, df.dtypes) if dtype == pl.String]

# Count invalid values (redacted or null) for each string column
# Collect columns that have all invalid values
invalid_cols = []

invalid_counts = df.select([
    (pl.col(c).str.contains(redacted_pattern) | pl.col(c).is_null())
    .sum()
    .alias(c)
    for c in string_cols
])

total_rows = df.height
invalid_cols = [
    col for col in invalid_counts.columns 
    if invalid_counts[col][0] == total_rows
]

# Combine and drop all columns
cols_to_drop = set(cols_to_drop + invalid_cols)
df = df.drop(*cols_to_drop)

# Create date versions of stay/detention book in/out datetime
# and remove seconds from datetime columns
datetime_cols = [c for c in df.columns if 'time' in c]

df = df.with_columns([
    # Extract dates from datetime columns
    pl.col("detention_book_in_date_time").cast(pl.Date).alias("detention_book_in_date"),
    pl.col("detention_book_out_date_time").cast(pl.Date).alias("detention_book_out_date"),
    pl.col("stay_book_in_date_time").cast(pl.Date).alias("stay_book_in_date"),
    # Truncate all datetime columns to minute precision
    *[pl.col(c).dt.truncate("1m") for c in datetime_cols]
])

detention_id_cols = ["unique_identifier", "detention_facility_code", "detention_book_in_date_time"]

# Create detention id
df = df.with_columns(
    pl.concat_str(
        [pl.col(c).cast(pl.String) for c in detention_id_cols],
        separator="_"
    ).map_batches(create_id).alias("detention_id")
)

# The stay_id uses the date rather than datetime because no two stays should begin on the same day
stay_id_cols = ["unique_identifier", "stay_book_in_date"]

# Remove people with stays that begin on the same day but end at different times
stays = (
    df.group_by(stay_id_cols)
    .agg(
        pl.col("stay_book_out_date_time").n_unique().alias("unique_count")
    )
    .filter(pl.col("unique_count") > 1)
    .select("unique_identifier")
)

df = df.join(stays, on="unique_identifier", how="anti")

# Create stay id
df = df.with_columns(
    pl.concat_str(
        [pl.col(c).cast(pl.String) for c in stay_id_cols],
        separator="_"
    ).map_batches(create_id).alias("stay_id")
)

# Drop rows that are missing a unique identifier
df = df.filter(pl.col("unique_identifier").is_not_null())

# Get duplicated stay_ids
stay_ids = (
    df.group_by("stay_id")
    .agg(
        pl.col("stay_book_out_date").n_unique().alias("unique_count")
    )
    .filter(pl.col("unique_count") > 1)
    .select("stay_id")
)

# Remove duplicate stays
df = df.join(stay_ids, on="stay_id", how="anti")

# Get each stay
stays = df.select("unique_identifier", "stay_book_in_date_time", "stay_book_out_date_time").unique()

# Get next stay using window function
stays = stays.sort("unique_identifier", "stay_book_in_date_time").with_columns(
    pl.col("stay_book_in_date_time")
    .shift(-1)
    .over("unique_identifier")
    .alias("next_stay_book_in_date_time")
)

# Get people with overlapping stays
stays = (
    stays.filter(
        pl.col("stay_book_out_date_time") > pl.col("next_stay_book_in_date_time")
    )
    .select("unique_identifier")
    .unique()
)

# Remove people with stays that overlap
df = df.join(stays, on="unique_identifier", how="anti")

# Keep only the first row (latest book out date) per detention_id
df = (
    df.sort("detention_book_out_date_time", descending=True, nulls_last=False)
    .unique("detention_id", keep="first")
)

# Remove 0 second detentions
df = df.filter(
    (pl.col("detention_book_out_date_time") != pl.col("detention_book_in_date_time")) |
    pl.col("detention_book_out_date_time").is_null()
)

# Order stay within person and add stay number
df = (
    df.sort("unique_identifier", "stay_book_in_date")
    .with_columns(
        # Add stay number for each person
        pl.col("stay_book_in_date")
        .rank(method="dense")
        .over("unique_identifier")
        .alias("stay_number"),
    )
    .with_columns(
        # Add indicator for whether the 'current' stay is the first stay for each person
        (pl.col("stay_number") == 1).alias("first_stay")
    )
)

# Add in facilities data
fac = pl.read_parquet("data/facilities_raw.parquet")
df = df.join(fac, on="detention_facility_code", how="left")

df = df.with_columns([
    # Keep only non-null and not "Transferred" detention release reasons
    pl.when(
        pl.col("detention_release_reason").is_not_null() & 
        (pl.col("detention_release_reason") != "Transferred")
    )
    .then(pl.col("detention_release_reason"))
    .alias("non_transfer_reason")
]).with_columns([
    # Get last non-transfer reason within each stay
    pl.col("non_transfer_reason")
    .last()
    .over("stay_id", order_by="detention_book_in_date_time")
    .alias("non_transfer_reason")
]).with_columns([
    # Replace "Transferred" or missing stay_release_reason with last non-transfer detention release reason
    # Therefore, if all detention_release_reasons within a stay are "Transferred", stay_release_reason is untouched
    pl.when(
        ((pl.col("stay_release_reason") == "Transferred") | pl.col("stay_release_reason").is_null()) & 
        pl.col("non_transfer_reason").is_not_null()
    )
    .then(pl.col("non_transfer_reason"))
    .otherwise(pl.col("stay_release_reason"))
    .alias("stay_release_reason")
])

# Get one row per stay with the removal indicator
stay_level = (
    df.group_by(["unique_identifier", "stay_number"])
    .agg([
        (pl.col("stay_release_reason").first() == "Removed").alias("removed")
    ])
    .sort(["unique_identifier", "stay_number"])
)

# Create the previously_removed indicator at stay level
stay_level = stay_level.with_columns(
    pl.col("removed")
    .cum_max()
    .shift(1, fill_value=False)
    .over("unique_identifier")
    .alias("previously_removed")
)

# Join back to original data
df = df.join(
    stay_level.select(["unique_identifier", "stay_number", "previously_removed"]),
    on=["unique_identifier", "stay_number"],
    how="left"
)

# Drop people that have multiple stays with missing book out dates
drop = (
    df.filter(pl.col("stay_book_out_date").is_null())
    .group_by("unique_identifier")
    .agg(
        pl.col("stay_book_in_date")
        .n_unique()
        .alias("unique_count")
    )
).filter(pl.col("unique_count") > 1).select("unique_identifier")

df = df.join(drop, on="unique_identifier", how="anti")

# Drop people that have multiple stays that end in transfer
drop = (
    df.filter(pl.col("stay_release_reason") == "Transferred")
    .group_by("unique_identifier")
    .agg(pl.col("stay_id").n_unique().alias("num_transfer_stays"))
).filter(pl.col("num_transfer_stays") > 1).select("unique_identifier")

df = df.join(drop, on="unique_identifier", how="anti")

# Get number of stays per person
df = df.with_columns(
    pl.col("stay_number")
    .max()
    .over("unique_identifier")
    .alias("total_num_stays")
)

# Get next stay_id
# Must get unique stays first because a stay can have multiple rows
stay_level = (
    df.select(["unique_identifier", "stay_id", "stay_number"])
    .unique()
    .sort("unique_identifier", "stay_number")
    .with_columns(
        pl.col("stay_id")
        .shift(-1)
        .over("unique_identifier")
        .alias("next_stay_id")
    )
).select(["stay_id", "next_stay_id"])

df = df.join(stay_level, on="stay_id", how="left")

# If stay_release_reason == "Transferred" and it is the last stay, set stay_release_reason to NULL
df = df.with_columns(
    pl.when(
        (pl.col("stay_release_reason") == "Transferred") &
          (pl.col("stay_number") == pl.col("total_num_stays"))
    )
    .then(None)
    .otherwise("stay_release_reason")
    .alias("stay_release_reason")
)

# If stay_release_reason == "Transferred" and it is the not last stay, set stay_id = next_stay_id
df = df.with_columns(
    pl.when(
        (pl.col("stay_release_reason") == "Transferred") &
          (pl.col("stay_number") < pl.col("total_num_stays"))
    )
    .then("next_stay_id")
    .otherwise("stay_id")
    .alias("stay_id")
)

df = df.with_columns(
    pl.col("stay_book_in_date")
    .min() # Set stay_book_in_date to earliest stay_book_in_date
    .over("stay_id")
    .alias("stay_book_in_date")
).with_columns(
    pl.col("stay_book_in_date_time")
    .min()
    .over("stay_id")
    .alias("stay_book_in_date_time")
).with_columns(
    pl.col("stay_book_out_date")
    .last() #Set stay_book_out_date to latest stay_book_out_date (using last to handle null)
    .over("stay_id", order_by="stay_number")
    .alias("stay_book_out_date")
).with_columns(
    pl.col("stay_book_out_date_time")
    .last()
    .over("stay_id", order_by="stay_number")
    .alias("stay_book_out_date_time")
).with_columns(
    pl.col("stay_release_reason")
    .last() # Set stay_release_reason to stay_release_reason where stay_number is highest
    .over("stay_id", order_by="stay_number")
    .alias("stay_release_reason")
)

# Recalculate stay number after combining transfer stays
df = df.with_columns(
        pl.col("stay_book_in_date_time")
        .rank(method="dense")
        .over("unique_identifier", order_by="stay_book_in_date")
        .alias("stay_number"),
)

# Add row number to identify the most recent detention for each stay
# and update the last detention_release_reason to match stay_release_reason (if not "Transferred")
df = (
    df.sort("detention_book_in_date_time", descending=True)
    .with_columns(
        pl.col("detention_book_in_date_time")
        .rank(method="ordinal", descending=True)
        .over("stay_id")
        .alias("rn")
    )
    .with_columns(
        pl.when(
            (pl.col("rn") == 1) & (pl.col("stay_release_reason") != "Transferred")
        )
        .then(pl.col("stay_release_reason"))
        .otherwise(pl.col("detention_release_reason"))
        .alias("detention_release_reason")
    )
)

# Drop stays that have a missing non-final detention_book_out_date_time
drop = df.filter(
    (pl.col("rn") != 1) & pl.col("detention_book_out_date_time").is_null()
                 ).select("stay_id").unique()

df = df.join(drop, on="stay_id", how="anti")

# Get next detentions's book in date and time for each person
df = (
    df.sort("detention_book_in_date_time")
    .with_columns([
        pl.col("detention_book_in_date_time")
        .shift(-1)
        .over("stay_id")
        .alias("next_book_in_date_time"),
        pl.col("detention_book_in_date")
        .shift(-1)
        .over("stay_id")
        .alias("next_book_in_date")
    ])
)

# If next detention book in time is before 'current' book out time, set book out to next book in
df = df.with_columns([
    pl.when(
        pl.col("next_book_in_date") < pl.col("detention_book_out_date")
    )
    .then("next_book_in_date")
    .otherwise("detention_book_out_date")
    .alias("detention_book_out_date"),
    pl.when(
        pl.col("next_book_in_date_time") < pl.col("detention_book_out_date_time")
    )
    .then("next_book_in_date_time")
    .otherwise("detention_book_out_date_time")
    .alias("detention_book_out_date_time")
])

df = df.with_columns(
    pl.col("detention_book_in_date_time")
    .shift(-1)
    .over("unique_identifier", order_by="detention_book_in_date_time")
    .alias("within_person_next_book_in_date_time")
)

# Remove anyone that has detentions that overlap across stays
df = df.with_columns(
    pl.col("detention_book_in_date_time")
    .shift(-1)
    .over("unique_identifier", order_by="detention_book_in_date_time")
    .alias("within_person_next_book_in_date_time")
)

drop = df.filter(
    pl.col("within_person_next_book_in_date_time") < pl.col("detention_book_out_date_time")
).select("unique_identifier").unique()

df = df.join(drop, on="unique_identifier", how="anti")

# Lowercase the final charge column
df = df.with_columns(
    pl.col("final_charge").map_elements(
        # Except the characters wrapped in parentheses (this is a code and case has meaning)
        lambda text: re.sub(r"[^()]+", lambda m: m.group(0).lower(), text) if text else text,
        return_dtype=pl.String
    )
)

# Remove people with multiple birth years
multiple = (
    df.group_by("unique_identifier")
    .agg(
        pl.col("birth_year")
        .n_unique()
        .alias("unique_count")
    )
).filter(pl.col("unique_count") > 1).select("unique_identifier")

df = df.join(multiple, on="unique_identifier", how="anti")

# Take last ethnicity within person
df = df.with_columns(
    pl.col("ethnicity")
    .last(ignore_nulls=True)
    .over("unique_identifier", order_by="detention_book_in_date")
    .alias("ethnicity")
)

# Take last gender within person
df = df.with_columns(
    pl.col("gender")
    .last(ignore_nulls=True)
    .over("gender", order_by="detention_book_in_date")
    .alias("gender")
)

# Replace unknown marital status with NULL
df = df.with_columns(
    pl.when(pl.col("marital_status") == "Unknown")
    .then(None)
    .otherwise("marital_status")
    .alias("marital_status")
)

# Take last marital status within stay
df = df.with_columns(
    pl.col("marital_status")
    .last(ignore_nulls=True)
    .over("stay_id", order_by="detention_book_in_date")
    .alias("marital_status")
)

# Make religions lowercase
df = df.with_columns(
    pl.col("religion").str.to_lowercase().alias('religion')
)

# Remove duplicate whitespace
df = df.with_columns(
    pl.col("religion").str.replace_all(r"\s{2,}", " ").alias("religion")
)

# Replace unknown with NULL
df = df.with_columns(
    pl.when(pl.col("religion").str.contains("(?i)(un|not|none|relig|known)"))
    .then(None)
    .otherwise(pl.col("religion"))
    .alias("religion")
)

# Convert final order indicator to boolean
df = df.with_columns(
    pl.when(pl.col("final_order_yes_no") == "YES")
    .then(True)
    .otherwise(False)
    .alias("final_order_yes_no")
)

# Indicator for whether final order came before book in


# Drop remaining helper columns and save
df = df.drop("within_person_next_book_in_date_time", "non_transfer_reason", "total_num_stays", "next_stay_id")

print("Writing data...")
Path("intermediate").mkdir(parents=True, exist_ok=True)
df.write_parquet("intermediate/detentions_clean.parquet")