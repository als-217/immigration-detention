import polars as pl
from pathlib import Path
from utils import find_distance

print("Reading data...")
df = pl.read_parquet("intermediate/detentions_clean.parquet")

print("Prepping data...")
df = df.with_columns(
    (pl.col("stay_book_out_date").is_null()).alias("in_detention")
)

# Set null book out dates to the max book out value
last_book_out = df.select(pl.col("detention_book_out_date").max()).item()
df = df.with_columns(
    pl.when(
        pl.col("detention_book_out_date").is_null()
    )
    .then(last_book_out)
    .otherwise(pl.col("detention_book_out_date"))
    .alias("detention_book_out_date")
)

# Create a within-person next book in date to handle a new stay beginning on the same day
df = df.with_columns(
    pl.col("detention_book_in_date")
    .shift(-1)
    .over("unique_identifier", order_by="detention_book_in_date")
    .alias("within_person_next_book_in_date")
)

# Add the last day of detention for each detention stint
df = df.with_columns(
    pl.when(
        # If the next detention or stay begins on the same day, then the person did not end the day in that facility
        pl.col("within_person_next_book_in_date") == pl.col("detention_book_out_date")
    )
    # So treat the day before the book out date as the last day of detention
    .then(pl.col("within_person_next_book_in_date") - pl.duration(days=1))
    # Else the person was transferred overnight or their next stay began later
    # In this case, the last day of detention is the book out date
    .otherwise(pl.col("detention_book_out_date"))
    .alias("last_detention_date")
)

# If the last date of detention is before the book in date, \
# then the person was booked in, booked out, and booked in again on the same day, drop these rows
df = df.filter(pl.col("detention_book_in_date") <= pl.col("last_detention_date"))

# Get next detention facility information
df = df.with_columns([
    pl.col("latitude")
    .shift(-1)
    .over("stay_id", order_by="detention_book_in_date_time")
    .alias("next_latitude"),
    pl.col("longitude")
    .shift(-1)
    .over("stay_id", order_by="detention_book_in_date_time")
    .alias("next_longitude"),
    pl.col("detention_facility_code")
    .shift(-1)
    .over("stay_id", order_by="detention_book_in_date_time")
    .alias("next_detention_facility_code"),
    pl.col("city")
    .shift(-1)
    .over("stay_id", order_by="detention_book_in_date_time")
    .alias("next_city"),
    pl.col("state")
    .shift(-1)
    .over("stay_id", order_by="detention_book_in_date_time")
    .alias("next_state"),
])

# Compute transfer distance
df = df.with_columns(
    pl.struct(["latitude", "longitude", "next_latitude", "next_longitude"])
    .map_batches(
        lambda s: pl.Series(find_distance(
            s.struct.field("latitude"),
            s.struct.field("longitude"),
            s.struct.field("next_latitude"),
            s.struct.field("next_longitude")
        ))
    )
    .alias("distance_km")
)

# Set minimum book in date to the earliest book out date
earliest_book_out = df.select(pl.col("detention_book_out_date").min()).item()
df = df.with_columns(
    pl.when(
        pl.col("detention_book_in_date") < earliest_book_out
    )
    .then(earliest_book_out)
    .otherwise(pl.col("detention_book_in_date"))
    .alias("detention_start_date")
)

df = df.drop("stay_book_out_date_time", "stay_book_out_date", "next_book_in_date_time",
              "next_book_in_date", "latitude", "longitude", "next_latitude", "next_longitude",
             "rn", "within_person_next_book_in_date")

print("Expanding data into a panel...")
panel = (
    df.with_columns(
        # Collect all dates in each detention and create a temporary column
        pl.date_ranges(
            pl.col("detention_start_date"),
            pl.col("last_detention_date"),
            interval="1d"
        ).alias("detention_date")
    )
    # Create one row for each date in the list
    .explode("detention_date")
    .sort("unique_identifier", "detention_date")
)

# Count number of detention_dates in current detention
panel = panel.with_columns(
    (pl.col("detention_date") - pl.col("detention_book_in_date"))
    .alias("days_in_current_detention")
)

# Count number of detention_dates in current stay
panel = panel.with_columns(
    (pl.col("detention_date") - pl.col("stay_book_in_date"))
    .alias("days_in_current_stay")
)

panel = panel.drop("detention_book_in_date_time", "detention_book_out_date_time",
                   "last_detention_date", "stay_book_in_date", "detention_start_date")

# Add a date id
panel = panel.with_columns(
    pl.col("detention_date").dt.strftime("%Y%m%d").cast(pl.Int32).alias("date_id")
)

print("Writing data...")
Path("processed").mkdir(parents=True, exist_ok=True)
panel.write_parquet("processed/detentions_panel.parquet")