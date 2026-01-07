from pathlib import Path
from pandas import ExcelFile
import polars as pl
import io
import requests
from polars.exceptions import SchemaError
from utils import harmonize_dtypes

# data_url.txt is a file with a link to the data
url = Path("data_url.txt").read_text(encoding="utf-8")

print("Downloading data...")
response = requests.get(url)
response.raise_for_status()

file = io.BytesIO(response.content)
print("Download finished.")

print("Reading data...")
sheets = ExcelFile(file).sheet_names

df = [
    pl.read_excel(
        file,
        sheet_name=sheet,
        read_options={"header_row": 6},
        engine="calamine"
    ) for sheet in sheets
]

try:
    df = pl.concat(df)
except SchemaError:
    # If all column names are the same, set string columns to the other sheet's dtype
    if df[0].columns != df[1].columns:
        raise Exception("Sheets have incongruent schemas")
    
    print("Casting columns to harmonize datatypes:")
    df[0], df[1] = harmonize_dtypes(df[0], df[1])
    df = pl.concat(df)
    
# Make column names lowercase and replace spaces with underscores
df = df.rename(lambda col: col.lower().replace(" ", "_"))

# Rename specific columns
df = df.rename({
    "book_in_date_time": "detention_book_in_date_time",
    "most_serious_conviction_(msc)_charge_code": "msc_charge_code"
})

print("Writing data...")
Path("data").mkdir(parents=True, exist_ok=True)
df.write_parquet("data/detentions_raw.parquet")
