from pathlib import Path
import polars as pl
import requests
import io

# facilties_url.txt is a file with a link to the data
url = Path("facilities_url.txt").read_text(encoding="utf-8")

print("Downloading facilities data...")
response = requests.get(url)
response.raise_for_status()

file = io.BytesIO(response.content)
print("Download finished.")

print("Reading data...")
df = pl.read_excel(
        file,
        sheet_name=None,
        engine="calamine"
    )

print("Writing data...")
Path("data").mkdir(parents=True, exist_ok=True)
df.write_parquet("data/facilities_raw.parquet")