#!/bin/bash

# Activate the venv
source id_venv/bin/activate

echo "Ingesting raw data..."
python3 ingest_data.py

if [ $? -eq 0 ]; then
    echo "Data ingested successfully."
else
    echo "Error: ingest_data.py failed. Exiting."
    exit 1 # Exit the bash script if a Python script fails
fi