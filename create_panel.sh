#!/bin/bash

# Activate the venv
source id_venv/bin/activate

echo "Ingesting raw data..."
python3 ingest_data.py

if [ $? -eq 0 ]; then
    echo "Data ingested successfully."
else
    echo "Error: ingest_data.py failed. Exiting."
    exit 1
fi

echo "Ingesting facilities data..."
python3 ingest_facilities.py

if [ $? -eq 0 ]; then
    echo "Facilities data ingested successfully."
else
    echo "Error: facilities_data.py failed. Exiting."
    exit 1
fi

echo "Creating clean dataset..."
python3 clean_data.py

if [ $? -eq 0 ]; then
    echo "Data cleaned successfully."
else
    echo "Error: clean_data.py failed. Exiting."
fi