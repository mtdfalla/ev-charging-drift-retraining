# Data

The data are public and are not redistributed in this repository. Download them as below.

## UrbanEV (main data)

* Li, H. et al. UrbanEV: An open benchmark dataset for urban electric vehicle charging demand prediction. Scientific Data 12, 523 (2025). doi:10.1038/s41597-025-04874-4
* Repository: https://github.com/IntelligentSystemsLab/UrbanEV (CC0 1.0). Mirror on Dryad: doi:10.5061/dryad.np5hqc04z
* Coverage: 275 traffic zones in Shenzhen, hourly, 1 September 2022 to 28 February 2023 (4,344 steps), aggregated from 1,362 stations.

Steps:

    git clone --depth 1 https://github.com/IntelligentSystemsLab/UrbanEV.git
    # copy UrbanEV/data/*.csv into data/raw/

The scripts use `volume.csv` (hourly charging volume per zone, kWh, the target), `occupancy.csv` and `duration.csv` (loaded in step 01) and `inf.csv` (zone coordinates for the drift map). Column headers are the zone IDs and the first column is the time.

## City of Palo Alto ChargePoint sessions (robustness check)

* City of Palo Alto, Electric vehicle charging station usage (ChargePoint) open data, sessions 2011 to 2020.
* A copy of the file is kept in the open-data collection of Amara-Ouali et al., A review of electric vehicle load open data and models, Energies 14(8), 2233 (2021): https://github.com/yvenn-amara/ev-load-open-data (folder `1. Input Data/4. City of Palo Alto/`).

Steps:

    git clone --depth 1 https://github.com/yvenn-amara/ev-load-open-data.git
    python code/06_paloalto_ingest.py "ev-load-open-data/1. Input Data/4. City of Palo Alto/ChargePoint Data CY20Q4.csv"

This writes `data_paloalto/raw/volume.csv`: station-hourly energy for the 30 stations active in at least 1% of hours over 2018 to 2020, with each session's energy assigned to its start hour.
