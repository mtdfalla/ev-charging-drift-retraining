# ev-charging-drift-retraining

Code and processing scripts for the paper:

> M. Abdelmagid, M. Maalouf. *Non-Stationarity in Urban Electric-Vehicle Charging Demand: A Forecasting Benchmark and Drift-Triggered Retraining Rule*. IEEE International Conference on Industrial Engineering and Engineering Management (IEEM 2026), Singapore, December 2026.

MIT license for the code. The data are not redistributed here. `data/README.md` explains where to download them.

The study measures how non-stationary zone-level charging demand is, benchmarks naive, adaptive and static learned forecasters under a leakage-free day-ahead protocol, and frames retraining as a decision under a refit budget with a Page-Hinkley trigger on delayed live errors. The main data are the 275 traffic zones of the UrbanEV benchmark (Shenzhen, hourly, September 2022 to February 2023). City of Palo Alto ChargePoint sessions (30 stations, 2018 to 2020) serve as a robustness check.

## Environment

Python 3.11 or later. Tested with Python 3.13, numpy 2.5, pandas 3.0, scikit-learn 1.9, scipy 1.18, statsmodels 0.15 and matplotlib 3.11. Everything runs on a CPU.

    pip install -r requirements.txt

## Running

UrbanEV, from the repository root:

    python code/01_load_data.py                       # processed targets and the 60/20/20 split
    python code/02_drift_analysis.py                  # drift descriptors (Table I, Figs. 1 and 2)
    python code/03_forecast_benchmark.py reset
    python code/03_forecast_benchmark.py chunk 0 275 275   # naive and adaptive models, all zones
    python code/03_forecast_benchmark.py global 275        # static ridge, MLP and GBT
    python code/03_forecast_benchmark.py agg
    python code/05_tables.py                               # Tables II and III, paired tests
    python code/04_retrain_rule.py reset
    python code/04_retrain_rule.py chunk 0 60 60           # 60 drift-stratified zones
    python code/04_retrain_rule.py agg                     # Table IV and Fig. 4

Palo Alto (PowerShell shown, use `export` on Linux or macOS):

    python code/06_paloalto_ingest.py "<path>/ChargePoint Data CY20Q4.csv"
    $env:DATAROOT="data_paloalto/raw"; $env:RESULTS_DIR="experiments/results_pa"; $env:FIG_DIR="experiments/figures_pa"
    python code/02_drift_analysis.py
    python code/03_forecast_benchmark.py reset
    python code/03_forecast_benchmark.py chunk 0 30 30
    python code/03_forecast_benchmark.py global 30
    python code/03_forecast_benchmark.py agg

Then, with the variables cleared, `python code/07_cross_dataset.py` builds the two-city comparison (Fig. 3). The full UrbanEV run takes a few minutes on a laptop. Chunks can be split (for example `chunk 0 100 275`, then `chunk 100 275 275`) and are appended to a partial file before `agg`.

## Results

`experiments/results/` and `experiments/results_pa/` hold the result files behind the paper, so every table can be checked without a rerun:

* `drift_summary_volume.csv`: per-zone drift descriptors (PSI, KS, Wasserstein, mean shift, Page-Hinkley changes, STL trend and seasonal strength, drift-type label, coordinates).
* `benchmark_volume.csv` and `benchmark_volume_agg.csv`: per-zone and mean MASE, sMAPE, MAE and RMSE for every model at h = 1 and h = 24.
* `retrain_volume.csv` and `retrain_volume_agg.csv`: the five retraining policies on the 60 zones, with refits, gain and cost.
* `cross_dataset_h24_v2.csv`: day-ahead MASE in both cities.
* `table2_volume.csv` and `table3_volume.csv`: Tables II and III as built by `code/05_tables.py`.

## Notes

* **Leakage-free adaptation.** A forecast made at origin t for t + h is scored and used for learning only after y(t + h) is observed. Adaptive models update at t with the pair (x(t - h), y(t)). The retraining trigger reads only delayed errors.
* **MASE scale.** The in-sample naive error for h = 1 and the seasonal-naive (24 h) error for h = 24, computed on the training period of each zone.
* **Reproducibility.** A rerun reproduces the drift descriptors, every naive, adaptive, ridge and MLP result and the retraining table exactly. The histogram gradient-boosted trees (GBT) can differ slightly across scikit-learn versions: the day-ahead mean MASE was 0.796 in the paper and 0.794 with scikit-learn 1.9. The comparisons in the paper hold.
* **Palo Alto.** Session energy is assigned to the start hour, which can concentrate energy at the session start. The paper treats Palo Alto as a robustness check for this reason.
* **Seeds.** Fixed at 42 in `code/utils.py`.

## Contents

* `code/` data loading, drift analysis, forecasting benchmark, tables and tests, retraining rule, Palo Alto ingestion and the two-city comparison.
* `data/README.md` data sources, licences and download steps.
* `experiments/` result files of the paper.

## Citation

See `CITATION.cff`.
