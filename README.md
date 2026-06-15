# Geomagnetic Storm Event Dataset Builder

This project builds a lightweight sliding-window index for geomagnetic storm
event detection. It is designed for continuous 1-minute INTERMAGNET station
data and 1-minute SYM-H labels.

The dataset builder does not save millions of duplicated raw, PSD, or CWT
window files. It stores one continuous array per station-year and a partitioned
Parquet window index. Training code reads each window on demand.

## Current task

- Input: continuous station XYZ time series.
- Target: detect medium-or-stronger geomagnetic storm events in long sequences.
- Label source: SYM-H only; SYM-H is never included in model input.
- Positive threshold: `SYM-H <= -50 nT`.
- Low-value fragments separated by at most 12 hours are grouped into one event.
- Window: 6 hours, left-closed and right-open.
- Stride: 1 hour.
- Missing station samples: reject the affected window; do not interpolate.

## Accepted station data

Station files must use the IAGA-2002 text format with 1-minute XYZ columns.
The builder reads the actual `Data Type` header and does not trust the filename.

Accepted:

- Definitive
- Quasi-definitive
- Provisional

Rejected:

- Variation

If multiple versions exist for the same station-year, the builder selects:

```text
Definitive > Quasi-definitive > Provisional
```

## AutoDL directory layout

The default [config.yaml](config.yaml) expects:

```text
/root/autodl-tmp/data/
├── intermagnet_raw/
│   ├── FUR/2015/*.txt
│   ├── NGK/2015/*.txt
│   └── ...
├── symh_raw/
│   ├── 2008.txt
│   ├── 2009.txt
│   └── 2022.txt
└── geomag_dataset/              # generated output
```

SYM-H input rows must contain:

```text
year day_of_year hour minute symh_nt
```

## Installation on AutoDL

```bash
cd /root/autodl-tmp/cibao
git pull
python -m pip install -r requirements.txt
```

Dataset construction is CPU and disk work. It does not require a GPU. AutoDL
no-card mode can run small tests, but processing all 240 station-year files is
better done in normal mode because no-card mode has limited CPU and memory.

## Recommended execution

Run each stage separately so failures are easy to inspect.

### 1. Audit station files

```bash
python main.py --config config.yaml --stage audit
```

Review:

```text
/root/autodl-tmp/data/geomag_dataset/audit/station_year_sources.csv
```

### 2. Convert station files

```bash
python main.py --config config.yaml --stage convert
```

This produces continuous memory-mapped arrays:

```text
continuous/FUR/2015_data.npy
continuous/FUR/2015_valid.npy
continuous/FUR/2015_metadata.json
```

### 3. Build SYM-H events

```bash
python main.py --config config.yaml --stage symh
```

Output:

```text
labels/storm_events.csv
```

### 4. Build window index

```bash
python main.py --config config.yaml --stage windows
```

Output is partitioned by year:

```text
windows/
├── year=2008/FUR.parquet
├── year=2008/NGK.parquet
└── ...
```

To run everything:

```bash
python main.py --config config.yaml --stage all
```

### 5. Draw quality-control figures

Before model training, draw diverse Storm, NonStorm, and Uncertain examples:

```bash
python inspect_windows.py --config config.yaml --samples-per-group 10
```

This writes at most 90 figures:

```text
figures_check/
├── train/Storm/
├── train/NonStorm/
├── train/Uncertain/
├── val/...
├── test/...
└── manifest.csv
```

Each figure shows delta XYZ, first-difference XYZ, and the matching SYM-H
sequence. Sampling is performed across distinct event/date groups to avoid
drawing many adjacent windows from the same storm.

Run a small test first:

```bash
python main.py --config config.yaml --stage all --stations FUR --years 2015 --overwrite
```

For storm generation, a multi-year run should use consecutive years. The full
2008-2022 configuration is the intended production run.

## Labels

Each complete station window receives:

- `1 / Storm`: window center is inside a merged storm event.
- `0 / NonStorm`: the whole window is outside the 24-hour event buffer and
  every valid SYM-H value is above `-30 nT`.
- `-1 / Uncertain`: event boundary, weak disturbance, recovery interval, or
  missing SYM-H.

`Uncertain` is a label-quality state, not a third physical class. It is excluded
from the default training dataset.

## Data split

The default temporal split is:

```text
train: 2008-2017
val:   2018-2019
test:  2020-2022
```

All stations and windows belonging to the same storm event share the same
`event_id` and split. Events crossing a split boundary are marked `excluded`.

## Dynamic model input

`IndexedWindowDataset` reads a raw `[3, 360]` window and dynamically creates:

```text
raw_delta = raw - median(first 60 minutes)
raw_diff  = first temporal difference
```

The default model input is:

```text
[delta_X, delta_Y, delta_Z, diff_X, diff_Y, diff_Z]
shape = [6, 360]
```

No normalization is performed by the dataset builder.

Example:

```python
from geomag_dataset.dataset import IndexedWindowDataset

dataset = IndexedWindowDataset(
    "/root/autodl-tmp/data/geomag_dataset",
    split="train",
)

sample = dataset[0]
print(sample["time_domain"].shape)  # (6, 360)
print(sample["label"])
```

PSD and CWT are disabled by default and calculated only when enabled in
`features` inside `config.yaml`. They are not saved for every window.

## Useful overrides

Process selected stations:

```bash
python main.py --config config.yaml --stage convert --stations FUR,NGK
```

Process selected years:

```bash
python main.py --config config.yaml --stage convert --years 2015:2017
```

Rebuild existing output:

```bash
python main.py --config config.yaml --stage windows --overwrite
```

Logs are written to:

```text
/root/autodl-tmp/data/geomag_dataset/logs/build_dataset.log
```
