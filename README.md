# 2D Data Matrix Reader

Streamlit app and benchmark scripts for reading cropped Data Matrix images.

The app reads each image independently. As soon as one preprocessing path and decoder returns a valid Data Matrix result, that image stops scanning and the batch runner moves on to the next file.

## Run

```bash
source /home/wseokyang/miniconda3/etc/profile.d/conda.sh
conda activate 2d12
streamlit run app.py
```

## Benchmark Samples

```bash
source /home/wseokyang/miniconda3/etc/profile.d/conda.sh
conda activate 2d12
python scripts/benchmark_cases.py --mode aggressive --max-seconds 6
```

The benchmark writes:

- `reports/benchmark_cases.csv`
- `reports/benchmark_cases.json`

## Compare Engines

```bash
source /home/wseokyang/miniconda3/etc/profile.d/conda.sh
conda activate 2d12
python scripts/compare_engines.py --cases test_cases2 --per-engine-seconds 18 --candidate-limit 520
```

Local `test_cases2` benchmark result:

- `zxing-cpp`: 10 / 24
- `pylibdmtx`: 21 / 24
- OpenCV barcode detector: 0 / 24
- Combined: 21 / 24
- Current app reader after grid reconstruction fallback: 21 / 24

The comparison writes:

- `reports/test_cases2_engine_compare_final.csv`
- `reports/test_cases2_engine_compare_final.json`

## Hard-Case Search

```bash
source /home/wseokyang/miniconda3/etc/profile.d/conda.sh
conda activate 2d12
python scripts/targeted_failure_search.py --seconds 600
```

`pyzxing` needs Java. In this workspace, `openjdk` was installed into the `2d12` conda environment.

Local `test_cases` benchmark result after targeted search and grid reconstruction:

- Combined reader: 23 / 24
- Newly recovered hard case: `{B3CB698A-59D5-41A0-8DC4-01E686BF9AB9}.png`
- Grid-reconstructed hard case: `{0783A71B-ACC9-4179-A3AF-25FC9042B4FC}.png`
- Still not decoded by local decoders: `{E7CA298A-DC03-45A1-BABC-BEF2C7D89DFB}.png`

The targeted search writes:

- `reports/targeted_failure_search.csv`
- `reports/targeted_failure_search.json`

The latest grid reconstruction benchmark writes:

- `reports/test_cases_after_grid_reconstruction.csv`
- `reports/test_cases_after_grid_reconstruction.json`
- `reports/test_cases2_after_grid_reconstruction.csv`
- `reports/test_cases2_after_grid_reconstruction.json`
- `reports/catalog_match_probe.csv`
- `reports/catalog_match_probe.json`

## Notes

The reader tries the original image first, then scans adjusted grayscale/color-channel variants with contrast, gamma, thresholding, quiet-zone padding, scaling, and small-angle rotations. `zxing-cpp` is tried first for fast wins, and `pylibdmtx` is used heavily for tight crops and direct-part-mark style images.

For very low-contrast crops, the reader also has a compact grid reconstruction fallback. It samples likely 14x14/16x16 Data Matrix module grids, enforces only the fixed finder/timing border, re-renders a clean symbol, then accepts the result only when `zxing-cpp` successfully ECC-decodes the reconstructed symbol. This recovered the red low-contrast `{0783A71B-ACC9-4179-A3AF-25FC9042B4FC}.png` case. The remaining `{E7CA298A-DC03-45A1-BABC-BEF2C7D89DFB}.png` crop is too blurred/low-contrast for the tested local decoders and did not produce a reliable catalog match.

The public repository includes the local `test_cases/` samples and generated `reports/` artifacts for reproducibility. `test_cases2/` remains ignored unless it is explicitly prepared for public release.
