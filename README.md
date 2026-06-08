# 2D Data Matrix Reader

Streamlit app and benchmark scripts for reading cropped Data Matrix images.

The app reads each image independently. As soon as one preprocessing path and decoder returns a valid Data Matrix result, that image stops scanning and the batch runner moves on to the next file.

## Run

```bash
source /home/wseokyang/miniconda3/etc/profile.d/conda.sh
conda activate 2d12
streamlit run app.py
```

## Reproducible Runtime

The most reproducible path is Docker, because `pylibdmtx` depends on the native OS `libdmtx` shared library, not only the Python package.

```bash
docker build -t 2d-reading .
docker run --rm -p 8501:8501 2d-reading
```

Then open:

```text
http://localhost:8501
```

To verify the decoder stack inside the container:

```bash
docker run --rm 2d-reading python scripts/check_runtime.py --benchmark --max-seconds 30
```

Expected local benchmark target for `test_cases/` is 23 / 24 with a 30 second per-image limit. A result around 20 / 24 usually means one of these is different:

- native `libdmtx` is missing or a different OS package is loaded
- `pylibdmtx` is installed but cannot find `libdmtx`
- the per-image timeout is too short for the slower `pylibdmtx` cases
- Python package versions differ from `requirements.txt`

For a native Ubuntu/WSL setup, install the OS library before Python packages:

```bash
sudo apt-get update
sudo apt-get install -y libdmtx0t64 openjdk-17-jre-headless
pip install -r requirements.txt
python scripts/check_runtime.py --benchmark --max-seconds 30
```

On older Debian/Ubuntu images the package may be named `libdmtx0b` instead of `libdmtx0t64`. If native setup is inconsistent, use Docker.

## Benchmark Samples

```bash
source /home/wseokyang/miniconda3/etc/profile.d/conda.sh
conda activate 2d12
python scripts/benchmark_cases.py --mode aggressive --max-seconds 30
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
