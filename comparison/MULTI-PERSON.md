# Multi-person exploratory comparison

This guide covers the existing Face-API WASM and SFace comparison on the
[synthetic multi-person dataset](../multi-person/README.md). The test uses
intended labels and example thresholds, not independently verified identities.

## Prerequisites

Prepare the Node dependencies and model assets from [COMPARISON.md](COMPARISON.md).
Python requires NumPy and an OpenCV build with YuNet and SFace support; the
Python environment is not locked. Record OS and runtime versions. After setup,
batch inference uses local assets. The example uses Windows PowerShell; other
platforms require verification.

## Run and inspect results

From the repository root, run both local engines on `../multi-person/manifest.csv`, then open the generated HTML report:

```powershell
cd comparison
python -B multi-compare.py
```

The command creates a new timestamped JSON file and an HTML file in `comparison/multi-person-results/`; it does not overwrite previous runs. Open the HTML file directly in a browser. For the default dataset and output directory, thumbnails refer to local images. Filter the comparison table by probe identity or intended relationship. The image table shows every manifest row, missing files, invalid or missing labels, detection failures and each engine's error. The comparison table includes every probe against every listed enrollment; missing intended enrollments also receive an explicit unavailable row. The JSON contains the same detailed results without face descriptors. A nonzero exit means the dataset is incomplete, malformed, has detection failures, or an engine is unavailable; the generated report still records the results when possible. A missing or invalid manifest prevents a run. `--manifest PATH` and `--output-dir PATH` select an alternate dataset and output location, useful for small tests; image paths are relative to the manifest directory and cannot escape it. Thumbnail links in the report are intended for the default dataset location.

The exact CSV headers are `person_id,image,role,scenario,identity_review`. The expected dataset has 20 intended identities, each with one `enrollment` and four `probe` rows. Other counts, duplicate image paths, missing fields and invalid roles are reported. Paths must be relative, forward-slash paths. These labels remain unverified regardless of the `identity_review` value. Review them independently before drawing identity conclusions.

The existing locked Node dependencies and local models described in [COMPARISON.md](COMPARISON.md) are used without network access. Python needs NumPy and OpenCV with YuNet and SFace support; its OpenCV version is recorded, but this environment is not locked by the project. Face-API checks that both TF.js and Face-API share the WASM engine before and after inference. Face-API uses SSD MobileNet V1 (0.5 confidence), landmarks and Euclidean distance ≤ 0.6; SFace uses YuNet (0.9 confidence), 640-pixel maximum dimension, aligned SFace features and cosine similarity ≥ 0.363. Multiple detected faces are excluded. The report records runtime versions, detector settings, the manifest SHA-256 and model SHA-256 values.

The thresholds are examples, not calibrated on this dataset. There is no assumed calibration/evaluation split. Intended same/different labels are not ground truth; the synthetic set cannot establish accuracy, liveness or security. The score scales differ. Node WASM and Python/OpenCV have different detectors, preprocessing and runtimes, so their timings are not a fair speed comparison; the multi-person report makes no speed claim. Complete image generation and independent identity review are separate prerequisites for interpreting the full dataset.

## Saved observation

The saved `multi-comparison-2026-10-27T09-34-58-670808Z.json` contains
100 detected images per engine and 1,600 scored pairs. Both engines matched
all 80 intended-same pairs. Face-API matched 166 of 1,520 intended-different
pairs; SFace matched 242. These counts use the example thresholds and pending
labels above; they are not validated false-accept rates.

The report includes a paginated visual review of intended different-person pairs, filterable by whether both engines matched or only one did. It displays the probe and reference images side by side and retains their model scores. Visual resemblance and model outputs do not establish identity, so this review does not change the pending identity labels.

Run focused logic checks with `python -B test-multi-compare.py`. The existing one-person runners and their saved results are independent of this workflow.
