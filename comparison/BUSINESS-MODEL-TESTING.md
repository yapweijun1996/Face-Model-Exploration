# Local business-use model testing plan

This is a plan for later testing, not a completed benchmark or a ranking by
accuracy. The ten artifacts below represent four model families: variants and
conversions are not independent models. Use the exact published artifact and
record its source, version, hash and license. A license permitting commercial
use does not by itself establish suitability, accuracy or legal compliance.

## Candidate inventory

| # | Artifact | Business-use evidence | Platform and project status |
| --- | --- | --- | --- |
| 1 | AuraFace v1 `glintr100.onnx` | [Pinned model card](https://huggingface.co/fal/AuraFace-v1/blob/686774cad65e40933022896195e07e01ee06bee9/README.md) declares Apache-2.0. The pinned revision has no `LICENSE.md`; the [later license file](https://huggingface.co/fal/AuraFace-v1/blob/main/LICENSE.md) should be reviewed against the intended artifact before distribution. | ONNX Runtime supports Windows, Linux and macOS. Existing experimental runner also requires and runs EdgeFace; no AuraFace-only runner exists. |
| 2 | SFace `face_recognition_sface_2021dec.onnx` | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/blob/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/README.md) states the directory files are Apache-2.0. The exact checkpoint's training data is the subject of an [open clarification request](https://github.com/opencv/opencv_zoo/issues/313). | OpenCV on Windows, Linux and macOS; existing local runner. |
| 3 | SFace `face_recognition_sface_2021dec_int8.onnx` | [OpenCV Zoo SFace directory](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/README.md), Apache-2.0. | Quantized variant; backend and platform compatibility unverified here; no runner. |
| 4 | SFace `face_recognition_sface_2021dec_int8bq.onnx` | [OpenCV Zoo SFace directory](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/README.md), Apache-2.0. | Block-quantized variant; backend and platform compatibility unverified here; no runner. |
| 5 | SeetaFace6 `face_recognizer.csta` | [Publisher](https://github.com/seetafaceengine/SeetaFace6) says its open edition is free for commercial and personal use. | General model; publisher lists Windows and Linux packages, macOS is listed as coming soon; no project runner. |
| 6 | SeetaFace6 `face_recognizer_light.csta` | Same [publisher statement](https://github.com/seetafaceengine/SeetaFace6). | Lightweight variant; same platform caveat; no project runner. |
| 7 | SeetaFace6 `face_recognizer_mask.csta` | Same [publisher statement](https://github.com/seetafaceengine/SeetaFace6). | Mask variant; same platform caveat; no project runner. |
| 8 | dlib `dlib_face_recognition_resnet_model_v1.dat` | [Author's model repository](https://github.com/davisking/dlib-models) states the models are released into the public domain and describes the training sources. | dlib supports desktop platforms; installation and specific dependencies need verification; no project runner. |
| 9 | AuraFace Core ML `FaceEmbedding.mlmodel` | [Third-party conversion](https://huggingface.co/RuiSumida/AuraFace-v1-CoreML) declares Apache-2.0. Verify provenance and conversion parity. | Apple Core ML; not a native Windows or Linux model; no project runner. |
| 10 | SFace GGUF | [Third-party conversion](https://huggingface.co/cstr/sface-GGUF/blob/main/README.md) declares Apache-2.0. Verify provenance and conversion parity. | Requires a compatible GGUF runtime; [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed) documents Windows, Linux and macOS builds; no project runner. |

Neither the third-party conversions nor quantized variants have been validated
by this project. Check the licenses of runtimes, detector and alignment assets
separately, and retain applicable notices when distributing. This inventory
does not require prior model-provider approval under the cited terms; verify
the precise artifact and applicable terms for the intended business use.

## Step 1: Prepare the test record

1. Select a single machine and record OS, architecture, CPU, memory, Python and
   OpenCV versions; record GPU/backend if used. Repeat on other platforms later.
2. Start with the repository's synthetic images and manifests. Labels are
   intended identities and remain unverified; inspect missing images, identity
   drift and detection failures before interpreting any scores.
3. Record model filename, source URL, revision, SHA-256, detector, alignment,
   thresholds and runtime versions. Store results separately for each run.
4. Inference should use local files. Setup may require downloads. Do not place
   sensitive images or biometric data in the repository or generated reports.

## Step 2: Run the existing SFace smoke test

The runner expects the standard SFace and YuNet model files in
`comparison/vendor/models/`. Their pinned URLs and expected hashes are in
[`download.py`](download.py); YuNet's model license is [MIT](https://github.com/opencv/opencv_zoo/blob/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/LICENSE).
Install compatible Python, NumPy and OpenCV locally as described in
[COMPARISON.md](COMPARISON.md). Obtain the two verified assets before the run.
`python -B download.py` is a convenient setup option, but downloads additional
Face-API assets too and uses the network; it does not overwrite existing files.

From the repository root on Windows:

```powershell
cd comparison
python -B run-sface.py
```

On Linux or macOS, use `cd comparison` and `python3 -B run-sface.py` with the
same dependencies and assets. These platforms have not been tested here.

The runner writes a new timestamped `results-sface-*.json` with 40 image rows.
Check enrollment, all row statuses, missing files, zero or multiple detections,
processing errors and scores. Exit zero alone does not mean every image was
detected: detection failures also require inspection. The example cosine
threshold 0.363 is not calibrated for a business use case.

## Step 3: Compare AuraFace

The current [AuraFace exploratory runner](exploratory-onnx/README.md) requires
both AuraFace and EdgeFace, a previous multi-person comparison, and their exact
model hashes. It cannot run AuraFace independently. EdgeFace's official model
card has non-commercial terms and the third-party conversion's provenance and
rights remain unresolved. Do not treat this combined runner as a business-use
only test. An AuraFace-only runner is future work; after it exists, run the same
reviewed images with recorded detector, alignment, errors and raw cosine scores.
Do not reuse SFace thresholds for AuraFace.

## Step 4: Evaluate remaining candidates one at a time

No runners for items 3–10 are implemented here. For each candidate, verify its
exact artifact and license, install its supported runtime, record hashes and
versions, then build or obtain an appropriate local test harness. Run the same
reviewed dataset; report unsupported backends, missing images and detection or
inference failures. For conversions and quantized variants, compare outputs with
their base model and assess any loss of accuracy. Confirm SeetaFace model and
SDK version compatibility, particularly on macOS, before planning that test.

## Step 5: Plan a business-relevant evaluation

Use appropriately authorized, independently labeled images representative of
the intended users, devices and conditions. Separate threshold calibration from
final evaluation by identity or another appropriate independent split. Report
false accepts and false rejects at specified thresholds, detection failures,
sample counts, latency and relevant demographic or capture-condition results.
Evaluate liveness separately if required. Never present synthetic smoke-test
counts as validated real-world accuracy or security.

## Test record template

| Field | Record |
| --- | --- |
| Date, operator, OS / architecture | |
| Model file, source, revision, SHA-256, license | |
| Runtime, version, hardware, detector and alignment | |
| Dataset, label review and evaluation split | |
| Threshold, scores and decision rule | |
| Enrollment / probes, detection failures and errors | |
| False accepts / rejects, latency and limitations | |
| Result file and next action | |
