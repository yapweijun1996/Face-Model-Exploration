# Local comparison smoke test

This guide covers the single-person dataset and local Face-API WASM and SFace
runners. It is an exploratory smoke test. For the planned business-use model
evaluation, see [BUSINESS-MODEL-TESTING.md](BUSINESS-MODEL-TESTING.md).

## Prerequisites and preparation

- Run commands from the repository root unless stated otherwise.
- Face-API requires Node.js, locked `comparison/package-lock.json`
  dependencies and local Face-API model files.
- SFace requires Python 3, NumPy and OpenCV with `FaceDetectorYN` and
  `FaceRecognizerSF`, plus YuNet and SFace model files. Python dependencies are
  not locked; record the actual versions used.
- `download.py` fetches public assets and checks the pinned ONNX hashes. It
  also fetches Face-API assets. Setup requires network access; the offline
  batch runners use local files.
- Examples use Windows PowerShell. Verify dependency availability and commands
  separately on Linux and macOS.

## Existing observations

Both offline pipelines completed the 40-image smoke test. The intended same-person probes were matched in 29/29 cases by both pipelines. Of 10 intended different-person probes, Face-API matched 5 and SFace matched 4. Face-API used a distance threshold of 0.6; SFace used a cosine similarity threshold of 0.363. The labels have not been independently verified and the images are synthetic; these counts do not establish real-world accuracy, liveness, or security. See the timestamped `results-face-wasm-*.json` and `results-sface-*.json` files for individual scores and statuses.

## Offline Face-API WASM batch runner

`face-wasm.js` reads the 40 local PNG files and existing Face-API models without a server, browser, or network access during inference. It checks that both TensorFlow.js and Face-API share the same engine and report the `wasm` backend, then writes a new timestamped `results-face-wasm-*.json` containing per-image outcomes and grouped counts. It enrolls `face-01.png`; compares the other images using Euclidean distance <= 0.6; records missing files, detection failures and processing errors; and exits nonzero for missing files, processing errors, or unavailable enrollment. No output is overwritten. Detection uses SSD MobileNet V1 at 0.5 confidence and the 68-point landmark and recognition models. PNG decoding uses `pngjs` and input is passed as an RGB tensor. Node tensor preprocessing and timing may differ from the browser path.

The direct dependencies are pinned in `package.json` and transitive dependencies in `package-lock.json`: `@vladmandic/face-api` 1.7.15, `@tensorflow/tfjs` 4.22.0, `@tensorflow/tfjs-backend-wasm` 4.22.0, and `pngjs` 5.0.0. Face-API's [package metadata](https://github.com/vladmandic/face-api/blob/master/package.json) specifies TF.js 4.22.0 for development, and its [Node WASM example](https://github.com/vladmandic/face-api/blob/master/demo/node-wasm.js) uses the dedicated Node WASM build. The backend package contains the matching WASM binaries. Public tarball URLs for the direct dependencies are:

- https://registry.npmjs.org/@vladmandic/face-api/-/face-api-1.7.15.tgz
- https://registry.npmjs.org/@tensorflow/tfjs/-/tfjs-4.22.0.tgz
- https://registry.npmjs.org/@tensorflow/tfjs-backend-wasm/-/tfjs-backend-wasm-4.22.0.tgz
- https://registry.npmjs.org/pngjs/-/pngjs-5.0.0.tgz

From the repository root, install the locked dependencies and run with lifecycle scripts disabled:

```powershell
cd comparison
npm.cmd ci --ignore-scripts --no-audit --no-fund
node face-wasm.js
```

Verify the output reports `backend: wasm`, all 40 rows, and the status counts before using it. A nonzero exit or absent output means the run is incomplete. The browser tool described below uses CPU, not WASM.

## Offline SFace batch runner

With Python 3 and OpenCV installed, run `python -B run-sface.py` from this directory. It writes a new timestamped results file without overwriting previous runs. The two tested SFace runs gave the same grouped outcomes. The Python/OpenCV and Node/WASM pipelines have different detectors, image preprocessing, and runtimes; their timing is not an apples-to-apples browser performance benchmark.

Inspect all 40 rows, enrollment, detection and error statuses in the saved
JSON. Exit zero does not guarantee that every image contains exactly one
detected face. Missing or invalid model assets can prevent a result from being
created; check filenames, hashes and installed OpenCV capabilities first.

## Legacy browser and SFace setup

The page first displays the latest saved Face-API WASM and SFace results side by side, with the enrollment image, per-photo scores and thresholds, and filters for intended same- and different-person probes. Summary percentages use only completed matched or not-matched probes as the denominator; unavailable probes are reported separately. These rates describe this unverified synthetic set and are not per-image identity probabilities. The separate optional browser rerun below uses the Face-API CPU backend; it does not rerun the offline WASM batch.

Requires Python 3, NumPy and OpenCV Python with `FaceDetectorYN` and `FaceRecognizerSF` (the implementation machine has Python 3.13 and OpenCV 5.0.0). An existing compatible environment is sufficient; if missing, use a project-local virtual environment and install a suitable pinned OpenCV package and NumPy, without global installation. From the repository root, download public assets once from a network-enabled environment:

```powershell
cd comparison
python -B download.py
python -B server.py
```

Open `http://127.0.0.1:18765/` in a browser to see saved results, or click **Run comparison** to run the optional browser CPU and SFace pipelines. Stop the server with Ctrl+C. Download errors or missing models are shown as pipeline errors. The script does not overwrite existing assets; it reports their SHA-256 and checks the two ONNX hashes. Verify existing Face-API assets independently if their origin is uncertain. No images are modified or sent to external services by the tool.

Check that the result shows one enrollment and 39 probes per pipeline, including separate counts for missing images, zero/multiple detected faces, recognition decisions, processing errors, and unavailable enrollment. The browser results are in memory and are not automatically persisted. If either pipeline fails or the reference is not enrolled, its recognition totals are incomplete. The count of matches is not an accuracy estimate.

## Provenance and behavior

- Browser library and Face-API weights: `@vladmandic/face-api` **1.7.15**, MIT, from [the pinned npm release](https://www.npmjs.com/package/@vladmandic/face-api/v/1.7.15); exact asset URLs are in `download.py`. The bundle includes TF.js and reports its exact version in results. SSD MobileNet V1, 68-point landmarks and face descriptors use Face-API's own preprocessing. It uses the CPU backend, not the WASM backend, so timing must not be interpreted as a WASM comparison. The user's [Face-API-WASM repository](https://github.com/yapweijun1996/Face-API-WASM) documents the same detector/landmark/descriptor pipeline and the 0.6 Euclidean demo threshold.
- YuNet `face_detection_yunet_2023mar.onnx` and SFace `face_recognition_sface_2021dec.onnx` come from [OpenCV Zoo commit 47534e27c9851bb1128ccc0102f1145e27f23f98](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models). YuNet is [MIT](https://github.com/opencv/opencv_zoo/blob/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/LICENSE); SFace is [Apache-2.0](https://github.com/opencv/opencv_zoo/blob/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/LICENSE). Expected SHA-256 values and exact download URLs are in `download.py`. OpenCV's [tutorial](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html) specifies YuNet detection, SFace `alignCrop`, feature extraction, and an example cosine similarity threshold of 0.363. Images are resized to fit within 640 pixels on the longest edge before detection and alignment. YuNet score threshold is 0.9. Multiple detections are excluded rather than silently choosing a face.
- Face-API and SFace scores and thresholds have different meanings. Per-image processing times include image loading, detection, alignment and recognition; total time includes initialization. The browser and local Python processes use different runtimes, so timings are diagnostic only.
- The server binds only `127.0.0.1:18765` and serves a fixed path allowlist. It exposes no upload or arbitrary file endpoint. `results-sface.json` records a separate local SFace run.
