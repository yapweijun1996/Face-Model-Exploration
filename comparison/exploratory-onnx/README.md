# Local EdgeFace and AuraFace exploratory comparison

This independent pipeline reads `../multi-person/manifest.csv` and the prior `multi-person-results/multi-comparison-2026-10-27T09-34-58-670808Z.json`. It verifies that their manifest SHA-256 values match. It never writes to either dataset or prior results. A run creates a fresh directory under `../exploratory-runs/` containing `report.json` and `report.html`; open the HTML directly to review pairs of local photos. It includes prior scores, new cosine scores, detection/inference errors and score distributions. Embeddings and crops are kept in memory and are not saved. No network access is used during inference.

Intended same/different labels in this synthetic dataset are unverified, not ground truth. Cosines are not probabilities. This experiment makes no match decision for the two new models and cannot establish accuracy, liveness or security. Previous engine decisions and thresholds remain in the original report. Face-API distance and the cosine scores have different scales. The new models' ONNX inference call timings exclude detection, decoding, alignment and model loading, include a cold first inference and cannot be compared directly to the earlier end-to-end pipeline times.

## Assets and licensing

Place only the following files in `../vendor/models/`. All three hashes are verified before a run; a different same-named model is rejected. Downloads and dependencies must be obtained separately in an environment with access; this program never downloads them.

* `edgeface_xs_gamma_06.onnx`, SHA-256 `9206e2eb13a2761d7b5b76e13016d4b9acd3fa3535a9a09939f3adacd139a5ff`: [pinned third-party UniFace export](https://huggingface.co/yakhyo/uniface-weights/resolve/13f64660d271d74aa7cc2e86c43d39da551848a5/edgeface_xs_gamma_06.onnx). The [official EdgeFace XS model card](https://huggingface.co/Idiap/EdgeFace-XS-GAMMA) specifies CC BY-NC-SA 4.0; this is non-commercial and share-alike. The ONNX conversion is not an official Idiap release, and its equivalence to the original checkpoint has not been independently validated here. Verify the conversion provenance and applicable rights before use or redistribution. The [official code](https://github.com/otroshi/edgeface) and its license do not automatically establish a more permissive weight license.
* `glintr100.onnx`, SHA-256 `a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60`: [official AuraFace v1, pinned revision](https://huggingface.co/fal/AuraFace-v1/resolve/686774cad65e40933022896195e07e01ee06bee9/glintr100.onnx). The [pinned model card](https://huggingface.co/fal/AuraFace-v1/blob/686774cad65e40933022896195e07e01ee06bee9/README.md) declares Apache-2.0; the previously linked `LICENSE.md` is absent at that revision. Resolve the license and review the model-card data and fairness caveats before use or redistribution. Do not assume other assets in that repository have the same rights.
* `face_detection_yunet_2023mar.onnx`, SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`: already used by the existing comparison; [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and its model license apply.

## Runtime and preprocessing

Use Node.js 22 or later and exactly `onnxruntime-node@1.22.0` for CPU inference. The full 100-image run was verified with Node.js 25.2.1; `package-lock.json` records transitive dependencies. The Python helper uses OpenCV's YuNet detector and NumPy for image decoding and alignment only; Python packages are not locked and their actual versions are recorded. The tested environment had Python 3.13.7, OpenCV 5.0.0 and NumPy 2.3.5. Execution metadata includes all model hashes, manifest and prior-result hashes, Node, runtime, Python, OpenCV and NumPy versions.

The manifest hash verifies the CSV bytes, not the contents of the referenced
images. The earlier result is identified by its recorded hash, not checked
against an independent trusted hash; changing an image without changing the
manifest can make prior and new scores refer to different image contents.

YuNet uses score 0.9, NMS 0.3, top-k 5000 and maximum dimension 640, requiring exactly one face. Its five landmarks are mapped by a least-squares similarity transform to the [ArcFace 112-pixel template](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py); OpenCV uses linear interpolation and black borders. Both models receive RGB float32 NCHW, `(pixel - 127.5) / 127.5`. This is consistent with the [EdgeFace export's preprocessing](https://github.com/yakhyo/uniface/blob/main/uniface/recognition/edgeface.py) and [recognition alignment flow](https://github.com/yakhyo/uniface/blob/main/uniface/recognition/base.py), and AuraFace's ArcFace usage. The [original EdgeFace example](https://github.com/otroshi/edgeface/blob/main/face_alignment/align.py) uses MTCNN; substituting YuNet is an experimental change, and numerical parity with that example remains unverified. YuNet detection scores are detector scores, not probabilities.

## Run

From this directory after obtaining the assets and dependencies:

```powershell
npm ci --ignore-scripts --no-audit --no-fund
node run.js
```

The report path is printed. Exit 1 means setup failed; exit 2 means a report was produced with detection or inference errors. A missing or mismatched asset prevents a run before output creation. The pipeline is deliberately fixed to the existing manifest and prior run; inspect their provenance before relying on the comparison. Run focused tests without model files with `node --test test.js` and `python -B test_prepare.py`.
