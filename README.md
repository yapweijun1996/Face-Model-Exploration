# Face Model Exploration

This repository compares four local face-recognition pipelines on synthetic
portraits. It is an exploratory inference and reporting project: it does not
train or fine-tune a model and is not a production identity-verification system.

## Pipelines

| Pipeline | Runtime | Comparison |
| --- | --- | --- |
| Face-API | Node.js and TensorFlow.js WASM | Euclidean distance; example threshold 0.6 |
| SFace | Python and OpenCV | Cosine similarity; example threshold 0.363 |
| EdgeFace XS | Node.js and ONNX Runtime CPU | Cosine similarity; no match threshold |
| AuraFace | Node.js and ONNX Runtime CPU | Cosine similarity; no match threshold |

Face-API uses SSD MobileNet V1 detection and 68-point landmarks. The other
pipelines use YuNet detection; the EdgeFace and AuraFace experiment uses
five-point alignment before inference. Detection, preprocessing and runtimes
differ, so model scores and recorded timings are not directly interchangeable.
The optional Face-API rerun in the browser uses CPU, not WASM.

## Guides and saved results

- [Single-person comparison and setup](comparison/COMPARISON.md): 40 images,
  Face-API WASM and SFace results, and local server instructions.
- [Multi-person dataset](multi-person/README.md) and
  [comparison guide](comparison/MULTI-PERSON.md): 20 intended identities and
  100 images, with each probe compared against every enrollment.
- [EdgeFace and AuraFace guide](comparison/exploratory-onnx/README.md):
  model assets, licensing, preprocessing and four-model reports.
- [Business-use candidates and staged testing](comparison/BUSINESS-MODEL-TESTING.md):
  licensing evidence, platform limitations and a plan for later testing.
- [Visual review notes](multi-person/REVIEW-NOTES.md): observations on four
  selected pairs; the notes do not verify dataset identities.

Start with the single-person guide for the existing SFace smoke test. Use the
multi-person guide for broader exploratory comparisons, then the ONNX guide for
the separate EdgeFace and AuraFace experiment. The ONNX runner requires both
models; an AuraFace-only runner is not implemented. Setup may require network
access; the documented offline batch inference uses local assets.

Saved results are exploratory observations, not validated accuracy or security
measurements. Intended identity labels remain unverified, and comparisons
reuse images and identities. No liveness or spoof-resistance test is included.
Model files and installed dependencies are not committed; obtain them separately
as described in the guides. The local server has no user authentication and is
not suitable as a security boundary for sensitive biometric data.

## Dataset overview

This folder contains 30 separately generated PNG images, `face-01.png` through
`face-30.png`, intended to depict the same fictional adult. `face-01.png` is the
reference; the other images were generated using that reference. These are
synthetic variants, not independent camera captures of a real person. Ten
additional independent images, `negative-01.png` through `negative-10.png`,
are intended to depict ten other fictional people.

The built-in ImageGen tool was used. The reference prompt requested a natural,
photorealistic head-and-shoulders portrait of one fictional adult with an
unobstructed face. Each subsequent prompt requested the same facial identity
while varying pose, expression, lighting, glasses, clothing, or background.
The ten negative samples were generated separately without the first person's
reference image. The requested variation for each image is recorded in
`manifest.csv`; it is a prompt description, not a verified measurement of the
resulting image.

The CSV contains one row per image with `filename`, `intended_identity`,
`synthetic`, `requested_variation`, and `identity_review`. The shared identity
is an intended label, not a verified ground truth; all reviews are `pending`.
Review images and their intended identity labels before interpreting results.

The ten other intended identities are too few to estimate a reliable
false-accept rate. See the guides above for execution, interpretation and
the larger synthetic dataset.
