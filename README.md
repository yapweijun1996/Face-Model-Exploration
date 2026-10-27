# Synthetic Face Recognition Spike Dataset

This folder contains 30 separately generated PNG images, `face-01.png` through
`face-30.png`, intended to depict the same fictional adult. `face-01.png` is the
reference; the other images were generated using that reference. These are
synthetic variants, not independent camera captures of a real person. Ten
additional independent images, `negative-01.png` through `negative-10.png`,
are intended to depict ten other fictional people.

## Generation

The built-in ImageGen tool was used. The reference prompt requested a natural,
photorealistic head-and-shoulders portrait of one fictional adult with an
unobstructed face. Each subsequent prompt requested the same facial identity
while varying pose, expression, lighting, glasses, clothing, or background.
The ten negative samples were generated separately without the first person's
reference image. The requested variation for each image is recorded in
`manifest.csv`; it is a prompt description, not a verified measurement of the
resulting image.

## Manifest

The CSV contains one row per image with `filename`, `intended_identity`,
`synthetic`, `requested_variation`, and `identity_review`. The shared identity
is an intended label, not a verified ground truth; all reviews are `pending`.
Review images and their intended identity labels before interpreting results.

## Suggested spike

An initial smoke test can use the reference for enrollment and the remaining
reviewed variants as probes. Record face detection failures, match distances,
thresholds, device, backend, model version, and elapsed time. Initial Face-API
WASM and SFace smoke tests, commands, and per-image results are documented in
`comparison/COMPARISON.md`; the labels have not been independently verified.
Images share a common generated reference, so this set is not
an independent real-world sample and cannot establish real-world accuracy.

The ten other intended identities allow an initial different-person smoke
test, but they are too few to estimate a reliable false-accept rate. A
meaningful comparison also needs more verified different-person and
unknown-person examples, plus consented real-world samples representative of
the intended use. This image set does not test liveness or resistance to
spoofing.
