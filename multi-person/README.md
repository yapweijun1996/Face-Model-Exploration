# Synthetic multi-person comparison dataset

For preparation, commands and expected output, use the [multi-person comparison
guide](../comparison/MULTI-PERSON.md). Selected visual observations are in
[REVIEW-NOTES.md](REVIEW-NOTES.md).

This dataset has 20 independently generated fictional adult reference portraits.
For each reference, four individually generated images were requested to show the
same fictional person in different conditions: a head turn, an outdoor smile,
glasses and indoor lighting, and a single-person office activity. The reference
image was supplied to the image generator for each variation. Prompt descriptions
are requests, not verified properties of the generated pictures.

`manifest.csv` assigns an intended identity to each image and indicates whether it
is an enrollment image or a probe. All `identity_review` values are `pending`.
The labels may contain identity drift or accidental resemblances. The images are
synthetic, not independent camera captures of real people. This dataset supports
exploratory testing, not a claim about real-world accuracy, demographic performance,
or liveness. Activity images are single-person images and do not test group-photo
face detection.

For each probe, compare with its own reference and other identities' references.
Always report missing images, detection failures, and incomplete comparisons as
well as scores and thresholds. Comparisons sharing images or identities are not
statistically independent. Do not calibrate a threshold on these same probes and
then claim the tuned result is an independent evaluation.
