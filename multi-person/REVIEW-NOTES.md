# Exploratory visual review

These notes cover four selected comparisons from the run dated 2026-10-27;
they are not a review of all images or comparisons. Enrollment portraits were
generated independently and variations were generated from their respective
reference image. Intended identity labels remain pending: visual inspection and
model scores cannot establish identity or eliminate generator drift.

| Probe | Reference | Face-API distance / decision | SFace similarity / decision | Visual observation |
| --- | --- | --- | --- | --- |
| person-01-02 | person-15-01 | 0.3026 / matched | 0.4643 / matched | The portraits have similar-looking facial features, but different hair and framing. Their independently generated labels remain unverified. |
| person-01-04 | person-03-01 | 0.3823 / matched | 0.3573 / not matched | Hair, glasses, pose and lighting differ; the SFace score is close to its 0.363 example threshold. |
| person-06-02 | person-05-01 | 0.9092 / not matched | 0.4581 / matched | The portraits have markedly different visible facial appearance and hair; SFace's match warrants inspection. |
| person-18-03 | person-03-01 | 0.7242 / not matched | 0.4820 / matched | The portraits have visibly different appearance, hair and backgrounds; SFace's match warrants inspection. |

These decisions use the example thresholds Face-API distance <= 0.6 and SFace
cosine similarity >= 0.363. The thresholds are not calibrated to this set. All
interpretations are exploratory and should not be generalized to real people,
demographics, independent accuracy, or a security decision.
