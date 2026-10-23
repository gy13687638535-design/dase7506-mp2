# AI assistance and reuse disclosure

OpenAI Codex provided substantive assistance in both research rounds: assignment
interpretation, model/feature and experiment design, implementation, CPU training
and evaluation, verification, analysis, report writing, figures and packaging.
This assistance is not limited to proofreading. The student should review,
understand and reproduce the work before submitting; this document does not
assert that such review has already occurred.

Actual executed experiment records support the reported numbers. Round 1's
aggregate test score was already visible when the student requested further
research. Round 2 used only training references and official validation for
development, recorded a prospective plan, froze a new artifact and then ran
one final official test evaluation. The unchanged test accuracy is reported
without claiming a gain, and no method was selected by comparing test scores.

The implementation reuses the supplied DASE7506 data, fixed embeddings, numeric
parser, data checker, prediction runner and scorer. It also uses NumPy, SciPy,
scikit-learn and CPU-only PyTorch, with Matplotlib/ReportLab for presentation.
The original round-1 verifier, whose source and training routine are supplied,
is reused as a baseline and ensemble member. Neural networks are trained from
scratch; no external datasets, pretrained weights or inference API calls are
used. The supplied GSM8K license and Qwen-generation attribution are retained.
