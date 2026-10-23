# DASE7506 MP2 - Round 2 frozen submission

**Official test accuracy: 80.94218415417559% (378/467).**
The second research round improved disagreement-validation accuracy from
299/355 to 300/355, but **did not improve the test score**. The report documents
this limited result, rather than claiming that the new architecture is superior.

The submission is the candidate selected by the prospective development rule:
50% normalized round-1 verifier plus 50% of a three-seed listwise MLP ensemble.
No decision to retain or replace a method was based on the new test result.

| Item | Location |
|---|---|
| Final report, at most 10 pages | `output/pdf/report.pdf` |
| Frozen inference assets | `code/runs/model_bundle/` |
| Official test score and predictions | `code/runs/test-score.json`, `test-predictions.json` |
| Exact method and asset identity | `code/runs/freeze.json` |
| Resource measurements and verification | `code/runs/test-resources.json`, `verification.json` |
| Prospective experimental plan | `code/research_plan.json` |
| Complete research records | `code/research_runs/` |
| Prior-round evidence and baseline checkpoint | `history/round1/` |
| AI assistance disclosure | `AI_DISCLOSURE.md` |
| Chinese reading and submission guide | `中文说明.md` |
| Delivery hashes and audit | `DELIVERY_MANIFEST.json` |
| Score and pending publication fields | `SUBMISSION_METADATA.json` |

## Install and reproduce the submitted score

Use **Python 3.12**. The complete ZIP includes the original data. The smaller
code/checkpoint ZIP omits original JSONL and embedding NPZ files: obtain the
[official repository ZIP](https://github.com/xudongwu-0/DASE7506-MP2/archive/refs/heads/main.zip)
and copy its `code/data/` directory into this project. Keep its original manifest
and license files. The official parser, prediction runner and scorer are unchanged.

From the repository root:

```bash
cd code
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements-method.txt
python check_data.py
python evaluate.py --split validation
python evaluate.py --split test
```

Install `requirements-method.txt` **instead of** the preserved upstream
`requirements.txt`; the latter has a different NumPy pin. Our complete tested
inference environment is NumPy 2.1.3, SciPy 1.15.3, scikit-learn 1.6.1, joblib
1.4.2 and threadpoolctl 3.5.0. PyTorch is unnecessary for inference: learned MLP
layers are stored as ordinary uncompressed NumPy arrays and evaluated in NumPy.

`evaluate.py` sets four threads and launches fresh official prediction and
scoring processes serially. The test command checks the frozen hashes first.
Repeated evaluation is for reproduction, not further method selection.

Equivalent official commands, from `code/`:

```bash
python predict.py --method submission.py --checkpoint runs/model_bundle --split test --output runs/test-predictions.json
python score_predictions.py --split test --predictions runs/test-predictions.json --output runs/test-score.json
```

Before direct official commands, set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to `4`. In PowerShell:

```powershell
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:NUMEXPR_NUM_THREADS='4'
```

For unseen audit inputs:

```bash
python predict.py --method submission.py --checkpoint runs/model_bundle --split audit --data-dir /path/to/audit_inputs --output runs/audit-predictions.json
```

No IDs, cached predictions, reference answers, network calls or fixed bank sizes
are used by the predictor. It selects only supplied candidates.

## Reproduce training and development

CPU-only PyTorch was used to train small networks from scratch; no pretrained
weights were downloaded. Install training dependencies into a Python 3.12
environment (the supplied CPU-wheel pin targets Windows/Linux):

```bash
python -m pip install -r requirements-training.txt
python research.py --output tmp/reproduction
```

This reproduces the full round-2 search into a separate directory and does not
modify the submitted assets. The baseline checkpoint in `history/round1/` is
part of the package. Its fitting and original search can independently be
reproduced with:

```bash
python train.py --output-dir tmp/round1-reproduction
```

Do not run `train.py` with its default destination over the submitted model.
Both training programs load only train/validation references, never test data.
The round-2 script deliberately refuses to overwrite a completed research run.

Research protocol:

1. Stratified question-level 80/20 split inside the training bank, seed 20261023.
   Strata are counts of correct candidates. There are 951 holdout questions,
   including 419 disagreement questions; 2,010 informative groups fit inner models.
2. Compare pairwise linear C in {0.001, 0.01, 0.1}; compare BCE, pairwise and
   multi-positive listwise MLP losses, plus a context-augmented listwise MLP.
3. Neural grid: hidden width 64, dropout 0.1, AdamW learning rate 0.001,
   weight decay {0.001, 0.01}, epochs {10,20,40,60}, seeds {7506,7507,7508}.
   All scalers fit only the relevant training partition; all features are clipped
   after standardization to [-8,8]. Hyperparameters use mean inner-holdout results.
4. Refit the selected configuration per family on 2,513 informative training
   groups. Compare normalized seed ensembles and five fusion weights on official
   validation. Promotion requires a strict validation-hard gain, no inner-hard
   reduction relative to a refitted baseline, and improvement in at least two
   of three seed-specific validation fusions.
5. Freeze before the final test. The lone eligible result is listwise MLP fusion
   with new-model weight 0.5, weight decay 0.01 and 10 epochs. Its advantage is
   just one validation question and is not statistically established.

Round 2 fitted 3 inner pairwise-linear models, 24 inner neural training
trajectories (96 epoch evaluations), 13 final refits, plus 2 classifiers in the
inner baseline. It assessed 25 fusion settings, including duplicate baseline
endpoints. Total recorded research runtime was 80.80 seconds, including input
loading, features and exports, excluding interpreter startup. Prior round-1
development recorded 51.85 training/search seconds. These are measured local
costs, not reference-machine guarantees.

Score normalization subtracts each group's mean and divides by a standard
deviation fitted on informative training groups. It is not probability
calibration. The final MLP input is 1,083 dimensions: normalized embeddings
(1,024) plus the original scalar features (59). Each shared candidate scorer is
1083 -> 64 -> 1 with ReLU; dropout is disabled at inference. The three networks
have 69,441 learned layer parameters each. All preprocessing arrays and score
scales are included in the checkpoint.

## Diagnostics and document reproduction

```bash
python verify_round2.py
python analyze_round2.py
python benchmark_baseline.py
```

Verification checks full validation export agreement, empty inputs, unseen IDs,
question/candidate reorderings, variable batch sizes, arithmetic safety and
actual file/network access during inference. It asserts that torch is not
imported for inference. Analysis uses only saved validation labels/scores and
validation input text. Reported validation confidence and significance analyses
reuse development data and do not establish out-of-sample calibration.

For optional figure/PDF regeneration, in a document environment:

```bash
python -m pip install matplotlib==3.10.0 reportlab==4.4.9 pypdf==6.10.0
python code/make_figures.py
python code/make_report.py
python code/package_delivery.py
```

The document commands are run from the repository root. They do not train or
change the selected method. Main evaluation needs none of these document packages.

## Resource limits and provenance

Test inference plus scoring took 3.203 seconds on an AMD Ryzen 7 7745HX under
Windows 11, Python 3.12.14, four threads. Selected assets occupy **1,411,186 bytes
(1.346 MiB)** uncompressed. The five required files are `config.json`,
`legacy.pkl`, `seed7506.npz`, `seed7507.npz`, `seed7508.npz`. All five must be
provided together; an individual NPZ is not a complete checkpoint.

The historical checkpoint is retained for baseline/research reproduction; the
delivery audit also counts historical weights and configuration conservatively.
The reference Xeon machine, not the local CPU, determines the 30-second limit.
The original numeric parser/scorer and all data hashes are verified unchanged.

**Prior test exposure:** round 1's aggregate test result was known before this
user-requested second research round. Round 2 did not use test references,
predictions or scores for fitting or selection. `research_plan.json` and its
timestamped hash were recorded before the search. The final method was frozen
at 2026-10-23 14:50:47 UTC+8, then evaluated once on the test bank. No model was
changed or reverted after observing its score. This is a public-bank repeated
development history, not a claim of a newly untouched external test set.

## Final submission

Local artifacts are complete; **no GitHub submission issue has been created**.
The student ID and public repository/checkpoint links are still needed. Review
the report and understand the implementation and AI disclosure before submitting.

Publish the repository including report, experiment records, `runs/model_bundle`,
official score and hashes. Record the exact repository commit. Use a permanent
commit link for code and a matching checkpoint directory/release link. Enter
**80.94218415417559**, as a percentage, on the
[course website](https://xudongwu-0.github.io/courses/dase7506/mp2/), along with
the student ID and artifact links, then actually create the prepared GitHub
issue. Deadline: **31 October 2026, 23:59, Hong Kong/Beijing time (UTC+8)**.

Official source: [DASE7506-MP2](https://github.com/xudongwu-0/DASE7506-MP2).
Retain `code/data/LICENSE.txt` and `NOTICE.txt` when redistributing the supplied
GSM8K data and Qwen-generated trace/embedding package. The original package
manifest describes the upstream submission stub; the new freeze manifest and
delivery manifest identify the submitted implementation.
