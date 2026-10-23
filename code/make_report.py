"""Build the eight-page report from frozen records; no fitting or evaluation."""
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/pdf/report.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
read = lambda p: json.loads((ROOT/p).read_text())
analysis = read('code/runs/analysis.json')
selection = read('code/research_runs/selection.json')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Body', fontName='Helvetica', fontSize=10.1, leading=13.5, spaceAfter=7))
styles.add(ParagraphStyle(name='SmallText', parent=styles['Body'], fontSize=8.7, leading=11.4, spaceAfter=6))
styles.add(ParagraphStyle(name='PageTitle', fontName='Helvetica-Bold', fontSize=20, leading=25, textColor=colors.HexColor('#123952'), spaceAfter=16))
styles.add(ParagraphStyle(name='Sub', fontName='Helvetica-Bold', fontSize=12, leading=16, spaceBefore=8, spaceAfter=7))
story=[]
def p(text, style='Body'): story.append(Paragraph(text, styles[style]))
def title(n,text):
    if n>1: story.append(PageBreak())
    p('DASE7506 / MINI-PROJECT 2 / RESEARCH ROUND 2', 'SmallText')
    p(text,'PageTitle')
def sub(text): p(text,'Sub')
def table(rows,widths):
    cells=[[Paragraph(str(v),styles['SmallText']) for v in row] for row in rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eff4')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#7f9caf')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dce3e8'))]))
    story.extend([t,Spacer(1,10)])
def figure(name,w,h,caption):
    story.append(Image(str(ROOT/'output/figures'/name),width=w,height=h))
    p(caption,'SmallText')

title(1,'Selecting reliable LLM reasoning')
p('<b>Research question:</b> Can a ranking objective and comparison between candidates improve an existing small verifier enough to justify added computation?')
p('The task supplies four completed solutions and fixed 1,024-dimensional representations per question. The selector chooses one candidate; it does not train or query the language model that generated the solutions. Correctness means the final numeric answer matches the supplied reference, not that every reasoning step is valid [1, 2].')
sub('Outcome')
table([['Method','Validation: all','Validation: disagreement','Official test'],['Prior verifier','742/798 (92.98%)','299/355 (84.23%)','378/467 (80.94%)'],['Frozen round-2 fusion','743/798 (93.11%)','300/355 (84.51%)','378/467 (80.94%)']],[128,110,143,126])
p('The selected method combines the prior verifier with three small, independently initialized listwise MLPs. Its validation gain is one question. Eight previous mistakes are corrected and seven previously correct choices become wrong. The final test accuracy is unchanged. These observations do not establish a reliable accuracy advantage for the more complex model.')
p('Complete fresh-process test prediction plus independent scoring took <b>3.203 seconds</b> locally with four CPU threads. Required inference assets total <b>1,411,186 bytes (1.346 MiB)</b>. Both are within the local budget checks; eligibility on the instructor\'s reference CPU remains subject to reproduction.')
sub('Hypothesis and evidence')
p('The hypothesis was that optimizing relative scores within a question would help choose among plausible but conflicting answers, and that candidate context would add useful evidence. Standalone ranking models did not beat the prior verifier. A 50/50 score fusion was the only candidate satisfying the recorded promotion rule. Context augmentation did not improve this search. The results support retaining a strong baseline and treating small validation gains cautiously.')
sub('Development history and scope')
p('The prior round\'s aggregate test score was known before this user-requested follow-up. A bounded round-2 plan was recorded before training; selection used training holdout and official validation only. The artifact was frozen on 23 October 2026 at 14:50:47 UTC+8, then tested once. No method was changed or reverted after that score. This disclosure is material: this is repeated development on a public course bank, not a newly untouched external test.')
p('All numerical experiment results in this report come from executed local records. Interpretations and future proposals are identified as such. Code, checkpoints, records, AI disclosure and reproduction instructions accompany this report.','SmallText')

title(2,'Data, supervision and evaluation protocol')
table([['Bank','Questions','Permitted use'],['Training','4,755','Fit features, transformations and model parameters'],['Validation','798','Development, method selection and descriptive analysis'],['Test','467','Frozen final scoring and reproduction only']],[95,80,332])
p('Each question has four candidates and at least one correct answer. Several candidates may be correct. Test groups contain at least two distinct valid parsed answers. Train and validation also contain easier groups [1]. We define the disagreement subset using distinct parsed answers, without reference labels. It contains 355 validation questions; full validation accuracy is inflated by easy agreement groups. Both metrics are reported, with disagreement accuracy driving development.')
sub('Internal training split')
p('An 80/20 question-level split of training uses seed 20261023, stratified by the number of correct candidates. There are 951 internal holdout questions, including 419 disagreement questions. The fitting subset excludes all-positive groups, leaving 2,010 informative questions; full-training refits use 2,513 informative groups. Candidates from one question never cross this internal boundary. Removing all-positive groups focuses ranking on comparisons with incorrect alternatives, but also changes the training distribution.')
p('Inner models fit their scalers and any PCA only on their own fitting partition. The prior verifier is refitted on this same partition to form the internal baseline. This avoids comparing new models against a baseline trained on the internal holdout. Final refits use only the supplied training bank. The original full-training verifier remains the official validation comparator.')
sub('Selection sequence')
p('1. Record the finite search plan and SHA-256 before training. 2. Select each family\'s regularization and epoch using mean internal disagreement accuracy across three seeds, then overall accuracy; exact ties retain the first declared setting. 3. Refit those configurations on training. 4. Compare score fusions on official validation under the prespecified promotion rule. 5. Freeze sources and assets before official testing.')
p('The internal holdout is itself reused across a search, and official validation selects a fusion. Neither is an independent confirmatory dataset. Bootstrap intervals and confidence diagnostics later in this report are descriptive and do not correct for selection. Only one post-freeze test run was used in this round; test errors were not used to redesign the method.')
sub('Rules enforced')
p('Only supplied data and fixed embeddings are used. There are no external datasets, pretrained neural weights, inference API calls, answer tables keyed by IDs, cached-prediction shortcuts, or new answer generation. The original data, numeric parser, prediction runner and scorer are unchanged. Inference takes arbitrary new bank inputs and returns candidate indices 0-3. Training was CPU-only; evaluation used four threads. All learned transformations, scales, ensemble members and configuration files are counted as assets [1].')

title(3,'Baseline, features and learning objectives')
sub('Prior verifier and feature information')
p('The prior verifier combines a histogram gradient-boosted tree (weight 0.25) with embedding logistic regression (weight 0.75). The tree uses 59 scalar features plus a 32-component PCA of group-relative embeddings. It uses 160 boosting iterations, learning rate 0.06, 15 leaves, minimum leaf size 30 and L2 regularization 10. Logistic regression uses standardized normalized embeddings with C=0.1. Full source and the original search records are retained.')
p('Scalar features summarize parsed answers, trace length/formatting, bounded arithmetic checks, and comparisons with the other candidates. They can detect agreement or local calculation consistency, but do not constitute a proof checker. Embeddings are normalized per candidate; the context variant additionally includes the embedding minus its group mean. This contextual vector is a fixed transformation, not an attention model.')
table([['Family','Scoring / training design'],['Pairwise linear','One shared linear score; logistic loss on correct-minus-incorrect feature differences; both signs, group-balanced weights.'],['MLP: BCE','Shared 1083 -> 64 -> 1 network; independent binary cross entropy for each candidate.'],['MLP: pairwise','Same network; average softplus(s_wrong - s_correct) within each group.'],['MLP: listwise','Same network; logsumexp(all scores) minus logsumexp(correct scores).'],['Context listwise MLP','2107 -> 64 -> 1; adds 1,024 group-relative embedding dimensions; same listwise loss.']],[136,371])
p('The 1,083-dimensional ordinary input concatenates normalized embeddings (1,024) and scalar features (59). Every new model standardizes features using training-only means/scales and clips them to [-8,8]. Neural scorers use one ReLU hidden layer with 64 units and dropout 0.1 during training. The final network has 69,441 trainable layer parameters per seed. All networks are trained from scratch.')
sub('Why the objectives differ')
p('BCE learns absolute candidate correctness. Pairwise loss explicitly orders every correct/incorrect pair and weights questions equally. The multi-positive listwise objective maximizes probability mass assigned to any correct candidate, matching the task\'s acceptance of multiple correct answers. It need not distribute that mass equally among correct candidates. This alignment is a hypothesis about useful training signals, not evidence that listwise loss must win.')
p('Invalid parsed answers are masked at selection. The maximum-scoring valid candidate is returned; exact ties follow candidate order. Because no new solution can be generated, the method\'s scope is choosing among supplied alternatives. Correct-answer selection can still accept flawed reasoning that happens to produce the correct number.')

title(4,'Bounded search and frozen decision')
table([['Search dimension','Declared values'],['Linear regularization C','0.001, 0.01, 0.1'],['Neural families','BCE, pairwise, listwise, context listwise'],['Weight decay / epoch checkpoints','0.001 or 0.01 / 10, 20, 40, 60'],['Seeds / optimization','7506, 7507, 7508 / AdamW, lr 0.001, batch 128 groups'],['Fixed architecture','64 ReLU units, dropout 0.1, gradient norm clipped at 5'],['New-model fusion weight','0, 0.25, 0.5, 0.75, 1']],[175,332])
p('This search fitted three inner linear models and 24 inner neural training trajectories. Four epoch checkpoints per trajectory give 96 neural evaluations, not 96 separate fits. It then fitted 13 full-training models (one linear plus three seeds in each neural family), in addition to the two classifiers used for the internal baseline. All 25 family/weight fusion settings, including duplicate baseline endpoints, are recorded.')
sub('Promotion rule recorded before the search')
p('A candidate must strictly improve official validation disagreement accuracy, must not reduce internal disagreement accuracy relative to the refitted prior verifier, and must improve official validation in at least two of its three seed-specific fusions (one for the deterministic linear family). Eligible candidates are ranked by disagreement accuracy, then overall accuracy, then recorded new-member file bytes. Only one candidate qualified, so the final tie-break had no effect.')
sub('Selected model')
p('<b>Three listwise MLPs, weight decay 0.01, 10 epochs, fused at new-model weight 0.5.</b> For each component, subtract the mean score within its four-candidate group, then divide by a score standard deviation learned on informative training groups. Average the three normalized neural scores. The final score is 0.5 times the normalized prior-verifier score plus 0.5 times this neural average. All scale values are checkpointed; no evaluation labels enter normalization.')
p('The selected fusion matches the internal baseline at 341/419 (81.38%). Its three seed-specific validation fusions score 84.79%, 83.94% and 84.79%; two exceed the prior 84.23%. The three-seed fusion reaches 300/355 (84.51%). This seed gate reduces reliance on one initialization, but does not guarantee robustness to a different dataset or split.')
sub('Search cost and reproducibility')
p('Recorded round-2 wall time was 80.80 seconds, including input loading, feature extraction, fitting and exports, excluding interpreter startup. The prior round recorded 51.85 seconds of training/search. These figures exclude dependency installation, verification, report generation and human/AI development time. The machine was an AMD Ryzen 7 7745HX, Windows 11, Python 3.12.14, CPU-only PyTorch 2.9.0+cpu with four threads. Deterministic algorithms and explicit seeds were enabled; platform-level numerical differences remain possible.')
p('The saved plan, start timestamp/hash, inner results, selected configurations, refit results and fusion results are retained. Nonselected checkpoints are not needed for evaluation and are omitted from the delivery; the training script regenerates them. The selected checkpoint and historical baseline are included in full.')

title(5,'Comparisons and ablation results')
rows=[['Method (standalone unless noted)','Internal hard %','Validation hard %','Validation all %'],['Prior verifier','81.38','84.23','92.98']]
names={'pair_linear':'Pairwise linear','mlp_bce':'MLP: BCE','mlp_pairwise':'MLP: pairwise','mlp_listwise':'MLP: listwise','context_listwise':'Context MLP: listwise'}
for r in analysis['family_results']:
    rows.append([names[r['family']],f"{100*r['ensemble_internal_hard']:.2f}",f"{100*r['ensemble_hard']:.2f}",f"{100*r['ensemble_validation']:.2f}"])
rows.append(['Selected 50/50 fusion','81.38','84.51','93.11'])
table(rows,[201,102,102,102])
p('Neural rows are averages of normalized scores across three seeds, using each family\'s internally selected configuration. Internal results are development results on 419 disagreement groups; official validation has 355. They are different question sets, so their percentages should not be interpreted as a learning curve. All standalone new families underperform the prior verifier on official validation.','SmallText')
figure('fusion.png',507,225,'Figure 1. Fusion weight ablation on official validation. The dashed line is the prior verifier. More weight on the new model generally fails to help; the selected listwise blend has only a marginal gain.')
p('<b>Matched objective/context ablation:</b> at weight decay 0.01 and 10 epochs, mean internal disagreement accuracy across seeds is 79.08% (BCE), 80.19% (pairwise), 79.95% (listwise), and 79.16% (context listwise). Holding these settings fixed isolates the changed objective/context more closely than the independently tuned family table. The tested context addition did not help; this does not rule out better relational architectures.')
p('<b>Final test:</b> the frozen fusion scores 378/467 (80.94218415417559%), identical in aggregate to the prior round. We did not inspect test errors to choose another method. The evidence does not justify claiming a test improvement or a universally superior selector.')

title(6,'Reliability, uncertainty and failure cases')
p('The selected fusion fixes eight prior errors and introduces seven on disagreement validation. The net gain is +0.282 percentage points. A paired question bootstrap (10,000 resamples, seed 20261023) gives a 95% interval of <b>-1.97 to +2.54 points</b>; exact McNemar p=1.0. These calculations reuse selection data and ignore the search, so they are descriptive, not a confirmatory significance claim. Even this optimistic comparison provides no clear evidence of superiority.')
figure('coverage.png',507,182,'Figure 2. Descriptive selective-error curve on 355 disagreement-validation questions. Confidence is the gap between the best scores associated with distinct parsed answers. Scores are not calibrated probabilities.')
p('The highest-margin 89 questions have 87 correct selections (97.75%); at all 355 questions accuracy falls to 84.51%. This suggests a useful uncertainty ordering on development data. It does not prove calibration or support abstaining in the submitted task, which requires one choice for every question. Thresholds were not tuned on test data.')
table([['Validation example','Observed choice and interpretation'],['Corrected: overtime pay','40 hours at $12 plus overtime at $18; total $696. The fusion selects 52 hours, replacing 184. This is consistent with improved quantity interpretation, but the score alone cannot prove what the model understood.'],['Regressed: naps','3 naps/week, 2 hours/nap, over 70 days gives 60 hours. The prior method selects 60; the fusion selects 420. A unit/time conversion error remains plausible to the model.'],['Confident shared error: attendance','250 students, 140 girls present, twice as many girls present as boys. Correct absent boys = 110 - 70 = 40. Both methods select 180 despite a high distinct-answer margin (2.197). Three candidates repeat the wrong answer.']],[130,377])
p('These examples were selected after validation analysis and are illustrative, not representative frequency estimates. They show why agreement, local arithmetic and fluent structure can support a wrong interpretation. The attendance failure is consistent with a consensus-related weakness; it does not establish a causal feature attribution. More targeted training-only perturbations would be needed to test that explanation.')

title(7,'Resources and reproduction')
table([['Measurement','Prior verifier','Round-2 fusion'],['Validation full evaluation, fresh processes','3.414 s','3.474 s'],['Official test full evaluation','2.98 s (prior record)','3.203 s'],['Required inference assets','546,945 bytes','1,411,186 bytes'],['Local thread setting','4','4']],[221,143,143])
p('Validation timings were measured in this round using the same local hardware and include prediction startup, imports, asset/data loading, feature extraction, every ensemble member and independent scoring. Test timings come from separate rounds. Each is a single observation; the small differences do not establish a stable speed ranking. The new checkpoint is about 2.58 times the old size with no demonstrated test gain. The 30-second reference limit is determined on the instructor\'s Xeon Platinum 8457C [1].')
p('The five required files are config.json, legacy.pkl and seed7506.npz / seed7507.npz / seed7508.npz. NPZ files are uncompressed; their array payload totals 859,296 bytes. The 1,411,186-byte total also includes learned preprocessing, score scales and the legacy member. Counting the retained historical baseline as well gives 1,958,131 bytes (1.867 MiB), still below 16 MiB. Research logs/predictions are evidence, never inference inputs.')
sub('Reproduce the frozen score')
p('Use Python 3.12. From the repository root, enter code/, create and activate a virtual environment, then run the following commands. The full distribution includes the unchanged official data; the lightweight distribution requires copying code/data/ from the official repository first.','SmallText')
for cmd in ['python -m pip install -r requirements-method.txt','python check_data.py','python evaluate.py --split validation','python evaluate.py --split test']:
    p(cmd,'SmallText')
p('The evaluator sets four library threads and times fresh official prediction/scoring processes. It checks frozen source and asset hashes before test evaluation. Exact equivalents using predict.py and score_predictions.py, including thread environment variables and unseen audit-bank input instructions, are in README.md. CPU inference uses NumPy and scikit-learn; PyTorch is not imported.')
sub('Reproduce training and check the interface')
for cmd in ['python -m pip install -r requirements-training.txt','python research.py --output tmp/reproduction','python train.py --output-dir tmp/round1-reproduction','python verify_round2.py']:
    p(cmd,'SmallText')
p('The first research command rebuilds round 2 using the supplied historical baseline; train.py separately reproduces that baseline search. These destinations preserve the submitted assets. Verification passed full validation export agreement, empty inputs, unseen IDs, reordered questions/candidates, variable batches, bounded arithmetic, and output types. A runtime audit hook checks against reference/cache reads and network access on the tested calls; these checks are evidence about covered cases, not a formal proof for every possible input.','SmallText')
p('Dependencies are pinned in requirements-method.txt and requirements-training.txt. Frozen source/asset SHA-256 values are in code/runs/freeze.json; official JSON and timings are in code/runs/. The delivery manifest identifies the final documentation/package version.','SmallText')

title(8,'Limitations, disclosure and submission record')
sub('What remains unresolved')
p('This study examines a small grid, three neural seeds and one internal split. Stronger regularization, training-only semantic perturbations, group-equivariant attention, or learned routing among verifiers could potentially help, but none was trained or shown to improve here. A larger search would increase validation overfitting risk and development cost. It should be evaluated under a new prespecified protocol, without selecting from repeated test scores.')
p('The tested models use fixed embeddings and shallow text-derived features. They cannot reliably verify all mathematical steps or resolve every missing condition. A candidate may pass local arithmetic checks while answering the wrong question. The scored target is final-answer correctness, so results must not be presented as a guarantee of logically sound reasoning. The validation margin analysis is not out-of-distribution calibration.')
p('Prior test aggregate exposure, repeated validation reuse and the small one-question gain limit generalization claims. The final frozen model follows the prospective promotion rule; its unchanged test score is retained transparently. The increased model complexity has not produced an established practical accuracy benefit on this bank.')
sub('AI assistance and reuse')
p('OpenAI Codex provided substantive help with assignment interpretation, experiment and feature design, implementation, CPU training/evaluation, verification, analysis, report writing, figures and packaging. Assistance was not limited to proofreading. The student should understand, review and reproduce the work before submitting; this report does not assert that student review has already occurred. AI_DISCLOSURE.md records the same scope.')
p('The project reuses the course data, fixed embeddings and official evaluation tools, plus NumPy, SciPy, scikit-learn and CPU PyTorch. Matplotlib and ReportLab generate presentation artifacts. The prior verifier is original earlier-round work supplied with its training code. No external training dataset or additional pretrained weights are used. The provided GSM8K MIT license and Qwen-generation attribution are retained [1, 2].')
sub('Artifact and submission status')
p('The frozen method is identified by code/runs/freeze.json, timestamp 2026-10-23T06:50:47.911503+00:00. Its matching official accuracy is <b>80.94218415417559</b>, on the required 0-100 scale. All five inference assets must accompany the code. Final publication must supply a public repository with this report, exact code commit and a matching checkpoint/configuration link.')
p('<b>No submission issue has been created.</b> Student ID and public artifact URLs remain to be supplied. By 31 October 2026, 23:59 UTC+8, enter these with the score on the course website and actually create the GitHub issue it prepares. A prepared link or local ZIP alone is not a completed submission. The course subsequently provides a seven-day peer-review period [1].')
sub('Primary references and local evidence')
p('[1] DASE7506 MP2 guide: <link href="https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md" color="#146c94">Task, constraints and assessment</link>. [2] Official <link href="https://github.com/xudongwu-0/DASE7506-MP2/blob/main/code/README.md" color="#146c94">installation and evaluation instructions</link> and code/data/manifest.json. The guide and original attribution are included in this package.','SmallText')
p('Experiment evidence: code/research_plan.json; code/research_runs/{started,inner_results,inner_selection,refit_results,fusion_results,selection}.json; code/runs/{analysis,verification,freeze,test-score,test-resources}.json; history/round1/experiments.json. Figures are generated directly from saved validation records.','SmallText')

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#d8e1e8'));canvas.line(44,39,A4[0]-44,39)
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#516472'))
    canvas.drawString(44,26,'DASE7506 MP2 | Frozen round-2 research | 23 October 2026')
    canvas.drawRightString(A4[0]-44,26,str(doc.page))

doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=38,bottomMargin=53,title='Selecting Reliable LLM Reasoning - Round 2',author='DASE7506 MP2 submission; AI assistance disclosed')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
pages=len(PdfReader(OUT).pages)
assert pages==8, f'Expected 8 pages, got {pages}; inspect layout.'
print(f'{OUT}: {pages} pages')
