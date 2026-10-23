"""Regenerate descriptive figures from recorded validation experiments only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
read = lambda p: json.loads((ROOT / p).read_text())
rows = read('code/research_runs/fusion_results.json')
labels = {'pair_linear':'Pairwise linear', 'mlp_bce':'MLP: BCE',
          'mlp_pairwise':'MLP: pairwise', 'mlp_listwise':'MLP: listwise',
          'context_listwise':'Context MLP: listwise'}
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, ax = plt.subplots(figsize=(7.1,3.15), layout='constrained')
for family, label in labels.items():
    r = [x for x in rows if x['family']==family]
    ax.plot([x['new_weight'] for x in r], [100*x['hard_accuracy'] for x in r], 'o-', label=label, markersize=4)
ax.axhline(100*299/355, color='#555555', linestyle='--', linewidth=1)
ax.set(xlabel='Weight of new model (0 = prior verifier)', ylabel='Disagreement validation accuracy (%)', xticks=[0,.25,.5,.75,1])
ax.grid(alpha=.15)
ax.legend(loc='center left', bbox_to_anchor=(1, .5), frameon=False, fontsize=9)
fig.savefig(OUT/'fusion.png', dpi=190)
plt.close(fig)
rows = read('code/runs/analysis.json')['coverage']
fig, ax = plt.subplots(figsize=(7.1,2.55), layout='constrained')
ax.plot([100*r['coverage'] for r in rows], [100*(1-r['accuracy']) for r in rows], 'o-', color='#146c94')
for r in rows:
    first = r['coverage'] < .3
    ax.annotate(f"{r['correct']}/{r['n']} correct", (100*r['coverage'],100*(1-r['accuracy'])), xytext=(5 if first else -5,10), textcoords='offset points', ha='left' if first else 'right', fontsize=9)
ax.set(xlabel='Coverage: highest margin questions retained (%)', ylabel='Selection error (%)', ylim=(0,20), xlim=(20,105))
ax.grid(alpha=.15)
fig.savefig(OUT/'coverage.png', dpi=190)
plt.close(fig)
print(OUT)
