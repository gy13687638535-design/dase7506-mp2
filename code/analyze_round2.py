"""Post-freeze descriptive analysis of development data, never test references."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from common import load_bank,parse_answer


def main():
    folder=Path('research_runs');archive=np.load(folder/'validation_scores.npz')
    selection=json.loads((folder/'selection.json').read_text())
    key=selection['selected']['name'];y=archive['labels'];valid=archive['valid'];hard=archive['hard']
    def choices(name):return np.where(valid,archive[name],-1e9).argmax(axis=1)
    def correctness(name):return y[np.arange(len(y)),choices(name)]
    old,new=correctness('baseline'),correctness(key)
    delta=new[hard].astype(int)-old[hard].astype(int)
    rng=np.random.default_rng(20261023)
    boots=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(axis=1)
    gains=int((delta==1).sum());losses=int((delta==-1).sum())
    bank=load_bank('validation');scores=archive[key];margin=np.zeros(len(bank))
    for i,row in enumerate(bank):
        answer_scores={}
        for j,c in enumerate(row['candidates']):
            a=parse_answer(c['text'])
            if a is not None:answer_scores[a]=max(answer_scores.get(a,-1e9),scores[i,j])
        ordered=sorted(answer_scores.values(),reverse=True)
        margin[i]=ordered[0]-ordered[1] if len(ordered)>1 else np.inf
    indices=np.flatnonzero(hard);order=indices[np.argsort(-margin[indices],kind='stable')]
    coverage=[]
    for fraction in (.25,.5,.75,1.):
        selected=order[:int(np.ceil(fraction*len(order)))];correct=int(new[selected].sum())
        coverage.append({'coverage':len(selected)/len(order),'n':len(selected),'correct':correct,'accuracy':correct/len(selected)})
    family=[]
    inner=json.loads((folder/'inner_selection.json').read_text())
    fusion=json.loads((folder/'fusion_results.json').read_text())
    refits=json.loads((folder/'refit_results.json').read_text())
    for config in inner:
        name=config['family'];raw=next(r for r in fusion if r['family']==name and r['new_weight']==1)
        rows=[r for r in refits if r['family']==name]
        family.append({'family':name,'config':config,'ensemble_validation':raw['accuracy'],
            'ensemble_hard':raw['hard_accuracy'],'ensemble_internal_hard':raw['inner']['hard_accuracy'],
            'seed_hard':[r['hard_accuracy'] for r in rows],
            'seed_hard_std':float(np.std([r['hard_accuracy'] for r in rows],ddof=1)) if len(rows)>1 else 0.})
    categories=[('corrected',np.flatnonzero(hard & new & ~old)),
                ('regressed',np.flatnonzero(hard & ~new & old)),
                ('confident_error',np.array([i for i in order if not new[i]]))]
    cases=[]
    for category,ids in categories:
        for i in ids[:2]:
            i=int(i);row=bank[i]
            cases.append({'type':category,'row_zero_based':i,'question':row['question'],
                          'answers':[parse_answer(c['text']) for c in row['candidates']],
                          'correct_indices':np.flatnonzero(y[i]).tolist(),'old_choice':int(choices('baseline')[i]),
                          'new_choice':int(choices(key)[i]),'margin':float(margin[i]),
                          'candidates':[c['text'] for c in row['candidates']]})
    result={'family_results':family,'validation_paired_gain':gains,'validation_paired_loss':losses,
            'validation_delta':float(delta.mean()),'paired_bootstrap95':np.quantile(boots,[.025,.975]).tolist(),
            'bootstrap_replicates':10000,'bootstrap_seed':20261023,
            'mcnemar_exact_p':binomtest(gains,gains+losses,p=.5).pvalue if gains+losses else 1.,
            'coverage':coverage,
            'caution':'All development analyses reuse selection data. Confidence intervals/p-values do not adjust for search; no robust advantage is established.'}
    Path('runs/analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    Path('runs/validation_cases.json').write_text(json.dumps(cases,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
