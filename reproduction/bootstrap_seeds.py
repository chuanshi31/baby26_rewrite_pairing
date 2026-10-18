"""Paired bootstrap of saved per-seed metrics; does not train or rescore models."""
import argparse,json,random,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('summary');p.add_argument('--condition',required=True);p.add_argument('--metric',default='ewok');p.add_argument('--a',default='wwm7tok3');p.add_argument('--b',default='token');p.add_argument('--samples',type=int,default=10000);p.add_argument('--random-seed',type=int,default=0);args=p.parse_args()
rows=json.loads(Path(args.summary).read_text())['seed_metrics']
aa={r['seed']:r[args.metric] for r in rows if r['condition']==args.condition and r['masking']==args.a}
bb={r['seed']:r[args.metric] for r in rows if r['condition']==args.condition and r['masking']==args.b}
assert set(aa)==set(bb) and len(aa)>=2
d=[aa[s]-bb[s] for s in sorted(aa)];rng=random.Random(args.random_seed)
draws=sorted(statistics.mean(rng.choices(d,k=len(d))) for _ in range(args.samples))
print(json.dumps({'condition':args.condition,'metric':args.metric,'a':args.a,'b':args.b,'paired_seeds':sorted(aa),'mean_difference':statistics.mean(d),'bootstrap_samples':args.samples,'random_seed':args.random_seed,'percentile_95_interval':[draws[int(.025*(len(draws)-1))],draws[int(.975*(len(draws)-1))]],'note':'Reconstructed analysis on rounded saved reports; not a claim to recover an unrecorded original bootstrap RNG state.'},indent=2))
