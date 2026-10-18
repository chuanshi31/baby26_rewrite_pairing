"""Portable launcher for the released paper conditions. Prints command unless --execute."""
import argparse,json,shlex,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--condition',required=True)
p.add_argument('--masking',choices=['token','wwm7tok3'],default='token')
p.add_argument('--seed',type=int,choices=[0,1,2],default=0)
p.add_argument('--output',required=True)
p.add_argument('--execute',action='store_true')
a=p.parse_args()
if a.condition=='submission100m':
    data='data/fineweb_os_bcc_100m_simplified_pairs.txt'
    tokenizer='tokenizers/fineweb_os_bcc_100m_40k.model'
    hidden,intermediate,batch,acc=768,2560,2048,4
    extra=['--regular_mlm','--span_masking','span_geo','--span_epochs','0','--post_span_masking','token','--token_epochs','0','--span_geo_p','0.5','--span_stats_mode','average']
    script='train_mask.py'
else:
    rows=json.loads((ROOT/'reproduction/runs.json').read_text())
    row=next(x for x in rows if x['condition']==a.condition and x['masking']==a.masking and x['seed']==a.seed)
    data,tokenizer=row['train_data'],row['tokenizer']
    hidden,intermediate,batch,acc=384,1280,256,2
    script='train_mask.py'
    extra=['--regular_mlm','--span_masking','wwm' if a.masking=='wwm7tok3' else 'token','--span_epochs','7' if a.masking=='wwm7tok3' else '0','--token_epochs','0','--span_stats_mode','average']
cmd=[sys.executable,script,'--train_data',data,'--valid_data','data/bb26.dev','--tokenizer',tokenizer,'--model_path','configs/deberta-v3-base','--output_path',a.output,'--hidden_size',str(hidden),'--intermediate_size',str(intermediate),'--dropout','0.1','--max_seq_len','0:64,5:256','--batch_size',str(batch),'--grad_acc',str(acc),'--lr','0.007','--epochs','10','--weight_decay','0.01','--seed',str(a.seed),'--cpus','8','--mlm_prob','0.15','--mask_decay','0','--mask_replace_prob','0.8','--random_replace_prob','0.1','--mask_update_steps','500','--first_mask_update','0','--logging_steps','100','--eval_steps','500','--save_steps','999999999','--lamb']+extra
for f in [data,tokenizer,'data/bb26.dev','configs/deberta-v3-base/config.json']:
    if not (ROOT/f).is_file():raise SystemExit('Missing input: '+f+'; restore the 100M corpus first if applicable.')
print(shlex.join(cmd))
if a.execute:subprocess.run(cmd,cwd=ROOT,check=True)
