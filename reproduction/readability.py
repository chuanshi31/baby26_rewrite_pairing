"""Full-pair readability metrics using the definitions recorded with the paper data."""
import argparse,gzip,json,re,statistics
from pathlib import Path
WORD_RE=re.compile(r"[a-z]+(?:'[a-z]+)?")
def main():
    import textstat
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+');p.add_argument('--rewrite-field',default='simplified');p.add_argument('--output',required=True);a=p.parse_args()
    values={k:[] for k in ['ratio','lexical_jaccard','original_fre','rewrite_fre','original_fkgl','rewrite_fkgl','original_dale_chall','rewrite_dale_chall','original_difficult_word_ratio','rewrite_difficult_word_ratio']};skipped=0
    for name in a.inputs:
        op=gzip.open if name.endswith('.gz') else open
        with op(name,'rt',encoding='utf8') as f:
            for line in f:
                r=json.loads(line);original=r.get('original','').strip();rewrite=r.get(a.rewrite_field,'').strip()
                if not original or not rewrite:skipped+=1;continue
                x=set(WORD_RE.findall(original.lower()));y=set(WORD_RE.findall(rewrite.lower()))
                values['ratio'].append(len(rewrite.split())/max(1,len(original.split())))
                values['lexical_jaccard'].append(len(x&y)/max(1,len(x|y)))
                for key,text in [('original',original),('rewrite',rewrite)]:
                    for suffix,fun in [('fre',textstat.flesch_reading_ease),('fkgl',textstat.flesch_kincaid_grade),('dale_chall',textstat.dale_chall_readability_score)]:values[key+'_'+suffix].append(fun(text))
                    values[key+'_difficult_word_ratio'].append(textstat.difficult_words(text)/max(1,len(WORD_RE.findall(text.lower()))))
    out={'pairs':len(values['ratio']),'skipped':skipped,'metrics':{k:statistics.mean(v) if v else None for k,v in values.items()},'implementation':'Reconstructed from recorded metric definitions. The original temporary analysis script and its textstat version were not retained; exact last-decimal agreement is not certified.'}
    Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
