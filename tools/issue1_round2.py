"""Reproducible server-core experiment, batching only in this harness (no API changes)."""
import os,sys,pathlib,inspect,io,json,time,csv,collections
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.issue1_evaluate import metric,independent
import serve.generate_server as gs
OUT=pathlib.Path(os.environ['ISSUE1_OUT']);OUT.mkdir(parents=True,exist_ok=True)
# Public endpoint remains single-sample. This harness repeats identical texts for B independent noise draws.
ns=dict(gs.__dict__);ns['BATCH']=3
source=inspect.getsource(gs._sample).replace('args.num_samples = 1','args.num_samples = BATCH').replace('    args.num_repetitions = 1','    texts = texts * BATCH\n    args.num_repetitions = 1')
exec(source,ns);sample=ns['_sample'];state=gs._init_state();state.args.guidance_param=2.5
rows=[];A=[];timings=[]
def split(d,i):
 r=dict(d)
 for k in ('motion','hml_vec','lengths'):r[k]=d[k][i:i+1]
 r['decode_state']={k:v[i:i+1] for k,v in d['decode_state'].items()};r['num_samples']=1;return r
def run(tag,seed,L,batch=3,p40=False,text='A person waves the right hand.'):
 ns['BATCH']=batch;state.args.seed=seed;parts=[];start=time.perf_counter()
 for mode,n,prompt in [('raise',20,'A person raises the right hand.'),('wave',60,text),('lower',20,'A person lowers the hand.')]:
  req=gs.GenerateRequest(text=prompt,num_frames=n,session_id=tag,reset_session=mode=='raise',prefix_frames=40 if p40 and mode=='lower' else 20,blend_frames=12,lookahead_frames=L,is_final=mode=='lower')
  d=np.load(io.BytesIO(sample(state,req)),allow_pickle=True).item();assert d['motion'].shape==(batch,22,3,n);np.save(OUT/f'{tag}_{mode}.npy',d);parts.append(d)
 elapsed=time.perf_counter()-start;result=[]
 for i in range(batch):
  r=dict(tag=tag,seed=seed,trial=i+1,P=20,N=60,L=L,blend=12,guidance=2.5,candidates=batch,wave_text=text,**metric([split(d,i) for d in parts]));result.append(r)
 timings.append(dict(tag=tag,batch=batch,seconds=elapsed));state.session_motions.pop(tag,None);state.session_decode_states.pop(tag,None);return result

def save():
 (OUT/'metrics.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False));(OUT/'A_metrics.json').write_text(json.dumps(A,indent=2));(OUT/'timings.json').write_text(json.dumps(timings,indent=2))
 if rows:
  fields=[k for k in rows[0] if k!='seams']
  with (OUT/'metrics.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
for seed in range(1001,1017):
 ns['BATCH']=3;state.args.seed=seed;d=np.load(io.BytesIO(sample(state,gs.GenerateRequest(text='A person waves the right hand.',num_frames=60,session_id='A',reset_session=True,continue_from_previous=False,is_final=True))),allow_pickle=True).item();np.save(OUT/f'A_{seed}.npy',d)
 for i in range(3):A.append(dict(seed=seed,trial=i+1,**independent(split(d,i))))
 for L in (0,20,40,60,90):
  rows+=run(f'L{L}_{seed}',seed,L);save();print('SCAN',seed,L,flush=True)
summary={L:float(np.mean([r['success'] for r in rows if r['L']==L])) for L in (0,20,40,60,90)};best=max(summary,key=lambda L:(summary[L],-L));(OUT/'scan_summary.json').write_text(json.dumps(dict(rates=summary,best_L=best),indent=2));print('BEST',best,summary,flush=True)
for seed in range(1001,1017):
 for trial in range(3):
  effective=seed+100000*(trial+1);res=run(f'combo_{seed}_{trial+1}',effective,best,batch=8,p40=True)
  for i,r in enumerate(res):r.update(base_seed=seed,repeat=trial+1,candidate=i+1)
  rows+=res;save();print('COMBO',seed,trial+1,flush=True)
 rows+=run(f'repeated_{seed}',seed,best,text='A person waves the right hand repeatedly.');save()
(OUT/'COMPLETE').write_text('sampling complete\n')
