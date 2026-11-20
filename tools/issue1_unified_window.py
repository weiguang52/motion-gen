"""Unified raw-suffix window evaluation. Run after the reproduction gate."""
import pathlib,sys,json,collections,numpy as np,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from tools.issue1_evaluate import metric,xyz
from utils.sampler_util import smooth_continuation_boundary
from data_loaders.humanml.scripts.motion_process import recover_from_ric_stateful
P=pathlib.Path('/home/cxm2/data/motion-gen/outputs/issue1_window')
assert (P/'REPRO_COMPLETE').exists() and not (P/'BLOCKED').exists()
O=P/'offline';O.mkdir(exist_ok=True)
parts={'root':[0],'right_arm':[14,17,19,21],'left_arm':[13,16,18,20],'other':[1,2,3,4,5,6,7,8,9,10,11,12,15]}
rows=[];summary={};selected={}
def one(d,i):
 r=dict(d)
 for k in ('motion','hml_vec','raw_suffix_hml','lengths'):r[k]=d[k][i:i+1]
 r['decode_state']={k:v[i:i+1] for k,v in d['decode_state'].items()};r['num_samples']=1
 return r
def fix(ds,i,B,mode):
 fixed=[one(ds[0],i)];windows=[]
 for j in (1,2):
  d=one(ds[j],i);pre=np.concatenate([x['hml_vec'] for x in fixed],axis=-1)[...,-int(d['continuation_prefix_frames']):]
  raw=d['raw_suffix_hml'];b=min(B,pre.shape[-1],raw.shape[-1])
  h=smooth_continuation_boundary(torch.tensor(pre[:,:,None,:]),torch.tensor(raw[:,:,None,:]),B,blend_mode=mode).squeeze(2).numpy()
  state={k:torch.tensor(v) for k,v in fixed[-1]['decode_state'].items()}
  dec,st=recover_from_ric_stateful(torch.tensor(h.transpose(0,2,1)[:,None]),22,state=state)
  d['motion']=dec.view(-1,*dec.shape[2:]).permute(0,2,3,1).numpy();d['hml_vec']=h;d['decode_state']={k:v.numpy() for k,v in st.items()}
  d['blend_mode']=mode;d['requested_blend_frames']=B;d['effective_blend_frames']=b;fixed.append(d)
  prev,nxt=xyz(fixed[-2]),xyz(d)
  # Centers 0..b-1 include both seam differences and the window exit.
  joined=np.concatenate([prev[-2:],nxt]);acc=np.linalg.norm(np.diff(joined,n=2,axis=0),axis=2)*20
  normal=np.linalg.norm(np.diff(nxt,n=2,axis=0),axis=2)*20
  outside=normal[max(0,b-1):]
  available=len(outside)>0
  body={}
  for name,idx in parts.items():
   peak=float(acc[:b,idx].mean(axis=1).max());base=float(np.quantile(outside[:,idx].mean(axis=1),.95)) if available else None
   body[name]=dict(peak=peak,p95=base,ratio=peak/max(base,1e-6) if available else None)
  peak=float(acc[:b].mean(axis=1).max());base=float(np.quantile(outside.mean(axis=1),.95)) if available else None
  windows.append(dict(effective_B=b,normal_available=available,ratio=peak/max(base,1e-6) if available else None,body=body))
 met=metric(fixed);met['v2_success']=met['success'];passed=all(w['ratio'] is not None and w['ratio']<=2 for w in windows)
 met['window_pass']=passed;met['windows']=windows
 if not passed:met['failure_types'].append('修正窗口过冲或无窗口外基准')
 met['success']=not met['failure_types']
 return fixed,met
for f in sorted(P.glob('*_raise.npy')):
 tag=f.name[:-10];ds=[np.load(P/f'{tag}_{m}.npy',allow_pickle=True).item() for m in ('raise','wave','lower')]
 for B in (12,20,30):
  for mode in ('position','c1'):
   for i in range(len(ds[0]['motion'])):
    fixed,met=fix(ds,i,B,mode);r=dict(tag=tag,trial=i+1,B=B,mode=mode,**met);rows.append(r)
    dest=O/f'{mode}_B{B}_{tag}_{i+1}';dest.mkdir(exist_ok=True)
    for name,d in zip(('raise','wave','lower'),fixed):np.save(dest/f'{name}.npy',d)
for B in (12,20,30):
 for mode in ('position','c1'):
  for group in ('L40','L60','L90','combo'):
   rr=[r for r in rows if r['B']==B and r['mode']==mode and r['tag'].startswith(group+'_')]
   key=f'{group}_{mode}_B{B}'
   summary[key]=dict(n=len(rr),seam_pass=sum('接缝跳变' not in r['failure_types'] for r in rr),window_pass=sum(r['window_pass'] for r in rr),v2_success=sum(r['v2_success'] for r in rr),success=sum(r['success'] for r in rr),unavailable_windows=sum(not w['normal_available'] for r in rr for w in r['windows']))
  if mode=='c1':
   for K in (1,4,8):
    picked=[]
    for seed in range(1001,1017):
     for rep in (1,2,3):
      candidates=[r for r in rows if r['B']==B and r['mode']==mode and r['tag']==f'combo_{seed}_{rep}' and r['trial']<=K]
      # Fixed lexicographic acceptance score, then fewer failures, then lower max seam/window ratio.
      best=min(candidates,key=lambda r:(not r['success'],len(r['failure_types']),-r['wave_zero_crossings'],max([r['seam_max_ratio']]+[w['ratio'] if w['ratio'] is not None else 1e9 for w in r['windows']]),r['trial']))
      picked.append(best)
    selected[f'B{B}_bestof{K}']=picked
    summary[f'B{B}_bestof{K}']=dict(n=len(picked),success=sum(r['success'] for r in picked),v2_success=sum(r['v2_success'] for r in picked),failures=dict(collections.Counter(t for r in picked for t in r['failure_types'])))
for name,value in [('metrics',rows),('summary',summary),('selected',selected)]: (O/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False),flush=True)
