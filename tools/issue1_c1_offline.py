import pathlib,sys,json,collections,numpy as np,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from tools.issue1_evaluate import metric,reversals,xyz
from data_loaders.humanml.scripts.motion_process import recover_from_ric_stateful
P=pathlib.Path('/home/cxm2/data/motion-gen/outputs/issue1_round2');V=P.parent/'issue1_v2';O=P.parent/'issue1_c1';O.mkdir(exist_ok=True)
rows=json.loads((V/'metrics.json').read_text());selected=json.loads((V/'selected.json').read_text())['bestof8'];groups={'L40':[r for r in rows if r['tag'].startswith('L40_')],'L60':[r for r in rows if r['tag'].startswith('L60_')],'L90':[r for r in rows if r['tag'].startswith('L90_')],'bestof8':selected}
parts_groups={'root':[0],'torso_head':[3,6,9,12,15],'left_arm':[13,16,18,20],'right_arm':[14,17,19,21],'legs':[1,2,4,5,7,8,10,11]}
def one(d,i):
 r=dict(d)
 for k in ('motion','hml_vec','lengths'):r[k]=d[k][i:i+1]
 r['decode_state']={k:v[i:i+1] for k,v in d['decode_state'].items()};r['num_samples']=1;return r
def load(r):return [one(np.load(P/f'{r["tag"]}_{m}.npy',allow_pickle=True).item(),r['trial']-1) for m in ('raise','wave','lower')]
def waves(d):
 m=xyz(d);axis=m[:,2]-m[:,1];axis/=np.maximum(np.linalg.norm(axis,axis=1,keepdims=True),1e-8);x=np.sum((m[:,21]-m[:,17])*axis,axis=1);z=x-x.mean();f=np.fft.rfftfreq(len(z),1/20);power=np.abs(np.fft.rfft(z))**2;power[0]=0
 return dict(reversals=reversals(m),ptp_m=float(np.ptp(x)),dominant_hz=float(f[np.argmax(power)]) if np.ptp(x)>1e-6 else 0.)
wave_rows=[]
for seed in range(1001,1017):
 d=np.load(P/f'A_{seed}.npy',allow_pickle=True).item()
 for i in range(3):wave_rows.append(dict(group='A',seed=seed,trial=i+1,**waves(one(d,i))))
for group,rr in groups.items():
 for r in rr:wave_rows.append(dict(group=group,tag=r['tag'],seed=r['seed'],trial=r['trial'],**waves(load(r)[1])))
summary={'waving':{},'c1':{},'seam_localization':{}};raw=[]
for group in ('A','L40','L60','L90','bestof8'):
 rr=[r for r in wave_rows if r['group']==group];summary['waving'][group]=dict(n=len(rr),not_waving=sum(r['reversals']<2 for r in rr),quantiles={k:{str(q):float(np.quantile([r[k] for r in rr],q)) for q in (.1,.5,.9)} for k in ('reversals','ptp_m','dominant_hz')})
loc=[]
for r in rows:
 a,b,c=map(xyz,load(r))
 for j,(prev,nxt) in enumerate(((a,b),(b,c))):
  if max(r['seams'][j]['ratios'])<=2:continue
  acceleration=((nxt[1]-nxt[0])-(nxt[0]-prev[-1]))*20
  loc.append(dict(tag=r['tag'],trial=r['trial'],seam=j+1,**{k:float(np.linalg.norm(acceleration[idx],axis=1).mean()) for k,idx in parts_groups.items()}))
for j in (1,2):
 rr=[r for r in loc if r['seam']==j];summary['seam_localization'][str(j)]=dict(failures=len(rr),body_mean={k:float(np.mean([r[k] for r in rr])) for k in parts_groups})
for group,rr in groups.items():
 for B in (12,20):
  records=[]
  for r in rr:
   original=load(r);fixed=[original[0]];overs=[]
   for seg in (1,2):
    prev=fixed[-1];d=dict(original[seg]);h=np.array(d['hml_vec'],copy=True);pre=prev['hml_vec'];n=h.shape[-1];b=min(B,n);w=.5*(1+np.cos(np.linspace(0,np.pi,b)));dv=(pre[:,3:193,-1:]-pre[:,3:193,-2:-1])-(h[:,3:193,1:2]-h[:,3:193,:1]);h[:,3:193,:b]+=np.arange(b)[None,None,:]*dv*w[None,None,:]
    data=torch.tensor(h.transpose(0,2,1)[:,None],dtype=torch.float32);state={k:torch.tensor(v,dtype=torch.float32) for k,v in prev['decode_state'].items()};decoded,next_state=recover_from_ric_stateful(data,22,state=state);d['motion']=decoded.view(-1,*decoded.shape[2:]).permute(0,2,3,1).numpy();d['hml_vec']=h;d['decode_state']={k:v.numpy() for k,v in next_state.items()};d['offline_c1_velocity_window']=B;fixed.append(d)
    m=xyz(d);v=np.linalg.norm(np.diff(m,axis=0),axis=2).mean(axis=1)*20;acc=np.linalg.norm(np.diff(m,n=2,axis=0),axis=2).mean(axis=1)*400;baseline=xyz(original[seg]);tailv=np.linalg.norm(np.diff(baseline,axis=0),axis=2).mean(axis=1)*20;taila=np.linalg.norm(np.diff(baseline,n=2,axis=0),axis=2).mean(axis=1)*400
    overs.append(dict(speed_peak_ratio=float(v[:b].max()/max(np.quantile(tailv,.95),1e-6)),acc_peak_ratio=float(acc[:b].max()/max(np.quantile(taila,.95),1e-6)),baseline_speed_peak=float(np.linalg.norm(np.diff(xyz(original[seg]),axis=0),axis=2).mean(axis=1)[:b].max()*20)))
   met=metric(fixed);record=dict(group=group,B=B,tag=r['tag'],seed=r['seed'],trial=r['trial'],before_success=r['success'],before_seam_pass='接缝跳变' not in r['failure_types'],after_seam_pass='接缝跳变' not in met['failure_types'],overshoot=overs,**met);records.append(record);raw.append(record)
   out=O/f'{group}_B{B}_{r["tag"]}_{r["trial"]}';out.mkdir(exist_ok=True)
   for mode,d in zip(('raise','wave','lower'),fixed):np.save(out/(mode+'.npy'),d)
  summary['c1'][f'{group}_B{B}']=dict(n=len(records),before_seam_pass=sum(r['before_seam_pass'] for r in records),after_seam_pass=sum(r['after_seam_pass'] for r in records),after_success=sum(r['success'] for r in records),speed_ratio_p95=float(np.quantile([s['speed_peak_ratio'] for r in records for s in r['overshoot']],.95)),acc_ratio_p95=float(np.quantile([s['acc_peak_ratio'] for r in records for s in r['overshoot']],.95)))
for name,data in [('summary',summary),('metrics',raw),('waving',wave_rows),('localization',loc)]: (O/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False))
