import pathlib,json,csv,collections,numpy as np,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from tools.issue1_evaluate import metric,independent,seam,xyz,height
P=pathlib.Path('/home/cxm2/data/motion-gen/outputs/issue1_round2');O=pathlib.Path('/home/cxm2/data/motion-gen/outputs/issue1_v2');O.mkdir(exist_ok=True)
def split(d,i):
 r=dict(d);r['motion']=d['motion'][i:i+1];return r
rows=[];curves=collections.defaultdict(list)
for old in json.loads((P/'metrics.json').read_text()):
 i=old['trial']-1;parts=[split(np.load(P/f'{old["tag"]}_{m}.npy',allow_pickle=True).item(),i) for m in ('raise','wave','lower')];v=metric(parts);r={**old,**v,'v1_success':old['success'],'v1_failures':old['failure_types']};rows.append(r)
 group='repeated' if r['tag'].startswith('repeated') else 'combo' if r['tag'].startswith('combo') else f'L{r["L"]}';curves[group].append(np.concatenate([height(xyz(d)) for d in parts]))
summary={};selected={}
def stat(rr):return dict(n=len(rr),v1_successes=sum(r['v1_success'] for r in rr),v2_successes=sum(r['success'] for r in rr),changed=sum(r['v1_success']!=r['success'] for r in rr),failures=dict(collections.Counter(t for r in rr for t in r['failure_types'])))
for name in ('L0','L20','L40','L60','L90','repeated'):
 rr=[r for r in rows if r['tag'].startswith(name+'_')];summary[name]=stat(rr)
groups=collections.defaultdict(list)
for r in rows:
 if r['tag'].startswith('combo'):groups[r['tag']].append(r)
for k in (1,4,8):
 rr=[max(g[:k],key=lambda r:(r['success'],-len(r['failure_types']),r['above_fraction'],r['release_frame'],r['wave_zero_crossings'],-r['lower_max_rise_m'],-r['seam_max_ratio'])) for g in groups.values()];summary[f'bestof{k}']=stat(rr);selected[f'bestof{k}']=rr
# Compare old vs new best-of strategies separately; per-candidate label changes remain in rows.
a=[];cal=[]
for seed in range(1001,1017):
 d=np.load(P/f'A_{seed}.npy',allow_pickle=True).item()
 for i in range(3):
  one=split(d,i);m=xyz(one);a.append(dict(seed=seed,trial=i+1,**independent(one)))
  for cut in (20,40):
   s=seam(m[:cut],m[cut:]);cal.append(dict(seed=seed,trial=i+1,cut=cut,ratio=max(s['ratios']),velocity_ratio=s['ratios'][2],passed=max(s['ratios'])<=2))
summary['calibration']=dict(n=len(cal),pass_rate=float(np.mean([r['passed'] for r in cal])),quantiles={str(q):float(np.quantile([r['ratio'] for r in cal],q)) for q in (.5,.9,.95,.99)})
for name,data in [('metrics',rows),('A_metrics',a),('calibration',cal)]:
 (O/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));fields=list(dict.fromkeys(k for r in data for k in r if k!='seams'))
 with (O/f'{name}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(data)
(O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(O/'selected.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
curves.pop('combo',None)
for name,rr in selected.items():
 curves[name]=[np.concatenate([height(xyz(split(np.load(P/f'{r["tag"]}_{m}.npy',allow_pickle=True).item(),r['trial']-1))) for m in ('raise','wave','lower')]) for r in rr]
fig,axes=plt.subplots(2,5,figsize=(20,8),sharex=True,sharey=True)
for ax,(name,cs) in zip(axes.flat,curves.items()):
 ar=np.array(cs);mu=ar.mean(0);sd=ar.std(0);t=np.arange(100)/20;ax.plot(t,mu);ax.fill_between(t,mu-sd,mu+sd,alpha=.2);ax.axvline(1,color='r',linestyle='--');ax.axvline(4,color='r',linestyle='--');ax.axhline(0,color='gray');ax.set_title(name);ax.set_xlabel('seconds');ax.set_ylabel('wrist minus shoulder (m)')
fig.tight_layout();fig.savefig(O/'height_curves.png',dpi=160);fig.savefig(O/'height_curves.pdf');print(json.dumps(summary,ensure_ascii=False))
