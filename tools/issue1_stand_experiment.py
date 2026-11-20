"""Issue #1 standing-prefix and independent-window controls; fixed batched RNG."""
import pathlib,sys,os,io,json,inspect,time,numpy as np,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import serve.generate_server as gs
from infer import _load_continuation_prefix
from tools.issue1_evaluate import seam,reversals,xyz,height
P=pathlib.Path('/home/cxm2/data/motion-gen/outputs');O=P/'issue1_stand';O.mkdir(exist_ok=True)
os.environ['MDM_STAND_DATA_PREFIX_NPY']=str(O/'stand_data.npy')
ns=dict(gs.__dict__);ns['BATCH']=3
source=inspect.getsource(gs._sample).replace('args.num_samples = 1','args.num_samples = BATCH').replace('    args.num_repetitions = 1','    texts = texts * BATCH\n    args.num_repetitions = 1')
exec(source,ns);sample=ns['_sample'];state=gs._init_state();base=state.diffusion.logsnr_sample_loop;captured={}
def capture(*a,**kw):
 t=time.perf_counter();x=base(*a,**kw);captured['raw']=x.detach().cpu();captured['seconds']=time.perf_counter()-t;return x
state.diffusion.logsnr_sample_loop=capture;prefixes={}
for name,path in [('stand_mean',str(pathlib.Path.cwd()/'assets/ar_dit_zero_vel_stand_prefix.npy')),('stand_data',str(O/'stand_data.npy'))]:
 pre,dec=_load_continuation_prefix(path,state.mean,state.std,20,263);prefixes[name]=(pre,dec)
rows=[]
def one(d,i):
 r=dict(d);r['motion']=d['motion'][i:i+1];return r
def run(tag,seed,init,segments,manual=False):
 ns['BATCH']=3;state.args.seed=seed;sid=tag;state.session_motions.pop(sid,None);state.session_decode_states.pop(sid,None)
 if init and manual:
  pre,dec=prefixes[init];state.session_motions[sid]=pre.expand(3,-1,-1,-1).clone();state.session_decode_states[sid]={k:v.expand(3,*v.shape[1:]).clone() for k,v in dec.items()}
 outputs=[]
 for index,(mode,n,L) in enumerate(segments):
  text='A person raises the right hand.' if mode=='raise' else 'A person waves the right hand.'
  req=gs.GenerateRequest(text=text,num_frames=n,session_id=sid,init_prefix=init if index==0 and not manual else None,lookahead_frames=L,blend_mode='position',blend_frames=12)
  d=np.load(io.BytesIO(sample(state,req)),allow_pickle=True).item();p=d['continuation_prefix_frames'];raw=captured['raw'][...,p:p+n].squeeze(2).numpy()
  d['raw_suffix_hml']=raw*state.std[None,:,None]+state.mean[None,:,None];d['sampling_seconds']=captured['seconds'];d['init_prefix']=init;np.save(O/f'{tag}_{mode}.npy',d);outputs.append(d)
 for i in range(3):
  m=xyz(one(outputs[-1],i));axis=m[:,2]-m[:,1];axis/=np.maximum(np.linalg.norm(axis,axis=1,keepdims=True),1e-8);x=np.sum((m[:,21]-m[:,17])*axis,axis=1)
  power=np.abs(np.fft.rfft(x-x.mean()))**2;power[0]=0;freq=np.fft.rfftfreq(len(x),1/20);first=xyz(one(outputs[0],i));h0=height(first);rise=np.flatnonzero(h0>h0[0]+.03)
  row=dict(tag=tag,seed=seed,trial=i+1,prefix=init,segments=segments,reversals=reversals(m),ptp_m=float(np.ptp(x)),dominant_hz=float(freq[np.argmax(power)]),above_fraction=float(np.mean(height(m)>0)),wave_start_height=float(height(m)[0]),start_delay_frame=int(rise[0]) if len(rise) else None)
  if len(outputs)==2:row['raise_end_height']=float(height(xyz(one(outputs[0],i)))[-1])
  if init:
   pre,dec=prefixes[init];phys=pre.squeeze(0).squeeze(1).T.numpy()*state.std+state.mean
   from data_loaders.humanml.scripts.motion_process import recover_from_ric_stateful
   pm,_=recover_from_ric_stateful(torch.tensor(phys[None,None],dtype=torch.float32),22);row['start_seam']=seam(pm.numpy()[0,0],first)
  rows.append(row)
 (O/'metrics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));state.session_motions.pop(sid,None);state.session_decode_states.pop(sid,None);return outputs
for kind in prefixes:
 for L in (0,40):
  for seed in range(1001,1017):
   tag=f'SA_{kind}_L{L}_{seed}';ds=run(tag,seed,kind,[('wave',60,L)])
   if seed==1001 and L==0:
    manual=run(tag+'_manualcheck',seed,kind,[('wave',60,L)],manual=True);err=float(np.max(np.abs(ds[0]['motion']-manual[0]['motion'])));assert err<1e-6;print('INIT_EQ',kind,err,flush=True)
   print('SA',kind,L,seed,flush=True)
(O/'SA_COMPLETE').write_text('done')
for seed in range(1001,1017):
 for L in (0,40,90):run(f'I{L}_{seed}',seed,None,[('wave',60,L)]);print('I',L,seed,flush=True)
 for a in (0,40):
  for b in (0,40,90):run(f'D{a}_{b}_{seed}',seed,None,[('raise',20,a),('wave',60,b)]);print('D',a,b,seed,flush=True)
(O/'ID_COMPLETE').write_text('done')
for kind in prefixes:
 for a in (0,40):
  for b in (0,40):
   for seed in range(1001,1017):run(f'SB_{kind}_{a}_{b}_{seed}',seed,kind,[('raise',20,a),('wave',60,b)]);print('SB',kind,a,b,seed,flush=True)
(O/'COMPLETE').write_text('done')
