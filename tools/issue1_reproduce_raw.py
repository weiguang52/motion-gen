"""Capture raw suffixes, gated against the previously saved position-B12 run."""
import os,sys,pathlib,inspect,io,json,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import serve.generate_server as gs
P=pathlib.Path('/home/cxm2/data/motion-gen/outputs/issue1_round2');O=P.parent/'issue1_window';O.mkdir(exist_ok=True)
ns=dict(gs.__dict__);ns['BATCH']=3
source=inspect.getsource(gs._sample).replace('args.num_samples = 1','args.num_samples = BATCH').replace('    args.num_repetitions = 1','    texts = texts * BATCH\n    args.num_repetitions = 1')
exec(source,ns);sample=ns['_sample'];state=gs._init_state();base=state.diffusion.logsnr_sample_loop;cap={}
def capture(*a,**kw):
 x=base(*a,**kw);cap['raw']=x.detach().cpu();return x
state.diffusion.logsnr_sample_loop=capture;checks=[]
def run(tag,seed,L,batch,p40):
 ns['BATCH']=batch;state.args.seed=seed
 for mode,n,text in [('raise',20,'A person raises the right hand.'),('wave',60,'A person waves the right hand.'),('lower',20,'A person lowers the hand.')]:
  req=gs.GenerateRequest(text=text,num_frames=n,session_id=tag,reset_session=mode=='raise',lookahead_frames=L,is_final=mode=='lower',prefix_frames=40 if p40 and mode=='lower' else 20,blend_frames=12,blend_mode='position')
  d=np.load(io.BytesIO(sample(state,req)),allow_pickle=True).item();old=np.load(P/f'{tag}_{mode}.npy',allow_pickle=True).item()
  err=float(np.max(np.abs(d['motion']-old['motion'])));herr=float(np.max(np.abs(d['hml_vec']-old['hml_vec'])))
  checks.append(dict(tag=tag,mode=mode,xyz_error=err,hml_error=herr));(O/'reproduction.json').write_text(json.dumps(checks,indent=2))
  if max(err,herr)>=1e-4:
   (O/'BLOCKED').write_text('reproduction mismatch');raise RuntimeError((tag,mode,err,herr))
  p=d['continuation_prefix_frames'];raw=cap['raw'][...,p:p+n].squeeze(2).numpy();d['raw_suffix_hml']=raw*state.std[None,:,None]+state.mean[None,:,None];np.save(O/f'{tag}_{mode}.npy',d)
 state.session_motions.pop(tag,None);state.session_decode_states.pop(tag,None);print('REPRO_OK',tag,flush=True)
for L in (40,60,90):
 for seed in range(1001,1017):run(f'L{L}_{seed}',seed,L,3,False)
for seed in range(1001,1017):
 for repeat in range(1,4):run(f'combo_{seed}_{repeat}',seed+100000*repeat,40,8,True)
(O/'REPRO_COMPLETE').write_text('all match <1e-4')
