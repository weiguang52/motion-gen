"""Render issue #1 evidence on the server; never substitute stems for prompts."""
import pathlib,json,subprocess,os,numpy as np,importlib.util,sys
P=pathlib.Path('/home/cxm2/data/motion-gen/outputs');O=P/'issue1_stand';W=P/'issue1_window/offline';arm=pathlib.Path('/home/cxm2/data/ArmIK');py=str(arm/'.conda-env/bin/python')
env=os.environ.copy();env.update(MUJOCO_GL='egl',MPLBACKEND='Agg',OMP_NUM_THREADS='2')
spec=importlib.util.spec_from_file_location('render_source',arm/'tools/render_source_motion.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);orig=mod.plot_3d_motion
def one(d,i):
 d=dict(d)
 for k in ('motion','hml_vec','lengths'):d[k]=d[k][i:i+1]
 d['decode_state']={k:v[i:i+1] for k,v in d['decode_state'].items()};d['num_samples']=1;return d
def load(tag,i,modes=('wave',)):return [one(np.load(O/f'{tag}_{m}.npy',allow_pickle=True).item(),i) for m in modes]
tasks=[]
for seed in (1001,1002):
 for kind in (None,'stand_mean','stand_data'):
  tag=f'I0_{seed}' if kind is None else f'SA_{kind}_L0_{seed}'
  tasks.append((f'SA_compare_{seed}_{kind or "none"}',load(tag,0),dict(tag=tag,seed=seed,trial=1,prefix=kind,L=0,P=20 if kind else 0)))
for tag in ('I90_1001','D0_0_1001','D40_90_1001'):
 tasks.append((tag,load(tag,0,('raise','wave') if tag.startswith('D') else ('wave',)),dict(tag=tag,trial=1)))
sel=json.loads((W/'selected.json').read_text())['B12_bestof8']
# There are no window-accepted successes. Include old-v2 passes, explicitly labeled.
chosen=[r for r in sel if r['v2_success']][:2]+[r for r in sel if not r['v2_success']][:2]
for i,r in enumerate(chosen):
 dpath=W/f'c1_B12_{r["tag"]}_{r["trial"]}'
 ds=[np.load(dpath/f'{m}.npy',allow_pickle=True).item() for m in ('raise','wave','lower')]
 tasks.append((f'reselected_{i}',ds,r))
 if i<2:
  dpath=W/f'position_B12_{r["tag"]}_{r["trial"]}'
  before=[np.load(dpath/f'{m}.npy',allow_pickle=True).item() for m in ('raise','wave','lower')]
  tasks.append((f'position_compare_{i}',before,dict(r,mode='position')))
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axs=plt.subplots(3,4,figsize=(16,9))
  for name,parts in [('position',before),('c1',ds)]:
   x=np.concatenate([d['motion'][0].transpose(2,0,1) for d in parts]);v=np.linalg.norm(np.diff(x,axis=0),axis=2)*20;a=np.linalg.norm(np.diff(x,n=2,axis=0),axis=2)*400
   for row,(body,joints) in enumerate([('right arm',[14,17,19,21]),('left arm',[13,16,18,20]),('root',[0])]):
    for j,cut in enumerate((20,80)):
     for k,(values,unit,offset) in enumerate([(v,'m/s',1),(a,'m/s2',2)]):
      t=np.arange(offset,offset+len(values))-cut;mask=abs(t)<=15;ax=axs[row,2*j+k];ax.plot(t[mask],values[mask][:,joints].mean(axis=1),label=name);ax.set_title(f'{body}, seam{j+1}, {unit}');ax.axvline(0,color='gray');ax.legend()
  fig.tight_layout();fig.savefig(O/f'window_compare_{i}.png',dpi=120);plt.close(fig)
# Stand-start SB is rendered after all groups finish.
if (O/'COMPLETE').exists():
 summary=json.loads((O/'summary.json').read_text())
 # Select lowest not-waving count separately within each stand type; this is not a passing scheme.
 for kind in ('stand_mean','stand_data'):
  keys=[k for k in summary if k.startswith('SB_'+kind) and summary[k]['n']==48];g=min(keys,key=lambda k:(summary[k]['not_wave'],-summary[k]['above']));a,b=map(int,g.rsplit('_',2)[1:])
  for seed in (1001,1002):
   for tag in (f'{g}_{seed}',f'D{a}_{b}_{seed}'):
    tasks.append((tag,load(tag,0,('raise','wave')),dict(tag=tag,trial=1,selection='lowest not-waving count within stand type')))
manifest=[]
for label,ds,meta in tasks:
 out=O/'videos'/label;out.mkdir(exist_ok=True,parents=True);files=[];titles=[]
 for i,d in enumerate(ds):
  f=out/f'segment_{i}.npy';np.save(f,d);files.append(str(f));titles.extend([str(d['text'][0])]*int(d['lengths'][0]))
 if not (out/'human_robot.mp4').exists():
  def plot(*a,**kw):a=list(a);a[3]=titles;return orig(*a,**kw)
  mod.plot_3d_motion=plot;sys.argv=['render','--npy',*files,'--output',str(out/'human.mp4')];mod.main()
  cmds=[[py,str(arm/'RobotArmControl/humanoid_motion_stream_ik.py'),'--npy',*files,'--output',str(out/'joints.npz')],[py,str(arm/'RobotArmControl/export_humanoid_mujoco_video.py'),'--robot-npz',str(out/'joints.npz'),'--output',str(out/'robot.mp4'),'--width','480','--height','480']]
  with (out/'render.log').open('w') as log:
   for cmd in cmds:subprocess.run(cmd,env=env,cwd=str(arm),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-i',str(out/'human.mp4'),'-i',str(out/'robot.mp4'),'-filter_complex','[0:v]scale=480:480[a];[a][1:v]hstack=inputs=2','-an',str(out/'human_robot.mp4')],check=True)
 manifest.append(dict(label=label,video=str(out/'human_robot.mp4'),text=[str(d['text'][0]) for d in ds],guidance=2.5,blend=12,**meta))
 (O/'video_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('DONE',label,flush=True)
(O/'RENDER_COMPLETE').write_text('done')
