"""Frozen issue #1 metrics. Frame indices are zero-based; no threshold relaxation."""
import numpy as np

def xyz(d):return d['motion'][0].transpose(2,0,1)
def heading(m):
 v=m[:,2]-m[:,1];return np.unwrap(np.arctan2(v[:,2],v[:,0]))
def metric(parts):
 a,b,c=map(xyz,parts);h=b[:,21,1]-b[:,17,1];hc=c[:,21,1]-c[:,17,1]
 below=[i for i in range(len(h)-4) if np.all(h[i:i+5]<0)]
 # No smoothing/deadband: literal horizontal velocity sign reversals, excluding zeros.
 vx=np.diff(b[:,21,0]-b[:,17,0]);sg=np.sign(vx);sg=sg[sg!=0];cross=int(np.sum(sg[1:]!=sg[:-1]))
 ratios=[];details=[]
 for prev,nxt in ((a,b),(b,c)):
  root=float(np.linalg.norm(nxt[0,0]-prev[-1,0]));yaw=float(abs(np.angle(np.exp(1j*(heading(nxt)[0]-heading(prev)[-1])))))
  jump=float(np.linalg.norm((nxt[1]-nxt[0])-(prev[-1]-prev[-2]),axis=1).mean()*20)
  normal_root=np.concatenate([np.linalg.norm(np.diff(t[:,0],axis=0),axis=1) for t in (prev,nxt)])
  normal_yaw=np.concatenate([np.abs(np.diff(heading(t))) for t in (prev,nxt)])
  normal_acc=np.concatenate([np.linalg.norm(np.diff(t,n=2,axis=0),axis=2).mean(axis=1)*20 for t in (prev,nxt)])
  bounds=[float(np.quantile(v,.95)) for v in (normal_root,normal_yaw,normal_acc)]
  rr=[x/max(z,1e-6) for x,z in zip((root,yaw,jump),bounds)];ratios+=rr;details.append(dict(root_m=root,yaw_rad=yaw,velocity_jump_m_s=jump,normal_p95=bounds,ratios=rr))
 rise=float(np.max(hc-hc[0]));first=below[0] if below else len(h);above=float(np.mean(h>0));fail=[]
 if above<.8 or first<54:fail.append('提前放下')
 if cross<2:fail.append('举着不挥')
 if rise>=.05 or hc[-1]>=0:fail.append('第三段重复抬手或未放下')
 if max(ratios)>2:fail.append('接缝跳变')
 return dict(above_fraction=above,release_frame=first,release_observed=bool(below),wave_zero_crossings=cross,lower_max_rise_m=rise,lower_end_height_m=float(hc[-1]),seam_max_ratio=max(ratios),seams=details,success=not fail,failure_types=fail)
def independent(d):
 m=xyz(d);h=m[:,21,1]-m[:,17,1];v=[i for i in range(len(h)-4) if np.all(h[i:i+5]<0)]
 return dict(release_frame=v[0] if v else len(h),release_observed=bool(v),end_height=float(h[-1]))
