"""Issue1 v2: consistent seam differences, armed release, body-axis hysteresis."""
import numpy as np
from tools.issue1_evaluate_v1 import xyz,heading

def height(m):return m[:,21,1]-m[:,17,1]
def release(heights,start=0):
 armed=False;up=0
 for i,h in enumerate(heights):
  up=up+1 if h>0 else 0
  if up>=10:armed=True
  if armed and i>=start and i+5<=len(heights) and np.all(heights[i:i+5]<0):return i-start,True
 return len(heights)-start,False

def reversals(m,deadband=.03):
 axis=m[:,2]-m[:,1];axis/=np.maximum(np.linalg.norm(axis,axis=1,keepdims=True),1e-8)
 x=np.sum((m[:,21]-m[:,17])*axis,axis=1)
 if len(x)<2:return 0
 pivot=extreme=float(x[0]);direction=0;count=0
 for value in x[1:]:
  value=float(value)
  if direction==0:
   if abs(value-pivot)>=deadband:direction=1 if value>pivot else -1;extreme=value
  elif direction>0:
   extreme=max(extreme,value)
   if extreme-value>=deadband:count+=1;direction=-1;pivot=extreme;extreme=value
  else:
   extreme=min(extreme,value)
   if value-extreme>=deadband:count+=1;direction=1;pivot=extreme;extreme=value
 return count

def seam(prev,nxt,fps=20):
 bridge=nxt[0]-prev[-1]
 j1=float(np.linalg.norm(bridge-(prev[-1]-prev[-2]),axis=1).mean()*fps)
 j2=float(np.linalg.norm((nxt[1]-nxt[0])-bridge,axis=1).mean()*fps)
 root=float(np.linalg.norm(bridge[0]));yaw=float(abs(np.angle(np.exp(1j*(heading(nxt)[0]-heading(prev)[-1])))))
 normal=[np.concatenate([np.linalg.norm(np.diff(t[:,0],axis=0),axis=1) for t in (prev,nxt)]),np.concatenate([np.abs(np.diff(heading(t))) for t in (prev,nxt)]),np.concatenate([np.linalg.norm(np.diff(t,n=2,axis=0),axis=2).mean(axis=1)*fps for t in (prev,nxt)])]
 bounds=[float(np.quantile(v,.95)) for v in normal];jump=max(j1,j2);ratios=[v/max(z,1e-6) for v,z in zip((root,yaw,jump),bounds)]
 return dict(root_m=root,yaw_rad=yaw,velocity_jump_m_s=jump,j1=j1,j2=j2,normal_p95=bounds,ratios=ratios)

def metric(parts):
 a,b,c=map(xyz,parts);h=height(b);hc=height(c);frame,observed=release(np.r_[height(a),h],len(a));cross=reversals(b);ss=[seam(a,b),seam(b,c)];ratio=max(v for s in ss for v in s['ratios']);rise=float(np.max(hc-hc[0]));above=float(np.mean(h>0));fail=[]
 if h[0]<0:fail.append('抬手不到位')
 if above<.8 or frame<54:fail.append('提前放下')
 if cross<2:fail.append('举着不挥')
 if rise>=.05 or hc[-1]>=0:fail.append('第三段重复抬手或未放下')
 if ratio>2:fail.append('接缝跳变')
 return dict(above_fraction=above,release_frame=frame,release_observed=observed,wave_zero_crossings=cross,lower_max_rise_m=rise,lower_end_height_m=float(hc[-1]),raise_end_height_m=float(height(a)[-1]),wave_start_height_m=float(h[0]),seam_max_ratio=ratio,seams=ss,success=not fail,failure_types=fail)

def independent(d):
 m=xyz(d);h=height(m);f,obs=release(h);return dict(release_frame=f,release_observed=obs,end_height=float(h[-1]),last30_above_fraction=float(np.mean(h[-30:]>0)))
