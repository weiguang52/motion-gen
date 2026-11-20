import unittest,numpy as np
from tools.issue1_evaluate import seam,reversals,release
class V2Tests(unittest.TestCase):
 def test_smooth_split(self):
  t=np.arange(100)/20;m=np.zeros((100,22,3));m[:,:,0]=np.sin(t[:,None]*3+np.arange(22)[None,:]*.1)*.2;m[:,:,1]=np.cos(t[:,None]*2+np.arange(22)[None,:]*.1)*.1;m[:,:,2]=t[:,None]*.2;m[:,2,0]+=1
  for cut in (20,80):
   r=seam(m[:cut],m[cut:]);self.assertLessEqual(max(r['ratios']),2);self.assertGreater(r['ratios'][2],.3);self.assertLess(r['ratios'][2],1.5)
 def test_stationary_raise_no_wave(self):
  m=np.zeros((60,22,3));m[:,2,0]=1;m[:,21,1]=1;self.assertEqual(reversals(m),0)
 def test_small_jitter_ignored(self):
  m=np.zeros((60,22,3));m[:,2,0]=1;m[:,21,0]=np.sin(np.arange(60))*0.005;self.assertEqual(reversals(m),0)
 def test_armed_release(self):
  self.assertEqual(release(np.r_[np.full(5,-1),np.ones(10),np.full(5,-1)],5),(10,True))
  self.assertFalse(release(np.full(60,-1))[1])
