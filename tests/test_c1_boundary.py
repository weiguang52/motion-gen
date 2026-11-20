import unittest,torch
from utils.sampler_util import smooth_continuation_boundary
class C1Tests(unittest.TestCase):
 def test_constant_velocity(self):
  pre=torch.zeros(1,263,1,20);suf=torch.zeros(1,263,1,30);pre[:,3:193]=torch.arange(20.).reshape(1,1,1,-1);suf[:,3:193]=100-torch.arange(30.).reshape(1,1,1,-1)
  pos=smooth_continuation_boundary(pre,suf,12);c1=smooth_continuation_boundary(pre,suf,12,'c1');self.assertTrue(torch.equal(c1[:,:3],pos[:,:3]));self.assertTrue(torch.equal(c1[...,0],pos[...,0]));self.assertLess(float(torch.abs((c1[:,3:193,:,1]-c1[:,3:193,:,0])-1).max()),.05);self.assertTrue(torch.equal(c1[...,12:],suf[...,12:]))
 def test_default_unchanged(self):
  p=torch.randn(2,263,1,20);s=torch.randn(2,263,1,60);self.assertTrue(torch.equal(smooth_continuation_boundary(p,s,12),smooth_continuation_boundary(p,s,12,'position')))
 def test_short_and_invalid(self):
  p=torch.zeros(1,263,1,1);s=torch.ones(1,263,1,1);self.assertTrue(torch.isfinite(smooth_continuation_boundary(p,s,1,'c1')).all())
  with self.assertRaises(ValueError):smooth_continuation_boundary(p,s,0,'bad')
