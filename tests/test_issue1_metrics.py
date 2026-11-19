import unittest
import numpy as np
from tools.issue1_evaluate import metric,independent
class MetricsTests(unittest.TestCase):
 def data(self,n,height):
  m=np.zeros((1,22,3,n));m[0,2,0,:]=1;m[0,21,1,:]=height;return {'motion':m}
 def test_no_release_is_censored(self):
  r=independent(self.data(60,1));self.assertEqual(r['release_frame'],60);self.assertFalse(r['release_observed'])
 def test_first_five_below_includes_start(self):
  r=independent(self.data(60,-1));self.assertEqual(r['release_frame'],0);self.assertTrue(r['release_observed'])
 def test_raised_stationary_is_failure(self):
  r=metric([self.data(20,1),self.data(60,1),self.data(20,-1)]);self.assertFalse(r['success']);self.assertIn('举着不挥',r['failure_types'])
 def test_repeated_raise_detected(self):
  c=self.data(20,-1);c['motion'][0,21,1,5]=0
  r=metric([self.data(20,1),self.data(60,1),c]);self.assertIn('第三段重复抬手或未放下',r['failure_types'])
