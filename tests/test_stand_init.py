"""Standing initialization contracts, without loading model weights."""
import unittest,io
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np,torch
class StandTests(unittest.TestCase):
 def state(self,history=False):
  class Model:
   njoints=263;nfeats=1
   def encode_text(self,text):return torch.zeros(len(text),1)
  class Diffusion:
   def logsnr_sample_loop(self,model,shape,**kw):return torch.zeros(shape)
  return SimpleNamespace(args=SimpleNamespace(arch='dit',autoregressive=False,context_len=0,pred_len=0,guidance_param=1,seed=1,sampling_steps=50),model=Model(),guided_model=None,diffusion=Diffusion(),mean=np.zeros(263),std=np.ones(263),max_frames=196,fps=20,continuation_prefix_frames=20,continuation_blend_frames=12,session_motions={'s':torch.zeros(1,263,1,20)} if history else {},session_decode_states={})
 def test_request_init_and_manual_state_are_equal(self):
  import serve.generate_server as gs
  pre=torch.zeros(1,263,1,20);pre[:,3:193]=.1
  dec={'root_rot_ang':torch.zeros(1,1),'root_pos':torch.zeros(1,1,3)}
  a=self.state();b=self.state();b.session_motions['s']=pre;b.session_decode_states['s']=dec
  with patch('infer._load_continuation_prefix',return_value=(pre,dec)) as load,patch.object(gs.dist_util,'dev',return_value=torch.device('cpu')):
   x=np.load(io.BytesIO(gs._sample(a,gs.GenerateRequest(text='wave',num_frames=60,session_id='s',init_prefix='stand_mean'))),allow_pickle=True).item()
   y=np.load(io.BytesIO(gs._sample(b,gs.GenerateRequest(text='wave',num_frames=60,session_id='s'))),allow_pickle=True).item()
  load.assert_called_once();np.testing.assert_array_equal(x['motion'],y['motion']);self.assertEqual(x['motion'].shape[-1],60);self.assertEqual(a.session_motions['s'].shape[-1],60)
 def test_existing_history_not_replaced(self):
  import serve.generate_server as gs
  with patch('infer._load_continuation_prefix') as load,patch.object(gs.dist_util,'dev',return_value=torch.device('cpu')):
   gs._sample(self.state(True),gs.GenerateRequest(text='wave',num_frames=60,session_id='s',init_prefix='stand_data'))
  load.assert_not_called()
 def test_default_off_and_invalid_mode(self):
  from serve.generate_server import GenerateRequest
  self.assertIsNone(GenerateRequest(text='wave').init_prefix)
  with self.assertRaises(ValueError):GenerateRequest(text='wave',init_prefix='unknown')
 def test_requires_continuation(self):
  import serve.generate_server as gs
  with self.assertRaises(gs.HTTPException):
   gs._sample(self.state(),gs.GenerateRequest(text='wave',num_frames=60,init_prefix='stand_mean',continue_from_previous=False))
