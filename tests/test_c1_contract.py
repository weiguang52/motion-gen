import unittest,io,copy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np,torch
from utils.sampler_util import smooth_continuation_boundary
class ContractTests(unittest.TestCase):
 def test_cli_and_request_defaults(self):
  import infer
  from serve.generate_server import GenerateRequest
  import tempfile,json
  from pathlib import Path
  with tempfile.TemporaryDirectory() as temp:
   Path(temp,'args.json').write_text(json.dumps({'arch':'dit'}))
   argv=['infer.py','--model_path',str(Path(temp,'model.pt')),'--text_prompt','A person waves the right hand.']
   with patch('sys.argv',argv):self.assertEqual(infer._build_args().continuation_blend_mode,'position')
   with patch('sys.argv',argv+['--continuation_blend_mode','c1']):self.assertEqual(infer._build_args().continuation_blend_mode,'c1')
  self.assertEqual(GenerateRequest(text='wave').blend_mode,'position')
  with self.assertRaises(ValueError):GenerateRequest(text='wave',blend_mode='bad')
 def test_service_frozen_sampler_routes_c1(self):
  import serve.generate_server as gs
  pre=torch.zeros(1,263,1,20);pre[:,3:193]=torch.arange(20.).view(1,1,1,-1)
  class Model:
   njoints=263;nfeats=1
   def encode_text(self,text):return torch.zeros(len(text),1)
  class Diffusion:
   def logsnr_sample_loop(self,model,shape,**kw):
    sample=torch.zeros(shape);sample[:,3:193]=100-torch.arange(shape[-1]).view(1,1,1,-1);return sample
  args=SimpleNamespace(arch='dit',autoregressive=False,context_len=0,pred_len=0,guidance_param=1,seed=1,sampling_steps=50)
  state=SimpleNamespace(args=args,model=Model(),guided_model=None,diffusion=Diffusion(),mean=np.zeros(263),std=np.ones(263),max_frames=196,fps=20,continuation_prefix_frames=20,continuation_blend_frames=12,session_motions={'test':pre},session_decode_states={})
  raw=Diffusion().logsnr_sample_loop(None,(1,263,1,80))
  expected=smooth_continuation_boundary(pre,raw[...,20:],12,'c1')
  with patch.object(gs.dist_util,'dev',return_value=torch.device('cpu')):
   d=np.load(io.BytesIO(gs._sample(state,gs.GenerateRequest(text='wave',num_frames=60,session_id='test',blend_mode='c1'))),allow_pickle=True).item()
  self.assertEqual(d['continuation_blend_mode'],'c1');np.testing.assert_allclose(d['hml_vec'],expected.squeeze(2).numpy(),atol=1e-5);self.assertEqual(state.session_motions['test'].shape[-1],60)
