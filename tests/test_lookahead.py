import ast,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

class LookaheadTests(unittest.TestCase):
 def test_cli_real_cropping_before_smoothing(self):
  tree=ast.parse(Path('infer.py').read_text())
  nodes=[]
  for n in ast.walk(tree):
   if isinstance(n,ast.If) and ast.unparse(n.test)=="args.arch == 'dit'" and 'prefix_frames + n_frames' in ast.unparse(n): nodes.append(n)
  self.assertEqual(len(nodes),1)
  for p,n,l in ((0,20,40),(20,60,40),(40,20,0)):
   sample=torch.arange(p+n+l).reshape(1,1,1,-1)
   env=dict(args=type('Args',(),{'arch':'dit'})(),sample=sample,prefix_frames=p,n_frames=n)
   exec(compile(ast.Module(body=nodes,type_ignores=[]),'crop','exec'),env)
   self.assertEqual(env['sample'][...,p:].shape[-1],n)
   self.assertEqual(env['sample'][...,-1].item(),p+n-1)
 def test_server_crops_before_session_and_decode(self):
  s=Path('serve/generate_server.py').read_text(); crop=s.index('sample = sample[..., :prefix_frames + n_frames]')
  self.assertLess(crop,s.index('state.session_motions[req.session_id] = sample'))
  self.assertLess(crop,s.index('motions, hml_vec, next_decode_state ='))
 def test_request_rejects_negative_lookahead(self):
  from serve.generate_server import GenerateRequest
  with self.assertRaises(ValueError): GenerateRequest(text='wave',lookahead_frames=-1)
  self.assertTrue(GenerateRequest(text='wave',is_final=True).is_final)

if __name__=='__main__':unittest.main()
