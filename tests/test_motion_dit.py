import unittest

import torch

from model.backbones.ar_dit import MotionARDiT
from model.backbones.dit import MotionDiT


class MotionDiTTests(unittest.TestCase):
    def test_pure_dit_cross_attn_forward_is_finite(self):
        model = MotionDiT(hidden_size=8, ff_size=16, depth=2, num_heads=2)
        x = torch.randn(6, 2, 8)
        cond = torch.randn(2, 8)
        key_padding_mask = torch.tensor(
            [
                [False, False, False, False, True, True],
                [False, False, False, False, False, False],
            ]
        )
        context = torch.randn(3, 2, 8)
        context_mask = torch.tensor(
            [
                [False, False, True],
                [False, False, False],
            ]
        )

        out = model(
            x,
            cond,
            key_padding_mask=key_padding_mask,
            context=context.permute(1, 0, 2),
            context_mask=context_mask,
        )

        self.assertEqual(tuple(out.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_ar_dit_cross_attn_forward_is_finite(self):
        model = MotionARDiT(hidden_size=8, ff_size=16, depth=2, num_heads=2)
        x = torch.randn(6, 2, 8)
        cond = torch.randn(2, 8)
        key_padding_mask = torch.tensor(
            [
                [False, False, False, False, True, True],
                [False, False, False, False, False, False],
            ]
        )
        context = torch.randn(5, 2, 8)
        context_mask = torch.tensor(
            [
                [False, False, False, True, True],
                [False, False, False, False, False],
            ]
        )

        out = model(
            x,
            cond,
            key_padding_mask=key_padding_mask,
            context=context.permute(1, 0, 2),
            context_mask=context_mask,
        )

        self.assertEqual(tuple(out.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(out).all().item())


if __name__ == "__main__":
    unittest.main()
