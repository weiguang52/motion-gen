import unittest

import numpy as np
import torch
import torch.nn as nn

from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)


class _ZeroEpsilonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x, timesteps, **kwargs):
        return torch.zeros_like(x) + self.anchor


class DiffusionInpaintingTests(unittest.TestCase):
    def _diffusion(self):
        return GaussianDiffusion(
            betas=np.linspace(0.0001, 0.02, 20, dtype=np.float64),
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
        )

    def test_logsnr_sampling_preserves_known_prefix_exactly(self):
        diffusion = self._diffusion()
        model = _ZeroEpsilonModel()
        shape = (1, 3, 1, 8)
        known_motion = torch.zeros(shape)
        known_motion[..., :2] = torch.tensor([1.25, -0.75])
        known_mask = torch.zeros(shape, dtype=torch.bool)
        known_mask[..., :2] = True

        sample = diffusion.logsnr_sample_loop(
            model,
            shape,
            steps=5,
            known_motion=known_motion,
            known_mask=known_mask,
        )

        self.assertEqual(tuple(sample.shape), shape)
        self.assertTrue(torch.equal(sample[..., :2], known_motion[..., :2]))
        self.assertTrue(torch.isfinite(sample).all().item())

    def test_known_motion_and_mask_must_be_provided_together(self):
        diffusion = self._diffusion()
        model = _ZeroEpsilonModel()

        with self.assertRaisesRegex(ValueError, "provided together"):
            diffusion.p_sample_loop(
                model,
                (1, 1, 1, 4),
                known_motion=torch.zeros(1, 1, 1, 4),
            )


if __name__ == "__main__":
    unittest.main()
