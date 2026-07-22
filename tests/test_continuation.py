import unittest

import torch

from data_loaders.humanml.scripts.motion_process import recover_from_ric_stateful
from utils.sampler_util import smooth_continuation_boundary


class ContinuationTests(unittest.TestCase):
    def test_boundary_smoothing_matches_velocity_and_fades_to_sample(self):
        prefix = torch.zeros(1, 263, 1, 3)
        prefix[..., -2] = 1.0
        prefix[..., -1] = 2.0
        suffix = torch.full((1, 263, 1, 6), 10.0)

        smoothed = smooth_continuation_boundary(prefix, suffix, blend_frames=4)

        # Dynamic channels hold their last value; state-like channels extrapolate one step.
        self.assertTrue(torch.equal(smoothed[:, :3, :, 0], prefix[:, :3, :, -1]))
        self.assertTrue(torch.equal(smoothed[:, 3:193, :, 0], torch.full((1, 190, 1), 3.0)))
        self.assertTrue(torch.equal(smoothed[:, 193:, :, 0], prefix[:, 193:, :, -1]))
        self.assertTrue(torch.allclose(smoothed[..., 3], suffix[..., 3]))
        self.assertTrue(torch.equal(smoothed[..., 4:], suffix[..., 4:]))

    def test_stateful_decode_matches_single_pass_decode(self):
        torch.manual_seed(7)
        hml_vec = torch.randn(1, 1, 10, 263) * 0.02
        hml_vec[..., 3] = 0.9

        full_motion, full_state = recover_from_ric_stateful(hml_vec, 22)
        first_motion, first_state = recover_from_ric_stateful(hml_vec[..., :4, :], 22)
        second_motion, second_state = recover_from_ric_stateful(
            hml_vec[..., 4:, :],
            22,
            state=first_state,
        )
        chained_motion = torch.cat([first_motion, second_motion], dim=-3)

        self.assertTrue(torch.allclose(chained_motion, full_motion, atol=1e-6))
        self.assertTrue(
            torch.allclose(second_state["root_rot_ang"], full_state["root_rot_ang"], atol=1e-6)
        )
        self.assertTrue(torch.allclose(second_state["root_pos"], full_state["root_pos"], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
