import torch


class Rotation2xyz:
    """Minimal stub used by the inference-only package."""

    def __init__(self, device, dataset="humanml"):
        self.device = device
        self.dataset = dataset

    def __call__(
        self,
        x,
        mask,
        pose_rep,
        translation,
        glob,
        jointstype,
        vertstrans,
        betas=None,
        beta=0,
        glob_rot=None,
        get_rotations_back=False,
        **kwargs,
    ):
        del mask, translation, glob, jointstype, vertstrans, betas, beta, glob_rot, kwargs
        if pose_rep != "xyz":
            raise NotImplementedError("rotation2xyz stub only supports pose_rep='xyz'")
        if get_rotations_back:
            return x, None, None
        return x
