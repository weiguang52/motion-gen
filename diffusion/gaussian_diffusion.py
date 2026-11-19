# This code is based on https://github.com/openai/guided-diffusion
"""
Minimal DDPM inference utilities for DiT / AR-DiT motion generation.
"""
import enum
import math

import numpy as np
import torch as th

from diffusion.spacing import build_spaced_timesteps


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.0):
    if schedule_name == "linear":
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    if schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self in (LossType.KL, LossType.RESCALED_KL)


class GaussianDiffusion:
    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        **kwargs,
    ):
        del kwargs, loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.rescale_timesteps = rescale_timesteps

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped,
            t,
            x_t.shape,
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        """Diffuse a clean sample to timestep ``t``."""
        if noise is None:
            noise = th.randn_like(x_start)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1,
                t,
                x_t.shape,
            )
            * x_t
        )

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def _continuous_model_timestep(self, t):
        if t.is_floating_point():
            return t.to(dtype=th.float32).clamp(0.0, 1.0)
        return t.to(dtype=th.float32) / float(max(self.num_timesteps - 1, 1))

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        batch_size, channels = x.shape[:2]
        assert t.shape == (batch_size,)
        model_output = model(x, self._continuous_model_timestep(t), **model_kwargs)

        if self.model_var_type == ModelVarType.FIXED_LARGE:
            model_variance = np.append(self.posterior_variance[1], self.betas[1:])
            model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        elif self.model_var_type == ModelVarType.FIXED_SMALL:
            model_variance = self.posterior_variance
            model_log_variance = self.posterior_log_variance_clipped
        else:
            raise NotImplementedError("Only fixed variance DDPM inference is supported.")

        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x_start):
            if denoised_fn is not None:
                x_start = denoised_fn(x_start)
            if clip_denoised:
                x_start = x_start.clamp(-1, 1)
            return x_start

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
        elif self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))
        else:
            raise NotImplementedError(self.model_mean_type)

        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        assert model_output.shape[:2] == (batch_size, channels)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        known_motion=None,
        known_mask=None,
        repaint_repeats=1,
    ):
        final = None
        if (
            model_kwargs is not None
            and "y" in model_kwargs
            and "text" in model_kwargs["y"]
            and "text_embed" not in model_kwargs["y"]
        ):
            model_kwargs["y"]["text_embed"] = model.encode_text(model_kwargs["y"]["text"])

        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            known_motion=known_motion,
            known_mask=known_mask,
            repaint_repeats=repaint_repeats,
        ):
            final = sample
        return final["sample"]

    def logsnr_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        steps=100,
        known_motion=None,
        known_mask=None,
        repaint_repeats=1,
    ):
        if device is None:
            device = next(model.parameters()).device
        if model_kwargs is None:
            model_kwargs = {}

        if "y" in model_kwargs and "text" in model_kwargs["y"]:
            model_kwargs["y"]["text_embed"] = model.encode_text(model_kwargs["y"]["text"])

        use_timesteps = build_spaced_timesteps(
            self.num_timesteps,
            steps,
            spacing="logsnr",
            betas=self.betas,
        )

        from diffusion.respace import SpacedDiffusion

        eval_diffusion = SpacedDiffusion(
            use_timesteps=use_timesteps,
            betas=self.betas,
            model_mean_type=self.model_mean_type,
            model_var_type=self.model_var_type,
            loss_type=LossType.MSE,
            rescale_timesteps=self.rescale_timesteps,
        )
        return eval_diffusion.p_sample_loop(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            known_motion=known_motion,
            known_mask=known_mask,
            repaint_repeats=repaint_repeats,
        )

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        known_motion=None,
        known_mask=None,
        repaint_repeats=1,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = noise if noise is not None else th.randn(*shape, device=device)

        if (known_motion is None) != (known_mask is None):
            raise ValueError("known_motion and known_mask must be provided together.")

        known_noise = None
        if known_motion is not None:
            known_motion = known_motion.to(device=device, dtype=img.dtype)
            known_mask = known_mask.to(device=device, dtype=th.bool)
            try:
                known_motion = known_motion.expand(*shape)
                known_mask = known_mask.expand(*shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"known_motion and known_mask must broadcast to sample shape {tuple(shape)}."
                ) from exc
            known_noise = th.randn_like(known_motion)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                if known_motion is not None:
                    known_at_t = self.q_sample(known_motion, t, noise=known_noise)
                    img = th.where(known_mask, known_at_t, img)
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if known_motion is not None and i >= self.num_timesteps // 2 and i > 0:
                    for _ in range(max(0, int(repaint_repeats) - 1)):
                        # Forward one spaced diffusion step, then denoise again.
                        alpha = float(self.alphas_cumprod[i] / self.alphas_cumprod[i - 1])
                        img = out["sample"] * (alpha ** 0.5) + th.randn_like(img) * ((1 - alpha) ** 0.5)
                        img = th.where(known_mask, self.q_sample(known_motion, t, noise=known_noise), img)
                        out = self.p_sample(model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
                if known_motion is not None:
                    if i == 0:
                        known_next = known_motion
                    else:
                        next_t = th.tensor([i - 1] * shape[0], device=device)
                        known_next = self.q_sample(known_motion, next_t, noise=known_noise)
                    out["sample"] = th.where(known_mask, known_next, out["sample"])
                    out["pred_xstart"] = th.where(known_mask, known_motion, out["pred_xstart"])
                yield out
                img = out["sample"]


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
