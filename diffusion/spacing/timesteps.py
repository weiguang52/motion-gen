import numpy as np


def compute_alphas_cumprod(betas):
    betas = np.asarray(betas, dtype=np.float64)
    alphas = 1.0 - betas
    return np.cumprod(alphas, axis=0)


def logsnr_from_alphas_cumprod(alphas_cumprod, eps=1e-12):
    alphas_cumprod = np.asarray(alphas_cumprod, dtype=np.float64)
    return np.log(alphas_cumprod + eps) - np.log(1.0 - alphas_cumprod + eps)


def _ensure_step_count(indices, num_timesteps, num_steps):
    indices = np.clip(np.round(indices).astype(int), 0, num_timesteps - 1)
    indices = np.unique(indices)

    if 0 not in indices:
        indices = np.insert(indices, 0, 0)
    if (num_timesteps - 1) not in indices:
        indices = np.append(indices, num_timesteps - 1)

    if len(indices) < num_steps:
        extra = np.round(np.linspace(0, num_timesteps - 1, num_steps)).astype(int)
        indices = np.unique(np.concatenate([indices, extra]))

    if len(indices) > num_steps:
        pick = np.linspace(0, len(indices) - 1, num_steps)
        indices = indices[np.round(pick).astype(int)]

    return np.sort(indices).tolist()


def build_spaced_timesteps(
    num_timesteps,
    num_steps,
    spacing="uniform",
    betas=None,
    alphas_cumprod=None,
):
    if num_steps >= num_timesteps:
        return list(range(num_timesteps))

    spacing = spacing.lower()
    if spacing not in {"uniform", "quadratic", "cosine", "logsnr"}:
        raise ValueError(f"Unsupported spacing: {spacing}")

    if spacing == "logsnr":
        if alphas_cumprod is None:
            if betas is None:
                raise ValueError("logsnr spacing requires betas or alphas_cumprod.")
            alphas_cumprod = compute_alphas_cumprod(betas)
        logsnr = logsnr_from_alphas_cumprod(alphas_cumprod)

        if np.any(np.diff(logsnr) > 0):
            order = np.argsort(logsnr)
            logsnr = logsnr[order]
            t_values = np.arange(num_timesteps)[order]
        else:
            logsnr = logsnr[::-1]
            t_values = np.arange(num_timesteps)[::-1]

        target = np.linspace(logsnr[0], logsnr[-1], num_steps)
        positions = np.interp(target, logsnr, t_values)
        return _ensure_step_count(positions, num_timesteps, num_steps)

    s = np.linspace(0.0, 1.0, num_steps)
    if spacing == "quadratic":
        s = s ** 2
    elif spacing == "cosine":
        s = 1.0 - np.cos(s * np.pi / 2.0)

    positions = s * (num_timesteps - 1)
    return _ensure_step_count(positions, num_timesteps, num_steps)
