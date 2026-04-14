"""
Brief:
提供采样包装器与自回归采样工具，主要为多个模块提供可复用的基础能力。
"""
import torch
import torch.nn as nn
from copy import deepcopy
from utils.wrapper_util import wrapped_getattr


def _get_text_embed_item(text_embed, new_key, legacy_key):
    if new_key in text_embed:
        return text_embed[new_key]
    return text_embed.get(legacy_key)


def repeat_text_embed_for_dynamic(text_embed, num_samples):
    if isinstance(text_embed, dict):
        repeated = {}
        ctxt = _get_text_embed_item(text_embed, 'ctxt', 'context_tokens')
        ctxt_mask = _get_text_embed_item(text_embed, 'ctxt_mask', 'context_mask')
        vtxt = _get_text_embed_item(text_embed, 'vtxt', 'global_text')
        if ctxt is not None:
            repeated['ctxt'] = ctxt.unsqueeze(0).repeat(num_samples, 1, 1, 1)
        if ctxt_mask is not None:
            repeated['ctxt_mask'] = ctxt_mask.unsqueeze(0).repeat(num_samples, 1, 1)
        if vtxt is not None:
            repeated['vtxt'] = vtxt.unsqueeze(0).repeat(num_samples, 1, 1, 1)
        return repeated
    return (
        text_embed[0].unsqueeze(0).repeat(num_samples, 1, 1, 1),
        text_embed[1].unsqueeze(0).repeat(num_samples, 1, 1),
    )


def select_dynamic_text_embed_step(text_embed, step_idx):
    if isinstance(text_embed, dict):
        step_embed = {}
        ctxt = _get_text_embed_item(text_embed, 'ctxt', 'context_tokens')
        ctxt_mask = _get_text_embed_item(text_embed, 'ctxt_mask', 'context_mask')
        vtxt = _get_text_embed_item(text_embed, 'vtxt', 'global_text')
        if ctxt is not None:
            step_embed['ctxt'] = ctxt[:, :, step_idx]
        if ctxt_mask is not None:
            step_embed['ctxt_mask'] = ctxt_mask[:, step_idx]
        if vtxt is not None:
            step_embed['vtxt'] = vtxt[:, :, step_idx]
        return step_embed
    return (
        text_embed[0][:, :, step_idx],
        text_embed[1][:, step_idx],
    )


def select_dynamic_text_embed_indices(text_embed, step_indices):
    if isinstance(text_embed, dict):
        step_embed = {}
        ctxt = _get_text_embed_item(text_embed, 'ctxt', 'context_tokens')
        ctxt_mask = _get_text_embed_item(text_embed, 'ctxt_mask', 'context_mask')
        vtxt = _get_text_embed_item(text_embed, 'vtxt', 'global_text')
        if ctxt is not None:
            step_embed['ctxt'] = torch.stack(
                [ctxt[sample_i, :, idx] for sample_i, idx in enumerate(step_indices)],
                dim=1,
            )
        if ctxt_mask is not None:
            step_embed['ctxt_mask'] = torch.stack(
                [ctxt_mask[sample_i, idx] for sample_i, idx in enumerate(step_indices)],
                dim=0,
            )
        if vtxt is not None:
            step_embed['vtxt'] = torch.stack(
                [vtxt[sample_i, :, idx] for sample_i, idx in enumerate(step_indices)],
                dim=1,
            )
        return step_embed
    return (
        torch.stack(
            [text_embed[0][sample_i, :, idx] for sample_i, idx in enumerate(step_indices)],
            dim=1,
        ),
        torch.stack(
            [text_embed[1][sample_i, idx] for sample_i, idx in enumerate(step_indices)],
            dim=0,
        ),
    )


def blend_dynamic_text_embed(prev_embed, cur_embed, blend):
    blend = max(0.0, min(1.0, float(blend)))
    if blend == 0.0:
        return cur_embed
    if isinstance(cur_embed, dict):
        mixed = dict(cur_embed)
        prev_ctxt = _get_text_embed_item(prev_embed, 'ctxt', 'context_tokens')
        cur_ctxt = _get_text_embed_item(cur_embed, 'ctxt', 'context_tokens')
        prev_mask = _get_text_embed_item(prev_embed, 'ctxt_mask', 'context_mask')
        cur_mask = _get_text_embed_item(cur_embed, 'ctxt_mask', 'context_mask')
        prev_vtxt = _get_text_embed_item(prev_embed, 'vtxt', 'global_text')
        cur_vtxt = _get_text_embed_item(cur_embed, 'vtxt', 'global_text')
        if prev_ctxt is not None and cur_ctxt is not None:
            mixed['ctxt'] = (blend * prev_ctxt) + ((1.0 - blend) * cur_ctxt)
        if prev_mask is not None and cur_mask is not None:
            mixed['ctxt_mask'] = cur_mask if blend < 0.5 else (cur_mask & prev_mask)
        if prev_vtxt is not None and cur_vtxt is not None:
            mixed['vtxt'] = (blend * prev_vtxt) + ((1.0 - blend) * cur_vtxt)
        return mixed
    prev_tokens, prev_mask = prev_embed
    cur_tokens, cur_mask = cur_embed
    mixed_tokens = (blend * prev_tokens) + ((1.0 - blend) * cur_tokens)
    mixed_mask = cur_mask if blend < 0.5 else (cur_mask & prev_mask)
    return mixed_tokens, mixed_mask


# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        self.rot2xyz = self.model.rot2xyz
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = self.model.cond_mode
        self.encode_text = self.model.encode_text

    def forward(self, x, timesteps, y=None):
        cond_mode = self.model.cond_mode
        assert cond_mode in ['text', 'no_cond']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))

    def __getattr__(self, name, default=None):
        # this method is reached only if name is not in self.__dict__.
        return wrapped_getattr(self, name, default=None)


class AutoRegressiveSampler():
    def __init__(self, args, sample_fn, required_frames=196):
        self.sample_fn = sample_fn
        self.args = args
        self.required_frames = required_frames
    
    def sample(self, model, shape, **kargs):
        samples_buf = []
        cur_prefix = deepcopy(kargs['model_kwargs']['y'].get('prefix'))
        if self.args.autoregressive_include_prefix and cur_prefix is not None:
            samples_buf.append(cur_prefix)
        for chunk, _ in self.iter_chunks(model, shape, **kargs):
            samples_buf.append(chunk)

        full_batch = torch.cat(samples_buf, dim=-1)[..., :self.required_frames]  # 200 -> 196
        return full_batch

    def iter_chunks(self, model, shape, **kargs):
        n_iterations = (self.required_frames // self.args.pred_len) + int(self.required_frames % self.args.pred_len > 0)
        cur_prefix = deepcopy(kargs['model_kwargs']['y'].get('prefix'))
        dynamic_text_mode = isinstance(kargs['model_kwargs']['y']['text'][0], list)
        autoregressive_shape = list(deepcopy(shape))
        autoregressive_shape[-1] = self.args.pred_len
        prev_chunk = None

        for i in range(n_iterations):
            cur_kargs = deepcopy(kargs)
            if cur_prefix is None:
                cur_kargs['model_kwargs']['y'].pop('prefix', None)
            else:
                cur_kargs['model_kwargs']['y']['prefix'] = cur_prefix
            cur_text = cur_kargs['model_kwargs']['y']['text']

            if dynamic_text_mode:
                seg_idx = [min(i, len(s) - 1) for s in kargs['model_kwargs']['y']['text']]
                cur_text = [s[idx] for s, idx in zip(kargs['model_kwargs']['y']['text'], seg_idx)]
                cur_kargs['model_kwargs']['y']['text'] = cur_text
                if model.text_encoder_type != 'bert_clip':
                    raise NotImplementedError(
                        'Dynamic text sampling currently supports only bert_clip text encoders.'
                    )
                text_embed = cur_kargs['model_kwargs']['y']['text_embed']
                cur_embed = select_dynamic_text_embed_indices(text_embed, seg_idx)
                blend = float(getattr(self.args, 'dynamic_text_blend', 0.0) or 0.0)
                if blend > 0.0:
                    prev_idx = [max(0, idx - 1) for idx in seg_idx]
                    prev_embed = select_dynamic_text_embed_indices(text_embed, prev_idx)
                    cur_embed = blend_dynamic_text_embed(prev_embed, cur_embed, blend)
                cur_kargs['model_kwargs']['y']['text_embed'] = cur_embed

            sample = self.sample_fn(model, autoregressive_shape, **cur_kargs)
            chunk = sample.clone()[..., -self.args.pred_len:]
            if prev_chunk is not None:
                chunk = self._blend_chunk_motion(prev_chunk, chunk)
            prev_chunk = chunk
            cur_prefix = sample.clone()[..., -self.args.context_len:]
            yield chunk, cur_text

    def _blend_chunk_motion(self, prev_chunk, cur_chunk):
        blend_frames = int(getattr(self.args, 'autoregressive_motion_blend_frames', 0) or 0)
        blend_frames = min(blend_frames, prev_chunk.shape[-1], cur_chunk.shape[-1])
        if blend_frames <= 0:
            return cur_chunk

        blended = cur_chunk.clone()
        if prev_chunk.shape[-1] >= 2:
            last_frame = prev_chunk[..., -1:]
            prev_frame = prev_chunk[..., -2:-1]
            velocity = last_frame - prev_frame
            steps = torch.arange(
                1,
                blend_frames + 1,
                device=cur_chunk.device,
                dtype=cur_chunk.dtype,
            ).view(1, 1, 1, blend_frames)
            motion_anchor = last_frame + (velocity * steps)
        else:
            motion_anchor = prev_chunk[..., -1:].expand(-1, -1, -1, blend_frames)

        alpha = torch.linspace(
            1.0 / blend_frames,
            1.0,
            blend_frames,
            device=cur_chunk.device,
            dtype=cur_chunk.dtype,
        ).view(1, 1, 1, blend_frames)
        blended[..., :blend_frames] = ((1.0 - alpha) * motion_anchor) + (
            alpha * cur_chunk[..., :blend_frames]
        )
        return blended
