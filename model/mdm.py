"""
Brief:
定义主 MDM 文本动作生成模型，主要服务于文本到动作模型的前向计算与表示变换。
"""
import torch
import torch.nn as nn
from model.backbones.ar_dit import MotionARDiT
from model.backbones.dit import MotionDiT
from model.components.mdm_modules import (
    InputProcess,
    OutputProcess,
    EmbedTargetLocSingle,
    EmbedTargetLocSplit,
    EmbedTargetLocMulti,
)
from model.components.time_encoding import PositionalEncoding, build_timestep_embedder
from model.conditioning.prefix_conditioning import (
    PrefixConditioning,
    compose_cross_context,
    should_use_prefix_cross_context,
)
from model.conditioning.text_conditioning_stack import TextConditioningStack
from model.geometry.rotation2xyz import Rotation2xyz


class MDM(nn.Module):
    def __init__(self, njoints, nfeats, translation, pose_rep, glob,
                 model_dim=512, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", data_rep='rot6d', dataset='humanml',
                 arch='dit', clip_version='ViT-B/32', **kargs):
        super().__init__()

        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep
        self.dataset = dataset
        
        self.pose_rep = pose_rep
        self.glob = glob
        self.translation = translation

        self.model_dim = model_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.input_feats = self.njoints * self.nfeats

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.1)
        self.arch = arch
        self.ar_dit_dual_cross_attn = bool(kargs.get('ar_dit_dual_cross_attn', False))
        self.input_process = InputProcess(self.data_rep, self.input_feats, self.model_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.model_dim, self.dropout, max_len=kargs.get('pos_embed_max_len', 5000))

        self.pred_len = kargs.get('pred_len', 0)
        self.context_len = kargs.get('context_len', 0)
        self.total_len = self.pred_len + self.context_len
        self.is_prefix_comp = self.total_len > 0
        self.all_goal_joint_names = kargs.get('all_goal_joint_names', [])
        
        self.multi_target_cond = kargs.get('multi_target_cond', False)
        self.multi_encoder_type = kargs.get('multi_encoder_type', 'multi')
        self.target_enc_layers = kargs.get('target_enc_layers', 1)
        self._init_target_cond()
        self._init_backbone(kargs)

        self.embed_timestep = build_timestep_embedder(
            self.model_dim,
            self.sequence_pos_encoder,
            mode=kargs.get('time_embed_mode', 'fm'),
            num_timesteps=kargs.get('diffusion_steps', 1000),
            rescale_timesteps=kargs.get('rescale_timesteps', False),
        )

        self.text_conditioning = TextConditioningStack(self)
        self._init_conditioning(kargs, clip_version)
        self._init_prefix_conditioning()

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.model_dim, self.njoints,
                                            self.nfeats)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        # Kept for backward compatibility with existing call sites.
        return [p for name, p in self.named_parameters() if not self._is_language_parameter_name(name)]

    @staticmethod
    def _is_language_parameter_name(name):
        # Exclude language backbones (CLIP/BERT), including legacy attribute prefixes.
        return (
            name.startswith('clip_text_model.')
            or name.startswith('clip_model.')
            or name.startswith('global_clip_model.')
            or name.startswith('bert_text_model.')
            or name.startswith('bert_model.')
            or name.startswith('text_conditioner.clip_text_model.')
            or name.startswith('text_conditioner.bert_text_model.')
        )

    def _init_target_cond(self):
        if self.multi_target_cond:
            if self.multi_encoder_type == 'multi':
                self.embed_target_cond = EmbedTargetLocMulti(self.all_goal_joint_names, self.model_dim)
            elif self.multi_encoder_type == 'single':
                self.embed_target_cond = EmbedTargetLocSingle(self.all_goal_joint_names, self.model_dim, self.target_enc_layers)
            elif self.multi_encoder_type == 'split':
                self.embed_target_cond = EmbedTargetLocSplit(self.all_goal_joint_names, self.model_dim, self.target_enc_layers)

    def _init_backbone(self, kargs):
        if self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.model_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'dit':
            print("DiT init")
            self.seqDiT = MotionDiT(
                hidden_size=self.model_dim,
                ff_size=self.ff_size,
                depth=self.num_layers,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )
        elif self.arch == 'ar_dit':
            print("AR DiT init")
            self.seqDiT = MotionARDiT(
                hidden_size=self.model_dim,
                ff_size=self.ff_size,
                depth=self.num_layers,
                num_heads=self.num_heads,
                dropout=self.dropout,
                use_dual_cross_attn=self.ar_dit_dual_cross_attn,
            )
        else:
            raise ValueError('Please choose correct architecture [trans_dec, dit, ar_dit]')

    def _init_conditioning(self, kargs, clip_version):
        self.text_conditioning.init_conditioning(kargs, clip_version)

    def _init_prefix_conditioning(self):
        self.prefix_conditioning = None
        if should_use_prefix_cross_context(self.arch, self.is_prefix_comp, self.context_len):
            self.prefix_conditioning = PrefixConditioning(
                input_feats=self.input_feats,
                model_dim=self.model_dim,
                context_len=self.context_len,
                pred_len=self.pred_len,
                num_heads=self.num_heads,
                ff_size=self.ff_size,
                dropout=self.dropout,
            )

    def _use_prefix_as_cross_context(self):
        return self.prefix_conditioning is not None

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[-2]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def _apply_target_cond(self, time_emb, y):
        if 'target_cond' in y.keys():
            # NOTE: We don't use CFG for joints - but we do wat to support uncond sampling for generation and eval!
            time_emb += self.mask_cond(self.embed_target_cond(y['target_cond'], y['target_joint_names'], y['is_heading'])[None], force_mask=y.get('target_uncond', False))  # For uncond support and CFG
            # time_emb += self.embed_target_cond(y['target_cond'], y['target_joint_names'], y['is_heading'])[None]
        return time_emb

    def _apply_prefix_completion(self, x, y, bs):
        if self.is_prefix_comp:
            x = self.prefix_conditioning.apply_prefix_completion(
                x,
                y,
                bs,
                is_prefix_comp=self.is_prefix_comp,
                use_cross_context=self._use_prefix_as_cross_context(),
            ) if self.prefix_conditioning is not None else x
            if self.prefix_conditioning is None and not self._use_prefix_as_cross_context():
                x = torch.cat([y['prefix'], x], dim=-1)
                y['mask'] = torch.cat([torch.ones([bs, 1, 1, self.context_len], dtype=y['mask'].dtype, device=y['mask'].device),
                                       y['mask']], dim=-1)
        return x

    def _build_cond_emb(self, time_emb, y, bs):
        return self.text_conditioning.build_condition_embeddings(
            time_emb,
            y,
            bs,
            mask_cond_fn=self.mask_cond,
        )

    def _build_frames_mask(self, y, x):
        frames_mask = None
        is_valid_mask = y['mask'].shape[-1] > 1  # Don't use mask with the generate script
        if is_valid_mask:
            frames_mask = torch.logical_not(y['mask'][..., :x.shape[0]].squeeze(1).squeeze(1)).to(device=x.device)
        return frames_mask

    def _build_prefix_context(self, y):
        if self.prefix_conditioning is None:
            return None, None
        return self.prefix_conditioning.build_prefix_context(
            y,
            use_cross_context=self._use_prefix_as_cross_context(),
        )

    def _compose_cross_context(self, text_emb, text_mask, prefix_context, prefix_mask):
        return compose_cross_context(text_emb, text_mask, prefix_context, prefix_mask)

    def _forward_trans_dec(self, x, cond_emb, frames_mask, text_mask):
        xseq = x
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqTransDecoder(
            tgt=xseq,
            memory=cond_emb,
            memory_key_padding_mask=text_mask,
            tgt_key_padding_mask=frames_mask,
        )  # Rotem's bug fix
        return output

    def _build_transformer_condition_vector(self, global_cond, text_emb, text_mask, text_global_emb):
        cond_vec = self._compute_dit_condition(global_cond, None)
        if text_global_emb is not None:
            pooled_text = self._compute_dit_condition(text_global_emb, None)
            cond_vec = cond_vec + pooled_text
        elif text_emb is not None:
            pooled_text = self._compute_dit_condition(text_emb, text_mask)
            cond_vec = cond_vec + pooled_text
        return cond_vec

    def _build_transformer_context(
        self,
        text_emb,
        text_mask,
        prefix_context=None,
        prefix_mask=None,
        include_prefix=False,
    ):
        if include_prefix:
            return self._compose_cross_context(text_emb, text_mask, prefix_context, prefix_mask)
        if text_emb is None:
            return None, None
        return text_emb.permute(1, 0, 2), text_mask

    def _forward_dit(self, x, cond_emb, global_cond, frames_mask, text_mask, text_emb, text_global_emb):
        del cond_emb
        xseq = x
        xseq = self.sequence_pos_encoder(xseq)
        cond_vec = self._build_transformer_condition_vector(global_cond, text_emb, text_mask, text_global_emb)
        ctx, ctx_mask = self._build_transformer_context(text_emb, text_mask)

        output = self.seqDiT(xseq, cond_vec, key_padding_mask=frames_mask, context=ctx, context_mask=ctx_mask)
        return output

    def _forward_ar_dit(
        self,
        x,
        cond_emb,
        global_cond,
        frames_mask,
        text_mask,
        text_emb,
        text_global_emb,
        prefix_context,
        prefix_mask,
    ):
        del cond_emb
        xseq = x
        xseq = self.sequence_pos_encoder(xseq)
        cond_vec = self._build_transformer_condition_vector(global_cond, text_emb, text_mask, text_global_emb)
        if self.ar_dit_dual_cross_attn:
            text_ctx, text_ctx_mask = self._build_transformer_context(text_emb, text_mask)
            output = self.seqDiT(
                xseq,
                cond_vec,
                key_padding_mask=frames_mask,
                text_context=text_ctx,
                text_context_mask=text_ctx_mask,
                prefix_context=prefix_context,
                prefix_context_mask=prefix_mask,
            )
        else:
            ctx, ctx_mask = self._build_transformer_context(
                text_emb,
                text_mask,
                prefix_context=prefix_context,
                prefix_mask=prefix_mask,
                include_prefix=True,
            )
            output = self.seqDiT(
                xseq,
                cond_vec,
                key_padding_mask=frames_mask,
                context=ctx,
                context_mask=ctx_mask,
            )
        return output

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] continuous diffusion time in [0, 1]
        """
        bs, njoints, nfeats, nframes = x.shape
        time_emb = self.embed_timestep(timesteps)  # [1, bs, d]

        time_emb = self._apply_target_cond(time_emb, y)
        x = self._apply_prefix_completion(x, y, bs)
        cond_emb, text_emb, text_mask, global_cond, text_global_emb = self._build_cond_emb(time_emb, y, bs)
        prefix_context, prefix_mask = self._build_prefix_context(y)

        x = self.input_process(x)

        frames_mask = self._build_frames_mask(y, x)

        if self.arch == 'trans_dec':
            output = self._forward_trans_dec(x, cond_emb, frames_mask, text_mask)

        elif self.arch == 'dit':
            output = self._forward_dit(
                x, cond_emb, global_cond, frames_mask, text_mask, text_emb, text_global_emb
            )
        elif self.arch == 'ar_dit':
            output = self._forward_ar_dit(
                x,
                cond_emb,
                global_cond,
                frames_mask,
                text_mask,
                text_emb,
                text_global_emb,
                prefix_context,
                prefix_mask,
            )
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")

        # Extract completed suffix
        if self.is_prefix_comp and not self._use_prefix_as_cross_context():
            output = output[self.context_len:]
            y['mask'] = y['mask'][..., self.context_len:]
        
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output


    def _compute_dit_condition(self, cond_emb, text_mask):
        """
        Reduce (seq_len, batch, dim) conditions to a single vector per sample for DiT blocks.
        """
        if cond_emb.dim() == 3:  # [seq, bs, dim]
            cond_tokens = cond_emb.permute(1, 0, 2)  # [bs, seq, dim]
            # cond = self.cond_pool(cond_tokens, mask=text_mask)  # [bs, dim]
            if text_mask is not None:
                # text_mask True -> padding
                valid = (~text_mask).float().unsqueeze(-1)  # [bs, seq, 1]
                denom = valid.sum(dim=1).clamp(min=1e-6)  # [bs, 1, 1] broadcast
                cond = (cond_tokens * valid).sum(dim=1) / denom  # [bs, dim]
            else:
                cond = cond_tokens.mean(dim=1)  # [bs, dim]
        elif cond_emb.dim() == 2:
            cond = cond_emb  # [bs, dim]
        else:
            cond = cond_emb.squeeze(0)  # [bs, dim]
        return cond

    def clear_text_embedding_cache(self):
        self.text_conditioning.clear_cache()


    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self.rot2xyz, "smpl_model"):
            self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if hasattr(self.rot2xyz, "smpl_model"):
            self.rot2xyz.smpl_model.train(*args, **kwargs)
