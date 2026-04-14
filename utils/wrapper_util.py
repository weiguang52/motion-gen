"""
Brief:
提供轻量包装器辅助函数，主要服务于 model wrapper 的属性转发。
"""
import torch


def wrapped_getattr(self, name, default=None, wrapped_member_name='model'):
    """Forward missing attributes to the wrapped inner object."""
    if isinstance(self, torch.nn.Module):
        try:
            attr = torch.nn.Module.__getattr__(self, name)
        except AttributeError:
            wrapped_member = torch.nn.Module.__getattr__(self, wrapped_member_name)
            attr = getattr(wrapped_member, name, default)
    else:
        wrapped_member = getattr(self, wrapped_member_name)
        attr = getattr(wrapped_member, name, default)
    return attr
