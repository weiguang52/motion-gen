import torch
from data_loaders.humanml.common.quaternion import qinv, qrot


# Expose for gaussian_diffusion imports
qinv = qinv
qrot = qrot


# Recover global angle and positions for rotation dataset
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    # Y-axis rotation from velocity
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    # Add Y-axis rotation to root position
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_root_rot_pos_stateful(data, state=None):
    rot_vel = data[..., 0]
    batch_shape = rot_vel.shape[:-1]

    if state is None:
        init_rot_ang = torch.zeros(batch_shape, device=data.device, dtype=data.dtype)
        init_root_pos = torch.zeros(batch_shape + (3,), device=data.device, dtype=data.dtype)
    else:
        init_rot_ang = state["root_rot_ang"].to(device=data.device, dtype=data.dtype)
        init_root_pos = state["root_pos"].to(device=data.device, dtype=data.dtype)

    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    r_rot_ang[..., 0] = init_rot_ang
    if rot_vel.shape[-1] > 1:
        r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), device=data.device, dtype=data.dtype)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,), device=data.device, dtype=data.dtype)
    if rot_vel.shape[-1] > 1:
        r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos = r_pos + init_root_pos.unsqueeze(-2)
    r_pos[..., 1] = data[..., 3]

    next_root_rot_ang = init_rot_ang + rot_vel.sum(dim=-1)
    next_root_pos = r_pos[..., -1, :].clone()
    next_rot_quat = torch.zeros(batch_shape + (4,), device=data.device, dtype=data.dtype)
    next_rot_quat[..., 0] = torch.cos(next_root_rot_ang)
    next_rot_quat[..., 2] = torch.sin(next_root_rot_ang)
    last_step = torch.zeros(batch_shape + (3,), device=data.device, dtype=data.dtype)
    last_step[..., [0, 2]] = data[..., -1, 1:3]
    last_step = qrot(qinv(next_rot_quat), last_step)
    next_root_pos[..., 0] += last_step[..., 0]
    next_root_pos[..., 2] += last_step[..., 2]
    next_root_pos[..., 1] = data[..., -1, 3]

    next_state = {
        "root_rot_ang": next_root_rot_ang,
        "root_pos": next_root_pos,
    }
    return r_rot_quat, r_pos, next_state


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # Add Y-axis rotation to local joints
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    # Add root XZ to joints
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # Concatenate root and joints
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


def recover_from_ric_stateful(data, joints_num, state=None):
    r_rot_quat, r_pos, next_state = recover_root_rot_pos_stateful(data, state=state)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions, next_state


def get_target_location(*args, **kwargs):
    raise NotImplementedError("Target conditioning is not supported in minimal inference.")
