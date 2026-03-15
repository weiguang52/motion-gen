"""
Brief:
Collate motion batches for text-conditioned HumanML-style inference.
"""
import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b["inp"] for b in notnone_batches]
    if "lengths" in notnone_batches[0]:
        lenbatch = [b["lengths"] for b in notnone_batches]
    else:
        lenbatch = [len(b["inp"][0][0]) for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {"y": {"mask": maskbatchTensor, "lengths": lenbatchTensor}}

    if "text" in notnone_batches[0]:
        textbatch = [b["text"] for b in notnone_batches]
        cond["y"].update({"text": textbatch})

    if "tokens" in notnone_batches[0]:
        textbatch = [b["tokens"] for b in notnone_batches]
        cond["y"].update({"tokens": textbatch})

    if "prefix" in notnone_batches[0]:
        prefixbatch = [b["prefix"] for b in notnone_batches]
        cond["y"].update({"prefix": collate_tensors(prefixbatch)})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch, target_batch_size):
    # HumanML/KIT 训练用 collate：把原始样本重复/截断到期望 batch 大小，并构造成统一张量
    repeat_factor = -(-target_batch_size // len(batch))  # Ceiling division
    repeated_batch = batch * repeat_factor 
    full_batch = repeated_batch[:target_batch_size]  # Truncate to the target batch size
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
        'key': b[7] if len(b) > 7 else None,
    } for b in full_batch]
    return collate(adapted_batch)
