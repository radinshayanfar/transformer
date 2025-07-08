import os
import torch
import math

def save_args(args, filepath):
    with open(filepath, 'w') as fp:
        fp.write(str(args))

def set_seed(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # random.seed(seed)
    # np.random.seed(seed)

def load_checkpoint(filepath, model, optim, device):
    if not os.path.exists(filepath):
        print("No checkpoint found at", filepath)
        return 0
    print("Loading checkpoint from", filepath)
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optim_state_dict"])
    batch = checkpoint["batch"]
    return batch

def save_checkpoint(filepath, model, optim, batch):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optim.state_dict(),
        "batch": batch,
    }, filepath)
    print("Checkpoint saved to", filepath, "at batch", batch)

def call_on_device(func, device, *args, **kwargs):
    args = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, (torch.nn.Module, torch.Tensor)):
            args[i] = arg.to(device)
    for key, val in kwargs.items():
        if isinstance(val, (torch.nn.Module, torch.Tensor)):
            kwargs[key] = val.to(device)
    return func(*args, **kwargs)


def distribute_call(funcs, *args, **kwargs):
    if not torch.cuda.is_available():
        return [func(*args, **kwargs) for func in funcs]
    cuda_count = torch.cuda.device_count()
    returns = []
    for i, func in enumerate(funcs):
        device = func.device if hasattr(func, "device") else torch.device(f"cuda:{i % cuda_count}")
        returns.append(call_on_device(func, device, *args, **kwargs))
    return returns


def distribute_objects(objects):
    if not torch.cuda.is_available():
        return objects
    device_count = torch.cuda.device_count()
    for i, obj in enumerate(objects):
        obj.device = torch.device(f"cuda:{i % device_count}")
        objects[i] = obj.to(obj.device)
    return objects


def get_positional_encoding_table(max_context_len, d_model):
    table = torch.zeros((d_model, max_context_len))

    for dim in range(d_model):
        sine_func = math.sin if dim % 2 == 0 else math.cos
        denom = math.pow(10_000, dim / d_model)
        for pos in range(max_context_len):
            table[dim, pos] = sine_func(pos/denom)
    
    return table.T


def pad_and_mask(encodings, tokenizer, pad_token):
    ids = [torch.tensor(e.ids, dtype=torch.long) for e in encodings]
    max_len = max(len(x) for x in ids)
    pad_id = tokenizer.token_to_id(pad_token)
    input_ids = torch.stack([
        torch.cat([x, torch.full((max_len - len(x),), pad_id, dtype=torch.long)])
        for x in ids
    ])
    attention_mask = torch.stack([
        torch.cat([torch.ones(len(x), dtype=torch.long),
                torch.zeros(max_len - len(x), dtype=torch.long)])
        for x in ids
    ])
    return input_ids, attention_mask


def tensor_to_sequence(token_ids, pad_mask):
    sequences = []
    for ids, mask in zip(token_ids, pad_mask):
        trimmed = ids[mask.bool()].tolist()
        sequences.append(trimmed)
    return sequences

def ids_to_tokens(sequence, tokenizer, remove_eos="[EOS]"):
    eos_token = tokenizer.token_to_id(remove_eos) if remove_eos is not None else -1
    return [tokenizer.id_to_token(i) for i in sequence if i != eos_token]
