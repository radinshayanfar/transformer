import torch
import math

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
    
    return table
