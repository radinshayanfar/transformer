import torch
import math

class Tokenizer:
    pass

def get_positional_encoding_table(max_context_len, d_model):
    table = torch.zeros((d_model, max_context_len))

    for dim in range(d_model):
        sine_func = math.sin if dim % 2 == 0 else math.cos
        denom = math.pow(10_000, dim / d_model)
        for pos in range(max_context_len):
            table[dim, pos] = sine_func(pos/denom)
    
    return table
