import math
import torch
import torch.nn as nn

class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_k: int):  # we assume d_k == d_v and only use d_k here
        super().__init__()
        self.scale = math.sqrt(d_k)
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_k)

        self.att_mask = torch.full((d_k, d_k), float("-inf")).triu(1)  # setting diagonal to 1 to exclude the main diagonal

    def forward(self, x, masked=False, enc_dec_key=None, enc_dec_value=None):
        Q = self.W_Q(x)
        if enc_dec_key is None:  # this is not a encoder decoder attention layer, so we have to compute K and V
            K = self.W_K(x)
        else:
            K = enc_dec_key
        if enc_dec_value is None:  # this is not a encoder decoder attention layer, so we have to compute K and V
            V = self.W_V(x)
        else:
            V = enc_dec_value
        
        scaled = (Q @ K.T) / self.scale
        if masked:  # set next tokens to -inf for decoder blocks
            scaled += self.att_mask
        # softmaxing the scores accros each token, so that each row sums to 1
        attention = torch.softmax(scaled, dim=1) @ V
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, d_v):
        super().__init__()
        self.heads = [AttentionHead(d_model, d_v) for _ in range(h)]
        self.W_O = nn.Linear(h*d_v, d_model)

    def forward(self, x, masked=False, enc_dec_key=None, enc_dec_value=None):
        zs = [head(x, masked, enc_dec_key, enc_dec_value) for head in self.heads]
        z = torch.cat(zs, dim=1)
        z = self.W_O(z)

        return z


class EncoderBlock(nn.Module):
    pass

class DecoderBlock:
    pass

class EncoderStack:
    pass

class DecoderStack:
    pass
