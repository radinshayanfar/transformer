import math
import torch
import torch.nn as nn

class AttentionHead(nn.Module):
    def __init__(self, d_model, d_k, masked=False):  # we assume d_k == d_v and only use d_k here
        super().__init__()
        self.scale = math.sqrt(d_k)
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_k)

        self.masked = masked

    def forward(self, x, enc_dec_layer_input=None):
        assert self.masked == False or enc_dec_layer_input is None, "Either masked could be True or enc_dec_layer_input can have value, but not both"
        Q = self.W_Q(x)
        if enc_dec_layer_input is None:  # this is not a encoder decoder attention layer, so we have to compute K and V based on x
            K = self.W_K(x)
            V = self.W_V(x)
        else:
            K = self.W_K(enc_dec_layer_input)
            V = self.W_V(enc_dec_layer_input)
        
        scaled = (Q @ K.T) / self.scale
        if self.masked:  # set next tokens to -inf for decoder blocks
            att_mask = torch.full_like(scaled, float("-inf")).triu(1)  # setting diagonal to 1 to exclude the main diagonal
            scaled += att_mask
        # softmaxing the scores accros each token, so that each row sums to 1
        attention = torch.softmax(scaled, dim=1) @ V
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, d_v, masked=False):
        super().__init__()
        self.heads = [AttentionHead(d_model, d_v, masked=masked) for _ in range(h)]
        self.W_O = nn.Linear(h*d_v, d_model)

    def forward(self, x, enc_dec_layer_input=None):
        zs = [head(x, enc_dec_layer_input) for head in self.heads]
        z = torch.cat(zs, dim=1)
        z = self.W_O(z)

        return z


class EncoderBlock(nn.Module):
    def __init__(self, h, d_model, d_k, d_ff):
        super().__init__()

        self.mult_head_att = MultiHeadAttention(h, d_model, d_k)

        # using conv1d with filter size 1 to mimic separate FFN application
        self.ffn_l1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.ffn_l2 = nn.Conv1d(d_ff, d_model, kernel_size=1)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        z = self.mult_head_att(x)
        x = self.layer_norm1(z + x)
        # TODO: apply dropout to x
        
        # add a dummy dim for conv layers
        y = torch.relu(self.ffn_l1(x.unsqueeze(dim=-1)))
        y = self.ffn_l2(y)
        y = y.squeeze()
        # TODO: apply dropout to y

        x = self.layer_norm2(y + x)

        return x

class DecoderBlock(nn.Module):
    def __init__(self, h, d_model, d_k, d_ff, decoder_only=False):
        super().__init__()

        self.masked_att = MultiHeadAttention(h, d_model, d_k, masked=True)

        # using conv1d with filter size 1 to mimic separate FFN application
        self.ffn_l1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.ffn_l2 = nn.Conv1d(d_ff, d_model, kernel_size=1)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        self.decoder_only = decoder_only
        if not self.decoder_only:
            self.encdec_att = MultiHeadAttention(h, d_model, d_k)
            self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, enc_dec_layer_input=None):
        assert self.decoder_only or enc_dec_layer_input is not None, "enc_dec_layer_input should be provided in non-decoder only blocks"
        z = self.masked_att(x)
        x = self.layer_norm1(z + x)
        # TODO: apply dropout to x

        if not self.decoder_only:
            z = self.encdec_att(x, enc_dec_layer_input)
            x = self.layer_norm2(z + x)
            # TODO: apply dropout to x
        
        # add a dummy dim for conv layers
        y = torch.relu(self.ffn_l1(x.unsqueeze(dim=-1)))
        y = self.ffn_l2(y)
        y = y.squeeze()
        # TODO: apply dropout to y

        x = self.layer_norm3(y + x)

        return x

class EncoderStack:
    pass

class DecoderStack:
    pass
