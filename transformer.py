import math
import torch
import torch.nn as nn
from typing import Optional, Literal

from utils import get_positional_encoding_table


class AttentionHead(nn.Module):
    def __init__(self, d_model, d_k, masked=False):  # we assume d_k == d_v and only use d_k here
        super().__init__()
        self.scale = math.sqrt(d_k)
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_k)

        self.masked = masked

    def forward(self, x, enc_dec_layer_input=None, padding_mask=None, padding_mask_enc_dec=None):
        assert self.masked == False or enc_dec_layer_input is None, "Either masked could be True or enc_dec_layer_input can have value, but not both"
        Q = self.W_Q(x)
        if enc_dec_layer_input is None:  # this is not a encoder decoder attention layer, so we have to compute K and V based on x
            K = self.W_K(x)
            V = self.W_V(x)
        else:
            K = self.W_K(enc_dec_layer_input)
            V = self.W_V(enc_dec_layer_input)
        
        # print("Q", Q.shape, "K", K.shape, "K.T", K.transpose(-2, -1).shape, "V", V.shape)
        scaled = (Q @ K.transpose(-2, -1)) / self.scale
        if self.masked:  # set next tokens to -inf for decoder blocks
            # att_mask = torch.full(scaled.shape[-2:], float("-inf"), device=scaled.device).triu(1)  # setting diagonal to 1 to exclude the main diagonal
            att_mask = torch.full_like(scaled, float("-inf")).triu(1)  # setting diagonal to 1 to exclude the main diagonal
            scaled += att_mask
        if padding_mask is not None:
            padding_mask1 = padding_mask.unsqueeze(1)
            padding_mask2 = padding_mask_enc_dec.unsqueeze(1) if enc_dec_layer_input is not None else padding_mask1
            padding_mask = padding_mask1.transpose(-2, -1) @ padding_mask2  # outer product to convert to 2D
            padding_mask = padding_mask.float()
            padding_mask[padding_mask == 0] = float("-inf")
            padding_mask[padding_mask == 1] = 0
            scaled += padding_mask
        # softmaxing the scores accros each token, so that each row sums to 1
        attention = torch.softmax(scaled, dim=-1) @ V
        attention = attention.nan_to_num()  # to prevent nan to propagate to other positions in multihead linear layer
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, d_v, masked=False):
        super().__init__()
        self.heads = [AttentionHead(d_model, d_v, masked=masked) for _ in range(h)]
        self.W_O = nn.Linear(h*d_v, d_model)

    def forward(self, x, enc_dec_layer_input=None, padding_mask=None, padding_mask_enc_dec=None):  # if enc_dec attention, padding_mask is for encoder input
        zs = [head(x, enc_dec_layer_input, padding_mask, padding_mask_enc_dec) for head in self.heads]
        z = torch.cat(zs, dim=-1)
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
    
    def forward(self, x, padding_mask=None):
        z = self.mult_head_att(x, padding_mask=padding_mask)
        x = self.layer_norm1(z + x)
        # TODO: apply dropout to x
        
        # add a dummy dim for conv layers
        B, L = x.shape[:2]
        y = x.reshape(-1, *x.shape[2:])
        y = torch.relu(self.ffn_l1(y.unsqueeze(dim=-1)))
        y = self.ffn_l2(y)
        y = y.squeeze()
        y = y.reshape(B, L, *y.shape[1:])
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
    
    def forward(self, x, enc_dec_layer_input=None, padding_mask_x=None, padding_mask_enc_dec=None):
        assert self.decoder_only or enc_dec_layer_input is not None, "enc_dec_layer_input should be provided in non-decoder only blocks"
        z = self.masked_att(x, padding_mask=padding_mask_x)
        x = self.layer_norm1(z + x)
        # TODO: apply dropout to x

        if not self.decoder_only:
            z = self.encdec_att(x, enc_dec_layer_input, padding_mask=padding_mask_x, padding_mask_enc_dec=padding_mask_enc_dec)
            x = self.layer_norm2(z + x)
            # TODO: apply dropout to x
        
        # add a dummy dim for conv layers
        B, L = x.shape[:2]
        y = x.reshape(-1, *x.shape[2:])
        y = torch.relu(self.ffn_l1(y.unsqueeze(dim=-1)))
        y = self.ffn_l2(y)
        y = y.squeeze()
        y = y.reshape(B, L, *y.shape[1:])
        # TODO: apply dropout to y

        x = self.layer_norm3(y + x)

        return x


class EncoderStack(nn.Module):
    def __init__(self, n_blocks, h, d_model, d_k, d_ff):
        super().__init__()
        self.blocks = [EncoderBlock(h, d_model, d_k, d_ff) for _ in range(n_blocks)]
    
    def forward(self, x, padding_mask=None):
        for block in self.blocks:
            x = block(x, padding_mask)
        return x


# we use a separate class for decoder stack to add more flexibility
class DecoderStack(nn.Module):
    def __init__(self, n_blocks, h, d_model, d_k, d_ff, decoder_only=False):
        super().__init__()
        self.blocks = [DecoderBlock(h, d_model, d_k, d_ff, decoder_only) for _ in range(n_blocks)]

    def forward(self, x, enc_dec_layer_input=None, padding_mask_x=None, padding_mask_enc_dec=None):
        for block in self.blocks:
            x = block(x, enc_dec_layer_input, padding_mask_x, padding_mask_enc_dec)

        return x


class Transformer(nn.Module):
    def __init__(self, n_blocks, h, d_model, d_k, d_ff, n_vocab, max_context_len, arch: Optional[Literal["decoder", "encoder", "both"]] = None):
        super().__init__()
        self.arch = arch

        if self.arch != "decoder":
            self.encoder_stack = EncoderStack(n_blocks, h, d_model, d_k, d_ff)
        if self.arch != "encoder":
            self.decoder_stack = DecoderStack(n_blocks, h, d_model, d_k, d_ff, True if self.arch == "decoder" else False)

        self.pos_emb_table = get_positional_encoding_table(max_context_len, d_model)

        self.emb_table = nn.Parameter(torch.randn((d_model, n_vocab)))

        self.linear = nn.Linear(d_model, n_vocab)

    
    def embed_inputs(self, input_ids):
        B, L = input_ids.shape
        input_embs = self.emb_table[:, input_ids.reshape(-1)].T.reshape(B, L, -1)
        return input_embs
    

    def forward(self, x, y=None, padding_mask_x=None, padding_mask_y=None):
        x = self.embed_inputs(x)
        if y is not None:
            y = self.embed_inputs(y)

        encoder_output = None
        if self.arch != "decoder":
            encoder_output = self.encoder_stack(x, padding_mask=padding_mask_x)
        if self.arch != "encoder":
            dec_input = x if self.arch == "decoder" else y
            padding_mask_dec = padding_mask_x if self.arch == "decoder" else padding_mask_y
            padding_mask_enc = padding_mask_x if self.arch != "decoder" else None
            decoder_output = self.decoder_stack(dec_input, encoder_output, padding_mask_x=padding_mask_dec, padding_mask_enc_dec=padding_mask_enc)
        
        if self.arch == "encoder":
            encoder_output = self.linear(encoder_output)
            encoder_probs = torch.softmax(encoder_output, dim=-1)
            return encoder_probs
        decoder_output = self.linear(decoder_output)
        decoder_probs = torch.softmax(decoder_output, dim=-1)
        # if self.arch == "decoder":
        #     return decoder_output
        return decoder_probs
            

    def generate(self):
        pass
