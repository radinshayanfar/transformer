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
        # softmaxing the scores accros each token, so that each row sums to 1
        softmaxed = torch.softmax(scaled, dim=-1)
        if self.masked:  # causal attention
            att_mask = torch.ones_like(softmaxed).tril()
            softmaxed = softmaxed * att_mask.detach()
        if padding_mask is not None:
            padding_mask1 = padding_mask.unsqueeze(1)
            padding_mask2 = padding_mask_enc_dec.unsqueeze(1) if enc_dec_layer_input is not None else padding_mask1
            padding_mask = padding_mask1.transpose(-2, -1).to(torch.float16) @ padding_mask2.to(torch.float16)  # outer product to convert to 2D - float16 for cuda support
            softmaxed = softmaxed * padding_mask.detach()
        attention = softmaxed @ V
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, d_v, masked=False):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(d_model, d_v, masked=masked) for _ in range(h)])
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
        self.blocks = nn.ModuleList([EncoderBlock(h, d_model, d_k, d_ff) for _ in range(n_blocks)])
    
    def forward(self, x, padding_mask=None):
        for block in self.blocks:
            x = block(x, padding_mask)
        return x


# we use a separate class for decoder stack to add more flexibility
class DecoderStack(nn.Module):
    def __init__(self, n_blocks, h, d_model, d_k, d_ff, decoder_only=False):
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock(h, d_model, d_k, d_ff, decoder_only) for _ in range(n_blocks)])

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
    

    def forward(self, encoder_x=None, decoder_x=None, enc_pad_mask=None, dec_pad_mask=None, logits=True):
        assert encoder_x is not None or decoder_x is not None, "Either encoder or decoder should have input"
        if encoder_x is not None:
            encoder_x = self.embed_inputs(encoder_x)
        if decoder_x is not None:
            decoder_x = self.embed_inputs(decoder_x)

        encoder_output = None
        if self.arch != "decoder":
            encoder_output = self.encoder_stack(encoder_x, padding_mask=enc_pad_mask)
        if self.arch != "encoder":
            decoder_output = self.decoder_stack(decoder_x, encoder_output, padding_mask_x=dec_pad_mask, padding_mask_enc_dec=enc_pad_mask)
        
        output = encoder_output if self.arch == "encoder" else decoder_output
        output = self.linear(output)
        if not logits:
            output = torch.softmax(output, dim=-1)
        return output

    def generate(self, eos_token_id, pad_token_id, encoder_x=None, decoder_x=None, enc_pad_mask=None, dec_pad_mask=None, max_length=256):
        assert self.arch != "encoder", "Auto regressive generation doesn't work with encoder-only models"

        def mask_if_present(x, mask):
            if x is None:
                return x
            return x[mask]

        while True:
            last_token_index = dec_pad_mask.sum(dim=1) - 1
            if (last_token_index >= max_length).any():
                break
            continue_mask = decoder_x[torch.arange(decoder_x.shape[0]), last_token_index] != eos_token_id
            if not continue_mask.any():
                break

            if (last_token_index[continue_mask] == decoder_x.shape[1] - 1).any():  # tensor needs to grow - we double in size
                decoder_x = torch.cat((decoder_x, torch.full_like(decoder_x, pad_token_id)), dim=1)
                dec_pad_mask = torch.cat((dec_pad_mask, torch.zeros_like(dec_pad_mask)), dim=1)
            
            decoder_probs = self(
                mask_if_present(encoder_x, continue_mask),
                mask_if_present(decoder_x, continue_mask),
                mask_if_present(enc_pad_mask, continue_mask),
                mask_if_present(dec_pad_mask, continue_mask),
                logits=False,
            )

            gen_tokens = decoder_probs[torch.arange(decoder_probs.shape[0]), last_token_index[continue_mask]].argmax(dim=-1)  # using greedy decoding
            decoder_x[continue_mask, last_token_index[continue_mask]+1] = gen_tokens
            dec_pad_mask[continue_mask, last_token_index[continue_mask]+1] = 1
        
        return decoder_x, dec_pad_mask
