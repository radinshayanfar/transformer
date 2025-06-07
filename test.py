import torch
from transformer import AttentionHead, MultiHeadAttention, EncoderBlock

if __name__ == "__main__":
    d_model = 16
    d_k = 4
    d_ff = 2048
    head = AttentionHead(d_model, d_k)
    att = head(torch.randn(2, d_model))
    print(att)

    mult_head = MultiHeadAttention(1, d_model, d_k)
    print(mult_head(torch.randn(2, d_model)).shape)

    mult_head = MultiHeadAttention(8, d_model, d_k)
    print(mult_head(torch.randn(2, d_model)).shape)

    enc_block = EncoderBlock(8, d_model, d_k, d_ff)
    block = enc_block(torch.randn(2, d_model))
    print(block.shape)

    dec_head = MultiHeadAttention(8, d_model, d_k)
    att = dec_head(torch.randn(2, d_model), False, torch.randn(5, d_model))
    print(att.shape)
