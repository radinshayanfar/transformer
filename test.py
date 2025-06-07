import torch
from transformer import AttentionHead, MultiHeadAttention

if __name__ == "__main__":
    d_model = 16
    d_k = 4
    head = AttentionHead(d_model, d_k)
    att = head(torch.randn(2, d_model))
    print(att)

    mult_head = MultiHeadAttention(1, d_model, d_k)
    print(mult_head(torch.randn(2, d_model)).shape)

    mult_head = MultiHeadAttention(8, d_model, d_k)
    print(mult_head(torch.randn(2, d_model)).shape)
