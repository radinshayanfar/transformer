import torch
from transformer import AttentionHead

if __name__ == "__main__":
    d_model = 16
    d_k = 4
    head = AttentionHead(d_model, d_k)
    att = head(torch.randn(2, d_model))
    print(att)
