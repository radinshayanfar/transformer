import torch
from transformer import AttentionHead, MultiHeadAttention, EncoderBlock, DecoderBlock, EncoderStack, DecoderStack

if __name__ == "__main__":
    d_model = 16
    d_k = 4
    d_ff = 2048
    head = AttentionHead(d_model, d_k)
    att = head(torch.randn(2, d_model))
    print(att)

    mult_head = MultiHeadAttention(8, d_model, d_k)
    print(mult_head(torch.randn(2, d_model)).shape)

    mult_head = MultiHeadAttention(1, d_model, d_k, masked=True)
    print(mult_head(torch.randn(2, d_model)).shape)

    enc_block = EncoderBlock(8, d_model, d_k, d_ff)
    block = enc_block(torch.randn(2, d_model))
    print(block.shape)

    dec_head = MultiHeadAttention(8, d_model, d_k)
    att = dec_head(torch.randn(2, d_model), torch.randn(5, d_model))
    print(att.shape)
    
    dec_block = DecoderBlock(8, d_model, d_k, d_ff)
    block = dec_block(torch.randn(2, d_model), torch.randn(5, d_model))
    print(block.shape)

    dec_block = DecoderBlock(8, d_model, d_k, d_ff, decoder_only=True)
    block = enc_block(torch.randn(2, d_model))
    print(block.shape)

    enc_stack = EncoderStack(6, 8, d_model, d_k, d_ff)
    stack = enc_stack(torch.randn(2, d_model))
    print(stack.shape)

    dec_stack = DecoderStack(6, 8, d_model, d_k, d_ff, 5000)
    stack = dec_stack(torch.randn(2, d_model), torch.randn(5, d_model))
    print(stack.shape)

    dec_stack = DecoderStack(6, 8, d_model, d_k, d_ff, 2, True)
    stack = dec_stack(torch.randn(2, d_model))
    print(stack)
