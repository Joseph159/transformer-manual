import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

def scaled_dot_product(q : Tensor, k : Tensor, v : Tensor, mask : Tensor = None) -> Tensor:
    """
    缩放点积注意力的实现
    
    Args:
        q (Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, depth)
        k (Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, depth)
        v (Tensor): Value tensor of shape (batch_size, num_heads, seq_len_v, depth_v)
        mask (Tensor, optional): Mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k)
    
    Returns:
        Tensor: Output tensor after applying attention mechanism.
    """
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)

    # Scale matmul_qk
    dk = k.size()[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    # Add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, float('-inf'))  # Add large negative value to masked positions

    # Softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)  # (..., seq_len_q, seq_len_k)

    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output


class AttentionHead(nn.Module):
    def __init__(self, dim_input: int, dim_q: int, dim_k: int):
        super().__init__()

        self.linear_query = nn.Linear(dim_input, dim_q)
        self.linear_key = nn.Linear(dim_input, dim_k)
        self.linear_value = nn.Linear(dim_input, dim_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:# (batch_size, 1, seq_len, d_v)

        return scaled_dot_product(
            self.linear_query(query),
            self.linear_key(key),
            self.linear_value(value),
            mask
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_input: int, num_heads: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(dim_input, dim_q, dim_k) for _ in range(num_heads)
        ])
        self.linear_out = nn.Linear(num_heads * dim_k, dim_input)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        head_outputs = [head(query, key, value, mask) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        return self.linear_out(concatenated)