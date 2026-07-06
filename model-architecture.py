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