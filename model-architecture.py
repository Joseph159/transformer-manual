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


def feed_forward(input_dim: int, intermediate_dim: int = 2048) -> nn.Sequential:

    return nn.Sequential(
        nn.Linear(input_dim, intermediate_dim),
        nn.ReLU(),
        nn.Linear(intermediate_dim, input_dim)
    )

class Residual(nn.Module):
    """
    残差连接模块
    """
    def __init__(self, sublayer: nn.Module, dimension: int, dropout_rate: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        # 层归一化
        self.norm = nn.LayerNorm(dimension)
        # 随机丢弃一部分特征，防止过拟合
        self.dropout = nn.Dropout(dropout_rate)
        

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.norm(x + self.dropout(self.sublayer(x, *args, **kwargs)))


def position_encoding(seq_len: int, d_model: int) -> Tensor:
    """
    生成位置编码
    """
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    
    angle_rads = get_angles(np.arange(seq_len)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # 偶数索引使用sin
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # 奇数索引使用cos

    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    """
    def __init__(self, model_dim: int, heads_count: int, feed_forward_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        query_dim = key_dim = max(model_dim // heads_count, 1)  # Ensure at least dimension of 1 for query and key
        self.multi_head_attention = Residual(
            MultiHeadAttention(model_dim, heads_count, query_dim, key_dim),
            model_dim,
            dropout_rate=dropout_rate
        )

        self.feedforward_network = Residual(
            feed_forward(model_dim, feed_forward_dim),
            model_dim, 
            dropout_rate=dropout_rate
        )


    def forward(self, src: Tensor, mask: Tensor = None) -> Tensor:
        src = self.multi_head_attention(src, src, src, mask)
        return self.feedforward_network(src)


class TransformerEncoder(nn.Module):
    """
    Transformer编码器
    """
    def __init__(self, layer_count: int = 6, model_dim: int = 512, heads_count: int = 6, feed_forward_dim: int = 2048, dropout_rate: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, heads_count, feed_forward_dim, dropout_rate)
            for _ in range(layer_count)
        ])

    def forward(self, src: Tensor, mask: Tensor = None) -> Tensor:
        seq_legth, dimension = src.size(1), src.size(2)
        src += position_encoding(seq_legth, dimension)  # Add positional encoding
        for layer in self.layers:
            src = layer(src, mask)
        return src
        