# Transformer 解码器层与解码器阅读文档

## 1. 代码原文

```python
class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器层
    """
    def __init__(self, model_dim: int, heads_count: int, feed_forward_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        query_dim = key_dim = max(model_dim // heads_count, 1)  # Ensure at least dimension of 1 for query and key
        self.self_attention = Residual(
            MultiHeadAttention(model_dim, heads_count, query_dim, key_dim),
            model_dim,
            dropout_rate=dropout_rate
        )
        self.cross_attention = Residual(
            MultiHeadAttention(model_dim, heads_count, query_dim, key_dim),
            model_dim,
            dropout_rate=dropout_rate
        )
        self.feedforward_network = Residual(
            feed_forward(model_dim, feed_forward_dim),
            model_dim,
            dropout_rate=dropout_rate
        )

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt = self.self_attention(tgt, tgt, tgt)
        tgt = self.cross_attention(tgt, memory, memory)
        return self.feedforward_network(tgt)

class TransformerDecoder(nn.Module):
    """
    Transformer解码器
    """
    def __init__(self, layer_count: int = 6, model_dim: int = 512, heads_count: int = 6, feed_forward_dim: int = 2048, dropout_rate: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(model_dim, heads_count, feed_forward_dim, dropout_rate)
            for _ in range(layer_count)
        ])
        self.final_linear = nn.Linear(model_dim, model_dim)  # Final linear layer to project to output dimension

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        seq_legth, dimension = tgt.size(1), tgt.size(2)
        tgt += position_encoding(seq_legth, dimension)  # Add positional encoding
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return torch.softmax(self.final_linear(tgt), dim=-1)  # Apply softmax to the final output
```

---

# 2. 先用一句话理解 Decoder

Transformer Decoder 的作用是：

```text
根据已经生成的目标 token，以及 Encoder 读懂的源序列信息，继续生成下一个 token。
```

例如机器翻译：

```text
源句子：
I love China

目标句子：
我 爱 中国
```

Encoder 负责读懂源句子：

```text
I love China
  ↓
Encoder
  ↓
memory
```

Decoder 负责生成目标句子：

```text
<BOS>
  ↓
我
  ↓
爱
  ↓
中国
  ↓
<EOS>
```

其中：

```text
<BOS>：句子开始标记
<EOS>：句子结束标记
```

---

# 3. Decoder 的两个输入是什么？

在代码中：

```python
def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
```

Decoder 有两个输入：

| 参数       | 含义                        |
| -------- | ------------------------- |
| `tgt`    | 目标序列，也就是已经生成过的目标 token 表示 |
| `memory` | Encoder 的输出，也就是源句子被编码后的表示 |

---

## 3.1 tgt 是什么？

`tgt` 是 Decoder 当前已经拥有的目标序列表示。

训练时，`tgt` 一般是右移后的目标序列。

例如目标句子是：

```text
我 爱 中国
```

训练时 Decoder 输入是：

```text
<BOS> 我 爱
```

Decoder 要预测的是：

```text
我 爱 中国
```

所以：

```text
Decoder 输入：<BOS> 我 爱
Decoder 目标：我    爱 中国
```

在进入 Decoder 之前，token 会经过 embedding，所以 `tgt` 形状一般是：

```text
(batch_size, tgt_seq_len, model_dim)
```

例如：

```text
tgt: (2, 4, 512)
```

含义：

```text
2 个样本
目标序列长度是 4
每个 token 是 512 维向量
```

---

## 3.2 memory 是什么？

`memory` 是 Encoder 的输出。

比如源句子是：

```text
I love China
```

Encoder 输出：

```text
memory = Encoder(src)
```

它的形状一般是：

```text
(batch_size, src_seq_len, model_dim)
```

例如：

```text
memory: (2, 3, 512)
```

含义：

```text
2 个样本
源序列长度是 3
每个源 token 被编码成 512 维向量
```

注意：

```text
memory 不是原始词向量，而是 Encoder 处理后的上下文语义表示。
```

---

# 第一部分：TransformerDecoderLayer 解码器层

---

## 4. TransformerDecoderLayer 是什么？

```python
class TransformerDecoderLayer(nn.Module):
```

这定义了 Transformer 的一个解码器层。

注意：

```text
它不是完整 Decoder。
它只是 Decoder 里面的一层。
```

完整 Decoder 是由多个 `TransformerDecoderLayer` 堆叠起来的。

例如：

```text
TransformerDecoder
  ├── TransformerDecoderLayer 1
  ├── TransformerDecoderLayer 2
  ├── TransformerDecoderLayer 3
  ├── ...
  └── TransformerDecoderLayer N
```

---

## 5. 一个 DecoderLayer 里面有什么？

一个 DecoderLayer 里面有三大块：

```text
1. Self-Attention
2. Cross-Attention
3. Feed Forward
```

对应代码：

```python
self.self_attention = Residual(...)
self.cross_attention = Residual(...)
self.feedforward_network = Residual(...)
```

每一块外面都包了：

```text
Residual + Dropout + LayerNorm
```

所以一个 DecoderLayer 的整体流程是：

```text
输入 tgt
  ↓
Masked Self-Attention
  ↓
Residual + Dropout + LayerNorm
  ↓
Cross-Attention，看 Encoder 输出 memory
  ↓
Residual + Dropout + LayerNorm
  ↓
Feed Forward
  ↓
Residual + Dropout + LayerNorm
  ↓
输出
```

---

## 6. 初始化函数

```python
def __init__(self, model_dim: int, heads_count: int, feed_forward_dim: int, dropout_rate: float = 0.1):
```

参数含义：

| 参数                 | 含义                      |
| ------------------ | ----------------------- |
| `model_dim`        | 模型主维度，也就是每个 token 的向量维度 |
| `heads_count`      | 多头注意力的头数                |
| `feed_forward_dim` | 前馈神经网络中间层维度             |
| `dropout_rate`     | Dropout 概率，默认 0.1       |

例如：

```python
model_dim = 512
heads_count = 8
feed_forward_dim = 2048
dropout_rate = 0.1
```

表示：

```text
每个 token 是 512 维
多头注意力有 8 个头
前馈网络中间层是 2048 维
Dropout 概率是 0.1
```

---

## 7. 调用父类初始化

```python
super().__init__()
```

这一行是 PyTorch 模块里常见的写法。

它的作用是初始化父类 `nn.Module`。

这样 PyTorch 才能正确管理这个模块里面的子模块和参数。

---

## 8. 计算每个注意力头的维度

```python
query_dim = key_dim = max(model_dim // heads_count, 1)
```

这一行是在计算每个注意力头内部的 Query 和 Key 维度。

假设：

```python
model_dim = 512
heads_count = 8
```

那么：

```text
512 // 8 = 64
```

所以：

```text
query_dim = 64
key_dim = 64
```

意思是：

```text
每个注意力头的 Q/K 维度是 64。
```

---

## 9. 为什么要除以 heads_count？

多头注意力会把总维度拆给多个头。

例如：

```text
model_dim = 512
heads_count = 8
```

那么每个头负责：

```text
512 / 8 = 64 维
```

8 个头拼接回来：

```text
8 × 64 = 512
```

刚好回到 `model_dim`。

---

## 10. `max(..., 1)` 的作用

```python
max(model_dim // heads_count, 1)
```

这是为了防止出现 0 维。

例如：

```python
model_dim = 4
heads_count = 8
```

那么：

```text
4 // 8 = 0
```

如果 Query/Key 是 0 维，注意力无法计算。

所以用：

```python
max(0, 1)
```

保证最少是 1 维。

---

# 11. 第一块：Self-Attention

```python
self.self_attention = Residual(
    MultiHeadAttention(model_dim, heads_count, query_dim, key_dim),
    model_dim,
    dropout_rate=dropout_rate
)
```

这创建了 Decoder 的第一块注意力：

```text
Self-Attention
```

它的作用是：

```text
让目标序列内部的 token 互相看。
```

例如当前目标序列是：

```text
<BOS> 我 爱
```

Decoder 在预测下一个词时，需要看已经生成过的：

```text
<BOS>
我
爱
```

所以 self-attention 是：

```text
目标序列自己看自己。
```

---

## 11.1 Decoder Self-Attention 理论上应该是 Masked Self-Attention

在标准 Transformer 里，Decoder 的 self-attention 应该是：

```text
Masked Multi-Head Self-Attention
```

也就是需要 causal mask。

原因是：

```text
预测当前位置时，不能偷看未来位置。
```

例如训练时：

```text
Decoder 输入：<BOS> 我 爱
预测目标：我    爱 中国
```

当模型预测“我”时，只能看到 `<BOS>`，不能看到后面的“我”“爱”。

当模型预测“爱”时，只能看到 `<BOS> 我`，不能看到后面的“爱”。

所以需要 mask。

但是你的代码里：

```python
tgt = self.self_attention(tgt, tgt, tgt)
```

没有传 mask。

这意味着：

```text
如果用于标准自回归生成，这里还缺少 causal mask。
```

更标准的 forward 应该类似：

```python
tgt = self.self_attention(tgt, tgt, tgt, tgt_mask)
```

前提是你的 `Residual` 和 `MultiHeadAttention` 支持传 mask。

---

# 12. 第二块：Cross-Attention

```python
self.cross_attention = Residual(
    MultiHeadAttention(model_dim, heads_count, query_dim, key_dim),
    model_dim,
    dropout_rate=dropout_rate
)
```

这创建了 Decoder 的第二块注意力：

```text
Cross-Attention
```

也叫：

```text
Encoder-Decoder Attention
```

它的作用是：

```text
让 Decoder 去看 Encoder 输出的 memory。
```

---

## 12.1 Cross-Attention 中 Q/K/V 来自哪里？

在 DecoderLayer 的 forward 里：

```python
tgt = self.cross_attention(tgt, memory, memory)
```

这里传参是：

```text
Q = tgt
K = memory
V = memory
```

也就是：

```text
Query 来自 Decoder 当前状态
Key 来自 Encoder 输出
Value 来自 Encoder 输出
```

写成注意力公式就是：

```text
Attention(Q_decoder, K_encoder, V_encoder)
```

---

## 12.2 为什么 Q 来自 Decoder？

因为 Decoder 当前正在生成目标序列。

它需要问：

```text
我现在要生成下一个 token，我应该去源句子里找什么信息？
```

这个“问题”就是 Query。

所以：

```text
Q 来自 Decoder
```

---

## 12.3 为什么 K/V 来自 Encoder？

Encoder 已经读懂了源句子。

例如源句子：

```text
I love China
```

Encoder 输出：

```text
memory = [
  I 的上下文向量,
  love 的上下文向量,
  China 的上下文向量
]
```

Decoder 需要从这些信息里找答案。

所以：

```text
K 来自 Encoder：用来匹配 Decoder 的查询
V 来自 Encoder：提供真正被取走的信息
```

---

## 12.4 Cross-Attention 的直观例子

假设源句子：

```text
I love China
```

Encoder 输出：

```text
memory =
[
  I 的向量,
  love 的向量,
  China 的向量
]
```

Decoder 当前已经生成：

```text
<BOS> 我 爱
```

现在要预测下一个 token。

Decoder 产生一个 Query，去和 Encoder 的 Key 匹配：

```text
Query 和 I 的 Key 匹配
Query 和 love 的 Key 匹配
Query 和 China 的 Key 匹配
```

得到注意力权重：

```text
I:      0.05
love:   0.10
China:  0.85
```

说明当前最应该关注：

```text
China
```

然后用这些权重加权 Encoder 的 Value：

```text
0.05 * V(I)
+ 0.10 * V(love)
+ 0.85 * V(China)
```

得到一个上下文向量。

这个向量会帮助 Decoder 生成：

```text
中国
```

---

# 13. 第三块：Feed Forward

```python
self.feedforward_network = Residual(
    feed_forward(model_dim, feed_forward_dim),
    model_dim,
    dropout_rate=dropout_rate
)
```

这创建了 Decoder 的前馈神经网络块。

如果：

```python
model_dim = 512
feed_forward_dim = 2048
```

那么前馈神经网络大概是：

```python
nn.Sequential(
    nn.Linear(512, 2048),
    nn.ReLU(),
    nn.Linear(2048, 512)
)
```

也就是：

```text
512 → 2048 → 512
```

它的作用是：

```text
对每个 token 自己的特征做进一步加工。
```

---

# 14. DecoderLayer 的 forward 函数

```python
def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
    tgt = self.self_attention(tgt, tgt, tgt)
    tgt = self.cross_attention(tgt, memory, memory)
    return self.feedforward_network(tgt)
```

这个 forward 定义了解码器层的数据流动过程。

---

## 14.1 第一步：Self-Attention

```python
tgt = self.self_attention(tgt, tgt, tgt)
```

这里传的是：

```text
Q = tgt
K = tgt
V = tgt
```

所以这是：

```text
目标序列内部的自注意力。
```

展开理解：

```text
tgt = LayerNorm(tgt + Dropout(SelfAttention(tgt, tgt, tgt)))
```

标准 Decoder 中，这一步应该带 mask：

```text
tgt = LayerNorm(tgt + Dropout(MaskedSelfAttention(tgt, tgt, tgt, tgt_mask)))
```

---

## 14.2 第二步：Cross-Attention

```python
tgt = self.cross_attention(tgt, memory, memory)
```

这里传的是：

```text
Q = tgt
K = memory
V = memory
```

所以这是：

```text
Decoder 看 Encoder 输出。
```

展开理解：

```text
tgt = LayerNorm(tgt + Dropout(CrossAttention(tgt, memory, memory)))
```

这一步的作用是：

```text
让 Decoder 在生成目标 token 时参考源句子信息。
```

---

## 14.3 第三步：Feed Forward

```python
return self.feedforward_network(tgt)
```

这一行把经过 self-attention 和 cross-attention 的结果送入前馈网络。

展开理解：

```text
output = LayerNorm(tgt + Dropout(FeedForward(tgt)))
```

---

# 15. DecoderLayer forward 完整展开

原代码：

```python
def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
    tgt = self.self_attention(tgt, tgt, tgt)
    tgt = self.cross_attention(tgt, memory, memory)
    return self.feedforward_network(tgt)
```

可以理解成：

```python
def forward(self, tgt, memory):
    # 1. 目标序列内部自注意力
    tgt = LayerNorm(tgt + Dropout(SelfAttention(tgt, tgt, tgt)))

    # 2. 目标序列去看 Encoder 输出
    tgt = LayerNorm(tgt + Dropout(CrossAttention(tgt, memory, memory)))

    # 3. 前馈网络加工
    tgt = LayerNorm(tgt + Dropout(FeedForward(tgt)))

    return tgt
```

标准自回归 Decoder 更应该是：

```python
def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
    tgt = LayerNorm(tgt + Dropout(MaskedSelfAttention(tgt, tgt, tgt, tgt_mask)))
    tgt = LayerNorm(tgt + Dropout(CrossAttention(tgt, memory, memory, memory_mask)))
    tgt = LayerNorm(tgt + Dropout(FeedForward(tgt)))
    return tgt
```

---

# 16. DecoderLayer 形状变化示例

假设：

```python
batch_size = 2
src_seq_len = 3
tgt_seq_len = 4
model_dim = 512
heads_count = 8
feed_forward_dim = 2048
```

输入：

```text
tgt:    (2, 4, 512)
memory: (2, 3, 512)
```

含义：

```text
tgt 是目标序列表示，长度 4
memory 是源序列编码结果，长度 3
```

---

## 16.1 Self-Attention 部分

输入：

```text
tgt: (2, 4, 512)
```

每个头维度：

```text
query_dim = key_dim = 512 // 8 = 64
```

每个头里：

```text
Q: (2, 4, 64)
K: (2, 4, 64)
V: (2, 4, 64)
```

注意力分数：

```text
QK^T: (2, 4, 4)
```

这里的 `(4, 4)` 表示：

```text
目标序列中 4 个 token 两两计算注意力。
```

输出：

```text
self_attention_output: (2, 4, 512)
```

残差相加和 LayerNorm 后：

```text
tgt: (2, 4, 512)
```

---

## 16.2 Cross-Attention 部分

输入：

```text
tgt:    (2, 4, 512)
memory: (2, 3, 512)
```

Cross-Attention 中：

```text
Q 来自 tgt
K 来自 memory
V 来自 memory
```

每个头：

```text
Q: (2, 4, 64)
K: (2, 3, 64)
V: (2, 3, 64)
```

注意力分数：

```text
QK^T: (2, 4, 3)
```

这里的 `(4, 3)` 表示：

```text
目标序列 4 个位置，分别去关注源序列 3 个位置。
```

也就是：

```text
每个目标 token 都会去看源句子的每个 token。
```

输出：

```text
cross_attention_output: (2, 4, 512)
```

残差相加和 LayerNorm 后：

```text
tgt: (2, 4, 512)
```

---

## 16.3 Feed Forward 部分

前馈网络：

```text
Linear(512, 2048)
ReLU
Linear(2048, 512)
```

形状变化：

```text
(2, 4, 512)
  ↓ Linear(512, 2048)
(2, 4, 2048)
  ↓ ReLU
(2, 4, 2048)
  ↓ Linear(2048, 512)
(2, 4, 512)
```

残差相加和 LayerNorm 后：

```text
output: (2, 4, 512)
```

---

# 第二部分：TransformerDecoder 完整解码器

---

## 17. TransformerDecoder 是什么？

```python
class TransformerDecoder(nn.Module):
```

这个类实现的是完整的 Transformer 解码器。

它由多个 `TransformerDecoderLayer` 组成。

如果：

```python
layer_count = 6
```

那么它里面有：

```text
6 个 TransformerDecoderLayer
```

整体结构是：

```text
tgt
  ↓
加位置编码
  ↓
DecoderLayer 1
  ↓
DecoderLayer 2
  ↓
DecoderLayer 3
  ↓
...
  ↓
DecoderLayer N
  ↓
final_linear
  ↓
softmax
  ↓
输出概率
```

---

## 18. Decoder 初始化函数

```python
def __init__(self, layer_count: int = 6, model_dim: int = 512, heads_count: int = 6, feed_forward_dim: int = 2048, dropout_rate: float = 0.1):
```

参数含义：

| 参数                 | 含义                |
| ------------------ | ----------------- |
| `layer_count`      | 解码器层数，默认 6        |
| `model_dim`        | 模型主维度，默认 512      |
| `heads_count`      | 注意力头数，默认 6        |
| `feed_forward_dim` | 前馈网络中间层维度，默认 2048 |
| `dropout_rate`     | Dropout 概率，默认 0.1 |

---

## 19. 创建多个 DecoderLayer

```python
self.layers = nn.ModuleList([
    TransformerDecoderLayer(model_dim, heads_count, feed_forward_dim, dropout_rate)
    for _ in range(layer_count)
])
```

这一段创建多个解码器层，并用 `nn.ModuleList` 保存。

如果：

```python
layer_count = 6
```

那么会创建：

```text
6 个 TransformerDecoderLayer
```

---

## 20. 为什么使用 nn.ModuleList？

```python
self.layers = nn.ModuleList([...])
```

`nn.ModuleList` 是 PyTorch 专门用来保存多个子模块的容器。

不要随便换成普通 Python list。

原因是：

```text
nn.ModuleList 可以让 PyTorch 正确注册里面每一层的参数。
```

这样优化器才能找到这些层的参数并更新。

---

## 21. final_linear 是什么？

```python
self.final_linear = nn.Linear(model_dim, model_dim)
```

这行创建了最后的线性层。

它的作用是把 Decoder 输出映射到最终输出空间。

但是这里需要注意：

```text
在真正的语言生成任务里，final_linear 通常应该映射到 vocab_size，而不是 model_dim。
```

也就是说，更常见写法是：

```python
self.final_linear = nn.Linear(model_dim, vocab_size)
```

因为最后要预测的是：

```text
词表中每个 token 的概率。
```

如果词表大小是：

```text
vocab_size = 30000
```

那么最终输出应该是：

```text
(batch_size, tgt_seq_len, 30000)
```

表示每个位置上 30000 个词的概率。

而你现在的代码：

```python
nn.Linear(model_dim, model_dim)
```

如果：

```text
model_dim = 512
```

最终只输出：

```text
512 维
```

这更像是投影回模型维度，而不是预测词表概率。

---

# 22. Decoder 的 forward 函数

```python
def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
    seq_legth, dimension = tgt.size(1), tgt.size(2)
    tgt += position_encoding(seq_legth, dimension)  # Add positional encoding
    for layer in self.layers:
        tgt = layer(tgt, memory)
    return torch.softmax(self.final_linear(tgt), dim=-1)  # Apply softmax to the final output
```

这个 forward 定义了完整 Decoder 的数据流。

---

## 23. 取得目标序列长度和维度

```python
seq_legth, dimension = tgt.size(1), tgt.size(2)
```

假设：

```text
tgt: (2, 4, 512)
```

那么：

```python
tgt.size(1)
```

得到：

```text
4
```

表示目标序列长度。

```python
tgt.size(2)
```

得到：

```text
512
```

表示模型维度。

所以：

```text
seq_legth = 4
dimension = 512
```

注意：变量名 `seq_legth` 拼写有误，更标准应该是：

```python
seq_length
```

---

## 24. 给 tgt 加位置编码

```python
tgt += position_encoding(seq_legth, dimension)
```

这一步给目标序列加位置信息。

因为 Decoder 也需要知道目标 token 的顺序。

例如：

```text
<BOS> 我 爱
```

Decoder 要知道：

```text
<BOS> 在第 0 个位置
我 在第 1 个位置
爱 在第 2 个位置
```

`tgt` 形状：

```text
(batch_size, tgt_seq_len, model_dim)
```

例如：

```text
(2, 4, 512)
```

位置编码形状：

```text
(1, tgt_seq_len, model_dim)
```

例如：

```text
(1, 4, 512)
```

相加：

```text
(2, 4, 512) + (1, 4, 512) = (2, 4, 512)
```

---

## 25. 逐层经过 DecoderLayer

```python
for layer in self.layers:
    tgt = layer(tgt, memory)
```

这一段表示：

```text
让 tgt 依次通过每一个 DecoderLayer。
```

如果有 6 层，展开就是：

```python
tgt = layer1(tgt, memory)
tgt = layer2(tgt, memory)
tgt = layer3(tgt, memory)
tgt = layer4(tgt, memory)
tgt = layer5(tgt, memory)
tgt = layer6(tgt, memory)
```

每一层都会做：

```text
Self-Attention
Cross-Attention
Feed Forward
```

---

## 26. final_linear + softmax

```python
return torch.softmax(self.final_linear(tgt), dim=-1)
```

这一步做最终输出。

先经过：

```python
self.final_linear(tgt)
```

把 Decoder 输出映射到输出空间。

然后：

```python
torch.softmax(..., dim=-1)
```

在最后一个维度上做 softmax，把 logits 变成概率分布。

---

## 27. softmax 的作用

softmax 会把一组分数变成概率。

例如某个位置输出分数：

```text
我: 5.0
你: 1.0
爱: 3.0
中国: 4.0
```

经过 softmax 后可能变成：

```text
我: 0.65
你: 0.01
爱: 0.09
中国: 0.25
```

模型会选择概率最高的 token 作为输出，或者按采样策略选择。

---

# 第三部分：Decoder 生成 token 的过程

---

## 28. 训练阶段 Decoder 输入是什么？

训练时，Decoder 输入是右移后的目标序列。

例如目标句子：

```text
我 爱 中国
```

Decoder 输入：

```text
<BOS> 我 爱
```

Decoder 预测目标：

```text
我 爱 中国
```

也就是：

| Decoder 输入  | 预测目标 |
| ----------- | ---- |
| `<BOS>`     | `我`  |
| `<BOS> 我`   | `爱`  |
| `<BOS> 我 爱` | `中国` |

---

## 29. 推理阶段 Decoder 输入是什么？

推理时没有标准答案，所以要一个 token 一个 token 生成。

第一步：

```text
输入：<BOS>
输出：我
```

第二步：

```text
输入：<BOS> 我
输出：爱
```

第三步：

```text
输入：<BOS> 我 爱
输出：中国
```

第四步：

```text
输入：<BOS> 我 爱 中国
输出：<EOS>
```

最终得到：

```text
我 爱 中国
```

---

## 30. 第一个词是如何生成的？

第一个词依赖两个东西：

```text
1. <BOS>
2. Encoder 输出 memory
```

流程是：

```text
<BOS>
  ↓
Embedding
  ↓
加位置编码
  ↓
Decoder Self-Attention
  ↓
Cross-Attention 参考 memory
  ↓
Feed Forward
  ↓
Linear
  ↓
Softmax
  ↓
预测第一个 token
```

比如源句子：

```text
I love China
```

Decoder 第一步：

```text
输入：<BOS>
参考：memory = Encoder(I love China)
输出概率：
我：0.80
你：0.05
他：0.03
爱：0.02
...
```

于是生成：

```text
我
```

---

# 第四部分：Decoder 和 Encoder 的区别

---

## 31. Encoder 做什么？

Encoder 负责读懂输入序列。

例如：

```text
I love China
```

Encoder 输出：

```text
memory
```

可以理解成：

```text
源句子的上下文语义表示。
```

Encoder 内部主要是：

```text
Self-Attention
Feed Forward
```

---

## 32. Decoder 做什么？

Decoder 负责生成目标序列。

它既要看已经生成的目标 token，也要看 Encoder 输出的源句子信息。

Decoder 内部是：

```text
Self-Attention
Cross-Attention
Feed Forward
```

---

## 33. EncoderLayer 和 DecoderLayer 对比

| 模块                    | EncoderLayer | DecoderLayer |
| --------------------- | ------------ | ------------ |
| Self-Attention        | 有            | 有            |
| Masked Self-Attention | 不需要          | 需要           |
| Cross-Attention       | 没有           | 有            |
| Feed Forward          | 有            | 有            |
| 输入                    | src          | tgt + memory |
| 作用                    | 理解源序列        | 生成目标序列       |

---

# 第五部分：Decoder 和 RNN/LSTM 的区别

---

## 34. Decoder 是否需要之前 token？

需要。

Transformer Decoder 生成时仍然是自回归的。

也就是说：

```text
生成下一个 token 时，需要前面已经生成的 token。
```

例如：

```text
<BOS> → 我
<BOS> 我 → 爱
<BOS> 我 爱 → 中国
```

---

## 35. 那和 RNN 有什么区别？

区别不是是否需要历史 token。

区别是：

```text
RNN 靠上一步 hidden state 传递历史信息。
Transformer 靠 Self-Attention 直接看所有历史 token。
```

RNN 像传话：

```text
第 1 个 token → 第 2 个 token → 第 3 个 token → 第 4 个 token
```

Transformer 像开会：

```text
当前 token 可以直接看前面所有 token。
```

---

## 36. 训练时 Transformer 可以并行

RNN 训练时通常要按时间步顺序算。

Transformer 训练时可以一次性把目标序列喂进去：

```text
Decoder 输入：<BOS> 我 爱
预测目标：我 爱 中国
```

然后通过 mask 防止看到未来。

所以训练时可以并行计算每个位置。

---

## 37. 推理时 Transformer 仍然一个一个生成

推理时没有答案，所以必须：

```text
先生成第一个 token
再生成第二个 token
再生成第三个 token
```

但是它可以使用 KV Cache 缓存历史 token 的 Key/Value，减少重复计算。

---

# 第六部分：代码中需要注意的问题

---

## 38. 目前代码缺少 mask

你的 DecoderLayer 代码是：

```python
def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
    tgt = self.self_attention(tgt, tgt, tgt)
    tgt = self.cross_attention(tgt, memory, memory)
    return self.feedforward_network(tgt)
```

标准 Decoder 自注意力应该带 causal mask。

否则训练时模型可能看到未来 token。

更标准的接口可以写成：

```python
def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor = None, memory_mask: Tensor = None) -> Tensor:
    tgt = self.self_attention(tgt, tgt, tgt, tgt_mask)
    tgt = self.cross_attention(tgt, memory, memory, memory_mask)
    return self.feedforward_network(tgt)
```

---

## 39. final_linear 应该映射到 vocab_size

现在代码：

```python
self.final_linear = nn.Linear(model_dim, model_dim)
```

如果用于语言生成，更标准应该是：

```python
self.final_linear = nn.Linear(model_dim, vocab_size)
```

因为最终要预测词表中每个 token 的概率。

---

## 40. heads_count 默认 6 不太适合 model_dim 512

代码默认：

```python
model_dim = 512
heads_count = 6
```

那么：

```text
512 // 6 = 85
6 × 85 = 510
```

不是 512。

虽然你的 `linear_out` 可能能映射回来，但更标准的配置是：

```python
model_dim = 512
heads_count = 8
```

因为：

```text
512 / 8 = 64
```

比较整齐。

---

## 41. seq_legth 拼写错误

代码中：

```python
seq_legth
```

更标准应该是：

```python
seq_length
```

虽然只要前后变量名一致就能运行，但建议改正。

---

## 42. 原地加法不太推荐

代码：

```python
tgt += position_encoding(seq_legth, dimension)
```

这是原地加法。

更推荐：

```python
tgt = tgt + position_encoding(seq_length, dimension).to(tgt.device)
```

这样更安全，也避免 CPU/GPU 设备不一致的问题。

---

## 43. 更推荐的 Decoder forward 写法

可以改成：

```python
def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor = None, memory_mask: Tensor = None) -> Tensor:
    seq_length, dimension = tgt.size(1), tgt.size(2)

    pos = position_encoding(seq_length, dimension).to(tgt.device)
    tgt = tgt + pos

    for layer in self.layers:
        tgt = layer(tgt, memory, tgt_mask, memory_mask)

    logits = self.final_linear(tgt)
    return torch.softmax(logits, dim=-1)
```

前提是 `TransformerDecoderLayer` 也支持 mask。

---

# 第七部分：一句话总结

---

## 44. TransformerDecoderLayer 总结

```text
TransformerDecoderLayer 是解码器的一层。

它先用 Self-Attention 看已经生成的目标 token，
再用 Cross-Attention 看 Encoder 输出的源句子信息，
最后用 Feed Forward 加工每个 token 的特征。

三个子层外面都包了 Residual + Dropout + LayerNorm。
```

---

## 45. TransformerDecoder 总结

```text
TransformerDecoder 是完整解码器。

它先给目标序列加位置编码，
再让目标序列依次通过多个 DecoderLayer，
最后通过 Linear + Softmax 输出每个位置的 token 概率。
```

---

## 46. 最终记忆版

```text
DecoderLayer 里面有三步：

1. Self-Attention：
   看自己已经生成过的目标 token。

2. Cross-Attention：
   看 Encoder 读懂的源句子 memory。

3. Feed Forward：
   加工当前 token 的特征。

完整 Decoder：
   tgt + position encoding
   → 多层 DecoderLayer
   → Linear
   → Softmax
   → token 概率
```

---

## 47. 最核心的一句话

```text
Encoder 负责读懂输入；
Decoder 负责生成输出；
Decoder 的 self-attention 看已经生成的目标词；
Decoder 的 cross-attention 看 Encoder 读懂的源句子。
```
