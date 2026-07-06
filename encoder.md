# Transformer 编码器层与编码器阅读文档

## 1. 代码原文

```python
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
```

---

## 2. 先用一句话理解

这段代码实现了 Transformer 的 **Encoder 编码器部分**。

它分成两层结构：

```text
TransformerEncoderLayer：一个编码器层

TransformerEncoder：完整编码器，由多个 EncoderLayer 堆叠而成
```

也就是说：

```text
TransformerEncoder
  ├── TransformerEncoderLayer 1
  ├── TransformerEncoderLayer 2
  ├── TransformerEncoderLayer 3
  ├── ...
  └── TransformerEncoderLayer N
```

一个 EncoderLayer 内部包含：

```text
多头自注意力 Multi-Head Self-Attention
前馈神经网络 Feed Forward Network
```

并且每个子层外面都包了：

```text
Residual + Dropout + LayerNorm
```

---

## 3. 整体结构图

```text
输入 src
  ↓
加位置编码 position_encoding
  ↓
EncoderLayer 1
  ↓
EncoderLayer 2
  ↓
EncoderLayer 3
  ↓
...
  ↓
EncoderLayer N
  ↓
编码器输出
```

其中每一个 `EncoderLayer` 内部是：

```text
输入 src
  ↓
Multi-Head Self-Attention
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

# 第一部分：TransformerEncoderLayer 编码器层

---

## 4. TransformerEncoderLayer 是什么？

```python
class TransformerEncoderLayer(nn.Module):
```

这定义了一个 Transformer 编码器层。

注意：

```text
它不是完整 Encoder。
它只是 Encoder 里面的一层。
```

完整 Encoder 是由很多个这样的层堆起来的。

例如原始 Transformer 论文中，Encoder 默认堆叠 6 层：

```text
EncoderLayer × 6
```

---

## 5. 初始化函数

```python
def __init__(self, model_dim: int, heads_count: int, feed_forward_dim: int, dropout_rate: float = 0.1):
```

这个初始化函数用于创建一个编码器层内部需要的模块。

参数含义如下：

| 参数                 | 含义                      |
| ------------------ | ----------------------- |
| `model_dim`        | 模型主维度，也就是每个 token 向量的维度 |
| `heads_count`      | 多头注意力的头数                |
| `feed_forward_dim` | 前馈神经网络中间层维度             |
| `dropout_rate`     | Dropout 概率，默认是 0.1      |

比如：

```python
model_dim = 512
heads_count = 8
feed_forward_dim = 2048
dropout_rate = 0.1
```

可以理解为：

```text
每个 token 是 512 维
多头注意力有 8 个头
前馈网络中间层是 2048 维
训练时 Dropout 概率为 0.1
```

---

## 6. 调用父类初始化

```python
super().__init__()
```

这一行是 PyTorch 模块必须要写的。

因为这个类继承自：

```python
nn.Module
```

调用：

```python
super().__init__()
```

可以让 PyTorch 正确管理这个模块里面的子模块和参数。

如果不写，下面这些模块可能无法被 PyTorch 正确识别：

```python
self.multi_head_attention
self.feedforward_network
```

---

## 7. 计算每个注意力头的维度

```python
query_dim = key_dim = max(model_dim // heads_count, 1)
```

这一行是在计算：

```text
每个注意力头里面 Query 和 Key 的维度。
```

---

### 7.1 `model_dim // heads_count`

假设：

```python
model_dim = 512
heads_count = 8
```

那么：

```python
model_dim // heads_count
```

就是：

```text
512 // 8 = 64
```

所以：

```text
query_dim = 64
key_dim = 64
```

也就是说：

```text
每个注意力头使用 64 维的 Q 和 K。
```

---

### 7.2 为什么要除以 heads_count？

因为多头注意力会把总维度拆给多个头。

如果：

```text
model_dim = 512
heads_count = 8
```

那么每个头一般分到：

```text
512 / 8 = 64 维
```

8 个头拼接回来：

```text
8 × 64 = 512
```

刚好回到模型主维度。

---

### 7.3 `//` 是什么？

```python
model_dim // heads_count
```

`//` 是 Python 里的整除。

例如：

```python
512 // 8 = 64
513 // 8 = 64
```

它会直接舍掉小数部分。

---

### 7.4 `max(..., 1)` 的作用

```python
max(model_dim // heads_count, 1)
```

这是为了避免每个头的维度变成 0。

例如：

```python
model_dim = 4
heads_count = 8
```

那么：

```text
4 // 8 = 0
```

如果 Query 和 Key 的维度是 0，模型肯定无法计算。

所以用：

```python
max(0, 1)
```

得到：

```text
1
```

保证至少有 1 维。

---

### 7.5 这行代码的整体意思

```python
query_dim = key_dim = max(model_dim // heads_count, 1)
```

可以理解为：

```text
每个注意力头的 Query 维度 = 每个注意力头的 Key 维度
通常等于 model_dim / heads_count
如果算出来小于 1，就强制设为 1
```

---

## 8. 创建多头注意力残差块

```python
self.multi_head_attention = Residual(
    MultiHeadAttention(model_dim, heads_count, query_dim, key_dim),
    model_dim,
    dropout_rate=dropout_rate
)
```

这一段创建的是编码器层的第一大块：

```text
多头自注意力 + 残差连接 + Dropout + LayerNorm
```

---

### 8.1 先看里面的 MultiHeadAttention

```python
MultiHeadAttention(model_dim, heads_count, query_dim, key_dim)
```

这会创建一个多头注意力模块。

它的作用是：

```text
让每个 token 去看同一句话里的其他 token。
```

比如一句话：

```text
我 喜欢 学习 AI
```

多头注意力会让：

```text
“我” 可以看 “喜欢”
“学习” 可以看 “AI”
“AI” 可以看 “学习”
```

也就是让 token 之间交换信息。

---

### 8.2 再看外面的 Residual

```python
Residual(
    MultiHeadAttention(...),
    model_dim,
    dropout_rate=dropout_rate
)
```

`Residual` 会把子层包装成：

```text
LayerNorm(x + Dropout(Sublayer(x)))
```

这里的 `Sublayer` 就是：

```text
MultiHeadAttention
```

所以这一整块等价于：

```text
LayerNorm(src + Dropout(MultiHeadAttention(src, src, src, mask)))
```

也就是 Transformer 里的：

```text
Add & Norm
```

---

## 9. 创建前馈神经网络残差块

```python
self.feedforward_network = Residual(
    feed_forward(model_dim, feed_forward_dim),
    model_dim, 
    dropout_rate=dropout_rate
)
```

这一段创建的是编码器层的第二大块：

```text
前馈神经网络 + 残差连接 + Dropout + LayerNorm
```

---

### 9.1 feed_forward 是什么？

如果：

```python
model_dim = 512
feed_forward_dim = 2048
```

那么：

```python
feed_forward(model_dim, feed_forward_dim)
```

大概等价于：

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

前馈神经网络的作用是：

```text
对每个 token 自己的特征做进一步加工。
```

---

### 9.2 注意力层和前馈层的区别

注意力层负责：

```text
token 和 token 之间交流信息。
```

前馈层负责：

```text
每个 token 自己加工自己的特征。
```

一句话：

```text
注意力：看别人
前馈网络：加工自己
```

---

## 10. EncoderLayer 的 forward 函数

```python
def forward(self, src: Tensor, mask: Tensor = None) -> Tensor:
    src = self.multi_head_attention(src, src, src, mask)
    return self.feedforward_network(src)
```

这个 `forward` 定义了一个编码器层的数据流动过程。

---

## 11. forward 第一步：多头自注意力

```python
src = self.multi_head_attention(src, src, src, mask)
```

这一行调用的是：

```python
self.multi_head_attention
```

也就是前面创建的：

```text
Residual(MultiHeadAttention(...))
```

---

### 11.1 为什么传三个 src？

多头注意力需要三个输入：

```text
Query
Key
Value
```

而这里传的是：

```python
src, src, src
```

也就是：

```text
Query = src
Key   = src
Value = src
```

这叫：

```text
Self-Attention 自注意力
```

意思是：

```text
Q、K、V 都来自同一个输入序列。
```

所以这一层是让输入序列内部的 token 互相看。

---

### 11.2 这一行展开后是什么？

因为 `self.multi_head_attention` 是一个 `Residual` 模块，所以：

```python
src = self.multi_head_attention(src, src, src, mask)
```

内部大概等价于：

```text
src = LayerNorm(src + Dropout(MultiHeadAttention(src, src, src, mask)))
```

也就是：

```text
先做多头自注意力
再做 Dropout
再和原始 src 相加
最后做 LayerNorm
```

---

## 12. forward 第二步：前馈神经网络

```python
return self.feedforward_network(src)
```

这一行调用的是：

```python
self.feedforward_network
```

也就是：

```text
Residual(feed_forward(...))
```

所以它内部大概等价于：

```text
output = LayerNorm(src + Dropout(FeedForward(src)))
```

---

## 13. EncoderLayer forward 完整展开

原始代码：

```python
def forward(self, src: Tensor, mask: Tensor = None) -> Tensor:
    src = self.multi_head_attention(src, src, src, mask)
    return self.feedforward_network(src)
```

可以展开理解为：

```python
def forward(self, src, mask=None):
    # 1. 多头自注意力 + 残差连接 + LayerNorm
    src = LayerNorm(src + Dropout(MultiHeadAttention(src, src, src, mask)))

    # 2. 前馈神经网络 + 残差连接 + LayerNorm
    src = LayerNorm(src + Dropout(FeedForward(src)))

    return src
```

---

## 14. EncoderLayer 形状变化例子

假设：

```python
batch_size = 2
seq_len = 5
model_dim = 512
heads_count = 8
feed_forward_dim = 2048
```

输入：

```text
src: (2, 5, 512)
```

含义：

```text
2 个样本
每个样本 5 个 token
每个 token 是 512 维
```

---

### 14.1 多头注意力部分

每个头的维度：

```text
query_dim = key_dim = 512 // 8 = 64
```

每个头内部：

```text
Q: (2, 5, 64)
K: (2, 5, 64)
V: (2, 5, 64)
```

计算注意力分数：

```text
QK^T: (2, 5, 5)
```

这里的 `(5, 5)` 表示：

```text
5 个 token 两两之间的注意力关系。
```

每个头输出：

```text
head_output: (2, 5, 64)
```

8 个头拼接：

```text
concat: (2, 5, 512)
```

经过输出线性层：

```text
attention_output: (2, 5, 512)
```

残差相加：

```text
src + attention_output
(2, 5, 512) + (2, 5, 512) = (2, 5, 512)
```

LayerNorm 后：

```text
src: (2, 5, 512)
```

---

### 14.2 前馈网络部分

前馈网络是：

```text
Linear(512, 2048)
ReLU
Linear(2048, 512)
```

形状变化：

```text
(2, 5, 512)
  ↓ Linear(512, 2048)
(2, 5, 2048)
  ↓ ReLU
(2, 5, 2048)
  ↓ Linear(2048, 512)
(2, 5, 512)
```

残差相加：

```text
src + feedforward_output
(2, 5, 512) + (2, 5, 512) = (2, 5, 512)
```

最终输出：

```text
(2, 5, 512)
```

---

# 第二部分：TransformerEncoder 编码器

---

## 15. TransformerEncoder 是什么？

```python
class TransformerEncoder(nn.Module):
```

这个类实现的是完整的 Transformer 编码器。

它不是一层，而是由多层 `TransformerEncoderLayer` 堆叠起来。

如果：

```python
layer_count = 6
```

那么它里面有：

```text
6 个 TransformerEncoderLayer
```

整体结构是：

```text
输入 src
  ↓
加位置编码
  ↓
EncoderLayer 1
  ↓
EncoderLayer 2
  ↓
EncoderLayer 3
  ↓
EncoderLayer 4
  ↓
EncoderLayer 5
  ↓
EncoderLayer 6
  ↓
输出
```

---

## 16. Encoder 初始化函数

```python
def __init__(self, layer_count: int = 6, model_dim: int = 512, heads_count: int = 6, feed_forward_dim: int = 2048, dropout_rate: float = 0.1):
```

参数含义：

| 参数                 | 含义                |
| ------------------ | ----------------- |
| `layer_count`      | 编码器层数，默认 6        |
| `model_dim`        | 模型主维度，默认 512      |
| `heads_count`      | 注意力头数，默认 6        |
| `feed_forward_dim` | 前馈网络中间层维度，默认 2048 |
| `dropout_rate`     | Dropout 概率，默认 0.1 |

---

## 17. 创建多层 EncoderLayer

```python
self.layers = nn.ModuleList([
    TransformerEncoderLayer(model_dim, heads_count, feed_forward_dim, dropout_rate)
    for _ in range(layer_count)
])
```

这一段非常重要。

它创建了多个编码器层，并用 `nn.ModuleList` 保存起来。

---

### 17.1 `range(layer_count)`

如果：

```python
layer_count = 6
```

那么：

```python
for _ in range(layer_count)
```

会循环 6 次。

每次创建一个：

```python
TransformerEncoderLayer(...)
```

所以最终创建 6 层。

---

### 17.2 为什么用 `_`？

```python
for _ in range(layer_count)
```

这里的 `_` 表示：

```text
这个循环变量本身不用。
```

也就是说，我们只关心循环几次，不关心当前是第几次。

---

### 17.3 为什么用 nn.ModuleList？

```python
self.layers = nn.ModuleList([...])
```

`nn.ModuleList` 是 PyTorch 专门用来保存多个子模块的容器。

不能随便用普通 Python list 替代。

原因是：

```text
普通 list 可能导致 PyTorch 不能正确注册子模块参数。
nn.ModuleList 可以让 PyTorch 正确管理每一层的参数。
```

这样优化器才能找到这些层里的参数并更新。

---

## 18. Encoder 的 forward 函数

```python
def forward(self, src: Tensor, mask: Tensor = None) -> Tensor:
    seq_legth, dimension = src.size(1), src.size(2)
    src += position_encoding(seq_legth, dimension)  # Add positional encoding
    for layer in self.layers:
        src = layer(src, mask)
    return src
```

这个 `forward` 定义了完整编码器的数据流动过程。

---

## 19. 取得序列长度和维度

```python
seq_legth, dimension = src.size(1), src.size(2)
```

假设输入：

```text
src: (batch_size, seq_len, model_dim)
```

例如：

```text
src: (2, 5, 512)
```

那么：

```python
src.size(1)
```

得到：

```text
5
```

表示序列长度。

```python
src.size(2)
```

得到：

```text
512
```

表示每个 token 的维度。

所以：

```text
seq_legth = 5
dimension = 512
```

注意：这里变量名写成了 `seq_legth`，应该是拼写错误，更标准应该写成：

```python
seq_length
```

不过只要后面也用 `seq_legth`，程序仍然能运行。

---

## 20. 加位置编码

```python
src += position_encoding(seq_legth, dimension)
```

这一行给输入加上位置编码。

输入 `src` 形状是：

```text
(batch_size, seq_len, model_dim)
```

例如：

```text
src: (2, 5, 512)
```

位置编码形状是：

```text
(1, seq_len, model_dim)
```

例如：

```text
position_encoding: (1, 5, 512)
```

两者相加：

```text
(2, 5, 512) + (1, 5, 512) = (2, 5, 512)
```

这一步的作用是：

```text
把 token 的位置信息加入输入表示。
```

因为 Transformer 自注意力本身不知道顺序，所以需要位置编码告诉它：

```text
哪个 token 在第 0 个位置
哪个 token 在第 1 个位置
哪个 token 在第 2 个位置
...
```

---

## 21. 逐层通过 EncoderLayer

```python
for layer in self.layers:
    src = layer(src, mask)
```

这一段表示：

```text
让 src 依次通过每一个 EncoderLayer。
```

如果有 6 层，展开就是：

```python
src = layer1(src, mask)
src = layer2(src, mask)
src = layer3(src, mask)
src = layer4(src, mask)
src = layer5(src, mask)
src = layer6(src, mask)
```

每一层都会做：

```text
多头自注意力
残差连接 + LayerNorm
前馈神经网络
残差连接 + LayerNorm
```

---

## 22. 返回最终编码结果

```python
return src
```

最后返回经过所有编码器层处理后的结果。

输出形状仍然是：

```text
(batch_size, seq_len, model_dim)
```

例如：

```text
(2, 5, 512)
```

虽然形状不变，但是里面的内容已经变了。

每个 token 的向量已经融合了上下文信息。

例如原来：

```text
“学习” 只表示自己
```

经过编码器后：

```text
“学习” 的向量可能已经融合了 “我”“喜欢”“AI” 等上下文信息
```

---

# 第三部分：forward 是如何嵌套调用的？

---

## 23. 总体调用链

当你调用：

```python
output = encoder(src, mask)
```

实际会触发：

```text
TransformerEncoder.forward(src, mask)
```

然后里面会循环调用每个层：

```text
TransformerEncoderLayer.forward(src, mask)
```

EncoderLayer 内部又会调用：

```text
Residual.forward(...)
```

Residual 内部再调用：

```text
MultiHeadAttention.forward(...)
```

MultiHeadAttention 内部再调用：

```text
AttentionHead.forward(...)
```

AttentionHead 内部再调用：

```text
Linear.forward(...)
scaled_dot_product(...)
```

所以整体是层层嵌套的。

---

## 24. 完整 forward 调用链

```text
TransformerEncoder.forward(src, mask)
│
├── 加 position_encoding
│
├── EncoderLayer 1.forward(src, mask)
│   │
│   ├── Residual.forward(src, src, src, mask)
│   │   │
│   │   ├── MultiHeadAttention.forward(src, src, src, mask)
│   │   │   │
│   │   │   ├── AttentionHead 1.forward(...)
│   │   │   ├── AttentionHead 2.forward(...)
│   │   │   ├── ...
│   │   │   └── linear_out(...)
│   │   │
│   │   ├── Dropout
│   │   ├── 残差相加
│   │   └── LayerNorm
│   │
│   └── Residual.forward(src)
│       │
│       ├── FeedForward.forward(src)
│       │   │
│       │   ├── Linear
│       │   ├── ReLU
│       │   └── Linear
│       │
│       ├── Dropout
│       ├── 残差相加
│       └── LayerNorm
│
├── EncoderLayer 2.forward(src, mask)
│
├── EncoderLayer 3.forward(src, mask)
│
├── ...
│
└── return src
```

---

## 25. PyTorch 中为什么写 `layer(src, mask)` 而不是 `layer.forward(src, mask)`？

标准写法是：

```python
layer(src, mask)
```

而不是：

```python
layer.forward(src, mask)
```

因为在 PyTorch 里：

```python
layer(src, mask)
```

内部会自动调用：

```python
layer.forward(src, mask)
```

并且在调用 forward 前后，PyTorch 还会处理一些额外逻辑，比如 hook、状态管理等。

所以标准写法永远是：

```python
module(input)
```

而不是直接手写：

```python
module.forward(input)
```

---

# 第四部分：完整形状流动示例

---

## 26. 假设输入参数

```python
batch_size = 2
seq_len = 5
model_dim = 512
layer_count = 6
heads_count = 8
feed_forward_dim = 2048
```

输入：

```text
src: (2, 5, 512)
```

---

## 27. 加位置编码

位置编码：

```text
position_encoding(5, 512): (1, 5, 512)
```

相加：

```text
src:              (2, 5, 512)
position_encoding:(1, 5, 512)
--------------------------------
result:           (2, 5, 512)
```

形状不变。

---

## 28. 通过 EncoderLayer 1

输入：

```text
(2, 5, 512)
```

多头自注意力后：

```text
(2, 5, 512)
```

前馈网络后：

```text
(2, 5, 512)
```

输出：

```text
(2, 5, 512)
```

---

## 29. 通过 6 层后

每一层都保持形状不变：

```text
Layer 1: (2, 5, 512) → (2, 5, 512)
Layer 2: (2, 5, 512) → (2, 5, 512)
Layer 3: (2, 5, 512) → (2, 5, 512)
Layer 4: (2, 5, 512) → (2, 5, 512)
Layer 5: (2, 5, 512) → (2, 5, 512)
Layer 6: (2, 5, 512) → (2, 5, 512)
```

最终输出：

```text
(2, 5, 512)
```

虽然形状没有变，但每个 token 的向量内容已经融合了上下文信息。

---

# 第五部分：这段代码需要注意的问题

---

## 30. heads_count 默认值是 6，和 model_dim=512 不太匹配

代码里：

```python
layer_count: int = 6
model_dim: int = 512
heads_count: int = 6
```

如果：

```text
model_dim = 512
heads_count = 6
```

那么：

```text
512 // 6 = 85
```

每个头是 85 维。

6 个头拼接：

```text
6 × 85 = 510
```

不是 512。

如果你的 `MultiHeadAttention` 最后有：

```python
nn.Linear(heads_count * key_dim, model_dim)
```

那么它仍然可以从 510 映射回 512。

但是更标准的 Transformer 通常要求：

```text
model_dim 能被 heads_count 整除。
```

例如：

```text
512 / 8 = 64
768 / 12 = 64
1024 / 16 = 64
```

所以更常见的配置是：

```python
model_dim = 512
heads_count = 8
```

或者加一个检查：

```python
assert model_dim % heads_count == 0
```

---

## 31. seq_legth 拼写问题

代码里写的是：

```python
seq_legth
```

更标准应该是：

```python
seq_length
```

虽然不影响运行，但建议修改，方便阅读。

可以改成：

```python
seq_length, dimension = src.size(1), src.size(2)
src += position_encoding(seq_length, dimension)
```

---

## 32. `src += position_encoding(...)` 是原地加法

代码：

```python
src += position_encoding(seq_legth, dimension)
```

这是原地加法。

更安全、更常见的写法是：

```python
src = src + position_encoding(seq_length, dimension)
```

这样不会原地修改原来的 `src`，更容易避免一些梯度计算或调试问题。

---

## 33. position_encoding 的设备问题

如果 `src` 在 GPU 上，而 `position_encoding(...)` 默认生成在 CPU 上，可能会报错。

例如：

```text
src 在 cuda
position_encoding 在 cpu
```

相加会失败。

更稳妥的写法是：

```python
pos = position_encoding(seq_length, dimension).to(src.device)
src = src + pos
```

---

## 34. 更推荐的 forward 写法

可以把 Encoder 的 forward 改成：

```python
def forward(self, src: Tensor, mask: Tensor = None) -> Tensor:
    seq_length, dimension = src.size(1), src.size(2)

    pos = position_encoding(seq_length, dimension).to(src.device)
    src = src + pos

    for layer in self.layers:
        src = layer(src, mask)

    return src
```

这样更清晰，也更安全。

---

# 第六部分：一句话总结

---

## 35. TransformerEncoderLayer 总结

```text
TransformerEncoderLayer 是编码器的一层。

它先用多头自注意力让 token 之间交换信息，
再用前馈神经网络加工每个 token 自己的特征。

两个子层外面都包了 Residual + Dropout + LayerNorm。
```

公式可以写成：

```text
src = LayerNorm(src + Dropout(MultiHeadAttention(src, src, src, mask)))

output = LayerNorm(src + Dropout(FeedForward(src)))
```

---

## 36. TransformerEncoder 总结

```text
TransformerEncoder 是完整编码器。

它先给输入加位置编码，
然后让输入依次通过多个 TransformerEncoderLayer。
```

公式可以写成：

```text
src = src + position_encoding

for layer in layers:
    src = layer(src, mask)

return src
```

---

## 37. 最终记忆版

```text
EncoderLayer = 一层加工

Encoder = 多层 EncoderLayer 堆叠

EncoderLayer 里面：
1. Attention：看别人
2. FeedForward：加工自己
3. Residual：保留原始信息
4. LayerNorm：稳定训练

Encoder 里面：
1. 先加位置编码
2. 再一层一层通过 EncoderLayer
```
