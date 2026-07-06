# 缩放点积注意力机制实现流程详解

## 1. 背景：什么是缩放点积注意力？

缩放点积注意力是 Transformer 中最核心的计算模块之一，英文叫：

```text
Scaled Dot-Product Attention
```

它的核心公式是：

```text
Attention(Q, K, V) = softmax(QKᵀ / √dk) V
```

其中：

| 符号      | 含义                 |
| ------- | ------------------ |
| Q       | Query，查询向量         |
| K       | Key，键向量            |
| V       | Value，值向量          |
| dk      | Key 向量的维度          |
| QKᵀ     | Query 和 Key 的相似度分数 |
| softmax | 将分数转换为概率权重         |
| mask    | 掩码，用来屏蔽不该关注的位置     |

一句话理解：

> Q 和 K 用来计算“应该关注谁”，V 用来提供“真正被聚合的信息”。

---

## 2. 代码实现

下面是缩放点积注意力的 PyTorch 实现：

```python
def scaled_dot_product(q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
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
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))

    dk = k.size()[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, v)

    return output
```

---

## 3. 整体实现流程

缩放点积注意力的执行流程如下：

```text
输入 q, k, v
    ↓
计算 QKᵀ
    ↓
除以 √dk 进行缩放
    ↓
如果有 mask，则屏蔽不该关注的位置
    ↓
对最后一维做 softmax，得到注意力权重
    ↓
用注意力权重乘 V
    ↓
得到最终输出
```

对应公式：

```text
Attention(Q, K, V) = softmax(QKᵀ / √dk) V
```

---

# 4. 逐步拆解代码

## 4.1 计算 Q 和 K 的相似度

```python
matmul_qk = torch.matmul(q, k.transpose(-2, -1))
```

这一行对应公式中的：

```text
QKᵀ
```

假设：

```text
q: (batch_size, num_heads, seq_len_q, depth)
k: (batch_size, num_heads, seq_len_k, depth)
```

由于矩阵乘法要求中间维度对齐，所以不能直接用 `q @ k`。

需要先对 `k` 的最后两个维度进行转置：

```python
k.transpose(-2, -1)
```

转置后：

```text
k:              (batch_size, num_heads, seq_len_k, depth)
k.transpose:    (batch_size, num_heads, depth, seq_len_k)
```

然后：

```text
q:   (batch_size, num_heads, seq_len_q, depth)
kᵀ:  (batch_size, num_heads, depth, seq_len_k)
```

最后两个维度做矩阵乘法：

```text
(seq_len_q, depth) @ (depth, seq_len_k)
= (seq_len_q, seq_len_k)
```

所以结果是：

```text
matmul_qk: (batch_size, num_heads, seq_len_q, seq_len_k)
```

含义是：

> 每个 Query token 和每个 Key token 都计算一个相关性分数。

---

## 4.2 获取 dk

```python
dk = k.size()[-1]
```

`k.size()` 表示获取 `k` 的形状。

例如：

```text
k.shape = (2, 8, 10, 64)
```

那么：

```python
k.size()[-1]
```

取的是最后一个维度，也就是：

```text
dk = 64
```

这里的 `dk` 表示 Key 向量的维度。

---

## 4.3 除以根号 dk

```python
scaled_attention_logits = matmul_qk / np.sqrt(dk)
```

这一行对应公式中的：

```text
QKᵀ / √dk
```

它的作用是对注意力分数进行缩放，避免点积结果过大。

如果不缩放，`QKᵀ` 的数值可能很大，进入 softmax 后会导致概率分布过于极端。

例如：

```text
softmax([20, 1, 0]) ≈ [1.0, 0.0, 0.0]
```

这样会导致梯度很小，不利于训练。

如果 `dk = 64`，那么：

```text
√dk = 8
```

原始分数：

```text
[16, 8, 4]
```

缩放后：

```text
[2, 1, 0.5]
```

softmax 之后会更加平滑，训练也更稳定。

---

## 4.4 加入 mask

```python
if mask is not None:
    scaled_attention_logits += (mask * -1e9)
```

mask 的作用是屏蔽某些位置，让模型不要关注它们。

常见 mask 有两类：

### 1. Padding Mask

用于屏蔽 `<pad>` 位置。

例如句子长度不一致时，会补齐：

```text
我 喜欢 AI <pad> <pad>
```

模型不应该关注 `<pad>`，所以要把这些位置 mask 掉。

### 2. Causal Mask / Look-ahead Mask

用于 GPT 这类自回归模型。

在生成第 3 个 token 时，只能看前面的 token，不能提前看到第 4、第 5 个 token。

例如：

```text
当前位置：第 3 个 token

允许看到：
第 1 个 token
第 2 个 token
第 3 个 token

不允许看到：
第 4 个 token
第 5 个 token
```

---

# 5. 问题答疑

---

## 问题 1：matmul 是什么含义？

`matmul` 是 `matrix multiplication` 的缩写，意思是：

```text
矩阵乘法
```

在 PyTorch 中：

```python
torch.matmul(a, b)
```

表示对 `a` 和 `b` 做矩阵乘法。

它不是逐元素相乘。

例如：

```python
a * b
```

表示对应位置相乘。

而：

```python
torch.matmul(a, b)
```

表示矩阵乘法，也就是：

```text
行 × 列，然后求和
```

---

## 问题 1.1：二维矩阵乘法规则

如果：

```text
A: (m, n)
B: (n, p)
```

那么：

```text
A @ B = C
```

结果是：

```text
C: (m, p)
```

要求：

```text
A 的列数 = B 的行数
```

也就是：

```text
(m, n) @ (n, p) = (m, p)
```

例如：

```text
(3, 4) @ (4, 5) = (3, 5)
```

---

## 问题 1.2：多维矩阵乘法规则

对于多维张量，`torch.matmul` 的规则是：

```text
最后两个维度做矩阵乘法，前面的维度当作 batch 维度。
```

通用公式：

```text
(..., m, n) @ (..., n, p) = (..., m, p)
```

其中：

| 符号    | 含义       |
| ----- | -------- |
| `...` | batch 维度 |
| `m`   | 左边矩阵的行数  |
| `n`   | 中间匹配维度   |
| `p`   | 右边矩阵的列数  |

例如：

```text
A: (2, 8, 10, 64)
B: (2, 8, 64, 12)
```

最后两个维度参与矩阵乘法：

```text
(10, 64) @ (64, 12) = (10, 12)
```

前面的 batch 维度 `(2, 8)` 保持不变。

所以结果是：

```text
(2, 8, 10, 12)
```

---

## 问题 2：transpose 是什么含义？

`transpose` 的意思是：

```text
交换两个维度
```

在 PyTorch 中：

```python
k.transpose(-2, -1)
```

表示交换 `k` 的倒数第二个维度和倒数第一个维度。

假设：

```text
k: (batch_size, num_heads, seq_len_k, depth)
```

那么：

```python
k.transpose(-2, -1)
```

结果是：

```text
k.transpose: (batch_size, num_heads, depth, seq_len_k)
```

为什么要这么做？

因为我们要计算：

```text
QKᵀ
```

也就是要用 `Q` 乘以 `K` 的转置。

如果不转置：

```text
q: (batch_size, num_heads, seq_len_q, depth)
k: (batch_size, num_heads, seq_len_k, depth)
```

最后两个维度是：

```text
(seq_len_q, depth) @ (seq_len_k, depth)
```

中间维度对不上，无法做矩阵乘法。

转置后：

```text
q:  (batch_size, num_heads, seq_len_q, depth)
kᵀ: (batch_size, num_heads, depth, seq_len_k)
```

最后两个维度是：

```text
(seq_len_q, depth) @ (depth, seq_len_k)
```

此时可以相乘。

---

## 问题 3：整体张量形状变化过程

假设：

```text
batch_size = 2
num_heads = 8
seq_len_q = 10
seq_len_k = 10
seq_len_v = 10
depth = 64
depth_v = 64
```

那么初始输入形状是：

```text
q: (2, 8, 10, 64)
k: (2, 8, 10, 64)
v: (2, 8, 10, 64)
```

### 第一步：对 k 转置

```python
k.transpose(-2, -1)
```

形状变化：

```text
k:    (2, 8, 10, 64)
kᵀ:   (2, 8, 64, 10)
```

---

### 第二步：计算 QKᵀ

```python
matmul_qk = torch.matmul(q, k.transpose(-2, -1))
```

形状变化：

```text
q:    (2, 8, 10, 64)
kᵀ:   (2, 8, 64, 10)
```

最后两个维度相乘：

```text
(10, 64) @ (64, 10) = (10, 10)
```

结果：

```text
matmul_qk: (2, 8, 10, 10)
```

含义：

```text
每个样本、每个注意力头中，每个 Query token 对所有 Key token 的注意力分数。
```

---

### 第三步：缩放

```python
scaled_attention_logits = matmul_qk / np.sqrt(dk)
```

形状不变：

```text
scaled_attention_logits: (2, 8, 10, 10)
```

---

### 第四步：加入 mask

```python
scaled_attention_logits += (mask * -1e9)
```

如果 mask 的形状是：

```text
mask: (2, 1, 10, 10)
```

它可以广播到：

```text
(2, 8, 10, 10)
```

所以加入 mask 后形状仍然是：

```text
scaled_attention_logits: (2, 8, 10, 10)
```

---

### 第五步：softmax 得到注意力权重

```python
attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
```

形状不变：

```text
attention_weights: (2, 8, 10, 10)
```

含义是：

```text
每个 Query token 对所有 Key token 的注意力权重。
```

最后一维的权重加起来等于 1。

---

### 第六步：乘以 V

```python
output = torch.matmul(attention_weights, v)
```

形状：

```text
attention_weights: (2, 8, 10, 10)
v:                 (2, 8, 10, 64)
```

最后两个维度做矩阵乘法：

```text
(10, 10) @ (10, 64) = (10, 64)
```

结果：

```text
output: (2, 8, 10, 64)
```

---

### 总结表

| 步骤      | 操作                      | 形状               |
| ------- | ----------------------- | ---------------- |
| 输入 q    | 原始 Query                | `(2, 8, 10, 64)` |
| 输入 k    | 原始 Key                  | `(2, 8, 10, 64)` |
| 输入 v    | 原始 Value                | `(2, 8, 10, 64)` |
| k 转置    | `k.transpose(-2, -1)`   | `(2, 8, 64, 10)` |
| QKᵀ     | `torch.matmul(q, kᵀ)`   | `(2, 8, 10, 10)` |
| 缩放      | `/ sqrt(dk)`            | `(2, 8, 10, 10)` |
| 加 mask  | `+ mask * -1e9`         | `(2, 8, 10, 10)` |
| softmax | `softmax(..., dim=-1)`  | `(2, 8, 10, 10)` |
| 乘 V     | `attention_weights @ v` | `(2, 8, 10, 64)` |

---

## 问题 4：为什么需要掩码？如何参与计算？

掩码的作用是：

```text
让模型不要关注某些位置。
```

例如：

```text
原始 attention logits:
[2.0, 1.0, 0.5, 3.0]
```

假设最后一个位置不能被关注：

```text
mask:
[0, 0, 0, 1]
```

代码中执行：

```python
scaled_attention_logits += (mask * -1e9)
```

计算后：

```text
[2.0, 1.0, 0.5, 3.0] + [0, 0, 0, -1e9]
= [2.0, 1.0, 0.5, -999999997.0]
```

然后经过 softmax：

```text
softmax([2.0, 1.0, 0.5, -1e9])
≈ [0.63, 0.23, 0.14, 0.00]
```

最后一个位置的权重几乎为 0，因此模型不会关注它。

所以 mask 不是直接删除某个 token，而是通过给它加一个极大的负数，让它在 softmax 后变成接近 0 的概率。

---

## 问题 5：为什么要除以根号 dk？

在注意力中，Query 和 Key 会做点积：

```text
QKᵀ
```

如果向量维度 `dk` 很大，点积结果的方差也会变大，分数可能变得很大。

分数过大时，softmax 会变得非常尖锐。

例如：

```text
softmax([100, 1, 0]) ≈ [1.0, 0.0, 0.0]
```

这种情况下，大部分位置的概率都接近 0，梯度也会变小。

除以 `√dk` 的作用是：

```text
控制点积结果的数值范围，让 softmax 不至于过度饱和。
```

例如：

```text
dk = 64
√dk = 8
```

原始分数：

```text
[16, 8, 4]
```

缩放后：

```text
[2, 1, 0.5]
```

缩放后的分数进入 softmax，会得到更平滑的概率分布，有利于训练稳定。

---

## 问题 6：softmax 的函数实现是什么？为什么要使用 softmax？

softmax 的作用是：

```text
把一组任意实数转换成一组概率。
```

它有两个特点：

```text
1. 每个值都大于 0
2. 所有值加起来等于 1
```

softmax 的公式是：

```text
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

例如输入：

```text
x = [2.0, 1.0, 0.5]
```

计算步骤：

```text
exp(2.0) ≈ 7.39
exp(1.0) ≈ 2.72
exp(0.5) ≈ 1.65
```

总和：

```text
7.39 + 2.72 + 1.65 = 11.76
```

所以：

```text
softmax(x) ≈ [
  7.39 / 11.76,
  2.72 / 11.76,
  1.65 / 11.76
]
```

结果约为：

```text
[0.63, 0.23, 0.14]
```

这组数可以理解为注意力权重。

---

### softmax 的简单 Python 实现

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)
```

但是这个版本存在数值稳定性问题。

如果输入很大：

```python
x = np.array([1000, 1001, 1002])
```

`np.exp(x)` 可能会溢出。

所以实际实现通常会减去最大值：

```python
import numpy as np

def stable_softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)
```

对于多维张量，一般要指定维度：

```python
def stable_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

在 PyTorch 中直接使用：

```python
torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
```

其中：

```text
dim=-1
```

表示在最后一个维度上做 softmax。

在注意力机制中，最后一个维度是：

```text
seq_len_k
```

所以它表示：

```text
对每个 Query token 来说，在所有 Key token 上分配注意力权重。
```

---

## 为什么注意力机制要用 softmax？

因为 `QKᵀ / √dk` 得到的只是原始分数，不是概率。

例如：

```text
[2.0, 1.0, 0.5]
```

这些数只能表示相对大小，但不能直接表示“关注比例”。

经过 softmax 后：

```text
[0.63, 0.23, 0.14]
```

就可以解释为：

```text
关注第 1 个 token 的权重是 63%
关注第 2 个 token 的权重是 23%
关注第 3 个 token 的权重是 14%
```

然后再用这些权重去加权求和 V：

```text
output = 0.63 * V1 + 0.23 * V2 + 0.14 * V3
```

所以 softmax 的作用是：

```text
把注意力分数转换成可以加权求和的注意力概率。
```

---

# 6. 最终总结

缩放点积注意力可以总结为六步：

```text
1. Q 和 Kᵀ 做矩阵乘法，得到注意力分数
2. 除以 √dk，防止分数过大
3. 加入 mask，屏蔽不该关注的位置
4. 使用 softmax，将分数变成概率权重
5. 用注意力权重乘 V，汇总上下文信息
6. 输出新的 token 表示
```

最核心的公式是：

```text
Attention(Q, K, V) = softmax(QKᵀ / √dk) V
```

一句话理解：

> QKᵀ 决定“关注谁”，softmax 决定“关注多少”，V 提供“被聚合的信息”。
