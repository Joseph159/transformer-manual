# 单头注意力与多头注意力机制阅读文档

本文基于以下两个模块进行解释：

* `AttentionHead`：单头注意力
* `MultiHeadAttention`：多头注意力

核心思路是：

```text
单头注意力：
query / key / value
  ↓
分别经过线性层生成 Q / K / V
  ↓
计算 scaled_dot_product
  ↓
得到一个注意力头的输出

多头注意力：
多个 AttentionHead 并行计算
  ↓
把多个 head 的输出拼接
  ↓
经过 linear_out 融合
  ↓
得到最终输出
```

---

# 一、单头注意力代码

```python
class AttentionHead(nn.Module):
    """
    单个注意力头的实现
    Args:
        d_model (int): 输入特征的维度
        d_k (int): Query和Key的维度
        d_v (int): Value的维度
    """
    def __init__(self, dim_input: int, dim_q: int, dim_k: int):
        super().__init__()

        self.linear_query = nn.Linear(dim_input, dim_q)
        self.linear_key = nn.Linear(dim_input, dim_k)
        self.linear_value = nn.Linear(dim_input, dim_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        return scaled_dot_product(
            self.linear_query(query),
            self.linear_key(key),
            self.linear_value(value),
            mask
        )
```

---

# 二、单头注意力整体流程

单头注意力的执行流程是：

```text
输入 query, key, value
  ↓
linear_query(query) 生成 Q
linear_key(key)     生成 K
linear_value(value) 生成 V
  ↓
调用 scaled_dot_product(Q, K, V, mask)
  ↓
计算 QKᵀ / √dk
  ↓
加 mask
  ↓
softmax 得到注意力权重
  ↓
注意力权重乘 V
  ↓
输出单个注意力头的结果
```

也就是：

```text
Attention(Q, K, V) = softmax(QKᵀ / √dk) V
```

---

# 三、单头注意力逐行解释

## 1. 定义类

```python
class AttentionHead(nn.Module):
```

定义一个 `AttentionHead` 类，表示一个注意力头。

它继承自 `nn.Module`，所以它是一个 PyTorch 神经网络模块，可以被训练，可以被优化器更新参数。

---

## 2. 初始化函数

```python
def __init__(self, dim_input: int, dim_q: int, dim_k: int):
```

初始化单头注意力。

参数含义如下：

| 参数          | 含义                           |
| ----------- | ---------------------------- |
| `dim_input` | 输入 token 的特征维度               |
| `dim_q`     | Query 的维度                    |
| `dim_k`     | Key 的维度，同时这段代码里也作为 Value 的维度 |

例如：

```python
head = AttentionHead(dim_input=512, dim_q=64, dim_k=64)
```

表示：

```text
输入 token 是 512 维
Query 被映射成 64 维
Key 被映射成 64 维
Value 也被映射成 64 维
```

---

## 3. 调用父类初始化

```python
super().__init__()
```

调用 `nn.Module` 的初始化方法。

这一行的作用是让 PyTorch 正确管理这个模块里面的参数，例如：

```python
self.linear_query
self.linear_key
self.linear_value
```

如果不调用 `super().__init__()`，这些子模块可能无法被 PyTorch 正确注册。

---

## 4. Query 线性层

```python
self.linear_query = nn.Linear(dim_input, dim_q)
```

定义一个全连接层，把输入映射成 Query。

假设：

```text
query: (batch_size, seq_len_q, dim_input)
```

经过这个线性层后：

```text
Q: (batch_size, seq_len_q, dim_q)
```

例如：

```text
query: (2, 5, 512)
dim_q = 64
```

那么：

```text
Q: (2, 5, 64)
```

含义是：

> 每个 query token 从 512 维变成 64 维，用来表示“我想关注谁”。

---

## 5. Key 线性层

```python
self.linear_key = nn.Linear(dim_input, dim_k)
```

定义一个全连接层，把输入映射成 Key。

假设：

```text
key: (batch_size, seq_len_k, dim_input)
```

经过这个线性层后：

```text
K: (batch_size, seq_len_k, dim_k)
```

例如：

```text
key: (2, 7, 512)
dim_k = 64
```

那么：

```text
K: (2, 7, 64)
```

含义是：

> 每个 key token 从 512 维变成 64 维，用来表示“我有什么特征，是否值得被关注”。

---

## 6. Value 线性层

```python
self.linear_value = nn.Linear(dim_input, dim_k)
```

定义一个全连接层，把输入映射成 Value。

假设：

```text
value: (batch_size, seq_len_v, dim_input)
```

经过这个线性层后：

```text
V: (batch_size, seq_len_v, dim_k)
```

例如：

```text
value: (2, 7, 512)
dim_k = 64
```

那么：

```text
V: (2, 7, 64)
```

含义是：

> 每个 value token 从 512 维变成 64 维，用来提供最终被加权汇总的信息。

注意：这里代码中 `linear_value` 的输出维度是 `dim_k`，说明作者默认：

```text
Value 的维度 = Key 的维度
```

更标准的写法通常会单独设置 `dim_v`：

```python
self.linear_value = nn.Linear(dim_input, dim_v)
```

---

## 7. forward 函数

```python
def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
```

`forward` 是前向传播函数。

当你调用：

```python
output = head(query, key, value, mask)
```

PyTorch 实际上会自动调用：

```python
head.forward(query, key, value, mask)
```

它接收四个输入：

| 参数      | 含义                |
| ------- | ----------------- |
| `query` | 用来生成 Q            |
| `key`   | 用来生成 K            |
| `value` | 用来生成 V            |
| `mask`  | 掩码，可选，用来屏蔽不该关注的位置 |

---

## 8. 生成 Q、K、V 并计算注意力

```python
return scaled_dot_product(
    self.linear_query(query),
    self.linear_key(key),
    self.linear_value(value),
    mask
)
```

这段代码等价于：

```python
q = self.linear_query(query)
k = self.linear_key(key)
v = self.linear_value(value)

output = scaled_dot_product(q, k, v, mask)

return output
```

也就是说：

```text
query → linear_query → Q
key   → linear_key   → K
value → linear_value → V
```

然后送入缩放点积注意力：

```text
scaled_dot_product(Q, K, V, mask)
```

---

# 四、多头注意力代码

```python
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制的实现
    Args:
        d_model (int): 输入特征的维度
        num_heads (int): 注意力头的数量
    """
    def __init__(self, dim_input: int, num_heads: int, dim_q: int, dim_k: int):
        super().__init__()
        self.headers = nn.ModuleList([
            AttentionHead(dim_input, dim_q, dim_k) for _ in range(num_heads)
        ])
        self.linear_out = nn.Linear(num_heads * dim_k, dim_input)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        """
        前向传播
        Args:
            query (Tensor): Query tensor of shape (batch_size, seq_len_q, dim_input)
            key (Tensor): Key tensor of shape (batch_size, seq_len_k, dim_input)
            value (Tensor): Value tensor of shape (batch_size, seq_len_v, dim_input)
            mask (Tensor, optional): Mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k)

        Returns:
            Tensor: Output tensor after applying multi-head attention mechanism.
        """
        head_outputs = [head(query, key, value, mask) for head in self.headers]
        concatenated = torch.cat(head_outputs, dim=-1)
        return self.linear_out(concatenated)
```

---

# 五、多头注意力整体流程

多头注意力的执行流程是：

```text
输入 query, key, value
  ↓
创建 num_heads 个 AttentionHead
  ↓
每个 AttentionHead 独立计算一次注意力
  ↓
得到多个 head_output
  ↓
沿最后一维拼接多个 head_output
  ↓
经过 linear_out 线性层融合
  ↓
输出最终结果
```

也就是：

```text
head_1 = AttentionHead_1(query, key, value)
head_2 = AttentionHead_2(query, key, value)
...
head_n = AttentionHead_n(query, key, value)

concatenated = concat(head_1, head_2, ..., head_n)

output = linear_out(concatenated)
```

---

# 六、多头注意力逐行解释

## 1. 定义类

```python
class MultiHeadAttention(nn.Module):
```

定义一个 `MultiHeadAttention` 类，表示多头注意力模块。

它继承自 `nn.Module`，因此它也是一个可训练的神经网络模块。

---

## 2. 初始化函数

```python
def __init__(self, dim_input: int, num_heads: int, dim_q: int, dim_k: int):
```

初始化多头注意力。

参数含义如下：

| 参数          | 含义                   |
| ----------- | -------------------- |
| `dim_input` | 输入 token 的特征维度       |
| `num_heads` | 注意力头的数量              |
| `dim_q`     | 每个头中 Query 的维度       |
| `dim_k`     | 每个头中 Key 和 Value 的维度 |

例如：

```python
mha = MultiHeadAttention(
    dim_input=512,
    num_heads=8,
    dim_q=64,
    dim_k=64
)
```

表示：

```text
输入 token 是 512 维
一共有 8 个注意力头
每个头的 Query 是 64 维
每个头的 Key 是 64 维
每个头的 Value 也是 64 维
```

---

## 3. 调用父类初始化

```python
super().__init__()
```

调用 `nn.Module` 的初始化方法，让 PyTorch 可以管理这个模块中的子模块和参数。

---

## 4. 创建多个注意力头

```python
self.headers = nn.ModuleList([
    AttentionHead(dim_input, dim_q, dim_k) for _ in range(num_heads)
])
```

这行代码是多头注意力的核心。

它创建了 `num_heads` 个单头注意力。

如果：

```text
num_heads = 8
```

那么这行代码等价于创建了 8 个 `AttentionHead`：

```python
self.headers = nn.ModuleList([
    AttentionHead(dim_input, dim_q, dim_k),
    AttentionHead(dim_input, dim_q, dim_k),
    AttentionHead(dim_input, dim_q, dim_k),
    AttentionHead(dim_input, dim_q, dim_k),
    AttentionHead(dim_input, dim_q, dim_k),
    AttentionHead(dim_input, dim_q, dim_k),
    AttentionHead(dim_input, dim_q, dim_k),
    AttentionHead(dim_input, dim_q, dim_k),
])
```

每个 `AttentionHead` 都有自己独立的参数：

```text
linear_query
linear_key
linear_value
```

所以每个头都会从不同角度生成 Q、K、V。

---

## 5. 为什么用 ModuleList？

```python
nn.ModuleList([...])
```

`ModuleList` 是 PyTorch 用来保存多个子模块的容器。

不能简单使用普通 Python 列表：

```python
self.headers = [
    AttentionHead(...),
    AttentionHead(...)
]
```

因为普通列表里的模块可能无法被 PyTorch 正确注册。

使用 `ModuleList` 后，PyTorch 才能知道：

```text
这些 AttentionHead 都是模型的一部分
它们的参数都需要训练
它们都应该出现在 model.parameters() 中
```

---

## 6. 定义输出融合层

```python
self.linear_out = nn.Linear(num_heads * dim_k, dim_input)
```

这一行定义了多头注意力最后的输出线性层。

因为每个头输出的最后一维是：

```text
dim_k
```

如果有 `num_heads` 个头，拼接后最后一维就是：

```text
num_heads * dim_k
```

例如：

```text
num_heads = 8
dim_k = 64
```

那么拼接后：

```text
8 * 64 = 512
```

所以：

```python
self.linear_out = nn.Linear(512, 512)
```

如果：

```text
dim_input = 512
```

最终会把多头拼接结果映射回输入维度。

这个输出层的作用是：

```text
把多个注意力头的结果融合起来，并恢复到模型主干需要的维度。
```

---

## 7. 多头 forward 函数

```python
def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
```

这是多头注意力的前向传播函数。

输入包括：

```text
query
key
value
mask
```

其中：

```text
query: (batch_size, seq_len_q, dim_input)
key:   (batch_size, seq_len_k, dim_input)
value: (batch_size, seq_len_v, dim_input)
```

通常：

```text
seq_len_k = seq_len_v
```

因为每个 Key 通常对应一个 Value。

---

## 8. 多个头分别计算

```python
head_outputs = [head(query, key, value, mask) for head in self.headers]
```

这一行让每个注意力头都单独计算一次注意力。

等价于：

```python
head_outputs = []

for head in self.headers:
    output = head(query, key, value, mask)
    head_outputs.append(output)
```

如果 `num_heads = 8`，就会得到 8 个输出：

```text
head_outputs = [
  head_1_output,
  head_2_output,
  head_3_output,
  head_4_output,
  head_5_output,
  head_6_output,
  head_7_output,
  head_8_output
]
```

每个 head 都接收同样的 `query、key、value`，但是由于每个 head 内部的线性层参数不同，所以它们学到的关注模式可以不同。

---

## 9. 拼接多个头的输出

```python
concatenated = torch.cat(head_outputs, dim=-1)
```

这行代码把多个头的输出沿着最后一维拼接。

如果每个头输出：

```text
(batch_size, seq_len_q, dim_k)
```

一共有 `num_heads` 个头，那么拼接后：

```text
(batch_size, seq_len_q, num_heads * dim_k)
```

例如：

```text
head_1_output: (2, 5, 64)
head_2_output: (2, 5, 64)
...
head_8_output: (2, 5, 64)
```

沿最后一维拼接：

```text
concatenated: (2, 5, 512)
```

为什么是 `dim=-1`？

因为最后一维是 token 的特征维度。

多个头的输出本质上是多个不同角度的特征，所以应该拼在特征维度上。

---

## 10. 输出融合

```python
return self.linear_out(concatenated)
```

最后把拼接后的多头输出送入 `linear_out`。

假设：

```text
concatenated: (2, 5, 512)
linear_out: 512 → 512
```

那么输出：

```text
output: (2, 5, 512)
```

最终返回的形状通常是：

```text
(batch_size, seq_len_q, dim_input)
```

---

# 七、QA 1：为什么要添加线性层？

## 1. 单头注意力中为什么要添加线性层？

在单头注意力中，有三层线性层：

```python
self.linear_query = nn.Linear(dim_input, dim_q)
self.linear_key = nn.Linear(dim_input, dim_k)
self.linear_value = nn.Linear(dim_input, dim_k)
```

它们的作用是把原始输入映射成不同角色：

```text
query → Q
key   → K
value → V
```

虽然它们可能来自同一个输入，但它们在注意力机制中的功能不同：

| 向量 | 作用                |
| -- | ----------------- |
| Q  | 用来查询：我想关注谁        |
| K  | 用来匹配：我是否值得被关注     |
| V  | 用来提供信息：被关注后贡献什么内容 |

如果不加线性层，模型只能直接使用原始输入：

```text
Q = query
K = key
V = value
```

这样模型表达能力会比较弱，因为同一个输入表示要同时承担查询、匹配和提供信息三种角色。

加上线性层后，模型可以学习三套不同的映射：

```text
输入 → 查询空间
输入 → 匹配空间
输入 → 内容空间
```

这可以增强模型表达能力。

---

## 2. 多头注意力中为什么还需要 linear_out？

多头注意力中，多个 head 会输出多个结果。

例如：

```text
head_1_output: (2, 5, 64)
head_2_output: (2, 5, 64)
...
head_8_output: (2, 5, 64)
```

拼接后：

```text
concatenated: (2, 5, 512)
```

`linear_out` 的作用是：

```text
融合多个头的信息
并把维度映射回 dim_input
```

所以多头注意力中的线性层分为两类：

```text
1. 每个 head 内部的 Q/K/V 线性层
2. 多头拼接后的输出融合线性层 linear_out
```

---

# 八、QA 2：forward 函数的作用是什么？

`forward` 函数表示模块的前向传播逻辑。

在 PyTorch 中，当我们写：

```python
output = model(input)
```

实际上 PyTorch 会调用：

```python
model.forward(input)
```

对于单头注意力：

```python
def forward(self, query, key, value, mask=None):
    return scaled_dot_product(
        self.linear_query(query),
        self.linear_key(key),
        self.linear_value(value),
        mask
    )
```

它定义的是：

```text
如何从 query/key/value 计算出一个注意力头的输出。
```

对于多头注意力：

```python
def forward(self, query, key, value, mask=None):
    head_outputs = [head(query, key, value, mask) for head in self.headers]
    concatenated = torch.cat(head_outputs, dim=-1)
    return self.linear_out(concatenated)
```

它定义的是：

```text
如何调用多个注意力头
如何拼接多个头的输出
如何通过 linear_out 融合最终结果
```

所以一句话理解：

```text
__init__ 负责定义有哪些层；
forward 负责定义数据如何流过这些层。
```

---

# 九、QA 3：多头注意力的“多头”体现在哪里？

多头体现在这一行：

```python
self.headers = nn.ModuleList([
    AttentionHead(dim_input, dim_q, dim_k) for _ in range(num_heads)
])
```

这里创建了多个 `AttentionHead`。

如果：

```text
num_heads = 8
```

就会有 8 个独立的注意力头。

每个头内部都有自己的参数：

```text
head_1: linear_query_1, linear_key_1, linear_value_1
head_2: linear_query_2, linear_key_2, linear_value_2
...
head_8: linear_query_8, linear_key_8, linear_value_8
```

所以它们不是完全相同的计算，而是不同参数下的注意力计算。

虽然每个头都看同样的输入：

```text
query, key, value
```

但由于参数不同，每个头可以学到不同的关注方式。

例如：

```text
head 1 关注局部相邻词
head 2 关注主谓关系
head 3 关注指代关系
head 4 关注实体关系
...
```

这些关注模式不是人工设定的，而是模型训练过程中自动学习出来的。

---

# 十、QA 4：多头注意力相对于单头，额外的流程有哪些？

单头注意力流程是：

```text
query/key/value
  ↓
生成 Q/K/V
  ↓
scaled_dot_product
  ↓
输出一个 head 的结果
```

多头注意力额外多了三步：

```text
1. 创建多个 AttentionHead
2. 每个 AttentionHead 独立计算注意力
3. 把多个 head 输出拼接后，再通过 linear_out 融合
```

也就是说：

```text
单头：
一个 head 看输入，输出一个结果

多头：
多个 head 同时看输入，输出多个结果
然后 concat 拼接
最后 linear_out 融合
```

代码对应关系：

```python
head_outputs = [head(query, key, value, mask) for head in self.headers]
```

对应：

```text
多个 head 分别计算
```

```python
concatenated = torch.cat(head_outputs, dim=-1)
```

对应：

```text
拼接多个 head 的输出
```

```python
return self.linear_out(concatenated)
```

对应：

```text
融合多头结果
```

---

# 十一、QA 5：整体形状是如何变化的？

下面用一个具体例子说明。

假设：

```text
batch_size = 2
seq_len_q = 5
seq_len_k = 7
seq_len_v = 7
dim_input = 512
num_heads = 8
dim_q = 64
dim_k = 64
```

输入形状：

```text
query: (2, 5, 512)
key:   (2, 7, 512)
value: (2, 7, 512)
```

---

## 1. 单头注意力内部形状变化

### 第一步：生成 Q

```python
q = self.linear_query(query)
```

形状变化：

```text
query: (2, 5, 512)
q:     (2, 5, 64)
```

---

### 第二步：生成 K

```python
k = self.linear_key(key)
```

形状变化：

```text
key: (2, 7, 512)
k:   (2, 7, 64)
```

---

### 第三步：生成 V

```python
v = self.linear_value(value)
```

形状变化：

```text
value: (2, 7, 512)
v:     (2, 7, 64)
```

---

### 第四步：计算 QKᵀ

在 `scaled_dot_product` 中：

```python
matmul_qk = torch.matmul(q, k.transpose(-2, -1))
```

先转置 K：

```text
k:  (2, 7, 64)
kᵀ: (2, 64, 7)
```

然后矩阵乘法：

```text
q:  (2, 5, 64)
kᵀ: (2, 64, 7)
```

结果：

```text
QKᵀ: (2, 5, 7)
```

含义是：

```text
每个 query token 对 7 个 key token 都计算一个注意力分数。
```

---

### 第五步：softmax

```python
attention_weights = softmax(scaled_attention_logits, dim=-1)
```

形状不变：

```text
attention_weights: (2, 5, 7)
```

最后一维 `7` 表示对所有 key token 分配注意力权重。

---

### 第六步：注意力权重乘 V

```python
output = torch.matmul(attention_weights, v)
```

形状：

```text
attention_weights: (2, 5, 7)
v:                 (2, 7, 64)
```

矩阵乘法：

```text
(5, 7) @ (7, 64) = (5, 64)
```

结果：

```text
head_output: (2, 5, 64)
```

所以单个注意力头输出：

```text
(batch_size, seq_len_q, dim_k)
```

---

## 2. 多头注意力形状变化

如果有 8 个头，每个头都输出：

```text
head_output: (2, 5, 64)
```

那么：

```python
head_outputs = [head(query, key, value, mask) for head in self.headers]
```

得到：

```text
head_outputs = [
  (2, 5, 64),
  (2, 5, 64),
  (2, 5, 64),
  (2, 5, 64),
  (2, 5, 64),
  (2, 5, 64),
  (2, 5, 64),
  (2, 5, 64)
]
```

然后拼接：

```python
concatenated = torch.cat(head_outputs, dim=-1)
```

沿最后一维拼接：

```text
(2, 5, 64) × 8 → (2, 5, 512)
```

得到：

```text
concatenated: (2, 5, 512)
```

最后经过输出层：

```python
output = self.linear_out(concatenated)
```

因为：

```python
self.linear_out = nn.Linear(num_heads * dim_k, dim_input)
```

所以：

```text
512 → 512
```

最终输出：

```text
output: (2, 5, 512)
```

---

# 十二、整体形状变化总结表

## 单头注意力

| 步骤        | 张量                      | 形状            |
| --------- | ----------------------- | ------------- |
| 输入 query  | `query`                 | `(2, 5, 512)` |
| 输入 key    | `key`                   | `(2, 7, 512)` |
| 输入 value  | `value`                 | `(2, 7, 512)` |
| Query 线性层 | `linear_query(query)`   | `(2, 5, 64)`  |
| Key 线性层   | `linear_key(key)`       | `(2, 7, 64)`  |
| Value 线性层 | `linear_value(value)`   | `(2, 7, 64)`  |
| K 转置      | `k.transpose(-2, -1)`   | `(2, 64, 7)`  |
| 注意力分数     | `q @ kᵀ`                | `(2, 5, 7)`   |
| softmax   | `attention_weights`     | `(2, 5, 7)`   |
| 乘 V       | `attention_weights @ v` | `(2, 5, 64)`  |

---

## 多头注意力

| 步骤            | 张量                         | 形状               |
| ------------- | -------------------------- | ---------------- |
| 每个 head 输出    | `head_output`              | `(2, 5, 64)`     |
| 8 个 head 输出列表 | `head_outputs`             | `8 × (2, 5, 64)` |
| 拼接多个 head     | `torch.cat(..., dim=-1)`   | `(2, 5, 512)`    |
| 输出线性层         | `linear_out(concatenated)` | `(2, 5, 512)`    |

---

# 十三、需要注意的代码细节

## 1. `headers` 命名问题

代码里写的是：

```python
self.headers
```

更常见、更准确的命名是：

```python
self.heads
```

因为这里表示的是多个 attention heads，而不是 headers。

---

## 2. `dim_q` 和 `dim_k` 最好相等

因为单头注意力内部要计算：

```text
QKᵀ
```

如果：

```text
Q: (batch_size, seq_len_q, dim_q)
K: (batch_size, seq_len_k, dim_k)
```

那么：

```text
Kᵀ: (batch_size, dim_k, seq_len_k)
```

矩阵乘法要求中间维度相等：

```text
dim_q 必须等于 dim_k
```

否则：

```text
Q @ Kᵀ
```

无法计算。

---

## 3. Value 的维度最好单独命名

当前代码中：

```python
self.linear_value = nn.Linear(dim_input, dim_k)
```

说明 Value 的输出维度也使用 `dim_k`。

这不是错误，但不够灵活。

更标准的写法可以是：

```python
self.linear_value = nn.Linear(dim_input, dim_v)
```

对应多头输出层也要改成：

```python
self.linear_out = nn.Linear(num_heads * dim_v, dim_input)
```

---

## 4. mask 形状要和 scaled_dot_product 适配

当前单头注意力中，传入 `scaled_dot_product` 的 Q/K/V 是三维：

```text
q: (batch_size, seq_len_q, dim_q)
k: (batch_size, seq_len_k, dim_k)
v: (batch_size, seq_len_v, dim_k)
```

因此注意力分数通常是：

```text
(batch_size, seq_len_q, seq_len_k)
```

如果你的 `scaled_dot_product` 是按照四维多头格式写的：

```text
(batch_size, num_heads, seq_len_q, seq_len_k)
```

那么就需要在单头里加一个 head 维度：

```python
q = self.linear_query(query).unsqueeze(1)
k = self.linear_key(key).unsqueeze(1)
v = self.linear_value(value).unsqueeze(1)

output = scaled_dot_product(q, k, v, mask)

return output.squeeze(1)
```

---

# 十四、最终总结

单头注意力：

```text
用一组 Q/K/V 投影参数，从一个角度计算注意力。
```

多头注意力：

```text
用多组 Q/K/V 投影参数，从多个角度分别计算注意力，然后拼接融合。
```

最核心区别是：

```text
单头：
一个 AttentionHead

多头：
多个 AttentionHead + concat + linear_out
```

代码核心对应：

```python
head_outputs = [head(query, key, value, mask) for head in self.headers]
concatenated = torch.cat(head_outputs, dim=-1)
return self.linear_out(concatenated)
```

一句话理解：

> 单头注意力是一个专家看问题；多头注意力是多个专家分别看问题，最后把多个专家的判断合并成一个结果。
