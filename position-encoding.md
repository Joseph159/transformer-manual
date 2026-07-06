# Transformer 位置编码阅读文档：超小白逐行版

## 0. 先说人话：位置编码到底是干什么的？

Transformer 里面的注意力机制很强，但是它有一个问题：

```text
它只看一堆 token，不天然知道谁在前、谁在后。
```

比如这句话：

```text
我 喜欢 学习 AI
```

对人来说，顺序很明显：

```text
我      在第 0 个位置
喜欢    在第 1 个位置
学习    在第 2 个位置
AI      在第 3 个位置
```

但是 Transformer 的注意力计算本身更像是在看一堆向量。它需要额外知道：

```text
这个 token 在第几个位置。
```

所以就需要位置编码：

```text
token embedding：告诉模型“这个词是什么意思”
position encoding：告诉模型“这个词在哪里”
```

最后会相加：

```python
x = token_embedding + position_encoding
```

也就是：

```text
输入 = 词义信息 + 位置信息
```

---

## 1. 代码原文

```python
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
```

---

## 2. 这段代码最终生成什么？

假设：

```python
seq_len = 3
d_model = 4
```

含义是：

```text
seq_len = 3：一句话最多有 3 个 token
d_model = 4：每个 token 的向量是 4 维
```

最终生成的位置编码形状是：

```text
(1, 3, 4)
```

含义：

```text
1：batch 维度，方便后面和多个样本相加
3：有 3 个位置
4：每个位置有 4 维位置编码
```

---

## 3. 最重要的理解：两个维度组成一个“小位置编码单元”

位置编码不是只有：

```text
[sin, cos]
```

而是：

```text
[sin, cos, sin, cos, sin, cos, ...]
```

也就是：

```text
每两个维度是一对 sin/cos。
```

例如：

```text
d_model = 4
```

那么位置编码有 4 维：

```text
[第0维, 第1维, 第2维, 第3维]
```

按照 sin/cos 分组：

```text
第 0、1 维是一组：
[sin, cos]

第 2、3 维是一组：
[sin, cos]
```

所以：

```text
d_model = 4   → 2 对 sin/cos
d_model = 8   → 4 对 sin/cos
d_model = 512 → 256 对 sin/cos
```

### 为什么两个维度才算一组？

因为单独一个 sin 不够完整。

比如：

```text
sin(0°) = 0
sin(180°) = 0
```

只看 sin，0° 和 180° 都是 0，分不清。

但是如果同时看 sin 和 cos：

```text
0°:
[sin(0°), cos(0°)] = [0, 1]

180°:
[sin(180°), cos(180°)] = [0, -1]
```

就能区分了。

所以你可以这样理解：

```text
两个维度组成一个小钟表：
sin 是一个方向
cos 是另一个方向
合起来表示这个小钟表的指针位置
```

完整位置编码就是很多个不同速度的小钟表拼起来。

---

## 4. 逐行解释

### 4.1 定义函数

```python
def position_encoding(seq_len: int, d_model: int) -> Tensor:
```

定义一个函数，名字叫：

```text
position_encoding
```

它有两个参数：

| 参数 | 含义 |
|---|---|
| `seq_len` | 序列长度，也就是最多有几个 token |
| `d_model` | 每个 token 向量的维度 |

比如：

```python
position_encoding(seq_len=3, d_model=4)
```

意思是：

```text
给 3 个位置生成位置编码；
每个位置编码是 4 维。
```

---

### 4.2 定义内部函数 get_angles

```python
def get_angles(pos, i, d_model):
```

这个内部函数用于生成“角度矩阵”。

注意：

```text
get_angles 还没有真正做 sin/cos。
```

它只是先计算：

```text
每个位置、每个维度，对应的角度是多少。
```

后面才会把这些角度丢进 sin 或 cos。

---

### 4.3 计算 angle_rates

```python
angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
```

这行最难，但可以拆开看。

它的作用是：

```text
给每一维分配一个频率。
```

频率可以理解为：

```text
位置变化时，这一维变化得快还是慢。
```

靠前的维度变化快，靠后的维度变化慢。

---

## 5. 重点拆解 angle_rates

我们用一个具体例子：

```python
d_model = 4
```

那么维度编号是：

```text
i = [0, 1, 2, 3]
```

### 5.1 第一步：i // 2

```python
i // 2
```

`//` 是整除。

计算：

```text
0 // 2 = 0
1 // 2 = 0
2 // 2 = 1
3 // 2 = 1
```

所以：

```text
i       = [0, 1, 2, 3]
i // 2  = [0, 0, 1, 1]
```

这一行的作用：

```text
把相邻两个维度分成一组。
```

也就是：

```text
第 0、1 维是一组
第 2、3 维是一组
```

---

### 5.2 第二步：2 * (i // 2)

```python
2 * (i // 2)
```

刚才：

```text
i // 2 = [0, 0, 1, 1]
```

乘以 2：

```text
2 * (i // 2) = [0, 0, 2, 2]
```

这一行的作用：

```text
让一对 sin/cos 使用同一个频率。
```

也就是说：

```text
第 0、1 维共用一个频率
第 2、3 维共用一个频率
```

这就是前面说的：

```text
两个维度组成一个小位置编码单元。
```

---

### 5.3 第三步：除以 d_model

```python
(2 * (i // 2)) / np.float32(d_model)
```

现在：

```text
2 * (i // 2) = [0, 0, 2, 2]
d_model = 4
```

所以：

```text
[0, 0, 2, 2] / 4 = [0, 0, 0.5, 0.5]
```

这里的：

```python
np.float32(d_model)
```

只是把 `d_model` 转成小数类型，避免整数计算。

---

### 5.4 第四步：10000 的这些次方

```python
np.power(10000, [0, 0, 0.5, 0.5])
```

就是：

```text
10000^0
10000^0
10000^0.5
10000^0.5
```

计算：

```text
10000^0 = 1
10000^0.5 = 100
```

所以得到：

```text
[1, 1, 100, 100]
```

---

### 5.5 第五步：取倒数

```python
angle_rates = 1 / [1, 1, 100, 100]
```

得到：

```text
angle_rates = [1, 1, 0.01, 0.01]
```

这就是每一维对应的频率。

解释一下：

```text
第 0、1 维：频率 1，变化快
第 2、3 维：频率 0.01，变化慢
```

所以：

```text
第一对 sin/cos 是快钟表
第二对 sin/cos 是慢钟表
```

---

## 6. return pos * angle_rates 是什么？

```python
return pos * angle_rates
```

这一步是：

```text
位置 pos × 频率
```

得到：

```text
角度 angle
```

也就是：

```text
angle = pos * angle_rate
```

如果：

```text
位置 pos = 0, 1, 2
频率 angle_rates = [1, 1, 0.01, 0.01]
```

那么：

```text
pos = 0:
[0×1, 0×1, 0×0.01, 0×0.01]
= [0, 0, 0, 0]

pos = 1:
[1×1, 1×1, 1×0.01, 1×0.01]
= [1, 1, 0.01, 0.01]

pos = 2:
[2×1, 2×1, 2×0.01, 2×0.01]
= [2, 2, 0.02, 0.02]
```

这些数是角度，还不是最终位置编码。

---

## 7. 生成位置索引和维度索引

代码：

```python
angle_rads = get_angles(np.arange(seq_len)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
```

这行是在调用 `get_angles`。

### 7.1 np.arange(seq_len)

```python
np.arange(seq_len)
```

假设：

```python
seq_len = 3
```

那么：

```python
np.arange(3)
```

结果是：

```text
[0, 1, 2]
```

表示 3 个位置：

```text
位置 0
位置 1
位置 2
```

### 7.2 [:, np.newaxis]

```python
np.arange(seq_len)[:, np.newaxis]
```

原来：

```text
[0, 1, 2]
```

形状是：

```text
(3,)
```

加上 `[:, np.newaxis]` 之后，变成列向量：

```text
[
  [0],
  [1],
  [2]
]
```

形状变成：

```text
(3, 1)
```

### 7.3 np.arange(d_model)

```python
np.arange(d_model)
```

假设：

```python
d_model = 4
```

那么：

```text
[0, 1, 2, 3]
```

表示 4 个维度。

### 7.4 [np.newaxis, :]

```python
np.arange(d_model)[np.newaxis, :]
```

原来：

```text
[0, 1, 2, 3]
```

加上 `[np.newaxis, :]` 之后，变成行向量：

```text
[
  [0, 1, 2, 3]
]
```

形状变成：

```text
(1, 4)
```

### 7.5 为什么一个是列向量，一个是行向量？

因为这样可以利用广播机制：

```text
(3, 1) 和 (1, 4)
```

组合成：

```text
(3, 4)
```

也就是：

```text
3 个位置 × 4 个维度
```

这样就能一次性算出：

```text
每个位置在每个维度上的角度。
```

---

## 8. 完整手算示例：seq_len = 3, d_model = 4

现在完整算一遍。

### 8.1 输入参数

```python
seq_len = 3
d_model = 4
```

含义：

```text
3 个位置
每个位置 4 维
```

### 8.2 位置 pos

```text
pos =
[
  [0],
  [1],
  [2]
]
```

形状：

```text
(3, 1)
```

### 8.3 维度 i

```text
i =
[
  [0, 1, 2, 3]
]
```

形状：

```text
(1, 4)
```

### 8.4 计算频率 angle_rates

先算：

```text
i // 2 = [0, 0, 1, 1]
```

再算：

```text
2 * (i // 2) = [0, 0, 2, 2]
```

再除以 `d_model = 4`：

```text
[0, 0, 2, 2] / 4 = [0, 0, 0.5, 0.5]
```

再算：

```text
10000^[0, 0, 0.5, 0.5] = [1, 1, 100, 100]
```

取倒数：

```text
angle_rates = [1, 1, 0.01, 0.01]
```

### 8.5 计算角度矩阵 angle_rads

```text
angle_rads = pos * angle_rates
```

也就是：

```text
[
  [0],
  [1],
  [2]
]
*
[
  [1, 1, 0.01, 0.01]
]
```

广播后得到：

```text
[
  [0×1, 0×1, 0×0.01, 0×0.01],
  [1×1, 1×1, 1×0.01, 1×0.01],
  [2×1, 2×1, 2×0.01, 2×0.01]
]
```

也就是：

```text
angle_rads =
[
  [0, 0, 0,    0],
  [1, 1, 0.01, 0.01],
  [2, 2, 0.02, 0.02]
]
```

注意：

```text
这里还是角度，不是最终位置编码。
```

---

## 9. 偶数维用 sin

代码：

```python
angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
```

这里的：

```python
0::2
```

表示：

```text
从第 0 维开始，每隔 2 个取一个。
```

如果维度是：

```text
0, 1, 2, 3
```

那么 `0::2` 取的是：

```text
0, 2
```

所以：

```text
第 0 维用 sin
第 2 维用 sin
```

原来的角度矩阵：

```text
[
  [0, 0, 0,    0],
  [1, 1, 0.01, 0.01],
  [2, 2, 0.02, 0.02]
]
```

第 0、2 维取出来：

```text
[
  [0, 0],
  [1, 0.01],
  [2, 0.02]
]
```

做 sin：

```text
sin(0)    = 0
sin(1)    ≈ 0.84147
sin(2)    ≈ 0.90930
sin(0.01) ≈ 0.01000
sin(0.02) ≈ 0.02000
```

所以：

```text
第 0 维：
pos0: sin(0) = 0
pos1: sin(1) ≈ 0.84147
pos2: sin(2) ≈ 0.90930

第 2 维：
pos0: sin(0) = 0
pos1: sin(0.01) ≈ 0.01000
pos2: sin(0.02) ≈ 0.02000
```

---

## 10. 奇数维用 cos

代码：

```python
angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
```

这里的：

```python
1::2
```

表示：

```text
从第 1 维开始，每隔 2 个取一个。
```

如果维度是：

```text
0, 1, 2, 3
```

那么 `1::2` 取的是：

```text
1, 3
```

所以：

```text
第 1 维用 cos
第 3 维用 cos
```

原来的角度矩阵中，第 1、3 维是：

```text
[
  [0, 0],
  [1, 0.01],
  [2, 0.02]
]
```

做 cos：

```text
cos(0)    = 1
cos(1)    ≈ 0.54030
cos(2)    ≈ -0.41615
cos(0.01) ≈ 0.99995
cos(0.02) ≈ 0.99980
```

所以：

```text
第 1 维：
pos0: cos(0) = 1
pos1: cos(1) ≈ 0.54030
pos2: cos(2) ≈ -0.41615

第 3 维：
pos0: cos(0) = 1
pos1: cos(0.01) ≈ 0.99995
pos2: cos(0.02) ≈ 0.99980
```

---

## 11. 得到最终二维位置编码矩阵

最终：

```text
angle_rads =
[
  [sin(0), cos(0), sin(0),    cos(0)],
  [sin(1), cos(1), sin(0.01), cos(0.01)],
  [sin(2), cos(2), sin(0.02), cos(0.02)]
]
```

代入近似值：

```text
angle_rads =
[
  [0.00000,  1.00000, 0.00000, 1.00000],
  [0.84147,  0.54030, 0.01000, 0.99995],
  [0.90930, -0.41615, 0.02000, 0.99980]
]
```

形状：

```text
(3, 4)
```

含义：

```text
3 个位置，每个位置 4 维位置编码。
```

---

## 12. 增加 batch 维度

代码：

```python
pos_encoding = angle_rads[np.newaxis, ...]
```

原来的形状是：

```text
(3, 4)
```

加上 `np.newaxis` 后变成：

```text
(1, 3, 4)
```

内容变成：

```text
[
  [
    [0.00000,  1.00000, 0.00000, 1.00000],
    [0.84147,  0.54030, 0.01000, 0.99995],
    [0.90930, -0.41615, 0.02000, 0.99980]
  ]
]
```

为什么要加这个 `1`？

因为输入 embedding 通常是：

```text
(batch_size, seq_len, d_model)
```

比如：

```text
(32, 3, 4)
```

而位置编码是：

```text
(1, 3, 4)
```

它可以自动广播：

```text
(32, 3, 4) + (1, 3, 4) = (32, 3, 4)
```

也就是：

```text
同一套位置编码，加到 batch 里的每一个样本上。
```

---

## 13. 转成 PyTorch Tensor

代码：

```python
return torch.tensor(pos_encoding, dtype=torch.float32)
```

这一步把 NumPy 数组转换成 PyTorch Tensor。

并且指定类型：

```text
torch.float32
```

最终返回：

```text
shape = (1, 3, 4)
```

---

## 14. 最终结果汇总

调用：

```python
position_encoding(seq_len=3, d_model=4)
```

最终大约返回：

```python
tensor([
  [
    [0.00000,  1.00000, 0.00000, 1.00000],
    [0.84147,  0.54030, 0.01000, 0.99995],
    [0.90930, -0.41615, 0.02000, 0.99980]
  ]
])
```

形状：

```text
(1, 3, 4)
```

---

## 15. 最简单记忆版

```text
位置编码 = 很多个小钟表

每两个维度 = 一个小钟表

第 0、1 维：快钟表
第 2、3 维：慢钟表
第 4、5 维：更慢的钟表
...

sin 和 cos 合起来表示一个钟表指针的位置。

多个钟表一起看，就能知道 token 在序列中的位置。
```

---

## 16. 常见问题

### Q1：为什么不是只有一个 sin 和一个 cos？

因为 token embedding 通常不是 2 维，而是很多维。

比如：

```text
d_model = 512
```

那么位置编码也必须是 512 维，才能和 embedding 相加。

所以：

```text
512 维位置编码 = 256 对 sin/cos
```

---

### Q2：为什么两个维度组成一个小位置编码单元？

因为单独一个 sin 会丢信息。

例如：

```text
sin(0°) = 0
sin(180°) = 0
```

只看 sin 分不清。

但看一对 sin/cos：

```text
0°    → [0, 1]
180°  → [0, -1]
```

就能区分。

---

### Q3：为什么不同维度频率不同？

因为要同时表达近距离和远距离。

```text
高频维度：变化快，适合区分近距离
低频维度：变化慢，适合表达长距离
```

类似：

```text
秒针：看短时间变化
分针：看中等时间变化
时针：看长时间变化
```

---

### Q4：位置编码最后为什么是 `(1, seq_len, d_model)`？

因为要方便和输入相加。

输入 embedding 通常是：

```text
(batch_size, seq_len, d_model)
```

位置编码是：

```text
(1, seq_len, d_model)
```

相加时自动广播：

```text
(batch_size, seq_len, d_model) + (1, seq_len, d_model)
= (batch_size, seq_len, d_model)
```

---

## 17. 最终总结

这段函数的核心不是生成词向量，而是生成位置向量。

```text
词向量告诉模型：这个 token 是什么。
位置编码告诉模型：这个 token 在哪里。
```

最终：

```text
输入向量 = 词向量 + 位置编码
```

而位置编码内部：

```text
每两个维度是一对 sin/cos；
每一对像一个小钟表；
不同小钟表转速不同；
所有小钟表拼起来表示完整位置。
```
