
# 2_pytorch_basics

## 1. PyTorch训练整体逻辑

深度学习训练本质：

```text
输入数据
→ 模型预测
→ 计算loss
→ 计算梯度
→ 更新参数
```

对应代码：

```python
output = model(x)

loss = criterion(output, target)

loss.backward()

optimizer.step()
```

---

# 2. nn.Module、**init** 和 forward

定义模型：

```python
class MyModel(nn.Module):
```

表示创建一个继承自 `nn.Module` 的神经网络类。

`nn.Module` 主要负责：

- 组织神经网络的各个层和子模块
- 自动注册并管理模型参数，例如 `weight` 和 `bias`
- 提供 `model.parameters()` 获取需要训练的参数
- 提供 `model.train()` 和 `model.eval()` 切换训练/评估状态
- 提供 `model(x)` 的调用机制，最终执行 `forward(x)`

注意：

`nn.Module` 本身不是负责自动求导的核心机制。

PyTorch 的自动求导主要由 **autograd（automatic differentiation，自动微分）机制**完成，Tensor 是参与这一机制的核心对象。

`loss.backward()` 会沿着计算图反向计算梯度，并把需要求梯度的叶子 Tensor 的梯度累积到 `.grad` 中；对模型参数来说，就是例如 `weight.grad` 和 `bias.grad`。

## `__init__`

负责定义模型结构，即“模型里面有什么”。

例如：

```python
def __init__(self):
    super().__init__()
    self.linear = nn.Linear(10,1)
```

表示模型包含一个Linear层。

`Linear`内部包含需要学习的参数：

- weight
    
- bias
    

---

## `forward()`

负责定义数据如何经过模型，即“数据怎么计算”。

例如：

```python
def forward(self,x):
    return self.linear(x)
```

表示输入 `x` 经过Linear层得到输出。

注意：

不要在 `forward()` 中创建新的层：

```python
def forward(self,x):
    linear = nn.Linear(10,1)
```

因为每次调用都会重新创建参数，模型无法学习。

正确逻辑：

```text
__init__
创建模型参数

↓

forward
使用模型参数计算
```

---

# 3. model(x)、**call** 和 forward

代码：

```python
output = model(x)
```

不是直接调用 `forward()`。

实际流程：

```text
model(x)
→ model.__call__(x)
→ forward(x)
→ output
```

原因：

`nn.Module`实现了 `__call__()`，内部会自动调用用户定义的 `forward()`。

所以 PyTorch推荐：

```python
model(x)
```

而不是：

```python
model.forward(x)
```

---

# 4. Tensor

Tensor 是 PyTorch中的核心数据结构，类似 NumPy 数组。

生信中：

```text
AnnData:

adata.X

cells × genes
```

进入 PyTorch 后通常转换为：

```text
Tensor:

batch × genes
```

作为模型输入。

---

# 5. Dataset 和 DataLoader

## Dataset

负责定义：

> 一个样本在哪里，以及如何取出。

核心方法：

```python
__len__()
```

定义数据数量。

```python
__getitem__(index)
```

定义如何获取第 `index` 个样本。

---

## DataLoader

负责将 Dataset 中的样本组织成 batch。

例如：

```python
loader = DataLoader(
    dataset,
    batch_size=32
)
```

表示每次取32个样本输入模型。

关系：

```text
Dataset
提供单个样本

↓

DataLoader
组织batch

↓

模型输入
```

---

# 6. Loss（损失）

创建损失函数对象：

```python
criterion = nn.MSELoss()
```

含义：

创建一个 MSELoss 实例，不是计算loss。

计算：

```python
loss = criterion(prediction, target)
```

过程：

```text
criterion()
→ __call__()
→ forward()
→ 计算误差
→ loss
```

loss表示：

模型预测结果和真实值之间的差距。

---

# 7. Weight 和 Bias

Linear层：

[  
y=wx+b  
]

其中：

- weight：权重，表示输入特征的重要程度。
    
- bias：偏置，用于调整整体偏移。
    

二者都是模型需要学习的参数。

---

# 8. Gradient（梯度）

grad = gradient（梯度）。

作用：

告诉模型：

> 参数应该向哪个方向调整。

代码：

```python
loss.backward()
```

作用：

计算梯度，不直接更新参数。

得到：

```python
weight.grad
bias.grad
```

---

# 9. Optimizer（优化器）

创建：

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)
```

含义：

创建一个SGD优化器，让它负责更新模型参数。

---

## model.parameters()

这是模型的方法。

作用：

返回模型中所有需要训练的参数。

例如：

```text
model
 ↓
linear
 ↓
weight
bias
```

返回：

```text
weight
bias
```

交给optimizer管理。

---

## lr

learning rate（学习率）。

控制每次参数更新的幅度。

---

# 10. 完整训练循环

```python
for epoch in range(10):

    for batch_x, batch_y in loader:

        prediction = model(batch_x)

        loss = criterion(
            prediction,
            batch_y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
```

执行逻辑：

1. DataLoader提供batch。
    
2. `model(batch_x)`进行前向计算，得到prediction。
    
3. `criterion(prediction, batch_y)`计算loss。
    
4. `loss.backward()`计算梯度。
    
5. `optimizer.step()`根据梯度更新weight和bias。
    

---

# Final Mental Model

```text
Dataset
↓
DataLoader
↓
Tensor batch
↓
model(batch)
↓
forward()
↓
prediction
↓
criterion()
↓
loss
↓
backward()
↓
gradient
↓
optimizer.step()
↓
更新weight/bias
```

# 11. Embedding（嵌入层）

## 11.1 embedding 的含义

`embedding`

中文：嵌入

在深度学习中通常指：

> 将离散的对象转换成连续的向量表示。

神经网络不能直接理解文字、基因等离散对象，因此需要先转换成数字表示。

例如：

```text
cat

↓

token ID = 5

↓

Embedding

↓

[0.2, 0.5, -0.3, ...]
````

其中：

```text
token ID
```

只是编号：

```text
cat → 5
dog → 8
```

数字本身没有大小关系。

Embedding 的作用是：

> 学习一个能够表示对象特征的向量。

---

# 11.2 nn.Embedding()

PyTorch 提供：

```python
nn.Embedding(num_embeddings, embedding_dim)
```

例如：

```python
embedding = nn.Embedding(10000, 200)
```

表示：

```text
10000
=
共有10000个离散对象(token)

200
=
每个对象用200维向量表示
```

内部可以理解为一个可学习矩阵：

```text
10000 × 200
```

其中：

- 每一行对应一个 token
    
- 每一行是该 token 的 embedding vector
    

例如：

```text
Embedding Matrix

        dim1   dim2   dim3

cat     0.2    0.5    0.8

dog     0.1    0.7    0.4

fish    0.9    0.3    0.2
```

输入：

```python
token_id = 0
```

实际上就是：

> 查找 Embedding Matrix 第 0 行。

输出：

```text
[0.2,0.5,0.8]
```

---

# 11.3 embedding 在生物信息学中的对应

NLP 中：

```text
word

↓

word embedding
```

生物数据中：

```text
gene

↓

gene embedding
```

例如：

```text
CD3D

↓

gene ID

↓

gene embedding
```

因此：

scGPT、Geneformer 等模型中，也会使用类似：

```text
gene token
↓

embedding

↓

Transformer
```

的流程。

---

# 12. Tensor shape 思维

阅读深度学习代码时，需要持续关注：

> Tensor 的 shape 如何变化。

因为不同模型中，每个维度代表不同含义。

# 12.1 CNN 中的 shape

图像输入通常：

```text
(batch, channel, height, width)
```

例如：

```python
x.shape = [32, 1, 28, 28]
```

表示：

```text
32
=
batch size

1
=
channel

28
=
height

28
=
width
```

CNN 中，阅读代码时需要关注：

```text
输入shape

↓

经过卷积/池化

↓

输出shape
```

如何变化。

# 12.2 序列模型中的 shape

文本、基因序列等数据通常包含：

```text
sequence length
batch size
embedding dimension
```

例如：

token ID：

```text
[35, 20]
```

表示：

```text
35
=
sequence length

20
=
batch size
```

经过 Embedding：

```text
[35,20,200]
```

表示：

```text
35
=
sequence length

20
=
batch size

200
=
embedding dimension
```

即：

```text
token ID Tensor

↓

Embedding

↓

(sequence length, batch size, embedding dimension)
```

不同项目也可能使用：

```text
(batch size, sequence length, embedding dimension)
```

因此不能只记固定顺序，需要结合代码判断每个维度的意义。

---

# 12.3 为什么 shape 思维重要？

例如：

```python
x = self.embedding(input)
```

不能只知道：

> 做了 embedding。

还需要知道：

输入：

```text
什么shape？

↓

输出：

什么shape？
```

因为下一层：

```python
self.rnn(x)
```

或者：

```python
self.transformer(x)
```

需要匹配特定输入格式。

---

# 13. Dropout（随机失活）

## 13.1 Dropout 的作用

Dropout：是一种正则化方法，用于减少模型过拟合。

训练过程中：

随机将部分神经元输出设置为 0。

例如：

原始向量：

```text
[0.2, 0.5, 0.8, 0.1]
```

经过 Dropout：

```text
[0.2, 0, 0.8, 0]
```

---

# 13.2 为什么需要 Dropout？

如果模型长期依赖某些特征：

例如：

```text
某几个神经元

↓

预测结果
```

模型可能学习到过于简单的规律。

Dropout 会随机关闭部分神经元：

```text
训练过程中：

这次不用这个特征

下一次不用另一个特征
```

迫使模型学习更加稳定的表示。

---

# 13.3 nn.Dropout()

PyTorch：

```python
self.dropout = nn.Dropout(p=0.5)
```

其中：

```text
p
=
dropout probability
=
丢弃概率
```

例如：

```python
p=0.5
```

表示：

训练时约一半神经元会被随机关闭。

注意：

Dropout 只在训练模式启用：

```python
model.train()
```

启用。

推理时：

```python
model.eval()
```

关闭。

---

# 14. 从真实代码理解数据流

例如：

```python
emb = self.drop(self.encoder(input))
```

执行顺序：

从内向外：

```text
self.encoder(input)

↓

self.drop()

↓

emb
```

---

第一步：

```python
self.encoder(input)
```

如果：

```python
self.encoder = nn.Embedding(...)
```

则：

```text
token ID

↓

Embedding

↓

embedding vector
```

第二步：

```python
self.drop()
```

对 embedding 结果进行 Dropout。

最终：

```text
input

(token ID)

↓

Embedding

↓

embedding vector

↓

Dropout

↓

emb
```


持续更新中...
