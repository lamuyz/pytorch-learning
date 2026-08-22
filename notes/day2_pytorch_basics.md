
# Day2_pytorch_basics

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
