# Day3_mnist_cnn_training_flow

## 1. Overall execution flow

Dataset
→ DataLoader
→ batch(data, target)
→ model(data)
→ forward()
→ output
→ loss
→ backward()
→ optimizer.step()
→ test()
→ scheduler.step()
→ next epoch

---

## 2. Dataset and DataLoader

```python
dataset1 = datasets.MNIST(
    '../data',
    train=True,
    download=True,
    transform=transform
)
````

- `Dataset`：定义单个样本如何被读取。
    
- MNIST 一个样本返回：
    
    - image
        
    - label / target
        
- `../data`：MNIST 数据所在目录，`..` 表示当前目录的上一级。
    

```python
train_loader = torch.utils.data.DataLoader(
    dataset1,
    **train_kwargs
)
```

- `DataLoader`：把 Dataset 中的单个样本组织成 batch。
    
- 默认 `batch_size = 64`。
    

一个训练 batch：

```text
data.shape   = [64, 1, 28, 28]
target.shape = [64]
```

含义：

```text
data:
64张灰度图片
每张图片 shape = [1, 28, 28]

target:
64张图片对应的64个真实标签
```

注意：

```text
target.shape = [64]
```

不是 `target = 64`。

对于分类：

```text
data    [batch, channel, height, width]
output  [batch, number_of_classes]
target  [batch]
```

MNIST 中：

```text
data    [64, 1, 28, 28]
output  [64, 10]
target  [64]
```

---

## 3. CNN model

CNN = Convolutional Neural Network（卷积神经网络）。

模型结构：

```python
self.conv1 = nn.Conv2d(1, 32, 3, 1)
self.conv2 = nn.Conv2d(32, 64, 3, 1)

self.dropout1 = nn.Dropout(0.25)
self.dropout2 = nn.Dropout(0.5)

self.fc1 = nn.Linear(9216, 128)
self.fc2 = nn.Linear(128, 10)
```

`__init__()`：  
定义模型包含哪些层。

`forward()`：  
定义数据经过这些层的顺序。

```text
model(data)
→ __call__()
→ forward()
→ output
```

---

## 4. Conv2d

```python
nn.Conv2d(1, 32, 3, 1)
```

表示：

```text
1  = input channels
32 = output channels
3  = kernel_size = 3×3
1  = stride = 1
```

### Kernel

卷积核是一个小的可学习权重窗口，例如：

```text
3 × 3
```

它在输入图像上滑动并进行局部计算。

### Stride

`stride=1`：  
每次移动 1 个像素。

对于长度 28：

```text
1~3
2~4
...
26~28
```

共有 26 个合法位置，所以：

```text
28 → 26
```

第二个卷积同理：

```text
26 → 24
```

### Padding

padding = 在图像边缘补像素。

当前代码没有设置 padding，因此默认：

```text
padding = 0
```

所以卷积后高和宽会变小。

---

## 5. Shape flow through the CNN

输入：

```text
[64, 1, 28, 28]
```

经过：

```text
Conv2d(1,32,3,1)
→ [64, 32, 26, 26]

ReLU
→ [64, 32, 26, 26]

Conv2d(32,64,3,1)
→ [64, 64, 24, 24]

ReLU
→ [64, 64, 24, 24]

MaxPool2d(2)
→ [64, 64, 12, 12]

Dropout
→ [64, 64, 12, 12]

Flatten
→ [64, 9216]

Linear(9216,128)
→ [64, 128]

ReLU + Dropout
→ [64, 128]

Linear(128,10)
→ [64, 10]

log_softmax
→ [64, 10]
```

其中：

```text
9216 = 64 × 12 × 12
```

---

## 6. Max pooling

```python
F.max_pool2d(x, 2)
```

`2` 表示使用 `2×2` 区域。

每个 `2×2` 区域只保留一个最大值，因此：

```text
24×24
→
12×12
```

Max pooling 是局部下采样，不是从整个 Tensor 中只取一个最大值。

---

## 7. Flatten

```python
x = torch.flatten(x, 1)
```

`1` 表示：

```text
start_dim = 1
```

即：

> 保留第 0 维 batch，从第 1 维开始把后面的维度全部摊平。

例如：

```text
[64, 64, 12, 12]
→
[64, 9216]
```

所以：

```text
flatten(x, 1)
= 保留 batch 维，把每个样本的所有特征摊成一维
```

---

## 8. Dropout

```python
nn.Dropout(0.25)
nn.Dropout(0.5)
```

训练时会随机屏蔽一部分 activation。

作用：

```text
减少模型过度依赖少数特征
降低过拟合风险
```

Dropout 不改变 Tensor shape。

训练模式：

```python
model.train()
```

评估模式：

```python
model.eval()
```

在 eval mode 下 Dropout 不再随机屏蔽输出。

---

## 9. Output, target and loss

模型最后：

```python
self.fc2 = nn.Linear(128, 10)
```

输出：

```text
[batch, 10]
```

因为 MNIST 有 10 个类别：

```text
0~9
```

`target` 是每张图片的真实类别：

```text
target.shape = [batch]
```

例如：

```text
target = [5, 0, 4, 1, ...]
```

### log_softmax

```python
F.log_softmax(x, dim=1)
```

逻辑：

```text
logits
→ softmax probabilities
→ log probabilities
```

### NLLLoss

```python
loss = F.nll_loss(output, target)
```

NLLLoss = Negative Log Likelihood Loss。

这里是分类任务使用的损失函数。

直觉：

```text
模型越相信正确类别
→ loss 越小

模型越不相信正确类别
→ loss 越大
```

`loss function` = 损失函数  
`loss` = 损失 / 损失值

---

## 10. Training loop

核心代码：

```python
optimizer.zero_grad()
output = model(data)
loss = F.nll_loss(output, target)
loss.backward()
optimizer.step()
```

执行逻辑：

```text
optimizer.zero_grad()
→ 清除旧梯度

output = model(data)
→ forward，得到预测结果

loss = ...
→ 用 output 和 target 计算损失

loss.backward()
→ 反向传播，计算各参数的 gradient

optimizer.step()
→ 根据 gradient 更新模型参数
```

关键区别：

```text
backward()
= 计算梯度

optimizer.step()
= 真正更新参数
```

optimizer = 优化器。

---

## 11. Evaluation

```python
model.eval()

with torch.no_grad():
```

`model.eval()`：  
切换到评估模式。

`torch.no_grad()`：  
当前代码块不记录梯度。

测试阶段：

```text
forward
→ loss / prediction
→ accuracy
```

不需要：

```text
backward()
optimizer.step()
```

因为测试阶段只评估当前模型，不学习、不更新参数。

### argmax

```python
pred = output.argmax(dim=1)
```

`argmax` 返回最大值所在的 index。

对于 MNIST，这个 index 就是预测数字。

---

## 12. Optimizer and scheduler

```python
optimizer = optim.Adadelta(
    model.parameters(),
    lr=args.lr
)
```

optimizer：

```text
根据 gradient 更新模型的 weight / bias
```

```python
scheduler = StepLR(
    optimizer,
    step_size=1,
    gamma=args.gamma
)
```

scheduler：

```text
负责调整 optimizer 的 learning rate
```

例如 `gamma=0.7`：

```text
1.0
→ 0.7
→ 0.49
→ ...
```

---

## 13. args and epoch loop

```python
args = parser.parse_args()
```

`args` 保存命令行参数，例如：

```text
args.batch_size
args.epochs
args.lr
args.gamma
```

```python
for epoch in range(1, args.epochs + 1):
```

Python 的：

```text
range(start, stop)
```

是左闭右开：

```text
[start, stop)
```

因此默认 `epochs=14`：

```text
range(1, 15)
→ 1, 2, ..., 14
```

---

## 14. Mental model

```text
Dataset
→ DataLoader
→ data + target
→ model(data)
→ output
→ loss(output, target)
→ backward()
→ gradient
→ optimizer.step()
→ update parameters
→ test()
→ scheduler.step()
→ next epoch
```

### Key shapes

```text
data    [64, 1, 28, 28]
target  [64]
output  [64, 10]
```

### Core training loop

```python
optimizer.zero_grad()
output = model(data)
loss = F.nll_loss(output, target)
loss.backward()
optimizer.step()
```
