# 5. Project Reading

本阶段继续阅读 PyTorch 项目：

word_language_model

此前已经通过 MNIST CNN 项目熟悉了基本 PyTorch 训练流程：

- Dataset / DataLoader
- nn.Module
- forward()
- Tensor shape
- loss 与 optimizer

本次阅读重点从图像分类模型扩展到序列模型项目，关注：

- 多文件 PyTorch 项目的组织方式
- 数据处理流程
- Embedding 与序列模型的数据流
- model 与 forward 的关系
- 如何阅读不同类型的 PyTorch 项目

---

# 1. PyTorch 项目常见结构

本项目主要包含：
```

word_language_model/

├── main.py  
├── data.py  
├── model.py  
└── generate.py

```

不同文件负责不同功能。

## data.py

负责数据处理。

主要任务：
```

原始数据
↓
预处理
↓
Tensor

```

在本项目中：
```
文本
↓
token
↓
token ID
↓
Tensor

````

## model.py

负责定义神经网络模型。

例如：

```python
class RNNModel(nn.Module):
````

以及：

```python
class TransformerModel(nn.Module):
```

主要包含：

- 模型结构
    
- 网络层定义
    
- forward() 方法

## main.py

训练入口。

负责连接：

```
数据
↓
模型
↓
loss
↓
optimizer
↓
参数更新
```

主要流程：

```
解析参数
↓
加载数据
↓
创建模型
↓
定义 loss
↓
定义 optimizer
↓
训练
↓
验证
↓
保存模型
```

## generate.py

负责模型推理（inference）。

它不是训练程序。

区别：

训练：

```
输入数据
↓
模型预测
↓
计算 loss
↓
backward()
↓
更新参数
```

推理：

```
输入数据
↓
模型预测
↓
得到结果
```

推理阶段不会更新模型参数。

---

# 2. 阅读 PyTorch 项目的方法

阅读新的 PyTorch 项目时，可以按照以下顺序：

## Step 1：寻找入口文件

通常：

```
main.py
train.py
run.py
```

是程序入口。

需要找到：

- 数据在哪里加载
    
- 模型在哪里创建
    
- 训练在哪里开始

## Step 2：追踪数据流

阅读代码时，不要只看函数名字。

应该关注：

```
输入是什么？
↓
经过什么处理？
↓
进入模型的是什么？
↓
输出是什么？
```

本项目的数据流：

```
文本
↓
token
↓
token ID
↓
Tensor
↓
Embedding
↓
模型
↓
prediction
```

## Step 3：寻找模型定义

通常查看：

```
model.py
```

重点：

```python
class XXX(nn.Module):
```

以及：

```python
forward()
```

因为：

```python
model(x)
```

最终会调用：

```python
forward(x)
```

## Step 4：寻找训练循环

重点寻找：

```python
loss.backward()

optimizer.step()
```

因为这是 **PyTorch 更新模型参数的核心流程。

# 3. main.py 中的整体流程

main.py 负责将各个模块连接起来。

例如：

## 加载数据

```python
corpus = data.Corpus(args.data)
```

表示：

```
main.py
↓
调用 data.py 中的 Corpus 类
↓
得到处理后的数据对象
```

## 创建模型

例如：

```python
model = RNNModel(...)
```

表示：

```
main.py
↓
调用 model.py 中定义的模型
↓
创建模型对象
```

---

# 4. 数据从文本到 Tensor 的过程

该项目的数据处理流程：

```
原始文本
↓
tokenization
↓
token ID
↓
Tensor
```

## token

token 是模型处理的基本单位。

例如：

文本：

```
I love cats
```

可以拆成：

```
I
love
cats
```

每一个单位可以作为 token。

## token ID

神经网络不能直接处理：

```
cat
```

这样的字符串。

因此建立词表：

```
I    → 0
love → 1
cats → 2
```

转换：

```
I love cats
↓
0 1 2
↓
Tensor([0,1,2])
```

---

# 5. model() 与 forward()

PyTorch 中：

```python
model(x)
```

并不是直接调用一个普通函数。

执行过程：

```
model(x)
↓
nn.Module.__call__()
↓
forward()
↓
返回结果
```

因此，定义：

```python
def forward(self, x):
```

决定模型的数据流。

---

# 6. forward 中的数据流

例如：

```python
emb = self.drop(self.encoder(input))
```

执行顺序：

从内向外：

```
self.encoder(input)
↓
self.drop()
↓
emb
```

第一步：

```python
self.encoder(input)
```

其中：

```python
self.encoder = nn.Embedding(...)
```

所以：

```
token ID
↓
Embedding
↓
向量表示
```

第二步：

```python
self.drop(...)
```

对结果执行 Dropout。

最终：

```
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

---

# 7. RNN 中 hidden 的作用

RNN 的 forward：

```python
forward(self, input, hidden)
```

有两个输入：

```
input

hidden
```

其中：

```
input

=
当前 token 的 embedding
```

而：

```
hidden

=
RNN 保存的历史信息
```

处理序列时：

```
token1
↓
hidden1
token2 + hidden1
↓
hidden2
token3 + hidden2
↓
hidden3
```

需要注意：

```
hidden ≠ 上一个 token 的 embedding
```

区别：

```
embedding
=
当前 token 的表示

hidden
=
模型结合历史信息计算得到的内部状态
```

---

# 8. Tensor shape 阅读习惯

阅读模型代码时，需要持续关注：

```
Tensor 的 shape 如何变化
```

因为：

深度学习模型本质上是在不同 Tensor 表示之间进行转换。

例如：

```
输入 Tensor
↓
某一层
↓
输出 Tensor
```

需要知道：

- 每个维度代表什么
    
- 为什么需要这种 shape
    
- 下一层是否能够接收

在本项目中：

输入：

```
token ID Tensor
```

经过：

```
Embedding
```

变成：

```
embedding Tensor
```

再进入：

```
RNN / Transformer
```

---

# 9. 训练流程与之前学习的连接

虽然真实项目代码更加复杂，但是核心训练流程仍然是：

```
输入 Tensor
↓
model()
↓
forward()
↓
prediction
↓
loss
↓
backward()
↓
gradient
↓
optimizer.step()
↓
更新参数
```

对应 PyTorch 基础：

```python
loss.backward()

optimizer.step()
```

仍然是模型学习的核心。

---

# 10. 此次阅读获得的工程经验

通过阅读这个 PyTorch 项目，理解：

## 一个深度学习项目通常拆分：

```
数据处理
↓
模型定义
↓
训练流程
↓
推理流程
```

## 阅读代码时：

优先关注：

```
1. 输入是什么？

2. Tensor shape 是什么？

3. 经过哪些层？

4. 输出是什么？

5. 参数在哪里更新？
```