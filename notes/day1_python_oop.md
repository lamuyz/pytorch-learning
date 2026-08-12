
# Day1_python_oop

## 1. class / object / instance

### 核心概念

```python
class = 类（模板）
object / instance = 对象 / 实例（具体创建出来的东西）
```

例子：

```python
class Cell:
    pass


cell1 = Cell()
```

理解：

```
Cell → 类

cell1 → Cell的实例（对象）
```

`()`：

> 调用类，创建实例

---

## 自测

```python
cell1 = Cell()
```

回答：

- Cell是什么？
    
- cell1是什么？
    
- `()`是什么作用？
    

---
# 2. **init** 初始化方法

### 定义

`__init__`

> 特殊方法，在创建实例时自动执行。

代码：

```python
class Cell:

    def __init__(self, name):
        self.name = name


cell1 = Cell("Neuron")
```

等价：

```python
Cell.__init__(cell1, "Neuron")
```

其中：

```python
self = cell1
```

所以：

```python
self.name = name
```

等价：

```python
cell1.name = "Neuron"
```

---
# 3. self

### 定义

self：

> 当前对象自己。

例如：

```python
cell1 = Cell("Neuron")
```

执行：

```python
self.name = name
```

实际：

```python
cell1.name = "Neuron"
```

---
# 4. attribute 和 method

## Attribute（属性）

对象拥有的数据。

通常：

```python
self.xxx = xxx
```

例：

```python
self.name = name
```

调用：

```python
cell1.name
```

---

## Method（方法）

class里面定义的函数。

代码：

```python
class Cell:

    def show_info(self):
        print(self.name)
```

调用：

```python
cell1.show_info()
```

---
# 5. inheritance（继承）

代码：

```python
class MyModel(nn.Module):
```

含义：

```text
MyModel继承nn.Module
```

关系：

```
nn.Module（父类）
        ↓
MyModel（子类）
```

作用：

获得父类已有功能。

---

## 自测

为什么：

```python
class MyModel(nn.Module)
```

括号里面不是参数？

---
# 6. super()

代码：

```python
super().__init__()
```

拆解：

```
super()
↓
找到父类

.
↓
访问父类方法

__init__()
↓
调用父类初始化方法
```

作用：

执行：

```python
nn.Module.__init__()
```

---
## 自测

为什么 MyModel 需要：

```python
super().__init__()
```

---

# 7. forward 和 **call**

## forward

代码：

```python
def forward(self,x):
```

这是：

> 方法（method）

因为它定义在 class 内。

---

PyTorch：

```python
model(x)
```

实际：

```
model(x)

↓

model.__call__(x)

↓

model.forward(x)
```

---
# 8. pathlib

全称：

> pathlib = path library

作用：

处理文件路径。

代码：

```python
from pathlib import Path


data = Path("data")

file = data / "sample.txt"
```

优势：

跨平台。

---
# 9. argparse

全称：

> argument parser

中文：

参数解析器。

作用：

从命令行读取参数。

代码：

```python
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--epochs")

args = parser.parse_args()
```

运行：

```bash
python train.py --epochs 100
```

获取：

```python
args.epochs
```

---
# 10. config file

作用：

保存实验参数。

config.yaml:

```yaml
epochs: 100
lr: 0.001
```

读取：

```python
config["epochs"]
```

---

区别：

argparse：

临时修改

config：

保存实验配置，可复现。

---
# 11. logging

替代：

```python
print()
```

代码：

```python
import logging

logging.info("start")
logging.error("failed")
```

作用：

记录：

- 时间
    
- 等级
    
- 信息
    

---
# 12. Dataset

PyTorch已有：

```python
torch.utils.data.Dataset
```

自己继承：

```python
class MyDataset(Dataset):
```

关系：

```
Dataset
   ↓
MyDataset
```

---

代码：

```python
class MyDataset(Dataset):

    def __len__(self):
        return 100


    def __getitem__(self,index):
        return index
```

---
# 13. DataLoader

代码：

```python
loader = DataLoader(
    dataset,
    batch_size=10
)
```

含义：

不是加载10个数据集。

而是：

> 每次取10个sample组成一个batch。

关系：

```
Dataset
负责:
    一个样本怎么取

DataLoader
负责:
    怎么批量组织
```
