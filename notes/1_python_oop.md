
# 1_python_oop

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

# 14. Python module、class 与 object

在真实项目中，代码通常会拆分到多个 `.py` 文件中。

例如：
```

project/

├── data.py  
├── model.py  
└── main.py

````

每一个 `.py` 文件都可以看作一个 Python module（模块）。

---

## module.class()

例如：

```python
import data

corpus = data.Corpus(path)
````

执行过程：

```
data
↓
data.py 模块

Corpus
↓
data.py 中定义的类

Corpus()
↓
调用类创建对象
```

因此：

```python
data.Corpus()
```

表示：

> 从 data 模块中找到 Corpus 类，并创建一个 Corpus 对象。

这种写法在 PyTorch 中也很常见：

```python
nn.Linear()
```

拆开：

```
nn
↓
torch.nn 模块

Linear
↓
Linear 类

()
↓
创建 Linear 层对象
```

所以：

```python
module.class()
```

本质是：

```
模块
↓
类
↓
创建对象
```

---

# 15. import 模块 vs import 类

Python 中常见两种导入方式。

## 方式1：导入整个模块

```python
import data
```

之后需要：

```python
data.Corpus()
```

因为：

```text
data
=
模块名
```

## 方式2：直接导入类

```python
from data import Corpus
```

之后：

```python
Corpus()
```

即可。

因为已经把类直接导入当前环境。

---

区别：

```python
import data

data.Corpus()
```

强调：

> 从哪个模块里面找类。

```python
from data import Corpus

Corpus()
```

强调：

> 直接使用这个类。

---

# 16. 如何判断 import 的是模块还是类？

可以看 import 语句。

例如：

```python
import torch
```

这里：

```text
torch
=
模块
```

所以：

```python
torch.Tensor()
```

表示：

调用 torch 模块中的 Tensor。

例如：

```python
from torch import Tensor
```

这里：

```text
Tensor
=
直接导入的类
```

所以：

```python
Tensor()
```

可以直接创建对象。

判断方法：

```python
import xxx
```

通常导入模块。

```python
from xxx import yyy
```

通常导入模块中的某个对象（可能是类、函数、变量）。

---

# 17. os.path.join()

代码：

```python
os.path.join(path, "train.txt")
```

拆开：
## os

全称：

```
Operating System
```

Python 的 `os` 模块提供与操作系统交互的功能。

---

## path

表示 路径处理功能。

## join

> 连接、拼接。

因此：

```python
os.path.join()
```

意思：

> 按照当前操作系统规则，将多个路径连接起来。

例如：

```python
path = "./data"

os.path.join(path, "train.txt")
```

结果：

```
./data/train.txt
```

为什么不用：

```python
path + "/train.txt"
```

因为不同操作系统路径格式可能不同。

例如：

Windows：

```
data\train.txt
```

Linux/macOS：

```
data/train.txt
```

`os.path.join()` 可以自动处理这些差异。

---

# 18. list 与 append()

Python 中：

```python
[]
```

表示：

> 空列表（list）

例如：

```python
ids = []
```

表示：

创建一个空 list：

```
ids = []
```

注意：

```python
{}
```

才通常表示：

> 空字典（dictionary）

例如：

```python
gene_dict = {}
```

## append()

```
append
=
追加
```

代码：

```python
ids.append(word)
```

执行：

```
ids
↓
调用 list 对象的 append 方法
↓
把 word 添加到列表末尾
```

例如：

```python
ids = []

ids.append(10)
ids.append(20)
```

结果：

```python
ids = [10, 20]
```

这里：

```
ids
=
list 对象

append()
=
list 提供的方法
```

方法属于对象。

这与之前学习的：

```python
model.parameters()
```

类似：

```
model
↓
对象

parameters()
↓
对象的方法
```

---

# 19. 阅读 Python 代码的方法：从右往左、从里往外

真实项目中经常遇到复杂表达式。

例如：

```python
args = parser.parse_args()
```

不要先看：

```python
args =
```

而应该先看：

```python
parser.parse_args()
```

因为：

```
parser.parse_args()
↓
产生结果

args
↓
保存这个结果
```

对于嵌套函数：

```python
self.tokenize(
    os.path.join(path, "train.txt")
)
```

执行顺序：

第一步：

```python
os.path.join(path, "train.txt")
```

得到文件路径。

第二步：

```python
self.tokenize(文件路径)
```

处理这个文件。

因此：

> 阅读嵌套代码时，可以从最里面开始理解，再逐层向外分析。


持续补充中...

