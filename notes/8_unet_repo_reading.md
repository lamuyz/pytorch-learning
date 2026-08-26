# milesial/Pytorch-UNet reading

仓库：

```text
milesial/Pytorch-UNet
```

---

## 1. 仓库主要结构

目前重点关注：

```text
Pytorch-UNet/
├── unet/
│   ├── unet_model.py
│   └── unet_parts.py
│
├── utils/
│   ├── data_loading.py
│   └── dice_score.py
│
├── train.py
├── evaluate.py
└── predict.py
```

各文件作用：

```text
unet_model.py
→ 组装完整 U-Net

unet_parts.py
→ 定义 U-Net 的基础模块

data_loading.py
→ 数据读取与预处理

dice_score.py
→ Dice 相关计算

train.py
→ 模型训练

evaluate.py
→ 模型验证 / 评价

predict.py
→ 使用训练好的模型进行预测
```

`utils` 是 `utilities` 的缩写，可以理解为“辅助工具”。

---

## 2. `unet_model.py`

核心类：

```python
class UNet(nn.Module):
```

说明 U-Net 本身仍然是一个标准的 PyTorch `nn.Module`。

初始化函数中：

```python
def __init__(self, n_channels, n_classes, bilinear=False):
```

这里：

```text
n_channels
n_classes
bilinear
```

是 `__init__()` 的参数。

而：

```python
self.n_channels = n_channels
```

之后：

```text
self.n_channels
```

才是对象的属性。

---

## 3. U-Net 的组装方式

模型主要由以下模块组成：

```python
self.inc = DoubleConv(...)

self.down1 = Down(...)
self.down2 = Down(...)
self.down3 = Down(...)
self.down4 = Down(...)

self.up1 = Up(...)
self.up2 = Up(...)
self.up3 = Up(...)
self.up4 = Up(...)

self.outc = OutConv(...)
```

可以翻译成：

```text
输入
↓
DoubleConv
↓
Encoder
↓
最深层特征
↓
Decoder
↓
OutConv
↓
logits
```

---

## 4. `forward()` 的执行流

核心结构：

```python
x1 = self.inc(x)

x2 = self.down1(x1)
x3 = self.down2(x2)
x4 = self.down3(x3)
x5 = self.down4(x4)

x = self.up1(x5, x4)
x = self.up2(x, x3)
x = self.up3(x, x2)
x = self.up4(x, x1)

logits = self.outc(x)

return logits
```

这段代码基本就是 U-Net 的完整数据流。

Encoder：

```text
x
↓
x1
↓
x2
↓
x3
↓
x4
↓
x5
```

Decoder：

```text
x5 + x4
↓
up1

结果 + x3
↓
up2

结果 + x2
↓
up3

结果 + x1
↓
up4
```

这里保存 `x1 ~ x4` 的主要原因，是后续 skip connection 需要再次使用这些特征。

---

## 5. `unet_model.py` 和 `unet_parts.py` 的关系

可以简单理解为：

```text
unet_parts.py
→ 定义零件

unet_model.py
→ 使用零件组装完整 U-Net
```

`unet_model.py` 不负责具体实现每一次卷积，而是规定整个网络的数据流和模块连接关系。

---

# 6. `unet_parts.py`

目前读到四个核心模块：

```text
DoubleConv
Down
Up
OutConv
```

---

### 6.1. DoubleConv

结构：

```text
Conv2d
↓
BatchNorm
↓
ReLU
↓
Conv2d
↓
BatchNorm
↓
ReLU
```

因此叫：

```text
DoubleConv
```

源码使用：

```python
nn.Sequential(...)
```

`Sequential` 中文可以理解为“顺序容器”。

它的作用是把多个网络层按顺序放在一起：

```text
输入
↓
第一层
↓
第二层
↓
第三层
↓
输出
```

这样 `forward()` 可以直接：

```python
return self.double_conv(x)
```

---

### 6.2 `in_channels`、`mid_channels`、`out_channels`

DoubleConv 的数据流可以表示为：

```text
in_channels
↓
第一次 Conv
↓
mid_channels
↓
第二次 Conv
↓
out_channels
```

默认情况下：

```text
mid_channels = out_channels
```

例如：

```python
DoubleConv(3, 64)
```

大致表示：

```text
3 channels
↓
64 channels
↓
64 channels
```

---

### 6.3 `kernel_size=3, padding=1`

卷积使用：

```python
kernel_size=3
padding=1
```

表示：

```text
3×3 卷积核
+
外围填充 1 个像素
```

因此通常可以保持空间尺寸不变：

```text
256×256
↓
Conv2d
↓
256×256
```

---

### 6.4 `bias=False`

Conv 中的 bias 是偏置项。

可以类比：

```text
输出 = 加权结果 + bias
```

bias 可以让输出整体发生平移。

这个仓库中的 Conv 后面紧跟 BatchNorm，因此卷积层设置：

```python
bias=False
```

因为 BatchNorm 自身已经包含可学习的平移和缩放参数。

---

## 7. Down

这个仓库自己定义：

```text
Down
=
MaxPool
+
DoubleConv
```

所以在主模型中：

```python
x2 = self.down1(x1)
```

内部实际发生：

```text
x1
↓
MaxPool
↓
DoubleConv
↓
x2
```

不需要在 `unet_model.py` 中再显式写一次 pooling。

注意：

```text
Down
```

不是 PyTorch 内置模块，而是这个仓库自己定义的类。

---

## 8. Up

目前先理解总体结构：

```text
上采样
↓
和 Encoder 保存的 feature 对齐
↓
concatenation
↓
DoubleConv
```

`Up` 的 `forward()` 接收两个输入：

```text
x1
→ Decoder 当前特征

x2
→ Encoder 保存的特征
```

例如：

```python
x = self.up1(x5, x4)
```

表示：

```text
x5
→ Decoder 当前深层 feature

x4
→ Encoder 保存的 feature
```

---

## 9. Skip Connection 和 `torch.cat`

真实代码中：

```python
torch.cat([x2, x1], dim=1)
```

就是 U-Net 的 skip connection 在 Decoder 中进行特征融合的关键步骤。

`cat` 来自：

```text
concatenate
```

`concatenation` 中文是“拼接”。

`dim=1` 表示沿 channel 维拼接。

例如：

```text
[B, 256, 64, 64]

+

[B, 256, 64, 64]

↓ torch.cat(..., dim=1)

[B, 512, 64, 64]
```

---

## 10. `F.pad`

`F` 来自：

```python
import torch.nn.functional as F
```

所以：

```text
F = torch.nn.functional
```

`pad` 来自 `padding`，中文可以理解为“填充”。

在这里主要用于解决 Decoder 和 Encoder feature 空间尺寸可能相差一两个像素的问题。

例如：

```text
Decoder feature:
64×64

Encoder feature:
65×65
```

先：

```text
64×64
↓
F.pad
↓
65×65
```

然后才能进行：

```python
torch.cat(...)
```

---

## 11. OutConv

`OutConv` 使用：

```text
1×1 Conv
```

例如：

```text
[B, 64, H, W]
↓
OutConv
↓
[B, 1, H, W]
```

最终输出的是：

```text
logits
```

`UNet.forward()` 本身没有做 Sigmoid，而是直接：

```python
return logits
```

后续训练或预测代码再决定如何处理 logits。

---
