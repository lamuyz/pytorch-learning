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

## 12. `train.py`：训练流程

`train.py` 负责把数据、模型、损失函数和优化器真正串起来。

```text
Dataset
↓
DataLoader
↓
image + ground truth mask
↓
U-Net
↓
logits
↓
计算 loss
↓
backward()
↓
optimizer.step()
↓
更新模型参数
```

训练数据会拆成：

```text
training set
→ 用于训练、计算梯度、更新参数

validation set
→ 训练过程中检查模型表现，不更新参数
```

训练集通常：

```python
shuffle=True
```

`shuffle` 表示打乱样本顺序。

---

### 12.1 Optimizer

`optimizer` 是优化器。

```text
loss.backward()
↓
计算参数梯度

optimizer.step()
↓
根据梯度真正更新参数
```

可以简单理解为：

```text
backward()
→ 算应该怎么改

optimizer.step()
→ 真正去改
```

---

### 12.2 BCEWithLogitsLoss 和 CrossEntropyLoss

仓库根据 `n_classes` 选择损失函数：

```python
criterion = (
    nn.CrossEntropyLoss()
    if model.n_classes > 1
    else nn.BCEWithLogitsLoss()
)
```

这是 Python 的条件表达式：

```text
A if 条件 else B
```

等价于：

```python
if model.n_classes > 1:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.BCEWithLogitsLoss()
```

二分类分割：

```text
背景
vs
目标
```

通常每个像素输出一个 logit：

```text
[B, 1, H, W]
```

使用：

```text
BCEWithLogitsLoss
```

多分类分割：

```text
class 0
class 1
class 2
...
```

每个像素对应多个类别 logits：

```text
[B, C, H, W]
```

使用：

```text
CrossEntropyLoss
```

需要注意：

```text
高级细胞分割
≠
一定是多分类
```

很多细胞分割问题仍然是前景 / 背景等少数语义类别，但还需要进一步解决：

```text
哪些像素属于同一个细胞
不同细胞如何彼此分开
```

这属于实例分割问题，模型输出和 loss 往往会更复杂。

---

### 12.3 `masks_pred` 和 `squeeze(1)`

代码：

```python
masks_pred = model(images)
```

这里的 `masks_pred` 虽然名字像“预测 mask”，但此时实际上还是：

```text
logits
```

还没有经过 Sigmoid 和 threshold。

二分类模型输出可能是：

```text
[B, 1, H, W]
```

而 ground truth 常存成：

```text
[B, H, W]
```

因此：

```python
masks_pred.squeeze(1)
```

会把长度为 `1` 的 channel 维删除：

```text
[B, 1, H, W]
↓
squeeze(1)
↓
[B, H, W]
```

这样 prediction 和 ground truth 的 shape 可以对齐。

---

## 13. BCE Loss 和 Dice Loss

二分类训练时，logits 分成两条路径。

```text
logits
↓
BCEWithLogitsLoss
↓
BCE Loss
```

这里不需要手动执行 Sigmoid，因为 `BCEWithLogitsLoss` 内部已经包含。

另一条：

```text
logits
↓
Sigmoid
↓
0~1 连续概率
↓
Soft Dice Loss
```

最后：

```text
BCE Loss
+
Dice Loss
↓
总 loss
↓
backward()
↓
optimizer.step()
```

训练 Dice Loss 时不会先用 threshold 把预测硬切成 `0/1`，因为需要保留连续概率带来的梯度信息。

---

## 14. `dice_score.py`

`dice_score.py` 主要负责 Dice 相关计算：

```text
dice_coeff
→ Dice score

multiclass_dice_coeff
→ 多分类 Dice

dice_loss
→ Dice Loss
```

Dice coefficient：

```text
Dice
=
2 × 重合区域
/
(预测区域 + 真实区域)
```

Dice 越接近 `1`，说明预测 mask 和真实 mask 重合得越好。

---

### 14.1 Soft Dice 的代码对应

例如：

```text
prediction:
0.1 0.2 0.8 0.7

target:
0   0   1   1
```

逐元素相乘：

```text
0.1×0 + 0.2×0 + 0.8×1 + 0.7×1
=
1.5
```

代码中的：

```text
inter
=
2 × 1.5
=
3
```

同时：

```text
prediction.sum() = 1.8
target.sum() = 2
```

所以：

```text
sets_sum
=
1.8 + 2
=
3.8
```

最终：

```text
Dice
≈
3 / 3.8
≈
0.789
```

对应代码：

```python
dice = (inter + epsilon) / (sets_sum + epsilon)
```

`epsilon` 是希腊字母 `ε`（艾普西隆），在这里表示一个非常小的数，用来避免分母为 0 等数值问题。

---

### 14.2 Dice score 和 Dice Loss

```text
Dice score
→ 评价预测 mask 与真实 mask 的重合度
→ 越大越好
```

```text
Dice Loss
=
1 - Dice
→ 用于训练
→ 越小越好
```

---

## 15. `evaluate.py`：验证模型

验证阶段：

```text
有 image
+
有 ground truth
```

但是：

```text
不 backward()
不 optimizer.step()
```

只检查当前模型表现，不修改参数。

### 15.1 `net.eval()`

```python
net.eval()
```

表示让模型进入评估模式。

这会影响：

```text
BatchNorm
Dropout
```

等训练和验证阶段行为不同的模块。

### 15.2 `@torch.inference_mode()`

`inference` 中文是“推理”。

在机器学习中表示：

```text
训练好的模型
+
输入数据
↓
直接计算预测结果
```

`@torch.inference_mode()` 是 Python 装饰器，表示：

```text
这个函数只做推理
↓
不记录反向传播需要的梯度信息
```

作用：

```text
减少内存占用
+
提高推理效率
```

### 15.3 threshold

二分类验证时：

```text
logits
↓
Sigmoid
↓
0~1 概率
↓
threshold
↓
0/1 prediction mask
```

常见：

```text
threshold = 0.5
```

例如：

```text
0.73 > 0.5
↓
1
↓
目标
```

```text
0.21 < 0.5
↓
0
↓
背景
```

`0.5` 是常用默认分界，但不是固定规定。

阈值降低：

```text
更多像素容易被判成目标
```

阈值提高：

```text
只有更高概率的像素才会被判成目标
```

验证时：

```text
0/1 prediction mask
+
ground truth mask
↓
Dice score
```

---

## 16. Training、Validation 和 Test

```text
training set
→ 用来训练和更新参数

validation set
→ 训练过程中检查模型表现

test set
→ 模型确定后做最终评价
```

可以理解成：

```text
training set
= 平时练习

validation set
= 阶段测验

test set
= 最终考试
```

---

## 17. Scheduler

`scheduler` 是学习率调度器。

如果验证 Dice 长时间不再提高，例如：

```text
0.800
0.801
0.800
0.802
```

可以降低 learning rate：

```text
原来用较大的步子更新参数
↓
改成更小的步子继续调整
```

这个仓库监控 Dice score，因此目标是：

```text
Dice 越大越好
```

---

## 18. Checkpoint

`checkpoint` 可以理解为模型训练过程中的存档。

例如：

```text
Epoch 10
↓
保存 checkpoint
```

以后加载后：

```text
不会重新随机初始化
```

而是：

```text
从 Epoch 10 已经学到的参数继续训练
```

如果想完整恢复训练状态，还可以同时保存：

```text
optimizer state
scheduler state
epoch
```

checkpoint 也可以直接加载到模型中用于预测。

---

## 19. `predict.py`：真正预测新图片

`predict.py` 和 `evaluate.py` 的主要区别：

```text
evaluate
→ 有 ground truth
→ 可以计算 Dice

predict
→ 通常没有 ground truth
→ 只需要得到预测 mask
```

预测流程：

```text
加载模型 / checkpoint
↓
读取新图片
↓
预处理
↓
U-Net
↓
logits
↓
Sigmoid
↓
概率
↓
threshold
↓
最终预测 mask
```

预测阶段：

```text
不 backward()
不 optimizer.step()
不需要 BCE Loss
不需要 Dice Loss
```

如果没有 ground truth，也无法计算真正的 Dice score。

---

## 20. `data_loading.py`

这个文件负责：

```text
硬盘上的 image + mask
↓
Dataset
↓
DataLoader
```

核心仍然是：

```text
__len__()
__getitem__()
```

`__len__()`：

```text
告诉 DataLoader dataset 中有多少个样本
```

`__getitem__()`：

```text
根据 index 取出一对 image + mask
```

通常返回类似：

```python
{
    'image': image,
    'mask': mask
}
```

因此 `train.py` 才可以使用：

```python
batch['image']
batch['mask']
```

图像分割中，image 和 ground truth mask 的空间变换必须保持一致：

```text
image resize
↓
mask 也必须对应 resize
```

否则图像像素和标签位置无法对应。

---

## 21. 完整仓库执行流

现在可以把整个仓库串成一条完整流程：

```text
硬盘上的 image + mask
↓
data_loading.py
↓
Dataset
↓
DataLoader
↓
batch
├── image
└── ground truth mask
↓
train.py
↓
UNet
↓
unet_model.py
↓
调用 unet_parts.py
↓
Encoder
↓
Decoder
↓
OutConv
↓
logits
```

训练阶段：

```text
logits
├────────────→ BCEWithLogitsLoss / CrossEntropyLoss
│
└→ Sigmoid 等处理
   ↓
   Dice Loss
↓
总 loss
↓
backward()
↓
计算梯度
↓
optimizer.step()
↓
更新模型参数
```

训练过程中：

```text
validation set
↓
evaluate.py
↓
预测结果
↓
Dice score
↓
检查当前模型表现
↓
scheduler 可能调整 learning rate
```

训练结束：

```text
checkpoint
↓
保存模型当前已经学到的参数
```

之后：

```text
checkpoint
↓
predict.py
↓
新图片
↓
模型
↓
logits
↓
概率
↓
threshold
↓
最终 mask
```

---

## 22. 当前对仓库各文件的理解

```text
unet/unet_parts.py
→ 定义 DoubleConv / Down / Up / OutConv

unet/unet_model.py
→ 把这些模块组装成完整 U-Net

utils/data_loading.py
→ 读取 image 和 mask，构建 Dataset

utils/dice_score.py
→ Dice score / Dice Loss

train.py
→ 训练模型

evaluate.py
→ 验证模型并计算 Dice

predict.py
→ 使用训练好的模型预测新图片
```

整个仓库可以浓缩成：

```text
数据读取
↓
模型
↓
训练
↓
验证
↓
保存参数
↓
预测
```

---

## 23. 当前阅读结论

目前已经能够解释这个仓库的主执行流：

```text
image + mask
↓
Dataset / DataLoader
↓
U-Net
↓
DoubleConv / Down / Up / skip connection
↓
logits
↓
BCE / CrossEntropy + Dice Loss
↓
backward()
↓
optimizer
↓
validation Dice
↓
checkpoint
↓
predict
```

还没有深入的部分主要是：

```text
bilinear / factor
AMP
gradient clipping
argparse
logging / wandb
更完整的 checkpoint state
```

这些工程细节暂时不影响理解这个 repo 的核心工作方式。

---

## 24. 下一步建议

下一步建议先不要立刻进入更复杂的细胞分割仓库。

当前已经完成：

```text
读懂
```

但还应该补一次：

```text
修改
↓
运行
↓
观察结果
```

### 建议修改 1：prediction threshold

分别尝试：

```text
threshold = 0.3
threshold = 0.5
threshold = 0.7
```

观察：

```text
threshold 降低
↓
更多像素被判成目标
```

以及：

```text
threshold 提高
↓
只有更高概率的像素才被判成目标
```

这个修改可以直接验证：

```text
logit
↓
Sigmoid
↓
概率
↓
threshold
↓
最终 mask
```

### 建议修改 2：打印关键 tensor shape

可以在 `forward()` 中暂时打印：

```text
x1
x2
x3
x4
x5
up1
up2
up3
up4
logits
```

验证自己对：

```text
Encoder 下采样
↓
Bottleneck
↓
Decoder 上采样
```

过程中 shape 变化的理解。

完成一次：

```text
修改
+
运行
+
验证
```

之后，再进入更接近真实细胞实例分割的仓库更合适。

下一阶段重点可以从：

```text
semantic segmentation
```

进入：

```text
instance segmentation
↓
多个细胞怎样彼此分开
↓
模型为什么需要更复杂的输出
↓
为什么 loss 不再只是 BCE / CrossEntropy + Dice
```
