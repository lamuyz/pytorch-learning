# U-Net Core Concepts

## 1. U-Net 

U-Net 是一种常用于图像分割的神经网络。

它的整体结构由两部分组成：

```text
输入图像
↓
Encoder
↓
Bottleneck
↓
Decoder
↓
分割结果
```

之所以叫 U-Net，是因为 Encoder 和 Decoder 组成的整体结构看起来像字母 `U`。

---

## 2. 分割任务中的 mask

在二分类分割中，mask 是一张像素级标签图。

通常：

```text
1 = 目标 / 前景
0 = 背景
```

所以背景也属于 mask 的一部分，mask 并不只代表目标区域。

例如：

```text
真实 mask：

0 0 0 0
0 1 1 0
0 1 1 0
0 0 0 0
```

其中 `1` 表示目标区域，`0` 表示背景。

---

## 3. Ground Truth

Ground truth 指真实标签或真实标注。

在监督式图像分割中：

```text
输入图像
+
ground truth mask
↓
模型预测
↓
和 ground truth 比较
↓
计算 loss
↓
反向传播
↓
更新模型参数
```

因此，可靠的 ground truth 对训练和评价都非常重要。

---

## 4. Logit、Sigmoid 和预测概率

U-Net 最后一层首先输出的是 `logit`，而不是概率。

logit 可以是任意实数：

```text
-3.2
0.7
2.3
5.1
```

经过 Sigmoid 后，会被转换到 `0~1` 之间：

```text
logit
↓
Sigmoid
↓
0~1 之间的概率
```

例如：

```text
logit = 1
↓
Sigmoid
↓
0.73
```

可以理解为：

```text
模型认为这个像素属于目标的概率约为 73%
```

这里的“连续概率”指概率值本身可以是：

```text
0.13
0.57
0.82
0.96
```

而不是只能取 `0` 或 `1`。

---

## 5. BCE Loss

BCE 全称：

> Binary Cross Entropy，二元交叉熵

它主要逐像素比较模型预测和真实标签之间的差异。

例如：

```text
真实标签 = 1
预测概率 = 0.9
```

误差较小。

如果：

```text
真实标签 = 1
预测概率 = 0.1
```

误差较大。

PyTorch 中更常见：

```python
nn.BCEWithLogitsLoss()
```

它可以直接接受 logits，内部完成：

```text
logits
↓
Sigmoid
↓
BCE
```

因此不需要手动先执行 `Sigmoid`。

---

## 6. Dice 和 Dice Loss

Dice coefficient（Dice 系数）用于衡量预测区域与真实区域的重合程度。

公式：

```text
Dice = 2 × 重合区域 / (预测区域 + 真实区域)
```

范围通常为：

```text
0 ~ 1
```

其中：

```text
Dice = 1
→ 完全重合

Dice = 0
→ 完全不重合
```

因此：

```text
Dice 越大越好
```

为了把它用于神经网络训练，通常定义：

```text
Dice Loss = 1 - Dice
```

因此：

```text
Dice Loss 越小越好
```

可以简单理解：

```text
Dice = 成绩
Dice Loss = 扣分
```

---

## 7. Soft Dice

训练时，不能在计算 Dice Loss 前先用 threshold 把模型预测硬切成 `0/1`。

例如模型输出概率：

```text
0.1  0.2
0.8  0.7
```

真实 mask：

```text
0    0
1    1
```

Soft Dice 直接使用这些概率值计算。

逐元素相乘：

```text
0.1 × 0 = 0
0.2 × 0 = 0
0.8 × 1 = 0.8
0.7 × 1 = 0.7
```

得到：

```text
软交集 = 1.5
```

预测概率总和：

```text
0.1 + 0.2 + 0.8 + 0.7 = 1.8
```

真实目标总和：

```text
0 + 0 + 1 + 1 = 2
```

于是：

```text
Soft Dice
= 2 × 1.5 / (1.8 + 2)
≈ 0.789
```

训练时保留连续概率，是为了让 loss 能随预测值平滑变化，从而进行梯度计算和反向传播。

---

## 8. U-Net 的基本模块

常见 U-Net 实现可以拆成：

```text
DoubleConv
Down
Up
OutConv
```

### DoubleConv

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

两次卷积，因此叫 `DoubleConv`。

### Down

```text
MaxPool
↓
DoubleConv
```

用于 Encoder 下采样。

通常表现为：

```text
空间尺寸 ↓
channel 数 ↑
```

### Up

```text
上采样
+
Encoder 保存的 feature
↓
Concatenation
↓
DoubleConv
```

用于 Decoder 恢复空间分辨率。

### OutConv

通常使用：

```text
1×1 Conv
```

把最后的内部 feature channels 转换成最终分割输出。

---

## 9. Skip Connection

U-Net 会把 Encoder 中保存的高分辨率特征直接送到对应的 Decoder 层。

例如：

```text
Encoder feature
        │
        │ skip connection
        ↓
Decoder feature
↓
concatenation
↓
DoubleConv
```

它的作用是让 Decoder 在恢复图像空间结构时，同时利用 Encoder 中保存的细节信息。

---

## 10. Concatenation

代码中常缩写为 `concat`。

PyTorch 中常见：

```python
torch.cat([x2, x1], dim=1)
```

`dim=1` 表示沿 channel 维拼接。

例如：

```text
[B, 256, 64, 64]
+
[B, 256, 64, 64]
↓
[B, 512, 64, 64]
```

空间尺寸不变，channel 数相加。

---

## 12. BatchNorm

BatchNorm 全称：

> Batch Normalization，批归一化

它不是单细胞分析中的 batch correction。

它主要对中间特征值进行标准化和重新缩放，使训练过程更稳定。

基本过程可以理解为：

```text
卷积输出
↓
计算当前 batch 中特征值的均值和方差
↓
标准化
↓
再通过可学习参数重新缩放和平移
```

---
