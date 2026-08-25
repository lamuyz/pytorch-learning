# U-Net Segmentation

## 1. 从分类CNN到图像分割

### 1.1 为什么需要分割模型？

此前 MNIST CNN 学习的是 Image Classification（图像分类）。

分类任务回答：

``` text
这张图片是什么？
```

例如：

``` text
28×28 image
↓
CNN
↓
Linear
↓
0-9类别
```

而细胞分割任务需要预测每一个像素属于什么。

例如：

``` text
DAPI image
↓
segmentation mask
```

------------------------------------------------------------------------

### 1.2 为什么分类CNN不能直接用于分割？

分类CNN：

``` text
image
↓
CNN feature extraction
↓
feature map
↓
Flatten
↓
Linear
↓
class prediction
```

Flatten会破坏空间信息。

分类需要知道：

``` text
有什么feature
```

分割需要知道：

``` text
feature在哪里
```

因此需要保留空间结构。

------------------------------------------------------------------------

## 2. U-Net整体结构

### 2.1 U-Net简介

U-Net全称：

> U-Net: Convolutional Networks for Biomedical Image Segmentation

用于生物医学图像分割。

核心思想：

结合：

``` text
高级语义信息
+
空间细节信息
```

整体结构：

``` text
Input image
↓
Encoder
↓
Bottleneck
↓
Decoder
↓
Segmentation mask
```

------------------------------------------------------------------------

### 2.2 Encoder

Encoder本质是CNN特征提取部分。

作用：

从图片中提取越来越抽象的feature。

``` text
pixel
↓
浅层feature
↓
中层feature
↓
深层feature
```

浅层：

-   边缘
-   纹理

深层：

-   形状
-   结构
-   语义信息

------------------------------------------------------------------------

### 2.3 Bottleneck（瓶颈层）

Bottleneck位于Encoder和Decoder之间。

``` text
Encoder
↓
64×64×512
↓
Bottleneck
↓
Decoder
```

特点：

空间尺寸最小，feature最抽象。

------------------------------------------------------------------------

### 2.4 Decoder

Decoder负责恢复空间尺寸。

``` text
64×64
↓
128×128
↓
256×256
↓
512×512
```

并生成pixel-level prediction。

------------------------------------------------------------------------

## 3. U-Net核心机制

### 3.1 Downsampling（下采样）

Encoder逐渐降低空间尺寸：

``` text
512×512
↓
256×256
↓
128×128
↓
64×64
```

常见方法：

Max Pooling：

``` python
nn.MaxPool2d(2)
```

Strided Convolution：

``` python
nn.Conv2d(
    64,
    128,
    kernel_size=3,
    stride=2
)
```

------------------------------------------------------------------------

### 3.2 Channel变化

CNN shape：

``` text
batch × channel × height × width
```

Encoder中：

``` text
空间尺寸减少

512×512
↓
64×64


channel增加

64
↓
128
↓
256
↓
512
```

channel增加代表保存更多种类feature。

------------------------------------------------------------------------

### 3.3 Upsampling（上采样）

Decoder恢复空间尺寸。

常见方式：

Interpolation + Conv：

``` text
64×64
↓
128×128
↓
Conv
```

Transposed Convolution：

``` python
nn.ConvTranspose2d()
```

用于Decoder上采样。

------------------------------------------------------------------------

### 3.4 Skip Connection（跳跃连接）

Skip connection：

将Encoder中间层产生的feature map传递给Decoder。

不是：

``` text
原始pixel
```

而是：

``` text
Encoder feature map
```

作用：

结合：

``` text
高级语义
+
空间细节
```

------------------------------------------------------------------------

### 3.5 Concat

Concat：

concatenation（拼接）。

U-Net沿channel方向拼接。

例如：

``` text
Decoder:
128×128×256

Encoder skip:
128×128×256

Concat:

128×128×512
```

之后通过Conv重新学习feature组合。

------------------------------------------------------------------------

## 4. U-Net输出与任务目标

### 4.1 为什么U-Net输出mask？

分割任务需要pixel-level prediction。

因此：

不是：

``` text
Flatten
↓
Linear
```

而是：

``` text
Feature map
↓
Conv
↓
Mask
```

例如：

``` text
512×512×64
↓
1×1 Conv
↓
512×512×1
```

------------------------------------------------------------------------

### 4.2 Classification CNN vs U-Net

Classification CNN：

``` text
Image
↓
CNN
↓
Flatten
↓
Linear
↓
Class
```

输出一个类别。

U-Net：

``` text
Image
↓
Encoder
↓
Bottleneck
↓
Decoder
↓
Mask
```

输出pixel-level mask。

------------------------------------------------------------------------

## 5. 与细胞分割任务的联系

在细胞分割中：

输入：

``` text
H&E / DAPI image
```

Encoder学习：

``` text
边缘
↓
细胞结构
```

Decoder恢复：

``` text
pixel位置
```

Skip connection帮助：

``` text
保持细胞边界细节
```

最终输出：

``` text
cell mask
```
