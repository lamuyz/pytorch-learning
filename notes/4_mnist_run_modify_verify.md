# 4_mnist_run_modify_verify
## 1. 今日目标

从“能读懂代码”进一步过渡到：

预测 → 修改 → 运行 → 验证

---
## 2. 完整运行 MNIST

运行：

```bash
python main.py --epochs 1
````

这次不使用 `--dry-run`。

`--dry-run` 不只是检查语法，它实际上也会执行：

forward → loss → backward → optimizer.step()

只是训练时会很快停止，所以主要用于快速检查整条训练流程是否能正常执行。

---

## 3. 修改 batch size 并预测 shape

运行：

```bash
python main.py --epochs 1 --batch-size 32
```

默认 `batch_size=64`，修改后为 32。

运行前预测：

- `data.shape = [32, 1, 28, 28]`
    
- `target.shape = [32]`
    
- `output.shape = [32, 10]`
    

含义：

- `data`：32 张 MNIST 图片，每张图片 shape 为 `[1, 28, 28]`
    
- `target`：32 张图片对应的 32 个真实标签
    
- `output`：32 张图片，每张图片有 10 个类别输出
    

注意：

`target.shape = [32]` 不等于 `target = 32`，而是表示 target 中有 32 个标签。

---

## 4. 用真实代码验证 shape

在 `train()` 中临时加入：

```python
if batch_idx == 0:
    print("data shape:", data.shape)
    print("target shape:", target.shape)
    print("output shape:", output.shape)
```

真实结果：

```text
data shape: torch.Size([32, 1, 28, 28])
target shape: torch.Size([32])
output shape: torch.Size([32, 10])
```

结果与预测一致。

---

## 5. 修改模型结构

原模型：

```python
self.fc1 = nn.Linear(9216, 128)
self.fc2 = nn.Linear(128, 10)
```

修改为：

```python
self.fc1 = nn.Linear(9216, 64)
self.fc2 = nn.Linear(64, 10)
```

对应 shape：

`[32, 9216] → [32, 64] → [32, 10]`

关键关系：

`fc1` 的输出维度必须等于 `fc2` 的输入维度。

因此如果把：

`Linear(9216, 128)`

改成：

`Linear(9216, 64)`

那么下一层也必须从：

`Linear(128, 10)`

改成：

`Linear(64, 10)`

否则 Tensor shape 无法连接。

---

## 6. Training log

当：

- `batch_size = 32`
    
- `log_interval = 10`
    

时，每 10 个 batch 打印一次训练日志。

因为：

`10 × 32 = 320`

所以训练进度中会看到：

```text
0
320
640
960
...
```

例如：

```text
Train Epoch: 1 [0/60000 (0%)]     Loss: ...
Train Epoch: 1 [320/60000 (1%)]   Loss: ...
Train Epoch: 1 [640/60000 (1%)]   Loss: ...
```

Loss 在训练过程中通常总体下降，但不要求每个 batch 都严格下降。

---

## 7. CNN shape 回顾

当 `batch_size=32`：

`[32, 1, 28, 28]`  
→ Conv2d(1, 32, 3, 1)  
→ `[32, 32, 26, 26]`  
→ Conv2d(32, 64, 3, 1)  
→ `[32, 64, 24, 24]`  
→ MaxPool2d(2)  
→ `[32, 64, 12, 12]`  
→ Flatten  
→ `[32, 9216]`  
→ Linear(9216, 128)  
→ `[32, 128]`  
→ Linear(128, 10)  
→ `[32, 10]`

如果隐藏层改成 64：

`[32, 9216] → [32, 64] → [32, 10]`
