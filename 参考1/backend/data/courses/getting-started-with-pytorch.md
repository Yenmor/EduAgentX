---
course_id: pytorch
course_name: PyTorch 深度学习实践
level: 深度学习实践层
source_file: getting-started-with-pytorch.md
knowledge_base: 高校人工智能专业能力进阶课程群
---

# PyTorch 深度学习实践

## 课程定位

使用 PyTorch 完成模型构建、训练、调试、评估和实验复现。

## 先修课程

Python 程序设计, 神经网络与深度学习

## 后续课程

大语言模型基础, 计算机视觉与 CNN

## 核心知识点

- Tensor
- Autograd
- nn.Module
- Dataset
- DataLoader
- 训练循环
- 模型保存

## 适配资源类型

- 代码实验
- 讲解文档
- 项目任务
- 测验

﻿# PyTorch入门

## Chapter 1 Pytorch Fundamentals Setup

### PyTorch 是什么？

## 原始内容：PyTorch 是什么？

PyTorch 是一个主要由 Meta AI 开发的开源机器学习 (machine learning)库。它在研究和工业界都受到了广泛欢迎，用于构建和训练深度学习 (deep learning)模型。它是一个强大的工具集，专门为满足现代机器学习应用的需求而设计。

其核心功能在于，PyTorch 提供了两个使其非常有效的基本功能：

1. **张量计算：** 与 NumPy 数组类似，PyTorch 提供了称为张量的多维数组。然而，PyTorch 张量带有一个主要优势：它们可以方便地在图形处理器（GPU）或其他专用硬件加速器上处理。这种能力显著加快了训练大型神经网络 (neural network)所需的数值计算，与仅使用 CPU 执行相比，通常能将速度提高几个数量级。如果你使用过 NumPy，你会发现操作这些张量的 API 会让你感到熟悉，从而简化了转换过程。
2. **自动微分：** 训练神经网络涉及根据损失函数 (loss function)的梯度调整模型参数 (parameter)。手动计算这些梯度既复杂又容易出错，特别是对于深层架构。PyTorch 包含一个复杂的自动微分引擎，称为 `Autograd`。当对张量执行操作时，`Autograd` 会动态构建一个计算图。这个图会记录操作序列，允许 PyTorch 在需要时（通常通过调用 `.backward()`）使用链式法则自动计算梯度。与需要预先定义静态图的框架相比，这种动态特性在模型设计上提供了相当大的灵活性。

### 为何选用 PyTorch？

有几个因素促成了 PyTorch 的广泛使用：

- **Python 风格的接口：** PyTorch 与 Python 数据科学环境紧密结合。其 API 设计直观自然，对于 Python 开发者来说相对容易学习和使用。调试通常感觉就像调试标准的 Python 代码。
- **灵活性：** 动态计算图（由运行定义）意味着网络的结构可以在执行期间改变。这对某些类型的模型特别有用，如循环神经网络 (neural network) (RNN)，其中序列长度可能不同，或者在模型内部实现复杂的控制流。
- **丰富的生态体系：** PyTorch 得益于一套丰富的支持库和工具。像 `torchvision`、`torchaudio` 和 `torchtext` 这样的库针对特定应用领域（分别是计算机视觉、音频和自然语言处理）提供预构建的数据集、模型架构和数据转换。与 TensorBoard 等可视化工具的集成进一步提升了开发流程。
- **研究与生产：** 虽然最初因其灵活性而在研究社区中广受欢迎，PyTorch 已大幅成熟，现在包含了 TorchServe 和 TorchScript 等工具，使其成为将模型部署到生产环境的可行选择。

本章将带你开始了解 PyTorch，通过关注第一个基本功能：张量。你将学习如何创建它们、操作它们、执行基本运算，并理解它们与 NumPy 数组的关系。掌握这些基础元素是第一步，以使用 PyTorch 的强大功能构建和训练复杂的深度学习 (deep learning)模型。

获取即时帮助、个性化解释和交互式代码示例。

---

### 安装与环境配置

### 安装与环境配置

在开始构建深度学习 (deep learning)模型之前，您需要配置好 PyTorch 开发环境。这包括安装核心库和设置您的工作区。由于本课程假定您熟悉 Python，我们将侧重于将 PyTorch 集成到标准的 Python 环境中。

### 选择安装方式：Conda 对比 Pip

PyTorch 可以使用两种主要的包管理工具安装：Conda（Anaconda 或 Miniconda 发行版的一部分）和 Pip（Python 默认的包安装器）。

- **Conda：** 在数据科学和机器学习 (machine learning)场景中常用。Conda 同时管理包和环境，简化了复杂依赖项的处理，包括 GPU 加速所需的非 Python 库，如 CUDA 工具包。如果您已经在使用 Anaconda，这通常是最顺畅的路径。
- **Pip：** 标准的 Python 包安装器。它运行良好，特别是当您倾向于使用 `venv` 等工具管理 Python 环境时。如果您需要 GPU 支持，可能需要单独处理 NVIDIA 驱动和 CUDA 工具包的安装。

为了保证可重现性并避免与其他项目冲突，强烈建议将 PyTorch 安装在专门的**虚拟环境**中，无论您选择 Conda 还是 Pip。


start

需要安装 PyTorch 吗？

condaᵤser

使用 Anaconda/Miniconda 吗？

start->condaᵤser

use\_conda

1. 创建 Conda 环境
conda create -n pytorchₑnv python=3.9
conda activate pytorchₑnv
2. 使用 Conda 安装命令
（请查阅 pytorch.org 获取最新命令）

condaᵤser->use\_conda

是

useₚip

1. 创建虚拟环境 (venv)
python -m venv pytorchₑnv
source pytorchₑnv/bin/activate # Linux/macOS
.\pytorchₑnv\Scripts\activate # Windows
2. 使用 Pip 安装命令
（请查阅 pytorch.org 获取最新命令）

condaᵤser->useₚip

否（或偏好 pip）

need\_gpu

需要 GPU 加速吗？

use\_conda->need\_gpu

useₚip->need\_gpu

cpu\_command

从 PyTorch 官网选择仅限 CPU 的命令

need\_gpu->cpu\_command

否

gpu\_command

1. 安装 NVIDIA 驱动/CUDA 工具包
（请参考 NVIDIA/PyTorch 文档）
2. 从 PyTorch 官网选择支持 CUDA 的命令

need\_gpu->gpu\_command

是

verify

验证安装：
`python -c 'import torch; print(torch.\_ᵥersion\_₎; print(torch.cuda.isₐvailable())'`

cpu\_command->verify

gpu\_command->verify

end

完成！

verify->end

> 安装决策流程图，总结了 Conda 和 Pip 之间的选择，考虑了 GPU 要求，并以验证结束。

### 安装步骤

确切的安装命令取决于您的操作系统（Linux、macOS、Windows）、包管理工具（Conda、Pip）以及您需要仅支持 CPU 还是支持 GPU (CUDA)。

**最佳做法是始终查阅 PyTorch 官方网站 ([pytorch.org](https://pytorch.org/))**，获取最新的安装命令生成器。选择您的偏好，它将提供确切的运行命令。

**1. 使用 Conda**

首先，创建并激活一个新的 Conda 环境（将 `pytorch_env` 替换为您喜欢的名称并选择合适的 Python 版本）：

```bash

conda create -n pytorch_env python=3.9

conda activate pytorch_env
```

现在，访问 PyTorch 网站，选择您的操作系统、Conda、Python 版本和所需的 CUDA 版本（或 CPU）。复制生成的命令。它看起来会像这样（仅为示例，**请从官网获取最新命令！**）：

```bash

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

在您已激活的 Conda 环境中运行网站提供的命令。

**2. 使用 Pip**

首先，使用 `venv` 创建并激活一个新的虚拟环境：

```bash

python -m venv pytorch_env

source pytorch_env/bin/activate

.\pytorch_env\Scripts\activate

.\pytorch_env\Scripts\Activate.ps1
```

接下来，访问 PyTorch 网站，选择您的操作系统、Pip、Python 版本和所需的 CUDA 版本（或 CPU）。复制生成的命令。它通常会涉及指定一个索引 URL 和可能的 CUDA 版本（仅为示例，**请从官网获取最新命令！**）：

```bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

在您已激活的虚拟环境中运行网站提供的命令。确保您的 `pip` 已更新到最新版本（`pip install --upgrade pip`）。

### GPU 支持 (CUDA)

深度学习 (deep learning)计算，特别是矩阵乘法，在使用 CUDA 并行计算平台的 NVIDIA GPU 上运行速度明显加快。

- **要求：** 您需要一块支持 CUDA 的 NVIDIA GPU。
- **驱动和工具包：** 您必须在安装支持 CUDA 的 PyTorch 版本*之前*，在系统上安装相应的 NVIDIA 显示驱动程序和 CUDA 工具包。所需的 CUDA 工具包版本必须与 PyTorch 构建时所用的版本匹配（例如，11.7、11.8、12.1）。Conda 可能会为您处理工具包的安装，而使用 Pip 时，您通常需要先在系统范围内安装它。请参阅 NVIDIA 的文档和 PyTorch 安装指南以获取具体信息。
- **安装命令：** 当使用 PyTorch 网站的命令生成器时，选择 CUDA 版本将定制命令以安装 GPU 兼容的二进制文件。

如果您没有兼容的 GPU 或初期不需要 GPU 加速，只需在安装时选择“CPU”选项。您可以在 CPU 上执行所有操作，尽管对于大型模型来说可能会慢一些。

### 验证安装

安装完成后，您可以通过打开 Python 解释器（在您已激活的虚拟环境中）并运行以下代码来验证安装：

```python
import torch

print(f"PyTorch Version: {torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:

    print(f"Number of GPUs: {torch.cuda.device_count()}")

    print(f"Current GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("PyTorch is using CPU.")

x = torch.rand(2, 3)
print("成功创建了一个张量:")
print(x)
```

如果这些命令运行无误并报告了正确的版本和 CUDA 状态（如果您安装了 GPU 版本并且硬件/驱动程序已设置好则为 True，否则为 False），您的安装就成功了。

### 环境配置注意事项

- **一致性：** 在开始 PyTorch 项目或运行脚本之前，请务必激活您的专用 `pytorch_env`（或您命名的其他环境）。
- **常用库：** 您通常会搭配 PyTorch 使用其他库。使用 `conda install <package>` 或 `pip install <package>` 将它们安装到相同的环境中：
  - `numpy`：用于数值操作和与其他库的接口。
  - `matplotlib` 或 `seaborn`：用于绘图和可视化。
  - `scikit-learn`：用于传统机器学习 (machine learning)任务和实用工具。
  - `jupyterlab` 或 `notebook`：用于交互式开发。
- **IDE：** 将您的集成开发环境 (IDE)，如 VS Code 或 PyCharm，配置为使用位于虚拟环境中的 Python 解释器。这可确保代码自动补全和调试功能与 PyTorch 正常协作。

PyTorch 安装完毕且环境配置好后，您现在就可以开始学习其核心部分：张量。

获取即时帮助、个性化解释和交互式代码示例。

---

### 张量介绍

### 张量介绍

\*\*张量（Tensor）\*\*是 PyTorch 中将要使用的主要数据结构。如果你使用过 NumPy，你会发现 PyTorch 张量非常熟悉。从本质上讲，张量是多维数组，与 NumPy 的 `ndarray` 非常相似。

可以把张量看作是常见数学对象的推广：

- 一个**标量**（单个数字，例如 7）是 0 维张量（或 0 阶张量）。
- 一个**向量 (vector)**（一维数字列表或数组，例如 `[1, 2, 3]`）是 1 维张量（或 1 阶张量）。
- 一个**矩阵**（二维数字网格，例如 `[[1, 2], [3, 4]]`）是 2 维张量（或 2 阶张量）。
- 依此类推，可以有 3 维张量（例如数字立方体，常用于 RGB 图像：高 x 宽 x 通道）、4 维张量（常用于图像批次：批大小 x 高 x 宽 x 通道）等。


Scalar

标量 (7)
0维张量

Vector

向量 [1, 2, 3]
1维张量

Scalar->Vector

增加维度

Matrix

矩阵 [[1, 2],
       [3, 4]]
2维张量

Vector->Matrix

增加维度

Tensor3D

3维张量
(例如：图像 高x宽x通道)

Matrix->Tensor3D

增加维度

TensorND

高维张量
(例如：批次 批大小x高x宽x通道)

Tensor3D->TensorND

...

> 一种将张量看作标量、向量和矩阵的推广的方式，维度逐渐增加。

在深度学习 (deep learning)中，张量用来表示几乎所有数据：

- **输入数据：** 图像批次、文本序列或特征表格。
- **模型参数 (parameter)：** 神经网络 (neural network)层的权重 (weight)和偏置 (bias)。
- **中间激活：** 网络内部各层的输出。
- **梯度：** 反向传播 (backpropagation)过程中计算的值，用于更新模型参数。

与标准 Python 列表甚至 NumPy 数组相比，是什么让 PyTorch 张量特别适合深度学习呢？

1. **GPU 加速：** 张量可以轻松地移动到图形处理器 (GPU) 或其他硬件加速器上进行处理。这为深度学习中常见的计算密集型操作提供了大幅加速。
2. **自动微分：** PyTorch 张量通过 `Autograd` 系统内置了对自动微分的支持（我们将在第 3 章中介绍）。这种机制自动计算梯度，这对通过反向传播训练神经网络来说非常重要。

尽管其原理与 NumPy 数组相似，但这两个特性使得 PyTorch 张量成为高效构建和训练模型的主要工具。在接下来的部分中，我们将介绍如何创建和操作这些重要的数据结构。

获取即时帮助、个性化解释和交互式代码示例。

---

### 创建张量

### 创建张量

张量是 PyTorch 中的基本数据结构，代表深度学习 (deep learning)模型中使用的多维数组。代码中提供了创建张量的实用方法。PyTorch 提供了一系列灵活的函数来创建张量，无论使用现有数据，还是需要特定形状和初始值。

### 从现有数据创建张量

创建张量最直接的方法通常是使用现有的 Python 数据结构，比如列表或 NumPy 数组。主要用于此目的的函数是 `torch.tensor()`。这个函数会根据输入数据自动判断数据类型 (`dtype`)，但您也可以明确指定。重要的是，`torch.tensor()` 总是复制输入数据。

我们来看看实际操作。首先，请确保已导入 PyTorch：

```python
import torch
import numpy as np
```

现在，从 Python 列表创建一个张量：

```python

data_list = [[1, 2], [3, 4]]
tensor_from_list = torch.tensor(data_list)

print("从列表生成的张量:")
print(tensor_from_list)
print(f"数据类型: {tensor_from_list.dtype}")
print(f"形状: {tensor_from_list.shape}")
```

这将输出：

```
从列表生成的张量:
tensor([[1, 2],
        [3, 4]])
数据类型: torch.int64
形状: torch.Size([2, 2])
```

请注意 PyTorch 如何根据输入整数判断数据类型为 `torch.int64`（一个 64 位整数）。

您也可以从 NumPy 数组开始，获得相同的结果：

```python

data_numpy = np.array([[5.0, 6.0], [7.0, 8.0]])
tensor_from_numpy = torch.tensor(data_numpy)

print("\n从 NumPy 数组生成的张量:")
print(tensor_from_numpy)
print(f"数据类型: {tensor_from_numpy.dtype}")
print(f"形状: {tensor_from_numpy.shape}")
```

输出：

```
从 NumPy 数组生成的张量:
tensor([[5., 6.],
        [7., 8.]], dtype=torch.float64)
数据类型: torch.float64
形状: torch.Size([2, 2])
```

这里，数据类型被判断为 `torch.float64`，因为 NumPy 数组包含浮点数。本章后面将讨论 PyTorch 张量与 NumPy 数组之间的关系，包括更有效的转换方法。

### 创建具有特定形状和值的张量

很多时候，您需要在没有预先数据的情况下创建特定大小的张量，例如用零、一或随机值进行初始化。PyTorch 为这些情况提供了专门的函数。形状通常作为元组或整数序列传递。

**零张量、一张量或未初始化数据张量：**

- `torch.zeros(*size, ...)`: 创建一个填充零的张量。
- `torch.ones(*size, ...)`: 创建一个填充一的张量。
- `torch.empty(*size, ...)`: 创建一个指定大小的张量，但其内容未被设定。其中的值是分配内存时该位置的任意数据。如果您计划在创建后立即填充张量，使用 `torch.empty` 会快一些，但请注意，初始值是不确定的。

```python

shape = (2, 3)

zeros_tensor = torch.zeros(shape)
ones_tensor = torch.ones(shape)
empty_tensor = torch.empty(shape)

print(f"\n零张量 (形状 {shape}):")
print(zeros_tensor)

print(f"\n一张量 (形状 {shape}):")
print(ones_tensor)

print(f"\n空张量 (形状 {shape}):")
print(empty_tensor)
```

输出（请注意 `empty_tensor` 的值会有所不同）：

```
零张量 (形状 (2, 3)):
tensor([[0., 0., 0.],
        [0., 0., 0.]])

一张量 (形状 (2, 3)):
tensor([[1., 1., 1.],
        [1., 1., 1.]])

空张量 (形状 (2, 3)):
tensor([[2.1321e-38, 5.0837e+24, 3.0763e-41],
        [3.0763e-41, 1.2326e-32, 1.8750e+00]])
```

默认情况下，这些函数创建的数据类型为 `dtype=torch.float32` 的张量。您可以使用 `dtype` 参数 (parameter)指定不同的数据类型：

```python

ones_int_tensor = torch.ones(shape, dtype=torch.int32)
print(f"\n一张量 (dtype=torch.int32):")
print(ones_int_tensor)
```

输出：

```
一张量 (dtype=torch.int32):
tensor([[1, 1, 1],
        [1, 1, 1]], dtype=torch.int32)
```

**带随机值的张量：**

初始化模型参数通常会从随机值开始。PyTorch 为此提供了函数：

- `torch.rand(*size, ...)`: 创建一个从区间 [0, 1) 均匀采样的张量。
- `torch.randn(*size, ...)`: 创建一个从标准正态分布（均值 0，方差 1）采样的张量。

```python

rand_tensor = torch.rand(shape)
randn_tensor = torch.randn(shape)

print(f"\n随机张量 (均匀分布 [0, 1), 形状 {shape}):")
print(rand_tensor)

print(f"\n随机张量 (标准正态分布, 形状 {shape}):")
print(randn_tensor)
```

输出（每次运行结果会有所不同）：

```
随机张量 (均匀分布 [0, 1), 形状 (2, 3)):
tensor([[0.6580, 0.5089, 0.1642],
        [0.3742, 0.5989, 0.7775]])

随机张量 (标准正态分布, 形状 (2, 3)):
tensor([[-0.2651, -0.3249, -1.0134],
        [ 1.1314,  1.1751, -0.1411]])
```

### 基于其他张量创建张量

有时，您需要创建一个新张量，使其与现有张量具有相同的属性（如形状和 `dtype`）。这对于确保操作兼容性很有帮助。PyTorch 为此提供了 `_like` 变体函数：

- `torch.zeros_like(input_tensor, ...)`
- `torch.ones_like(input_tensor, ...)`
- `torch.rand_like(input_tensor, ...)`
- `torch.randn_like(input_tensor, ...)`

这些函数接受一个现有张量作为输入，并返回一个新张量，其内容为指定的值（零、一、随机），但会与输入张量的形状和 `dtype` 匹配，除非明确覆盖。

```python

base_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(f"\n基础张量 (形状 {base_tensor.shape}, dtype {base_tensor.dtype}):")
print(base_tensor)

zeros_like_base = torch.zeros_like(base_tensor)
rand_like_base = torch.rand_like(base_tensor)

print("\n类似基础张量的零张量:")
print(zeros_like_base)
print(f"形状: {zeros_like_base.shape}, dtype: {zeros_like_base.dtype}")

print("\n类似基础张量的随机张量:")
print(rand_like_base)
print(f"形状: {rand_like_base.shape}, dtype: {rand_like_base.dtype}")
```

输出：

```
基础张量 (形状 torch.Size([2, 2]), dtype torch.float32):
tensor([[1., 2.],
        [3., 4.]])

类似基础张量的零张量:
tensor([[0., 0.],
        [0., 0.]])
形状: torch.Size([2, 2]), dtype: torch.float32

类似基础张量的随机张量:
tensor([[0.1216, 0.5908],
        [0.3264, 0.9272]])
形状: torch.Size([2, 2]), dtype: torch.float32
```

如您所见，新张量 `zeros_like_base` 和 `rand_like_base` 自动从 `base_tensor` 继承了 `shape` (torch.Size([2, 2])) 和 `dtype` (torch.float32)。

这些方法为您生成所需的张量提供了一套全面的方式，它们支撑着深度学习 (deep learning)模型中的计算。在接下来的章节中，我们将了解如何操作这些张量并进行重要运算。

获取即时帮助、个性化解释和交互式代码示例。

---

### 基本张量操作

### 基本张量操作

对 PyTorch 张量执行基本操作是一项主要技能。张量支持多种数学和逻辑运算。这些运算许多都与 NumPy 中的对应功能类似，都是按元素进行的。这些运算是神经网络 (neural network)计算的根本。

### 按元素算术运算

最常见的运算是将标准算术函数独立应用于参与运算的张量的每个元素。这些运算通常要求张量具有兼容的形状（关于形状兼容性，我们将在下一章讨论广播时详细说明）。

您可以使用标准 Python 算术运算符或等效的 `torch` 函数：

- **加法**: `+` 或 `torch.add()`
- **减法**: `-` 或 `torch.sub()`
- **乘法**: `*` 或 `torch.mul()`
- **除法**: `/` 或 `torch.div()`
- **幂运算**: `**` 或 `torch.pow()`

让我们看看这些操作如何运行：

```python
import torch

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])

sum_tensor = a + b
print("加法 (a + b):\n", sum_tensor)
print("加法 (torch.add(a, b)):\n", torch.add(a, b))

diff_tensor = a - b
print("\n减法 (a - b):\n", diff_tensor)

mul_tensor = a * b
print("\n按元素乘法 (a * b):\n", mul_tensor)
print("按元素乘法 (torch.mul(a, b)):\n", torch.mul(a, b))

div_tensor = a / b
print("\n除法 (a / b):\n", div_tensor)

pow_tensor = a ** 2
print("\n幂运算 (a ** 2):\n", pow_tensor)
print("幂运算 (torch.pow(a, 2)):\n", torch.pow(a, 2))
```

这些操作会创建包含结果的*新*张量。原始张量 `a` 和 `b` 保持不变。

### 就地操作

PyTorch 也提供许多操作的就地版本。这些操作直接修改张量，而不创建新对象，这可以节省内存。就地函数通常可以通过名称中末尾的下划线 `_` 来识别（例如，`add_`、`mul_`）。

```python
import torch

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])

print("原始张量 'a':\n", a)

a.add_(b)
print("\na.add_(b) 后张量 'a':\n", a)

a.mul_(2)
print("\na.mul_(2) 后张量 'a':\n", a)
```

虽然就地操作可以提高内存效率，但请谨慎使用。如果在计算图中的其他地方需要原始值，就地修改张量可能会导致自动求导（Autograd）（第 3 章会介绍）中的梯度计算问题。通常更稳妥的做法，尤其是在学习时，是使用返回新张量的标准操作。

### 标量操作

您可以对张量和单个数字（标量）执行算术运算。PyTorch 会自动扩展标量以匹配张量的形状，从而进行按元素运算。

```python
import torch

t = torch.tensor([[1, 2, 3], [4, 5, 6]])
scalar = 10

print("t + 标量:\n", t + scalar)

print("\nt * 标量:\n", t * scalar)

print("\nt - 标量:\n", t - scalar)
```

### 其他数学函数

PyTorch 提供丰富的数学函数库，这些函数按元素对张量进行操作，类似于 NumPy 的通用函数（ufuncs）。

```python
import torch

t = torch.tensor([[1., 4.], [9., 16.]])

print("平方根 (torch.sqrt(t)):\n", torch.sqrt(t))

print("\n指数 (torch.exp(t)):\n", torch.exp(t))

t_pos = torch.abs(t) + 1e-6
print("\n自然对数 (torch.log(t_pos)):\n", torch.log(t_pos))

t_neg = torch.tensor([[-1., 2.], [-3., 4.]])
print("\n绝对值 (torch.abs(t_neg)):\n", torch.abs(t_neg))
```

`torch` 模块中还有许多其他函数，例如 `torch.sin()`、`torch.cos()`、`torch.tanh()`、`torch.sigmoid()` 等。

### 归约操作

归约操作会减少张量中的元素数量，通常用于汇总信息。常见例子包括求和、求平均值、求最小值和最大值。

```python
import torch

t = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print("原始张量:\n", t)

total_sum = torch.sum(t)
print("\n所有元素的和 (torch.sum(t)):", total_sum)

mean_val = torch.mean(t.float())
print("所有元素的平均值 (torch.mean(t.float())):", mean_val)

max_val = torch.max(t)
print("张量中的最大值 (torch.max(t)):", max_val)

min_val = torch.min(t)
print("张量中的最小值 (torch.min(t)):", min_val)
```

您还可以使用 `dim` 参数 (parameter)沿着特定维度执行归约操作。这会折叠指定的维度，返回一个维度减少的张量。

```python
import torch

t = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print("原始张量:\n", t)

sum_dim0 = torch.sum(t, dim=0)
print("\n沿着 dim=0 求和（列）:\n", sum_dim0)

sum_dim1 = torch.sum(t, dim=1)
print("\n沿着 dim=1 求和（行）:\n", sum_dim1)

mean_dim1 = torch.mean(t.float(), dim=1)
print("\n沿着 dim=1 求平均值（行）:\n", mean_dim1)
```

051015行 0行 1

> 对张量 `[[1, 2, 3], [4, 5, 6]]` 沿着 `dim=1` 求和，结果为 `[1+2+3, 4+5+6] = [6, 15]`。

了解 `dim` 的工作原理对许多深度学习 (deep learning)操作很重要，例如计算每个批次项的损失或应用批归一化 (normalization)。

### 比较操作

您可以使用标准比较运算符（`>`、`<`、`>=`、`<=`、`==`、`!=`）按元素比较张量。结果是一个布尔值张量（`torch.bool`）。

```python
import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[1, 5], [0, 4]])

print("张量 'a':\n", a)
print("张量 'b':\n", b)

print("\na == b:\n", a == b)

print("\na > b:\n", a > b)

print("\na <= b:\n", a <= b)
```

布尔张量对掩码操作很有用，您将在更高级的张量操作中遇到它们。

### 逻辑操作

逻辑操作（`torch.logical_and()`、`torch.logical_or()`、`torch.logical_not()`）按元素对布尔张量或可在布尔上下文 (context)中求值的张量进行操作（其中 0 为假，非零为真）。

```python
import torch

bool_a = torch.tensor([[True, False], [True, True]])
bool_b = torch.tensor([[False, True], [True, False]])

print("布尔张量 'bool_a':\n", bool_a)
print("布尔张量 'bool_b':\n", bool_b)

print("\ntorch.logical_and(bool_a, bool_b):\n", torch.logical_and(bool_a, bool_b))

print("\ntorch.logical_or(bool_a, bool_b):\n", torch.logical_or(bool_a, bool_b))

print("\ntorch.logical_not(bool_a):\n", torch.logical_not(bool_a))
```

这些基本操作为更复杂的计算提供了基本组成部分。掌握它们是在 PyTorch 中实现数值算法和神经网络 (neural network)层的第一步。在下一章中，我们将介绍更高级的张量操作方法，包括索引、切片、重塑和广播。

获取即时帮助、个性化解释和交互式代码示例。

---

### 与 NumPy 的关联

### 与 NumPy 的关联

如果你对 Python 中的科学计算有经验，你可能很熟悉 NumPy 及其 `ndarray` 对象。NumPy 提供了一种强大的 N 维数组结构，已成为 Python 中数值操作的标准。PyTorch 认识到这种普遍性，并提供了与 NumPy 数组出色的互操作性。事实上，PyTorch 张量与 NumPy 数组非常相似：它们都是数字多维网格的抽象。

这种密切关联使得两者之间的切换变得简单，让你在 PyTorch 生态系统中工作时，能够运用现有的 NumPy 代码或库。

### 相似结构，不同能力

从本质上看，PyTorch 张量和 NumPy 数组都表示多维的密集数值数据。你可能在 NumPy 数组上执行的许多操作，在 PyTorch 张量中都有直接对应，通常具有相似的命名约定：

- **创建：** 两个库都提供函数来创建填充零、一、随机数或从现有 Python 列表创建数组/张量。
- **数学运算：** 逐元素加法、减法、乘法、除法、求幂、三角函数等，操作方式相似。
- **索引和切片：** 访问和修改元素或子数组/子张量使用可比较的语法。
- **形状操作：** 改变数组/张量的形状、转置和连接遵循相似的原则。

然而，主要由于 PyTorch 对深度学习 (deep learning)的侧重，两者之间存在一些基本区别：

1. **GPU 加速：** PyTorch 张量可以移到图形处理单元 (GPU) 上处理。这使得大规模并行计算成为可能，为深度学习中常见的矩阵乘法及其他操作提供了显著的速度提升。NumPy 数组主要为 CPU 计算设计。
2. **自动微分：** PyTorch 张量通过 `Autograd` 系统（第 3 章会讲到）内置支持自动微分。这种机制会自动跟踪对需要梯度的张量执行的操作，并在反向传播 (backpropagation)期间计算这些梯度，这对训练神经网络 (neural network)来说很重要。NumPy 数组不具备此能力。

### NumPy 数组与张量之间的转换

PyTorch 使得在这两种数据结构之间转换变得简单。

#### NumPy 数组到 PyTorch 张量

你可以使用 `torch.from_numpy()` 函数直接从 NumPy 数组创建 PyTorch 张量。

```python
import numpy as np
import torch

numpy_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
print(f"NumPy 数组:\n{numpy_array}")
print(f"NumPy 数组类型: {numpy_array.dtype}")

pytorch_tensor = torch.from_numpy(numpy_array)
print(f"\nPyTorch 张量:\n{pytorch_tensor}")
print(f"PyTorch 张量类型: {pytorch_tensor.dtype}")
```

**内存共享的重要说明：** 使用 `torch.from_numpy()` 时，生成的 PyTorch 张量和原始 NumPy 数组在 CPU 上共享相同的底层内存位置。这意味着修改一个对象会影响另一个。这种行为很高效，因为它避免了数据复制，但你需要注意这一点。

```python

numpy_array[0, 0] = 99
print(f"\n修改后的 NumPy 数组:\n{numpy_array}")
print(f"修改 NumPy 数组后的 PyTorch 张量:\n{pytorch_tensor}")

pytorch_tensor[1, 1] = -1
print(f"\n修改后的 PyTorch 张量:\n{pytorch_tensor}")
print(f"修改 PyTorch 张量后的 NumPy 数组:\n{numpy_array}")
```

如你所见，更改会反映在两个对象中，因为它们指向内存中的相同数据。

#### PyTorch 张量到 NumPy 数组

反之，你可以使用 `.numpy()` 方法将位于 CPU 上的 PyTorch 张量转换回 NumPy 数组。

```python

cpu_tensor = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
print(f"原始 PyTorch 张量 (CPU):\n{cpu_tensor}")

numpy_array_converted = cpu_tensor.numpy()
print(f"\n转换后的 NumPy 数组:\n{numpy_array_converted}")
print(f"NumPy 数组类型: {numpy_array_converted.dtype}")
```

同样，生成的 NumPy 数组和原始 CPU 张量共享相同的底层内存。对一个的修改会影响另一个。

```python

cpu_tensor[0, 1] = 25.0
print(f"\n修改后的 PyTorch 张量:\n{cpu_tensor}")
print(f"修改张量后的 NumPy 数组:\n{numpy_array_converted}")

numpy_array_converted[1, 0] = 35.0
print(f"\n修改后的 NumPy 数组:\n{numpy_array_converted}")
print(f"修改 NumPy 数组后的张量:\n{cpu_tensor}")
```

**GPU 张量：** `.numpy()` 方法仅适用于存储在 CPU 上的张量。如果你的张量在 GPU 上，你必须先使用 `.cpu()` 方法将其移到 CPU，然后才能将其转换为 NumPy 数组。直接在 GPU 张量上调用 `.numpy()` 会导致错误。

```python

if torch.cuda.is_available():
    gpu_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    print(f"\nGPU 上的张量:\n{gpu_tensor}")

    cpu_tensor_from_gpu = gpu_tensor.cpu()
    numpy_from_gpu = cpu_tensor_from_gpu.numpy()
    print(f"\n转换后的 NumPy 数组 (来自 GPU 张量):\n{numpy_from_gpu}")

else:
    print("\nCUDA 不可用，跳过 GPU 到 NumPy 的示例。")
```

### 结合两者的优势

轻松地在 NumPy 数组和 PyTorch 张量之间转换的能力非常实用。你可能使用熟悉的 NumPy 函数或其他操作 NumPy 数组的库来执行初始数据加载和预处理。然后，当需要构建或训练深度学习 (deep learning)模型时，你可以将数据转换为 PyTorch 张量，以借助于 GPU 加速和自动微分。同样，模型输出（即张量）可以转换回 NumPy 数组，以便使用 Matplotlib 或 Seaborn 等库进行分析或可视化。

理解这种关联和内存共享的含义，让你能够编写高效的代码，有效连接通用科学 Python 生态系统与 PyTorch 提供的专门深度学习能力。

获取即时帮助、个性化解释和交互式代码示例。

---

### 动手实践：环境配置与张量基本操作

### 动手实践：环境配置与张量基本操作

这些实践练习将提供PyTorch安装和基本张量对象的实践经验。它们将巩固对环境配置和执行基本张量操作的理解，这些都是使用PyTorch构建任何深度学习 (deep learning)模型的前提条件。

### 验证你的PyTorch安装

首先，让我们确保PyTorch已正确安装并在你的Python环境中可用。打开你的Python解释器或Jupyter Notebook并运行以下命令：

```python
import torch
import numpy as np

print(f"PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")

    device = torch.device("cuda")
else:
    print("CUDA not available. Using CPU.")
    device = torch.device("cpu")

print(f"Default device: {device}")
```

执行这段代码可以确认`torch`库能够被导入并显示其版本。它还会检查GPU的可用性，这对于后续章节中加速计算很重要。如果你有兼容的NVIDIA GPU并且安装了正确的PyTorch版本，应该会看到CUDA被报告为可用。目前，我们将主要使用CPU，但知道如何检查GPU也很重要。

### 创建张量

让我们练习使用之前介绍的多种方法创建张量。

**1. 从Python列表创建：**
从一个嵌套的Python列表创建一个2x3的张量。

```python

data = [[1, 2, 3], [4, 5, 6]]
tensor_from_list = torch.tensor(data)

print("从列表创建的张量：")
print(tensor_from_list)
print(f"形状: {tensor_from_list.shape}")
print(f"数据类型: {tensor_from_list.dtype}")
```

**2. 指定数据类型：**
创建相同的张量，但显式地将其数据类型设置为32位浮点数。

```python

tensor_float32 = torch.tensor(data, dtype=torch.float32)

print("\nfloat32数据类型的张量：")
print(tensor_float32)
print(f"形状: {tensor_float32.shape}")
print(f"数据类型: {tensor_float32.dtype}")
```

注意`dtype`的变化以及数字的表示方式（例如，`1.` 而不是 `1`）。

**3. 使用工厂函数：**
创建具有特定形状和初始值的张量。

```python

zeros_tensor = torch.zeros(3, 4)
print("\n全零张量 (3x4)：")
print(zeros_tensor)

ones_tensor_int = torch.ones(2, 2, dtype=torch.int32)
print("\n全一张量 (2x2, int32)：")
print(ones_tensor_int)

range_tensor = torch.arange(start=0, end=5, step=1)
print("\n范围张量 (0到4)：")
print(range_tensor)

rand_tensor = torch.rand(2, 3)
print("\n随机张量 (2x3)：")
print(rand_tensor)
```

这些工厂函数便于初始化张量，无需使用像列表这样的预先存在的数据结构。

### 张量基本操作

现在，让我们对已创建的张量执行一些基本操作。

```python

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.ones(2, 2)

print("张量 'a'：")
print(a)
print("张量 'b'：")
print(b)

sum_tensor = a + b

print("\n元素级和 (a + b)：")
print(sum_tensor)

prod_tensor = a * b

print("\n元素级积 (a * b)：")
print(prod_tensor)

scalar_mult = a * 3
print("\n标量乘法 (a * 3)：")
print(scalar_mult)

print(f"\n就地加法前 'a'：ID {id(a)}")
a.add_(b)
print("就地加法后 'a' (a.add_(b))：")
print(a)
print(f"就地加法后 'a'：ID {id(a)}")

x = torch.rand(2, 3)
y = torch.rand(3, 2)
matmul_result = torch.matmul(x, y)

print("\n矩阵乘法 (x @ y)：")
print(f"张量 x 的形状: {x.shape}, 张量 y 的形状: {y.shape}")
print(f"结果形状: {matmul_result.shape}")
print(matmul_result)
```

请注意元素级操作（如`+`、`*`）与矩阵乘法（`torch.matmul`或`@`）之间的区别。此外，还要注意就地操作（如`add_`）如何直接修改张量，而不创建新对象。

### 与NumPy的交互

PyTorch与NumPy良好配合。让我们练习在NumPy数组和PyTorch张量之间进行转换。

```python

numpy_array = np.array([[1.0, 2.0], [3.0, 4.0]])
print("\nNumPy数组：")
print(numpy_array)
print(f"类型: {type(numpy_array)}")

tensor_from_numpy = torch.from_numpy(numpy_array)
print("\n从NumPy数组创建的张量：")
print(tensor_from_numpy)
print(f"类型: {type(tensor_from_numpy)}")

numpy_array[0, 0] = 99.0
print("\n修改后的NumPy数组：")
print(numpy_array)
print("修改NumPy数组后的张量（共享内存）：")
print(tensor_from_numpy)

another_tensor = torch.tensor([[5, 6], [7, 8]], dtype=torch.float64)
print("\n另一个PyTorch张量：")
print(another_tensor)

numpy_from_tensor = another_tensor.numpy()
print("\n从张量创建的NumPy数组：")
print(numpy_from_tensor)
print(f"类型: {type(numpy_from_tensor)}")

another_tensor[1, 1] = 100.0
print("\n修改后的张量：")
print(another_tensor)
print("修改张量后的NumPy数组（共享内存）：")
print(numpy_from_tensor)
```

NumPy数组和CPU张量之间的这种内存共享行为效率高，但需要谨慎处理，因为可能会发生意外修改。如果你需要一个独立的副本，可以在转换之前使用张量的`.clone()`方法，或者使用标准的Python/NumPy复制机制。

本实践环节涵盖了验证你的配置、以多种方式创建张量、执行基本的算术和矩阵操作，以及在PyTorch张量和NumPy数组之间进行转换。掌握这些基础知识很重要，因为我们将在后续章节转向自动微分和构建神经网络 (neural network)模块等更复杂的主题。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 2 Advanced Tensor Manipulations

### 张量索引与切片

### 张量索引与切片

访问和修改张量的特定部分是处理深度学习 (deep learning)数据时常有的需求。无论您是需要选择单个数据点、提取一批训练样本、裁剪图像补丁，还是挑选特定特征，PyTorch都提供了强大且灵活的索引和切片机制，类似于NumPy数组中的那些，但PyTorch的机制与GPU加速和自动微分功能相集成。

### 基本索引

访问张量元素最直接的方式是使用标准的Python整数索引。请记住，PyTorch张量与Python列表和NumPy数组一样，使用0作为起始索引。

对于一维张量，您可以使用其索引访问元素：

```python
import torch

x_1d = torch.tensor([10, 11, 12, 13, 14])
print(f"原始一维张量:\n{x_1d}")

first_element = x_1d[0]
print(f"\n第一个元素 (x_1d[0]): {first_element}, 类型: {type(first_element)}")

last_element = x_1d[-1]
print(f"最后一个元素 (x_1d[-1]): {last_element}")

x_1d[1] = 110
print(f"\n修改后的张量:\n{x_1d}")
```

注意，访问单个元素会返回一个包含单个值的`torch.Tensor`（一个0维张量或标量），而不是一个标准的Python数字，除非您使用`.item()`明确提取它。元素的修改是原地进行的。

对于多维张量，您需要为每个维度提供索引，并用逗号分隔：

```python

x_2d = torch.tensor([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
print(f"原始二维张量:\n{x_2d}")

element_0_1 = x_2d[0, 1]
print(f"\n在 [0, 1] 的元素: {element_0_1}")

first_row = x_2d[0]
print(f"\n第一行 (x_2d[0]): {first_row}")

second_col = x_2d[:, 1]
print(f"第二列 (x_2d[:, 1]): {second_col}")

x_2d[1, 1] = 55
print(f"\n修改后的二维张量:\n{x_2d}")
```

提供的索引数量少于维数时，会沿剩余维度选择一个完整的子张量。例如，`x_2d[0]` 会选择整个第一行。

### 张量切片

切片允许您沿张量维度选择一系列元素。语法是 `start:stop:step`，其中 `start` 是包含的，`stop` 是不包含的，而 `step` 定义了间隔。省略 `start` 默认为0，省略 `stop` 默认为维度的末尾，省略 `step` 默认为1。

```python

y_1d = torch.arange(10)
print(f"原始一维张量: {y_1d}")

slice1 = y_1d[2:5]
print(f"\n切片 y_1d[2:5]: {slice1}")

slice2 = y_1d[:4]
print(f"切片 y_1d[:4]: {slice2}")

slice3 = y_1d[6:]
print(f"切片 y_1d[6:]: {slice3}")

slice4 = y_1d[::2]
print(f"切片 y_1d[::2]: {slice4}")

slice5 = y_1d[1:8:2]
print(f"切片 y_1d[1:8:2]: {slice5}")

slice6 = y_1d.flip(dims=[0])
print(f"使用 .flip(dims=[0]) 反转后的张量: {slice6}")
```

切片对多维张量的工作方式类似。您可以将整数索引和切片结合使用：

```python

x_2d = torch.tensor([[ 0,  1,  2,  3],
                     [ 4,  5,  6,  7],
                     [ 8,  9, 10, 11]])
print(f"原始二维张量:\n{x_2d}")

sub_tensor1 = x_2d[0:2, 1:3]
print(f"\n切片 x_2d[0:2, 1:3]:\n{sub_tensor1}")

sub_tensor2 = x_2d[:, -2:]
print(f"\n切片 x_2d[:, -2:]:\n{sub_tensor2}")

sub_tensor3 = x_2d[0, 1:]
print(f"\n切片 x_2d[0, 1:]:\n{sub_tensor3}")

sub_tensor4 = x_2d[::2, :]
print(f"\n切片 x_2d[::2, :]:\n{sub_tensor4}")
```


cluster₀

原始张量 (x₂d)

cluster₁

切片: x₂d[0:2, 1:3]

a

0

1

2

3

4

5

6

7

8

9

10

11

b

1

2

5

6

> 张量 `x_2d` 使用 `x_2d[0:2, 1:3]` 进行切片的视觉表示。由于切片索引遵循“左闭右开”原则（即不包含结束索引），这选择了索引为 0 和 1 的行，以及索引为 1 和 2 的列。

切片的一个重要特性（与某些其他形式的索引不同）是，返回的张量通常与原始张量共享底层存储。修改切片会修改原始张量。

```python
print(f"修改切片前的原始 x_2d:\n{x_2d}")

sub_tensor = x_2d[0:2, 1:3]

sub_tensor[0, 0] = 101

print(f"\n修改后的切片:\n{sub_tensor}")
print(f"\n修改切片后的原始 x_2d:\n{x_2d}")
```

如果您需要一个不共享内存的副本，请在切片上使用 `.clone()`: `sub_tensor_copy = x_2d[0:2, 1:3].clone()`。

### 布尔索引 (遮罩)

您可以使用布尔张量来索引另一个张量。布尔张量的形状必须能够广播到被索引张量的形状（通常，它们的形状完全相同）。只有布尔张量中对应 `True` 值的元素（即“遮罩”）才会被选中。这对于根据条件筛选数据非常有用。

```python

data = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(f"原始数据张量:\n{data}")

mask = data > 3
print(f"\n布尔遮罩 (data > 3):\n{mask}")

selected_elements = data[mask]
print(f"\n通过遮罩选择的元素:\n{selected_elements}")
print(f"所选元素的形状: {selected_elements.shape}")

data[data <= 3] = 0
print(f"\n将小于等于3的元素设置为零后的数据:\n{data}")
```

布尔索引通常返回一个包含所有选定元素的一维张量。与切片不同，它*不*保留原始形状。此外，布尔索引通常会创建一个副本，而不是一个视图。

您可以将布尔索引与其他形式结合使用。例如，根据应用于其中一列的条件选择行：

```python

row_mask = data[:, 0] > 2
print(f"\n行遮罩 (data[:, 0] > 2): {row_mask}")

selected_rows = data[row_mask, :]

print(f"\n第一列大于2的行:\n{selected_rows}")
```

### 整数数组索引

除了单个整数和切片，您还可以使用列表或一维整数张量沿维度进行索引。这使得您可以按任意顺序选择元素，或多次选择相同的元素。

```python
x = torch.arange(10, 20)
print(f"原始一维张量: {x}")

indices = torch.tensor([0, 4, 2, 2])
selected = x[indices]
print(f"\n使用索引 {indices} 选择的元素: {selected}")

y = torch.arange(12).reshape(3, 4)

print(f"\n原始二维张量:\n{y}")

row_indices = torch.tensor([0, 2])
selected_rows = y[row_indices]
print(f"\n使用索引 {row_indices} 选择的行:\n{selected_rows}")

col_indices = torch.tensor([1, 3])
selected_cols = y[:, col_indices]
print(f"\n使用索引 {col_indices} 选择的列:\n{selected_cols}")

row_idx = torch.tensor([0, 1, 2])
col_idx = torch.tensor([1, 3, 0])
selected_elements = y[row_idx, col_idx]
print(f"\n使用 (row_idx, col_idx) 选择的特定元素:\n{selected_elements}")
```

与布尔索引类似，整数数组索引通常返回一个新的张量（一个副本），而不是原始张量存储的视图。输出的形状取决于索引方法。当选择完整的行或列时，其他维度会被保留。当为多个维度提供索引数组（例如 `y[row_idx, col_idx]`）时，结果通常是一个对应于所选元素的一维张量。

掌握这些索引和切片技术能够精准地控制张量数据，为后续步骤中的数据准备、特征提取以及模型输入输出的操作奠定根基。

获取即时帮助、个性化解释和交互式代码示例。

---

### 张量的重塑与维度调整

### 张量的重塑与维度调整

通常，你会发现现有张量的结构不太适合后续计算步骤，尤其是在将数据送入特定的神经网络 (neural network)层时。PyTorch 提供了灵活的工具，可以在不改变底层数据元素本身的情况下，改变张量的形状或调整其维度。用于这些操作的主要方法是：`view()`、`reshape()` 和 `permute()`。

### 使用 `view()` 和 `reshape()` 改变形状

`view()` 和 `reshape()` 都允许你改变张量的维度，前提是总元素数量保持不变。它们在将多维张量展平后传递给线性层，或增加/移除大小为1的维度等任务中非常有用。

#### 使用 `view()`

`view()` 方法返回一个新的张量，该张量与原始张量共享相同的底层数据，但具有不同的形状。它非常高效，因为它避免了数据复制。然而，`view()` 要求张量在内存中是*连续的*。连续张量是指其元素在内存中按维度顺序连续存储，没有间隙的张量。大多数新创建的张量是连续的，但某些操作（如切片或使用 `t()` 进行转置）会产生非连续张量。

我们来看一个例子：

```python
import torch

x = torch.arange(12)
print(f"原始张量: {x}")
print(f"原始形状: {x.shape}")
print(f"是否连续? {x.is_contiguous()}")

y = x.view(3, 4)
print("\nview(3, 4) 后的张量:")
print(y)
print(f"新形状: {y.shape}")
print(f"与 x 共享存储吗? {y.storage().data_ptr() == x.storage().data_ptr()}")
print(f"y 是否连续? {y.is_contiguous()}")

z = y.view(2, 6)
print("\nview(2, 6) 后的张量:")
print(z)
print(f"新形状: {z.shape}")
print(f"与 x 共享存储吗? {z.storage().data_ptr() == x.storage().data_ptr()}")
print(f"z 是否连续? {z.is_contiguous()}")
```

你可以在 `view()` 调用中对一个维度使用 `-1`，PyTorch 将根据总元素数量和其它维度的尺寸自动推断出该维度的正确尺寸。

```python

w = x.view(2, 2, -1)
print("\nview(2, 2, -1) 后的张量:")
print(w)
print(f"新形状: {w.shape}")
```

如果你尝试在非连续张量上调用 `view()`，你会得到一个 `RuntimeError`。

```python

a = torch.arange(12).view(3, 4)
b = a.t()
print(f"\nb 是否连续? {b.is_contiguous()}")

try:
    c = b.view(12)
except RuntimeError as e:
    print(f"\n尝试 b.view(12) 时出错: {e}")
```

#### 使用 `reshape()`

`reshape()` 方法的行为类似于 `view()`，但提供了更多灵活性。如果张量对于目标形状是连续的，它会尝试返回一个视图。如果无法返回视图（例如，因为原始张量在与新形状兼容的方式上不是连续的），`reshape()` 将把数据*复制*到一个新的、具有所需形状的连续张量中。这使得 `reshape()` 通常更安全、更通用，尽管如果发生复制，性能可能会降低。

我们再次查看使用 `reshape()` 的转置例子：

```python

print(f"\n原始非连续张量 b:\n{b}")
print(f"b 的形状: {b.shape}")
print(f"b 是否连续? {b.is_contiguous()}")

c = b.reshape(12)
print(f"\nb.reshape(12) 后的张量 c:\n{c}")
print(f"c 的形状: {c.shape}")
print(f"c 是否连续? {c.is_contiguous()}")

print(f"与 b 共享存储吗? {c.storage().data_ptr() == b.storage().data_ptr()}")

d = b.reshape(2, -1)
print(f"\nb.reshape(2, -1) 后的张量 d:\n{d}")
print(f"d 的形状: {d.shape}")
```

**何时使用哪个方法？**

- 如果你确定张量是连续的，并且希望确保不发生数据复制以获得最高性能，请使用 `view()`。如果连续性假设有误，请准备好处理可能的 `RuntimeError`。
- `reshape()` 适用于连续和非连续张量。如果可能，它会返回一个视图，否则会创建一个副本。除非性能绝对关键且你能保证连续性，否则这通常是优选方法。

### 使用 `permute()` 调整维度顺序

`view()` 和 `reshape()` 通过重新安排元素在维度间的解释方式来改变形状，而 `permute()` 则明确地交换维度本身。它不改变总元素数量，也不改变每个轴上元素*数量*方面的形状，但它改变的是*哪个*轴对应哪个原始维度。

假设你有一个图像数据，存储为（通道，高，宽）格式，但为了特定的库或可视化需求，需要其格式为（高，宽，通道）。`permute()` 就是为此而设计的工具。你将所需的维度顺序作为参数 (parameter)提供。

```python

image_tensor = torch.randn(3, 32, 32)
print(f"原始形状: {image_tensor.shape}")

permuted_tensor = image_tensor.permute(1, 2, 0)
print(f"调整后的形状: {permuted_tensor.shape}")

print(f"permuted_tensor 是否连续? {permuted_tensor.is_contiguous()}")

original_again = permuted_tensor.permute(2, 0, 1)
print(f"调回后的形状: {original_again.shape}")
print(f"original_again 是否连续? {original_again.is_contiguous()}")

print(f"与原始张量共享存储吗? {original_again.storage().data_ptr() == image_tensor.storage().data_ptr()}")
```

和 `view()` 一样，`permute()` 返回一个与原始张量共享底层数据的张量。它不复制数据。然而，生成的张量通常*不是*连续的。如果你在调换维度后需要一个连续张量（例如，为了后续使用 `view()`），你可以链式调用 `.contiguous()` 方法：

```python

contiguous_permuted = permuted_tensor.contiguous()
print(f"\ncontiguous_permuted 是否连续? {contiguous_permuted.is_contiguous()}")

flattened_permuted = contiguous_permuted.view(-1)
print(f"展平后的形状: {flattened_permuted.shape}")
```

掌握 `view()`、`reshape()` 和 `permute()` 让你能够精确控制张量的结构，这是将数据适配到不同 PyTorch 操作和模型层要求所需的一项必备技能。请记住这些权衡：`view()` 速度快但要求连续性，`reshape()` 灵活但可能会复制，而 `permute()` 交换维度而不复制，但通常会产生非连续张量。

获取即时帮助、个性化解释和交互式代码示例。

---

### 张量的合并与分割

### 张量的合并与分割

许多深度学习 (deep learning)情境下，你需要将多个张量组合成一个，或将一个更大的张量拆分为更小的部分。这可能涉及汇总不同处理步骤的结果、准备数据批次或分离特征。PyTorch 提供了几个函数，用于有效合并和分割张量。

### 合并张量

组合张量是一个常见操作，尤其是在处理数据批次或合并特征表示时。PyTorch 提供了两种主要方式来合并张量：拼接 (`torch.cat`) 和堆叠 (`torch.stack`)。主要区别在于它们是沿着现有维度操作，还是引入一个新维度。

#### 使用 `torch.cat` 进行拼接

`torch.cat` 函数沿着现有维度拼接一系列张量。序列中的所有张量必须形状相同（除了拼接维度），或者为空。

```python
import torch

tensor_a = torch.randn(2, 3)
tensor_b = torch.randn(2, 3)
print(f"Tensor A (Shape: {tensor_a.shape}):\n{tensor_a}")
print(f"Tensor B (Shape: {tensor_b.shape}):\n{tensor_b}\n")

cat_dim0 = torch.cat((tensor_a, tensor_b), dim=0)
print(f"沿着维度0拼接 (形状: {cat_dim0.shape}):\n{cat_dim0}\n")

cat_dim1 = torch.cat((tensor_a, tensor_b), dim=1)
print(f"沿着维度1拼接 (形状: {cat_dim1.shape}):\n{cat_dim1}")

tensor_c = torch.randn(1, 2, 3)
tensor_d = torch.randn(1, 2, 3)

cat_3d_dim0 = torch.cat((tensor_c, tensor_d), dim=0)
print(f"\n3D张量沿着维度0拼接 (形状: {cat_3d_dim0.shape})")
```

请注意，`torch.cat` 增加了指定维度的大小，同时保持其他维度不变。张量在所有维度上都必须大小匹配，*除了*你进行拼接的那个维度。


clusterₐ

张量 A (2x3)

cluster\_b

张量 B (2x3)

cluster\_cat0

torch.cat((A, B), dim=0)
(4x3)

cluster\_cat1

torch.cat((A, B), dim=1)
(2x6)

a1

a11

a12

a13

a2

a21

a22

a23

b1

b11

b12

b13

b2

b21

b22

b23

cat0ₐ1

a11

a12

a13

cat0ₐ2

a21

a22

a23

cat0\_b1

b11

b12

b13

cat0\_b2

b21

b22

b23

cat1ᵣ1

a11

a12

a13

b11

b12

b13

cat1ᵣ2

a21

a22

a23

b21

b22

b23

clusterₐ

clusterₐ

op1

+ 维度0

op2

+ 维度1

cluster\_b

cluster\_b

cluster\_cat0

cluster\_cat0

cluster\_cat1

cluster\_cat1

> `torch.cat` 沿维度0和维度1对两个2x3张量进行拼接的视觉比较。

#### 使用 `torch.stack` 进行堆叠

与 `cat` 不同，`torch.stack` 沿着一个*新*维度连接一系列张量。当你希望从单个示例创建批次或将相关张量分组时，这会很有用。为了 `stack` 能够工作，输入序列中的所有张量必须具有完全相同的形状。

```python
import torch

tensor_e = torch.arange(6).reshape(2, 3)
tensor_f = torch.arange(6, 12).reshape(2, 3)
print(f"Tensor E (Shape: {tensor_e.shape}):\n{tensor_e}")
print(f"Tensor F (Shape: {tensor_f.shape}):\n{tensor_f}\n")

stack_dim0 = torch.stack((tensor_e, tensor_f), dim=0)
print(f"沿着新维度0堆叠 (形状: {stack_dim0.shape}):\n{stack_dim0}\n")

stack_dim1 = torch.stack((tensor_e, tensor_f), dim=1)
print(f"沿着新维度1堆叠 (形状: {stack_dim1.shape}):\n{stack_dim1}\n")

stack_dim2 = torch.stack((tensor_e, tensor_f), dim=2)
print(f"沿着新维度2堆叠 (形状: {stack_dim2.shape}):\n{stack_dim2}")
```


clusterₑ

张量 E (2x3)

cluster\_f

张量 F (2x3)

clusterₛtack0

torch.stack((E, F), dim=0)
(2x2x3)

clusterₛtack0ₑ

切片 0

clusterₛtack0\_f

切片 1

clusterₛtack1

torch.stack((E, F), dim=1)
(2x2x3)

e1

e11

e12

e13

e2

e21

e22

e23

f1

f11

f12

f13

f2

f21

f22

f23

s0ₑ1

e11

e12

e13

s0ₑ2

e21

e22

e23

s0\_f1

f11

f12

f13

s0\_f2

f21

f22

f23

clusterₛtack0ₑ

clusterₛtack0ₑ

clusterₛtack0\_f

clusterₛtack0\_f

s1ᵣ1

e11

e12

e13

f11

f12

f13

s1ᵣ2

e21

e22

e23

f21

f22

f23

clusterₑ

clusterₑ

op1

堆叠 维度0

op2

堆叠 维度1

cluster\_f

cluster\_f

clusterₛtack0

clusterₛtack0

clusterₛtack1

clusterₛtack1

> `torch.stack` 在 `dim=0` 和 `dim=1` 处插入新维度的视觉比较。请注意原始张量如何成为新张量中的切片。

选择 `cat` 还是 `stack` 取决于你是想沿着现有维度合并，还是创建一个新维度。`cat` 通常用于水平/垂直组合批次或特征，`stack` 则常用于从单个样本创建批次。

### 分割张量

正如你可以合并张量一样，你也经常需要将它们分开。这可能涉及将一个批次拆分回单个样本、将特征与标签分离或为并行处理划分数据。PyTorch 为这些任务提供了 `torch.split` 和 `torch.chunk` 函数。

#### 使用 `torch.split` 按特定大小分割

`torch.split` 函数沿着指定维度将张量分割成块。你可以指定每个块的大小（如果你想要等份），或者提供一个包含每个所需块大小的列表。

```python
import torch

tensor_g = torch.arange(12).reshape(6, 2)
print(f"原始张量 (形状: {tensor_g.shape}):\n{tensor_g}\n")

split_equal = torch.split(tensor_g, 2, dim=0)
print("分割成大小为2的等份（dim=0）:")
for i, chunk in enumerate(split_equal):
    print(f" 块 {i} (形状: {chunk.shape}):\n{chunk}")

print("-" * 20)

split_unequal = torch.split(tensor_g, [1, 2, 3], dim=0)
print("\n分割成大小不等的块 [1, 2, 3]（dim=0）:")
for i, chunk in enumerate(split_unequal):
    print(f" 块 {i} (形状: {chunk.shape}):\n{chunk}")

print("-" * 20)

split_dim1 = torch.split(tensor_g, 1, dim=1)
print("\n分割成大小为1的等份（dim=1）:")
for i, chunk in enumerate(split_dim1):

    print(f" 块 {i} (形状: {chunk.shape}):\n{chunk.squeeze()}")
```

`torch.split` 返回一个张量元组。如果你为 `split_size_or_sections` 参数 (parameter)提供一个整数，PyTorch 会沿着指定的 `dim` 将张量分割成该大小的块。如果维度大小不能被分割大小完全整除，最后一个块会更小。如果你提供一个大小列表，它们的总和必须等于被分割维度的大小。

#### 使用 `torch.chunk` 按数量分割

另一种方法是，`torch.chunk` 沿着给定维度将张量分割成指定*数量*的块。PyTorch 会尝试使这些块的大小尽可能相等。与需要指定块大小的 `torch.split` 不同，`chunk` 只需指定所需的块数量。

```python
import torch

tensor_h = torch.arange(10).reshape(5, 2)
print(f"原始张量 (形状: {tensor_h.shape}):\n{tensor_h}\n")

chunked_tensor = torch.chunk(tensor_h, 3, dim=0)
print("分割成3个部分（dim=0）:")
for i, chunk in enumerate(chunked_tensor):
    print(f" 块 {i} (形状: {chunk.shape}):\n{chunk}")

print("-" * 20)

tensor_i = torch.arange(12).reshape(3, 4)
print(f"\n原始张量 (形状: {tensor_i.shape}):\n{tensor_i}\n")

chunked_tensor_dim1 = torch.chunk(tensor_i, 2, dim=1)
print("分割成2个部分（dim=1）:")
for i, chunk in enumerate(chunked_tensor_dim1):
    print(f" 块 {i} (形状: {chunk.shape}):\n{chunk}")
```

当你知道想要多少个部分，而不关心维度大小是否能被均匀整除时，`torch.chunk` 很方便。当你需要大小精确且可能变化的块时，`torch.split` 提供了更多的控制。

掌握这些合并和分割操作很重要，可以帮助你有效处理数据，因为它会流经你的深度学习 (deep learning)管线的不同阶段，从初始加载和预处理，到训练的批处理，以及模型输出的分析。

获取即时帮助、个性化解释和交互式代码示例。

---

### 理解广播机制

### 理解广播机制

当对张量执行逐元素操作（如加法、减法或乘法）时，它们的形状通常需要对齐 (alignment)。但是，手动调整或重复张量以匹配形状可能会很繁琐且效率低下，尤其是在处理大型数据集时。PyTorch 通过一种称为\*\*广播（broadcasting）\*\*的机制解决了这个问题。

广播提供了一套规则，允许 PyTorch 在执行操作时自动扩展张量维度，前提是它们的形状满足特定的兼容标准。这在许多常见情况下省去了显式维度扩展的需要，使得代码更简洁，内存使用更优化，因为实际数据并未重复；只有计算行为像数据重复了一样。

### 广播规则

PyTorch 通过逐元素比较两个张量的形状来判断它们是否“可广播”，比较从*末尾*（最右侧）维度开始。如果满足以下条件，则两个张量可兼容进行广播（从右到左比较每个维度对）：

1. **维度相等：** 维度大小相等。
2. **其中一个维度为 1：** 两个维度中的一个为 1。
3. **缺少维度：** 一个张量不具备该维度（在此比较中，其大小被视为 1）。

如果所有维度对都满足这些条件，则张量是可广播的。结果张量的形状将沿每个维度对取最大尺寸。如果任何维度对不满足条件（即，维度不同且都不为 1），则会引发 `RuntimeError`。

我们来分析一下这个过程：

1. **对齐 (alignment)形状：** 张量根据它们的末尾维度进行对齐。如果一个张量的维度少于另一个，那么为了对齐，会在其形状前面添加大小为 1 的维度。
2. **检查兼容性并确定结果形状：** 从最右侧维度开始，比较尺寸：
   - 如果维度相等，则结果维度大小就是该尺寸。
   - 如果一个维度为 1，则结果维度大小是另一个（较大）维度的大小。
   - 如果一个张量缺少某个维度（由于对齐），则结果维度大小是另一个张量中该维度的大小。
3. **执行操作：** 操作的执行方式，就像是沿着给定维度大小为 1 的张量，其值被复制以匹配另一个张量中对应维度的大小一样。

### 广播示例

我们用代码示例来说明。

#### 标量与张量

将标量（一个 0 维张量）添加到任何张量时，总是通过广播机制生效。标量会有效地扩展以匹配张量的形状。

```python
import torch

a = torch.tensor([[1, 2, 3], [4, 5, 6]])

b = torch.tensor(10)

c = a + b

print(f"Shape of a: {a.shape}")

print(f"Shape of b: {b.shape}")

print(f"Shape of c: {c.shape}")

print(f"Result c:\n{c}")
```

这里，`b`（形状 `[]`）被广播到形状 `[2, 3]` 以匹配 `a`。

#### 行向量 (vector)与矩阵

考虑将一个行向量（形状 `[3]`）添加到一个矩阵（形状 `[2, 3]`）中。

```python

a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

b = torch.tensor([10, 20, 30])

c = a + b

print(f"Shape of a: {a.shape}")
print(f"Shape of b: {b.shape}")
print(f"Shape of c: {c.shape}")
print(f"Result c:\n{c}")
```

- **对齐 (alignment)：** `a` 的形状为 `[2, 3]`。`b` 的形状为 `[3]`。右侧对齐结果如下：

  ```
    张量 A:   2 x 3
    张量 B:       3
  ```
- **兼容性检查：**
  - 末尾维度：`3` 等于 `3`。兼容。结果维度大小为 `3`。
  - 下一个维度：`a` 为 `2`，`b` 在此处没有维度（隐式大小为 `1`）。兼容。结果维度大小为 `2`。
- **结果形状：** `[2, 3]`。
- **扩展：** 张量 `b` 被视为形状 `[1, 3]`，并且其单行沿第一个维度复制以匹配 `a` 的形状 `[2, 3]`。

#### 列向量与矩阵

现在，我们来将一个列向量（形状 `[2, 1]`）添加至同一矩阵（形状 `[2, 3]`）。

```python

a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

b = torch.tensor([[10], [20]])

c = a + b

print(f"Shape of a: {a.shape}")
print(f"Shape of b: {b.shape}")
print(f"Shape of c: {c.shape}")
print(f"Result c:\n{c}")
```

- **对齐：**

  ```
    张量 A:   2 x 3
    张量 B:   2 x 1
  ```
- **兼容性检查：**
  - 末尾维度：`a` 为 `3`，`b` 为 `1`。兼容（其中一个为 1）。结果维度大小为 `3`。
  - 下一个维度：`a` 为 `2`，`b` 为 `2`。兼容（相等）。结果维度大小为 `2`。
- **结果形状：** `[2, 3]`。
- **扩展：** 张量 `b` 中大小为 `1` 的维度（列维度）通过跨列复制值来扩展，以匹配 `a` 的形状 `[2, 3]`。

#### 可视化示例

我们来可视化张量 `A`（形状 `[3, 1]`）和 `B`（形状 `[4]`）的广播过程。


cluster\_A

张量 A (形状: [3, 1])

cluster\_B

张量 B (形状: [4])

cluster\_Broadcast

广播 A + B -> 结果 (形状: [3, 4])


A1

A2

A3


B1

B2

B3

B4

Aₑxpanded

A1, A1, A1, A1

A2, A2, A2, A2

A3, A3, A3, A3

A->Aₑxpanded

扩展维度 1
(大小 1 -> 4)

Bₑxpanded

B1, B2, B3, B4

B1, B2, B3, B4

B1, B2, B3, B4

B->Bₑxpanded

添加维度 0 并扩展
(大小 [4] -> [1,4] -> [3,4])

Result

A1+B1, A1+B2, A1+B3, A1+B4

A2+B1, A2+B2, A2+B3, A2+B4

A3+B1, A3+B2, A3+B3, A3+B4

Aₑxpanded->Result

+

Bₑxpanded->Result

+

> 张量 A (形状 [3, 1]) 和张量 B (形状 [4]) 进行广播加法的示意图。张量 A 的第二个维度（大小 1）扩展到 4。张量 B 获得一个大小为 1 的前置维度（变为形状 [1, 4]），然后扩展到大小 3。两者都有效地变为形状 [3, 4] 以进行逐元素加法。

#### 不兼容的形状

如果非匹配维度不为 1，则广播会失败。

```python

a = torch.tensor([[1, 2, 3], [4, 5, 6]])

b = torch.tensor([10, 20])

try:
    c = a + b
except RuntimeError as e:
    print(f"Error: {e}")
```

- **对齐：**

  ```
    张量 A:   2 x 3
    张量 B:       2
  ```
- **兼容性检查：**
  - 末尾维度：`a` 为 `3`，`b` 为 `2`。两者都不为 1。**不兼容。** 操作失败。

### 常见用途

广播在神经网络 (neural network)中经常使用：

- **添加偏置 (bias)：** 将偏置向量 (vector)（形状 `[output_features]`）添加到线性层的输出（形状 `[batch_size, output_features]`）。
- **归一化 (normalization)：** 从一批数据中减去均值（标量或按特征向量）并除以标准差（标量或按特征向量）。
- **应用掩码：** 将数据与可能具有较少维度的布尔掩码进行逐元素乘法。

理解广播对于编写简洁高效的 PyTorch 代码非常重要。它允许你自然地对不同形状的张量执行操作，只要它们遵守兼容性规则，从而简化了许多常见的数据处理和建模任务。

获取即时帮助、个性化解释和交互式代码示例。

---

### 张量数据类型

### 张量数据类型

张量具有数据类型，通常称为 `dtype`。数据类型决定了张量可以存储的数值种类（例如整数或浮点数）以及每个元素占用多少内存。选择合适的数据类型对于管理计算资源和确保深度学习 (deep learning)模型中的数值精度非常重要。

PyTorch 支持多种数值数据类型，与 NumPy 中的类似。每种类型都有不同的用途，平衡了内存使用、计算速度以及可表示数字的范围或精度。

### 理解 `dtype`

每个张量都有一个 `dtype` 属性，用于指定其元素的类型。默认情况下，PyTorch 创建浮点张量时使用 `torch.float32`，整数张量时使用 `torch.int64`。你可以这样检查张量的数据类型：

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
print(f"Tensor a: {a}")
print(f"dtype of a: {a.dtype}")

b = torch.tensor([1, 2, 3])
print(f"\nTensor b: {b}")
print(f"dtype of b: {b.dtype}")
```

输出：

```
Tensor a: tensor([1., 2., 3.])
dtype of a: torch.float32

Tensor b: tensor([1, 2, 3])
dtype of b: torch.int64
```

在创建张量时，你也可以明确指定 `dtype`：

```python

c = torch.tensor([1.0, 2.0], dtype=torch.float64)
print(f"\nTensor c: {c}")
print(f"dtype of c: {c.dtype}")

d = torch.ones(2, 2, dtype=torch.int32)
print(f"\nTensor d:\n{d}")
print(f"dtype of d: {d.dtype}")
```

输出：

```
Tensor c: tensor([1., 2.], dtype=torch.float64)
dtype of c: torch.float64

Tensor d:
tensor([[1, 1],
        [1, 1]], dtype=torch.int32)
dtype of d: torch.int32
```

你可以使用 `torch.get_default_dtype()` 查看 PyTorch 默认使用的浮点类型。

### 常用数据类型

以下是 PyTorch 中一些最常用的数据类型：

- **浮点类型：**

  - `torch.float32` (或 `torch.float`)：标准的32位单精度浮点数。由于它在CPU和GPU上兼顾了精度和性能，因此是模型参数 (parameter)和一般计算中最常见的类型。
  - `torch.float64` (或 `torch.double`)：64位双精度浮点数。提供更高的精度，但占用两倍内存，并且速度可能明显较慢，尤其是在未针对双精度进行优化的GPU上。当绝对需要高数值精度时使用它。
  - `torch.float16` (或 `torch.half`)：16位半精度浮点数。占用更少内存，并可在现代GPU（如NVIDIA Tensor Cores）上显著加快计算速度。然而，其有限的范围和精度有时可能导致数值不稳定（溢出或下溢）。常用于混合精度训练。
  - `torch.bfloat16`：一种替代的16位格式（脑浮点）。它与 `float32` 具有相似的范围，但精度较低。它正变得越来越受兼容硬件（例如，较新的NVIDIA GPU、Google TPU）上深度学习 (deep learning)训练的欢迎，因为它提供了内存节省和速度提升，同时通常比 `float16` 保持更好的稳定性。
- **整数类型：**

  - `torch.int64` (或 `torch.long`)：64位有符号整数。默认整数类型。常用于张量索引和分类任务中表示类别标签。
  - `torch.int32` (或 `torch.int`)：32位有符号整数。
  - `torch.int16`：16位有符号整数。
  - `torch.int8`：8位有符号整数。较小的整数类型可以节省内存，并对某些操作更快，常用于模型量化 (quantization)。
  - 相应的无符号整数类型也存在 (`torch.uint8`)。
- **布尔类型：**

  - `torch.bool`：表示布尔值 `True` 或 `False`。对于逻辑操作、使用掩码进行索引以及条件逻辑非常重要。

### 类型转换

你经常需要将张量从一种数据类型转换为另一种。这就是所谓的类型转换。转换张量的主要方式是使用 `.to()` 方法，该方法我们在张量在设备（CPU/GPU）之间移动的背景下也见过。

```python
float_tensor = torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32)
print(f"Original tensor: {float_tensor}, dtype: {float_tensor.dtype}")

int_tensor = float_tensor.to(torch.int64)
print(f"Casted to int64: {int_tensor}, dtype: {int_tensor.dtype}")

half_tensor = int_tensor.to(dtype=torch.float16)
print(f"Casted to float16: {half_tensor}, dtype: {half_tensor.dtype}")
```

输出：

```
Original tensor: tensor([1.1000, 2.2000, 3.3000]), dtype: torch.float32
Casted to int64: tensor([1, 2, 3]), dtype: torch.int64
Casted to float16: tensor([1., 2., 3.], dtype=torch.float16), dtype: torch.float16
```

注意，从浮点数转换为整数会截断小数部分。

PyTorch 还提供了便捷方法来进行常见的类型转换：

```python
tensor_a = torch.tensor([0, 1, 0, 1])

tensor_b = tensor_a.float()
print(f"\n.float(): {tensor_b}, dtype: {tensor_b.dtype}")

tensor_c = tensor_b.long()
print(f".long(): {tensor_c}, dtype: {tensor_c.dtype}")

tensor_d = tensor_a.bool()
print(f".bool(): {tensor_d}, dtype: {tensor_d.dtype}")
```

输出：

```
.float(): tensor([0., 1., 0., 1.]), dtype: torch.float32
.long(): tensor([0, 1, 0, 1]), dtype: torch.int64
.bool(): tensor([False,  True, False,  True]), dtype: torch.bool
```

请记住，类型转换通常会在内存中创建一个具有指定数据类型的*新*张量，而不是就地修改原始张量。

### 操作中的类型提升

当你对不同数据类型的张量执行操作时，PyTorch 通常会自动提升类型以确保兼容性。一般规则是，整数类型与浮点类型进行操作时，结果将是浮点类型。不同浮点类型之间的操作通常会得到更高精度的类型。

```python
int_t = torch.tensor([1, 2], dtype=torch.int32)
float_t = torch.tensor([0.5, 0.5], dtype=torch.float32)
double_t = torch.tensor([0.1, 0.1], dtype=torch.float64)

result1 = int_t + float_t
print(f"\nint32 + float32 = {result1}, dtype: {result1.dtype}")

result2 = float_t + double_t
print(f"float32 + float64 = {result2}, dtype: {result2.dtype}")
```

输出：

```
int32 + float32 = tensor([1.5000, 2.5000]), dtype: torch.float32
float32 + float64 = tensor([0.6000, 0.6000], dtype=torch.float64), dtype: torch.float64
```

虽然方便，但要注意自动类型提升，因为它如果未被预料到，可能会导致意外结果或性能影响。使用 `.to()` 进行显式类型转换可以让你对计算中所需的数据类型有更清晰的控制。

理解和管理张量数据类型是高效 PyTorch 编程的重要组成部分。它能让你控制内存占用，运用硬件加速（如GPU上的FP16），并为你的特定深度学习 (deep learning)任务保持必要的数值精度。

获取即时帮助、个性化解释和交互式代码示例。

---

### CPU 与 GPU 张量

### CPU 与 GPU 张量

深度学习 (deep learning)计算，特别是涉及大型张量和复杂模型的计算，需要强大的计算能力。中央处理器（CPU）用途广泛，而图形处理器（GPU）提供大规模并行能力，可以大幅加速构成神经网络 (neural network)核心的矩阵和向量 (vector)运算。PyTorch 提供简单直接的机制来管理张量的存放位置和计算发生地。了解如何在 CPU 和 GPU 之间移动张量是高效模型训练和推理 (inference)的必备技能。

### CPU：默认计算中心

默认情况下，当您创建 PyTorch 张量时未指定设备，它会分配在 CPU 上。

```python
import torch

cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
print(f"默认张量设备：{cpu_tensor.device}")
```

CPU 完全适用于许多任务，包括预处理步骤、较小规模的计算，或在没有兼容 GPU 时运行模型。然而，对于训练大型深度学习 (deep learning)模型，由于 CPU 的顺序处理特性与 GPU 的并行架构相比，仅依赖 CPU 常常会导致训练时间过长，难以承受。

### GPU：使用并行加速深度学习 (deep learning)

GPU 包含数百或数千个核心，旨在同时执行大量计算。这种架构非常适合深度学习中常见的运算类型，如大型矩阵乘法和卷积。PyTorch 通过 CUDA（统一计算设备架构）平台使用 NVIDIA GPU。

要使用 GPU，您需要：

1. 一个兼容 CUDA 的 NVIDIA GPU。
2. 已安装适当的 CUDA 工具包。
3. 安装了 CUDA 支持的 PyTorch 版本。

### 检查 GPU 可用性并设置设备

在尝试使用 GPU 之前，最好检查它是否可用且已为 PyTorch 正确配置。`torch.cuda.is_available()` 函数在 PyTorch 可以访问支持 CUDA 的 GPU 时返回 `True`。

之后我们可以创建一个 `torch.device` 对象来表示我们的目标计算设备（CPU 或 GPU）。这使得代码具有适应性，如果 GPU 可用则自动使用 GPU，否则回退到 CPU。

```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA (GPU) 可用。使用设备：{device}")

else:
    device = torch.device("cpu")
    print(f"CUDA (GPU) 不可用。使用设备：{device}")
```

### 直接在设备上创建张量

您可以在张量创建期间使用 `device` 参数 (parameter)直接指定目标设备。这通常比在 CPU 上创建然后再移动更高效。

```python

try:

    device_tensor = torch.randn(3, 4, device=device)
    print(f"张量创建于：{device_tensor.device}")
except RuntimeError as e:
    print(f"无法直接在 {device} 上创建张量：{e}")
```

### 在 CPU 和 GPU 之间移动张量

通常，您需要在设备之间传输现有张量。例如，从磁盘加载的数据通常位于 CPU 上，但您的模型可能在 GPU 上以加快计算。移动张量的主要方法是 `.to()` 方法。

`.to()` 方法接受 `torch.device` 对象、设备字符串（例如 `'cuda'`、`'cpu'`），甚至另一个张量（在这种情况下，张量会移动到与参数 (parameter)张量相同的设备上）作为输入。它在指定设备上返回一个 *新* 张量。原始张量在其原始设备上保持不变。

```python

cpu_tensor = torch.ones(2, 2)
print(f"原始张量：{cpu_tensor.device}")

moved_tensor = cpu_tensor.to(device)
print(f"移动后的张量：{moved_tensor.device}")

if moved_tensor.is_cuda:
    back_to_cpu = moved_tensor.to("cpu")
    print(f"张量移回至：{back_to_cpu.device}")
```

PyTorch 还提供便利方法：`.cpu()` 和 `.cuda()`。它们分别是 `.to('cpu')` 和 `.to('cuda:0')`（或当前默认的 CUDA 设备）的简写。

```python

if device.type == 'cuda':

    gpu_tensor_alt = cpu_tensor.cuda()
    print(f"使用 .cuda()：{gpu_tensor_alt.device}")

    cpu_tensor_alt = gpu_tensor_alt.cpu()
    print(f"使用 .cpu()：{cpu_tensor_alt.device}")
```

### 设备管理的重要考量

1. **设备一致性：** 涉及多个张量的操作（例如，加法、矩阵乘法）通常要求所有参与的张量都在 *同一* 设备上。尝试在 CPU 张量和 GPU 张量之间进行操作将导致 `RuntimeError`。在执行操作前，请确保您的数据和模型位于同一设备上。

   ```python

   cpu_a = torch.randn(2, 2)
   gpu_b = torch.randn(2, 2, device=device)
   try:

       c = cpu_a + gpu_b
   except RuntimeError as e:
       print(f"在不同设备上执行操作时出错：{e}")
   ```
2. **数据传输开销：** 在 CPU 和 GPU 内存之间移动数据并非瞬间完成。虽然 GPU 计算速度快，但如果管理不当，数据的来回传输可能成为瓶颈。为了最佳性能，请尝试在 GPU 上尽可能多地执行操作，然后在需要时（例如，保存到磁盘或转换为 NumPy）再将最终结果移回 CPU。
3. **模型放置：** 与张量一样，使用 `torch.nn.Module` 定义的神经网络 (neural network)模型也需要使用 `.to(device)` 方法移动到适当的设备。这确保模型的参数 (parameter)（本身也是张量）位于目标设备上进行计算。在讨论模型构建时，会更详细地介绍这一点。

掌握使用 `device` 和 `.to()` 方法进行张量放置是发挥 GPU 计算能力和编写高效、硬件感知型 PyTorch 代码不可或缺的要素。请记住始终检查设备一致性并注意数据传输成本。

获取即时帮助、个性化解释和交互式代码示例。

---

### 练习：张量操作技巧

### 练习：张量操作技巧

练习张量操作技术，包括索引、形状改变、组合、广播、数据类型以及张量在不同设备间的移动。通过这些示例进行练习是巩固理解的最佳途径。请确保已导入PyTorch。

```python
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")
```

### 索引和切片练习

索引和切片是访问和修改张量部分内容的基本操作。让我们尝试选择特定的数据点。

**任务1：** 创建一个二维张量，并选择第二行第三列的元素。

```python

data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
tensor_2d = torch.tensor(data)
print("原始张量:\n", tensor_2d)

element = tensor_2d[1, 2]
print("\n在 [1, 2] 的元素:", element)
print("值:", element.item())
```

**任务2：** 选择 `tensor_2d` 的整个第二行。

```python

row_1 = tensor_2d[1]
print("\n第二行（索引1）:\n", row_1)

row_1_alt = tensor_2d[1, :]
print("\n第二行（替代方法）:\n", row_1_alt)
```

**任务3：** 选择 `tensor_2d` 的第三列。

```python

col_2 = tensor_2d[:, 2]
print("\n第三列（索引2）:\n", col_2)
```

**任务4：** 创建一个布尔掩码以选择 `tensor_2d` 中所有大于7的元素，然后使用该掩码提取这些元素。

```python

mask = tensor_2d > 7
print("\n布尔掩码（张量 > 7）:\n", mask)

selected_elements = tensor_2d[mask]
print("\n大于7的元素:\n", selected_elements)
```

这些练习说明了标准Python索引如何结合类似NumPy的切片和布尔掩码，以提供灵活的数据访问方式。

### 形状改变与重排练习

在不改变张量数据的情况下改变其形状是很常见的，尤其是在为不同神经网络 (neural network)层准备输入时。

**任务1：** 创建一个包含12个元素的一维张量，并将其形状改为3x4的张量。

```python
tensor_1d = torch.arange(12)
print("\n原始一维张量:", tensor_1d)

reshaped_tensor = tensor_1d.reshape(3, 4)
print("\n改变形状为3x4:\n", reshaped_tensor)

view_tensor = tensor_1d.view(3, 4)
print("\n视为3x4:\n", view_tensor)
```

请记住，`view` 要求张量在内存中是连续的，并共享底层数据。`reshape` 可能会返回一个副本或视图，具体取决于连续性。

**任务2：** 给定 `reshaped_tensor`（3x4），使用 `permute` 交换其维度，得到一个4x3的张量。

```python

print("\n原始3x4张量:\n", reshaped_tensor)

permuted_tensor = reshaped_tensor.permute(1, 0)
print("\n置换为4x3:\n", permuted_tensor)
print("原始形状:", reshaped_tensor.shape)
print("置换后形状:", permuted_tensor.shape)
```

`permute` 在改变图像维度顺序等任务中有用（例如，从通道数 x 高度 x 宽度 变为 高度 x 宽度 x 通道数）。

### 连接与拆分练习

合并或拆分张量通常是必要的，尤其是在处理批次或不同特征集时。

**任务1：** 创建两个2x3张量，并沿维度0（行）连接它们。

```python
tensor_a = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor_b = torch.tensor([[7, 8, 9], [10, 11, 12]])
print("\n张量A:\n", tensor_a)
print("张量B:\n", tensor_b)

concatenated_rows = torch.cat((tensor_a, tensor_b), dim=0)
print("\n沿行连接（dim=0）:\n", concatenated_rows)
print("形状:", concatenated_rows.shape)
```

**任务2：** 沿维度1（列）连接 `tensor_a` 和 `tensor_b`。

```python

concatenated_cols = torch.cat((tensor_a, tensor_b), dim=1)
print("\n沿列连接（dim=1）:\n", concatenated_cols)
print("形状:", concatenated_cols.shape)
```

**任务3：** 使用 `stack` 组合 `tensor_a` 和 `tensor_b`，形成一个形状为2x2x3的新张量。

```python

stacked_tensor = torch.stack((tensor_a, tensor_b), dim=0)
print("\n堆叠的张量（dim=0）:\n", stacked_tensor)
print("形状:", stacked_tensor.shape)

stacked_tensor_dim1 = torch.stack((tensor_a, tensor_b), dim=1)
print("\n堆叠的张量（dim=1）:\n", stacked_tensor_dim1)
print("形状:", stacked_tensor_dim1.shape)
```

请注意 `stack` 如何添加一个新维度，而 `cat` 沿着现有维度连接。

**任务4：** 创建一个6x4张量，并沿维度0将其拆分为三个等大小的部分。

```python
tensor_to_split = torch.arange(24).reshape(6, 4)
print("\n待拆分张量（6x4）:\n", tensor_to_split)

chunks = torch.chunk(tensor_to_split, chunks=3, dim=0)
print("\n拆分为3个部分:")
for i, chunk in enumerate(chunks):
    print(f"部分 {i}（形状 {chunk.shape}）:\n", chunk)
```

### 广播练习

广播简化了不同形状张量之间的操作。

**任务1：** 创建一个3x3张量和一个1x3张量（行向量 (vector)）。将它们相加。

```python
matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_vector = torch.tensor([[10, 20, 30]])
print("\n矩阵（3x3）:\n", matrix)
print("行向量（1x3）:\n", row_vector)

result = matrix + row_vector
print("\n矩阵 + 行向量（广播）:\n", result)
```

PyTorch 自动扩展了 `row_vector`（形状1x3）的行，使其形状变为3x3，从而允许与 `matrix` 进行逐元素相加。

**任务2：** 创建一个3x3张量和一个3x1张量（列向量）。将它们相加。

```python
col_vector = torch.tensor([[100], [200], [300]])
print("\n矩阵（3x3）:\n", matrix)
print("列向量（3x1）:\n", col_vector)

result_col = matrix + col_vector
print("\n矩阵 + 列向量（广播）:\n", result_col)
```

这里，`col_vector`（形状3x1）被广播到各列，以匹配 `matrix` 的3x3形状。

### 数据类型练习

管理数据类型对于内存效率和数值稳定性很重要。

**任务1：** 创建一个整数张量并检查其 `dtype`。然后将其转换为浮点张量。

```python
int_tensor = torch.tensor([1, 2, 3, 4])
print("\n整数张量:", int_tensor)
print("数据类型:", int_tensor.dtype)

float_tensor = int_tensor.to(torch.float32)

print("\n转换为浮点张量:", float_tensor)
print("数据类型:", float_tensor.dtype)
```

**任务2：** 创建一个浮点张量并将其转换为整数张量。观察任何变化。

```python
float_tensor_orig = torch.tensor([1.1, 2.7, 3.5, 4.9])
print("\n原始浮点张量:", float_tensor_orig)
print("数据类型:", float_tensor_orig.dtype)

int_tensor_cast = float_tensor_orig.to(torch.int32)

print("\n转换为整数张量:", int_tensor_cast)
print("数据类型:", int_tensor_cast.dtype)
```

请注意，从浮点数转换为整数会截断小数部分。请注意可能的数据精度损失。

### CPU 与 GPU 练习

将张量移动到合适的设备（CPU或GPU）是必要的，以发挥硬件加速的优势。

**任务1：** 创建一个张量并检查其默认设备。然后，将其移动到GPU（如果可用），再移回CPU。

```python

cpu_tensor = torch.randn(2, 2)
print(f"\n张量在CPU上: {cpu_tensor.device}\n", cpu_tensor)

device_tensor = cpu_tensor.to(device)
print(f"\n张量已移动到 {device_tensor.device}:
", device_tensor)

cpu_tensor_again = device_tensor.to("cpu")
print(f"\n张量已移回CPU: {cpu_tensor_again.device}
", cpu_tensor_again)

if device_tensor.device != cpu_tensor.device:
    print("\n在不同设备上的张量相加会导致错误。")

    result_on_device = device_tensor + device_tensor
    print(f"在 {result_on_device.device} 上的操作结果:
", result_on_device)
else:
    print("\n两个张量都在CPU上，相加没问题。")
    result_on_cpu = cpu_tensor + cpu_tensor_again
    print(f"在 {result_on_cpu.device} 上的操作结果:
", result_on_cpu)
```

请记住，张量之间的操作通常要求它们在同一设备上进行。使用 `.to(device)` 显式移动张量是 PyTorch 代码中常见的做法，尤其是在为GPU训练准备数据和模型时。

本次实践涵盖了操作张量的主要技巧。随着您构建更复杂的模型，熟练掌握索引、形状改变、组合张量、理解广播、管理数据类型以及控制设备放置将变得愈发重要。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 3 Automatic Differentiation Autograd

### 自动微分的原理

### 自动微分的原理

训练神经网络 (neural network)包括迭代地调整模型参数 (parameter)（权重 (weight)和偏置 (bias)）以最小化损失函数 (loss function)。这种调整依赖于知晓每个参数的微小变化如何影响最终的损失值。从数学上讲，这种敏感性由损失函数对每个参数的梯度来体现。对于损失 LLL 和参数 www，我们需要计算 ∂L∂w\frac{\partial L}{\partial w}∂w∂L​。

对于非常简单的模型，手动使用微积分规则计算这些梯度是可行的，但对于当今常见的深度多层网络来说，这很快就会变得异常复杂且容易出错。想象一下为拥有数百万参数的模型推导导数！这时，\*\*自动微分（AD）\*\*就派上用场了。

AD 是一系列以数值方式评估由计算机程序定义的函数导数的方法。与*符号微分*（它操作数学表达式，常导致复杂且低效的公式）或*数值微分*（它使用有限差分近似导数，可能存在截断和舍入误差）不同，AD 通过在构成整体计算的基本运算（加法、乘法、三角函数等）层面系统地应用微积分的链式法则，高效地计算出精确的梯度。

### 链式法则：AD 的核心

其核心是，AD 依赖于链式法则。如果有一系列函数，例如 y=f(x)y = f(x)y=f(x) 和 z=g(y)z = g(y)z=g(y)，链式法则告诉我们如何找到复合函数 z=g(f(x))z = g(f(x))z=g(f(x)) 对 xxx 的导数：

dzdx=dzdy⋅dydx\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}dxdz​=dydz​⋅dxdy​

AD 将复杂的计算分解为一系列基本运算。然后，它计算每个小步骤的局部导数，并使用链式法则将它们组合起来以得到整体梯度。

考虑一个简单例子：L=(w⋅x+b)2L = (w \cdot x + b)^2L=(w⋅x+b)2。令 y=w⋅x+by = w \cdot x + by=w⋅x+b。则 L=y2L = y^2L=y2。
要找到 ∂L∂w\frac{\partial L}{\partial w}∂w∂L​，链式法则给出：

∂L∂w=∂L∂y⋅∂y∂w\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}∂w∂L​=∂y∂L​⋅∂w∂y​

我们知道 ∂L∂y=2y\frac{\partial L}{\partial y} = 2y∂y∂L​=2y 和 ∂y∂w=x\frac{\partial y}{\partial w} = x∂w∂y​=x。将 yyy 代回，我们得到：

∂L∂w=(2(w⋅x+b))⋅x\frac{\partial L}{\partial w} = (2(w \cdot x + b)) \cdot x∂w∂L​=(2(w⋅x+b))⋅x

AD 自动完成这个过程，即使运算链非常长。

### 正向模式与反向模式

在 AD 中应用链式法则主要有两种方式：

1. **正向模式（正向累积）：** 通过从输入到输出遍历计算步骤来计算导数。它计算*一个*输入的改变如何影响*所有*中间变量和最终输出。当输入数量相对于输出数量较少时，它比较高效。
2. **反向模式（反向累积）：** 通过从最终输出到输入反向遍历计算步骤来计算导数。它计算*最终输出*的改变如何受到*所有*中间变量和输入的影响。当输出数量相对于输入数量较少时，这种模式明显更高效，这正是深度学习 (deep learning)中的情况，因为我们通常只有一个标量损失值和数百万个参数 (parameter)（损失函数 (loss function)的输入）。

PyTorch 的 Autograd 系统使用**反向模式自动微分**。

### Autograd 如何使用 AD

当您对 `requires_grad` 属性设置为 `True` 的 PyTorch 张量执行操作时，PyTorch 会在后台构建一个有向无环图（DAG）。这个图通常被称为计算图，它记录了操作序列（节点）和涉及的张量（边）。

让我们直观地看一下 L=(a⋅x+b)2L = (a \cdot x + b)^2L=(a⋅x+b)2 的简单计算图，假设 aaa、xxx 和 bbb 是输入张量（或之前计算的结果），并且我们想得到 ∂L∂a\frac{\partial L}{\partial a}∂a∂L​、∂L∂x\frac{\partial L}{\partial x}∂x∂L​ 和 ∂L∂b\frac{\partial L}{\partial b}∂b∂L​。


clusterᵢnputs

输入

clusterₒps

运算

a

a

mul

\*

a->mul

x

x

x->mul

b

b

add

+

b->add

mul->a

dL/da

mul->x

dL/dx

mul->add

add->b

dL/db

add->mul

dL/d(a\*x)

pow

²

add->pow

pow->add

dL/dy


L (损失)

pow->L

L->pow

dL/dL=1

> L=(a⋅x+b)2L = (a \cdot x + b)^2L=(a⋅x+b)2 计算图的表示。实线表示正向传播，构建图。虚线表示反向传播 (backpropagation)期间梯度的流动，应用链式法则。

当您在最终输出张量（通常是标量损失 LLL）上调用 `.backward()` 时，Autograd 会从该输出开始并反向遍历图。在每个步骤（节点），它根据后续节点的梯度和当前节点执行运算的局部导数来计算梯度，有效地应用了链式法则。然后，对每个需要梯度的张量（如模型参数 (parameter)）计算出的梯度会累积到它们的 `.grad` 属性中。

这种机制使 PyTorch 能够自动计算由张量运算序列定义的任意复杂模型的梯度，让您摆脱手动推导的繁琐且容易出错的任务。接下来的章节将演示如何实际使用 Autograd 的功能：定义需要梯度的张量、隐式构建计算图、触发反向传播、访问梯度以及控制梯度计算。

获取即时帮助、个性化解释和交互式代码示例。

---

### PyTorch 计算图

### PyTorch 计算图

PyTorch 运用**计算图**作为其通过 Autograd 自动计算梯度的底层机制。每次你执行涉及需要梯度的张量运算时（你很快会了解如何指定这一点），PyTorch 都会动态构建一个图，表示计算序列。

可以将此图视为一个有向无环图 (DAG)，其中：

- **节点**代表张量或对其执行的运算。
- **边**代表数据（张量）的流动以及运算和张量之间的函数依赖关系。

考虑一个简单的计算：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y = w * x
z = y + b

print(f"Result z: {z}")
```

输出：

```
Result z: 7.0
```

当这些代码行执行时，PyTorch 会在后台构建一个图。它看起来像这样：


x

x (数据=2.0, requires\_grad=True)

mulₒp

\*

x->mulₒp

w

w (数据=3.0, requires\_grad=True)

w->mulₒp

b

b (数据=1.0, requires\_grad=True)

addₒp

+

b->addₒp

y

y (w\*x 的结果)

mulₒp->y

grad\_fn=<MulBackward0>

z

z (y+b 的结果)

addₒp->z

grad\_fn=<AddBackward0>

y->addₒp

> 这是 z=(w∗x)+bz = (w \* x) + bz=(w∗x)+b 的计算图表示。蓝色框代表输入张量，黄色椭圆代表运算，绿色框代表输出/中间张量。边表示数据流和运算间的依赖关系。由运算产生的张量上的 `grad_fn` 属性指向创建它们的函数。

### 动态特性

PyTorch 计算图的一个重要特点是它们的**动态**特性。与那些要求你在运行计算前定义整个图结构的框架不同，PyTorch 会在你的 Python 代码执行时动态构建图。

- **灵活性：** 这允许标准的 Python 控制流语句（如 `if` 条件或 `for` 循环）直接影响每次迭代中的图结构。如果你的模型架构需要在正向传播期间根据输入数据进行更改，PyTorch 会很自然地处理这种情况。
- **调试：** 动态图通常更容易使用标准 Python 工具进行调试，因为图的构建与你熟悉的程序执行同时进行。

### 正向传播和反向传播 (backpropagation)

1. **正向传播：** 当你执行张量运算（如 `y = w * x`）时，你正在进行正向传播。PyTorch 会记录所涉及的运算和张量，从而构建图。由 Autograd 追踪的运算产生的张量将具有 `grad_fn` 属性（如示例中的 `y` 和 `z`）。此属性引用了创建该张量的函数，并保存了对其输入的引用，从而形成了图中的反向链接。用户创建的张量（如 `x`、`w`、`b`）通常具有 `grad_fn=None`。
2. **反向传播：** 当你稍后在标量张量（通常是最终的损失值）上调用 `.backward()` 时，Autograd 会从该节点向后遍历此图。它使用微积分的链式法则，在每一步都由 `grad_fn` 指引，以计算标量输出相对于最初标记 (token)为 `requires_grad=True` 的张量（通常是模型参数 (parameter)或输入）的梯度。

### 叶张量和梯度

在 Autograd 中：

- **叶张量：** 这些是位于图“开始”处的张量。通常，它们是用户直接创建的张量（例如，使用 `torch.tensor()`、`torch.randn()`）且 `requires_grad=True`。模型参数 (parameter)（`nn.Parameter`，我们稍后会看到）也是叶张量。
- **非叶张量（中间张量）：** 这些是在图内运算后产生的张量（如上文的 `y` 和 `z`）。它们关联着 `grad_fn`。

默认情况下，在 `.backward()` 调用期间计算的梯度仅保留并累积在具有 `requires_grad=True` 的**叶张量**的 `.grad` 属性中。中间张量的梯度会进行计算，但使用后通常会被丢弃以节省内存，除非另有明确请求（例如，使用 `.retain_grad()`）。

理解这种图结构对于掌握 Autograd 的工作方式非常重要。它将你在模型中定义的正向计算与优化所需的梯度计算直接关联起来。接下来，我们将研究如何使用 `requires_grad` 来明确控制梯度跟踪。

获取即时帮助、个性化解释和交互式代码示例。

---

### 张量与梯度计算 (requires_grad)

### 张量与梯度计算 (`requires_grad`)

神经网络 (neural network)训练的根基在于计算损失函数 (loss function)相对于模型参数 (parameter)的梯度。PyTorch 的 Autograd 引擎自动处理这项复杂的任务。但是 Autograd 怎么知道*哪些*计算需要被追踪以便进行微分呢？答案在于 PyTorch 张量的一个特定属性：`requires_grad`。

### `requires_grad` 属性

每个 PyTorch 张量都有一个布尔属性，名为 `requires_grad`。此属性充当一个标志，告诉 Autograd 是否应记录涉及此张量的操作，以便稍后进行可能的梯度计算。

默认情况下，当你创建一个张量时，它的 `requires_grad` 属性被设置为 `False`。

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
print(f"Tensor x: {x}")
print(f"x.requires_grad: {x.requires_grad}")

y = torch.tensor([4.0, 5.0, 6.0], requires_grad=False)
print(f"\nTensor y: {y}")
print(f"y.requires_grad: {y.requires_grad}")
```

这种默认行为对效率来说是合理的。在典型的工作流程中，许多张量不需要梯度。例如，输入数据或目标标签通常是固定的，不需要计算相对于它们自身的梯度。不必要地追踪操作会消耗额外的内存和计算资源。

### 启用梯度追踪

要指示 PyTorch 追踪某个特定张量的操作并准备进行梯度计算，你需要将其 `requires_grad` 属性设置为 `True`。有两种主要方式可以做到这一点：

1. **在张量创建时：** 将 `requires_grad=True` 作为参数 (parameter)传递给张量创建函数。

   ```python

   w = torch.tensor([0.5, -1.0], requires_grad=True)
   print(f"Tensor w: {w}")
   print(f"w.requires_grad: {w.requires_grad}")
   ```
2. **在张量创建后（原地修改）：** 对现有张量使用原地方法 `.requires_grad_(True)`。

   ```python
   b = torch.tensor([0.1])
   print(f"Tensor b (before): {b}")
   print(f"b.requires_grad (before): {b.requires_grad}")

   b.requires_grad_(True)
   print(f"\nTensor b (after): {b}")
   print(f"b.requires_grad (after): {b.requires_grad}")
   ```

**重要提示：** 梯度计算通常只对浮点张量（如 `torch.float32` 或 `torch.float64`）有意义。导数涉及连续变化，这与浮点类型相符。尝试对整数张量设置 `requires_grad=True` 通常会导致错误或出现意料之外的行为，因为梯度并非以相同方式为离散值定义的。如果你尝试计算直接涉及被追踪操作的整数张量的梯度，PyTorch 通常会抛出 `RuntimeError`。

```python

try:
    int_tensor = torch.tensor([1, 2], dtype=torch.int64, requires_grad=True)

    print(f"Integer tensor created with requires_grad=True: {int_tensor.requires_grad}")

    result = int_tensor * 2.0
    print(f"Result requires_grad: {result.requires_grad}")

except RuntimeError as e:
    print(f"\n对整数张量设置 requires_grad 时出错: {e}")

float_tensor = torch.tensor([1.0, 2.0], requires_grad=True)
print(f"\n已创建 requires_grad=True 的浮点张量: {float_tensor.requires_grad}")
```

### `requires_grad` 的传播

重要的一点是，`requires_grad` 状态会在操作中传播。如果参与操作的*任何*输入张量具有 `requires_grad=True`，则该操作产生的输出张量将自动具有 `requires_grad=True`。这确保了涉及参数 (parameter)（通常具有 `requires_grad=True`）的整个计算链都得到追踪。

让我们通过一个例子说明：

```python

x = torch.tensor([1.0, 2.0])
w = torch.tensor([0.5, -1.0], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)

print(f"x requires_grad: {x.requires_grad}")
print(f"w requires_grad: {w.requires_grad}")
print(f"b requires_grad: {b.requires_grad}")

intermediate = w * x
print(f"\nintermediate (w * x) requires_grad: {intermediate.requires_grad}")

y = intermediate + b
print(f"y requires_grad: {y.requires_grad}")
```

注意，即使 `x` 不需要梯度，但由于 `w` 需要梯度，所以 `w * x` 的结果 (`intermediate`) 也需要梯度。接着，由于 `intermediate` 需要梯度（并且 `b` 也需要），最终输出 `y` 也具有 `requires_grad=True`。

### `.grad_fn` 属性

这种传播与 PyTorch 构建计算图的方式紧密关联。当一个新张量由某个操作创建，并且其 `requires_grad` 为 `True` 时，PyTorch 会将一个 `.grad_fn` 属性附加到这个新张量上。该属性引用了执行此操作的函数（例如 `AddBackward0` 或 `MulBackward0`），并且知道如何在反向传播 (backpropagation)过程中计算相应的梯度。

用户直接创建的张量（如我们上面的 `x`、`w` 和 `b` 示例）在图中被认为是“叶”张量。如果它们具有 `requires_grad=True`，它们的 `.grad_fn` 为 `None`，因为它们不是由图中被追踪的操作创建的。对需要梯度的张量进行操作所产生的张量是“非叶”张量，并将具有 `.grad_fn`。

让我们查看前面示例中的 `.grad_fn`：

```python
print(f"\nx.grad_fn: {x.grad_fn}")
print(f"w.grad_fn: {w.grad_fn}")
print(f"b.grad_fn: {b.grad_fn}")
print(f"intermediate.grad_fn: {intermediate.grad_fn}")
print(f"y.grad_fn: {y.grad_fn}")
```

你可以看到 `x`、`w` 和 `b`（我们的叶张量）的 `grad_fn` 为 `None`。相比之下，`intermediate` 有一个 `MulBackward0` 函数，而 `y` 有一个 `AddBackward0` 函数，这表明了创建它们的那些操作。这条 `grad_fn` 引用链*就是*Autograd 使用的动态计算图。


clusterᵢnputs

叶张量

clusterₒps

操作与结果

x

x
requires\_grad=False
grad\_fn=None

mulₒp

\*

x->mulₒp

输入

w

w
requires\_grad=True
grad\_fn=None

w->mulₒp

输入

b

b
requires\_grad=True
grad\_fn=None

addₒp

+

b->addₒp

输入

intermediate

intermediate
requires\_grad=True
grad\_fn=<MulBackward0>

mulₒp->intermediate

输出

y

y
requires\_grad=True
grad\_fn=<AddBackward0>

addₒp->y

输出

intermediate->addₒp

输入

> `y = w * x + b` 的计算图简化视图。需要梯度的张量用蓝色突出显示。注意操作符（`*`、`+`）如何创建新张量（`intermediate`、`y`），如果通过其输入启用了梯度追踪，这些新张量将通过 `grad_fn` 引用其创建操作。

通过对我们希望优化的张量（通常是模型参数 (parameter)，如权重 (weight) `w` 和偏置 (bias) `b`）设置 `requires_grad=True`，我们让 Autograd 能够构建此图，并将计算从最终输出（通常是损失）追溯到这些参数，为使用 `.backward()` 进行梯度计算的步骤做好准备，我们将在接下来介绍这一点。

获取即时帮助、个性化解释和交互式代码示例。

---

### 执行反向传播 (backward())

### 执行反向传播 (`backward()`)

在使用 PyTorch 时，张量会被设置，并且需要梯度的张量会被标记 (token)为 `requires_grad=True`。PyTorch 会在其动态计算图中忠实地跟踪这些张量上的操作。为了计算这些梯度，`backward()` 方法是必不可少的。

`backward()` 方法是驱动 PyTorch 自动微分的引擎。当在一个张量上调用它时，通常是模型最终的标量损失值，它会启动使用链式法则在整个计算图中计算梯度。它会计算被调用的张量相对于图中所有 `requires_grad=True` 的“叶”张量的梯度（这些通常是你的模型参数 (parameter)或你需要梯度的初始输入）。

### 启动梯度计算

你几乎总是在一个标量张量上调用 `backward()`，这通常是你的损失函数 (loss function)计算的结果。例如，如果 `loss` 包含代表模型批次误差的单个数值：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y = w * x + b
loss = y * y

print(f"Gradient for x before backward: {x.grad}")
print(f"Gradient for w before backward: {w.grad}")
print(f"Gradient for b before backward: {b.grad}")

loss.backward()

print(f"Gradient for x after backward: {x.grad}")
print(f"Gradient for w after backward: {w.grad}")
print(f"Gradient for b after backward: {b.grad}")
```

输出：

```text
Gradient for x before backward: None
Gradient for w before backward: None
Gradient for b before backward: None
Gradient for x after backward: 42.0
Gradient for w after backward: 28.0
Gradient for b after backward: 14.0
```

如你所见，调用 `loss.backward()` 计算了梯度 ∂loss∂x\frac{\partial \text{loss}}{\partial x}∂x∂loss​、∂loss∂w\frac{\partial \text{loss}}{\partial w}∂w∂loss​ 和 ∂loss∂b\frac{\partial \text{loss}}{\partial b}∂b∂loss​ 并将它们存储在 `x`、`w` 和 `b` 张量各自的 `.grad` 属性中。

### 为什么在标量上调用 `.backward()`？

Autograd 被设计用于计算雅可比向量 (vector)积 (JVP)。当你在一个标量张量 LLL 上调用 `backward()` 时，它隐式地等同于以 1.01.01.0 的起始梯度调用 `backward()`。这使得 PyTorch 可以使用链式法则，从标量损失向后传播，高效地计算所有参数 (parameter) p\mathbf{p}p 的梯度 ∂L∂p\frac{\partial L}{\partial \mathbf{p}}∂p∂L​。

如果你尝试在一个**非标量张量**（即包含多个元素的张量）上调用 `.backward()`，PyTorch 无法隐式知道如何根据最终（未显式）的标量损失来为该张量中每个元素的梯度加权。你将收到一个运行时错误，要求提供 `gradient` 参数：

```python

x_vector = torch.tensor([2.0, 4.0], requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y_non_scalar = w * x_vector + b

try:
    y_non_scalar.backward()
except RuntimeError as e:
    print(f"Error calling backward() on non-scalar: {e}")

grad_tensor = torch.ones_like(y_non_scalar)
y_non_scalar.backward(gradient=grad_tensor)
print(f"Gradient for x_vector after y_non_scalar.backward(gradient=...): {x_vector.grad}")
print(f"Gradient for w after y_non_scalar.backward(gradient=...): {w.grad}")
```

输出：

```text
Error calling backward() on non-scalar: grad can be implicitly created only for scalar outputs
Gradient for x_vector after y_non_scalar.backward(gradient=...): tensor([3., 3.])
Gradient for w after y_non_scalar.backward(gradient=...): 6.0
```

在大多数标准的训练循环中，你将计算一个单一的标量损失值，代表一个批次或样本的误差，并直接在该标量上调用 `loss.backward()`，无需提供 `gradient` 参数。

### 计算图和梯度流

`loss.backward()` 触发对创建 `loss` 的操作图进行反向遍历。


clusterᵢnputs

输入 (requires\_grad=True)

clusterₒps

操作

clusterₒutput

输出 (标量)

x

x=2.0

mulₒp

\*

x->mulₒp

w

w=3.0

w->mulₒp

b

b=1.0

addₒp

+

b->addₒp

mulₒp->x

 d(loss)/dx

mulₒp->w

 d(loss)/dw

y

y=7.0

mulₒp->y

addₒp->b

 d(loss)/db

addₒp->mulₒp

addₒp->y

w\*x+b

sqₒp

pow(2)

sqₒp->y

 d(loss)/dy

loss

loss=49.0

sqₒp->loss

 subgraph

 subgraph

clusterᵢntermediate

clusterᵢntermediate

y->addₒp

y->sqₒp

loss->sqₒp

 loss.backward() 从这里开始

> 一个简化的计算图，显示了输入 `x`、`w`、`b`、中间结果 `y` 和最终的标量 `loss`。虚线红色箭头说明了在 `loss.backward()` 过程中计算 `x`、`w` 和 `b` 梯度所经过的路径。

默认情况下，PyTorch 在 `backward()` 被调用后会清除计算图的中间缓冲区，以节省内存。这意味着如果你需要在图的*相同*部分多次调用 `backward()`（这种情况较少，通常用于高级技术或调试），你需要在第一次 `backward()` 调用时传入 `retain_graph=True`。然而，对于标准训练，你会构建一个图，计算损失，调用 `backward()`，更新权重 (weight)，然后为下一个批次重复这个过程，这会构建一个新的图。

理解 `backward()` 对于在 PyTorch 中训练模型非常重要。它是将模型输出和损失函数 (loss function)与需要调整的参数 (parameter)联系起来的机制。在接下来的章节中，我们将了解优化器如何访问和使用这些计算出的梯度。

Jul 19, 2025

修复了 backward() 方法部分的一个错误。将一个不正确的代码示例替换为准确的示例，以正确演示非标量张量的运行时错误及其正确的解决方案。

获取即时帮助、个性化解释和交互式代码示例。

---

### 访问梯度（.grad）

### 访问梯度（`.grad`）

在计算标量张量（通常是损失）相对于计算图中其他张量的梯度时，PyTorch 使用 `backward()` 方法。此方法触发梯度计算，但它不会直接返回梯度。相反，PyTorch 会将计算出的梯度存储在张量自身的一个特殊属性中：即 `.grad` 属性。

这个属性主要为计算图中的*叶*张量填充，即那些你通过设置 `requires_grad=True` 明确要求进行梯度跟踪的张量。请记住，叶张量通常是你直接创建的，例如模型参数 (parameter)或输入，而不是通过运算生成的中间张量。

`.grad` 属性包含一个与它所属的原始张量形状相同的张量。`.grad` 张量中的每个元素表示标量（调用 `backward()` 的对象）相对于原始张量中对应元素的偏导数。如果 LLL 是标量损失，www 是一个张量参数，那么在调用 `L.backward()` 之后，属性 `w.grad` 将包含表示 ∂L∂w\frac{\partial L}{\partial w}∂w∂L​ 的张量。

我们通过一个简单例子来说明这一点：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y = w * x + b

y.backward()

print(f"y 对 x 的梯度 (dy/dx): {x.grad}")
print(f"y 对 w 的梯度 (dy/dw): {w.grad}")
print(f"y 对 b 的梯度 (dy/db): {b.grad}")

z = torch.tensor(4.0, requires_grad=False)
print(f"张量 z 的梯度 (requires_grad=False): {z.grad}")
```

**预期输出：**

```
y 对 x 的梯度 (dy/dx): 3.0
y 对 w 的梯度 (dy/dw): 2.0
y 对 b 的梯度 (dy/db): 1.0
张量 z 的梯度 (requires_grad=False): None
```

在这个例子中：

1. 我们定义了 `x`、`w` 和 `b`，并设置 `requires_grad=True`，将它们标记 (token)为我们想要梯度的叶节点。
2. 我们执行了操作 y=w∗x+by = w \* x + by=w∗x+b。PyTorch 在后台构建了一个计算图。
3. 我们调用了 `y.backward()`。Autograd 从 `y` 开始向后遍历图以计算梯度。
   - ∂y∂x=w=3.0\frac{\partial y}{\partial x} = w = 3.0∂x∂y​=w=3.0
   - ∂y∂w=x=2.0\frac{\partial y}{\partial w} = x = 2.0∂w∂y​=x=2.0
   - ∂y∂b=1=1.0\frac{\partial y}{\partial b} = 1 = 1.0∂b∂y​=1=1.0
4. 计算出的梯度存储在 `x.grad`、`w.grad` 和 `b.grad` 中。访问这些属性会显示计算出的张量值。
5. 张量 `z` 是在 `requires_grad=False` 的情况下创建的，因此它未参与 Autograd 跟踪的梯度计算，其 `.grad` 属性保持为 `None`。

需要记住的是，梯度默认是累积的。如果你在不清除梯度的情况下，对图中可能不同的部分（或相同的部分）多次调用 `backward()`，新计算的梯度将被*添加*到 `.grad` 属性中已经存在的值上。这种行为是故意的，对于像跨小批量进行梯度累积这样的情况很有用，但在典型的训练循环中，你需要在每个反向传播 (backpropagation)步骤之前明确地将梯度清零。这通常通过 `optimizer.zero_grad()` 完成，我们将在构建训练循环时更详细地讨论它。

目前，要点是，在 `loss.backward()` 之后，更新模型参数所需的梯度可以直接在这些参数张量的 `.grad` 属性中可用。

获取即时帮助、个性化解释和交互式代码示例。

---

### 禁用梯度追踪

### 禁用梯度追踪

尽管Autograd自动追踪操作并计算梯度的能力对模型训练不可或缺，但在某些情况下，这种追踪是不必要甚至不希望的。具体来说，在模型评估（推理 (inference)）期间，或者当你执行不应影响梯度计算的操作时，追踪历史会消耗内存和计算资源，而没有任何益处。PyTorch提供了选择性禁用梯度追踪的方法。

### 为什么要禁用梯度追踪？

1. **模型评估（推理 (inference)）：** 当你使用训练好的模型对新数据进行预测时，你不会更新其权重 (weight)。在此阶段计算梯度没有意义。禁用梯度追踪可以显著减少内存使用（因为无需存储计算图），并加快前向传播的速度。
2. **冻结模型参数 (parameter)：** 在微调 (fine-tuning)期间，你可能希望冻结预训练 (pre-training)模型的一部分（例如，早期的卷积层），而只训练后面的层。你需要告知PyTorch不要为这些冻结的参数计算或存储梯度。
3. **内存效率：** Autograd存储的计算图会消耗大量内存，特别是对于复杂的模型或长序列。在不需要时关闭追踪可以避免这种额外开销。

### 使用 `torch.no_grad()` 上下文 (context)管理器

禁用代码块梯度追踪最常用且推荐的方式是使用 `torch.no_grad()` 上下文管理器。在此 `with` 块内执行的任何PyTorch操作都会表现得如同所有输入张量都不需要梯度，即使它们最初设置了 `requires_grad=True`。

```python
import torch

x = torch.randn(2, 2, requires_grad=True)
w = torch.randn(2, 2, requires_grad=True)
b = torch.randn(2, 2, requires_grad=True)

y = x * w + b
print(f"y.requires_grad: {y.requires_grad}")
print(f"y.grad_fn: {y.grad_fn}")

print("\n进入torch.no_grad()上下文：")
with torch.no_grad():
    z = x * w + b
    print(f"  z.requires_grad: {z.requires_grad}")
    print(f"  z.grad_fn: {z.grad_fn}")

    k = x * 5
    print(f"  k.requires_grad: {k.requires_grad}")

print("\n退出torch.no_grad()上下文：")
p = x * w
print(f"p.requires_grad: {p.requires_grad}")
print(f"p.grad_fn: {p.grad_fn}")
```

如示例所示，`with torch.no_grad():` 块内的操作会产生 `requires_grad=False` 且没有关联 `grad_fn` 的输出（`z`、`k`），这表明它们已脱离计算图历史。这正是你在评估循环中想要的结果：

```python

model.eval()
total_loss = 0
correct_predictions = 0

with torch.no_grad():
    for inputs, labels in validation_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
```

### 使用 `.detach()` 方法

另一种阻止特定张量进行梯度追踪的方法是使用 `.detach()` 方法。此方法会创建一个*新*张量，它与原始张量共享底层数据存储，但明确地脱离了当前计算图。它将拥有 `requires_grad=False`。

```python
import torch

a = torch.randn(3, 3, requires_grad=True)
b = a * 2
print(f"b.requires_grad: {b.requires_grad}")
print(f"b.grad_fn: {b.grad_fn}")

c = b.detach()
print(f"\n分离b以创建c后：")
print(f"c.requires_grad: {c.requires_grad}")
print(f"c.grad_fn: {c.grad_fn}")

print(f"\n原始张量 b 仍保持连接：")
print(f"b.requires_grad: {b.requires_grad}")
print(f"b.grad_fn: {b.grad_fn}")

d = c + 1
print(f"\n在分离张量 c 上的操作：")
print(f"d.requires_grad: {d.requires_grad}")
```

**何时使用 `.detach()` 与 `torch.no_grad()`？**

- 当你希望执行一*段*操作而不追踪梯度时，使用 `torch.no_grad()`，这通常用于推理 (inference)或评估代码段。为此目的，它通常更高效。
- 当你需要将*特定张量*从计算图中移除时，使用 `.detach()`，例如为了记录其值、在不应影响梯度的操作中使用它（如更新指标），或将其传递给期望非梯度追踪张量的函数，同时可能仍需要在其他地方使用原始张量的梯度历史。由于 `.detach()` 共享数据，*原地*修改分离的张量会影响原始张量，如果处理不当，这可能会对梯度计算产生影响。

### 原地修改 `requires_grad`

你也可以直接原地修改张量的 `requires_grad` 属性，但与上下文 (context)管理器或 `.detach()` 相比，这种临时禁用方法通常不太常见。它通常用于定义你明确不希望训练的参数 (parameter)。

```python
my_tensor = torch.randn(5, requires_grad=True)
print(f"初始requires_grad: {my_tensor.requires_grad}")

my_tensor.requires_grad_(False)
print(f"requires_grad_(False)后: {my_tensor.requires_grad}")
```

使用 `torch.no_grad()` 是进行高效推理 (inference)和评估的标准做法，而 `.detach()` 在你需要将特定张量从梯度历史中隔离时，提供更细致的控制。了解何时以及如何禁用梯度追踪对于编写高效且正确的PyTorch代码非常重要，特别是在你进一步学习基础训练循环之后。

获取即时帮助、个性化解释和交互式代码示例。

---

### 梯度累积

### 梯度累积

对标量张量（如损失值）调用 `.backward()` 会触发计算图中所有 `requires_grad=True` 的张量的梯度。需要了解的一个主要特性是，PyTorch 默认会*累积*梯度。

### 默认梯度累积行为

如果您在多次调用 `.backward()` 之间不清除梯度，PyTorch 会将新计算的梯度添加到叶张量（参数 (parameter)）的`.grad`属性中已有的值上。

我们用一个简单例子来说明这一点：

```python
import torch

x = torch.tensor([2.0], requires_grad=True)

y = x * x
z = y * 3

z.backward(retain_graph=True)
print(f"After first backward pass, x.grad: {x.grad}")

z.backward()

print(f"After second backward pass, x.grad: {x.grad}")

x.grad.zero_()
print(f"After zeroing, x.grad: {x.grad}")
```

运行此代码会产生类似于以下的输出：

```text
After first backward pass, x.grad: tensor([12.])
After second backward pass, x.grad: tensor([24.])
After zeroing, x.grad: tensor([0.])
```

请注意，第二次调用 `z.backward()` 如何将新计算的梯度（12）加到之前存储的梯度（12）上，结果为 24。这种累积是刻意为之的，并且有重要的用途。

### 为何累积梯度？模拟更大批量

这种默认行为的主要原因是为了方便**梯度累积**。当训练大型模型需要大批量数据以实现稳定收敛，但现有 GPU 内存无法一次性容纳如此大的批量时，此方法就很有用。

与其处理一个大批量，不如这样操作：

1. 将大批量分成若干个更小的迷你批量。
2. 每次处理一个迷你批量：执行前向传播并计算损失。
3. 对当前迷你批量的损失调用 `.backward()`。为此迷你批量计算的梯度将添加到模型参数 (parameter)的 `.grad` 属性中。
4. 对大批量内的所有迷你批量重复步骤 2-3。
5. *在*处理完所有迷你批量并累积其梯度后，使用 `optimizer.step()` 执行一次优化器更新步。此步骤使用所有迷你批量梯度的*总和*来更新模型权重 (weight)，从而有效地模拟了更大批量的一次更新步。
6. 非常重要的一点是，*在*开始处理*下一个*大批量（如果不是累积，则为下一个迷你批量）*之前*，使用 `optimizer.zero_grad()` **清除梯度**。

这使您可以使用模型所需的有效批量大小进行训练，即使它不能一次性完全载入内存，从而牺牲计算时间来提高内存效率。

### 标准训练中 `optimizer.zero_grad()` 的必要性

在标准的训练循环中，您在每次迭代中处理一个批量、计算损失、计算梯度并更新权重 (weight)，您通常*不希望*来自前一个批量的梯度影响当前的更新步。每个批量的梯度计算应该是独立的。

由于 PyTorch 默认累积梯度，如果在计算新批量的梯度之前未能清除它们，将导致不正确的更新。优化器将使用新旧梯度的混合，从而损害训练过程。

这就是为什么在标准的 PyTorch 训练循环中，您几乎总能看到 `optimizer.zero_grad()` 被调用的原因。它将优化器管理的所有参数 (parameter)的 `.grad` 属性重置，以确保随后的 `.backward()` 调用完全基于当前批量的损失来计算梯度。

典型的训练迭代结构如下所示：

```python

    inputs, labels = data_batch
    inputs, labels = inputs.to(device), labels.to(device)

    optimizer.zero_grad()

    outputs = model(inputs)

    loss = loss_fn(outputs, labels)

    loss.backward()

    optimizer.step()
```

`optimizer.zero_grad()` 的放置位置很重要。它应该在您计算当前迭代的损失并执行反向传播 (backpropagation)*之前*发生，确保当前批量的梯度计算有一个干净的开始。虽然它通常放在循环的开头，但技术上它只需要在 `loss.backward()` 之前发生。然而，将其放在开头是常见做法，并且能清楚地划分新批量处理的开始。

总而言之，梯度累积是 PyTorch 的一个内置功能，对于模拟更大的批量数据很有用。然而，在标准训练循环中，您必须通过在每次迭代开始时调用 `optimizer.zero_grad()` 来明确阻止这种累积，以确保模型更新仅基于当前批量的数据是正确的。

获取即时帮助、个性化解释和交互式代码示例。

---

### 动手实践：Autograd 运用

### 动手实践：Autograd 运用

实际例子演示 PyTorch Autograd 系统。这些练习会引导您设置梯度要求、执行反向传播 (backpropagation)、查看梯度、观察累积以及禁用梯度跟踪。请确保您已安装 PyTorch 并能导入 `torch` 库。

### 设置

首先，导入 PyTorch：

```python
import torch
```

### 例子 1：基本梯度计算

我们从一个非常简单的计算开始，并跟踪梯度。我们将定义两个张量 `x` 和 `w`，其中 `w` 表示我们想要优化的权重 (weight)。我们将计算一个简单的输出 `y`，然后计算一个标量损失 `L`。

1. **创建张量**：将 `x` 定义为一个包含一些数据的张量，将 `w` 定义为一个需要计算其梯度的张量（使用 `requires_grad=True`）。

   ```python

   x = torch.tensor([2.0, 4.0, 6.0])

   w = torch.tensor([0.5], requires_grad=True)

   print(f"x: {x}")
   print(f"w: {w}")
   print(f"x.requires_grad: {x.requires_grad}")
   print(f"w.requires_grad: {w.requires_grad}")
   ```

   请注意，`x` 默认情况下不需要梯度，而我们为 `w` 显式设置了它。
2. **定义计算**：执行一个简单的运算。任何通过涉及 `requires_grad=True` 的张量运算而得到的张量，其 `requires_grad` 也会是 `True`。

   ```python

   y = w * x

   L = y.mean()

   print(f"y: {y}")
   print(f"L: {L}")
   print(f"y.requires_grad: {y.requires_grad}")
   print(f"L.requires_grad: {L.requires_grad}")
   ```

   您会看到 `y` 和 `L` 现在都需要梯度，因为它们依赖于 `w`。
3. **计算梯度**：在最终的标量输出 (`L`) 上使用 `.backward()` 方法来计算整个图中的梯度。

   ```python

   L.backward()
   ```
4. **查看梯度**：查看张量 `w` 的 `.grad` 属性。

   ```python

   print(f"Gradient dL/dw: {w.grad}")

   print(f"Gradient dL/dx: {x.grad}")
   ```

   我们来分析 `w.grad` 的结果。计算过程为：
   yi=w∗xiy\_i = w \* x\_iyi​=w∗xi​
   L=13∑yi=13(wx1+wx2+wx3)L = \frac{1}{3} \sum y\_i = \frac{1}{3} (w x\_1 + w x\_2 + w x\_3)L=31​∑yi​=31​(wx1​+wx2​+wx3​)
   梯度 ∂L∂w\frac{\partial L}{\partial w}∂w∂L​ 为：

   ∂L∂w=13(x1+x2+x3)\frac{\partial L}{\partial w} = \frac{1}{3} (x\_1 + x\_2 + x\_3)∂w∂L​=31​(x1​+x2​+x3​)

   当 x=[2.0,4.0,6.0]x = [2.0, 4.0, 6.0]x=[2.0,4.0,6.0] 时，梯度为 13(2.0+4.0+6.0)=12.03=4.0\frac{1}{3} (2.0 + 4.0 + 6.0) = \frac{12.0}{3} = 4.031​(2.0+4.0+6.0)=312.0​=4.0。这与输出 `tensor([4.])` 相符。因为 `x` 在创建时没有设置 `requires_grad=True`，所以它的梯度未被计算，仍为 `None`。

### 例子 2：梯度与计算图

Autograd 动态构建图。我们来看一个稍微复杂一点的例子。

1. **创建张量**：

   ```python
   a = torch.tensor(2.0, requires_grad=True)
   b = torch.tensor(3.0, requires_grad=True)
   c = torch.tensor(4.0, requires_grad=False)

   print(f"a: {a}, requires_grad={a.requires_grad}")
   print(f"b: {b}, requires_grad={b.requires_grad}")
   print(f"c: {c}, requires_grad={c.requires_grad}")
   ```
2. **定义计算**：

   ```python
   d = a * b
   e = d + c
   f = e * 2

   print(f"d: {d}, requires_grad={d.requires_grad}")
   print(f"e: {e}, requires_grad={e.requires_grad}")
   print(f"f: {f}, requires_grad={f.requires_grad}")
   ```
3. **计算并查看梯度**：

   ```python

   f.backward()

   print(f"Gradient df/da: {a.grad}")
   print(f"Gradient df/db: {b.grad}")
   print(f"Gradient df/dc: {c.grad}")
   ```

   我们手动计算一下：
   d=a×bd = a \times bd=a×b
   e=d+c=a×b+ce = d + c = a \times b + ce=d+c=a×b+c
   f=2×e=2(a×b+c)f = 2 \times e = 2(a \times b + c)f=2×e=2(a×b+c)

   ∂f∂a=2×b=2×3.0=6.0\frac{\partial f}{\partial a} = 2 \times b = 2 \times 3.0 = 6.0∂a∂f​=2×b=2×3.0=6.0
   ∂f∂b=2×a=2×2.0=4.0\frac{\partial f}{\partial b} = 2 \times a = 2 \times 2.0 = 4.0∂b∂f​=2×a=2×2.0=4.0
   ∂f∂c=2\frac{\partial f}{\partial c} = 2∂c∂f​=2

   `a` 和 `b` 的计算梯度是匹配的。由于 `c` 定义时 `requires_grad=False`，Autograd 没有跟踪涉及 `c` 的操作来计算关于 `c` 本身的梯度，因此 `c.grad` 为 `None`。

### 例子 3：梯度累积

默认情况下，每次调用 `.backward()` 时，梯度都会累积到 `.grad` 属性中。这对于计算多个损失的梯度或模拟更大的批次大小等情况很有用，但在标准训练循环中，需要显式地将梯度清零。

1. **设置**：我们再次使用一个简单的设置。

   ```python
   x = torch.tensor(5.0, requires_grad=True)
   y = x * x
   print(f"Initial x.grad: {x.grad}")
   ```
2. **第一次反向传播 (backpropagation)**：

   ```python

   y.backward(retain_graph=True)
   print(f"x.grad after 1st backward: {x.grad}")
   ```
3. **第二次反向传播（累积）**：再次调用 `backward`，*不*清零梯度。

   ```python
   y.backward(retain_graph=True)
   print(f"x.grad after 2nd backward: {x.grad}")
   ```

   梯度被累积（相加）到之前的值上。
4. **清零梯度**：手动清零梯度。在典型的训练循环中，这通常通过 `optimizer.zero_grad()` 完成。

   ```python
   if x.grad is not None:
       x.grad.zero_()
   print(f"x.grad after zeroing: {x.grad}")
   ```
5. **第三次反向传播（清零后）**：

   ```python
   y.backward()
   print(f"x.grad after 3rd backward: {x.grad}")
   ```

   梯度清零后会重新计算。在训练循环中忘记清零梯度是常见的错误原因。

### 例子 4：禁用梯度跟踪

有时，您需要执行操作而不跟踪其梯度计算，最常见的情况是在模型评估（推理 (inference)）期间或在优化步骤之外调整参数 (parameter)时。

1. **使用 `torch.no_grad()`**：这个上下文 (context)管理器是禁用代码块梯度跟踪的标准方法。

   ```python
   a = torch.tensor(2.0, requires_grad=True)
   print(f"Outside context: a.requires_grad = {a.requires_grad}")

   with torch.no_grad():
       print(f"Inside context: a.requires_grad = {a.requires_grad}")
       b = a * 2
       print(f"Inside context: b = {b}, b.requires_grad = {b.requires_grad}")

   c = a * 3
   print(f"Outside context: c = {c}, c.requires_grad = {c.requires_grad}")
   ```

   在 `torch.no_grad()` 块内部，尽管 `a` 需要梯度，但生成的张量 `b` 却不需要。这使得块内的操作更节省内存且更快，因为反向传播 (backpropagation)的历史不会被保存。
2. **使用 `.detach()`**：这个方法会创建一个*新*张量，它共享相同的数据，但与计算历史分离。它不需要梯度。

   ```python
   a = torch.tensor(5.0, requires_grad=True)
   b = a * a

   c = a.detach()
   print(f"a.requires_grad: {a.requires_grad}")
   print(f"c.requires_grad: {c.requires_grad}")

   d = c * 3
   print(f"d.requires_grad: {d.requires_grad}")

   L1 = b.mean()
   L1.backward()
   print(f"Gradient dL1/da: {a.grad}")

   if a.grad is not None:
       a.grad.zero_()

   try:

       L2 = (a + d).mean()
       L2.backward()
       print(f"Gradient dL2/da: {a.grad}")
   except RuntimeError as e:
       print(f"Error demonstrating backward with detached: {e}")

   with torch.no_grad():
     c[0] = 100.0
   print(f"After modifying c, a = {a}")
   print(f"After modifying c, c = {c}")
   ```

   `detach()` 在您想在计算中使用张量的值但阻止梯度通过该特定路径回流时很有用，或者当您需要一个没有梯度历史的张量时（例如，用于绘图或日志记录）。请注意，它共享数据存储，因此原位修改会影响原始张量，除非您先 `.clone()` 它（`c = a.detach().clone()`）。

这些练习展示了 Autograd 的核心机制。您已经练习了启用梯度跟踪、执行反向传播、查看计算出的梯度、理解累积以及在需要时禁用跟踪。掌握这些操作对在 PyTorch 中构建和训练神经网络 (neural network)来说非常重要。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 4 Building Models Torch Nn

### torch.nn.Module基类

### `torch.nn.Module` 基类

构建神经网络 (neural network)在PyTorch中围绕着一个主要的理念：`torch.nn.Module`。可以将`nn.Module`视为一个基础蓝图或基类，所有的神经网络模型、层，甚至复杂的复合结构都是以此为基础构建的。它提供了一种标准化的方式来封装模型参数 (parameter)、管理这些参数的辅助函数（例如在CPU和GPU之间移动它们），以及定义输入数据在网络中流动的逻辑。

无论何时，当你在PyTorch中定义自定义神经网络时，通常会通过创建一个继承自`nn.Module`的Python类来完成。这种继承为你的自定义类提供了大量内置功能，这些功能对于深度学习 (deep learning)工作流程来说非常必要。

### `nn.Module`的核心结构

本质上，使用`nn.Module`需要在你的自定义类中实现两个主要方法：

1. **`__init__(self)`:** 构造函数。你在这里定义和初始化网络的组件，例如层（卷积层、线性层等）、激活函数 (activation function)，甚至其他`nn.Module`实例（子模块）。这些组件通常被作为类实例（`self`）的属性进行赋值。
2. **`forward(self, input_data)`:** 此方法定义了网络的*前向传播*。它规定了输入数据（`input_data`）如何流经在`__init__`中定义的层和组件。`forward`方法接收一个或多个输入张量，并返回一个或多个输出张量。PyTorch的Autograd系统会根据此`forward`方法中执行的操作自动构建计算图，从而实现自动微分。

以下是一个自定义模块的骨架：

```python
import torch
import torch.nn as nn

class MySimpleNetwork(nn.Module):
    def __init__(self):
        super(MySimpleNetwork, self).__init__()

        self.layer1 = nn.Linear(in_features=10, out_features=5)

        self.activation = nn.ReLU()

    def forward(self, x):

        x = self.layer1(x)
        x = self.activation(x)
        return x

model = MySimpleNetwork()
print(model)
```

执行此代码将打印出网络结构的表示，展示`nn.Module`如何帮助组织你的组件。

### 参数 (parameter)和子模块

`nn.Module`的一个重要特点是其自动注册和管理可学习参数的能力。当你在`__init__`方法中将一个PyTorch层（如`nn.Linear`、`nn.Conv2d`等）的实例作为属性赋值时，`nn.Module`会识别该层的内部参数（权重 (weight)和偏置 (bias)）。

这些参数是`torch.nn.Parameter`类的实例，它是`torch.Tensor`的一个特殊子类。主要区别在于`Parameter`对象默认自动设置`requires_grad=True`，并且它们会注册到父`nn.Module`中。这种注册使得PyTorch可以轻松收集模型的所有可学习参数，这对于在训练期间将它们传递给优化器来说非常重要。

你也可以直接使用`nn.Parameter`定义你自己的自定义可学习参数：

```python
class CustomModuleWithParameter(nn.Module):
    def __init__(self):
        super().__init__()

        self.my_weight = nn.Parameter(torch.randn(5, 2))

        self.my_info = torch.tensor([1.0, 2.0])

    def forward(self, x):

        return torch.matmul(x, self.my_weight)

module = CustomModuleWithParameter()

for name, param in module.named_parameters():
    print(f"Parameter name: {name}, Shape: {param.shape}, Requires grad: {param.requires_grad}")
```

注意`my_weight`被列为参数，而`my_info`则没有。这种自动跟踪简化了管理深度网络中可能数千或数百万参数的过程。

### `nn.Module`的主要功能

除了定义结构和管理参数 (parameter)之外，`nn.Module`还提供了一些有用的方法，可供你的自定义类继承：

- `parameters()`: 返回模块内（包括子模块中的）所有`nn.Parameter`对象的迭代器。这通常用于将模型的参数提供给优化器。
- `named_parameters()`: 类似于`parameters()`，但会生成（参数名，参数对象）的元组。这有助于检查或有选择地修改特定参数。
- `children()`: 返回直接子模块（定义为属性的子模块）的迭代器。
- `modules()`: 返回网络中所有模块的迭代器，从模块自身开始，然后递归遍历所有子模块。
- `state_dict()`: 返回一个Python字典，该字典包含模块的完整状态，主要将每个参数和缓冲区名称映射到其对应的张量。这对于保存模型权重 (weight)非常重要。
- `load_state_dict()`: 将状态（通常来自保存的文件）加载回模块，恢复参数和缓冲区。
- `to(device)`: 将模块的所有参数和缓冲区移动到指定设备（例如，GPU的`'cuda'`或CPU的`'cpu'`）。这对于硬件加速非常重要。
- `train()`: 将模块及其子模块设置为训练模式。这会影响像Dropout和BatchNorm这样的层，它们在训练和评估期间表现不同。
- `eval()`: 将模块及其子模块设置为评估模式。

理解`nn.Module`非常重要，因为它建立了在PyTorch中定义任何神经网络 (neural network)架构的标准模式。在接下来的章节中，我们将使用这个基类来构建包含各种层、激活函数 (activation function)和损失函数 (loss function)的网络。

获取即时帮助、个性化解释和交互式代码示例。

---

### 定义自定义网络架构

### 定义自定义网络架构

许多网络架构需要比简单的线性层堆叠更复杂的设计。尽管 `torch.nn.Sequential` 对于线性模型很方便，但更复杂的设计常常是必需的。你可能需要跳跃连接（如在 ResNet 中）、多输入/输出路径，或以非顺序方式使用的层。在这种情况下，通过继承 `torch.nn.Module` 来定义你自己的自定义网络架构就变得必不可少。这种方法在指定数据如何通过模型流动方面提供了最大的灵活性。

这个基本过程包含两个主要步骤：

1. **在构造函数 (`__init__`) 中定义层**：创建一个继承自 `torch.nn.Module` 的 Python 类。在其 `__init__` 方法中，你必须首先调用父类的构造函数 (`super().__init__()`)。然后，实例化网络所需的所有层（例如 `nn.Linear`、`nn.Conv2d`、`nn.ReLU` 等），并将它们作为类实例的属性（使用 `self`）进行分配。这些层就成为你的自定义模块的子模块。
2. **在 `forward` 方法中定义数据流**：为你的类实现 `forward` 方法。此方法将输入张量作为参数 (parameter)，并定义输入数据如何通过你在 `__init__` 中定义的层进行传播。此方法的输出是你的网络对于给定输入的最终输出。PyTorch 的 Autograd 系统会根据此 `forward` 方法中执行的操作自动构建计算图。

我们从一个基本例子开始：一个实现为自定义模块的简单线性回归模型。

```python
import torch
import torch.nn as nn

class SimpleLinearModel(nn.Module):
    def __init__(self, input_features, output_features):

        super().__init__()

        self.linear_layer = nn.Linear(input_features, output_features)
        print(f"已初始化 SimpleLinearModel，输入特征数={input_features}，输出特征数={output_features}")
        print(f"已定义层: {self.linear_layer}")

    def forward(self, x):

        print(f"前向传播输入形状: {x.shape}")
        output = self.linear_layer(x)
        print(f"前向传播输出形状: {output.shape}")
        return output

in_dim = 10
out_dim = 1

model = SimpleLinearModel(input_features=in_dim, output_features=out_dim)

dummy_input = torch.randn(5, in_dim)
print(f"\n模拟输入张量形状: {dummy_input.shape}")

output = model(dummy_input)
print(f"模型输出张量形状: {output.shape}")

print("\n模型参数:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  名称: {name}, 形状: {param.shape}")
```

在此示例中：

- `SimpleLinearModel` 继承自 `nn.Module`。
- `__init__` 调用 `super().__init__()` 并定义 `self.linear_layer = nn.Linear(...)`。此层现在是一个已注册的子模块。
- `forward(self, x)` 接收输入 `x` 并将其通过 `self.linear_layer`，然后返回结果。

PyTorch 会自动追踪 `nn.Linear` 层的参数（权重 (weight)和偏置 (bias)），因为它们被作为属性分配在 `nn.Module` 子类中。我们可以通过查看 `model.parameters()` 或 `model.named_parameters()` 来验证这一点。

### 构建多层感知机 (MLP)

现在，我们来构建一个稍微复杂一点的模型，一个两层 MLP，在层之间带有一个 ReLU 激活函数 (activation function)。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        print(f"已初始化 SimpleMLP: 输入={input_size}, 隐藏层={hidden_size}, 输出={output_size}")
        print(f"层 1: {self.layer1}")
        print(f"激活函数: {self.activation}")
        print(f"层 2: {self.layer2}")

    def forward(self, x):

        print(f"前向传播输入形状: {x.shape}")
        x = self.layer1(x)
        print(f"经过层 1 后的形状: {x.shape}")
        x = self.activation(x)

        print(f"经过激活函数后的形状: {x.shape}")
        x = self.layer2(x)
        print(f"经过层 2（输出）后的形状: {x.shape}")
        return x

in_size = 784
hidden_units = 128
out_size = 10

mlp_model = SimpleMLP(input_size=in_size, hidden_size=hidden_units, output_size=out_size)

dummy_mlp_input = torch.randn(32, in_size)
print(f"\n模拟 MLP 输入形状: {dummy_mlp_input.shape}")

mlp_output = mlp_model(dummy_mlp_input)
print(f"MLP 输出形状: {mlp_output.shape}")

print("\nMLP 模型参数:")
for name, param in mlp_model.named_parameters():
     if param.requires_grad:
        print(f"  名称: {name}, 形状: {param.shape}")
```

这里，`forward` 方法明确规定了序列：输入 -> `layer1` -> `activation` -> `layer2` -> 输出。请注意，`nn.ReLU` 等激活函数通常也在 `__init__` 中定义为层，并在 `forward` 中调用。另外，你也可以直接在 `forward` 方法中使用其函数式等效项（例如，导入 `torch.nn.functional as F` 后使用 `F.relu(x)`），特别是对于没有可学习参数 (parameter)的激活函数。

### 可视化 MLP 结构

我们可以可视化 `SimpleMLP` 的 `forward` 方法中定义的数据流。

SimpleMLP

输入 (x)

输入 (x)

层1 (线性)

层1 (线性)

输入 (x)->层1 (线性)

激活函数 (ReLU)

激活函数 (ReLU)

层1 (线性)->激活函数 (ReLU)

层2 (线性)

层2 (线性)

激活函数 (ReLU)->层2 (线性)

输出

输出

层2 (线性)->输出

> 数据流经 `SimpleMLP` 模型，如其 `forward` 方法中所定义。

### 继承 `nn.Module` 的优点

- **灵活性**：这是主要优势。你可以实现任何架构，包括具有多个输入/输出、残差连接（其中输入被加回到后续层的输出）、共享层或 `forward` 传递中自定义操作的架构。`nn.Sequential` 仅限于严格的线性层序列。
- **可读性和组织性**：复杂的架构通常在类结构中组织时更容易理解，其中层在 `__init__` 中定义，它们的交互方式在 `forward` 中定义。
- **参数 (parameter)管理**：PyTorch 会自动发现并注册在 `__init__` 方法中作为属性分配的任何 `nn.Module`（例如 `self.layer1 = nn.Linear(...)`）。这意味着 `model.parameters()` 将正确地给出所有子模块的所有可学习参数（权重 (weight)、偏置 (bias)），使其可以轻松传递给优化器。
- **嵌套**：自定义模块可以包含其他模块（包括 `nn.Sequential` 或其他自定义模块），从而允许你构建分层和可重用的组件。

通过继承 `nn.Module`，你可以完全控制网络的结构，从而实现针对特定任务的复杂深度学习 (deep learning)模型。这种方法是构建更复杂的前馈网络的标准做法。

获取即时帮助、个性化解释和交互式代码示例。

---

### 常见层：线性、卷积、循环

### 常见层：线性、卷积、循环

PyTorch 提供神经网络 (neural network)模型的基本构成单元，即层。`torch.nn` 包提供了多种预构建层，它们执行神经网络中常见的操作。这些层将可学习参数 (parameter)（权重 (weight)和偏置 (bias)）和操作本身都包含在内。这里将介绍三种主要类型：线性层、卷积层和循环层。

### 线性层 (`nn.Linear`)

神经网络 (neural network)层最基本的类型是**线性**层，也称为全连接层或密集层。它对输入数据进行线性变换。如果输入张量的形状为 (∗,Hin)(\*, H\_{in})(∗,Hin​)，其中 ∗\*∗ 代表任意数量的前导维度（如批大小），HinH\_{in}Hin​ 是输入特征的数量，那么 `nn.Linear` 层会将其转换为形状为 (∗,Hout)(\*, H\_{out})(∗,Hout​) 的输出张量，其中 HoutH\_{out}Hout​ 是为该层指定的输出特征数量。

在数学上，此操作表示为 y=xWT+by = x W^T + by=xWT+b，其中：

- xxx 是输入
- WWW 是权重 (weight)矩阵
- bbb 是偏置 (bias)向量 (vector)
- yyy 是输出

您可以通过指定输入特征和输出特征的数量来创建线性层。

```python
import torch
import torch.nn as nn

linear_layer = nn.Linear(in_features=20, out_features=30)

input_tensor = torch.randn(64, 20)

output_tensor = linear_layer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")

print(f"\nWeight shape: {linear_layer.weight.shape}")
print(f"Bias shape: {linear_layer.bias.shape}")
```

线性层是许多架构中的基本组成部分，包括简单的多层感知机（MLP），并且在像 CNN 和 RNN 这样更复杂的模型中，它们通常作为最终的分类或回归头部。

### 卷积层 (`nn.Conv2d`)

卷积层是现代计算机视觉模型的核心。与对扁平特征向量 (vector)进行操作的线性层不同，卷积层设计用于处理网格状数据（如图像），并保留空间关系。用于二维数据（如图像）的主要层是 `nn.Conv2d`。

它的工作原理是通过在输入空间维度（高和宽）上滑动小型滤波器（卷积核）。对于滤波器的每个位置，它计算滤波器权重 (weight)与滤波器下输入图像块的点积，从而在输出特征图中生成一个元素。这个过程有助于检测边缘、角点和纹理等空间模式。

`nn.Conv2d` 的主要参数 (parameter)包括：

- `in_channels`：输入图像中的通道数量（例如，RGB 图像为 3）。
- `out_channels`：要应用的滤波器数量。每个滤波器生成一个输出通道（特征图）。
- `kernel_size`：滤波器的大小（高，宽）。可以是单个整数用于方形卷积核，或一个元组 `(H, W)`。
- `stride` (可选，默认 1)：滤波器在每一步移动的像素数。
- `padding` (可选，默认 0)：添加到输入边界的零填充量。

```python

conv_layer = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)

input_image_batch = torch.randn(16, 3, 32, 32)

output_feature_maps = conv_layer(input_image_batch)

print(f"Input shape: {input_image_batch.shape}")
print(f"Output shape: {output_feature_maps.shape}")

print(f"\nWeight (filter) shape: {conv_layer.weight.shape}")
print(f"Bias shape: {conv_layer.bias.shape}")
```

`nn.Conv2d` 及其变体（`nn.Conv1d`、`nn.Conv3d`）对涉及空间层次的任务是不可或缺的，主要用于图像和视频分析，但有时也应用于序列数据。我们将在第 7 章更详细地了解如何构建 CNN。

### 循环层 (`nn.RNN`)

循环神经网络 (neural network)（RNN）设计用于处理序列数据，其中元素的顺序很重要。示例包括文本、时间序列数据或音频信号。RNN 层的核心思想是维护一个隐藏状态，该状态捕捉序列中先前元素的信息，并影响当前元素的处理。

PyTorch 中基本的 `nn.RNN` 层逐步处理输入序列。在每一步 ttt，它接收输入 xtx\_txt​ 和前一个隐藏状态 ht−1h\_{t-1}ht−1​，以计算输出 oto\_tot​（可选，通常只使用最终隐藏状态）和新的隐藏状态 hth\_tht​。

`nn.RNN` 的主要参数 (parameter)：

- `input_size`：每个时间步输入中的特征数量。
- `hidden_size`：隐藏状态中的特征数量。
- `num_layers` (可选，默认 1)：堆叠 RNN 层的数量。
- `batch_first` (可选，默认 False)：如果为 True，输入和输出张量将以 `(batch, seq_len, features)` 形式提供，而不是默认的 `(seq_len, batch, features)`。

```python

rnn_layer = nn.RNN(input_size=5, hidden_size=30, batch_first=True)

input_sequence_batch = torch.randn(10, 20, 5)

initial_hidden_state = torch.randn(1, 10, 30)

output_sequence, final_hidden_state = rnn_layer(input_sequence_batch, initial_hidden_state)

print(f"Input shape: {input_sequence_batch.shape}")
print(f"Initial hidden state shape: {initial_hidden_state.shape}")
print(f"Output sequence shape: {output_sequence.shape}")
print(f"Final hidden state shape: {final_hidden_state.shape}")
```

虽然 `nn.RNN` 展示了基本思想，但简单的 RNN 通常因梯度消失而难以处理长序列。在实际应用中，通常更偏好 `nn.LSTM`（长短期记忆）和 `nn.GRU`（门控循环单元 (GRU)）等更高级的循环层，因为它们包含门控机制，能更好地管理长距离依赖中的信息流。这些将在第 7 章再次提及。

这三种层类型（线性层、卷积层、循环层）代表了针对不同数据和任务的基本操作。`torch.nn` 提供了这些层以及许多其他层（如池化层、归一化 (normalization)层、dropout 层），它们可以在 `nn.Module` 子类中组合起来，以构建复杂的深度学习 (deep learning)模型。在接下来的部分中，我们将看到如何将这些层与非线性激活函数 (activation function)结合，并定义使用损失函数 (loss function)和优化器训练它们的标准。

获取即时帮助、个性化解释和交互式代码示例。

---

### 激活函数 (ReLU, Sigmoid, Tanh)

### 激活函数 (ReLU, Sigmoid, Tanh)

神经网络 (neural network)的表示能力很大程度上得益于在层之间引入非线性。如果只是简单地堆叠线性变换（如 `nn.Linear` 层）而没有任何介入函数，整个网络将简化为一个单一的等效线性变换。无论网络有多少层，它都只能学习输入与输出之间的线性关系。

激活函数 (activation function)是引入这些重要非线性的组成部分。它们逐元素应用于层的输出（常被称为预激活值或logit），在将值传递给下一层之前对其进行转换。PyTorch 在 `torch.nn` 模块中提供了各种各样的激活函数，通常通过将它们实例化为层在模型定义中使用。我们来看看其中最常见的三种：ReLU、Sigmoid 和 Tanh。

## ReLU (修正线性单元)

修正线性单元，简称ReLU，可以说是现代深度学习 (deep learning)中最受欢迎的激活函数 (activation function)，尤其是在卷积神经网络 (neural network) (CNN)中。它的定义非常简单：如果输入为正，它直接输出输入值，否则输出零。

其数学定义为：

ReLU(x)=max⁡(0,x)\text{ReLU}(x) = \max(0, x)ReLU(x)=max(0,x)

在 PyTorch 中，可以使用 `nn.ReLU`：

```python
import torch
import torch.nn as nn

relu_activation = nn.ReLU()
input_tensor = torch.randn(4)
output_tensor = relu_activation(input_tensor)

print(f"输入: {input_tensor}")
print(f"ReLU 输出: {output_tensor}")

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

model = SimpleNet()
```

−4−2024024

> ReLU 函数对负输入为零，对正输入为线性。

**优点：**

- **计算效率高：** 计算非常简单（max⁡(0,x)\max(0, x)max(0,x)）。
- **减少梯度消失：** 对于正输入，梯度为1，这有助于在训练期间梯度反向传播 (backpropagation)，相比于 Sigmoid 或 Tanh 等饱和函数。
- **引入稀疏性：** 由于负输入被映射到零，这可以导致网络中出现稀疏激活，有时可能是有益的。

**缺点：**

- **ReLU 死亡问题：** 输入始终落在负区间的神经元将输出零。因此，流经它们的梯度也将为零，这意味着它们的权重 (weight)在反向传播期间不会被更新。这些神经元实际上“死亡”了，不再对学习有贡献。Leaky ReLU 或 Parametric ReLU (PReLU) 等变体试图解决此问题。
- **非零中心：** 输出始终为非负值。

## Sigmoid

Sigmoid 函数，有时也称为逻辑函数，将其输入压缩到 0 到 1 的范围内。它在历史上很受欢迎，尤其是在二元分类模型的输出层，其中输出代表一个概率。

其数学形式为：

σ(x)=11+e−x\sigma(x) = \frac{1}{1 + e^{-x}}σ(x)=1+e−x1​

在 PyTorch 中，可以使用 `nn.Sigmoid`：

```python
import torch
import torch.nn as nn

sigmoid_activation = nn.Sigmoid()
input_tensor = torch.randn(4)
output_tensor = sigmoid_activation(input_tensor)

print(f"输入: {input_tensor}")
print(f"Sigmoid 输出: {output_tensor}")
```

−4−202400.51

> Sigmoid 函数将任意实数平滑地映射到 (0, 1) 的范围内。

**优点：**

- **输出易于理解：** (0, 1) 的范围便于表示概率。
- **梯度平滑：** 函数处处可微，提供平滑的梯度。

**缺点：**

- **梯度消失：** 对于非常大或非常小的输入，函数会饱和（输出接近 1 或 0），梯度变得非常接近零。这会严重减缓或停止深度网络的学习，因为梯度难以通过多层反向传播 (backpropagation)。
- **非零中心：** 输出始终为正，这有时会减缓收敛速度，相比于零中心激活函数 (activation function)。
- **计算成本更高：** 指数函数比 ReLU 的简单比较成本更高。

由于梯度消失问题，Sigmoid 在今天的深度网络隐藏层中不如 ReLU 常用，但它在特定任务（例如二元分类或多标签分类）的输出层中仍然适用。

## Tanh (双曲正切)

双曲正切函数，即 Tanh 函数，在数学上与 Sigmoid 相关，但将其输入压缩到 (-1, 1) 的范围内。

其定义为：

tanh⁡(x)=ex−e−xex+e−x=2σ(2x)−1\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} = 2 \sigma(2x) - 1tanh(x)=ex+e−xex−e−x​=2σ(2x)−1

在 PyTorch 中，可以使用 `nn.Tanh`：

```python
import torch
import torch.nn as nn

tanh_activation = nn.Tanh()
input_tensor = torch.randn(4)
output_tensor = tanh_activation(input_tensor)

print(f"输入: {input_tensor}")
print(f"Tanh 输出: {output_tensor}")
```

−4−2024−1−0.500.51

> Tanh 函数将任意实数平滑地映射到 (-1, 1) 的范围内。

**优点：**

- **零中心输出：** 与 Sigmoid 不同，Tanh 的输出以零为中心，这通常有助于模型在训练期间的收敛。零中心数据通常与基于梯度的优化方法配合得更好。
- **梯度平滑：** 与 Sigmoid 类似，它处处可微。

**缺点：**

- **梯度消失：** 与 Sigmoid 类似，Tanh 也会在很大正值或负值输入时出现饱和，导致深度网络中梯度消失。虽然由于其零中心性质，在隐藏层中它通常比 Sigmoid 更受青睐，但它仍然容易受到此问题的影响。
- **计算成本更高：** 涉及指数函数，使其比 ReLU 成本更高。

在 ReLU 兴起之前，Tanh 在隐藏层中通常比 Sigmoid 更受青睐，主要因为其零中心输出范围。它仍然常见于循环神经网络 (neural network) (RNN) 和 LSTM 中。

## 选择激活函数 (activation function)

没有一个“最佳”激活函数适用于所有情况。然而，有一些通用指导原则：

- **ReLU** 通常是前馈网络和 CNN 中隐藏层的默认选择，因为它高效且能有效缓解正输入时的梯度消失问题。从 ReLU 开始，如果遇到诸如死亡神经元之类的问题，再考虑其他替代方案。
- 如果怀疑存在“ReLU 死亡”问题，**Leaky ReLU** 或 **Parametric ReLU (PReLU)** 是不错的替代方案。它们为负输入引入了一个小的非零斜率。
- **Tanh** 在隐藏层中可能很有效，尤其是在 RNN 中，因为它有零中心输出。
- **Sigmoid** 通常保留用于 *输出层*，当你需要用于二元或多标签分类的概率时。因为梯度消失问题，避免在深度隐藏层中大量使用它。

通常需要进行实验，以找到适用于特定架构和数据集的最佳激活函数。在 PyTorch 中，更换激活函数很简单，通常只需更改一行代码，即激活模块实例化或在 `nn.Module` 的 `forward` 方法中被调用的位置。

获取即时帮助、个性化解释和交互式代码示例。

---

### 简单模型的顺序容器

### 简单模型的顺序容器

虽然通过子类化`torch.nn.Module`定义自定义网络架构提供了最大程度的灵活性，但许多常见模型都涉及层的一个直观序列，其中一层的输出直接作为下一层的输入。对于这些线性堆叠，PyTorch提供了一个方便的容器：`torch.nn.Sequential`。

`nn.Sequential`作为一个包装器，接收一个有序的模块序列（如层和激活函数 (activation function)），并在输入数据通过时，以该特定顺序执行它们。可以将其视为为您的数据转换构建一个流程。当您不需要复杂的数据流逻辑、跳跃连接或多输入/多输出路径时，这种方法简化了模型定义。

### 使用`nn.Sequential`定义模型

您可以通过将您想要包含的模块作为参数 (parameter)传递给其构造函数来创建`Sequential`模型。顺序很重要，因为它决定了数据流。

让我们构建一个简单的两层前馈网络，它接收一个784维的输入（例如扁平化的MNIST图像），通过一个包含128个单元和ReLU激活的隐藏层，最后生成一个10维输出（用于10个类别）。

```python
import torch
import torch.nn as nn
from collections import OrderedDict

input_size = 784
hidden_size = 128
output_size = 10

model_v1 = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
)

print("Model V1 (Unnamed Layers):")
print(model_v1)

dummy_input = torch.randn(64, input_size)
output = model_v1(dummy_input)
print("\nOutput shape:", output.shape)
```

这创建了一个模型，其中输入数据首先经过`nn.Linear(784, 128)`，然后应用`nn.ReLU()`激活，最后结果通过`nn.Linear(128, 10)`。请注意，定义是多么紧凑。`Sequential`容器自动处理将一个模块的输出作为下一个模块的输入传递。

### 在`nn.Sequential`中命名层

尽管前一种方法可行，但层只被分配了默认的数字索引（`0`、`1`、`2`等）。这可能会使得后续调试或访问特定层变得更难。为了清晰度和可访问性，更好的做法是使用Python `collections`模块中的`OrderedDict`来为您的层提供名称。

```python

model_v2 = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_size)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(hidden_size, output_size))
]))

print("\nModel V2 (Named Layers):")
print(model_v2)

print("\nAccessing fc1 weights shape:", model_v2.fc1.weight.shape)

print("Accessing layer at index 0:", model_v2[0])

print("Accessing layer by name 'relu1':", model_v2.relu1)
```

使用`OrderedDict`保留了插入顺序（这对`nn.Sequential`非常重要），同时允许您引用诸如`model_v2.fc1`或`model_v2.relu1`之类的层。这显著提高了代码的可读性和可维护性，特别是对于稍长的序列，使得检查模型的特定部分变得更容易。


clusterₛequential

nn.Sequential 容器

Input

输入
(批量, 784)

Layer1

fc1: nn.Linear(784, 128)

Input->Layer1

Activation1

relu1: nn.ReLU()

Layer1->Activation1

Layer2

fc2: nn.Linear(128, 10)

Activation1->Layer2

Output

输出
(批量, 10)

Layer2->Output

> 数据流经使用命名层通过`nn.Sequential`定义的`model_v2`。输入按线性顺序通过`fc1`、`relu1`和`fc2`。

### 何时使用`nn.Sequential`

`nn.Sequential`特别适合于：

1. **简单的全连接网络：** 模型中的层线性堆叠，没有分支或跳过，例如基本的多层感知机（MLP）或某些CNN的初始特征提取阶段。
2. **定义可复用模块：** 创建自包含的层模块（例如，一个包含`Conv2d`、`BatchNorm2d`和`ReLU`的卷积模块），然后可以将其作为单个模块整合到更大的自定义`nn.Module`结构中。
3. **快速原型开发：** 快速组装标准架构以测试想法或建立基线。

### 局限性

`nn.Sequential`的主要局限在于其严格的线性性质。它假定一个单一输入和一个单一输出，数据顺序流经所有包含的模块。您不能直接使用它来定义具有更复杂拓扑的模型，例如：

- **跳跃连接：** 像ResNet这样的架构，其中较早层的输出被添加到较晚层的输出中，这需要在自定义的`forward`方法中显式实现。
- **多输入或多输出：** 处理几个不同输入流或生成多个输出张量的模型不能仅凭`nn.Sequential`来表示。
- **共享层：** 架构中完全相同的层实例在网络拓扑的不同点被应用。
- **条件逻辑：** 任何数据流依赖于运行时条件或需要对从一层的输出到下一层输入的数据进行操作的场景。

对于任何表现出这些特点的架构，您必须通过子类化`torch.nn.Module`并自行实现`forward`方法来定义一个自定义模型，这将给予您对数据流的完全控制，如前所述在“定义自定义网络架构”一节中讨论的。

总之，`nn.Sequential`提供了一种清晰高效的方法来定义线性堆叠神经网络 (neural network)层的常见模式。它是一个有价值且方便的工具，适用于更简单的架构和组件模块，补充了自定义`nn.Module`类这种更灵活的方法。现在您可以使用`nn.Module`或`nn.Sequential`定义模型结构了，下一步是定义模型将优化的目标函数，这将引出损失函数 (loss function)。

获取即时帮助、个性化解释和交互式代码示例。

---

### 损失函数 (torch.nn损失)

### 损失函数 (`torch.nn` 损失)

神经网络 (neural network)，由堆叠的层和激活函数 (activation function)构建而成，进行预测。为了使这些网络有效学习，必须有一种方法来量化 (quantization)其预测值与真实目标值之间的偏差有多大。这种量化是**损失函数 (loss function)**的主要作用，损失函数也称为准则函数或目标函数。

`torch.nn` 包提供了深度学习 (deep learning)中常用的一系列标准损失函数。基本思路很简单：损失函数接收模型的输出（预测值）和真实值（目标值）作为输入，并计算出一个标量值，表示“误差”或“损失”。然后，PyTorch 的 Autograd 系统在反向传播 (backpropagation)过程中使用这个标量损失值来计算梯度，这些梯度进而指导优化器（如 `torch.optim` 中的 SGD 或 Adam）如何调整模型的参数 (parameter)（权重 (weight)和偏置 (bias)）以最小化此损失。

选择合适的损失函数非常重要，因为它直接定义了模型试图达成的目标。让我们看看 `torch.nn` 中一些最常用的损失函数。


InputData

输入数据 (X)

Model

模型 (nn.Module)

InputData->Model

Predictions

预测值 (ŷ)

Model->Predictions

LossFunction

损失函数 (例如，nn.MSELoss)

Predictions->LossFunction

TargetData

目标数据 (y)

TargetData->LossFunction

LossValue

标量损失值

LossFunction->LossValue

 用于
反向传播

> 损失函数比较模型预测值与目标数据，以生成一个标量损失值，该值通过反向传播指导参数更新。

### 常用损失函数 (loss function)

PyTorch 将损失函数实现为继承自 `nn.Module` 的类。你首先实例化损失函数类，然后使用模型的预测值和目标值调用该实例。

#### 回归损失

这些通常用于目标是预测连续值的情况。

1. **均方误差 (MSELoss)**: 可能是回归任务中最常用的损失函数。它衡量预测值和实际值之间差异的平方的平均值。

   公式如下：

   损失(y,y^)=1N∑i=1N(yi−y^i)2\text{损失}(y, \hat{y}) = \frac{1}{N} \sum\_{i=1}^{N} (y\_i - \hat{y}\_i)^2损失(y,y^​)=N1​i=1∑N​(yi​−y^​i​)2

   其中 NNN 是批次中的样本数量，yiy\_iyi​ 是真实值，y^i\hat{y}\_iy^​i​ 是预测值。对差异进行平方会更严厉地惩罚较大的误差。

   使用 `torch.nn.MSELoss`：

   ```python
   import torch
   import torch.nn as nn

   loss_fn = nn.MSELoss()

   predictions = torch.randn(3, 1, requires_grad=True)
   targets = torch.randn(3, 1)

   loss = loss_fn(predictions, targets)
   print(f"MSE Loss: {loss.item()}")
   ```
2. **平均绝对误差 (L1Loss)**: 另一种常用的回归损失。它衡量预测值和实际值之间绝对差异的平均值。

   公式如下：

   损失(y,y^)=1N∑i=1N∣yi−y^i∣\text{损失}(y, \hat{y}) = \frac{1}{N} \sum\_{i=1}^{N} |y\_i - \hat{y}\_i|损失(y,y^​)=N1​i=1∑N​∣yi​−y^​i​∣

   与 MSE 相比，L1 损失通常被认为对异常值不那么敏感，因为它不对误差进行平方。

   使用 `torch.nn.L1Loss`：

   ```python
   import torch
   import torch.nn as nn

   loss_fn_l1 = nn.L1Loss()
   predictions = torch.tensor([[1.0], [2.5], [0.0]], requires_grad=True)
   targets = torch.tensor([[1.2], [2.2], [0.5]])

   loss_l1 = loss_fn_l1(predictions, targets)
   print(f"L1 Loss: {loss_l1.item()}")
   ```

#### 分类损失

这些用于目标是预测离散类别标签的情况。

1. **交叉熵损失 (CrossEntropyLoss)**: 这是多类别分类问题的标准损失函数。当你的模型为每个类别输出原始分数（logits）时，它特别有效。

   `torch.nn.CrossEntropyLoss` 方便地将两个步骤合二为一：

   - 对模型的原始输出分数（logits）应用 `LogSoftmax` 函数。Softmax 将 logits 转换为和为 1 的概率，而 LogSoftmax 则取这些概率的对数。
   - 计算 LogSoftmax 输出和目标类别索引之间的负对数似然损失 (`NLLLoss`)。

   它需要：

   - **输入（预测值）：** 每个类别的原始、未归一化 (normalization)分数（logits）。形状通常为 `(N, C)`，其中 `N` 是批大小，`C` 是类别数量。
   - **目标：** 类别索引（从 0 到 C-1 的整数）。形状通常为 `(N)`。

   ```python
   import torch
   import torch.nn as nn

   loss_fn_ce = nn.CrossEntropyLoss()

   predictions_logits = torch.randn(3, 5, requires_grad=True)

   targets_classes = torch.tensor([1, 0, 4])

   loss_ce = loss_fn_ce(predictions_logits, targets_classes)
   print(f"Cross-Entropy Loss: {loss_ce.item()}")
   ```

   通常推荐使用 `nn.CrossEntropyLoss`，而不是手动应用 `LogSoftmax` 和 `NLLLoss`，因为前者具有更好的数值稳定性。
2. **二元交叉熵损失 (BCELoss 和 BCEWithLogitsLoss)**: 用于二元（两类别）分类问题或多标签分类（每个样本可以属于多个类别）。

   - `torch.nn.BCELoss`: 计算目标与输出之间的二元交叉熵。它期望模型的输出已经是概率（例如，在应用 Sigmoid 激活函数 (activation function)之后），通常在 [0, 1] 范围内。

     - 输入（预测值）：概率，形状 `(N, *)`。
     - 目标：概率（通常为 0.0 或 1.0），与输入形状相同。
   - `torch.nn.BCEWithLogitsLoss`: 这个版本在数值上比先使用 Sigmoid 层再使用 `BCELoss` 更稳定和方便。它将 Sigmoid 激活和 BCE 计算合为一步。它期望原始 logits 作为输入。

     - 输入（预测值）：原始 logits（Sigmoid 之前），形状 `(N, *)`。
     - 目标：概率（通常为 0.0 或 1.0），与输入形状相同。

   对于大多数二元分类任务，`BCEWithLogitsLoss` 是更优选择：

   ```python
   import torch
   import torch.nn as nn

   loss_fn_bce_logits = nn.BCEWithLogitsLoss()

   predictions_logits_bin = torch.randn(4, 1, requires_grad=True)

   targets_bin = torch.tensor([[1.0], [0.0], [0.0], [1.0]])

   loss_bce = loss_fn_bce_logits(predictions_logits_bin, targets_bin)
   print(f"BCE With Logits Loss: {loss_bce.item()}")
   ```

### 选择合适的损失函数 (loss function)

选择很大程度上取决于你的具体任务：

- **回归（预测连续值）：** 从 `nn.MSELoss` 开始。如果你怀疑异常值严重影响训练，可以考虑 `nn.L1Loss`。
- **二元分类（两个类别）：** 使用 `nn.BCEWithLogitsLoss`。确保你的模型有一个输出节点产生 logits。
- **多类别分类（每个样本只有一个正确类别）：** 使用 `nn.CrossEntropyLoss`。确保你的模型有 `C` 个输出节点产生 logits，其中 `C` 是类别数量。
- **多标签分类（每个样本可以有多个正确类别）：** 使用 `nn.BCEWithLogitsLoss`。确保你的模型有 `C` 个输出节点产生 logits，并且你的目标是多热编码（例如，如果存在类别 0 和 2，则为 `[1.0, 0.0, 1.0, 0.0]`）。

### 在训练中使用损失函数 (loss function)

在典型的训练循环中，你会在循环外部实例化一次所选择的损失函数。在循环内部，获取模型对一批数据的预测后，将预测值和相应的目标标签传递给损失函数实例，以计算该批次的损失。

```python

num_classes = 10
model = nn.Linear(784, num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

dummy_dataloader = [(torch.randn(64, 784), torch.randint(0, num_classes, (64,))) for _ in range(5)]

model.train()
for batch_idx, (data, target) in enumerate(dummy_dataloader):

    optimizer.zero_grad()

    predictions = model(data)

    loss = loss_fn(predictions, target)

    loss.backward()

    optimizer.step()

    if batch_idx % 2 == 0:
        print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
```

通过从 `torch.nn` 中选择合适的损失函数并将其正确集成到你的训练过程中，你为模型提供了一个明确的学习目标，与优化器和反向传播 (backpropagation)一起构成了模型训练机制的根本部分。

获取即时帮助、个性化解释和交互式代码示例。

---

### 优化器 (torch.optim)

### 优化器 (`torch.optim`)

优化器在神经网络 (neural network)训练中扮演着重要角色。在定义神经网络架构（例如使用 `torch.nn.Module`）并选定一个合适的损失函数 (loss function)来衡量模型预测与实际目标之间的差异后，优化器的作用是更新模型的参数 (parameter)（如权重 (weight)和偏置 (bias)），以最小化计算出的损失。`torch.optim` 包为深度学习 (deep learning)中常用的多种优化算法提供了实现。

回顾 Autograd 一章，调用 `loss.backward()` 会计算损失相对于所有 `requires_grad=True` 的模型参数的梯度。这些梯度指示了每个参数为减少损失所需的改变方向和大小。然而，仅仅计算梯度是不够的；我们需要一种机制来*应用*这些更新。优化器提供了这种机制。

## 优化过程

核心来说，训练神经网络 (neural network)是一个优化问题。我们希望找到一组参数 (parameter)（权重 (weight) www 和偏置 (bias) bbb）来最小化损失函数 (loss function) LLL。梯度下降 (gradient descent)是实现这一目的的基础算法。基本思路是迭代地沿着梯度的反方向调整参数：

θnew=θold−η∇θL\theta\_{new} = \theta\_{old} - \eta \nabla\_{\theta} Lθnew​=θold​−η∇θ​L

这里，θ\thetaθ 代表一个参数（如权重或偏置），∇θL\nabla\_{\theta} L∇θ​L 是损失 LLL 对 θ\thetaθ 的梯度，而 η\etaη (eta) 是学习率，一个控制步长的超参数 (hyperparameter)。

PyTorch 的 `torch.optim` 包实现了这一核心思想，以及一些旨在提高收敛速度和稳定性的更精巧变体。

## 使用 `torch.optim`

要在 PyTorch 中使用优化器，首先需要导入该包：

```python
import torch.optim as optim
```

接下来，实例化一个优化器对象。创建时，你必须告诉优化器它应该管理哪些参数 (parameter)。通常，你会使用 `model.parameters()` 方法传入模型的参数。你还需要指定学习率 (`lr`) 以及其他可能与算法相关的超参数 (hyperparameter)。

```python

optimizer = optim.SGD(model.parameters(), lr=0.01)

optimizer = optim.Adam(model.parameters(), lr=0.001)
```

`model.parameters()` 调用会返回一个迭代器，遍历你的模型中所有可学习的参数。优化器持有对这些张量的引用，并知道如何根据它们的 `.grad` 属性（该属性在 `loss.backward()` 调用期间填充）来更新它们。

## 常用优化器

尽管 `torch.optim` 提供了许多算法，但随机梯度下降 (gradient descent) (SGD) 和 Adam 是两个最常用的起始点。

### 随机梯度下降 (gradient descent) (SGD)

SGD 是一种经典的优化算法。在其 PyTorch 实现中，它可以在小批量数据（这是标准做法）而非单个样本上运行。它根据当前小批量数据计算出的梯度更新参数 (parameter)。

`optim.SGD` 优化器有几个重要参数：

- `params`: 要优化的参数的可迭代对象（例如，`model.parameters()`）。
- `lr`: 学习率 (η\etaη)。这是一个关键的超参数 (hyperparameter)。选择过小的值可能导致收敛缓慢，而过大的值可能导致不稳定或发散。
- `momentum`: 一种有助于加速 SGD 朝相关方向前进并抑制振荡的方法。它将之前更新向量 (vector)的一部分添加到当前更新向量中。典型值为 0.9。
- `weight_decay`: 在更新步骤中隐式地向损失函数 (loss function)添加 L2 正则化 (regularization)（对大权重 (weight)的惩罚）。这有助于防止过拟合 (overfitting)。

```python

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
```

### Adam (自适应矩估计)

Adam 是一种自适应学习率优化算法，这意味着它为不同参数 (parameter)计算各自的学习率。它结合了 RMSprop（根据最近梯度平方的平均值调整学习率）和动量的思想。Adam 通常比 SGD 收敛更快，且相对有效，常常在默认设置下表现良好。

`optim.Adam` 的重要参数：

- `params`: 要优化的参数。
- `lr`: 初始学习率（Adam 在内部进行调整）。常见的起始值是 `1e-3` 或 `0.001`。
- `betas`: 一个元组 `(beta1, beta2)`，控制动量估计的指数衰减率（通常是 `(0.9, 0.999)`）。
- `eps`: 添加到分母中的一个小项，用于数值稳定性（通常是 `1e-8`）。
- `weight_decay`: 添加 L2 正则化 (regularization)。

```python

optimizer = optim.Adam(model.parameters(), lr=0.001)

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)
```

其他常用优化器，如 `RMSprop`、`Adagrad` 和 `AdamW`（改进了权重 (weight)衰减处理的 Adam）也在 `torch.optim` 中提供。选择通常取决于具体问题和实际表现。

## 将优化器整合到训练循环中

在训练循环中使用优化器涉及每次迭代的两个主要步骤，通常在计算损失和梯度之后执行：

1. **`optimizer.zero_grad()`**: 在计算当前小批量数据的梯度（通过 `loss.backward()`）之前，必须清除之前迭代累积的梯度。PyTorch 默认在每次调用 `backward()` 时累积梯度。如果你忘记将其归零，来自多个批次的梯度将会混合，导致不正确的更新。这通常在循环开始时或在调用 `backward()` 之前执行。
2. **`optimizer.step()`**: 在用 `loss.backward()` 计算梯度后，调用 `optimizer.step()` 会更新所有已在优化器中注册的参数 (parameter)。它会使用计算出的梯度（存储在 `parameter.grad` 中）和学习率来应用特定的优化算法（如 SGD 或 Adam）。

以下是整合了优化器的训练迭代的简化结构：

```python

model.train()
for inputs, targets in data_loader:

    optimizer.zero_grad()

    outputs = model(inputs)

    loss = criterion(outputs, targets)

    loss.backward()

    optimizer.step()
```

下图展示了优化器在标准训练周期中的作用：

TrainingLoop

cluster\_Loop

训练迭代

ZeroGrad

optimizer.zero\_grad()

Forward

前向传播
(模型输出)

ZeroGrad->Forward

 下一次迭代

Loss

计算损失
(准则)

Forward->Loss

 下一次迭代

modelₚarams

模型参数
(权重, 偏置)

Backward

loss.backward()
(计算梯度)

Loss->Backward

 下一次迭代

Step

optimizer.step()
(更新权重)

Backward->Step

 下一次迭代

gradients

参数梯度
(.grad 属性)

Backward->gradients

 填充

Step->ZeroGrad

 下一次迭代

Step->modelₚarams

 修改

optimizer

优化器
(例如, Adam, SGD)

modelₚarams->optimizer

 初始化时传入

gradients->Step

 读取

optimizer->ZeroGrad

 清除 .grad

> 优化器使用 `loss.backward()` 计算出的梯度，通过 `optimizer.step()` 更新模型参数，在此之前确保使用 `optimizer.zero_grad()` 清除了之前的梯度。

## 调整学习率

有时，在训练期间调整学习率是有益的。例如，你可能希望以较大的学习率开始以加快初始进展，之后再降低学习率以更精细地调整参数 (parameter)。PyTorch 在 `torch.optim.lr_scheduler` 中为此提供了学习率调度器。这些调度器根据预设规则（例如，每隔几个 epoch 减少一次，或者当验证性能稳定时）调整与优化器相关的学习率。尽管它们功能强大，但调度器的详细用法通常在更高级的背景下介绍。

总结来说，`torch.optim` 是 PyTorch 中训练神经网络 (neural network)不可或缺的工具。通过选择合适的优化器、配置其学习率等超参数 (hyperparameter)，并将 `optimizer.zero_grad()` 和 `optimizer.step()` 正确整合到训练循环中，你便为模型提供了从数据中学习并最小化损失的机制。尝试不同的优化器和学习率是开发有效深度学习 (deep learning)模型的常规部分。

获取即时帮助、个性化解释和交互式代码示例。

---

### 练习：构建一个简单网络

### 练习：构建一个简单网络

让我们将本章的理念付诸实践，通过搭建一个简单的神经网络 (neural network)。我们将构建一个小型的前馈网络，专为二分类任务而设计。假设我们有包含两个特征的输入数据，并想将每个数据点归入两个类别（0或1）中的一个。

任何PyTorch模型的基础都是`torch.nn.Module`类。通过继承`nn.Module`可以创建自定义网络，并在`__init__`方法中定义层，在`forward`方法中定义数据流。

### 定义网络架构

我们将创建一个具有以下特点的网络：

1. 一个接受2个特征的输入层。
2. 一个包含10个神经元和ReLU激活函数 (activation function)的隐藏层。
3. 一个包含1个神经元的输出层，生成适合二分类的单个logit值。

以下是定义此架构的Python代码：

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)

        return out

input_features = 2
hidden_units = 10
output_classes = 1

model = SimpleNet(input_features, hidden_units, output_classes)

print(model)
```

运行此代码将打印我们新定义网络的结构，显示层及其顺序：

```
SimpleNet(
  (layer_1): Linear(in_features=2, out_features=10, bias=True)
  (relu): ReLU()
  (layer_2): Linear(in_features=10, out_features=1, bias=True)
)
```

此输出证实我们有一个线性层，将2个输入特征映射到10个隐藏单元，接着是ReLU激活，最后是另一个线性层，将10个隐藏单元映射到单个输出值。

### 准备训练组件

在我们开始训练之前（训练的详细内容将在后面介绍），我们需要实例化模型，定义损失函数 (loss function)，并选择一个优化器。让我们来设置这些。

```python

dummy_input = torch.randn(5, input_features)

dummy_labels = torch.randint(0, 2, (5, 1)).float()

criterion = nn.BCEWithLogitsLoss()

learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"\n使用的损失函数: {criterion}")
print(f"使用的优化器: {optimizer}")
```

### 模拟前向和反向传播 (backpropagation)

现在我们已经准备好模型、损失函数 (loss function)和优化器，让我们模拟训练过程中的单个步骤，来查看这些组件如何协同工作。这包括：

1. 将输入数据通过模型（前向传播）。
2. 计算模型输出与真实标签之间的损失。
3. 使用反向传播计算梯度。
4. 使用优化器更新模型的权重 (weight)。

```python

outputs = model(dummy_input)
print(f"\n模型输出（logits）形状：{outputs.shape}")

loss = criterion(outputs, dummy_labels)
print(f"计算的损失：{loss.item():.4f}")

optimizer.zero_grad()
loss.backward()

optimizer.step()
```

在此步骤中，我们执行了前向传播，从我们的`SimpleNet`获取原始输出（logits）。然后我们使用`BCEWithLogitsLoss`计算这些输出和我们的`dummy_labels`之间的差异。调用`loss.backward()`触发了Autograd计算所有`requires_grad=True`参数 (parameter)的梯度（默认包括我们`nn.Linear`层的权重和偏置 (bias)）。最后，`optimizer.step()`使用计算出的梯度和Adam优化算法更新了模型的参数。请记住在实际训练循环中的下一次反向传播之前调用`optimizer.zero_grad()`，以防止梯度累积。

您现在已成功使用`torch.nn`构建了一个简单的神经网络 (neural network)，定义了其层和前向传播，实例化了它，并将其与损失函数和优化器连接起来，为下一阶段：训练做好了准备。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 5 Efficient Data Handling

### 对专用数据加载器的需求

### 对专用数据加载器的需求

训练深度学习 (deep learning)模型需要处理大量数据。虽然使用 `torch.nn` 构建模型和使用 Autograd 计算梯度是基本步骤，但一个实际问题随之而来：如何在训练期间高效地将数据输入这些模型？

如果尝试手动处理数据加载，会遇到以下挑战：

1. **内存限制**: 现代数据集，特别是在计算机视觉或自然语言处理等方面，可能非常庞大，常常超出可用内存（RAM），更不用说 GPU 上的显存（VRAM）。一次性将整个数据集加载到内存中通常是不可行的。想象一下，尝试将整个 ImageNet 数据集（超过 1400 万张图像，数百 GB）直接加载到计算机的 RAM 中——对于大多数系统来说，这根本无法容纳。
2. **I/O 瓶颈**: 从磁盘读取数据的速度比 CPU 或 GPU 上的计算慢几个数量级。如果模型需要数据时你逐个加载数据样本，你速度极快的 GPU 将大部分时间处于空闲状态，等待下一批数据到来。这种顺序磁盘读取会成为一个主要瓶颈，极大地减缓训练过程。
3. **低效的预处理**: 数据很少以神经网络 (neural network)所需的精确格式存在。它通常需要预处理步骤，例如归一化 (normalization)、调整大小、数据类型转换或数据增强（随机修改样本以提高模型泛化能力）。与主要训练过程同步地逐个样本执行这些转换，会增加进一步的延迟。
4. **洗牌需求**: 为确保模型泛化能力并防止与数据顺序相关的偏差，标准做法是在每个训练周期前对数据集进行洗牌。实现高效的洗牌，特别是对于无法完全放入内存的数据集，会增加复杂性。
5. **批处理**: 神经网络通常在数据的小批量上进行训练，而不是单个样本。分批处理数据可以获得更稳定的梯度估计，并更好地使用 GPU 的并行处理能力。手动创建这些批次，确保它们的格式正确，以及处理最后一个可能较小的批次，都需要仔细编写代码。
6. **并行处理**: 为克服 I/O 瓶颈，高效的数据加载管道通常使用多个工作进程并行加载和预处理数据，在 GPU 忙于处理当前批次时准备未来的批次。正确实现这种并行，管理进程，并确保数据完整性是一项复杂的工程任务。

为每个项目从头解决所有这些问题会非常耗时且容易出错。你每次都相当于在重建一个重要的基础设施部分。


clusterₙaive

简化版朴素加载

clusterₚytorch

PyTorch DataLoader 方法

Dataset

大型数据集
(磁盘)

LoadSample

加载 + 处理
样本 (CPU)

Dataset->LoadSample

慢速读取

ToGPU

将样本
移至 GPU

LoadSample->ToGPU

Bottleneck

I/O 和 CPU 限制
GPU 常空闲

TrainStep

在样本上
训练 (GPU)

ToGPU->TrainStep

TrainStep->LoadSample

等待下一批

Bottleneck->TrainStep

减缓训练

PyDataset

大型数据集
(磁盘)

DataLoader

DataLoader
(并行工作器,
批处理, 洗牌)

PyDataset->DataLoader

BatchQueue

已准备的批次
(RAM)

DataLoader->BatchQueue

预取

ToGPUBatch

将批次
移至 GPU

BatchQueue->ToGPUBatch

TrainBatch

在批次上训练
(GPU 忙碌)

ToGPUBatch->TrainBatch

TrainBatch->BatchQueue

请求下一批

> 比较了导致瓶颈的朴素顺序数据加载方法与 PyTorch 数据工具提供的并行批处理方法。

认识到这些常见且重要的挑战，PyTorch 提供了 `torch.utils.data` 模块。这个模块提供专用工具，专门用于构建高效、灵活和并行的数据加载管道。它封装了洗牌、批处理、内存管理和并行加载的复杂性，让你能专注于定义数据集结构和所需的转换。

通过使用 PyTorch 的 `Dataset` 和 `DataLoader` 类（我们将在后续章节中介绍），你将获得：

- **效率**: 优化数据获取和预处理，通常在 CPU 核心间并行执行，确保 GPU 获得充足数据。
- **内存管理**: 通过仅在需要时将必要的批次加载到内存中来处理大型数据集。
- **灵活性**: 轻松集成自定义数据源和复杂的预处理/数据增强步骤。
- **简洁性**: 用于与数据集交互和创建数据迭代器的标准化 API。

这些工具是使用 PyTorch 构建实际深度学习应用的基本组成部分。让我们从 `Dataset` 类开始，了解它们如何工作。

获取即时帮助、个性化解释和交互式代码示例。

---

### 使用torch.utils.data.Dataset

### 使用 `torch.utils.data.Dataset`

高效加载和处理数据对训练深度学习 (deep learning)模型非常重要。PyTorch 提供了一种通过其 `torch.utils.data.Dataset` 抽象类来处理数据集的标准化方式。可以把 `Dataset` 看作一个约定：它定义了访问数据的标准接口，无论数据是存在内存中、磁盘上，还是需要即时生成。

### `Dataset` 抽象类

其核心是，`torch.utils.data.Dataset` 是一个表示数据集的抽象类。你在 PyTorch 中创建的任何自定义数据集都应该继承自这个类。为什么要使用这种结构？它确保了不同的数据集，无论是内置的还是自定义的，都能向其他 PyTorch 组件提供一致的 API，最值得一提的是 `DataLoader`，我们稍后会讲到。这种标准化简化了在相同训练代码中替换数据集或使用不同数据源的过程。

要创建自己的自定义数据集，你需要继承 `torch.utils.data.Dataset` 并重写两个必要的方法：

1. `__len__(self)`: 这个方法应该返回数据集中样本的总数。`DataLoader` 使用它来确定数据集的大小。
2. `__getitem__(self, idx)`: 这个方法负责根据给定索引 `idx` 从数据集中加载并返回一个样本。这是实际数据加载逻辑所在的地方（例如，读取图像文件、从 CSV 获取一行数据、访问列表中的元素）。`DataLoader` 会重复调用此方法来构建批次。

让我们用一个简单的例子来说明这一点。假设你的特征和对应的标签存储在 Python 列表或 NumPy 数组中。

```python
import torch
from torch.utils.data import Dataset
import numpy as np

class SimpleCustomDataset(Dataset):
    """一个带有特征和标签的简单数据集示例。"""

    def __init__(self, features, labels):
        """
        参数:
            features (列表或 np.array): 特征的列表或数组。
            labels (列表或 np.array): 标签的列表或数组。
        """

        assert len(features) == len(labels), "特征和标签的长度必须相同。"
        self.features = features
        self.labels = labels

    def __len__(self):
        """返回样本总数。"""
        return len(self.features)

    def __getitem__(self, idx):
        """
        生成一个数据样本。

        参数:
            idx (int): 元素的索引。

        返回:
            tuple: 给定索引对应的 (特征, 标签)。
        """

        feature = self.features[idx]
        label = self.labels[idx]

        sample = (torch.tensor(feature, dtype=torch.float32),
                  torch.tensor(label, dtype=torch.long))

        return sample

num_samples = 100
num_features = 10
features_data = np.random.randn(num_samples, num_features)
labels_data = np.random.randint(0, 5, size=num_samples)

my_dataset = SimpleCustomDataset(features_data, labels_data)

print(f"数据集大小: {len(my_dataset)}")

first_sample = my_dataset[0]
feature_sample, label_sample = first_sample
print(f"\n第一个样本特征:\n{feature_sample}")
print(f"第一个样本形状: {feature_sample.shape}")
print(f"第一个样本标签: {label_sample}")

tenth_sample = my_dataset[9]
print(f"\n第十个样本标签: {tenth_sample[1]}")
```

在此示例中：

- `__init__` 方法存储在实例化时传入的特征和标签数据。
- `__len__` 简单地返回特征列表的长度（这与标签列表的长度相同）。
- `__getitem__` 接受一个索引 `idx`，获取对应的特征和标签，将它们转换为 PyTorch 张量，并以元组形式返回。这种转换为张量的操作在 `__getitem__` 中很常见。

### 处理更复杂的场景

自定义 `Dataset` 的真正作用体现在处理那些不能直接在内存中获取的数据时。例如，你的图像文件路径和标签可能存储在一个 CSV 文件中。

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class ImageFilelistDataset(Dataset):
    """用于从 CSV 文件加载图像路径和标签的数据集。"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        参数:
            csv_file (字符串): 包含标注的 CSV 文件路径。
                               假设列有：'image_path', 'label'
            root_dir (字符串): 包含所有图像的目录。
            transform (可调用, 可选): 可选的数据变换，用于对样本进行处理。
                                           应用于样本。
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img_rel_path = self.annotations.iloc[idx, 0]
        img_full_path = os.path.join(self.root_dir, img_rel_path)

        try:
            image = Image.open(img_full_path).convert('RGB')
        except FileNotFoundError:
            print(f"错误：未在 {img_full_path} 找到图像")

            return None, None

        label = self.annotations.iloc[idx, 1]
        label = torch.tensor(int(label), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        if not isinstance(image, torch.Tensor):

             image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, label
```

在这个 `ImageFilelistDataset` 示例中：

- `__init__` 使用 pandas 读取 CSV 文件，并存储文件路径和根目录。它还接受一个可选的 `transform` 参数 (parameter)（我们很快会看到它的用法）。
- `__len__` 返回 CSV 文件中的行数。
- `__getitem__` 构建完整的图像路径，使用 PIL 加载图像，获取标签，应用任何指定的数据变换，确保图像是一个张量，并返回图像张量和标签张量。

请注意，`Dataset` 本身只定义了 *如何* 获取单个项目。它不会一次性将整个数据集加载到内存中（除非你的 `__init__` 明确这样做，但这对于大型数据集通常是避免的）。它也不处理批处理、打乱或并行加载。`DataLoader` 便是为此而生，它直接建立在 `Dataset` 提供的结构之上。通过实现 `__len__` 和 `__getitem__`，你为 `DataLoader` 高效访问数据样本提供了必要的结构。

获取即时帮助、个性化解释和交互式代码示例。

---

### 内置数据集（例如：TorchVision）

### 内置数据集（例如：TorchVision）

尽管创建自定义 `Dataset` 类为您的特定数据提供了最大的灵活性，但许多深度学习 (deep learning)任务，特别是在研究和基准测试中，使用标准化数据集。手动准备这些数据集涉及下载、解压、组织文件和编写解析逻辑，这可能既耗时又容易出错。

幸运的是，PyTorch 提供了配套库，可以简化常见领域的数据处理过程。对于计算机视觉，`torchvision` 包是一个不可或缺的工具。它不仅包含流行的数据集，还包含预训练 (pre-training)模型和常用的图像转换函数。本节主要介绍如何访问和使用 `torchvision.datasets` 提供的数据集。

### 使用 `torchvision.datasets` 访问数据集

`torchvision.datasets` 模块提供了对许多广泛使用的计算机视觉数据集的便捷访问，例如 MNIST、Fashion MNIST、CIFAR 10/100、ImageNet、COCO 等。使用这些数据集很简单。通常，您会从 `torchvision.datasets` 导入特定的数据集类并实例化它。

让我们看一个使用 CIFAR 10 数据集的例子，它包含 60,000 张 32x32 彩色图像，分为 10 个类别。

```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                             train=True,
                                             download=True,
                                             transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform)

print(f"CIFAR-10 training dataset size: {len(train_dataset)}")
print(f"CIFAR-10 test dataset size: {len(test_dataset)}")

img, label = train_dataset[0]
print(f"Image shape: {img.shape}")
print(f"Label: {label}")
```

当您首次运行此代码时，`torchvision` 会检查指定的 `root` 目录（在本例中为 `./data`）。如果 CIFAR 10 数据不存在，设置 `download=True` 会指示 `torchvision` 自动将数据集下载并解压到该目录中。后续运行将发现数据已存在于本地并跳过下载。

注意 `transform` 参数 (parameter)。您可以在此处指定数据预处理步骤，这些步骤在数据样本加载后但在 `__getitem__` 返回之前应用于每个样本。我们使用了 `transforms.ToTensor()`，它将 PIL 图像格式（`torchvision` 数据集常用）转换为 PyTorch 张量。数据转换将在下一节中进行更详细的介绍。

### 结构与使用

重要的是，`torchvision.datasets` 返回的对象（如上文的 `train_dataset` 和 `test_dataset`）是继承自 `torch.utils.data.Dataset` 的类实例。这意味着它们实现了必需的 `__len__` 和 `__getitem__` 方法，使其与 PyTorch 的 `DataLoader` 完全兼容。

- `len(train_dataset)` 返回数据集中样本的总数。
- `train_dataset[i]` 返回第 iii 个样本，通常是一个元组 `(data, target)`，其中 `data` 是预处理后的输入（例如，图像张量），`target` 是对应的标签或标注。

以下是 CIFAR-10 训练集中类别分布的简单可视化：

飞机汽车鸟猫鹿狗青蛙马船卡车010002000300040005000

> CIFAR-10 数据集是平衡的，每个类别恰好有 5,000 张训练图像。

### 在计算机视觉中

尽管 `torchvision` 最为完善，但其他领域也存在类似的库：

- **`torchaudio`**: 为音频处理任务提供数据集（如 SpeechCommands、LJSpeech 等）、模型和转换功能。
- **`torchtext`**: 为自然语言处理提供数据集（如 IMDb 情感分析、WikiText 语言建模）、分词 (tokenization)器 (tokenizer)和词汇工具。注意：`torchtext` 经历了重大的 API 变更，因此请查阅其文档以了解当前的使用模式。

使用这些库遵循相似的原则：导入所需的数据集类，实例化它（通常带有下载和预处理选项），然后将生成的 `Dataset` 对象与 `DataLoader` 一起使用。

依靠这些内置数据集可以显著加快开发和实验速度，使您能够专注于模型架构和训练，而不是数据获取和准备，尤其是在使用标准基准时。请记住，这些数据集对象直接与本章后面讨论的 `DataLoader` 集成，从而实现高效的批处理和洗牌。

获取即时帮助、个性化解释和交互式代码示例。

---

### 数据变换 (torchvision.transforms)

### 数据变换 (`torchvision.transforms`)

原始数据，如图像或文本，很少直接以完美适合神经网络 (neural network)输入的格式出现。模型通常需要特定大小和分布的数值张量。此外，为了提高模型的泛化能力并防止过拟合 (overfitting)，通常的做法是通过对现有数据应用随机修改来人工扩充训练数据集。这就是数据变换的作用。

PyTorch，特别是通过用于计算机视觉任务的 `torchvision` 库，提供了一个便捷的模块 `torchvision.transforms`，它包含多种常用操作，这些操作可以链式组合以创建数据处理流程。这些变换主要有两个作用：

1. **预处理：** 标准化数据格式、比例和大小。
2. **数据增强：** 对训练数据应用随机改动以增加其多样性。

让我们看一些基础变换。

### 常用预处理变换

这些变换通常应用于所有数据集划分（训练集、验证集和测试集），以确保一致性。

- **`transforms.ToTensor()`**：这通常是对使用PIL（Python图像库）或NumPy等库加载的图像数据最先应用的变换之一。它将PIL图像或NumPy数组（格式为高 x 宽 x 通道）转换为PyTorch `FloatTensor`（格式为通道 x 高 x 宽）。重要的是，它还将像素值从 [0, 255] 范围缩放到 [0.0, 1.0]。这种转换为张量和标准化范围对于模型输入是必需的。
- **`transforms.Resize(size)`**：将输入图像调整到给定 `size`。如果 `size` 是整数，图像的较短边将匹配此数字，同时保持宽高比。如果 `size` 是像 `(h, w)` 这样的序列，它会将图像调整为精确的高度 `h` 和宽度 `w`。这很重要，因为许多神经网络 (neural network)需要固定大小的输入。
- **`transforms.CenterCrop(size)`**：将图像的中心部分裁剪到给定 `size`。这通常在调整大小后使用，以确保最终图像尺寸精确，同时聚焦于中心区域。
- **`transforms.Normalize(mean, std)`**：使用为每个通道提供的均值和标准差对张量图像进行标准化。应用的操作是：
  输出=(输入−均值)/标准差输出 = (输入 - 均值) / 标准差输出=(输入−均值)/标准差
  标准化有助于稳定训练，并通过确保输入特征具有相似的比例（通常围绕零居中）来促进更快的收敛。`mean` 和 `std` 通常是值序列，每个输入通道对应一个值（例如，RGB图像有3个值）。来自ImageNet等大型数据集的预计算值通常用作默认值：`mean=[0.485, 0.456, 0.406]` 和 `std=[0.229, 0.224, 0.225]`。

### 常用数据增强变换

这些变换引入随机性，并且通常*仅*应用于训练数据集。这有助于模型学习对输入中的微小变化保持不变，从而降低过拟合 (overfitting)倾向。

- **`transforms.RandomHorizontalFlip(p=0.5)`**：以给定概率 `p`（默认为0.5，表示50%的机会）随机水平翻转图像。
- **`transforms.RandomRotation(degrees)`**：通过从 `(-degrees, +degrees)` 中均匀选择的随机角度旋转图像，或者如果 `degrees` 是序列 `(min, max)`，则在特定范围内旋转。
- **`transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)`**：随机改变图像的亮度、对比度、饱和度和色调。你可以指定每个属性的抖动范围。例如，`brightness=0.2` 意味着随机选择一个介于 `[max(0, 1 - 0.2), 1 + 0.2]` 之间的亮度因子。
- **`transforms.RandomResizedCrop(size)`**：裁剪图像的随机部分并将其调整到所需的 `size`。这是一种常用增强技术，特别适用于训练Inception网络等图像分类模型。

### 组合变换

你很少只应用一个变换。PyTorch 通过 `transforms.Compose` 方便地将多个变换链式组合起来。它接收一个变换对象列表并按顺序应用它们。

下面是为训练数据创建处理流程的示例，包括调整大小、增强、转换为张量和标准化：

```python
import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print("训练变换：")
print(train_transform)

print("\n测试变换：")
print(test_transform)
```

### 将变换与数据集结合

正如上一节关于 `Dataset` 对象所述，这些组合变换通常在实例化 `Dataset` 时作为参数 (parameter)（通常命名为 `transform` 或 `target_transform`）传入。对于 `torchvision.datasets` 中的内置数据集，这直接简单：

```python

from torchvision.datasets import ImageFolder
from pathlib import Path

train_data_path = Path("path/to/your/train_images")
test_data_path = Path("path/to/your/test_images")

train_dataset = ImageFolder(root=train_data_path, transform=train_transform)
test_dataset = ImageFolder(root=test_data_path, transform=test_transform)

sample_image, sample_label = train_dataset[0]
```

对于自定义 `Dataset` 类，您通常会在 `__init__` 方法中接受变换对象，并在返回样本之前在 `__getitem__` 方法中应用它。

```python
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

custom_train_dataset = CustomImageDataset(train_paths, train_labels, transform=train_transform)
custom_test_dataset = CustomImageDataset(test_paths, test_labels, transform=test_transform)
```

通过定义适当的变换并将其集成到您的 `Dataset` 中，您可以确保输入到模型的数据格式正确，并且对于训练数据而言，得到充分增强。这为下一步使用 `DataLoader` 有效地按批加载这些已处理的数据做好了准备。

获取即时帮助、个性化解释和交互式代码示例。

---

### 使用torch.utils.data.DataLoader

### 使用 `torch.utils.data.DataLoader`

尽管 `Dataset` 类提供了一种清晰的方式来抽象对单个数据样本的访问，但对大型数据集逐个样本进行迭代对于训练深度学习 (deep learning)模型通常效率不高。训练通常受益于分批处理数据。在这种情况下，`torch.utils.data.DataLoader` 在高效数据处理中发挥着核心作用。

`DataLoader` 封装了一个 `Dataset`（无论是内置的还是你自定义的实现）并提供了对其的迭代接口。它的主要职责是：

1. **分批处理**：将单个样本分组为小批量。
2. **打乱数据**：在每个周期随机打乱数据，以防止模型学习到样本的顺序并提高泛化能力。
3. **并行数据加载**：使用多个子进程并发加载数据，这可能会显著加快数据处理流程。

### 基本用法和迭代

创建 `DataLoader` 很简单。你主要需要提供 `Dataset` 实例并指定所需的 `batch_size`。

```python
import torch
from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.features = torch.randn(num_samples, 10)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = DummyDataset(num_samples=105)

train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

print(f"Dataset size: {len(dataset)}")
print(f"DataLoader batch size: {train_loader.batch_size}")

for epoch in range(1):
    print(f"\n--- Epoch {epoch+1} ---")
    for i, batch in enumerate(train_loader):

        features, labels = batch
        print(f"Batch {i+1}: Features shape={features.shape}, Labels shape={labels.shape}")
```

运行此代码将展示 `DataLoader` 如何产生数据批次。注意每个批次打印的形状反映了 `batch_size`（除了可能的最后一个批次）。

### 批次策略和 `drop_last`

默认情况下，如果 `Dataset` 中的总样本数不能被 `batch_size` 完全整除，最后一个批次将包含剩余样本，因此会更小。

在我们有 105 个样本、批次大小为 32 的示例中：

- 批次 1: 32 个样本
- 批次 2: 32 个样本
- 批次 3: 32 个样本
- 批次 4: 9 个样本 (105 - 3\*32 = 9)

有时，拥有可变批次大小，尤其是非常小的最后一个批次，可能会影响某些训练动态或特定层的要求（例如训练期间的 BatchNorm 层，尽管 PyTorch 处理得相当好）。如果你希望所有批次都具有精确的 `batch_size`，并丢弃较小的最后一个批次，你可以在创建 `DataLoader` 时设置 `drop_last=True`：

```python

train_loader_drop_last = DataLoader(dataset=dataset, batch_size=32, shuffle=True, drop_last=True)

print("\n--- DataLoader with drop_last=True ---")
for i, batch in enumerate(train_loader_drop_last):
    features, labels = batch
    print(f"Batch {i+1}: Features shape={features.shape}, Labels shape={labels.shape}")
```

### 打乱数据以获得更好的训练

在训练期间强烈建议设置 `shuffle=True`。它告诉 `DataLoader` 在为每个周期创建批次之前重新打乱数据集的索引。这确保模型每次都以不同的顺序查看数据，减少过度拟合数据呈现顺序的风险并提高模型健壮性。对于验证或测试，通常禁用打乱数据（`shuffle=False`），以确保不同运行之间评估指标的一致性。

### 使用 `num_workers` 并行加载数据

数据加载和预处理（应用变换）有时可能会成为瓶颈，特别是当变换很复杂或数据读取涉及大量 I/O 操作时。CPU 可能会花费大量时间准备下一个批次，而 GPU 则空闲等待数据。

`DataLoader` 允许你通过使用多个工作进程并行加载数据来缓解这个问题。你可以使用 `num_workers` 参数 (parameter)来指定工作进程的数量：

```python

fast_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=4)
```

当 `num_workers > 0` 时，`DataLoader` 会生成指定数量的 Python 进程。每个工作进程独立加载一个批次。这使得后续批次的数据加载和变换可以并行发生，而主进程则对当前批次执行模型训练步骤，通常通过更有效地利用 GPU 来显著提高速度。

请注意，增加 `num_workers` 也会增加 CPU 使用率和内存消耗，因为每个工作进程都会加载数据。将其设置过高有时会导致资源争用和收益递减，甚至减慢速度。它通常是根据你的具体硬件和数据集进行调整的参数。


cluster\_workers

并行加载 (如果 num\_workers > 0)

Dataset

Dataset 对象
(YourCustomDataset)

DataLoader

DataLoader
(批次大小, 打乱数据,
 工作进程数)

Dataset->DataLoader

 封装

Worker1

工作进程 1

DataLoader->Worker1

 分配索引

WorkerN

工作进程 N

DataLoader->WorkerN

 分配索引

Batch

批次
(特征, 标签)

Worker1->Batch

 获取并准备

WorkerN->Batch

 获取并准备

TrainingLoop

训练循环
(GPU/CPU)

Batch->TrainingLoop

 提供

TrainingLoop->DataLoader

 请求下一个

> `DataLoader` 的数据加载流程。`DataLoader` 封装了 `Dataset`，并且在 `num_workers > 0` 的情况下，使用工作进程获取并整理样本成批次，这些批次随后被训练循环使用。

### 使用 `pin_memory` 优化 GPU 传输

在 GPU 上训练时，由 `DataLoader` 加载的数据（位于标准 CPU 内存中）需要传输到 GPU 的内存中。这种传输需要时间。你通常可以通过在 `DataLoader` 中设置 `pin_memory=True` 来稍微加快速度。

```python

gpu_optimized_loader = DataLoader(dataset=dataset,
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)
```

设置 `pin_memory=True` 指示 `DataLoader` 在 CPU 端将张量分配到“固定”（页面锁定）内存中。从固定 CPU 内存到 GPU 内存的传输通常比从标准可分页 CPU 内存的传输快。这在与 `num_workers > 0` 一起使用时最有效。请注意，使用固定内存会消耗更多的 CPU 内存。

总之，`DataLoader` 是 PyTorch 中一个核心的工具，它简化并优化了向模型提供数据的过程。通过处理批次化、打乱数据和并行加载，它使你能够专注于模型架构和训练逻辑，同时确保你的数据处理流程高效且可扩展。

获取即时帮助、个性化解释和交互式代码示例。

---

### 自定义DataLoader用法

### 自定义DataLoader用法

“尽管默认的`DataLoader`提供了方便的批处理和随机排列功能，但许多应用需要更精细地控制数据如何抽样和整理成批次。PyTorch通过自定义采样器和`collate`函数提供灵活性，让你能根据具体需求调整数据加载过程，例如处理不平衡数据集或使用可变大小的输入。”

### 使用采样器控制样本选择

`DataLoader`使用`sampler`对象来决定从`Dataset`中抽取索引的顺序。默认情况下，如果`shuffle=True`，它使用`RandomSampler`；如果`shuffle=False`，则使用`SequentialSampler`。但是，你可以通过`sampler`参数 (parameter)显式传入自己的采样器实例（请注意：如果你提供了`sampler`，则必须将`shuffle`设为`False`，因为随机排列是由采样器本身定义的）。

PyTorch在`torch.utils.data`中提供了几种内置采样器：

- `SequentialSampler`：按顺序采样元素，总是以相同的顺序。
- `RandomSampler`：随机采样元素。如果`replacement=True`，则进行有放回采样。
- `SubsetRandomSampler`：从给定索引列表中随机采样元素。它适用于创建验证集划分，而无需修改原始数据集。
- `WeightedRandomSampler`：根据给定概率（权重 (weight)）从`[0,..,len(weights)-1]`中采样元素。这对于处理不平衡数据集特别有用，例如你想对少数类进行过采样或对多数类进行欠采样。

**示例：对不平衡数据使用`WeightedRandomSampler`**

设想一个分类数据集，其中类别“0”有900个样本，类别“1”有100个样本。简单的随机采样会导致批次严重偏向类别“0”。我们可以使用`WeightedRandomSampler`来提高类别“1”样本被选中的概率。

```python
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class_counts = torch.bincount(torch.tensor(targets))
num_samples = len(targets)

sample_weights = torch.tensor([1.0 / class_counts[t] for t in targets])

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)

dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

你也可以创建完全自定义的采样策略，通过继承`torch.utils.data.Sampler`并实现`__iter__`和`__len__`方法。

### 使用`collate_fn`自定义批次创建

一旦采样器为一个批次提供了索引列表，`DataLoader`会使用`dataset[index]`从`Dataset`中获取对应的样本。然后，它需要将这些单独的样本组装成一个批次。这个组装过程由`collate_fn`参数 (parameter)处理。

默认的`collate_fn`在许多标准情况下都能很好地工作。它会尝试：

- 将NumPy数组和Python数字转换为PyTorch张量。
- 保留数据结构（例如，如果你的`Dataset.__getitem__`返回一个字典，则整理后的批次将是一个字典，其中每个值是对应项目的批次）。
- 沿新维度（批次维度）堆叠张量。

但是，如果你的样本具有不同的大小（例如，不同长度的序列）或包含它不知道如何堆叠的数据类型，默认的`collate_fn`可能会失败或产生不理想的结果。

在这种情况下，你可以为`DataLoader`的`collate_fn`参数提供一个自定义函数。这个函数接收一个样本列表（其中每个样本是`Dataset.__getitem__`的输出），并负责以所需格式返回整理后的批次。

**示例：填充可变长度序列**

一个常见情况是涉及长度不同的序列（例如NLP中的句子）。默认的`collate`函数不能直接将它们堆叠成一个张量。自定义的`collate_fn`可以将每个批次中的序列填充到该批次中的最大长度。

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class VariableSequenceDataset(Dataset):
    def __init__(self, data):

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sequence = self.data[idx]
        label = len(sequence)
        return sequence, label

def pad_collate(batch):

    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    labels = torch.tensor(labels)

    return padded_sequences, labels

sequences = [torch.randn(torch.randint(5, 15, (1,)).item()) for _ in range(100)]
dataset = VariableSequenceDataset(sequences)

dataloader = DataLoader(dataset, batch_size=4, collate_fn=pad_collate)
```

这个自定义`collate_fn`使用`torch.nn.utils.rnn.pad_sequence`来处理填充，确保批次中的所有序列长度相同，使它们适合RNN等模型处理。

### 其他DataLoader自定义参数 (parameter)

除了`sampler`和`collate_fn`，其他参数也提供性能和行为调整：

- `num_workers` (整数，可选)：指定用于数据加载的子进程数量。将其设置为正整数可启用多进程数据加载，这可以显著加快数据获取速度，尤其当数据加载涉及磁盘I/O或CPU上的复杂预处理时。一个常见的起始设置是将其设为可用CPU核心的数量。默认值：`0`（数据加载在主进程中进行）。
- `pin_memory` (布尔值，可选)：如果为`True`，`DataLoader`在返回张量之前会将其复制到CUDA固定内存中。固定内存可以加快从CPU到GPU的数据传输。这仅在你使用GPU进行训练时才有效。默认值：`False`。
- `drop_last` (布尔值，可选)：如果为`True`，当数据集大小不能被批次大小整除时，将丢弃最后一个不完整的批次。如果为`False`（默认值），则最后一个批次可能小于`batch_size`。

通过理解和使用采样器、自定义`collate`函数以及其他`DataLoader`参数，你可以对数据管道获得精确的控制，从而能高效处理各种数据类型和结构，解决数据集不平衡问题，并优化数据加载性能以加快模型训练。

获取即时帮助、个性化解释和交互式代码示例。

---

### 动手实践：构建数据处理流程

### 动手实践：构建数据处理流程

构建一个完整的数据处理流程涉及从原始数据（为简化起见，将合成数据）开始，最终得到准备好输入模型的数据批次。这个过程需要创建一个自定义 `Dataset`，定义数据转换，并将所有内容封装在一个 `DataLoader` 中。

### 构建合成数据集

假设我们有一个数据集，包含特征向量 (vector)和对应的二元分类标签（0 或 1）。在本次练习中，我们将使用 PyTorch 张量直接生成这些数据。这避免了文件输入/输出的复杂性，让我们能够专注于数据处理机制。

```python
import torch
import torch.utils.data as data
from torchvision import transforms

num_samples = 100
num_features = 10

features = torch.randn(num_samples, num_features)

labels = torch.randint(0, 2, (num_samples,))

print(f"Shape of features: {features.shape}")
print(f"Shape of labels: {labels.shape}")
print(f"First 5 features:\n{features[:5]}")
print(f"First 5 labels:\n{labels[:5]}")
```

这使我们得到了两个张量：`features` 包含 100 个样本，每个样本有 10 个特征；`labels` 包含对应的 100 个标签。

### 创建自定义 `Dataset`

现在，我们需要使用 PyTorch 的 `Dataset` 类来组织这些数据。我们将创建一个自定义类，它继承自 `torch.utils.data.Dataset` 并实现两个重要方法：

1. `__len__(self)`: 返回数据集中样本的总数。
2. `__getitem__(self, idx)`: 返回指定索引 `idx` 处的样本（特征和标签）。

我们还将添加一个 `__init__` 方法来存储数据并可选地接受转换。

```python
class SyntheticDataset(data.Dataset):
    """一个用于合成特征和标签的自定义数据集。"""

    def __init__(self, features, labels, transform=None):
        """
        参数:
            features (Tensor): 包含特征数据的张量。
            labels (Tensor): 包含标签的张量。
            transform (callable, optional): 可选的样本转换。
        """

        assert features.shape[0] == labels.shape[0], \
            "特征和标签的样本数量必须一致"

        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """返回样本总数。"""
        return self.features.shape[0]

    def __getitem__(self, idx):
        """
        根据给定索引获取特征向量和标签。

        参数:
            idx (int): 要获取的样本索引。

        返回:
            tuple: (特征, 标签)，其中 feature 是特征向量，label 是对应的标签。
        """

        feature_sample = self.features[idx]
        label_sample = self.labels[idx]

        sample = {'feature': feature_sample, 'label': label_sample}

        if self.transform:
            sample = self.transform(sample)

        return sample['feature'], sample['label']

raw_dataset = SyntheticDataset(features, labels)

sample_idx = 0
feature_sample, label_sample = raw_dataset[sample_idx]
print(f"\nSample {sample_idx} - Feature: {feature_sample}")
print(f"Sample {sample_idx} - Label: {label_sample}")
print(f"Dataset length: {len(raw_dataset)}")
```

此时，`raw_dataset` 包含我们的数据并知道如何提供单个样本。

### 定义数据转换

通常，原始数据不适合直接输入神经网络 (neural network)。我们可能需要规范化特征、转换数据类型或应用数据增强（特别是对于图像）。`torchvision.transforms` 提供了方便的工具。即使我们的数据不是图像，我们也可以定义自定义转换或使用对张量进行操作的现有转换。

让我们定义一个简单转换流程：

1. 将特征张量转换为 `torch.float32`（模型输入的良好做法）。
2. 将标签张量转换为 `torch.long`（损失函数 (loss function)如 `CrossEntropyLoss` 常需要的）。
3. 对特征应用规范化（减去均值，除以标准差）。我们将在此示例中从合成数据集中计算这些统计量。

由于 `torchvision.transforms` 主要为图像（PIL 图像或张量）设计，将其直接应用于像我们 `sample` 这样的字典需要一些封装。我们将为此创建自定义可调用类或 lambda 函数。

```python

feature_mean = features.mean(dim=0)
feature_std = features.std(dim=0)

feature_std[feature_std == 0] = 1.0

class ToTensorAndType(object):
    """将特征转换为 FloatTensor，将标签转换为 LongTensor。"""
    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']
        return {'feature': feature.float(), 'label': label.long()}

class NormalizeFeatures(object):
    """规范化特征张量。"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']

        normalized_feature = (feature - self.mean) / self.std
        return {'feature': normalized_feature, 'label': label}

data_transforms = transforms.Compose([
    ToTensorAndType(),
    NormalizeFeatures(mean=feature_mean, std=feature_std)
])

transformed_dataset = SyntheticDataset(features, labels, transform=data_transforms)

sample_idx = 0
transformed_feature, transformed_label = transformed_dataset[sample_idx]

print(f"\n--- 转换后的样本 {sample_idx} ---")
print(f"原始特征:\n{features[sample_idx]}")
print(f"转换后特征:\n{transformed_feature}")
print(f"原始标签: {labels[sample_idx]} (dtype={labels.dtype})")
print(f"转换后标签: {transformed_label} (dtype={transformed_label.dtype})")

print(f"转换后特征均值: {transformed_feature.mean():.4f}")
```

注意到由于规范化，特征值已发生改变，并且特征和标签的数据类型现在分别是 `torch.float32` 和 `torch.int64` (LongTensor)。

### 使用 `DataLoader`

最后一步是使用 `DataLoader`。它接收我们的 `Dataset` 实例，并处理批处理、数据混洗以及可能的并行数据加载。

```python

batch_size = 16
shuffle_data = True
num_workers = 0

data_loader = data.DataLoader(
    transformed_dataset,
    batch_size=batch_size,
    shuffle=shuffle_data,
    num_workers=num_workers
)

print(f"\n--- 迭代 DataLoader (batch_size={batch_size}) ---")

feature_batch, label_batch = next(iter(data_loader))

print(f"Type of feature_batch: {type(feature_batch)}")
print(f"Shape of feature_batch: {feature_batch.shape}")
print(f"Shape of label_batch: {label_batch.shape}")
print(f"Data type of feature_batch: {feature_batch.dtype}")
print(f"Data type of label_batch: {label_batch.dtype}")
```

`DataLoader` 产生批次，其中第一个维度对应于 `batch_size`。我们的特征批次形状为 `[16, 10]`，标签批次形状为 `[16]`。数据类型反映了我们应用的转换。

### 数据处理流程可视化

我们可以可视化刚刚创建的流程：


原始数据

原始数据
(特征, 标签张量)

自定义数据集

自定义数据集
(SyntheticDataset)

原始数据->自定义数据集

 \_ᵢnit\_\_

转换

数据转换
(ToTensor, 规范化)

自定义数据集->转换

 \_\_getitem\_\_ 应用

数据加载器

DataLoader
(批处理, 混洗)

自定义数据集->数据加载器

 输入数据集

转换->自定义数据集

模型输入

模型输入
(批处理和已处理数据)

数据加载器->模型输入

 迭代获取批次

> 此图表展示了从原始张量到自定义 `Dataset` 的演进过程，在数据获取时应用转换（`__getitem__`），最后使用 `DataLoader` 生成适合模型训练的混洗批次。

你现在已成功使用 PyTorch 的核心数据工具构建了数据处理流程。你创建了一个 `Dataset` 来封装你的数据，应用了必要的 `transforms`，并使用 `DataLoader` 高效地生成批次。这种结构化方法是 PyTorch 项目中处理数据的基础，确保你的模型接收正确格式的数据并促进高效训练。这个流程现在已准备好集成到我们将在下一章构建的训练循环中。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 6 Implementing Training Loop

### 训练循环的构成

### 训练循环的构成

训练神经网络 (neural network)是一个迭代优化过程。你为模型提供数据，衡量其预测的准确度，然后微调 (fine-tuning)其内部参数 (parameter)（权重 (weight)和偏置 (bias)），以减少这种不准确性。这个循环会重复多次。管理这种重复过程的代码结构通常被称为**训练循环**。

### 整体结构：周期与批次

从宏观上看，训练通常包含两个嵌套循环：

1. **外层循环（周期）：** 一个周期表示对整个训练数据集的一次完整遍历。训练通常跨越多个周期，让模型能够多次查看并从每个数据样本中学习。周期的数量是你根据模型表现何时稳定或停止提升来选择的一个超参数 (parameter) (hyperparameter)。
2. **内层循环（批次）：** 由于内存限制，一次性处理整个数据集在计算上通常是不可行的。因此，在每个周期内，我们以称为批次的更小片段遍历数据集。你之前学过的`DataLoader`负责提供这些批次。与逐个处理样本或一次性使用整个数据集相比，以批次进行训练内存高效，还能带来更稳定的收敛和更好的泛化能力。

### 每次批次迭代中的核心步骤

对于在一个周期内处理的每个批次，训练循环执行一系列明确定义的步骤。让我们分解一下典型迭代中发生的情况：

1. **获取数据：** 从`DataLoader`获取下一批输入数据（特征）及其对应的目标标签。在此阶段，确保数据被传输到模型参数 (parameter)所在的正确计算设备（CPU或GPU）也很重要。
2. **梯度清零：** 在计算当前批次的梯度之前，你必须明确重置从*上一*次迭代积累的梯度。如果忘记这一步，梯度将在批次间累加，导致不正确的更新，并可能在训练期间发散。通过在优化器对象上调用`zero_grad()`方法来完成。

   ```python

   optimizer.zero_grad()
   ```
3. **前向传播：** 将输入特征批次送入你的模型。模型通过其层处理数据，应用学习到的权重 (weight)和激活函数 (activation function)，最终生成一批预测或输出。

   ```python

   predictions = model(input_batch)
   ```
4. **计算损失：** 使用你选择的损失函数 (loss function)（准则）将模型的`predictions`与真实的`target_batch`进行比较，例如用于分类的`nn.CrossEntropyLoss`或用于回归的`nn.MSELoss`。损失函数返回一个单一的标量值，表示当前批次的平均误差或差异。这个值显示了模型在这个特定批次上的表现好坏。

   ```python

   loss = criterion(predictions, target_batch)
   ```
5. **反向传播 (backpropagation)：** 这是PyTorch的自动微分引擎Autograd计算梯度的地方。调用`$loss.backward()`计算损失标量相对于每个`requires_grad=True`的模型参数的梯度（`nn.Module`中参数的默认设置）。这些梯度表示损失对每个参数变化的敏感度；本质上，它们告诉优化器如何调整每个权重以降低损失。

   ```python

   loss.backward()
   ```
6. **更新权重（优化器步进）：** 计算出梯度后，优化器现在可以调整模型的参数了。调用`$optimizer.step()`根据计算出的梯度和优化器的特定算法（如带动量的SGD、Adam等）更新每个参数。目标是朝着最小化损失的方向迈出一小步。

   ```python

   optimizer.step()
   ```

训练循环中的一次迭代是模型处理数据并更新参数的完整单元。这个循环为`DataLoader`提供的每个批次重复进行。一旦所有批次处理完毕，一个周期就完成了，外层循环开始下一个周期，重复整个批次迭代过程。

TrainingLoop

clusterₑpoch

训练周期

StartBatch

开始批次迭代

GetData

1. 获取批次
(数据 + 标签)

StartBatch->GetData

ZeroGrad

2. 梯度清零
optimizer.zero\_grad()

GetData->ZeroGrad

ForwardPass

3. 前向传播
predictions = model(inputs)

ZeroGrad->ForwardPass

CalcLoss

4. 计算损失
loss = criterion(preds, labels)

ForwardPass->CalcLoss

BackwardPass

5. 反向传播
loss.backward()

CalcLoss->BackwardPass

OptimizerStep

6. 更新权重
optimizer.step()

BackwardPass->OptimizerStep

EndBatch

结束批次迭代
(为下一个批次重复)

OptimizerStep->EndBatch

> 流程图显示了PyTorch训练循环单次批次迭代中的操作顺序。

获取即时帮助、个性化解释和交互式代码示例。

---

### 设置模型、损失函数和优化器

### 设置模型、损失函数和优化器

在进入训练的迭代过程之前，我们需要准备好核心部件：模型本身、衡量其误差的方法（损失函数 (loss function)）以及根据误差更新模型的机制（优化器）。这个准备阶段确保所有必需的组件都已初始化并为训练循环做好了准备。

### 实例化模型

首先，你需要一个神经网络 (neural network)模型的实例。定义自定义网络结构通常涉及继承`torch.nn.Module`。创建一个模型类的对象即可：

```python

model = SimpleNet(input_size=784, hidden_size=128, output_size=10)
print(model)
```

这会创建网络结构，包括其所有层和参数 (parameter)（权重 (weight)和偏置 (bias)）。最初，这些参数具有随机值（或者如果你实现了特定的初始化方案，则由这些方案确定的值）。

#### 将模型移动到正确设备

深度学习 (deep learning)计算，特别是训练，在GPU上速度要快得多。PyTorch使得将模型移动到合适的设备（CPU或GPU）变得简单。一种好的做法是尽早定义目标设备，然后始终将模型和数据都移动到该设备上。

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model.to(device)
```

执行`model.to(device)`会原地修改模型，如果CUDA可用，则将其所有参数和缓冲区移动到GPU内存中，否则保留在CPU上。请记住，任何参与模型计算的张量（如输入数据）也必须位于相同的设备上。我们将在训练循环内部处理数据张量的移动。

### 定义损失函数 (loss function)

损失函数，常被称为准则函数，衡量模型预测与实际目标值之间的距离。PyTorch在`torch.nn`模块中提供了许多标准损失函数。选择哪种损失函数很大程度上取决于你正在解决的问题类型（例如，回归、分类）。

对于多分类问题，`nn.CrossEntropyLoss`很常用。它在一个高效的类中结合了`nn.LogSoftmax`和`nn.NLLLoss`（负对数似然损失）。

```python

loss_fn = torch.nn.CrossEntropyLoss()
```

你像实例化模型一样实例化所选的损失函数。这个`loss_fn`对象稍后将在训练循环中被调用，通常接收模型的输出和真实标签作为输入，以计算一个标量损失值。

### 配置优化器

优化器实现了一种算法（如随机梯度下降 (gradient descent)或Adam），用于根据反向传播 (backpropagation)期间计算的梯度调整模型的参数 (parameter)。其作用是使损失函数 (loss function)最小化。优化器位于`torch.optim`包中。

初始化优化器时，你必须提供两个重要的参数：

1. **模型的参数：** 你告诉优化器它应该更新哪些张量。这可以通过使用`model.parameters()`轻松完成，该方法返回模型中所有可训练参数的迭代器。
2. **学习率（`lr`）：** 这个超参数 (hyperparameter)控制参数更新的步长。找到一个合适的学习率对有效的训练很有帮助。这通常需要尝试。

```python
import torch.optim as optim

learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```

在这里，我们创建了一个SGD优化器实例。它现在持有`model`所有参数的引用，并且知道当其`step()`方法稍后被调用时要使用的学习率。不同的优化器可能还有额外的超参数（如SGD的`momentum`或Adam的`betas`），你可以在初始化时进行配置。

随着模型被实例化并移动到正确设备，损失函数被定义，以及优化器被配置来更新模型的参数，我们已经设置好了所有必需的组件。我们现在准备继续进行训练过程的核心：在训练循环中迭代数据并执行前向传播、损失计算、反向传播和参数更新。

获取即时帮助、个性化解释和交互式代码示例。

---

### 使用 DataLoader 遍历数据

### 使用 DataLoader 遍历数据

`DataLoader` 对象设计用于管理训练数据，处理批处理、数据混洗以及可能的并行加载。它有助于在训练循环中高效地迭代数据。作为 Python 可迭代对象，`DataLoader` 能够简单地在每个训练周期中系统地为模型提供数据批次。

标准做法是使用 Python 的 `for` 循环。在每次迭代中，`DataLoader` 会生成一个数据批次，通常包含输入特征及其对应的目标标签。

```python

num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")

    model.train()

    for batch_idx, data_batch in enumerate(train_dataloader):

        inputs, labels = data_batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        if batch_idx % 100 == 0:
            current_batch_size = len(inputs)

            current_loss = 0.0
            print(f"  Batch {batch_idx}: [{current_batch_size} samples] Current Loss: {current_loss:.4f}")

print("训练完成！")
```

让我们来分析一下这个内部循环的主要部分：

1. **训练周期循环**：外部的 `for epoch in range(num_epochs):` 循环控制着我们遍历整个数据集的次数。
2. **设置训练模式**：`model.train()` 在每个训练周期开始时被调用。这很要紧，因为像 `torch.nn.Dropout` 或 `torch.nn.BatchNorm2d` 这样的层在训练期间（例如，应用 dropout、更新运行统计数据）与评估时有不同的行为。这个调用能确保它们处于正确的模式。
3. **遍历 DataLoader**：`for batch_idx, data_batch in enumerate(train_dataloader):` 是核心的迭代操作。`enumerate` 提供了一个批次计数器 (`batch_idx`)，而 `train_dataloader` 一次生成一个 `data_batch`。
4. **解包数据**：我们将 `data_batch` 解包成 `inputs` 和 `labels`。`DataLoader` 返回的结构与它所包装的 `Dataset` 的 `__getitem__` 方法返回的结构直接对应。对于典型的监督任务，这通常是包含特征和目标的元组或列表。
5. **将数据移动到设备**：`inputs.to(device)` 和 `labels.to(device)` 是不可或缺的步骤。神经网络 (neural network)计算，特别是模型的前向传播，要求模型的参数 (parameter)和输入数据位于*相同的*计算设备上（例如，都位于 CPU，或都位于特定的 GPU）。这一步将 `DataLoader` 获取的批次数据（通常在 CPU 内存中）移动到放置模型的设备上。未能进行此同步是运行时错误的常见原因。这还能确保，如果 `device` 设置为 CUDA 设备，计算可以从 GPU 加速中受益。

值得一提的是，如果你的 `DataLoader` 在初始化时设置了 `drop_last=False`（这是默认设置），一个训练周期中生成的最后一个批次可能包含比指定 `batch_size` 更少的样本。如果数据集中的样本总数不能被批次大小完全整除，就会发生这种情况。PyTorch 操作通常能很好地处理可变批次大小，但如果你进行任何假定固定批次大小的计算（例如对固定数量的损失求平均），请注意这一点。

当数据批次 (`inputs`, `labels`) 成功加载到目标设备后，你现在已为训练迭代循环中的主要计算步骤做好了充分准备：

- 将 `inputs` 送入模型以获得预测（前向传播）。
- 使用损失函数 (loss function)将预测与 `labels` 进行比较（计算损失）。
- 根据损失计算梯度（反向传播 (backpropagation)）。
- 使用优化器更新模型的权重 (weight)（更新权重）。

这些步骤针对每个批次重复执行，构成了模型训练过程的核心，也是后续章节的重点。

获取即时帮助、个性化解释和交互式代码示例。

---

### 前向传播：获取预测结果

### 前向传播：获取预测结果

设置好模型、损失函数 (loss function)、优化器和数据加载器后，训练循环的核心部分便开始了。每次迭代中的第一个操作步骤是**前向传播**。在此步骤中，模型接收当前批次的输入数据并生成预测结果。

可以将前向传播看作信息流经神经网络 (neural network)的过程，从输入层，经过所有隐藏层，最终到达输出层。每个层都会对其从前一个层接收到的数据执行其定义的计算。

### PyTorch中前向传播的工作原理

前向传播在PyTorch中是一个直接的过程。通过继承`torch.nn.Module`，用户可以在模型的`forward`方法中明确定义其操作的结构和顺序。

PyTorch的`nn.Module`基类提供了一个`__call__`方法。这使您能够将模型实例当作函数一样使用。当您调用`model(inputs)`时，PyTorch会隐式地执行您定义的`forward`方法，将`inputs`传递通过网络的各层。

### 实现前向传播

在您的训练循环内部，从`DataLoader`获取一个批次的数据（特征和标签）后，您将特征（输入）送入您的模型：

```python

inputs, labels = data_batch
inputs = inputs.to(device)
labels = labels.to(device)

outputs = model(inputs)
```

以下是此步骤的可视化表示：


DataLoader

DataLoader 批次
(输入, 标签)

Inputs

输入
(在设备上)

DataLoader->Inputs

 提取并
 移至设备

Model

model(inputs)

Inputs->Model

 前向送入

Outputs

输出
(预测/Logits)

Model->Outputs

> 在前向传播过程中数据流向：加载一个批次，输入数据被准备并发送到合适的设备，然后通过模型生成输出。

### 理解输出

前向传播生成的`outputs`张量包含了模型对给定输入批次的预测结果。这些预测结果的具体性质取决于模型的最后一层和任务：

- **分类：** 通常，输出代表每个类别的原始分数，称为 **logits**。这些logits通常会传递给像`nn.CrossEntropyLoss`这样的损失函数 (loss function)，该函数会在内部应用Softmax函数。
- **回归：** 输出可能直接代表预测的连续值。
- **其他任务：** 对于像分割或目标检测这样的任务，输出结构会更复杂，以反映任务的具体要求。

务必确保您的输入数据（`inputs`）和模型（`model`）位于相同的计算设备上（CPU或特定的GPU）。如果它们在不同的设备上，PyTorch会抛出运行时错误。如代码片段所示，使用`.to(device)`将数据批次移动到指定的`device`是防止这种情况发生的标准做法。

前向传播根据模型的当前参数 (parameter)（权重 (weight)和偏差）计算模型的预测结果。这些预测结果随后将在下一步：计算损失中与真实标签进行比较。

获取即时帮助、个性化解释和交互式代码示例。

---

### 计算损失

### 计算损失

模型根据输入数据批次生成预测（通常称为 `outputs` 或 `logits`）。为了评估这些预测与实际真实标签的匹配程度，需要一个评估机制。这就是损失函数 (loss function)的作用。

### 量化 (quantization)模型误差

一个**损失函数 (loss function)**，也称为**标准**或**目标函数**，它以数学方式衡量模型预测（y^\hat{y}y^​）与真实目标值（yyy）之间的差异。训练的目的通常是使这个损失值最小化。损失越小表示模型的预测越接近给定数据批次的实际目标。

在PyTorch中，损失函数在 `torch.nn` 模块中随即可用，就像模型层和激活函数 (activation function)一样。您通常在训练循环外部只实例化一次损失函数。常见选择包括：

- `nn.MSELoss`（均方误差）：常用于回归任务，目标是预测连续值。它计算预测和目标之间的平均平方差。
  LMSE=1N∑i=1N(yi−y^i)2L\_{MSE} = \frac{1}{N} \sum\_{i=1}^{N} (y\_i - \hat{y}\_i)^2LMSE​=N1​i=1∑N​(yi​−y^​i​)2
  其中 NNN 是批次中的样本数量。
- `nn.CrossEntropyLoss`：是多类别分类问题的标准选择。此标准在一个类中方便地结合了 `nn.LogSoftmax` 和 `nn.NLLLoss`（负对数似然损失）。它需要来自模型最后一层的原始、未归一化 (normalization)的分数（logits）作为输入，以及目标类别索引（整数）作为标签。
- `nn.BCEWithLogitsLoss`：用于二元分类或多标签分类任务。与 `CrossEntropyLoss` 类似，它在一个步骤中结合了 Sigmoid 层和二元交叉熵损失，以获得更好的数值稳定性。它也需要原始 logits 作为输入。

### 在PyTorch中计算损失

一旦您实例化了所选的标准（例如，`criterion = nn.CrossEntropyLoss()`），在循环中计算损失就很直接。您只需将模型的输出张量和包含真实标签的张量传递给标准对象即可：

```python

outputs = model(inputs)

loss = criterion(outputs, labels)
```

理解生成的 `loss` 变量代表什么很重要：

1. **标量值：** 它通常是一个数值（一个零维张量），表示整个批次的平均损失。
2. **计算图连接：** 重要的是，这个 `loss` 张量仍然连接着 PyTorch 在正向传播期间构建的计算图。它知道哪些操作和哪些模型参数 (parameter)促成了它的最终值。
3. **梯度计算已启用：** 因为它连接着计算图并依赖于 `requires_grad=True` 的参数，所以 `loss` 张量本身隐式地具有 `requires_grad=True`。这使得我们可以在下一步调用 `loss.backward()`，自动计算损失相对于模型所有可学习参数（∇θL\nabla\_{\theta} L∇θ​L）的梯度。

这个计算出的 `loss` 值作为反向传播 (backpropagation)过程的起点，它调整模型的权重 (weight)，以期在后续迭代中产生更低的损失。

获取即时帮助、个性化解释和交互式代码示例。

---

### 反向传播：计算梯度

### 反向传播：计算梯度

您已经成功计算出损失，它衡量了模型预测与实际目标值之间的偏离程度。接下来一个重要步骤是了解 *如何* 调整模型参数 (parameter)（权重 (weight)和偏置 (bias)）以减小此损失。这就是反向传播 (backpropagation)发挥作用的地方。

PyTorch 使用 `.backward()` 方法执行反向传播。当此方法在损失张量上调用时，模型参数的梯度便被计算出来。

```python

loss.backward()
```

调用 `loss.backward()` 会触发 PyTorch 的自动微分引擎 Autograd。回忆一下第 3 章，PyTorch 如何在正向传播过程中执行运算时动态构建计算图？此图记录了从输入数据和模型参数到最终损失值的运算序列。

`backward()` 调用会启动对此图的 *逆向* 遍历，从损失张量本身开始。利用微积分链式法则，Autograd 高效地计算图中每个 `requires_grad` 属性设为 `True` 的张量相对于损失的梯度。

具体来说，对于模型中参与了损失计算的每个参数 θ\thetaθ（例如 `nn.Linear` 或 `nn.Conv2d` 层中的权重和偏置），`loss.backward()` 会计算偏导数：

∂L∂θ\frac{\partial L}{\partial \theta}∂θ∂L​

这个梯度 ∂L∂θ\frac{\partial L}{\partial \theta}∂θ∂L​ 代表了损失 LLL 对参数 θ\thetaθ 微小变化的敏感度。它说明了 θ\thetaθ 为减小损失所需变化的方向和大小。

这些计算出的梯度会存放在哪里？PyTorch 会直接将其存储在对应参数张量的 `.grad` 属性中。

```python

print(model.weight.grad)
print(model.bias.grad)
```

在调用 `loss.backward()` 后，您通常会在模型参数的 `.grad` 属性中发现非 `None` 值。计算图中的中间张量的梯度通常不保留以节省内存，尽管如果调试或高级技术需要，可以修改此行为。通过 `nn.Module` 定义的模型参数被认为是图中的“叶”节点，它们的梯度会被保留。


cluster\_forward

正向传播

cluster\_backward

反向传播 (loss.backward())


输入 (X)

Yₚred

预测值 (Yₚred)

X->Yₚred

 \* W + B


权重 (W)
requires\_grad=True

W->Yₚred

grad\_W

dL/dW
存储于 W.grad


偏置 (B)
requires\_grad=True

B->Yₚred

grad\_B

dL/dB
存储于 B.grad

Loss

损失 (L)

Yₚred->Loss

 损失函数

Yₜrue

目标值 (Yₜrue)

Yₜrue->Loss

grad\_L

dL/dL = 1

Loss->grad\_L

grad\_Yₚred

dL/dYₚred

grad\_L->grad\_Yₚred

grad\_Yₚred->grad\_W

grad\_Yₚred->grad\_B

> 正向传播构建计算图（实线）。调用 `loss.backward()` 启动反向传播（虚线），从损失开始计算梯度，并将它们存储在 `W` 和 `B` 等张量的 `.grad` 属性中。

### 梯度累加

一个需要理解的重要行为是 PyTorch 会 *累加* 梯度。当您调用 `loss.backward()` 时，新计算出的梯度会 **添加** 到参数 (parameter)的 `.grad` 属性中已有的值上。它们不会覆盖之前的值。

这种累加是设计意图，且在特定情况下有用，例如模拟更大的批量大小或训练循环神经网络 (neural network) (RNN)。然而，在每次处理一个批量的标准训练循环中，您通常只想根据当前批量计算梯度。如果您不清除上次迭代的梯度，就会累积多个批量的梯度，从而导致参数更新不正确。

正因如此，如本章引言中所述并将在后面详细说明，您必须在每个训练迭代开始时，通常在正向传播 *之前* 或恰好在调用 `loss.backward()` 之前，显式地将梯度归零。执行此操作的标准方法涉及优化器，即使用 `optimizer.zero_grad()`。

在梯度计算完成并存储在参数的 `.grad` 属性中之后，下一步便是使用这些梯度更新模型的参数，这由优化器的 `step()` 方法处理。

获取即时帮助、个性化解释和交互式代码示例。

---

### 使用优化器更新权重

### 使用优化器更新权重

模型的参数 (parameter)（具有`requires_grad=True`的权重 (weight)和偏差）在其`.grad`属性中包含已计算的梯度。这些梯度是在损失计算并使用`loss.backward()`执行反向传播 (backpropagation)时生成的。它们，例如∇θL\nabla\_{\theta} L∇θ​L，表示在参数空间中会使损失增长最快的方向。为了使损失最小化，我们需要朝着*相反*的方向调整参数。这正是优化器的作用。

在第4章中，您学习了如何实例化优化器，例如`torch.optim.SGD`或`torch.optim.Adam`，并传入模型的参数（`model.parameters()`）以及学习率等配置信息。现在，在训练循环中，您使用优化器的`step()`方法来执行参数更新。

```python

outputs = model(inputs)

loss = criterion(outputs, labels)

loss.backward()

optimizer.step()
```

调用`optimizer.step()`会遍历在优化器初始化时注册的所有参数。对于每个参数`p`，它会使用存储在`p.grad`中的梯度来更新参数值`p.data`。

随机梯度下降 (gradient descent)（SGD）使用的最基本的更新规则是：

参数=参数−学习率×梯度\text{参数} = \text{参数} - \text{学习率} \times \text{梯度}参数=参数−学习率×梯度

或者更正式地，对于参数θ\thetaθ：

θnew=θold−η∇θL\theta\_{new} = \theta\_{old} - \eta \nabla\_{\theta} Lθnew​=θold​−η∇θ​L

其中η\etaη是学习率（类似于创建优化器时传入的`lr`参数），而∇θL\nabla\_{\theta} L∇θ​L是通过`loss.backward()`计算的梯度，优化器通过`parameter.grad`在内部访问它。

不同的优化器实现了更复杂的更新规则。例如，像Adam这样的优化器对每个参数使用自适应学习率并融入动量思想，但核心思想保持不变：使用计算出的梯度调整参数以最小化损失。`optimizer.step()`调用将所选优化算法定义的具体更新逻辑进行了封装。

在调用`optimizer.step()`*之前*，必须先调用`loss.backward()`。`backward()`调用计算梯度，而`step()`调用则使用这些梯度来更新权重。如果不先调用`backward()`，`.grad`属性将不会被填充（或者会包含来自上一次迭代的旧值），优化器也就无法知道如何有效地调整参数。

更新权重后的下一个重要步骤是在处理下一个批次之前清除梯度。我们将在下一节“梯度清零”中介绍这一点。

获取即时帮助、个性化解释和交互式代码示例。

---

### 梯度清零

### 梯度清零

调用 `$loss.backward()` 计算损失相对于模型参数 (parameter)（那些 `requires_grad=True` 的张量）的梯度。这些梯度表示为减少损失所需的改变方向和大小，它们存储在每个参数张量的 `.grad` 属性中。随后，`$optimizer.step()` 会根据所选的优化算法（如 SGD 或 Adam）使用这些存储的梯度来更新参数值。

然而，PyTorch 在反向传播 (backpropagation)过程中处理梯度的方式有一个不明显但非常重要的细节：**PyTorch 会累积梯度**。当您调用 `$loss.backward()$` 时，为每个参数新计算的梯度会 *添加* 到该参数 `.grad` 属性中已有的值上。

如果您不处理这个问题，请考虑训练循环在多次迭代中会发生什么：

1. **第一次迭代：** 计算损失 L1L\_1L1​，调用 `$loss_1.backward()`。梯度 ∇θL1\nabla\_{\theta} L\_1∇θ​L1​ 被计算并存储在 `param.grad` 中。调用 `$optimizer.step()$`。
2. **第二次迭代：** 计算损失 L2L\_2L2​，调用 `$loss_2.backward()`。新的梯度 ∇θL2\nabla\_{\theta} L\_2∇θ​L2​ 被计算。PyTorch 将这些梯度 *添加* 到现有梯度中，因此 `param.grad` 现在持有 ∇θL1+∇θL2\nabla\_{\theta} L\_1 + \nabla\_{\theta} L\_2∇θ​L1​+∇θ​L2​。调用 `$optimizer.step()$`。

第二次迭代中的优化器步骤使用了不正确的梯度信息。它使用了当前批次和上一批次的混合梯度。这会阻止模型良好地学习，因为权重 (weight)更新是基于来自不同数据点的陈旧且混合的梯度信号。

### `optimizer.zero_grad()` 的用途

为了防止这种累积并确保优化器 *仅* 基于当前批次的梯度来更新权重 (weight)，您必须在为下一次迭代计算梯度之前手动重置梯度。这正是 `optimizer.zero_grad()` 方法所做的。

调用 `optimizer.zero_grad()` 会遍历优化器被配置来管理的所有参数 (parameter) (θ\thetaθ)，并将其 `.grad` 属性重置为零（或 `None`）。

### 何时调用 `optimizer.zero_grad()`

您需要在每次训练迭代中调用 `optimizer.zero_grad()` 一次。最常见且推荐的做法是在循环的 *开始* 处调用它，即在处理下一个批次之前：

```python

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    for inputs, labels in dataloader:

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()
```

或者，您可以在 `optimizer.step()` *之后* 立即调用 `optimizer.zero_grad()`。在标准循环中，其功能结果是相同的。将其放在开始处可以清楚地划分新批次处理的开始。核心是，它必须在下一次 `loss.backward()` 调用 *之前* 被调用，以避免梯度在迭代之间累积。

忘记清零梯度是 PyTorch 训练循环中常见的错误源，这通常会导致模型无法收敛或表现出异常的学习情况。请务必确保 `optimizer.zero_grad()` 在您的训练迭代中位置正确。

虽然梯度累积通常是不希望发生的，但当 GPU 内存受限时，它可以被有意地用作一种方法来模拟更大的批次大小。在这种情况下，您可以对多个小批次执行前向和反向传播 (backpropagation)，通过在每次 `loss.backward()` 后 *不* 调用 `optimizer.zero_grad()` 来累积梯度，然后只在处理完所需数量的小批次后才调用 `optimizer.step()` 和 `optimizer.zero_grad()`。但是，对于常规训练，每次迭代清零梯度是常规步骤。

获取即时帮助、个性化解释和交互式代码示例。

---

### 实现评估循环

### 实现评估循环

为了客观评估模型在训练阶段后的表现，一种可靠的方法必不可少。仅凭训练损失可能产生误导，因为模型可能在训练数据上表现出色，但无法泛化到新的、未见过的数据。评估循环可以解决这一挑战。它的作用是衡量模型在独立数据集（如验证集或测试集）上的表现，这些数据未在权重 (weight)更新过程中使用。

### 为何需要独立的评估循环？

训练涉及根据训练数据调整模型参数 (parameter)。然而，评估纯粹是为了评测。我们想回答：“给定此输入，模型的预测与实际目标有多接近？”同时不改变模型本身。执行评估需要一个独特的流程，原因有以下几点：

1. **泛化能力评估：** 它衡量模型处理训练中未曾见过数据的能力，这是大多数机器学习 (machine learning)任务的最终目标。
2. **防止数据泄露：** 使用独立数据集可确保评估数据中的信息不会无意中影响训练过程（例如，被用于梯度更新）。
3. **模型选择与调优：** 验证集上的表现常用于选择最佳模型架构、决定何时停止训练（早期停止）或调整超参数 (hyperparameter)。
4. **检测过拟合 (overfitting)：** 比较训练集与验证集上的表现有助于识别过拟合。当模型对训练数据（包括其噪声）学习得过于透彻，从而失去泛化能力时，就会发生过拟合。此时训练数据上的表现可能持续提升，而验证数据上的表现却停滞不前或下降。

### 与训练循环的区别

评估循环与训练循环有相似之处（例如，遍历数据，执行前向传播），但存在重要区别：

1. **不计算梯度：** 由于我们只进行评估而不更新权重 (weight)，因此无需计算或存储梯度。这节省了内存和计算资源。
2. **不反向传播 (backpropagation)：** 因此，*不*调用`$loss.backward()`。
3. **不执行优化器步骤：** 模型的权重保持不变，因此*不*调用`$optimizer.step()`和`$optimizer.zero_grad()`。
4. **模型模式：** 模型应切换到评估模式。

### 将模型设置为评估模式：`model.eval()`

PyTorch 模型（`nn.Module`）有不同的训练和评估模式。你可以使用`model.train()`和`model.eval()`在它们之间切换。在开始评估循环之前调用`model.eval()`非常重要。此调用会通知Dropout和Batch Normalization等层，模型正处于评估阶段。

- **Dropout 层：** 在评估期间被停用。我们希望模型展现其完整的预测能力，而不是随机丢弃神经元。
- **Batch Normalization 层：** 使用训练期间计算的运行统计数据（均值和方差），而不是当前批次的统计数据。这确保了输出的一致性，不受评估批次统计数据的影响。

评估结束后，如果你计划恢复训练（例如，在每个周期后进行评估），请记得使用`model.train()`将模型切换回训练模式。

### 禁用梯度计算：`torch.no_grad()`

为防止PyTorch在评估期间跟踪操作并构建用于梯度计算的计算图，你应该将评估循环代码封装在`torch.no_grad()`上下文 (context)管理器中。

```python
with torch.no_grad():
```

使用`torch.no_grad()`主要有两个优点：

1. **效率：** 它减少了内存消耗，因为反向传播 (backpropagation)所需的中间激活不会被存储。操作也可能运行得更快。
2. **正确性：** 它确保你不会在不需要时意外地计算梯度或尝试执行反向传播。

### 评估循环的结构

以下是评估函数的一个典型结构：

```python
import torch

def evaluate_model(model, dataloader, criterion, device):
    """在提供的datasets上评估模型。"""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            _, predicted_labels = torch.max(outputs, dim=1)
            correct_predictions += (predicted_labels == targets).sum().item()
            total_samples += targets.size(0)

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    model.train()
    return average_loss, accuracy
```

**分步解析：**

1. **`model.eval()`：** 将模型切换到评估模式。
2. **初始化指标：** 设置变量以累加总损失和正确预测的数量（或其他相关指标）。同时，跟踪评估的总样本数。
3. **`with torch.no_grad():`：** 进入不计算梯度的上下文 (context)。
4. **遍历DataLoader：** 循环遍历评估`DataLoader`提供的批次。
5. **设备放置：** 确保输入数据和目标与模型位于同一设备上。
6. **前向传播：** 将输入数据通过模型（`outputs = model(inputs)`）。
7. **计算损失：** 使用准则计算损失。使用`loss.item()`获取当前批次损失的Python标量值，并在累加前乘以批次大小（`inputs.size(0)`），以处理最后一个批次大小可能存在的差异。
8. **计算指标：** 从模型输出中确定预测（例如，对于分类任务，使用`torch.max`获取最高概率的索引）。将预测与真实目标进行比较，并累加正确预测的数量和总样本数。
9. **汇总结果：** 遍历所有批次后，将累加的总量除以处理的总样本数，计算平均损失和总体准确率（或其他指标）。
10. **`model.train()`（可选）：** 如果此评估发生在训练周期之间，将模型切换回训练模式。

此评估循环为模型的泛化表现提供了必要的反馈，指导训练过程并帮助你构建更高效的深度学习 (deep learning)模型。同时监控这些评估指标和训练指标对理解模型行为非常重要。

2468100.511.5训练损失验证损失

> 此图显示了一个常见情况：在若干个周期后，验证损失开始增加，这表明过拟合 (overfitting)已经出现，即使训练损失持续下降。评估循环对于检测这种情况非常重要。

TrainingCycle

TrainEpoch

运行训练周期
(前向传播, 损失, 反向传播, 优化)

EvaluateModel

运行评估循环
(model.eval(), no\_grad, 前向传播, 指标)

TrainEpoch->EvaluateModel

 周期结束

LogMetrics

记录训练和验证指标

EvaluateModel->LogMetrics

CheckStop

检查停止条件
(例如：最大周期数, 早期停止)

CheckStop->TrainEpoch

 继续训练

Stop

训练完成

CheckStop->Stop

 停止训练

LogMetrics->CheckStop

> 典型的深度学习训练流程，在每个训练周期后加入评估，以监控表现并决定是否继续或停止训练。

获取即时帮助、个性化解释和交互式代码示例。

---

### 保存和加载模型检查点

### 保存和加载模型检查点

深度学习 (deep learning)模型的训练通常耗时，根据模型复杂度和数据集大小，可能需要数小时甚至数天。每次因系统中断、后续微调 (fine-tuning)或仅为预测而停止时都从头开始训练是不现实的。因此，保存和加载模型检查点变得非常重要。

检查点记录了训练过程在特定时刻的状态，方便您日后恢复。有效保存和加载 PyTorch 模型和训练状态是必要的组成部分。

### 应该保存什么？

保存检查点时，您需要根据目的决定需要哪些信息。通常，您至少会保存模型的参数 (parameter)。如果您打算恢复训练，还应保存优化器的状态，以及当前周期数和最新验证损失等其他元数据。

PyTorch 模型有一个内部状态字典，通过 `model.state_dict()` 访问，其中包含模型各层的所有学习参数（权重 (weight)和偏置 (bias)）。这是保存模型学习信息建议的方式。

为什么保存 `state_dict` 而不是整个模型对象（例如 `torch.save(model, PATH)`）？保存 `state_dict` 更具弹性，也更不容易出问题。对整个模型对象进行序列化保存会存储保存时使用的特定代码结构。如果您之后重构或更改模型类定义，加载序列化对象可能会失败或导致意外行为。仅保存状态字典将学习参数与代码结构分离，使加载更稳定。

同样，Adam 或 SGD 等优化器也有内部状态（例如，动量缓冲区、自适应学习率），这些状态在训练过程中会变化。为了精确地恢复训练，您应该使用 `optimizer.state_dict()` 保存优化器的状态。

### 使用 `torch.save` 保存检查点

PyTorch 使用 `torch.save()` 来序列化和保存对象。要保存检查点，您通常会创建一个字典，其中包含模型的状态字典、优化器的状态字典以及任何其他相关信息，然后保存这个字典。

以下是在训练循环中保存检查点的常见模式：

```python

checkpoint = {
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,

}

torch.save(checkpoint, PATH)
print(f"检查点已在周期 {epoch} 保存到 {PATH}")
```

您可以定期（例如，每 10 个周期）保存检查点，或在模型在验证集上取得新最佳表现时保存。

### 使用 `torch.load` 和 `load_state_dict` 加载检查点

要加载检查点，您首先使用 `torch.load()` 从文件中反序列化保存的字典。然后，您需要将状态字典加载回您的模型和优化器实例中。

**注意：** 在加载模型和优化器状态之前，您必须先创建它们的实例。`load_state_dict()` 方法将参数 (parameter)加载*到*一个现有对象中；它不会重新创建对象本身。

#### 加载用于推理 (inference)

如果您只需要模型进行预测（推理），并且不打算恢复训练，通常只需加载 `model_state_dict`。

```python

model = YourModelClass(*args, **kwargs)

PATH = "path/to/your/checkpoint.pth"

checkpoint = torch.load(PATH)

model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
```

设置 `model.eval()` 很关键，因为它会禁用 Dropout 等层，并使用运行统计数据对批量归一化 (normalization)层进行归一化，这是推理时的正确操作。

#### 加载以恢复训练

如果您想从上次中断的地方继续训练，您需要加载模型和优化器的状态，并获取其他已保存的元数据，例如周期数。

```python

model = YourModelClass(*args, **kwargs)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

PATH = "path/to/your/checkpoint.pth"
start_epoch = 0
best_loss = float('inf')

if os.path.exists(PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['loss']
    print(f"检查点已加载。从周期 {start_epoch} 继续训练")

model.train()
```

设置 `model.train()` 可确保 Dropout 和批量归一化等层在训练期间表现正常。

### 处理 CPU/GPU 设备映射

有时，您可能会保存一个在 GPU 上训练的模型，然后需要在只有 CPU 的机器上加载它，反之亦然。默认情况下，`torch.load()` 会尝试将张量加载到它们保存时所在的设备上。如果该设备不可用，这可能会导致错误。

为了处理这种情况，您可以使用 `torch.load()` 中的 `map_location` 参数 (parameter)。

```python

checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
```

保存和加载检查点是深度学习 (deep learning)工作流程中很普通的一部分。通过熟练掌握使用 `torch.save`、`torch.load` 和 `load_state_dict` 的这些方法，您可以保证训练进程安全，模型可以重复使用，并且训练过程面对中断时表现稳定。请记住，为了获得最大的适应性和稳定性，要保存模型和优化器的 `state_dict`，以及相关的元数据。

获取即时帮助、个性化解释和交互式代码示例。

---

### 动手实践：完整训练流程

### 动手实践：完整训练流程

既然您已经了解了训练的各个组成部分，我们将其整合到一个完整的、可运行的例子中。本实践练习将引导您设置模型、准备数据，并实现训练和评估循环，从而巩固本章讨论的内容。我们还会提到保存训练好的模型状态。

一个使用合成数据的简单线性回归问题将被解决。训练模型的目标是学习 y≈2x+1y \approx 2x + 1y≈2x+1 的关系。

### 1. 设置：导入与超参数 (parameter) (hyperparameter)

首先，我们导入必要的PyTorch模块，并为训练过程定义一些基本超参数。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

learning_rate = 0.01
num_epochs = 100
batch_size = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"正在使用设备: {device}")
```

### 2. 数据准备

我们将生成用于线性关系的合成数据，并使用 `TensorDataset` 和 `DataLoader` 对其进行包装。

```python

true_weight = torch.tensor([[2.0]])
true_bias = torch.tensor([1.0])

X_train_tensor = torch.randn(100, 1) * 5
y_train_tensor = true_weight * X_train_tensor + true_bias + torch.randn(100, 1) * 0.5

X_val_tensor = torch.randn(20, 1) * 5
y_val_tensor = true_weight * X_val_tensor + true_bias + torch.randn(20, 1) * 0.5

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
```

这里，`TensorDataset` 方便地将输入特征 (`X`) 和目标标签 (`y`) 张量包装起来。`DataLoader` 接着接收此数据集，并提供可迭代的批次，自动处理打乱和批次化。

### 3. 模型、损失与优化器

现在，定义模型架构、损失函数 (loss function)和优化器。由于我们正在模拟线性关系 y=wx+by=wx+by=wx+b，因此一个简单的线性层就足够了。

```python

model = nn.Linear(1, 1).to(device)

loss_fn = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

print("模型定义:")
print(model)
print("\n初始参数:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data.squeeze()}")
```

我们实例化 `nn.Linear`，它表示操作 y=Wx+by = Wx + by=Wx+b。PyTorch 会自动初始化权重 (weight) (WWW) 和偏置 (bias) (bbb) 参数 (parameter)。我们使用均方误差 (`nn.MSELoss`)，因为它是回归任务的标准方法，用于衡量预测值与真实值之间的平均平方差。选择随机梯度下降 (gradient descent) (`optim.SGD`) 来根据计算出的梯度更新模型的参数。请注意，我们将 `model.parameters()` 传递给优化器，以便它知道要更新哪些张量。最后，我们将模型移动到配置好的设备（CPU 或 GPU）上。

### 4. 训练循环

这是模型从数据中迭代学习过程的核心。

```python
print("\n开始训练...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for i, (features, labels) in enumerate(train_loader):

        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)

        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_epoch_loss = running_loss / num_batches
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}")

print("训练完成！")
```

我们来逐一分析训练轮次循环中的步骤：

1. `model.train()`：将模型设置为训练模式。这对于像 Dropout 或 BatchNorm 这样的层很重要，因为它们在训练和评估期间行为不同。
2. 我们遍历 `train_loader` 以获取 `features` 和 `labels` 的批次。
3. 数据被移动到模型所在的 `device` 上。这可以防止运行时错误。
4. **前向传播**：`outputs = model(features)` 计算模型对输入批次的预测。
5. **损失计算**：`loss = loss_fn(outputs, labels)` 使用 MSE 准则计算预测与实际标签之间的差异。
6. **反向传播 (backpropagation)**：
   - `optimizer.zero_grad()`：清除旧梯度。如果忘记此步骤，梯度将从之前的迭代中累积，导致不正确的更新。
   - `loss.backward()`：计算损失相对于所有 `requires_grad=True` 的模型参数 (parameter)的梯度。
7. **优化器步骤**：`optimizer.step()` 使用反向传播中计算的梯度和优化算法（本例中为 SGD）更新模型的参数 (`model.parameters()`)。
8. 我们跟踪 `running_loss` 来报告本轮的平均损失。

### 5. 评估循环

训练之后（或在训练期间定期，例如每轮结束后），我们需要在不更新模型权重 (weight)的情况下，评估模型在未见数据（验证集）上的表现。

```python
print("\n开始评估...")
model.eval()
total_val_loss = 0.0
num_val_batches = 0

with torch.no_grad():
    for features, labels in val_loader:

        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)

        loss = loss_fn(outputs, labels)
        total_val_loss += loss.item()
        num_val_batches += 1

avg_val_loss = total_val_loss / num_val_batches
print(f"验证损失: {avg_val_loss:.4f}")

print("\n学习到的参数:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data.squeeze()}")

print(f"(真实权重: {true_weight.item():.4f}, 真实偏置: {true_bias.item():.4f})")
```

评估循环中的主要区别：

1. `model.eval()`：将模型设置为评估模式。
2. `with torch.no_grad():`：此上下文 (context)管理器在该代码块内禁用梯度计算。这很重要，因为我们在评估时不需要梯度，并且它能减少内存消耗并加快计算速度。
3. 我们不调用 `loss.backward()` 或 `optimizer.step()`，因为我们只测量性能，而不进行训练。
4. 我们累加所有验证批次的损失，以获得平均验证损失。

评估后，我们打印学习到的参数 (parameter)。将它们与我们用于生成数据的 `true_weight` (2.0) 和 `true_bias` (1.0) 进行比较。在 100 轮训练后，它们应该相当接近。

### 6. 保存与加载模型状态

保存您训练好的模型非常重要。标准做法是保存模型的 `state_dict`，其中包含所有学习到的参数 (parameter)（权重 (weight)和偏置 (bias)）。

```python

model_save_path = 'linear_regression_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"\n模型 state_dict 已保存到 {model_save_path}")

loaded_model = nn.Linear(1, 1).to(device)

loaded_model.load_state_dict(torch.load(model_save_path))
print("模型 state_dict 加载成功。")

loaded_model.eval()

with torch.no_grad():
    sample_input = torch.tensor([[10.0]]).to(device)
    prediction = loaded_model(sample_input)
    print(f"输入 10.0 的预测值: {prediction.item():.4f}")
```

通常，保存 `state_dict` 比保存整个模型对象更好，因为它更灵活，且在底层代码更改时更不容易出错。要加载状态，您需要首先创建相同模型架构的实例，然后将字典加载到其中。

### 完整可运行示例

以下是结合所有部分的完整脚本：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

learning_rate = 0.01
num_epochs = 100
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"正在使用设备: {device}")

true_weight = torch.tensor([[2.0]])
true_bias = torch.tensor([1.0])
X_train_tensor = torch.randn(100, 1, device=device) * 5
y_train_tensor = true_weight.to(device) * X_train_tensor + true_bias.to(device) + torch.randn(100, 1, device=device) * 0.5
X_val_tensor = torch.randn(20, 1, device=device) * 5
y_val_tensor = true_weight.to(device) * X_val_tensor + true_bias.to(device) + torch.randn(20, 1, device=device) * 0.5

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

model = nn.Linear(1, 1).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

print("模型定义:")
print(model)
print("\n初始参数:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data.squeeze()}")

print("\n开始训练...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    num_batches = 0
    for i, (features, labels) in enumerate(train_loader):

        outputs = model(features)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1

    avg_epoch_loss = running_loss / num_batches
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}")
print("训练完成！")

print("\n开始评估...")
model.eval()
total_val_loss = 0.0
num_val_batches = 0
with torch.no_grad():
    for features, labels in val_loader:

        outputs = model(features)
        loss = loss_fn(outputs, labels)
        total_val_loss += loss.item()
        num_val_batches += 1

avg_val_loss = total_val_loss / num_val_batches
print(f"验证损失: {avg_val_loss:.4f}")

print("\n学习到的参数:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data.squeeze().item():.4f}")
print(f"(真实权重: {true_weight.item():.4f}, 真实偏置: {true_bias.item():.4f})")

model_save_path = 'linear_regression_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"\n模型 state_dict 已保存到 {model_save_path}")

loaded_model = nn.Linear(1, 1).to(device)
loaded_model.load_state_dict(torch.load(model_save_path))
loaded_model.eval()
print("模型 state_dict 加载成功。")

with torch.no_grad():
    sample_input = torch.tensor([[10.0]]).to(device)
    prediction = loaded_model(sample_input)
    print(f"输入 10.0 的预测值: {prediction.item():.4f}")
```

*（注：在合并脚本中，数据生成略有修改，直接在目标 `device` 上创建张量以提高效率，从而无需在循环内部对批次数据使用 `.to(device)`。）*

这个动手示例呈现了在 PyTorch 中训练几乎任何模型的基本结构。您现在有了一个结合数据加载、模型定义、训练迭代、评估和持久化的模板。您可以通过更改步骤3中的模型架构和步骤2中的数据准备来调整此结构以适应更复杂的模型和数据集。训练和评估循环的核心逻辑保持了显著的一致性。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 7 Introduction Common Architectures

### 卷积神经网络 (CNN) 概述

### 卷积神经网络 (CNN) 概述

标准神经网络 (neural network)层，如 `nn.Linear`，将输入数据视为一个扁平向量 (vector)。尽管功能强大，但这种方法未能内在理解图像等数据中存在的空间结构。对于图像来说，相互靠近的像素通常是关联的，它们构成边缘、纹理或物体的一部分。当直接应用于图像时，全连接层面临两个主要问题：

1. **参数 (parameter)效率低下：** 将一张中等大小的图像（例如，224x224 像素，3 个颜色通道）展平为向量会导致输入维度非常大。即使将其连接到一个中等大小的隐藏层，也需要大量权重 (weight)，使得模型容易过拟合 (overfitting)，并且计算成本高昂。
2. **空间信息丢失：** 展平图像会丢弃像素的 2D（或包含通道的 3D）排列。网络会丢失关于哪些像素最初是相邻的信息。

卷积神经网络 (CNN) 是一种专门设计用于处理具有网格状拓扑数据（如图像（2D 网格）或时间序列数据（1D 网格））的神经网络。它们通过结合两个主要思想来解决标准网络的局限：局部感受野（通过卷积）和空间下采样（通过池化）。

### 卷积操作：识别局部模式

CNN 的核心组成部分是**卷积层**。卷积层不将每个输入单元连接到每个输出单元，而是使用小的过滤器（也称为核），它们在输入数据上滑动。每个过滤器都是一个小的权重 (weight)矩阵。

想象一个微小的放大镜（即过滤器）在输入图像上滑动。在每个位置，过滤器会与其当前覆盖的图像区域执行元素级乘法，并将结果求和以在输出中生成一个单一值。这个过程在整个输入图像上重复进行，生成一个输出**特征图**。

ConvolutionOp

clusterᵢnput

输入区域

cluster\_filter

过滤器（核）

clusterₒutput

输出值

I1

I2

I3

I4

I5

I6

I7

I8

I9

F1

w


u03a3(输入 u00d7 过滤器)

F1->O

F2

w

F2->O

F3

w

F3->O

F4

w

F4->O

> 过滤器对输入的局部区域施加权重，以计算输出特征图中的一个值。

这种滑动过滤器方法具有两个显著优点：

1. **局部连接：** 特征图中的每个单元仅连接到输入的一个小区域（过滤器大小）。这使得网络能够在早期层中学习到局部模式，如边缘或角落。
2. **参数 (parameter)共享：** 相同的过滤器（具有相同的权重集合）在输入图像的不同位置重复使用。这与全连接层相比**大幅减少了**参数数量，并使网络对特征的平移具有等变性。如果一个模式（如垂直边缘）被过滤器学习，它可以在图像中任何位置检测到该模式。

通常，一个卷积层会使用多个过滤器，每个过滤器学习识别不同类型的特征（例如，一个过滤器识别水平边缘，另一个识别垂直边缘，还有一个识别特定纹理）。这些过滤器的输出堆叠在一起，形成该层的最终输出体。PyTorch 主要通过 `nn.Conv2d` 层来实现图像数据的这一操作。

### 激活函数 (activation function)

就像在标准网络中一样，非线性激活函数（例如 ReLU，在 PyTorch 中实现为 `nn.ReLU`）通常在卷积操作之后进行元素级应用。这使得网络能够学习特征之间复杂的非线性关系。

### 池化操作：下采样与不变性

在通过卷积层检测到特征后，通常有益于使表示更紧凑并对小的空间变异具有抵抗力。这通过使用**池化层**来实现。

最常见的类型是**最大池化**。它也涉及在特征图上滑动一个窗口（通常小于卷积过滤器且不重叠或带步幅）。但是，它不应用学习到的权重 (weight)，而只是简单地取出该窗口内的*最大*值。

MaxPooling

clusterᵢnput

特征图区域 (2x2)

clusterₒutput

输出值

FM1
1

max(1, 5, 3, 2) = 5

FM1->O

FM2
5

FM2->O

FM3
3

FM3->O

FM4
2

FM4->O

> 最大池化选择特征图局部窗口内的最大值。

池化提供多项益处：

1. **维度降低：** 它减少了特征图的空间维度（高度和宽度），降低了后续层的计算负担。
2. **平移不变性（局部）：** 通过用其最大激活来概括局部区域，池化使表示对特征在该区域内的确切位置更具稳定性。

PyTorch 提供了 `nn.MaxPool2d` 等池化层。

### 典型 CNN 架构

一个典型的 CNN 架构通常会堆叠这些组件：

1. 一个或多个**卷积 -> 激活 -> 池化**层块。早期层倾向于使用较小的过滤器来捕捉精细细节，而后期层可能使用较大的过滤器，或依赖于早期层的池化特征来捕捉更大空间区域上的更复杂模式。
2. 在经过多个卷积和池化层之后，得到的特征图通常会**展平**为一个向量 (vector)。
3. 然后，这个向量被馈入一个或多个**全连接 (`nn.Linear`) 层**，类似于标准前馈网络，用于最终的分类或回归。

CNN\_Architecture

Input

输入图像

Conv1

Conv + ReLU

Input->Conv1

Pool1

MaxPool

Conv1->Pool1

Conv2

Conv + ReLU

Pool1->Conv2

Pool2

MaxPool

Conv2->Pool2

Flatten

Flatten

Pool2->Flatten

FC1

Linear + ReLU

Flatten->FC1

Output

输出（分数）

FC1->Output

> 一个典型的 CNN 架构流程。

CNN **运用**卷积和池化，直接从网格状数据中自动学习特征的分层表示，这使得它们在图像识别、物体检测等任务中表现非常出色，甚至在文本得到适当表示时，也能用于自然语言处理。在下一节中，你将看到如何在 PyTorch 中实现像 `nn.Conv2d` 和 `nn.MaxPool2d` 这样的构建模块，以构建你的第一个 CNN。

获取即时帮助、个性化解释和交互式代码示例。

---

### 在PyTorch中构建一个简单的CNN

### 在PyTorch中构建一个简单的CNN

将卷积神经网络 (neural network) (CNN)的核心概念转化为可运行的PyTorch模型。CNN通常通过堆叠卷积层、激活函数 (activation function)和池化层来构建，之后通常跟随一个或多个全连接层，用于分类或回归。PyTorch的`torch.nn`模块提供了这些核心组件的预构建实现，以便高效构建。

我们的目标是构建一个能够处理图像数据的简单CNN。我们将从定义网络结构开始，将其作为一个Python类，并继承自`torch.nn.Module`。

### CNN的核心层在PyTorch中

1. **卷积层 (`nn.Conv2d`)**：此层对输入应用可学习的滤波器。主要参数 (parameter)有：

   - `in_channels`：输入张量的通道数（例如，灰度图像为1，RGB图像为3）。
   - `out_channels`：滤波器数量（也是输出张量的通道数）。每个滤波器学习检测不同的特征。
   - `kernel_size`：滤波器尺寸（高 x 宽）。单个整数`k`表示`k x k`的滤波器。
   - `stride`：滤波器每次移动的像素数（默认为1）。
   - `padding`：在输入周围添加填充，常用于控制输出的空间尺寸（默认为0）。

   ```python
   import torch
   import torch.nn as nn

   conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
   ```
2. **池化层 (`nn.MaxPool2d`)**：此层减小特征图的空间尺寸（高和宽），使表示更紧凑，并对特征位置的变化略微更具鲁棒性。

   - `kernel_size`：取最大值的窗口大小。
   - `stride`：窗口移动的距离。对于非重叠池化，通常设为等于`kernel_size`。

   ```python

   pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
   ```
3. **激活函数 (activation function) (例如，`nn.ReLU`)**：引入非线性，使网络能够学习复杂的模式。ReLU（修正线性单元）是常用选择。它逐元素应用：f(x)=max(0,x)f(x) = max(0, x)f(x)=max(0,x)。

   ```python

   relu1 = nn.ReLU()
   ```
4. **线性层 (`nn.Linear`)**：一个标准的全连接层。通常用于CNN的末尾，在空间特征被提取和展平之后。

   - `in_features`：输入特征的数量（需要展平卷积/池化层的输出）。
   - `out_features`：输出特征的数量（例如，分类任务中的类别数）。

   ```python

   fc1 = nn.Linear(in_features=512, out_features=10)
   ```

### 定义CNN结构

我们通过继承`nn.Module`来定义我们的CNN。各层通常在`__init__`方法中定义，而前向传播（数据如何流经各层）则在`forward`方法中定义。

我们来构建一个具有以下结构的CNN：

- 输入：[批大小, 1, 28, 28]（例如，像MNIST那样的灰度图像）
- Conv1：1个输入通道，16个输出通道，5x5核，步长1，填充2
- ReLU1
- MaxPool1：2x2核，步长2
- Conv2：16个输入通道，32个输出通道，5x5核，步长1，填充2
- ReLU2
- MaxPool2：2x2核，步长2
- 展平
- Linear1：（输入特征取决于MaxPool2的输出），128个输出特征
- ReLU3
- Linear2：128个输入特征，10个输出特征（例如，用于10个类别）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):

        x = self.pool1(F.relu(self.conv1(x)))

        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(-1, 32 * 7 * 7)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x
```

我们来可视化架构流程：

CNN\_Architecture

输入 (1x28x28)

输入 (1x28x28)

Conv1 (16x5x5, s1, p2)\nReLU

Conv1 (16x5x5, s1, p2)
ReLU

输入 (1x28x28)->Conv1 (16x5x5, s1, p2)\nReLU

MaxPool1 (2x2, s2)

MaxPool1 (2x2, s2)

Conv1 (16x5x5, s1, p2)\nReLU->MaxPool1 (2x2, s2)

16x28x28

Conv2 (32x5x5, s1, p2)\nReLU

Conv2 (32x5x5, s1, p2)
ReLU

MaxPool1 (2x2, s2)->Conv2 (32x5x5, s1, p2)\nReLU

16x14x14

MaxPool2 (2x2, s2)

MaxPool2 (2x2, s2)

Conv2 (32x5x5, s1, p2)\nReLU->MaxPool2 (2x2, s2)

32x14x14

展平

展平

MaxPool2 (2x2, s2)->展平

32x7x7

Linear1 (1568 -> 128)\nReLU

Linear1 (1568 -> 128)
ReLU

展平->Linear1 (1568 -> 128)\nReLU

1568

Linear2 (128 -> 10)

Linear2 (128 -> 10)

Linear1 (1568 -> 128)\nReLU->Linear2 (128 -> 10)

128

输出 (10)

输出 (10)

Linear2 (128 -> 10)->输出 (10)

> 数据和张量形状流经`SimpleCNN`模型。请注意，通道数增加，而空间维度（高/宽）减小。

### 使用模型

要使用这个模型，首先实例化该类。然后，您可以将输入数据（作为PyTorch张量）传入其中。输入张量必须具有预期的形状，包括批次维度。对于我们的`SimpleCNN`，这具体为`[N, 1, 28, 28]`，其中N是批次中的样本数量。

```python

model = SimpleCNN()
print(model)

dummy_input = torch.randn(4, 1, 28, 28)

output = model(dummy_input)

print(f"\nInput shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
```

运行此代码将打印模型的层结构，并确认输出张量形状符合我们的预期（`[4, 10]`），表示批次中每张图像的10个类别的得分。

这个例子展示了如何在`nn.Module`中组合`nn.Conv2d`、`nn.MaxPool2d`、`nn.ReLU`和`nn.Linear`层来创建一个基本的CNN。设计CNN时的一个重要细节是正确计算每个层之后张量形状的变化，特别是在连接卷积/池化部分和全连接部分时。我们将在下一节更详细地说明这些形状的跟踪。

获取即时帮助、个性化解释和交互式代码示例。

---

### 理解CNN层的输入/输出形状

### 理解CNN层的输入/输出形状

当你开始构建卷积神经网络 (neural network)（CNN）时，最常见的实际问题之一是确保一个层的输出形状能正确匹配下一个层的预期输入形状。与只需要考虑一个维度的简单全连接层不同，卷积层和池化层对多维网格状数据（如图像）进行操作，涉及高度、宽度和通道维度。了解这些维度如何变化对于构建有效的CNN架构非常重要。

让我们看一个用于二维CNN层（如`nn.Conv2d`或`nn.MaxPool2d`）的典型输入张量。它通常有四个维度：(N,Cin,Hin,Win)(N, C\_{in}, H\_{in}, W\_{in})(N,Cin​,Hin​,Win​)

- NNN: 批大小（同时处理的样本数量）。
- CinC\_{in}Cin​: 输入通道数（例如，RGB图像为3，灰度图像为1）。
- HinH\_{in}Hin​: 输入特征图的高度。
- WinW\_{in}Win​: 输入特征图的宽度。

批维度 NNN 通常保持不变。主要的变换发生在通道 (CCC)、高度 (HHH) 和宽度 (WWW) 上。

### 卷积层 (`nn.Conv2d`)

`torch.nn.Conv2d`层对由多个输入平面组成的输入信号应用二维卷积。影响输出形状最重要的参数 (parameter)是：

- `in_channels` (CinC\_{in}Cin​): 必须与输入张量中的通道数匹配。
- `out_channels` (CoutC\_{out}Cout​): 决定卷积产生的通道数。这是该层学习的滤波器数量。
- `kernel_size`: 卷积核（滤波器）的大小。可以是一个整数用于方形卷积核（例如，3表示3x3），也可以是一个元组`(kH, kW)`用于指定高度和宽度。
- `stride`: 卷积核在输入特征图上滑动时的步长。默认为1。可以是一个整数或一个元组`(sH, sW)`。较大的步长会导致输出特征图尺寸更小。
- `padding`: 输入边缘添加的零填充量。默认为0。可以是一个整数或一个元组`(padH, padW)`。填充有助于控制输出的空间维度，并能保留边界信息。
- `dilation`: 卷积核元素之间的间距。默认为1。较大的空洞（扩张）允许卷积核覆盖输入更广的区域，而不会增加参数数量（空洞卷积）。

输出形状 (N,Cout,Hout,Wout)(N, C\_{out}, H\_{out}, W\_{out})(N,Cout​,Hout​,Wout​) 如下确定：

1. **通道数 (CoutC\_{out}Cout​):** 这由`nn.Conv2d`层的`out_channels`参数直接设置。每个滤波器产生一个输出通道（特征图）。
2. **高度 (HoutH\_{out}Hout​) 和宽度 (WoutW\_{out}Wout​):** 这些取决于输入维度 (Hin,WinH\_{in}, W\_{in}Hin​,Win​) 和层的参数。计算输出高度的公式是：

   Hout=⌊Hin+2×填充[0]−空洞[0]×(卷积核尺寸[0]−1)−1步长[0]+1⌋H\_{out} = \lfloor \frac{H\_{in} + 2 \times \text{填充}[0] - \text{空洞}[0] \times (\text{卷积核尺寸}[0] - 1) - 1}{\text{步长}[0]} + 1 \rfloorHout​=⌊步长[0]Hin​+2×填充[0]−空洞[0]×(卷积核尺寸[0]−1)−1​+1⌋

   对于宽度 (WoutW\_{out}Wout​) 也是类似的：

   Wout=⌊Win+2×填充[1]−空洞[1]×(卷积核尺寸[1]−1)−1步长[1]+1⌋W\_{out} = \lfloor \frac{W\_{in} + 2 \times \text{填充}[1] - \text{空洞}[1] \times (\text{卷积核尺寸}[1] - 1) - 1}{\text{步长}[1]} + 1 \rfloorWout​=⌊步长[1]Win​+2×填充[1]−空洞[1]×(卷积核尺寸[1]−1)−1​+1⌋

   注意：如果`padding`、`dilation`、`kernel_size`或`stride`被指定为单个整数，它们将应用于高度和宽度两个维度（例如，`padding[0] = padding[1] = padding`）。符号 ⌊⋅⌋\lfloor \cdot \rfloor⌊⋅⌋ 表示向下取整函数（向下舍入到最接近的整数）。

让我们看一个当`dilation = 1`的常见情况。公式简化为：

Hout=⌊Hin+2×填充[0]−卷积核尺寸[0]步长[0]+1⌋H\_{out} = \lfloor \frac{H\_{in} + 2 \times \text{填充}[0] - \text{卷积核尺寸}[0]}{\text{步长}[0]} + 1 \rfloorHout​=⌊步长[0]Hin​+2×填充[0]−卷积核尺寸[0]​+1⌋
Wout=⌊Win+2×填充[1]−卷积核尺寸[1]步长[1]+1⌋W\_{out} = \lfloor \frac{W\_{in} + 2 \times \text{填充}[1] - \text{卷积核尺寸}[1]}{\text{步长}[1]} + 1 \rfloorWout​=⌊步长[1]Win​+2×填充[1]−卷积核尺寸[1]​+1⌋

**示例：**

假设我们有一个形状为`(16, 3, 32, 32)`的输入张量（批=16，通道=3，高=32，宽=32）。我们将其通过一个定义如下的`nn.Conv2d`层：

```python
import torch
import torch.nn as nn

conv_layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

input_tensor = torch.randn(16, 3, 32, 32)

output_tensor = conv_layer(input_tensor)
print(output_tensor.shape)
```

在此示例中，使用`kernel_size=3`、`stride=1`和`padding=1`是一种常见组合，它保持了输入的高度和宽度（`32x32` -> `32x32`），同时将通道数从3变为64。这有时被称为“相同”填充，尽管PyTorch不像其他一些框架那样有明确的`'same'`选项；你可以通过正确设置参数来实现。

如果我们将步长改为2（`stride=2`），输出维度将会减小：

```python
conv_layer_s2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)

output_tensor_s2 = conv_layer_s2(input_tensor)
print(output_tensor_s2.shape)
```

### 池化层 (`nn.MaxPool2d`)

池化层，例如`nn.MaxPool2d`，用于减小特征图的空间维度（下采样），使表示更紧凑且对小的平移具有鲁棒性。它们在每个通道上独立操作。

影响形状的主要参数 (parameter)与`nn.Conv2d`相似，但没有`out_channels`这个参数，因为池化不会改变通道数量：

- `kernel_size`: 池化窗口的大小。
- `stride`: 窗口的步长。通常设置为与`kernel_size`相等，以实现不重叠的池化（默认值是`kernel_size`）。
- `padding`: 添加的零填充量。
- `dilation`: 控制池化元素之间的间距。

输出形状 (N,Cout,Hout,Wout)(N, C\_{out}, H\_{out}, W\_{out})(N,Cout​,Hout​,Wout​) 中 HoutH\_{out}Hout​ 和 WoutW\_{out}Wout​ 的计算遵循与卷积层完全相同的公式：

Hout=⌊Hin+2×填充[0]−空洞[0]×(卷积核尺寸[0]−1)−1步长[0]+1⌋H\_{out} = \lfloor \frac{H\_{in} + 2 \times \text{填充}[0] - \text{空洞}[0] \times (\text{卷积核尺寸}[0] - 1) - 1}{\text{步长}[0]} + 1 \rfloorHout​=⌊步长[0]Hin​+2×填充[0]−空洞[0]×(卷积核尺寸[0]−1)−1​+1⌋
Wout=⌊Win+2×填充[1]−空洞[1]×(卷积核尺寸[1]−1)−1步长[1]+1⌋W\_{out} = \lfloor \frac{W\_{in} + 2 \times \text{填充}[1] - \text{空洞}[1] \times (\text{卷积核尺寸}[1] - 1) - 1}{\text{步长}[1]} + 1 \rfloorWout​=⌊步长[1]Win​+2×填充[1]−空洞[1]×(卷积核尺寸[1]−1)−1​+1⌋

**重要区别：** 池化层**不**改变通道数量。因此，Cout=CinC\_{out} = C\_{in}Cout​=Cin​。

**示例：**

让我们使用我们第一个`conv_layer`的输出（形状为`[16, 64, 32, 32]`）并将其通过一个常见的最大池化层：

```python

pool_layer = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

pooled_output = pool_layer(output_tensor)
print(pooled_output.shape)
```

在此，具有2x2卷积核和步长为2的池化层将高度和宽度维度减半（`32x32` -> `16x16`），而通道数量保持不变（64）。

CNN\_Shapes

Input

输入
(N, 3, 32, 32)

Conv1

nn.Conv2d
(输出=64, K=3, S=1, P=1)
保持高宽

Input->Conv1

 (N, 64, 32, 32)

Pool1

nn.MaxPool2d
(K=2, S=2, P=0)
高宽减半

Conv1->Pool1

 (N, 64, 16, 16)

Output

输出
(N, 64, 16, 16)

Pool1->Output

> 张量维度通过一个示例卷积和池化层序列的流向。

### 实际中跟踪形状

在构建复杂的CNN时，手动计算形状会变得繁琐且容易出错。这里有一些实用建议：

1. **打印形状：** 在初始开发阶段，在层之后添加`print(x.shape)`语句以验证维度。
2. **使用虚拟输入：** 创建一个具有预期形状的虚拟输入张量，并将其逐步或逐层地通过你的网络定义，以查看形状如何变化。
3. **辅助函数：** 编写一个小的辅助函数，以层和输入形状作为参数 (parameter)，并使用公式计算输出形状。
4. **库/工具：** 一些库或工具（如`torchinfo`或`pytorch-summary`）可以自动总结你的模型，显示给定输入尺寸下每个层的输出形状。

掌握形状计算是在设计和调试CNN时必要的一步。通过理解`kernel_size`、`stride`、`padding`和`dilation`如何影响空间维度，以及`out_channels`如何决定深度，你可以放心地堆叠层来构建有效的深度学习 (deep learning)模型。

获取即时帮助、个性化解释和交互式代码示例。

---

### 循环神经网络 (RNN) 概述

### 循环神经网络 (RNN) 概述

前馈网络是独立处理输入的。然而，许多问题都涉及序列数据，其中顺序很重要，并且先前项的背景信息会影响当前项。例如，理解一个句子、预测股价或转录语音。每个词、价格点或声音片段都依赖于它之前的内容。标准的前馈网络缺乏一种内在机制来‘记住’序列中的过去信息。

这就是循环神经网络 (neural network)（RNN）的作用所在。它们通过引入**循环**的构想，专门设计用于处理序列数据。

### 记忆的构想：隐状态

RNN 的决定性特征是其内部循环。在处理序列的每一步，网络不仅考虑当前输入，还会考虑它从先前步骤中保留下来的信息。这些保留的信息存储在所谓的**隐状态**中。

设想你正在阅读一个句子。你不会孤立地处理每个词。你对当前词的把握会受到你已阅读词语的很大影响。RNN 中的隐状态就像这种运行中的总结或背景信息。它捕获了序列中先前元素的相关信息。

### 逐步处理序列

RNN 一次处理序列中的一个元素（或“时间步”）时。对于每个时间步 ttt：

1. 它接收该时间步的输入，我们称之为 xtx\_txt​。
2. 它还接收来自前一个时间步的隐状态 ht−1h\_{t-1}ht−1​。
3. 它使用一组学习到的权重 (weight)结合 xtx\_txt​ 和 ht−1h\_{t-1}ht−1​，以计算新的隐状态 hth\_tht​。这个新的隐状态现在包含了从所有步骤直到 ttt 的信息。
4. 可选地，它可以为当前时间步生成一个输出 yty\_tyt​，这通常基于隐状态 hth\_tht​。

重要的是，**每个时间步**都使用相同的一组权重（结合输入和先前状态以及生成输出的规则）。这种权重共享使得 RNN 效率高，并能使其将模式推广到不同长度的序列。

### 可视化循环：时间上的展开

通常，通过在时间上“展开”RNN 会有所帮助。我们可以绘制一条链来表示网络在每个时间步的状态，而不是绘制循环。


clusterₜₘinus₁

时间 t-1

clusterₜ

时间 t

clusterₜₚlus₁

时间 t+1

htₘinus₁

h(t-1)

cellₜₘinus₁

RNN 单元

htₘinus₁->cellₜₘinus₁

xtₘinus₁

x(t-1)

xtₘinus₁->cellₜₘinus₁

ytₘinus₁

y(t-1)

cellₜₘinus₁->ytₘinus₁

ht

h(t)

cellₜₘinus₁->ht

placeholderᵢn

placeholderᵢn->htₘinus₁

 ...

cellₜₚlus₁

RNN 单元

ht->cellₜₚlus₁

xt

x(t)

cellₜ

RNN 单元

xt->cellₜ

yt

y(t)

cellₜ->ht

cellₜ->yt

htₚlus₁

h(t+1)

placeholderₒut

htₚlus₁->placeholderₒut

 ...

xtₚlus₁

x(t+1)

xtₚlus₁->cellₜₚlus₁

ytₚlus₁

y(t+1)

cellₜₚlus₁->ytₚlus₁

> 一个在时间上“展开”的 RNN。相同的 RNN 单元（代表共享权重 (weight)）处理输入 xtx\_txt​ 和先前的隐状态 ht−1h\_{t-1}ht−1​，以生成新的隐状态 hth\_tht​ 和可选的输出 yty\_tyt​。隐状态从一个时间步传递到下一个时间步。

从数学角度看，简单 RNN 单元在时间步 ttt 内的核心计算通常表示为：

计算新的隐状态 hth\_tht​：

ht=tanh⁡(Whhht−1+Wxhxt+bh)h\_t = \tanh(W\_{hh} h\_{t-1} + W\_{xh} x\_t + b\_h)ht​=tanh(Whh​ht−1​+Wxh​xt​+bh​)

计算输出 yty\_tyt​：

yt=Whyht+byy\_t = W\_{hy} h\_t + b\_yyt​=Why​ht​+by​

这里：

- xtx\_txt​ 是时间步 ttt 的输入。
- ht−1h\_{t-1}ht−1​ 是来自前一个时间步的隐状态。
- hth\_tht​ 是时间步 ttt 的新隐状态。
- yty\_tyt​ 是时间步 ttt 的输出。
- WhhW\_{hh}Whh​、WxhW\_{xh}Wxh​ 和 WhyW\_{hy}Why​ 是在训练期间学习到的权重矩阵。它们分别代表了先前隐状态、当前输入和当前隐状态的影响程度。这些权重在所有时间步之间是**共享**的。
- bhb\_hbh​ 和 byb\_yby​ 是偏置 (bias)向量 (vector)，也是学习得到的。
- tanh⁡\tanhtanh 是双曲正切激活函数 (activation function)，常用于简单的 RNN 中以引入非线性。根据具体任务，输出层可以使用其他激活函数（例如，用于分类的 Softmax）。

重要之处在于 hth\_tht​ 的循环公式，它同时依赖于当前输入 xtx\_txt​ 和先前的隐状态 ht−1h\_{t-1}ht−1​。正是这种依赖性赋予了 RNN 记忆能力。

### RNN 的应用场景

RNN 在处理序列模式的任务中表现出色：

- **自然语言处理（NLP）：** 语言建模（预测下一个词）、机器翻译、情感分析、文本生成。
- **语音识别：** 将口语音频转换为文本。
- **时间序列分析：** 预测股价、天气预报、传感器数据分析。
- **视频分析：** 理解视频帧中随时间发生的动作。

### 挑战与后续

尽管功能强大，但像上面描述的简单 RNN 在学习长距离依赖时可能会遇到困难。来自早期时间步的信息在通过多个步骤传播时可能会被稀释或丢失，这个问题通常被称为**梯度消失问题**。反之，梯度有时可能会变得过大，这被称为**梯度爆炸问题**。

这些挑战促成了更精密的循环架构的发展，如长短期记忆（LSTM）和门控循环单元（GRU），它们使用门控机制来更好地控制信息流和记忆。本章稍后将简要提及这些内容。

目前，掌握循环的核心思想、隐状态的作用以及逐步处理过程就足够了。在接下来的部分中，我们将了解如何使用 PyTorch 的 `nn.RNN` 模块实现一个基本的 RNN。

获取即时帮助、个性化解释和交互式代码示例。

---

### 在PyTorch中构建一个简单的RNN

### 在PyTorch中构建一个简单的RNN

构建一个简单的RNN模型，使用PyTorch的`torch.nn`库，以展示循环神经网络 (neural network) (RNN)如何利用隐藏状态处理序列数据。PyTorch提供了一个便捷的模块`nn.RNN`，它封装了RNN的核心逻辑。

### `nn.RNN` 模块

PyTorch中基本RNN的主要构成部分是`torch.nn.RNN`类。当你创建这个类的一个实例时，你就在创建一个可以处理序列的RNN层（或者可能是多个堆叠层）。

`nn.RNN`层主要接收一个输入序列和一个可选的初始隐藏状态。然后，它会遍历输入序列的每个时间步，根据当前输入和前一个隐藏状态更新其隐藏状态。它会生成一个输出序列（每个时间步的隐藏状态）以及处理完整个序列后的最终隐藏状态。

要初始化一个`nn.RNN`层，你需要指定几个重要参数 (parameter)：

- `input_size`：这定义了在每个时间步输入xxx中期望的特征数量。例如，如果你正在处理维度为300的词嵌入 (embedding)，那么`input_size`将是300。
- `hidden_size`：这决定了隐藏状态hhh中的特征数量。它也定义了每个时间步输出的维度。`hidden_size`的选择是一个超参数 (hyperparameter)，会影响模型的容量。
- `num_layers`：这允许你堆叠多个RNN层。第一个层的输出序列成为第二个层的输入序列，依此类推。默认值为1。堆叠层有时可以帮助模型学习更复杂的时序模式。
- `nonlinearity`：要使用的非线性函数。可以是`'tanh'`（默认）或`'relu'`。
- `batch_first`：一个布尔型参数。如果为`True`，则输入和输出张量以`(batch_size, seq_len, feature_dim)`的形式提供。如果为`False`（默认值），则为`(seq_len, batch_size, feature_dim)`。当使用生成序列批次的数据加载器时，将此参数设为`True`通常更直观。
- `dropout`：如果非零，则在除最后一层之外的每个RNN层的输出上引入一个Dropout层，其Dropout概率等于`dropout`。默认值：0。
- `bidirectional`：如果为`True`，则成为一个双向RNN。默认值：`False`。我们目前将专注于单向RNN。

### 输入和输出的形状

理解输入和输出张量的预期形状对于正确使用`nn.RNN`是必不可少的。为了便于理解，我们假设`batch_first=True`，因为它很常用。

- **输入：** 输入序列应该是一个形状为`(batch_size, seq_len, input_size)`的张量。
  - `batch_size`：批次中的序列数量。
  - `seq_len`：每个序列的长度（时间步数）。
  - `input_size`：每个时间步的特征数量（与`nn.RNN`的`input_size`参数 (parameter)匹配）。
- **初始隐藏状态 (`h_0`)：** （可选）如果你想提供一个初始隐藏状态，它的形状应为`(num_layers, batch_size, hidden_size)`。如果未提供，则默认为全零张量。
- **输出序列 (`output`)：** 该张量包含来自*最后一层*RNN的、*每个*时间步的输出特征（隐藏状态）。其形状为`(batch_size, seq_len, hidden_size)`。
- **最终隐藏状态 (`h_n`)：** 该张量包含处理完整个序列后，*每个*RNN层的最终隐藏状态。其形状为`(num_layers, batch_size, hidden_size)`。你可以将此最终隐藏状态用作后续层（例如用于分类的线性层）的输入。

如果`batch_first=False`（默认值），则输入和输出序列张量中的`batch_size`和`seq_len`维度会互换。隐藏状态张量（`h_0`，`h_n`）始终将`batch_size`作为第二个维度，无论`batch_first`设置如何。

### 实现一个简单的RNN模型

让我们使用`nn.Module`创建一个基本的RNN模型。该模型将包含一个`nn.RNN`层，后面跟着一个`nn.Linear`层，用于将序列的最终隐藏状态映射到输出预测。这种模式在序列分类任务中很常见。

```python
import torch
import torch.nn as nn

class SimpleRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_rnn_layers=1):
        """
        初始化SimpleRNNModel。

        Args:
            input_dim (int): 每个时间步输入特征的维度。
            hidden_dim (int): RNN隐藏状态的维度。
            output_dim (int): 最终输出的维度。
            num_rnn_layers (int): 堆叠RNN层的数量。默认值为1。
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_rnn_layers = num_rnn_layers

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            nonlinearity='tanh'
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        定义模型的正向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, input_dim)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, output_dim)。
        """

        batch_size = x.size(0)
        h0 = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_dim).to(x.device)

        rnn_out, hn = self.rnn(x, h0)

        last_layer_hidden_state = hn[-1]

        out = self.fc(last_layer_hidden_state)

        return out

INPUT_DIM = 10
HIDDEN_DIM = 20
OUTPUT_DIM = 5
NUM_LAYERS = 1

model = SimpleRNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)
print("模型结构:")
print(model)

BATCH_SIZE = 4
SEQ_LEN = 15
dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)

output = model(dummy_input)

print(f"\n输入形状: {dummy_input.shape}")
print(f"输出形状: {output.shape}")

assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
```

### 代码分析

1. **初始化 (`__init__`)**：我们定义`nn.RNN`层，指定`input_dim`、`hidden_dim`、`num_rnn_layers`，以及重要的`batch_first=True`。我们还定义了一个标准的`nn.Linear`层（`self.fc`），它将接收来自RNN的最终隐藏状态作为输入并生成模型的最终输出。
2. **正向传播 (`forward`)**：
   - 我们首先从输入张量`x`中确定`batch_size`。
   - 创建一个形状为`(num_layers, batch_size, hidden_dim)`的全零初始隐藏状态`h0`。我们使用`.to(x.device)`确保它与输入`x`位于相同的设备上。
   - 输入`x`和初始隐藏状态`h0`被传递给`self.rnn`层。它返回两个张量：`rnn_out`（来自最后一层的所有时间步的隐藏状态）和`hn`（经过最后一个时间步后所有层的最终隐藏状态）。
   - 由于我们的目标通常是序列分类或摘要，我们通常使用最终隐藏状态。`hn`的形状是`(num_layers, batch_size, hidden_size)`。我们使用`hn[-1]`选择*最后一层*的最终隐藏状态，这会得到一个形状为`(batch_size, hidden_size)`的张量。
   - 这个最终隐藏状态`hn[-1]`通过全连接层`self.fc`，以获得形状为`(batch_size, output_dim)`的最终输出张量。
3. **示例用法**：我们实例化模型并创建随机输入数据，使其符合预期的`(batch_size, seq_len, input_size)`格式（因为我们设置了`batch_first=True`）。运行模型会生成一个输出张量，我们验证其形状是否为`(batch_size, output_dim)`，这适用于多类别分类等任务，其中批次中的每个序列都会被分配一个输出向量 (vector)。

获取即时帮助、个性化解释和交互式代码示例。

---

### 循环神经网络（RNN）的序列数据输入处理

### 循环神经网络（RNN）的序列数据输入处理

循环神经网络 (neural network)（RNN）专门用于处理序列数据，无论是句子中的词语、音乐作品中的音符，还是随时间变化的测量值。与独立处理固定大小输入的传统前馈网络不同，RNN在序列的每个步骤都会更新一个内部*隐藏状态*，这使得之前步骤的信息能够影响当前和未来步骤的处理。这种序列处理方式需要特定的输入数据格式。

### 标准RNN输入形状

PyTorch的RNN层（如 `nn.RNN`、`nn.LSTM`、`nn.GRU`）默认期望输入数据为一个三维张量，具有以下维度：

**(序列长度, 批次大小, 输入特征)**

我们来逐一分析每个维度：

1. **序列长度 (`seq_len`)**: 这是序列中的时间步数量。例如，如果您正在处理句子，并且批次中最长的句子有15个词，那么您的`seq_len`通常就是15（较短的句子需要填充，稍后讨论）。
2. **批次大小 (`batch_size`)**: 这表示您同时处理的独立序列的数量。分批训练是提高效率和更好地估计梯度的标准做法。
3. **输入特征 (`input_size` 或 `features`)**: 这是表示*每个*时间步输入特征的维度。如果您处理的是用50维嵌入 (embedding)表示的词语，那么`input_size`将是50。如果您处理的是单变量时间序列（每个时间步只有一个值），那么`input_size`将是1。

**示例：** 假设您想处理一个包含32个句子的批次，其中每个句子表示为20个词的序列，并且每个词都转换为100维向量 (vector)嵌入。输入张量的形状将是 `(20, 32, 100)`。

```python
import torch

seq_len = 20
batch_size = 32
input_features = 100

rnn_input = torch.randn(seq_len, batch_size, input_features)

print(f"标准RNN输入形状: {rnn_input.shape}")
```

### `batch_first` 选项

尽管 `(序列长度, 批次大小, 输入大小)` 是默认设置，但许多人认为将批次维度放在首位更符合直觉，这也与数据常见组织方式以及其他层类型（如卷积层或线性层）通常处理输入的方式保持一致。PyTorch RNN层为此提供 `batch_first` 参数 (parameter)。

如果您使用 `batch_first=True` 初始化RNN层，它将期望输入张量形状为：

**(批次大小, 序列长度, 输入特征)**

**示例（接续前文）：** 如果使用 `batch_first=True`，相同数据的形状将是 `(32, 20, 100)`。

```python
import torch
import torch.nn as nn

seq_len = 20
batch_size = 32
input_features = 100
hidden_size = 50

rnn_input_batch_first = torch.randn(batch_size, seq_len, input_features)

rnn_layer = nn.RNN(input_size=input_features, hidden_size=hidden_size, batch_first=True)

output, hidden_state = rnn_layer(rnn_input_batch_first)

print(f"批次维度优先的RNN输入形状: {rnn_input_batch_first.shape}")
print(f"批次维度优先的RNN输出形状: {output.shape}")
```

使用 `batch_first=True` 通常可以简化数据准备流程，因为数据集通常在加载时就是批次维度优先。请记住，如果设置了 `batch_first=True`，输出形状也将采用 `(批次大小, 序列长度, 隐藏大小)` 格式。

RNN\_Input

InputTensor

输入张量
(序列长度, 批次大小, 特征)
或
(批次大小, 序列长度, 特征)

Seq1

T0

T1

...

T(seqₗen-1)

InputTensor->Seq1

一个序列

FeatureVec

F0

F1

...

F(features-1)

Seq1->FeatureVec

一个时间步

Seq2

T0

T1

...

T(seqₗen-1)

> RNN输入数据结构的视觉表示。输入通常是一个三维张量，表示多个序列（批次），每个序列包含多个时间步，并且每个时间步具有多个特征。

### 处理变长序列

"一个常见的问题是，数据集中的序列长度很少完全相同（例如，句子的词数不同）。由于张量要求统一的维度，您需要使批次中的序列长度一致。这通常通过以下方式完成："

1. **填充：** 较短的序列会在末尾填充一个特殊值（通常是零），直到它们达到批次中最长序列的长度（`seq_len`）。
2. **打包（可选但推荐）：** 为了防止RNN处理这些无意义的填充值，PyTorch提供了实用工具（`torch.nn.utils.rnn.pack_padded_sequence` 和 `torch.nn.utils.rnn.pad_packed_sequence`）。您可以在将填充序列送入RNN之前“打包”它们，告诉RNN批次中每个序列的真实长度。然后，RNN只处理实际的数据点。之后您再“填充”输出以获取标准张量。虽然我们在此不详细说明打包，但它是使用变长数据进行高效且准确RNN训练的重要技术。

理解期望的输入形状（`(序列长度, 批次大小, 特征)`，如果 `batch_first=True` 则是 `(批次大小, 序列长度, 特征)`）对于正确准备数据并将其送入PyTorch的循环层必不可少。始终查阅您所用特定层的文档，并确保您的数据预处理流程生成所需形状的张量。

获取即时帮助、个性化解释和交互式代码示例。

---

### LSTM 和 GRU 简要介绍

### LSTM 和 GRU 简要介绍

简单的循环神经网络 (neural network) (RNN) 层，例如 PyTorch 中的 `nn.RNN`，通过维护隐藏状态来处理序列。然而，这些网络在学习跨越长时间跨度的模式时经常遇到困难。这种困难主要是由于梯度消失问题，即在通过许多时间步进行反向传播 (backpropagation)时，梯度变得极其小，从而阻碍了模型根据早期输入有效更新权重 (weight)的能力。

为了解决这个局限，开发了更复杂的循环单元。PyTorch 中现成的两种最受欢迎且有效的替代方案是长短期记忆 (LSTM) 和门控循环单元 (GRU)。

### 长短期记忆 (LSTM)

LSTM，由 Hochreiter & Schmidhuber 于 1997 年提出，其设计目的是应对梯度消失问题并更好地捕获长距离依赖关系。LSTM 的核心改进在于其内部结构，它不仅包含像简单 RNN 一样的隐藏状态 (hth\_tht​)，还包含一个独立的 *单元状态* (ctc\_tct​)。

可以将单元状态视为一条信息高速公路，它允许早期时间步的相关信息相对顺畅地流经网络。信息进出此单元状态的流动由三种称为 *门* 的专门机制调控：

1. **遗忘门：** 决定从前一个单元状态 (ct−1c\_{t-1}ct−1​) 中丢弃哪些信息。
2. **输入门：** 决定将当前输入 (xtx\_txt​) 和前一个隐藏状态 (ht−1h\_{t-1}ht−1​) 中的哪些新信息存储到当前单元状态 (ctc\_tct​) 中。
3. **输出门：** 决定单元状态的哪些部分应作为新的隐藏状态 (hth\_tht​) 输出。

这些门使用 sigmoid 激活函数 (activation function)（输出值在 0 到 1 之间）来控制信息通过的程度。这种门控机制允许 LSTM 有选择地长时间记忆信息并遗忘不相关的细节，使它们在涉及复杂序列模式的任务中非常有效，例如机器翻译、语言建模和语音识别。

在 PyTorch 中，可以通过 `torch.nn.LSTM` 层使用 LSTM。其用法在预期的输入/输出形状和初始化参数 (parameter)（如 `input_size`、`hidden_size`、`num_layers`）方面与 `nn.RNN` 非常相似。

```python
import torch
import torch.nn as nn

input_size = 10
hidden_size = 20
num_layers = 2

lstm_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

batch_size = 5
seq_length = 15
dummy_input = torch.randn(batch_size, seq_length, input_size)

h0 = torch.randn(num_layers, batch_size, hidden_size)
c0 = torch.randn(num_layers, batch_size, hidden_size)

output, (hn, cn) = lstm_layer(dummy_input, (h0, c0))

print("LSTM Output shape:", output.shape)
print("LSTM Final Hidden State shape:", hn.shape)
print("LSTM Final Cell State shape:", cn.shape)
```

### 门控循环单元 (GRU)

GRU，由 Cho 等人于 2014 年提出，是新一代的门控循环单元，它简化了 LSTM 架构。它们也旨在解决梯度消失问题并捕获长期依赖关系，但通过稍微不同且计算量较小的结构实现这一点。

GRU 将单元状态和隐藏状态合并为一个隐藏状态 (hth\_tht​)。它们只使用两个门：

1. **重置门：** 决定在提出新的候选隐藏状态时，要遗忘多少前一个隐藏状态 (ht−1h\_{t-1}ht−1​) 的信息。
2. **更新门：** 决定保留多少前一个隐藏状态 (ht−1h\_{t-1}ht−1​) 的信息，以及将多少新的候选隐藏状态的信息并入最终隐藏状态 (hth\_tht​)。

由于门更少且没有独立的单元状态，GRU 在相同隐藏大小时比 LSTM 具有更少的参数 (parameter)。这可以使它们训练更快，并且在较小数据集上可能更不容易过拟合 (overfitting)，同时在许多任务上通常能达到与 LSTM 相当的性能。

PyTorch 提供了 `torch.nn.GRU` 层，其用法与 `nn.RNN` 和 `nn.LSTM` 遵循相同的模式。

```python

gru_layer = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

h0_gru = torch.randn(num_layers, batch_size, hidden_size)

output_gru, hn_gru = gru_layer(dummy_input, h0_gru)

print("\nGRU Output shape:", output_gru.shape)
print("GRU Final Hidden State shape:", hn_gru.shape)
```

实际上，在处理需要长期依赖的序列数据时，LSTM 和 GRU 都被广泛用于替代简单 RNN。LSTM 和 GRU 之间的选择通常取决于对特定任务和数据集的经验评估，尽管当计算资源或训练时间受限时，GRU 可能因其更简单的结构而更受青睐。PyTorch 使得对两者进行实验变得简单。

获取即时帮助、个性化解释和交互式代码示例。

---

### 实践：实现基本CNN和RNN

### 实践：实现基本CNN和RNN

你将使用PyTorch的`nn.Module`以及相关层，实现卷积神经网络 (neural network)（CNN）和循环神经网络（RNN）这两种架构的基本版本。这种动手实践将巩固你对这些模型如何构建以及数据如何在其中流动的理解。

我们将侧重于定义模型结构和理解输入/输出维度，直接基于第4章的`nn.Module`知识和本章前面部分对层的说明进行构建。请记住，这些是简化示例；将它们集成到完整的训练循环中需要添加数据加载（第5章）、损失函数 (loss function)、优化器以及训练逻辑（第6章）。

## 实现一个基本CNN

CNN擅长处理网格状数据，例如图像。让我们构建一个可用于图像分类的简单CNN。我们将定义一个包含卷积层、激活函数 (activation function)、池化层和最终全连接层的网络。

### 定义CNN架构

我们创建一个继承自`nn.Module`的类。在`__init__`中，我们定义所需的层：用于卷积的`nn.Conv2d`，用于激活的`nn.ReLU`，用于池化的`nn.MaxPool2d`，以及用于最终分类的`nn.Linear`层。`forward`方法定义了输入数据如何流经这些层。

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = out.view(out.size(0), -1)

        out = self.fc(out)
        return out
```

在此示例中：

- 我们假设输入是灰度图像（1通道），类似于MNIST数据集中的图像，尺寸为28x28。
- `nn.Conv2d(in_channels=1, out_channels=16, ...)`：接收1个输入通道，应用16个滤波器。`kernel_size=3`、`stride=1`、`padding=1`是常见的选择，它们在卷积后保持空间维度。
- `nn.MaxPool2d(kernel_size=2, stride=2)`：将高度和宽度减半。
- 第二个池化层的输出形状为（批次，32，7，7）。
- `out.view(out.size(0), -1)`：将张量从形状（批次，32，7，7）展平为（批次，32 \* 7 \* 7）=（批次，1568），以便可以将其输入到线性层。
- `nn.Linear(32 * 7 * 7, num_classes)`：最后一层将展平后的特征映射到所需数量的输出类别。

### 使用虚拟数据测试CNN

让我们创建一些匹配预期形状（批次大小，通道，高度，宽度）的虚拟输入数据，并将其传入我们的网络以查看输出形状。

```python

cnn_model = SimpleCNN(num_classes=10)

dummy_input_cnn = torch.randn(4, 1, 28, 28, requires_grad=False)

output_cnn = cnn_model(dummy_input_cnn)

print(f"Input shape: {dummy_input_cnn.shape}")
print(f"Output shape: {output_cnn.shape}")
```

运行这段代码应该输出：

```
Input shape: torch.Size([4, 1, 28, 28])
Output shape: torch.Size([4, 10])
```

这确认了我们的网络接收一个包含4张图像的批次，并为每张图像输出10个类别的预测。请注意`forward`方法如何决定数据流向，以及我们如何需要根据最终池化层的输出形状来计算线性层的展平尺寸。你可以回顾“理解CNN层的输入/输出形状”一节，练习手动计算这些维度。

## 实现一个基本RNN

RNN旨在处理序列数据。例如，让我们构建一个可以处理字符序列或传感器读数的简单RNN。

### 定义RNN架构

我们将使用`nn.RNN`层。请记住，RNN层期望的输入格式是（序列长度，批次大小，输入特征）。

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=False)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):

        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)

        out, hn = self.rnn(x, h0)

        out_last_step = out[-1, :, :]

        final_output = self.fc(out_last_step)

        return final_output, hn
```

在此示例中：

- `input_size`：序列中每个步长的特征数量。
- `hidden_size`：隐藏状态中的特征数量。
- `num_layers`：堆叠的RNN层数。
- `nn.RNN(...)`：核心RNN层。`batch_first=False`是默认值，表示序列长度维度在前。
- `forward`方法接受输入序列`x`和一个可选的初始隐藏状态`h0`。如果未提供`h0`，则将其初始化为零。
- `nn.RNN`层返回`out`（每个时间步的输出）和`hn`（最终隐藏状态）。
- 我们经常使用*最后一个*时间步的输出（`out[-1, :, :]`）进行序列分类或预测任务，并将其通过一个最终的线性层。

### 使用虚拟数据测试RNN

让我们创建一个虚拟序列并将其传入我们的RNN。

```python

input_features = 10
hidden_nodes = 20
output_classes = 5
sequence_length = 15
batch_size = 4

rnn_model = SimpleRNN(input_size=input_features, hidden_size=hidden_nodes, output_size=output_classes)

dummy_input_rnn = torch.randn(sequence_length, batch_size, input_features, requires_grad=False)

output_rnn, final_hidden_state = rnn_model(dummy_input_rnn)

print(f"Input sequence shape: {dummy_input_rnn.shape}")
print(f"Output prediction shape: {output_rnn.shape}")
print(f"Final hidden state shape: {final_hidden_state.shape}")
```

运行这段代码应该产生类似如下的输出：

```
Input sequence shape: torch.Size([15, 4, 10])
Output prediction shape: torch.Size([4, 5])
Final hidden state shape: torch.Size([1, 4, 20])
```

这表明模型处理一个包含4个序列的批次，每个序列长15步，每步有10个特征。它为批次中的每个序列输出一个大小为5的最终预测向量 (vector)，以及最终的隐藏状态。隐藏状态的形状反映了（层数，批次大小，隐藏尺寸）。

## 更多实践

现在你已经实现了这些架构的基本版本，请尝试进行试验：

1. **CNN变体：**
   - 更改`nn.Conv2d`层中的`kernel_size`、`stride`或`padding`。在运行代码之前预测输出形状。当步长为1时，`padding='same'`如何影响输出维度？
   - 添加另一个卷积/池化块。请记住重新计算`nn.Linear`层的输入尺寸。
   - 更改卷积层中`out_channels`的数量。
2. **RNN变体：**
   - 增加`SimpleRNN`中的`num_layers`。观察初始隐藏状态`h0`和最终隐藏状态`hn`的形状。
   - 更改`hidden_size`。
   - 将`nn.RNN`替换为`nn.LSTM`或`nn.GRU`。请注意，`nn.LSTM`处理一个*元组*隐藏状态（隐藏状态和单元状态）。你需要相应调整隐藏状态的初始化和处理方式。输入/输出形状大体遵循相同的模式。
   - 修改`forward`方法，以使用*所有*时间步的输出（`out`）而不是仅最后一个，例如通过对每个步应用线性层或使用平均等聚合方法。

这次实践为构建CNN和RNN提供了扎实基础。通过理解如何定义这些层、在`forward`方法中连接它们以及管理它们的输入/输出形状，你将具备良好能力，能够使用PyTorch构建和调整这些强大的架构以完成各种深度学习 (deep learning)任务。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 8 Monitoring Debugging Models

### PyTorch开发中常见错误

### PyTorch开发中常见错误

随着您开始构建较复杂的PyTorch模型和训练循环，您将不可避免地遇到错误，或模型行为不符预期的情况。有些错误会给出清晰的消息，而另一些则可能更难察觉，导致性能不佳却无明显崩溃。识别常见模式是高效调试的第一步。以下是一些PyTorch开发中常出现的问题。

### 形状不匹配

PyTorch中最常见的运行时错误可能涉及张量形状不兼容。这通常发生在某层的输出形状与下一层的预期输入形状不匹配，或输入数据的形状与模型第一层不一致时。

考虑一个简单序列：一个卷积层后接一个全连接（线性）层。`nn.Conv2d`层需要形状为 (Batch Size, Input Channels, Height, Width) 的输入张量，通常简写为 (N,Cin,Hin,Win)(N, C\_{in}, H\_{in}, W\_{in})(N,Cin​,Hin​,Win​)。它会生成形状为 (N,Cout,Hout,Wout)(N, C\_{out}, H\_{out}, W\_{out})(N,Cout​,Hout​,Wout​) 的输出。然而，`nn.Linear`层需要一个形状为 (Batch Size, Input Features) 的2D输入，或 (N,输入特征数)(N, \text{输入特征数})(N,输入特征数)。直接连接它们而不改变形状会导致错误。

```python
import torch
import torch.nn as nn

conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
linear_layer = nn.Linear(in_features=???, out_features=10)

input_data = torch.randn(64, 3, 32, 32)

conv_output = conv_layer(input_data)
print(f"Conv output shape: {conv_output.shape}")

flattened_output = conv_output.view(conv_output.size(0), -1)
print(f"Flattened output shape: {flattened_output.shape}")

correct_linear_layer = nn.Linear(in_features=16384, out_features=10)
output = correct_linear_layer(flattened_output)
print(f"Final output shape: {output.shape}")
```

错误通常看起来像：`RuntimeError: size mismatch, m1: [64 x 16384], m2: [? x 10]`。`m1` 通常指传递给层的输入张量，`m2` 指的是层的权重 (weight)矩阵。错误消息表明PyTorch尝试进行乘法运算的形状。调试这些错误包括：

1. 使用`.shape`打印前一层输出张量的形状。
2. 计算有问题层预期的输入特征数。对于卷积后的`nn.Linear`，这通常涉及使用`tensor.view(batch_size, -1)`将 (N,C,H,W)(N, C, H, W)(N,C,H,W) 输出展平为 (N,C∗H∗W)(N, C\*H\*W)(N,C∗H∗W)。
3. 确保`nn.Linear`层的`in_features`参数 (parameter)与展平后的尺寸匹配。

### 设备不匹配 (CPU/GPU)

PyTorch允许在不同设备上进行计算，主要是在CPU和NVIDIA GPU（使用CUDA）。当您尝试对位于不同设备上的张量进行操作时，会经常发生运行时错误。例如，如果您的模型被移至GPU (`model.to('cuda')`)，但您的输入数据保留在CPU上，前向传播将失败。

```python

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model = nn.Linear(10, 5)
input_cpu = torch.randn(1, 10)

model.to(device)
print(f"Model device: {next(model.parameters()).device}")

try:
    output = model(input_cpu)
except RuntimeError as e:
    print(f"Error: {e}")

input_gpu = input_cpu.to(device)
print(f"Input tensor device: {input_gpu.device}")

output = model(input_gpu)
print(f"Output tensor device: {output.device}")
print("Forward pass successful!")
```


cluster\_gpu

GPU 内存

Input\_CPU

输入张量
(CPU)

Model\_GPU

模型
(GPU)

Input\_CPU->Model\_GPU

错误！
设备不匹配

Input\_GPU

输入张量
(GPU)

Input\_CPU->Input\_GPU

.to(device)

Output\_GPU

输出张量
(GPU)

Model\_GPU->Output\_GPU

Input\_GPU->Model\_GPU

前向传播

> 导致设备不匹配错误的常见情况。在前向传播之前，输入张量必须移动到与模型相同的设备上（例如GPU）。

错误消息 `RuntimeError: Expected all tensors to be on the same device...` 相当明确。调试方法包括：

1. 在脚本早期确定目标`device`（检查`torch.cuda.is_available()`）。
2. 使用`model.to(device)`将模型移动到目标设备。
3. 在训练或评估循环中，使用`data = data.to(device)`和`targets = targets.to(device)`将输入数据张量明确地移动到同一目标设备。
4. 如果不确定，检查张量和模型参数 (parameter)的`.device`属性（`next(model.parameters()).device`）。

### 损失函数 (loss function)或目标形状/类型不正确

选择正确的损失函数很重要，但您还需要确保您的模型输出和目标标签具有该损失函数预期的形状和数据类型。使用错误的组合可能不会总是导致崩溃，但会导致“静默”失败，即损失减小但模型未能正确学习实际任务。

- **`nn.CrossEntropyLoss`**：常用于多类别分类。
  - 需要模型输出的原始、未归一化 (normalization)分数（logits），通常形状为 (N,C)(N, C)(N,C)，N 表示批量大小，C 表示类别数量。
  - 需要目标标签为类别索引（长整型），通常形状为 (N)(N)(N)。
  - 通常不应在此损失函数 *之前* 应用 `softmax` 函数，因为`CrossEntropyLoss`结合了`LogSoftmax`和`NLLLoss`。
- **`nn.MSELoss` (均方误差)**：常用于回归任务。
  - 需要模型输出和目标张量具有相同的形状（例如，(N,输出特征数)(N, \text{输出特征数})(N,输出特征数)）。
  - 两个张量通常都应是浮点类型。
- **`nn.BCEWithLogitsLoss`**：用于二元分类或多标签分类。
  - 需要模型输出的原始logits，形状为 (N,∗)(N, \*)(N,∗)，这里的 ∗\*∗ 是一个或多个维度。
  - 需要目标标签为相同形状 (N,∗)(N, \*)(N,∗) 的概率（浮点数），通常包含0和1。
  - 在此损失函数 *之前* 使用 `sigmoid` 是不正确的，因为它已包含了 sigmoid 计算。

此处的不匹配可能导致：

- 如果形状基本不兼容，则引发 `RuntimeError`。
- 如果数据类型错误（例如，向`CrossEntropyLoss`提供浮点型目标），则梯度计算不正确。
- 模型进行训练但学习效果不佳，因为损失函数衡量的是错误的东西（例如，将`MSELoss`用于分类索引）。

调试需要仔细阅读您选择的损失函数在PyTorch文档中的说明，并验证：

1. 模型输出张量的形状。
2. 目标张量的形状和数据类型（`.dtype`）。
3. 损失函数 *之前* 是否应应用任何激活函数 (activation function)（如`softmax`或`sigmoid`）。

### 梯度流问题

有时，梯度未能如预期地反向传播 (backpropagation)通过网络，导致参数 (parameter)没有得到更新。如果不监控，这可能会静默发生。

- **忘记 `requires_grad=True`:** 尽管`nn.Module`参数会自动带有`requires_grad=True`，但如果您创建的中间张量应是计算图的一部分，请确保它们正确设置了此标志。通常，如果它们是已需要梯度的张量运算的结果，这会自动处理。
- **原地操作:** 某些原地操作（如`tensor.add_()`）可能会干扰较旧PyTorch版本或复杂的计算图中的梯度跟踪。虽然PyTorch已改进其处理方式，但在需要梯度的网络计算中，通常更安全地使用非原地操作版本（`y = x + 1`而不是`x += 1`）。
- **使用NumPy:** 将张量转换为NumPy（`.numpy()`）会将其从计算图中分离。使用该NumPy数组进行的任何后续操作，即使转换回张量，也不会有梯度流回图的原始部分。
- **`.detach()`:** 对张量调用`.detach()`会明确地将其从计算图中移除。这有时是必要的（例如，在评估期间），但如果在训练期间意外使用它将停止梯度。

这种问题的症状是发现某些参数的`.grad`属性在`loss.backward()`之后保持`None`，或者尽管训练循环正在运行，模型性能却没有改善。本章后续您将学习如何更规范地检查梯度。

### 数据加载与预处理错误

错误也可能源于您的`Dataset`实现或数据转换。

- **`__getitem__`不正确:** 返回错误类型的数据（例如，如果模型期望张量，却返回NumPy数组而非张量图像）或不正确的形状。
- **大小不一致:** 如果`__getitem__`返回大小不一的张量（例如，变长序列或不同维度的图像），并且`DataLoader`的`collate_fn`未能正确处理这种填充或堆叠，这可能在批量创建期间导致错误。
- **转换错误:** 不正确地应用转换（例如，错误的归一化 (normalization)常数，在`transforms.Compose`流程中过早或过晚地转换为张量）可以静默地破坏输入到模型的数据。

调试这些错误通常包括：

1. 单独实例化您的`Dataset`。
2. 使用`dataset[i]`手动获取少量项目。
3. 在样本进入`DataLoader` *之前*，检查其形状、数据类型和值范围。

了解这些常见问题有助于您预见潜在问题，并在问题出现时提供诊断的起点。后续章节将提供更结构化的方法，用于调试和监控模型。

获取即时帮助、个性化解释和交互式代码示例。

---

### 调试张量形状不匹配

### 调试张量形状不匹配

在PyTorch中开发神经网络 (neural network)时，张量形状不匹配可能是最常遇到的运行时错误。当传递给层或操作的张量维度与该层或操作的预期不一致时，就会出现这些错误。诊断这些问题是调试过程的主要部分。弄清楚如何追踪和修正形状不兼容问题，对于构建能正常运行的模型很重要。

形状错误通常是因为不同层对其输入张量的维度和大小有特定要求而出现的。例如，线性层需要`(batch_size, in_features)`的二维输入，而二维卷积层需要`(batch_size, in_channels, height, width)`这样的四维输入。执行矩阵乘法或逐元素相加等操作时，也会对操作张量的形状有严格要求。

### 形状不匹配的常见原因

让我们看看一些形状错误出现的典型情形：

1. **线性层 (`nn.Linear`):** 定义为`nn.Linear(in_features, out_features)`的线性层，要求其输入张量`x`的最后一个维度与`in_features`匹配。一个常见错误是在展平卷积层输出后，给它提供了特征数量不正确的张量。

   - **错误示例:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x1024 and 512x10)` - 这里，输入批次有1024个特征，但线性层在定义时预期的是512个。
2. **卷积层 (`nn.Conv2d`):** 这些层要求输入张量形状为 `(N, C_in, H, W)`，其中 `N` 是批次大小，`C_in` 是输入通道数，`H`、`W` 是空间高度和宽度。如果通道维度不正确或输入张量不是四维的，就可能出现错误。

   - **错误示例:** `RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input[64, 1, 28, 28] to have 3 channels, but got 1 channels instead.` - 该层预期有3个输入通道（如RGB），但收到的输入只有1个通道（如灰度图）。
3. **展平操作:** 从卷积/池化层过渡到线性层时，张量需要展平。展平后维度大小的计算错误是后续线性层经常出错的原因。`nn.Flatten()`层可以帮助自动化此过程，但您仍然需要确保第一个线性层的`in_features`与展平后的总元素数量匹配。
4. **批次维度问题:** PyTorch层通常要求输入数据包含一个批次维度，即使批次大小为1。在将数据传递给模型之前，忘记添加此维度（例如，对于单个样本使用`tensor.unsqueeze(0)`）可能导致形状错误。
5. **矩阵乘法 (`torch.matmul`, `@`):** 标准矩阵乘法规则适用。对于 `A @ B`，`A` 的列数必须等于 `B` 的行数。如果这些维度不匹配，就会出现错误。
6. **逐元素操作:** 张量之间的加法 (`+`)、减法 (`-`) 或乘法 (`*`) 等操作通常要求张量具有完全相同的形状，或者根据广播规则兼容。如果形状不兼容且无法广播，则会发生`RuntimeError`。

### 调试形状错误的方法

系统地查找形状不匹配的来源，需要追踪张量维度在您的模型或操作中的变化。以下是有效的方法：

1. **打印张量形状:** 这是最直接的方式。在模型的`forward`方法或训练循环中的不同位置插入打印语句，以观察张量形状如何变化。使用f-string可以使代码更整洁：

   ```python
   import torch
   import torch.nn as nn

   x = some_layer(x)
   print(f"Shape after some_layer: {x.shape}")
   x = next_layer(x)
   print(f"Shape after next_layer: {x.shape}")
   ```

   通过比较打印出的形状与下一层的预期输入形状，您可以定位到不匹配发生的位置。
2. **理解错误信息:** PyTorch运行时错误通常提供详细信息，包括失败的操作和涉及的张量形状。仔细阅读这些信息。它们通常看起来像这样：
   `RuntimeError: size mismatch, m1: [A x B], m2: [C x D] ...`
   这会直接告知您在特定操作（通常是矩阵乘法）期间导致不兼容的形状（`[A x B]`、`[C x D]`）。
3. **查阅层文档:** 如果您不确定某个PyTorch层（例如`nn.Conv2d`、`nn.LSTM`、`nn.BatchNorm1d`）的预期输入或输出形状，请查阅PyTorch官方文档。它清楚地说明了所需的维度，以及如何根据核大小、步长、填充等参数 (parameter)计算输出形状。
4. **手动计算形状（尤其是对于CNN）:** 对于卷积层和池化层，了解如何计算输出空间维度很重要。公式涉及输入大小、核大小、填充和步长。手动计算几层后的预期输出形状有助于检查您的网络架构是否与您的设想一致。例如，`Conv2d`层的输出高度 HoutH\_{out}Hout​ 通常按如下方式计算：

   Hout=⌊Hin+2×padding−dilation×(kernel\_size−1)−1stride+1⌋H\_{out} = \lfloor \frac{H\_{in} + 2 \times \text{padding} - \text{dilation} \times (\text{kernel\\_size} - 1) - 1}{\text{stride}} + 1 \rfloorHout​=⌊strideHin​+2×padding−dilation×(kernel\_size−1)−1​+1⌋

   类似的公式也适用于宽度 WoutW\_{out}Wout​。了解这些有助于预测后续层所需的输入大小，特别是展平后的全连接层。
5. **明智地使用 `nn.Flatten`:** 从卷积层到线性层转换时，使用`nn.Flatten(start_dim=1)`通常比手动使用`view`进行重塑更安全。它会将从`start_dim`开始的所有维度（通常为1，以保持批次维度独立）展平为一个维度。但是，您仍然需要确保后续`nn.Linear`层的`in_features`与此展平后的大小匹配。

   ```python

   flatten = nn.Flatten()
   flat_output = flatten(conv_output)

   num_features = flat_output.shape[1]
   linear_layer = nn.Linear(num_features, num_classes)
   output = linear_layer(flat_output)
   ```
6. **使用调试器单步执行:** 对于复杂的模型或难以发现的错误，使用Python调试器（如本章后面讨论的`pdb`）可以逐行执行代码，并在模型的`forward`传递或训练循环中的每一步检查张量形状（`x.shape`）。

### 示例：修正展平不匹配

考虑一个简单的CNN，后接一个线性层：

```python
import torch
import torch.nn as nn

dummy_input = torch.randn(1, 1, 28, 28)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(10 * 10 * 10, 50)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.pool(self.relu(self.conv1(x)))
        print(f"Shape after conv/pool: {x.shape}")

        x = x.view(x.size(0), -1)
        print(f"Shape after flattening: {x.shape}")

        try:
            x = self.fc1(x)
        except RuntimeError as e:
            print(f"\nError occurred: {e}")
            print(f"Input shape to fc1: {x.shape}")
            print(f"fc1 expects input features: {self.fc1.in_features}")

model = SimpleNet()
model(dummy_input)
```

运行这段代码（包含`try-except`块）会打印出：

```
Input shape: torch.Size([1, 1, 28, 28])
Shape after conv/pool: torch.Size([1, 10, 12, 12])
Shape after flattening: torch.Size([1, 1440])

Error occurred: mat1 and mat2 shapes cannot be multiplied (1x1440 and 1000x50)
Input shape to fc1: torch.Size([1, 1440])
fc1 expects input features: 1000
```

打印出的语句和错误信息清楚地显示了不匹配：展平后的张量有1440个特征，但`fc1`在定义时`in_features=1000`。

**修正方法:** 使用从池化层输出计算出的正确输入特征数量（`10 * 12 * 12 = 1440`）重新定义`fc1`：

```python

self.fc1 = nn.Linear(10 * 12 * 12, 50)
```

另外，使用`nn.LazyLinear`会将`in_features`的初始化推迟到第一次前向传递时，自动正确设置它，尽管显式定义有助于理解。


Input

输入
(N, 1, 28, 28)

Conv1

nn.Conv2d(1, 10, ks=5)
(N, 10, 24, 24)

Input->Conv1

Pool1

nn.MaxPool2d(2)
(N, 10, 12, 12)

Conv1->Pool1

 ReLU

Flatten

展平
(N, 10\*12\*12 = 1440)

Pool1->Flatten

Linear1

nn.Linear(1440, 50)
(N, 50)

Flatten->Linear1

Output

输出
(N, 50)

Linear1->Output

> 张量形状通过一个简单CNN的流程，显示了线性层之前的展平步骤。N 代表批次大小。

调试形状不匹配通常感觉像侦探工作。通过系统地检查每一步的张量维度，弄清层要求，并仔细阅读错误信息，您可以有效处理这些常见问题，并确保您的模型架构实现正确。

获取即时帮助、个性化解释和交互式代码示例。

---

### 检查设备放置 (CPU/GPU)

### 检查设备放置 (CPU/GPU)

在 PyTorch 中使用 GPU 时，最常见的运行时错误之一源于尝试在位于不同设备（CPU 与 GPU）上的张量或模块之间执行操作。PyTorch 要求操作中涉及的张量以及执行操作的模型位于同一设备上。未能确保这种一致性会导致明确的错误，并停止执行。识别和纠正这些设备放置问题是主要关注点。

### 理解设备专属性

PyTorch 张量和模型参数 (parameter)有特定的关联设备：可以是 CPU 或特定的 GPU。默认情况下，张量是在 CPU 上创建的。为了使用 GPU 提供的加速，您必须将模型和数据都明确地移动到 GPU 上。

操作通常要求所有参与的张量都在同一设备上。例如，您不能直接将位于 CPU 上的张量与位于 GPU 上的张量相加。同样，位于 GPU 上的模型层（其中包含参数，参数本身也是张量）不能直接处理仍位于 CPU 上的输入张量。

### 识别设备不匹配

此问题最常见的表现是 `RuntimeError`，通常带有类似以下的消息：

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument mat1 in method wrapper_addmm)
```

此错误消息很有说明性。它告诉您：

1. 存在设备不匹配。
2. 它识别出涉及的设备（例如 `cpu` 和 `cuda:0`，它表示第一个 GPU）。
3. 它通常指向发生不匹配的具体操作（例如 `addmm`，它用于线性层）。

这通常发生在模型的正向传播或计算损失时，因为在这些地方模型参数 (parameter)直接与输入数据或标签进行交互。

### 检查张量和模型的设备

为了调试这些错误，您首先需要确定您的张量和模型参数 (parameter)位于何处。

**检查张量的设备：**
每个张量都有一个 `.device` 属性，它告诉您其当前位置。

```python
import torch

cpu_tensor = torch.randn(2, 2)
print(f"cpu_tensor 位于: {cpu_tensor.device}")

if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.to("cuda")
    print(f"gpu_tensor 位于: {gpu_tensor.device}")
else:
    print("GPU 不可用，无法创建 gpu_tensor。")
```

**检查模型的设备：**
使用 `torch.nn.Module` 定义的模型也需要位于正确的设备上。由于模型由包含参数（参数本身是张量）的层组成，您可以检查任何参数的设备，以推断模型的实际设备。一个常见方法是检查第一个参数的设备：

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

model = SimpleNet()

print(f"模型最初位于: {next(model.parameters()).device}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
    print(f"模型已移至: {next(model.parameters()).device}")
else:
    device = torch.device("cpu")
    print("GPU 不可用，模型仍留在 CPU 上。")
```

请注意，`model.to(device)` 会就地修改模型的参数和缓冲区，*如果*模型已在目标设备上，否则会返回一个已移动到设备上的*新*模型对象。通常的做法是重新赋值结果，例如 `model = model.to(device)`，尽管不重新赋值地调用 `model.to(device)` 通常也有效，因为它会修改内部状态。但是，明确的重新赋值更安全、更清晰。

### 纠正设备放置

一旦您识别出不匹配，解决方案是使用 `.to(device)` 方法将相关的对象（张量或模型）移动到所需的公共设备上。

**建立设备环境：**
一种标准做法是在脚本开始时定义一个 `device` 对象。此对象保存目标设备（如果 GPU 可用则为 GPU，否则为 CPU），并且可以在您的代码中重复使用。

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = SimpleNet().to(device)

for inputs, labels in data_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

通过在训练开始*前*持续应用 `.to(device)` 到您的模型，以及在训练循环*内*（对于每个批次）应用于输入数据，您可以确保所有计算都在目标设备上进行，从而防止设备不匹配错误。

### 设备错误的调试流程

如果您遇到指示设备不匹配的 `RuntimeError`：

1. **识别出错的行：** 查看堆栈跟踪以精确找出导致错误的具体操作。
2. **检查操作数：** 就在出错行之前，插入打印语句或使用调试器检查该操作中涉及的所有张量和模型参数 (parameter)的 `.device` 属性。例如，如果错误发生在 `outputs = model(inputs)` 期间，检查 `inputs.device` 和 `next(model.parameters()).device`。如果错误发生在 `loss = criterion(outputs, labels)` 期间，检查 `outputs.device` 和 `labels.device`。
3. **应用 `.to(device)`：** 确保任何被识别为位于错误设备上的张量或模型都使用 `.to(device)` 明确移动到出错操作发生*之前*。请记住在数据加载循环内移动输入和标签。

检查和管理设备放置是编写 PyTorch 代码的基本方面，特别是在使用 GPU 进行加速时。采纳尽早定义 `device` 对象并相应地持续移动模型和数据的做法将帮助您避免许多常见的运行时错误。

获取即时帮助、个性化解释和交互式代码示例。

---

### 检查梯度问题（消失/爆炸）

### 检查梯度问题（消失/爆炸）

有效的训练依赖于反向传播 (backpropagation)期间计算的梯度。这些梯度引导优化器更新模型参数 (parameter)以最小化损失函数 (loss function)。然而，这些梯度的大小有时会成为问题，导致训练不稳定或停滞。两个常见问题是梯度消失和梯度爆炸。了解如何检查梯度是诊断训练中出现的问题的一项重要技能。

### 理解梯度问题

在反向传播 (backpropagation)过程中，梯度使用链式法则逐层计算。在深度网络中，这涉及将许多小数字（导数）相乘。

- **梯度消失：** 当梯度从输出层向初始层反向传播时变得极其小，就会发生这种情况。结果是，初始层的权重 (weight)和偏置 (bias)更新得非常缓慢，甚至完全不更新。网络基本上停止从早期层的数据中学习有意义的特征。这在深度网络中尤其常见，当使用 sigmoid 或 tanh 等激活函数 (activation function)时，这些函数在大多数区域的导数都小于 1。
- **梯度爆炸：** 这是相反的问题，即在反向传播过程中梯度变得过大。大梯度会导致模型权重发生显著更新。这可能导致优化过程变得不稳定，损失剧烈波动甚至变成 `NaN`（非数字），从而有效地停止训练。梯度爆炸可能由于不佳的权重初始化、过高的学习率或某些网络结构引起，尤其是在循环神经网络 (neural network) (RNN)中。

### 在 PyTorch 中检测梯度问题

PyTorch 的 Autograd 系统计算梯度，并将其存储在 `requires_grad=True` 的张量的 `.grad` 属性中。这些梯度在 `loss.backward()` 调用后即可访问，并在 `optimizer.step()` 更新模型参数 (parameter)或 `optimizer.zero_grad()` 清除梯度之前保持可用。

#### 监控整体梯度范数

一个常用做法是监控模型中所有可训练参数的梯度整体大小（范数）。L2 范数（欧几里得范数）是常用的一种。非常小的范数表明梯度消失，而非常大或 `NaN` 的范数则表明梯度爆炸。

以下是在训练循环中计算并记录总梯度范数的方法：

```python

total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5

print(f"总梯度范数: {total_norm}")
```

随时间监控此值可以提供信息：

2468100.01110010k稳定梯度梯度爆炸梯度消失训练期间的总梯度范数训练步数总 L2 范数

> 模型梯度总 L2 范数随训练步数变化的趋势，以对数尺度显示。稳定的训练显示相对一致的范数，梯度爆炸显示快速增加（常导致 NaN），梯度消失则显示趋近于零的下降。

#### 逐层检查梯度

有时，梯度问题可能局限于特定层。你可以直接检查单个参数的梯度。

```python

if hasattr(model, 'conv1') and model.conv1.weight.grad is not None:
    conv1_grad_mean = model.conv1.weight.grad.abs().mean().item()
    conv1_grad_max = model.conv1.weight.grad.abs().max().item()
    print(f"层 conv1 - 平均绝对梯度: {conv1_grad_mean:.6f}, 最大绝对梯度: {conv1_grad_max:.6f}")

if hasattr(model, 'fc2') and model.fc2.bias.grad is not None:
    fc2_bias_grad_norm = model.fc2.bias.grad.norm(2).item()
    print(f"层 fc2 (偏置) - L2 范数: {fc2_bias_grad_norm:.6f}")
```

查看平均或最大绝对梯度值，或特定层的范数，可以帮助确定梯度是在减小还是在不受控制地增长。使用直方图（例如，使用 Matplotlib 或通过 TensorBoard 记录）来可视化某一层的梯度值分布也很有用。

#### 使用钩子进行更细致的检查

为了进行更详细的调试，PyTorch 提供了钩子。可以在任何 `nn.Module` 上注册一个反向钩子（`register_full_backward_hook`）。当为该模块计算了梯度时，此钩子会执行，允许你检查甚至修改通过它的梯度（`grad_input`，`grad_output`）。尽管功能强大，但钩子会增加复杂性，通常在简单检查方法不足时使用。

#### 观察损失行为

间接来看，训练损失本身就是一个强有力的指示器。

- **损失变为 `NaN`：** 几乎总是梯度爆炸或数学上无效操作（如 `log(0)`）的迹象。
- **损失下降极其缓慢或过早停滞：** 可能是梯度消失的症状，特别是如果涉及初始层。
- **损失剧烈波动：** 可能表示梯度爆炸或学习率过高。

### 缓解的初步措施

检测梯度问题是第一步。解决这些问题通常涉及其他地方更详细介绍的技术，但常见策略包括：

- **梯度裁剪：** 对于梯度爆炸，在优化器步骤之前限制梯度的最大范数或值。`torch.nn.utils.clip_grad_norm_` 或 `torch.nn.utils.clip_grad_value_` 是标准实用工具。
- **激活函数 (activation function)：** 在深度网络中用 ReLU 或其变体（Leaky ReLU, PReLU, ELU）替换 sigmoid/tanh，这些函数通常具有问题较少的导数特性。
- **权重 (weight)初始化：** 使用旨在维持各层方差的初始化方案，例如 Xavier/Glorot 或 He 初始化。
- **批量归一化 (normalization)：** 通过归一化层输入，有助于稳定学习并可以缓解梯度消失/爆炸问题。
- **网络架构：** 使用跳跃连接或残差连接（如 ResNets 中）为梯度流动提供替代路径，从而在非常深的神经网络 (neural network)中对抗梯度消失。
- **学习率调整：** 降低学习率有时可以帮助解决梯度爆炸问题，尽管它可能无法解决根本原因。

模型工作后，你不一定需要在每次训练运行时都检查梯度，但当训练不稳定或无效时，它是一个必不可少的诊断工具。通过监控梯度范数和检查单个层的梯度，你可以获得关于训练动态的有价值信息，并发现潜在的梯度消失或梯度爆炸问题。

获取即时帮助、个性化解释和交互式代码示例。

---

### 使用 TensorBoard 可视化训练进度

### 使用 TensorBoard 可视化训练进度

虽然打印语句和手动记录可以提供损失或准确率等指标的快照，但它们往往无法清晰地展示整个训练过程中的趋势和变化。损失是在持续下降，还是在剧烈波动？验证准确率是否停滞不前？使用可视化工具回答这些问题会容易得多。TensorBoard 是一个功能强大的可视化工具包，最初为 TensorFlow 开发，它通过 `torch.utils.tensorboard` 模块与 PyTorch 顺畅结合。它能让您在基于网络的仪表板中跟踪和可视化模型训练的各个方面。

### 设置 SummaryWriter

PyTorch 中用于将数据记录到 TensorBoard 的主要接口是 `SummaryWriter` 类。您通常会在训练脚本的开头实例化它。

```python
from torch.utils.tensorboard import SummaryWriter
import torch

log_dir = 'runs/my_first_experiment'
writer = SummaryWriter(log_dir)

print(f"TensorBoard 日志目录: {log_dir}")
```

`SummaryWriter` 会将事件文件写入指定的 `log_dir`。TensorBoard 读取这些文件来生成可视化图表。最佳做法是为不同实验（例如，改变超参数 (parameter) (hyperparameter)）使用不同的目录，以便您可以轻松比较各个运行。

### 记录标量值

TensorBoard 最常用的场景是记录随时间变化的标量值，例如损失和准确率。这通过 `add_scalar` 方法完成。

`writer.add_scalar(tag, scalar_value, global_step=None)`

- `tag` (字符串): 标量的名称，例如 'Training Loss' 或 'Validation Accuracy'。在标签中使用斜杠（例如，'Loss/train', 'Loss/validation'）有助于在 TensorBoard 用户界面中组织图表。
- `scalar_value` (浮点数或整数): 您想记录的值。请注意，这应该是一个 CPU 标量值。如果您的损失在 GPU 上，您需要使用 `.item()` 将其移至 CPU。
- `global_step` (整数): 与此数据点关联的步数，通常表示 epoch 数或批次迭代计数。这决定了图中 x 轴的值。

让我们看看如何将其整合到一个典型的训练和验证循环结构中：

```python

model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_train_samples = 0

    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_train_samples += inputs.size(0)

        log_interval = 100
        if i % log_interval == log_interval - 1:
            current_step = epoch * len(train_loader) + i
            avg_batch_loss = running_loss / (log_interval * train_loader.batch_size)
            writer.add_scalar('Loss/train_batch', avg_batch_loss, current_step)

    epoch_loss = running_loss / total_train_samples
    writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
    print(f'Epoch {epoch+1}/{num_epochs}, 训练损失: {epoch_loss:.4f}')

    model.eval()
    validation_loss = 0.0
    correct = 0
    total_val_samples = 0
    with torch.no_grad():
        for data in valid_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_val_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = validation_loss / total_val_samples
    accuracy = 100.0 * correct / total_val_samples

    writer.add_scalar('Loss/validation', avg_val_loss, epoch)
    writer.add_scalar('Accuracy/validation', accuracy, epoch)
    print(f'Epoch {epoch+1}/{num_epochs}, 验证损失: {avg_val_loss:.4f}, 准确率: {accuracy:.2f}%')

writer.close()
print("训练完成。TensorBoard 日志已保存。")
```

在这个例子中:

1. 我们在训练循环之前实例化了 `SummaryWriter`。
2. 在训练循环中，我们定期记录每个批次的平均训练损失以及每个 epoch 的平均损失。
3. 在验证循环中，我们计算并记录每个 epoch 的平均验证损失和准确率。
4. 我们将 `epoch` 数字用作 epoch 级别指标的 `global_step`。对于批次级别指标，我们根据 epoch 和批次索引计算组合步数。
5. 重要的是，我们在训练结束后调用 `writer.close()` 以确保所有缓冲数据都写入磁盘。

### 启动和使用 TensorBoard

您的脚本运行并生成日志文件后，您可以从终端启动 TensorBoard 界面。导航到包含 `runs`（或自定义日志）目录的父目录并运行：

```bash
tensorboard --logdir runs
```

如果您使用了特定的目录，例如 `logs/my_experiment_1`，您将使用：

```bash
tensorboard --logdir logs/my_experiment_1
```

TensorBoard 通常会启动一个网络服务器，通常在 `http://localhost:6006`。在您的网络浏览器中打开此地址。您应该会看到一个仪表板，您可以在其中查看记录的标量、比较不同运行（如果您的 `logdir` 中有多个子目录），并观察 epoch 或步数的变化趋势。

01234567890.511.52304050607080损失/训练 epoch损失/验证准确率/验证TensorBoard 指标示例周期损失准确率 (%)

> 一个类似于 TensorBoard 可能显示的图表示例，展示了训练损失、验证损失和验证准确率在不同 epoch 上的变化。

### 关于标量

记录标量是基础操作，但 `SummaryWriter` 也提供了可视化其他类型数据的方法，这对于更具体的调试情况很有用：

- `add_histogram(tag, values, global_step)`: 跟踪张量值随时间的变化分布。这对于监控不同层中权重 (weight)或梯度的分布非常有用，可以帮助诊断梯度消失或梯度爆炸等问题。
- `add_graph(model, input_to_model)`: 可视化模型的架构。您传入 `nn.Module` 和一个示例输入张量。TensorBoard 会显示操作图，这有助于验证连接和形状。请注意，动态控制流（例如依赖于张量值的 `if` 语句）可能无法完全显示。
- `add_image(tag, img_tensor, global_step)`: 记录图像。在计算机视觉任务中很有用，可以在训练期间查看示例输入、输出或生成的图像。`img_tensor` 的格式需要仔细处理（例如 `CHW` 或 `NCHW`）。
- `add_embedding(mat, metadata, label_img, global_step, tag)`: 使用 PCA 或 t-SNE 等技术将高维嵌入 (embedding)（如词嵌入或图像特征）可视化到较低维空间。

对于中级课程，掌握 `add_scalar` 是最重要的一步。随着遇到更复杂的调试难题，试用 `add_histogram` 用于权重/梯度以及 `add_graph` 用于模型结构是很好的后续步骤。

使用 TensorBoard 将调试从解读一串数字转变为分析视觉趋势。它能帮助了解收敛速度、潜在的过拟合 (overfitting)（比较训练损失与验证损失），以及学习过程的稳定性，使其成为实际深度学习 (deep learning)开发中必不可少的工具。

获取即时帮助、个性化解释和交互式代码示例。

---

### 训练/评估期间的指标记录

### 训练/评估期间的指标记录

尽管 TensorBoard 提供丰富的可视化功能，但有效监控的根本在于训练和评估代码中的系统性记录。仅仅运行循环是不够的；你需要记录主要的性能数据，以了解训练是如何进展的，并在问题出现时进行诊断。这里说明如何在 PyTorch 训练和评估例程中直接实现基础的指标记 (token)录。

### 为什么要记录指标？

记录指标有几个重要目的：

1. **性能追踪：** 观察损失和准确度（或其他相关指标）在不同训练轮次中的趋势，以判断模型是否有效学习。
2. **调试：** 及早发现潜在问题。损失是否在下降？是否停滞不前？验证性能是否在改善或变差？记录指标中的异常情况常常指向一些根本问题，比如之前讨论过的学习率问题或过拟合 (overfitting)。
3. **比较：** 记录的指标允许在不同模型架构、超参数 (parameter) (hyperparameter)或训练运行之间进行客观比较。
4. **可视化的根本：** 像 TensorBoard 这样的工具依赖这些记录的值来生成图表和仪表板。

### 要记录的指标

具体指标取决于你的任务，但常用指标包括：

- **损失：** 这是你的优化器试图最小化的值。你几乎总是应该记录损失。追踪训练损失（在训练循环中基于训练数据计算）和验证损失（在评估循环中基于单独的验证数据集计算）很重要。比较这两者对于识别过拟合 (overfitting)非常重要。
- **准确度：** 对于分类任务，准确度（正确分类样本的比例）是一个标准且易于理解的指标。
- **其他任务特定指标：** 根据问题类型，你可能记录精确度、召回率、F1分数（用于分类），平均绝对误差（MAE）或均方根误差（RMSE）（用于回归），交并比（IoU）（用于分割）等。

### 在训练循环中实现记录

在训练期间，你通常希望追踪每个训练轮次中的平均损失和准确度。为每个批次记录指标可能产生噪音，并且对整体趋势的指示性较差，尽管有时它对调试不稳定性有用。

下面是修改标准训练轮次函数以包含记录的方法：

```python
import torch

def train_one_epoch(model, train_dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    print(f"Training Epoch: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    return epoch_loss, epoch_acc
```

记录实现中的要点：

- 在轮次开始时初始化累加器（`running_loss`、`correct_predictions`、`total_samples`）。
- 在批次循环内部，计算损失和预测后，更新累加器。
  - 使用 `loss.item()` 获取当前批次损失张量的标量值，以防止计算图被保留。如果你想在最后平均之前得到总损失，则乘以 `inputs.size(0)`（批次大小）；否则，你可以平均批次损失，但如果最后一个批次较小，按批次大小加权会更准确。
  - 计算批次的准确度（或其他指标），并添加到累加总数中。在这里也使用 `.item()`。
- 循环结束后，通过将累加总数除以样本数量，计算整个轮次的平均损失和准确度。
- 打印或存储这些轮次级别的指标。

### 在评估循环中实现记录

在评估循环中进行记录是类似的，但存在重要区别：

- 它在 `torch.no_grad()` 下运行，以禁用梯度计算。
- 模型应处于评估模式（`model.eval()`），以禁用 dropout 并使用训练期间学习到的批归一化 (normalization)统计量。
- 没有反向传播 (backpropagation)或优化器步骤。

```python
import torch

def evaluate_model(model, val_dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    print(f"Validation: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc
```

### 存储和使用记录的指标

将指标打印到控制台对于即时反馈很有用。为了更系统的分析或可视化，你会希望存储它们。简单的 Python 列表或字典效果很好：

```python

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for epoch in range(num_epochs):
    print(f"--- Epoch {epoch+1}/{num_epochs} ---")
    train_loss, train_acc = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device)
    val_loss, val_acc = evaluate_model(model, val_dataloader, loss_fn, device)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

print("Training finished.")
```

这种结构使你能够收集整个训练过程中的性能数据。这些存储的列表（`train_losses`、`val_losses`等）正是你可以输入到 Matplotlib 等绘图库或传递给 TensorBoard `SummaryWriter`（如前一节所述）以创建如下可视化图表的内容。

2468100.40.60.811.21.41.61.8训练损失验证损失

> 训练和验证损失曲线在10个轮次上绘制。观察这些趋势有助于诊断过拟合 (overfitting)（验证损失增加而训练损失减少）或欠拟合 (underfitting)（两种损失都保持高位）。

通过在训练和评估期间持续记录指标，你对模型的行为有很好的了解，从而能够就超参数 (parameter) (hyperparameter)调整、模型调整和调试策略做出明智的决定。

获取即时帮助、个性化解释和交互式代码示例。

---

### 在 PyTorch 中使用 Python 调试器 (pdb)

### 在 PyTorch 中使用 Python 调试器 (pdb)

尽管像 TensorBoard 这样的工具可以帮助监视训练趋势，并且可视化网络图可以提供架构理解，但有时你需要在程序执行到某个特定点时，仔细检查其确切状态。形状不匹配可能在模型的 `forward` 传递中发生，梯度可能意外地变成 `NaN`，或者张量值可能无故发散。对于这些情况，使用调试器逐行执行代码通常是找出根本原因最直接的方法。

Python 的内置调试器 `pdb` 是一个强大的文本工具，它能与 PyTorch 代码一同使用。它允许你暂停执行、检查变量（包括张量）、逐行执行代码，并在问题发生时准确了解程序的流程。

### 启用调试器

使用 `pdb` 启动调试会话最常见的方法是，在你希望执行暂停的位置，直接将以下两行代码插入到你的 Python 脚本中：

```python
import pdb
pdb.set_trace()
```

当 Python 解释器遇到 `pdb.set_trace()` 时，它会停止执行，并将你带入终端中的 `pdb` 交互式控制台。`(Pdb)` 提示符表示你当前处于调试器中。

你应该把 `pdb.set_trace()` 放在哪里？

- **在可能出错的代码前：** 如果某行代码（例如，矩阵乘法、损失计算）引发错误，请将 `pdb.set_trace()` 放在紧邻该行之前。这能让你检查该操作的输入。
- **在模型的 `forward` 方法中：** 为了了解数据在经过层时如何变化，请将跟踪点放在 `forward` 方法内。你可以逐层执行应用，并检查张量的形状和值。
- **在训练循环中：** 为了查看特定迭代或 epoch 的状态，请将其放在循环内。你可以将其包装在条件语句中（例如，`if batch_idx == problematic_index: import pdb; pdb.set_trace()`）。
- **在 `loss.backward()` 之后：** 为了在梯度计算完毕但优化器步骤执行之前检查梯度，请将跟踪点放在 `backward()` 调用之后。

此外，你也可以从命令行在 `pdb` 的控制下启动整个脚本，这会在脚本的第一行启动调试器：

```bash
python -m pdb your_pytorch_script.py
```

这对于诊断脚本执行早期发生的问题很有用，例如导入错误或设置问题。

### PDB 基本命令

一旦你在 `(Pdb)` 提示符下，就可以使用各种命令来控制执行和检查状态。下面是一些最常用的命令：

- `n` (next)：执行当前行，并在当前函数的下一行停止。如果当前行是一个函数调用，`n` 会执行整个函数并在它返回*后*停止。
- `s` (step)：与 `n` 类似，但如果当前行是一个函数调用，`s` 会*进入*函数，并在其第一行停止。
- `c` (continue)：恢复正常执行，直到遇到下一个断点（或 `pdb.set_trace()` 调用），或者直到脚本结束或出错。
- `l` (list)：显示当前执行行周围的源代码。使用 `l .` 再次列出以当前行为中心的源代码。
- `p <expression>` (print)：在当前上下文 (context)中评估 `<expression>` 并打印其值。这可以说是调试 PyTorch 最重要的命令。你可以检查张量、变量、模型参数 (parameter)等。
  - `p my_tensor.shape`
  - `p my_tensor.dtype`
  - `p my_tensor.device`
  - `p my_tensor` (打印张量本身；可能很大)
  - `p model.layer1.weight.grad` (在 `backward()` 之后)
  - `p loss.item()`
- `a` (args)：打印当前函数的参数列表。
- `r` (return)：继续执行直到当前函数返回。
- `b <line_number>` (breakpoint)：在当前文件的特定 `<line_number>` 处设置断点。当执行到达该行时会暂停。你也可以在其他文件 (`b path/to/file.py:<line_number>`) 或方法上 (`b self.my_method`) 指定断点。
- `cl` 或 `clear`：清除所有断点。`cl <bp_number>` 清除特定断点。
- `q` (quit)：退出调试器并立即终止脚本。
- `h` (help)：显示可用命令列表。`h <command>` 提供特定命令的帮助。

### 使用 PDB 调试 PyTorch 代码：示例

让我们看看 `pdb` 如何帮助处理常见的 PyTorch 调试情况。

**情况 1：调试模型中的形状不匹配问题**

假设你有一个简单模型，并且在前向传播时遇到了形状不匹配错误。

```python
import torch
import torch.nn as nn
import pdb

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.activation = nn.ReLU()

        self.layer2 = nn.Linear(25, 5)

    def forward(self, x):
        print(f"Initial shape: {x.shape}")
        x = self.layer1(x)
        print(f"After layer1: {x.shape}")
        x = self.activation(x)
        print(f"After activation: {x.shape}")

        pdb.set_trace()

        x = self.layer2(x)
        print(f"After layer2: {x.shape}")
        return x

net = SimpleNet()

input_tensor = torch.randn(32, 10)
output = net(input_tensor)
```

当你运行这段代码时，它会打印形状，然后在 `pdb.set_trace()` 处停止。

```text
Initial shape: torch.Size([32, 10])
After layer1: torch.Size([32, 20])
After activation: torch.Size([32, 20])
-> x = self.layer2(x)
(Pdb)
```

在 `(Pdb)` 提示符下，你可以检查：

- `p x.shape`：这将输出 `torch.Size([32, 20])`。
- `p self.layer2`：这将显示定义 `Linear(in_features=25, out_features=5, bias=True)`。
- `p self.layer2.in_features`：这将输出 `25`。

通过比较输入形状 `[32, 20]`（特别是特征维度 `20`）与 `layer2.in_features`（`25`），不匹配之处变得显而易见。然后你可以退出 (`q`) 并修正 `nn.Linear` 的定义。

**情况 2：检查梯度**

假设你的损失没有下降，并且你怀疑出现了梯度消失或梯度爆炸。你可以在反向传播 (backpropagation)后检查它们。

```python

outputs = model(inputs)
loss = loss_fn(outputs, targets)

optimizer.zero_grad()
loss.backward()

import pdb
pdb.set_trace()
```

当执行暂停时，你可以检查特定参数 (parameter)的梯度：

- `p model.some_layer.weight.grad`：检查 `some_layer` 权重 (weight)部分的梯度张量。查看是否存在 `NaN` 值、非常大的值（爆炸）或非常小的值（消失）。
- `p model.some_layer.weight.grad.abs().mean()`：计算梯度的平均绝对值，以了解其大小。
- `p loss.item()`：提醒自己当前的损失值。

### 有效使用 PDB 的建议

- **有针对性：** 将 `pdb.set_trace()` 放置在离你怀疑问题所在位置尽可能近的地方。
- **使用 Print (`p`)：** 广泛运用 `p` 命令来检查张量形状、数据类型、设备位置和实际值。
- **谨慎执行：** 使用 `n`（next）逐行移动。只有当你需要检查你编写的函数调用时才使用 `s`（step）（检查 PyTorch 的内部函数可能会很冗长）。
- **移除跟踪：** 在最终确定代码之前，请记得移除或注释掉 `import pdb; pdb.set_trace()` 调用，尤其是在提交到版本控制或部署之前。
- **考虑长时间运行的替代方案：** `pdb` 会停止执行。对于仅在训练数小时后才出现的问题进行调试，交互式调试可能不切实际。在这种情况下，定期记录详细信息（张量形状、损失值、梯度范数）可能更适合，或许可以结合条件断点或断言检查。

有效使用 `pdb` 需要一些练习，但它是理解 PyTorch 代码详细的、逐步执行过程以及解决许多从堆栈跟踪或高级指标中不明显问题的必不可少的工具。

获取即时帮助、个性化解释和交互式代码示例。

---

### 实践：调试与可视化

### 实践：调试与可视化

动手练习旨在巩固应用常见调试难点和可视化工具的技能。这些练习侧重于识别错误、检查模型行为，并使用 TensorBoard 和标准调试方法监控训练进度。涵盖的场景包括形状不匹配、设备放置错误和设置可视化。

### 练习 1：修正形状不匹配

形状不匹配是构建或修改神经网络 (neural network)时常见的错误。请看下面这个简单模型，它旨在处理 28x28 灰度图像（如 MNIST），并将其展平为 784 元素的向量 (vector)：

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)

        self.layer3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)

        x = self.layer3(x)
        return x

dummy_input = torch.randn(4, 784)
model = SimpleMLP()

try:
    output = model(dummy_input)
    print("模型运行成功！")
except RuntimeError as e:
    print(f"捕获到错误：{e}")
```

1. **运行代码：** 执行上面的代码片段。你会遇到一个 `RuntimeError`。仔细查看错误信息。它通常指向尺寸不匹配，常常会提到特定层的预期输入尺寸和实际输入尺寸（在本例中是 `mat1 and mat2 shapes cannot be multiplied`）。
2. **诊断：** 错误发生在输入 `x` 到达 `self.layer3` 时。前一个层 `self.layer2` 输出一个形状为 `(batch_size, 64)` 的张量。然而，`self.layer3` 被定义为 `nn.Linear(100, 10)`，它预期输入有 100 个特征。这种不匹配导致了错误。

   - 你可以在报错行之前插入打印语句来确认形状：

     ```python

     print("layer3 之前的形状：", x.shape)
     x = self.layer3(x)
     ```
3. **修正代码：** 修改 `__init__` 方法中 `self.layer3` 的定义，以接受正确数量的输入特征（即 64，`self.layer2` 的输出大小）。

   ```python

   self.layer3 = nn.Linear(64, 10)
   ```
4. **验证：** 使用修正后的层定义重新运行脚本。前向传播现在应该能顺利完成，没有错误。

### 练习 2：修正设备放置

使用 GPU 时，很重要的一点是模型和数据都在同一个设备上。我们来模拟一个错误场景，其中模型被移到 GPU，但输入张量仍留在 CPU 上。

```python
import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("正在使用 GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("正在使用 CPU")

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

model = SimpleNet().to(device)
print(f"模型参数位于：{next(model.parameters()).device}")

input_data = torch.randn(8, 10)
print(f"输入数据位于：{input_data.device}")

try:
    output = model(input_data)
    print("前向传播成功！")
except RuntimeError as e:
    print(f"\n捕获到错误：{e}")
    print("\n提示：检查模型和输入数据是否在同一个设备上。")
```

1. **运行代码：** 如果你有支持 CUDA 的 GPU，运行这段代码会产生一个 `RuntimeError`。错误信息很可能会显示类似 `Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!` 的内容。
2. **诊断：** 打印语句确认模型位于 `cuda` 设备上（如果可用），而 `input_data` 位于 `cpu` 设备上。PyTorch 操作通常要求操作数位于同一个设备上。
3. **修正代码：** 在将 `input_data` 传递给模型之前，将其移动到模型所在的设备上。

   ```python

   input_data = input_data.to(device)
   print(f"输入数据已移至：{input_data.device}")

   output = model(input_data)
   print("数据移动后前向传播成功！")
   ```
4. **验证：** 重新运行修正后的脚本。前向传播应该能顺利执行，没有设备不匹配错误。记住这个原理也适用于训练循环中；从 `DataLoader` 获取的每个批次都需要被移动到相应的设备上。

### 练习 3：使用 TensorBoard 可视化训练

TensorBoard 为训练过程提供了宝贵的参考信息。让我们将其集成到一个简化训练循环中。我们将模拟训练数据并追踪一个模拟损失值。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time

writer = SummaryWriter('runs/simple_experiment')

model = nn.Linear(10, 2)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

inputs = torch.randn(100, 10)
targets = torch.randn(100, 2)

print("开始模拟训练...")
num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    simulated_loss = loss + torch.randn(1) * 0.1 + (num_epochs - epoch) / num_epochs

    simulated_loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:

        writer.add_scalar('Training/Loss', simulated_loss.item(), epoch)

        writer.add_histogram('Model/Weights', model.weight, epoch)
        writer.add_histogram('Model/Bias', model.bias, epoch)

        print(f'周期 [{epoch+1}/{num_epochs}]，模拟损失：{simulated_loss.item():.4f}')

    time.sleep(0.1)

writer.close()
print("模拟训练完成。TensorBoard 日志已保存到 'runs/simple_experiment'。")
print("在你的终端中运行 'tensorboard --logdir=runs' 来查看。")
```

1. **运行代码：** 执行 Python 脚本。它会打印周期进度并提及保存日志。
2. **启动 TensorBoard：** 打开你的终端或命令提示符，导航到 *包含* `runs` 文件夹的目录（而不是 `runs` 文件夹内部），并运行命令：

   ```bash
   tensorboard --logdir=runs
   ```
3. **在浏览器中查看：** TensorBoard 会输出一个 URL（通常是 `http://localhost:6006/`）。在你的网页浏览器中打开这个 URL。
4. **浏览：** 浏览 TensorBoard 界面。你应该能找到 `simple_experiment` 运行记录。在“标量”（Scalars）标签页下，你会看到“训练/损失”（Training/Loss）图表，它显示了我们模拟损失的递减趋势。在“直方图”（Histograms）或“分布”（Distributions）标签页下，你可以观察模型权重 (weight)和偏差的分布如何随周期变化（或者在这个简单模拟中变化不大）。如果你取消注释了 `add_graph` 那行代码，你还会发现模型架构的可视化图表在“图”（Graphs）标签页下。

> 下面这个图表显示了一个在 TensorBoard 中查看时，损失如何随周期递减的例子。
>
> 0102030400.511.5
>
> > 一个折线图，描绘了 50 个周期内模拟训练损失的递减情况，每 5 个周期记录一次。

### 练习 4：使用 Python 调试器 (pdb)

有时，打印语句不足以解决问题，你需要交互式地检查程序状态。Python 调试器（`pdb`）是一个强大的工具。让我们回顾练习 1 中的形状不匹配场景，并使用 `pdb`。

通过在顶部添加 `import pdb` 并在导致错误的行之前添加 `pdb.set_trace()`，修改练习 1 中 *原始的* 失败代码：

```python
import torch
import torch.nn as nn
import pdb

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)

        self.layer3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        print("即将进入 pdb...")
        pdb.set_trace()

        print("layer3 之前的形状：", x.shape)
        x = self.layer3(x)
        return x

dummy_input = torch.randn(4, 784)
model = SimpleMLP()
output = model(dummy_input)
```

1. **运行修改后的代码：** 执行脚本。当程序执行到 `pdb.set_trace()` 时，它会暂停，你会在终端中看到 `(Pdb)` 提示符。
2. **与 pdb 交互：**
   - 输入 `p x.shape`（打印 `x.shape`）然后按回车键。你会看到 `torch.Size([4, 64])`。
   - 输入 `p self.layer3` 然后按回车键。你会看到定义 `Linear(in_features=100, out_features=10, bias=True)`。
   - 比较输入形状（64 个特征）与层预期输入（100 个特征），可以清楚地看到不匹配。
   - 输入 `n`（下一步）然后按回车键。这将尝试执行下一行代码（`x = self.layer3(x)`），这会导致 `RuntimeError`，并可能退出调试器或在其中显示错误堆栈。
   - 或者，输入 `c`（继续）让程序继续运行直到下一个断点或出现错误。
   - 输入 `q`（退出）立即退出调试器并终止脚本。
3. **修正并删除：** 一旦你理解了问题，你可以输入 `q` 退出，然后像练习 1 中那样修正代码，并删除 `import pdb` 和 `pdb.set_trace()` 行。

这个练习为你在 PyTorch 项目中处理调试和监控任务打下了基础。记住，使用打印语句进行快速检查，`pdb` 进行交互式检查，以及 TensorBoard 用于可视化训练过程和模型结构。这些工具对于构建、训练和改进高效的深度学习 (deep learning)模型来说非常重要。

获取即时帮助、个性化解释和交互式代码示例。

---
