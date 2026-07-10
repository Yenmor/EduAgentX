# NumPy 和 Pandas 基本功

## Chapter 1 Intro Data Handling Python

### 什么是 NumPy 和 Pandas？

# 什么是 NumPy 和 Pandas？

NumPy 和 Pandas 是 Python 生态中用于处理数值数据和执行数据分析任务的基本库。它们是专门的工具集，在数据科学和人工智能的应用中，当需要处理数字集合、数据表格或时间序列信息时，它们会非常常用。

### NumPy：数值计算的核心

NumPy，是 Numerical Python 的简称，它是 Python 中许多科学计算构建的底层部分。它的主要贡献是 `ndarray` 对象，一个强大的 N 维数组。

NumPy 的数组有何不同之处？

1. **效率高：** NumPy 数组使用 C 语言实现，并针对数值运算进行了优化。在 NumPy 数组上逐元素执行的计算比在标准 Python 列表中执行的等效操作快很多。处理大型数据集时，这种速度很重要。
2. **内存占用：** NumPy 数组比 Python 列表更节省内存，特别是处理大量数值数据时，因为它们在内存中连续存储相同数据类型的元素。
3. **功能性：** NumPy 提供一整套高级数学函数，可直接对这些数组进行操作。这包括线性代数运算、傅里叶变换、随机数生成以及用于统计分析的工具。

本质上，如果您需要在数值数据块（如向量 (vector)、矩阵或更高维张量）上执行数学运算，NumPy 提供基本对象和函数来高效完成。它构建了许多其他数据分析和机器学习 (machine learning)库（包括 Pandas）赖以存在的支撑。

### Pandas：让数据分析和处理更便捷

NumPy 提供底层的数值支撑，而 Pandas 则提供更高级的数据结构和分析工具，这些工具旨在提高实用性和易用性，特别是处理 *表格* 数据，例如电子表格或 SQL 表。

Pandas 中两个主要的数据结构是：

1. **Series：** 一维带标签数组，类似于电子表格中的一列，或带有相关索引的单个数据向量 (vector)。它可以存储任何 NumPy 数据类型的数据。
2. **DataFrame：** 二维带标签数据结构，其列可以包含不同数据类型，非常类似于电子表格、SQL 表或 Series 对象的字典。这是最常用的 Pandas 对象。

Pandas 擅长处理以下方面：

- **数据处理：** 轻松读取各种文件格式（如 CSV、Excel、JSON、SQL 数据库）的数据，并将数据写回。
- **数据清洗：** 提供处理缺失数据（查找、填充或删除 `NaN` 值）、筛选行和转换数据的工具。
- **数据分析：** 提供选择数据子集（切片和切块）、根据条件对数据分组、执行计算和聚合（如求和、平均值、计数）以及合并或连接不同数据集的方法。
- **时间序列：** 包含处理时间戳数据的专门工具。

DataWorkflow

RawData

原始数据来源
(CSV, Excel, 数据库等)

Pandas

Pandas
(DataFrame, Series)
数据加载、清洗、
处理、合并

RawData->Pandas

 加载数据

NumPy

NumPy
(ndarray)
数值运算、
数组支撑

Pandas->NumPy

 用于底层操作

Analysis

分析 / AI 模型
可视化、统计、
机器学习

Pandas->Analysis

 准备数据以供

NumPy->Pandas

 提供数组结构

NumPy->Analysis

 直接数值输入

> 这是 Pandas 和 NumPy 如何融入典型数据工作流程的简化视图。Pandas 处理更高级的数据结构和输入/输出，通常在内部依赖 NumPy 进行高效的数值计算。

"总而言之，NumPy 提供优化的数组对象和数学运算机制，而 Pandas 则在此基础上提供灵活、易用的数据结构（Series 和 DataFrame）以及一套丰富的功能，用于加载、清洗、转换、合并和分析数据。在 Python 中进行以数据为中心的项目时，您几乎总会同时使用它们。本课程将指导您掌握有效使用这两者的实用技能。"

获取即时帮助、个性化解释和交互式代码示例。

---

### 对人工智能和数据科学的重要性

# 对人工智能和数据科学的重要性

人工智能和数据科学应用本质上由数据驱动。无论是训练机器学习 (machine learning)模型识别图像、分析客户行为，还是处理传感器读数，高效处理和操作数据的能力都十分必要。然而，原始数据很少以可直接用于分析或模型训练的格式出现。它通常需要大量的清理、转换和结构化工作。

Python及其丰富的库生态系统已成为人工智能和数据科学应用中的主导语言。NumPy和Pandas在Python数据处理中是实现高效数据操作和分析的必要工具。

### NumPy的高效数值计算

从本质上讲，数据科学很大一部分涉及数值计算，通常处理大型数据集。标准的Python列表和循环虽然灵活，但在处理大量数字进行数学运算时效率不高。

NumPy（Numerical Python）直接解决了这一难题。它提供：

1. **`ndarray` 对象：** 一种强大的N维数组对象，比标准Python序列在内存效率上更高，且数值运算速度快得多。操作通过优化过的预编译C代码实现。
2. **向量 (vector)化操作：** NumPy允许您对整个数组进行一次性操作，无需显式Python循环。例如，两个NumPy数组相加会进行逐元素相加。这种“向量化 (quantization)”极大地加快了计算速度。
3. **数学函数：** 它提供一个广泛的库，包含作用于这些数组的高级数学函数，涵盖线性代数、傅里叶变换、随机数生成等。

许多机器学习 (machine learning)算法要求输入数据为数值数组形式。NumPy提供了标准格式以及有效操作这些数组的工具，使其成为Python中科学计算和人工智能不可或缺的组成部分。

### Pandas的灵活数据结构与数据分析

“尽管NumPy擅长处理原始数值数组，但数据通常具有更多结构。我们可能拥有包含数字、文本、日期和类别信息的混合数据集，它们通常以带有有意义的行和列标签的表格形式组织（例如电子表格或数据库中的数据）。”

Pandas基于NumPy构建，并提供专门为此类表格和异构数据设计的更高级数据结构和分析工具：

1. **DataFrame和Series：** Pandas引入了两种主要数据结构：`DataFrame`（一个二维带标签数据结构，如同一个表格）和`Series`（一维带标签数组，如同一个单列）。这些结构允许您通过行（索引）和列的标签直观地处理数据。
2. **数据处理能力：** Pandas使常见数据任务变得简单明了。这包括从各种文件格式（CSV、Excel、SQL数据库）读取数据、处理缺失值、根据条件筛选和选择数据、重塑数据、合并多个数据集以及执行分组分析。
3. **集成性：** Pandas与NumPy以及其他数据科学库，如Matplotlib（用于绘图）和Scikit-learn（用于机器学习 (machine learning)），顺利集成。使用Pandas准备和清理的数据通常可以直接输入机器学习模型。

### 典型工作流程

在典型的人工智能或数据科学项目中，NumPy和Pandas贯穿于初期阶段的使用：

1. **加载：** Pandas用于将来自各种来源的数据加载到DataFrames中。
2. **检查与清理：** 这两个库都有助于检查数据。Pandas尤其适用于查找和处理缺失值、修正数据类型以及发现不一致之处。
3. **转换与特征工程：** 数据通常需要重塑、聚合或组合。可能需要从现有特征中创建新特征。Pandas为这些任务提供了强大的工具（如`groupby`、`merge`、`apply`）。NumPy函数可用于数值转换。
4. **模型准备：** 对于许多机器学习 (machine learning)模型，Pandas DataFrames中的结构化数据会被转换为NumPy数值数组，这是Scikit-learn等库所期望的格式。

如果没有NumPy高效的数值数组和Pandas灵活的数据结构及操作工具，在Python中为人工智能和数据科学任务准备数据将会显著更复杂且性能更低。因此，掌握这些库是使用Python进入这些领域的重要一步。它们提供了整理数据所需的必备工具包。

获取即时帮助、个性化解释和交互式代码示例。

---

### 环境配置

# 环境配置

在开始使用 NumPy 和 Pandas 之前，你需要在电脑上安装它们。这些库默认不内置于 Python，因此你需要添加它们。这个过程需要使用名为包管理器的工具，它们能帮助下载和安装我们需要的这类软件库。

有两种主要方式可以在 Python 中配置你的数据科学工作环境：使用 Anaconda 分发版或使用 Python 的标准包安装器 `pip`。我们建议初学者使用 Anaconda，因为它能大大简化这个过程。

### 方案一：安装 Anaconda（推荐）

Anaconda 是一个免费、开源的 Python 和 R 分发版，专门为科学计算和数据科学而设计。它捆绑了 Python、许多必需的库（包括 NumPy、Pandas 和 Jupyter Notebooks），以及它自己的包管理器 `conda`。

**为什么选择 Anaconda？**

- **便捷性：** 它能一次性安装 Python 和数百个流行的数据科学包。NumPy 和 Pandas 通常已包含在内，因此你甚至可能无需单独安装它们。
- **环境管理：** `conda` 使得为不同项目创建隔离环境变得容易，从而避免库版本之间的冲突。
- **跨平台：** 它可在 Windows、macOS 和 Linux 上运行。

**步骤：**

1. **下载：** 访问 Anaconda 官方分发网站 (<https://www.anaconda.com/products/distribution>)，下载适用于你操作系统的安装程序（Windows、macOS 或 Linux）。选择 Python 3.x 版本（通常建议选择最新的稳定版）。
2. **安装：** 运行下载的安装程序。按照屏幕上的说明进行操作。除非有特殊原因，我们建议接受默认设置。安装过程中有一个重要选择（尤其是在 Windows 上），即是否将 Anaconda 添加到系统 PATH。安装程序通常不建议这样做，而是建议你使用“Anaconda Prompt”（在 Windows 上）或常规终端（在 macOS/Linux 上），这是一个好的建议。
3. **验证（可选）：** 安装完成后，你可以打开 Anaconda Prompt (Windows) 或你的终端 (macOS/Linux)，并检查 `conda` 是否可用：

   ```bash
   conda --version
   ```

   你应该会看到 `conda` 的版本号被打印出来。
4. **检查/安装 NumPy 和 Pandas：** Anaconda 通常包含 NumPy 和 Pandas。你可以通过输入以下命令来检查：

   ```bash
   conda list numpy pandas
   ```

   如果它们已列出，你就可以开始了！如果未列出，或者你想确保拥有与该分发版兼容的最新版本，你可以安装或更新它们：

   ```bash

   conda install numpy pandas jupyterlab

   conda update numpy pandas jupyterlab
   ```

   我们在这里包含 `jupyterlab` 是因为它提供了我们即将使用的 Jupyter Notebook 环境。

### 方案二：使用 pip 和虚拟环境

如果你的系统上已经安装了 Python（从 [python.org](https://www.python.org) 下载或通过其他方式安装），并且你不想使用 Anaconda，那么你可以使用 `pip`，它是 Python 默认的包安装器。

**为什么选择 pip？**

- **标准：** 它是标准的 Python 包管理器。
- **精简：** 如果你不需要 Anaconda 提供的所有额外功能，它能提供一个更轻量级的设置。

**最佳实践：虚拟环境**

使用 `pip` 时，强烈建议使用虚拟环境。虚拟环境是一个隔离的目录，其中包含特定的 Python 版本及其自己安装的库集。这可以防止不同项目所需的包之间发生冲突。Python 通过 `venv` 模块内置支持此功能。

**步骤：**

1. **确保 Python 和 pip 已安装：** 首先，请确保你已安装 Python 3。打开你的终端或命令提示符并输入：

   ```bash
   python --version

   pip --version
   ```

   如果这些命令有效并显示版本，则说明你已准备就绪。如果无效，你需要先从 [python.org](https://www.python.org) 安装 Python。`pip` 通常包含在 Python 3.4 及更高版本中。
2. **创建虚拟环境：** 在终端中导航到你的项目目录（或创建一个），然后运行：

   ```bash

   python -m venv myenv
   ```

   这将创建一个名为 `myenv` 的目录，其中包含 Python 安装文件。
3. **激活环境：** 在安装包之前，你需要激活环境：

   - **在 macOS 和 Linux 上：**

     ```bash
     source myenv/bin/activate
     ```
   - **在 Windows (命令提示符) 上：**

     ```bash
     myenv\Scripts\activate.bat
     ```
   - **在 Windows (PowerShell) 上：**

     ```bash
     myenv\Scripts\Activate.ps1
     ```

     （你可能需要调整执行策略：`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`）

   你的终端提示符应该会改变，以表明环境（本例中为 `myenv`）已激活。
4. **安装库：** 现在，使用 `pip` 安装 NumPy、Pandas 和 JupyterLab：

   ```bash
   pip install numpy pandas jupyterlab
   ```

   `pip` 会将库下载并安装到你当前激活的虚拟环境中。
5. **停用（完成后）：** 当你完成项目工作时，只需输入以下命令即可停用环境：

   ```bash
   deactivate
   ```

### 验证你的安装

无论你选择了哪种方法，验证库是否正确安装并且可以在 Python 中导入都是一个好习惯。

1. **启动 Python 解释器：** 打开你的终端（或 Anaconda Prompt）。如果你使用了虚拟环境，请确保它已激活。输入 `python` 并按回车键。你应该会看到 Python 提示符 (`>>>`)。
2. **导入库：** 逐一输入以下命令：

   ```python
   import numpy as np
   import pandas as pd

   print(f"NumPy version: {np.__version__}")
   print(f"Pandas version: {pd.__version__}")
   ```
3. **检查输出：** 如果命令执行没有错误，并打印出 NumPy 和 Pandas 的版本号，那么你的安装就成功了！
4. **退出解释器：** 输入 `exit()` 并按回车键，退出 Python 解释器。

你的环境现在已配置完成，你已拥有基本的工具。在下一节中，我们将在 Jupyter Notebook（许多数据科学家偏爱的交互式环境）中运行一些基本代码示例，开始使用这些库。

获取即时帮助、个性化解释和交互式代码示例。

---

### 运行你的第一个代码片段

# 运行你的第一个代码片段

编写并执行你的第一批 NumPy 和 Pandas 代码是一个重要步骤。这有助于确认库是否正确安装，并让你初步了解它们如何运作。

### 验证你的安装

使用任何 Python 库的第一步是将其 `import` 到你当前的脚本或 Notebook 会话中。这使得库的函数和对象可供你使用。对于 NumPy 和 Pandas，有一些标准的社区导入惯例：

- NumPy 通常以别名 `np` 导入。
- Pandas 通常以别名 `pd` 导入。

使用这些别名让你的代码更短、更易读，因为它们被广泛认可。

打开你的 Jupyter Notebook（或 Python 解释器）并在一个单元格中输入以下代码：

```python

import numpy as np

import pandas as pd

print("NumPy 和 Pandas 导入成功！")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
```

要在 Jupyter 中运行此代码，请按 `Shift + Enter`。如果安装成功，你将在单元格下方看到打印的确认信息，以及 NumPy 和 Pandas 的已安装版本。如果你遇到 `ModuleNotFoundError` 这样的错误信息，请重新检查上一节（“设置你的环境”）中的安装步骤。

### 你的第一个 NumPy 数组

NumPy 的主要对象是 N 维数组，通常称为 `ndarray`。让我们从一个标准的 Python 列表创建一个简单的一维数组。

```python

my_list = [1, 2, 3, 4, 5]

np_array = np.array(my_list)

print("我的第一个 NumPy 数组：")
print(np_array)

print("对象的类型：")
print(type(np_array))
```

运行此单元格将输出：

```
我的第一个 NumPy 数组：
[1 2 3 4 5]
对象的类型：
<class 'numpy.ndarray'>
```

注意到输出 `[1 2 3 4 5]` 与 Python 列表 `[1, 2, 3, 4, 5]` 略有不同。这表明它是一个 NumPy 数组。`type()` 函数确认 `np_array` 确实是 `numpy.ndarray` 类的一个实例。

NumPy 数组支持高效的逐元素操作。让我们尝试一个简单的：将每个元素乘以 2。

```python

doubled_array = np_array * 2

print("乘法运算后的数组：")
print(doubled_array)
```

输出：

```
乘法运算后的数组：
[ 2  4  6  8 10]
```

对于标准的 Python 列表，你需要使用循环或列表推导来完成此操作。NumPy 在后台更高效地处理了这一点。

### 你的第一个 Pandas 对象：Series 和 DataFrame

Pandas 提供了两种主要数据结构：`Series`（一维）和 `DataFrame`（二维）。

让我们创建一个 `Series`。Series 类似于一维 NumPy 数组，但带有关联的索引。

```python

data_series = [10, 20, 30, 40, 50]
pd_series = pd.Series(data_series)

print("我的第一个 Pandas Series：")
print(pd_series)

print("\n对象的类型：")
print(type(pd_series))
```

运行此单元格将输出：

```
我的第一个 Pandas Series：
0    10
1    20
2    30
3    40
4    50
dtype: int64

对象的类型：
<class 'pandas.core.series.Series'>
```

注意到输出左侧包含值（10、20 等）和索引（0、1、2 等）。`dtype: int64` 表示 Series 中存储值的数据类型。

现在，让我们创建一个 `DataFrame`。DataFrame 是一种二维的、表格状的结构，其中列可以具有不同的数据类型。一种常见的方法是从 Python 字典创建它，其中键成为列名，值（列表或数组）成为列数据。

```python

data_dict = {
    'StudentID': [101, 102, 103, 104],
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Score': [85, 92, 78, 88]
}

df = pd.DataFrame(data_dict)

print("我的第一个 Pandas DataFrame：")
print(df)

print("\n对象的类型：")
print(type(df))
```

运行此单元格会生成：

```
我的第一个 Pandas DataFrame：
   StudentID     Name  Score
0        101    Alice     85
1        102      Bob     92
2        103  Charlie     78
3        104    David     88

对象的类型：
<class 'pandas.core.frame.DataFrame'>
```

此输出清晰地类似于一个带有标签列（'StudentID'、'Name'、'Score'）的表格和一个自动生成的行索引（0、1、2、3）。

执行这些简单的代码片段确认你的环境已就绪，并让你初步了解如何在 NumPy 和 Pandas 中创建和操作基本数据结构。在接下来的章节中，我们将在此基础上，更详细地研究这些库的功能。

获取即时帮助、个性化解释和交互式代码示例。

---

### Jupyter Notebook 简介

# Jupyter Notebook 简介

Jupyter Notebook 是您在交互式数据分析中可能会最常使用的工具。可以将其视为一个数字实验室笔记本，您可以在其中编写和运行 Python 代码，添加解释性文本，并显示表格和图表等结果，所有这些都可以在您的网页浏览器中完成。

### 什么是 Jupyter Notebook？

A Jupyter Notebook 是一个开源的 Web 应用程序，允许您创建和分享包含实时代码、公式、可视化内容和说明性文本的文档。“Jupyter”这个名字来源于它支持的核心编程语言：Julia、Python 和 R。笔记本在数据科学中被广泛使用，因为它们让代码试验变得简单，可以立即查看结果，并在整个过程中记录您的思考过程。

笔记本文档本身（`.ipynb` 文件）会存储所有内容：您的代码、生成的输出，以及您添加的任何文本或图像。这使得它们独立且易于分享。

### 笔记本界面：单元格和内核

Jupyter Notebook 的主要工作组件是**单元格**和**内核**。

- **单元格：** 它们是笔记本的构建块。主要有两种类型：

  - **代码单元格：** 这些包含您打算执行的 Python 代码。当您运行一个代码单元格时，代码会发送到内核执行，并且输出（如计算结果、表格或图表）会直接显示在该单元格下方。
  - **Markdown 单元格：** 这些包含使用 Markdown（一种简单的标记 (token)语言）格式化的文本。您使用这些单元格来编写说明、添加标题、创建列表、插入图像，并通常用于构建您的笔记本，使其易于理解。
- **内核：** 这是在后台运行的计算引擎。当您运行一个代码单元格时，内核会执行该代码。它维护您的计算状态，这意味着在一个单元格中定义的变量在同一会话的后续单元格中都可用。在本课程中，您将使用 Python 内核。

G

clusterₙotebook

Jupyter Notebook (浏览器)

cell1

代码单元格
(例如：import numpy as np)

cell2

Markdown 单元格
(例如：## 数据加载)

kernel

Python 内核
(执行代码，管理变量)

cell1->kernel

 发送代码

cell3

代码单元格
(例如：data = np.array(...))

cell4

输出区域

cell3->cell4

cell3->kernel

 发送代码

kernel->cell4

 发送输出

> Jupyter Notebook 中的基本交互流程。单元格中的代码发送到内核执行，结果显示回笔记本中。

### 单元格操作

当您打开 Jupyter Notebook 时，您会看到一个主要由这些单元格组成的界面。以下是基本操作：

1. **选择单元格：** 只需点击它。选中的单元格会有一个边框（通常是蓝色或绿色，取决于模式）。
2. **运行单元格：** 最常见的方法是按下 `Shift + Enter`。这会运行当前单元格中的代码并自动选择其下方的下一个单元格。`Ctrl + Enter` 会运行选定的单元格但保持焦点在该单元格上。您也可以使用工具栏中的“运行”按钮。运行代码单元格的顺序很重要，因为后面的单元格可能依赖于前面定义的变量或导入。
3. **更改单元格类型：** 您可以使用工具栏中的下拉菜单或键盘快捷键（通常是 `Esc`，然后按 `M` 转换为 Markdown，或按 `Y` 转换为 Code）在代码和 Markdown 类型之间切换。
4. **添加单元格：** 使用工具栏中的“+”按钮或键盘快捷键（例如 `Esc`，然后按 `A` 在上方插入，或按 `B` 在下方插入）。
5. **编辑单元格：** 双击 Markdown 单元格以编辑其文本。对于代码单元格，只需点击内部即可开始输入。
6. **删除单元格：** 选择单元格，然后使用工具栏中的“剪切”（剪刀）图标，或者按下 `Esc` 键，然后连续按两次 `D` 键。

### 代码单元格的实际应用

让我们看一个简单的例子。在一个代码单元格中，您可以输入：

```python
import numpy as np
import pandas as pd

print("NumPy and Pandas imported successfully!")
```

按下 `Shift + Enter` 会执行此操作。如果一切设置正确，您将看到输出：

```
NumPy and Pandas imported successfully!
```

现在，在*下一个*代码单元格中，您可以使用 `np` 和 `pd`，因为内核记住了之前执行中定义的它们：

```python
my_array = np.array([1, 2, 3, 4])
my_series = pd.Series(['a', 'b', 'c'])

print(my_array)
print(my_series)
```

运行此单元格会输出：

```
[1 2 3 4]
0    a
1    b
2    c
dtype: object
```

请注意，输出是如何直接出现在生成它的单元格下方的。

### 用于文档的 Markdown

Markdown 单元格允许您添加格式化文本。您可以创建结构并解释您的代码。以下是一些基本示例：

```markdown
# 这是一个一级标题

## 这是一个二级标题

这里有一些普通文本。您可以解释下一个代码单元格的作用。

使用星号或破折号创建项目符号列表：
* 项目 1
* 项目 2
  * 子项目

或者使用数字创建有序列表：
1. 第一步
2. 第二步

您可以使用反引号添加 `行内代码`。
```

当您运行 Markdown 单元格（`Shift + Enter`）时，文本将以指定的格式呈现。

### 为什么在数据分析中使用 Jupyter？

Jupyter Notebooks 特别适合您在本课程中将要进行的工作类型：

- **交互式分析：** 运行小段代码，即时查看结果，并根据输出调整方法。这对于理解数据非常有益。
- **集成环境：** 将您的代码、其输出（表格、图表）和您的笔记保存在一个文档中。
- **易于分享：** 与他人分享您的 `.ipynb` 文件，他们可以看到您的工作流程、结果和说明（他们需要安装 Jupyter 才能自己运行代码）。
- **可视化：** 像 Matplotlib 和 Seaborn（它们通常基于 NumPy 和 Pandas）这样的库可以直接与 Jupyter 集成，以便内联显示图表。

您通常会从终端（导航到您的工作目录后）通过输入 `jupyter notebook` 或 `jupyter lab` 来启动 Jupyter Notebook。Anaconda Navigator 也提供了图形界面启动方式。随着本课程的进展，您将获得更多使用笔记本处理 NumPy 和 Pandas 的实践经验。

获取即时帮助、个性化解释和交互式代码示例。

---

### 动手实践：设置与验证

# 动手实践：设置与验证

这个动手练习将引导你安装 NumPy 和 Pandas 库（如果尚未安装），并通过在 Jupyter Notebook 中运行一些基础代码来确认一切正常运行。

### 确认安装

根据你选择直接使用 Anaconda 还是 `pip`，安装步骤会略有不同。

#### 选项一：使用 Anaconda

Anaconda 简化了包管理。如果你按照“设置你的环境”部分的建议安装了 Anaconda，你很可能已经安装了 NumPy、Pandas 和 Jupyter。不过，我们还是来明确地确认或安装它们。

1. **打开 Anaconda Prompt（或 macOS/Linux 上的终端）：**

   - 在 Windows 上，在“开始”菜单中搜索“Anaconda Prompt”。
   - 在 macOS 或 Linux 上，打开你的标准终端应用程序。
2. **安装/更新库：** 确保你拥有最新版本是个好习惯。执行以下命令：

   ```bash
   conda install numpy pandas jupyterlab
   ```

   Conda 会检查这些包是否已安装，如果必要则更新它们，或者如果它们缺失则安装它们。它还会自动处理安装任何所需的依赖项。你可能会被提示确认安装计划；如果是，请输入 `y` 并按回车键。
3. **验证安装（可选）：** 你可以列出 Conda 管理的已安装包来检查：

   ```bash
   conda list numpy pandas
   ```

   此命令应显示 NumPy 和 Pandas 的已安装版本。

#### 选项二：使用 pip

如果你使用 `pip`（Python 的标准包安装器）直接管理你的 Python 环境，请按照以下步骤操作。

1. **打开终端或命令提示符：**

   - 在 Windows 上，打开命令提示符 (cmd) 或 PowerShell。
   - 在 macOS 或 Linux 上，打开你的终端。
2. **安装库：** 使用 `pip` 安装 NumPy、Pandas 和 JupyterLab（包含 Jupyter Notebook）：

   ```bash
   pip install numpy pandas jupyterlab
   ```

   *注意：* 根据你的系统配置，你可能需要使用 `pip3` 而不是 `pip`，特别是如果你同时安装了 Python 2 和 Python 3。在某些系统上，特别是 Linux 和 macOS，使用 `python -m pip install ...` 是一种更可靠的方式，可以确保你使用的是与你预期 Python 解释器关联的 `pip`。
3. **验证安装（可选）：** 你可以使用 `pip` 显示已安装包的详细信息：

   ```bash
   pip show numpy pandas
   ```

   如果 NumPy 和 Pandas 包成功安装，此命令将显示它们的信息。

### 启动 JupyterLab

库安装好后，让我们启动 JupyterLab，这是我们将在整个课程中使用的交互式环境。

1. **导航到你的项目目录（推荐）：** 打开你的终端或 Anaconda Prompt。使用 `cd`（更改目录）命令导航到你希望保存本课程笔记本的文件夹。例如：

   ```bash
   cd path/to/your/projects/essential-numpy-pandas
   ```

   将 `path/to/your/projects/essential-numpy-pandas` 替换为你计算机上的实际路径。在特定的项目文件夹中工作有助于使你的文件井井有条。
2. **启动 JupyterLab：** 输入以下命令并按回车键：

   ```bash
   jupyter lab
   ```

   此命令将启动 JupyterLab 服务器。你的默认网页浏览器应自动打开，显示 JupyterLab 界面。如果它没有自动打开，终端将提供一个 URL（通常以 `http://localhost:8888/lab` 开头），你可以复制并粘贴到浏览器的地址栏中。

保持终端窗口运行；关闭它将关闭 Jupyter 服务器。

### 创建并运行你的第一个笔记本

现在，让我们创建一个笔记本并运行一些代码来确认 NumPy 和 Pandas 已经准备就绪。

1. **创建新笔记本：** 在 JupyterLab 界面中（它在你的浏览器中打开），查找“启动器”选项卡。在“笔记本”下，点击“Python 3”内核图标（根据你的设置，它可能有一个稍微不同的名称，如“Python [conda env:base]”）。这将创建一个并打开一个新的、未命名的笔记本文件（`.ipynb`）。
2. **重命名笔记本（可选但推荐）：** 点击笔记本区域顶部的“Untitled.ipynb”名称，并将其重命名为有描述性的名称，例如 `01-Setup-Verification.ipynb`。
3. **输入并运行 NumPy 代码：** 在第一个代码单元格（旁边带有 `[ ]:` 的框）中，输入以下 Python 代码：

   ```python
   import numpy as np

   my_array = np.array([1, 2, 3, 4, 5])

   print("My first NumPy array:")
   print(my_array)

   print("Array shape:")
   print(my_array.shape)
   ```

   要运行此单元格中的代码，请点击单元格内部并按下 `Shift + Enter`。
4. **验证 NumPy 输出：** 在单元格下方，你应该看到输出：

   ```
   My first NumPy array:
   [1 2 3 4 5]
   Array shape:
   (5,)
   ```

   看到此输出确认 NumPy 已正确安装并正常运行。`import numpy as np` 行导入该库，通常为其赋予别名 `np`。然后我们创建了一个简单的 1 维数组并打印它，以及它的形状（沿着一个维度有 5 个元素）。
5. **输入并运行 Pandas 代码：** 输出下方应出现一个新的代码单元格。如果没有，点击笔记本工具栏中的 `+` 按钮。在这个新单元格中，输入以下代码：

   ```python
   import pandas as pd

   my_series = pd.Series({'a': 10, 'b': 20, 'c': 30})

   print("My first Pandas Series:")
   print(my_series)
   ```

   按下 `Shift + Enter` 运行此单元格。
6. **验证 Pandas 输出：** 你应该看到以下输出：

   ```
   My first Pandas Series:
   a    10
   b    20
   c    30
   dtype: int64
   ```

   此输出确认 Pandas 也已安装并正常运行。我们使用约定俗成的别名 `pd` 导入它，并从 Python 字典创建了一个基本的 Series（一维带标签数组）。

### 故障排除提示

如果你在安装或运行代码时遇到错误：

- **命令未找到：** 如果你的终端无法识别 `conda` 或 `pip`，这可能意味着 Anaconda 或 Python 在安装过程中没有添加到你系统的 PATH 环境变量中。请重新查看你操作系统的安装说明，或尝试再次运行安装，确保你选择将它添加到 PATH 的选项（如果可用且适合你的设置）。
- **ImportError：** 如果 Python 提示找不到模块（`No module named 'numpy'` 或 `No module named 'pandas'`），则安装可能失败，或者你可能正在使用与安装库时不同的 Python 环境运行笔记本。确保你在执行安装的相同环境（例如，相同的 Anaconda Prompt/终端会话）中运行 `jupyter lab`。请再次尝试安装命令。
- **检查版本：** 有时会出现兼容性问题。在笔记本单元格中导入它们后，你可以使用 `np.__version__` 和 `pd.__version__` 检查版本。

成功运行这些简单的代码片段意味着你的环境已正确配置 NumPy 和 Pandas，并且你熟悉在 Jupyter Notebook 中执行代码的基本知识。你现在可以继续学习后续章节，并开始使用这些功能强大的库了。记住保存你的笔记本（文件 -> 保存笔记本或 Ctrl+S/Cmd+S），你也可以通过回到运行 `jupyter lab` 的终端窗口并按两次 `Ctrl + C` 来关闭 Jupyter 服务器。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 2 Getting Started Numpy Arrays

### 了解 NumPy N 维数组

# 了解 NumPy N 维数组

NumPy 的核心是其主要数据结构：**N 维数组**，常称为 `ndarray`。可以将其视为专为数值数据设计的高效、灵活的容器。尽管标准的 Python 列表功能多用，但它们未针对数据科学和机器学习 (machine learning)中常见的大规模数值运算进行优化。NumPy 数组解决了这些限制。

`ndarray` 有何独特之处？

1. **同质数据类型：** 与 Python 列表不同，Python 列表可以存储不同类型的元素（例如，整数、字符串和浮点数都可以在一个列表中），而单个 NumPy 数组中的所有元素必须是*相同*的数据类型。这种同质性是 NumPy 性能的重要因素。了解每个元素（例如）是 64 位浮点数，使 NumPy 能够使用高度优化的底层 C 代码进行计算，从而避免了与 Python 动态类型相关的许多额外开销。
2. **固定大小：** 创建 NumPy 数组时，其大小通常是固定的。尽管 NumPy 提供了改变数组大小的函数，但这些操作通常涉及创建新数组和复制数据，而非原地调整原始数组大小。这种固定大小的特点有助于内存分配和访问的效率。
3. **多维性：** 顾名思义，`ndarray` 可以有多个维度。

   - **一维数组** 类似于列表或向量 (vector)。
   - **二维数组** 类似于带有行和列的表格或矩阵。
   - 数组可以有 3、4 甚至更多维度，允许表示复杂数据集，例如彩色图像（维度 1：高度，维度 2：宽度，维度 3：颜色通道）或矩阵序列。每个维度都称为一个**轴**。
4. **高效性：** 由于 `ndarray` 将数据存储在连续的内存块中并使用编译过的 C 代码执行操作，对其进行的数学运算明显快于使用循环在 Python 列表上执行的等效操作。这种能力，常称为**向量化 (quantization)**，允许您对数据执行批量操作而无需编写显式循环，从而产生简洁且更快的代码。

我们来直观地展示一维数组和二维数组之间的区别：

G

cluster₁d

一维数组 (向量)

cluster₂d

二维数组 (矩阵)

a0

值 0
(索引 0)

a1

值 1
(索引 1)

a0->a1

a2

值 2
(索引 2)

a1->a2

p00

p01

p10

p02

p11

p12

b00

值 (0,0)

b01

值 (0,1)

b10

值 (1,0)

b02

值 (0,2)

b11

值 (1,1)

b12

值 (1,2)

> 一个简单的一维数组（序列）和二维数组（网格）的可视化表示。

快速看一下 `ndarray` 对象在 Python 代码中是什么样子。我们很快将详细介绍创建方法。

```python

import numpy as np

python_list = [10, 20, 30, 40, 50]

numpy_array = np.array(python_list)

print("NumPy 数组:", numpy_array)
print("类型:", type(numpy_array))
```

运行此代码将输出：

```
NumPy Array: [10 20 30 40 50]
Type: <class 'numpy.ndarray'>
```

注意输出 `[10 20 30 40 50]` 看起来与列表相似，但没有逗号。这是 NumPy 一维数组的标准字符串表示形式。其类型证实我们现在正在使用 NumPy 的专用 `ndarray` 对象。

了解 `ndarray` 是运用 NumPy 能力进行高效数据处理和计算的第一步。随后的章节将引导您创建这些数组并更仔细地检查它们的属性。

获取即时帮助、个性化解释和交互式代码示例。

---

### 从 Python 列表创建数组

# 从 Python 列表创建数组

虽然 NumPy 提供了创建数组的专门函数（我们很快会看到），但一个非常常见的开始方法是将现有的 Python 数据结构，特别是列表，转换为 NumPy 数组。当你已经使用标准 Python 代码加载或生成了数据时，这通常是第一步。

用于此转换的主要函数是 `np.array()`。我们来看看它是如何工作的。

### 从列表创建一维数组

假设你有一个简单的 Python 数字列表：

```python
import numpy as np

python_list = [5, 10, 15, 20, 25]
print(f"原始 Python 列表: {python_list}")
print(f"类型: {type(python_list)}")

numpy_array = np.array(python_list)

print(f"转换后的 NumPy 数组: {numpy_array}")
print(f"类型: {type(numpy_array)}")
print(f"数组的数据类型 (dtype): {numpy_array.dtype}")
```

当你运行这段代码时，你会注意到以下几点：

1. `np.array()` 函数将 Python 列表作为输入。
2. 输出的 `numpy_array` 不再是标准的 Python 列表，而是 `numpy.ndarray` 类型的一个对象。这就是我们提到的 N 维数组。
3. NumPy 自动推断了元素的数据类型。由于所有元素都是整数，因此生成的数组的 `dtype` 为 `int64`（具体的整数类型，如 `int32` 或 `int64`，可能会因你的系统而略有不同）。

这种自动类型检测很方便。如果你的列表包含浮点数，NumPy 将创建一个浮点数组：

```python
float_list = [1.0, 2.5, 3.7, 4.2]
float_array = np.array(float_list)
print(f"浮点数组: {float_array}")
print(f"浮点数组 dtype: {float_array.dtype}")
```

ListToArray

list

Python 列表

[5, 10, 15, 20, 25]

array

NumPy 数组

5

10

15

20

25

(dtype=int64)

list->array

np.array()

> 使用 `np.array()` 将一维 Python 列表转换为 NumPy ndarray。

### 从嵌套列表创建多维数组

NumPy 数组可以具有多个维度。你可以通过将嵌套的 Python 列表（包含其他列表的列表）传递给 `np.array()` 来创建多维数组。

```python

nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(f"原始嵌套列表:\n{nested_list}")

numpy_2d_array = np.array(nested_list)

print(f"转换后的二维 NumPy 数组:\n{numpy_2d_array}")
print(f"数组形状: {numpy_2d_array.shape}")
print(f"数组维度: {numpy_2d_array.ndim}")
print(f"数组 dtype: {numpy_2d_array.dtype}")
```

这里，包含三个整数的三个列表组成的列表被转换为一个 3 行 3 列的二维数组（一个矩阵）。NumPy 会相应地排列数据。

NestedListToArray

list

Python 列表

[[1, 2, 3], [4, 5, 6], [7, 8, 9]]

array

NumPy 二维数组

1

2

3

4

5

6

7

8

9

shape=(3,3), dtype=int64

list->array

np.array()

> 将嵌套的 Python 列表转换为二维 NumPy ndarray。

为了让 NumPy 高效地创建标准的二维数组，内部列表必须具有相同的长度。如果它们不一致，NumPy 可能会创建一个 `dtype=object` 的数组，每个元素都是一个 Python 对象（如列表），这通常会丧失 NumPy 数组的性能优势。

```python

uneven_list = [[1, 2], [3, 4, 5]]

object_array = np.array(uneven_list, dtype=object)

print(f"不规则列表生成的数组:\n{object_array}")
print(f"数组 dtype: {object_array.dtype}")
```

### 指定数据类型 (`dtype`)

虽然 NumPy 的自动类型推断很有用，但有时你需要明确控制数组的数据类型。你可以使用 `np.array()` 函数中的 `dtype` 参数 (parameter)来指定所需的类型。这有助于：

- **内存管理：** 较小的数据类型（如 `float32` 而不是 `float64`，或 `int8` 而不是 `int64`）使用的内存更少。
- **精度：** 确保计算以足够的浮点精度执行。
- **兼容性：** 匹配其他库或函数所需的数据类型。

下面是如何从整数列表创建一个浮点数数组：

```python
integer_list = [1, 2, 3, 4]

float_array = np.array(integer_list, dtype=float)

print(f"原始整数列表: {integer_list}")
print(f"转换后的浮点数组: {float_array}")
print(f"数组 dtype: {float_array.dtype}")

int8_array = np.array(integer_list, dtype=np.int8)
print(f"转换后的 int8 数组: {int8_array}")
print(f"数组 dtype: {int8_array.dtype}")
```

常见的 NumPy 数据类型包括 `np.int8`、`np.int16`、`np.int32`、`np.int64`、`np.uint8`（无符号整数）、`np.float32`、`np.float64`、`np.complex64`、`np.complex128`、`np.bool_` 和 `np.object_`。选择正确的 `dtype` 是一个重要考量，尤其是在处理大型数据集时。

转换 Python 列表是开始使用 NumPy 数组的一种基本方法，它提供了一条从标准 Python 数据结构到 NumPy 强大数值计算能力的途径。在下一节中，我们将查看 NumPy 提供的直接创建数组的函数，无需预先存在 Python 列表。

获取即时帮助、个性化解释和交互式代码示例。

---

### 内置数组创建函数

# 内置数组创建函数

使用NumPy的专门函数生成数组通常更方便高效。这些函数允许您创建具有特定结构或初始值的数组，而无需先构建Python列表。以下是一些常用的函数。

### 使用 `arange` 创建序列

与Python的内置 `range` 函数类似，NumPy的 `arange` 函数在给定区间内创建一个包含等间距值的数组。然而，`range` 生成的是一个生成器，而 `arange` 则直接返回一个NumPy数组。

其基本语法是 `np.arange(start, stop, step)`，参数 (parameter)说明如下：

- `start`: 区间的起始值（包含）。如果未提供，默认为0。
- `stop`: 区间的结束值（不包含）。
- `step`: 值之间的间隔。默认为1。

```python
import numpy as np

arr1 = np.arange(5)
print(arr1)

arr2 = np.arange(2, 8)
print(arr2)

arr3 = np.arange(1, 10, 2)
print(arr3)
```

请注意，`arange` 和Python的 `range` 一样，结果中不包含 `stop` 值。此外，`arange` 可以使用浮点数作为步长，但由于潜在的浮点数精度问题，请谨慎使用。对于非整数步长，如果精确的点数更重要，通常更推荐使用 `linspace`（接下来会介绍）。

### 创建全零或全一数组

通常，您需要用占位值（通常是零或一）来初始化特定大小的数组。NumPy为此提供了 `zeros` 和 `ones` 函数。

- `np.zeros(shape, dtype=float)`: 创建一个填充零的数组。
- `np.ones(shape, dtype=float)`: 创建一个填充一的数组。

`shape` 参数 (parameter)是一个元组，用于指定数组的维度（例如，`(3,)` 表示一个大小为3的一维数组，`(2, 4)` 表示一个2行4列的二维数组）。`dtype` 参数是可选的，用于指定数据类型（默认为 `float64`）。

```python

zeros_arr_1d = np.zeros(4)
print(zeros_arr_1d)

ones_arr_2d_int = np.ones((2, 3), dtype=np.int64)
print(ones_arr_2d_int)

print(ones_arr_2d_int.dtype)
```

### 使用 `linspace` 创建等间距数组

有时，您需要在起始值和结束值之间创建一个包含特定数量等间距点的数组。`linspace` 在这种情况下很有用。

其语法是 `np.linspace(start, stop, num=50)`，参数 (parameter)如下：

- `start`: 序列的起始值（包含）。
- `stop`: 序列的结束值（默认为包含）。
- `num`: 要生成的样本数量。默认为50。

与 `arange` 不同，`linspace` 在数组中包含 `stop` 值。

```python

lin_arr1 = np.linspace(0, 1, 5)
print(lin_arr1)

lin_arr2 = np.linspace(0, 10, 11)
print(lin_arr2)

lin_arr3 = np.linspace(0, 1, 5, endpoint=False)
print(lin_arr3)
```

`linspace` 在生成绘图或模拟的坐标时特别方便。

### 使用 `eye` 创建单位矩阵

单位矩阵是一种方阵（行数等于列数），其主对角线（从左上到右下）上为一，其他地方为零。NumPy的 `eye` 函数可以创建这类矩阵。

其语法是 `np.eye(N, dtype=float)`，其中 `N` 是行数（和列数）。

```python

identity_matrix = np.eye(3)
print(identity_matrix)

identity_matrix_int = np.eye(4, dtype=int)
print(identity_matrix_int)
```

单位矩阵在线性代数运算中非常重要。

### 使用 `full` 创建填充特定值的数组

如果您需要一个给定形状、且完全填充除0或1之外的常数值的数组，可以使用 `np.full`。

其语法是 `np.full(shape, fill_value, dtype=None)`。

```python

full_arr = np.full((2, 4), 7)
print(full_arr)

pi_arr = np.full(3, np.pi)
print(pi_arr)
```

数据类型会从 `fill_value` 推断，除非使用 `dtype` 显式指定。

### 生成随机数组

NumPy还包含一个功能强大的子模块 `numpy.random`，用于创建从各种分布中抽取的随机数数组。以下是一些常见例子：

- `np.random.rand(d0, d1, ..., dn)`: 创建一个给定形状的数组，其中包含从 [0,1)[0, 1)[0,1) 均匀分布中抽取的随机样本。
- `np.random.randn(d0, d1, ..., dn)`: 创建一个给定形状的数组，其中包含从标准正态分布（均值为0，方差为1）中抽取的随机样本。
- `np.random.randint(low, high=None, size=None, dtype=int)`: 创建一个指定大小的数组，其中包含从 `low`（包含）到 `high`（不包含）的随机整数。

```python

rand_arr = np.random.rand(2, 3)
print(rand_arr)

randn_arr = np.random.randn(4)
print(randn_arr)

randint_arr = np.random.randint(10, 20, size=5)
print(randint_arr)
```

这些随机数函数对于模拟、统计建模以及机器学习 (machine learning)算法中的参数 (parameter)初始化都很重要。

这些内置函数提供了灵活高效的方式来创建NumPy数组，以应对各种计算任务，是Python中许多数值工作流程的重要构成部分。

获取即时帮助、个性化解释和交互式代码示例。

---

### 理解数组数据类型

# 理解数组数据类型

NumPy数组的一个基本特点是它们是同质的；单个数组中的所有元素必须是相同的数据类型。这与标准的Python列表不同，后者可以在同一个列表中包含各种类型的元素（如整数、字符串和浮点数）。这种同质性是NumPy高效能与性能的基础，使得数值运算能够进行优化、低层次的实现。

数组元素的具体数据类型存储在一个名为 `dtype`（数据类型的缩写）的特殊属性中。理解并有时控制 `dtype` 对于以下几个方面很重要：

1. **内存占用：** 不同的数据类型需要不同的内存量。一个64位整数（`int64`）比32位整数（`int32`）或8位整数（`int8`）使用更多的内存。选择最小的合适类型可以显著减少大型数组的内存占用。
2. **性能：** 对特定类型数组（特别是数值类型）的操作可以由底层的C或Fortran代码更快地执行，因为这些操作无需标准Python中所需的类型检查开销。
3. **精度和范围：** 浮点类型（`float32`，`float64`）提供不同级别的精度。整数类型能表示的数值范围也不同。选择正确的类型可以确保计算准确并避免潜在的溢出错误。

### 常见NumPy数据类型

NumPy支持比标准Python更丰富的数值类型。以下是一些最常见的类型：

| 类型 | 字符串代码 | 说明 | 示例 |
| --- | --- | --- | --- |
| `int8`, `int16`, `int32`, `int64` | `i1`, `i2`, `i4`, `i8` | 有符号整数（8、16、32或64位） | `-128` 到 `127` (`i1`) |
| `uint8`, `uint16`, `uint32`, `uint64` | `u1`, `u2`, `u4`, `u8` | 无符号整数（非负数） | `0` 到 `255` (`u1`) |
| `float16`, `float32`, `float64` | `f2`, `f4`, `f8` | 浮点数（半精度、单精度、双精度） | `3.14159` |
| `complex64`, `complex128` | `c8`, `c16` | 复数（由两个32位或64位浮点数表示） | `1 + 2j` |
| `bool` | `?` | 布尔类型，存储 `True` 和 `False` 值 | `True` |
| `object` | `O` | Python对象类型 | `(1, 'a')`, `[1, 2]` |
| `string_` | `S` | 固定长度ASCII字符串类型（例如，`S10`表示长度为10的字符串） | `'numpy'` |
| `unicode_` | `U` | 固定长度Unicode类型（例如，`U10`表示长度为10的字符串） | `'你好'` |

*注意：* 默认的整数类型（`int_`）和浮点类型（`float_`）通常分别对应`int64`和`float64`，这取决于您的系统架构。但在需要特定精度或大小时，明确指定是一个好的做法。

### 查看数组的数据类型

您可以使用NumPy数组的`dtype`属性轻松查看其数据类型。

```python
import numpy as np

arr_int = np.array([1, 2, 3, 4])
print(f"Array: {arr_int}")
print(f"Data Type: {arr_int.dtype}")

arr_float = np.array([1.0, 2.5, 3.0, 4.8])
print(f"\nArray: {arr_float}")
print(f"Data Type: {arr_float.dtype}")

arr_mixed = np.array([1, 2, 3.5, 4])
print(f"\nArray: {arr_mixed}")
print(f"Data Type: {arr_mixed.dtype}")
```

Output:

```
Array: [1 2 3 4]
Data Type: int64

Array: [1.  2.5 3.  4.8]
Data Type: float64

Array: [1.  2.  3.5 4. ]
Data Type: float64
```

请注意最后一个例子，由于列表中既包含整数又包含浮点数，NumPy自动推断出可以容纳所有元素的最通用类型，在本例中是`float64`。

### 创建时指定数据类型

您不必依赖NumPy的自动推断。在创建数组时，可以使用`dtype`参数 (parameter)明确指定所需的数据类型。这对于控制内存占用或确保特定精度很有用。

```python
import numpy as np

arr_float32 = np.array([1, 2, 3], dtype=np.float32)
print(f"Array: {arr_float32}")
print(f"Data Type: {arr_float32.dtype}")

arr_int8 = np.array([10, 20, 127], dtype=np.int8)
print(f"\nArray: {arr_int8}")
print(f"Data Type: {arr_int8.dtype}")

arr_complex = np.array([1+1j, 2+2j], dtype='c8')
print(f"\nArray: {arr_complex}")
print(f"Data Type: {arr_complex.dtype}")

zeros_uint16 = np.zeros(5, dtype=np.uint16)
print(f"\nArray: {zeros_uint16}")
print(f"Data Type: {zeros_uint16.dtype}")
```

Output:

```
Array: [1. 2. 3.]
Data Type: float32

Array: [ 10  20 127]
Data Type: int8

Array: [1.+1.j 2.+2.j]
Data Type: complex64

Array: [0 0 0 0 0]
Data Type: uint16
```

### 使用`astype`更改数据类型

有时，您需要将现有数组转换为不同的数据类型。`astype()`方法会创建一个*新*数组，其类型为您指定的类型，同时复制原始数据并根据需要进行类型转换。它不会修改原始数组，除非您将结果重新赋值回原始变量名。

```python
import numpy as np

arr_float = np.array([1.1, 2.7, 3.5, 4.9])
print(f"Original Array: {arr_float}")
print(f"Original dtype: {arr_float.dtype}")

arr_int = arr_float.astype(np.int32)
print(f"\nConverted to int32: {arr_int}")
print(f"New dtype: {arr_int.dtype}")

arr_num = np.array([0, 1, 5, 0, -2])
print(f"\nNumeric Array: {arr_num}")
arr_bool = arr_num.astype(np.bool_)
print(f"Converted to bool: {arr_bool}")
print(f"New dtype: {arr_bool.dtype}")

arr_str = arr_int.astype(np.string_)
print(f"\nConverted to string: {arr_str}")
print(f"New dtype: {arr_str.dtype}")
```

Output:

```
Original Array: [1.1 2.7 3.5 4.9]
Original dtype: float64

Converted to int32: [1 2 3 4]
New dtype: int32

Numeric Array: [ 0  1  5  0 -2]
Converted to bool: [False  True  True False  True]
New dtype: bool

Converted to string: [b'1' b'2' b'3' b'4']
New dtype: |S11
```

**注意：** 使用 `astype()` 时请注意。将浮点数转换为整数会截断小数部分，而不是四舍五入。将高精度类型（如`float64`）转换为低精度类型（如`float32`）可能导致精度丢失。将数据转换为范围较小的类型（如`int64`转换为`int8`）可能导致意料之外的结果或错误，如果值超出目标类型的限制。

理解NumPy的数据类型是一个基本步骤。它让您能够通过做出明智的选择，来存储和处理您的数值数据，从而编写出更节省内存、运行更快的代码。随着您处理更大规模的数据集，选择正确 `dtype` 的影响会变得越来越重要。

获取即时帮助、个性化解释和交互式代码示例。

---

### 数组基本属性

# 数组基本属性

理解 NumPy 数组的结构通常是必需的，尤其当数组很大时，而无需打印整个数组的内容。NumPy 数组带有一些实用的属性，它们提供关于数组本身的元数据。这些属性不像方法那样需要括号 `()`；你可以直接使用点表示法访问它们（例如，`my_array.attribute`）。主要的属性有：`ndim`、`shape`、`size` 和 `dtype`。

假设你已经创建了两个这样的数组：

```python
import numpy as np

arr1d = np.array([1, 2, 3, 4, 5])

arr2d = np.array([[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0],
                  [9.0, 10.0, 11.0, 12.0]])
```

现在，我们来查看它们的属性。

### `ndim`：轴（维度）的数量

`ndim` 属性告诉你数组的轴或维度的数量。一维数组有一个轴，二维数组有两个轴（就像电子表格中的行和列），三维数组有三个，以此类推。

```python
print(arr1d.ndim)

print(arr2d.ndim)
```

了解维度的数量对于理解如何索引数组以及操作可能如何表现非常重要。

### `shape`：数组的维度

`ndim` 告诉你轴的*数量*，而 `shape` 属性告诉你数组沿每个轴的*大小*。它返回一个整数元组，表示每个维度上的元素数量。

对于我们的一维数组 `arr1d`，其形状反映了它的长度：

```python
print(arr1d.shape)
```

请注意元组 `(5,)` 中的逗号。这表示它是一个只含一个元素的元组，指明一个长度为 5 的维度。

对于我们的二维数组 `arr2d`，其形状反映了其行和列的结构：

```python
print(arr2d.shape)
```

这个输出 `(3, 4)` 告诉我们数组有 2 个维度（因为元组中有两个数字，与 `arr2d.ndim` 匹配）。第一个轴的长度是 3（可以认为是行），第二个轴的长度是 4（可以认为是列）。

`shape` 是最常用的属性之一，因为它能快速概览数组的结构。

### `size`：元素总数

`size` 属性提供数组中元素的总数量。它简单地等于 `shape` 元组中所有元素的乘积。

```python
print(arr1d.size)

print(arr2d.size)
```

对于 `arr1d`，大小是 5（与它的形状 `(5,)` 匹配）。对于 `arr2d`，大小是 3×4=123 \times 4 = 123×4=12（与它的形状 `(3, 4)` 匹配）。这个属性对于快速了解你正在处理的数据量很方便。

### `dtype`：数组元素的数据类型

我们之前讨论过 NumPy 数组包含相同数据类型的元素。`dtype` 属性显示了该数据类型是什么。NumPy 在你创建数组时会自动推断出合适的数据类型，但你也可以明确指定它。

```python
print(arr1d.dtype)

print(arr2d.dtype)
```

在 `arr1d` 中，NumPy 检测到整数，所以它很可能分配了一个 64 位整数类型 (`int64`)。在 `arr2d` 中，我们使用了带小数点的数字（例如 `1.0`），所以 NumPy 分配了一个 64 位浮点类型 (`float64`)。

了解 `dtype` 很重要，原因如下：

- **内存占用：** 不同的数据类型占用不同大小的内存（例如，`int64` 比 `int8` 占用更多内存）。
- **精度：** 浮点类型（`float32`、`float64`）提供不同级别的精度。
- **兼容性：** 某些操作需要特定的数据类型。

这些属性（`ndim`、`shape`、`size`、`dtype`）是你检查和理解 NumPy 数组结构和性质的首要工具。定期检查它们有助于确保你的数组结构符合预期，然后再进行更复杂的操作。

获取即时帮助、个性化解释和交互式代码示例。

---

### 实践操作：创建和查看数组

# 实践操作：创建和查看数组

示例演示了使用多种方法创建多个 NumPy 数组。创建后，使用查看属性检查这些数组的结构。

首先，请确保已安装 NumPy 并将其导入。通常的约定是将其导入为别名 `np`。如果你在 Jupyter Notebook 或交互式 Python 环境中工作，请执行以下代码：

```python
import numpy as np
```

### 从 Python 列表创建数组

创建 NumPy 数组最直接的方法是从现有 Python 列表或元组创建。

**1. 一维数组：**
让我们从一个简单的整数列表开始。

```python

list_1d = [1, 3, 5, 7, 9]

array_1d = np.array(list_1d)

print(array_1d)
```

输出：

```
[1 3 5 7 9]
```

现在，让我们查看其属性：

```python
print(f"类型: {type(array_1d)}")
print(f"数据类型 (dtype): {array_1d.dtype}")
print(f"维度 (ndim): {array_1d.ndim}")
print(f"形状: {array_1d.shape}")
print(f"大小: {array_1d.size}")
```

输出：

```
类型: <class 'numpy.ndarray'>
数据类型 (dtype): int64
维度 (ndim): 1
形状: (5,)
大小: 5
```

请注意，NumPy 将数据类型推断为 `int64`（一个64位整数，默认整数类型通常取决于你的系统）。它是一个1维数组（`ndim=1`），包含5个元素（`size=5`），其形状表示为 `(5,)`，表示沿着单个轴有5个元素。

**2. 二维数组：**
现在，让我们使用嵌套列表来创建一个2维数组（类似于矩阵）。

```python

list_2d = [[1, 2, 3], [4, 5, 6]]

array_2d = np.array(list_2d)

print(array_2d)
```

输出：

```
[[1 2 3]
 [4 5 6]]
```

让我们查看这个2维数组：

```python
print(f"数据类型 (dtype): {array_2d.dtype}")
print(f"维度 (ndim): {array_2d.ndim}")
print(f"形状: {array_2d.shape}")
print(f"大小: {array_2d.size}")
```

输出：

```
数据类型 (dtype): int64
维度 (ndim): 2
形状: (2, 3)
大小: 6
```

这次，NumPy 正确识别出它是一个2维数组（`ndim=2`）。形状 `(2, 3)` 告诉我们它有2行3列。元素总数（`size`）是 2×3=62 \times 3 = 62×3=6。

### 使用内置数组创建函数

NumPy 提供无需从 Python 列表开始创建数组的函数，这通常更高效。

**1. `np.zeros` 和 `np.ones`：**
这些函数分别创建填充零或一的数组。你需要将所需形状作为元组提供。

```python

zeros_array = np.zeros((3, 4))
print("全零数组:
", zeros_array)
print(f"全零数组数据类型: {zeros_array.dtype}
")

ones_array = np.ones(5, dtype=np.int16)
print("全一数组:
", ones_array)
print(f"全一数组数据类型: {ones_array.dtype}")
print(f"全一数组形状: {ones_array.shape}")
```

输出：

```
全零数组:
 [[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
全零数组数据类型: float64

全一数组:
 [1 1 1 1 1]
全一数组数据类型: int16
全一数组形状: (5,)
```

注意 `np.zeros` 默认为浮点数（`float64`），而我们为 `np.ones` 明确要求了16位整数（`int16`）。

**2. `np.arange`：**
类似于 Python 的 `range`，但返回一个 NumPy 数组。

```python

range_array = np.arange(0, 10, 2)
print("Arange 数组:
", range_array)
print(f"Arange 形状: {range_array.shape}")
print(f"Arange 数据类型: {range_array.dtype}")
```

输出：

```
Arange 数组:
 [0 2 4 6 8]
Arange 形状: (5,)
Arange 数据类型: int64
```

**3. `np.linspace`：**
创建一个数组，包含指定数量的元素，这些元素在起始值和结束值之间均匀间隔（包含两端）。

```python

linspace_array = np.linspace(0, 1, 6)
print("Linspace 数组:
", linspace_array)
print(f"Linspace 形状: {linspace_array.shape}")
print(f"Linspace 数据类型: {linspace_array.dtype}")
```

输出：

```
Linspace 数组:
 [0.  0.2 0.4 0.6 0.8 1. ]
Linspace 形状: (6,)
Linspace 数据类型: float64
```

### 理解数据类型 (dtypes)

NumPy 数组之所以高效，是因为所有元素通常共享相同的数据类型。让我们看看当我们创建数组并指定类型时会发生什么。

```python

float_array = np.array([1.0, 2.5, 3.7, 4.2])
print("浮点数数组:", float_array)
print(f"数据类型: {float_array.dtype}
")

int_from_float = np.array([1.0, 2.5, 3.7, 4.2], dtype=np.int32)
print("从浮点数转换的整数数组:", int_from_float)
print(f"数据类型: {int_from_float.dtype}
")

mixed_array = np.array([1, 2.5, 3, 4.8])
print("混合数组:", mixed_array)
print(f"数据类型: {mixed_array.dtype}")
```

输出：

```
浮点数数组: [1.  2.5 3.7 4.2]
数据类型: float64

从浮点数转换的整数数组: [1 2 3 4]
数据类型: int32

混合数组: [1.  2.5 3.  4.8]
数据类型: float64
```

当我们强制将浮点数转换为 `int32` 数组时，小数部分被截断了。当从包含整数和浮点数的混合列表创建数组且未指定 `dtype` 时，NumPy 会智能地将所有元素向上转换为 `float64`，以容纳浮点数且不丢失信息。

### 数组结构可视化

有时，视觉表示有助于理解结构，特别是对于2维数组。让我们将之前创建的 `array_2d`（[[1,2,3],[4,5,6]][[1, 2, 3], [4, 5, 6]][[1,2,3],[4,5,6]]）可视化，它的形状是 `(2, 3)`。

123456列 0列 1列 2行 1行 0

> 热力图形象地表示了 `array_2d` 的2行3列。每个单元格对应数组中的一个元素，显示其值和位置。

本次实践环节涵盖了从列表创建 NumPy 数组，以及使用 `np.zeros`、`np.ones`、`np.arange` 和 `np.linspace` 等专用函数。我们还练习了查看数组的基本属性：`dtype`、`ndim`、`shape` 和 `size`。理解这些创建方法和属性是有效使用 NumPy 进行数值计算任务的重要前提。你现在已具备能力，可以创建 Python 中数据分析和科学计算的基本构成要素。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 3 Numpy Array Indexing Slicing

### 访问单个元素

# 访问单个元素

就像你在表格中按行和列号查找信息，或在街道上按门牌号找到特定房屋一样，你需要一种方法来定位并取出NumPy数组中存储的单个值。这个操作用到*索引*。

### 一维数组的索引

最简单的情况是一维数组，它就像一个值列表或序列。在Python中，因此也在NumPy中，索引从0开始。这表示第一个元素在索引0处，第二个在索引1处，依此类推。这通常被称为**零基索引**。

让我们创建一维数组：

```python
import numpy as np

arr1d = np.array([10, 11, 12, 13, 14])
print(arr1d)
```

```text
[10 11 12 13 14]
```

要访问一个元素，在数组名后使用方括号 `[]`，并将索引放在方括号内：

```python

first_element = arr1d[0]
print(f"第一个元素 (索引 0): {first_element}")

third_element = arr1d[2]
print(f"第三个元素 (索引 2): {third_element}")
```

```text
第一个元素 (索引 0): 10
第三个元素 (索引 2): 12
```

NumPy也支持负数索引，这对于从数组末尾访问元素很方便。索引 `-1` 表示最后一个元素，`-2` 表示倒数第二个，依此类推。

```python

last_element = arr1d[-1]
print(f"最后一个元素 (索引 -1): {last_element}")

second_last = arr1d[-2]
print(f"倒数第二个元素 (索引 -2): {second_last}")
```

```text
最后一个元素 (索引 -1): 14
倒数第二个元素 (索引 -2): 13
```

尝试访问有效范围之外的索引（0到length-1，或-1到-length）将导致 `IndexError`。确保索引在数组的大小范围内是很重要的。

### 二维数组（矩阵）的索引

二维数组，或称矩阵，有行和列。要访问二维数组中的单个元素，你需要同时指定行索引和列索引。同样，行索引和列索引都是零基的。

标准语法是 `array[行索引, 列索引]`。注意方括号内行索引和列索引之间有逗号分隔。

让我们创建一个二维数组：

```python

arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])
print("原始二维数组:\n", arr2d)
```

```text
原始二维数组:
 [[1 2 3]
  [4 5 6]]
```

现在，让我们访问一些元素：

```python

element_00 = arr2d[0, 0]
print(f"[0, 0]处的元素: {element_00}")

element_12 = arr2d[1, 2]
print(f"[1, 2]处的元素: {element_12}")

element_01 = arr2d[0, 1]
print(f"[0, 1]处的元素: {element_01}")
```

```text
[0, 0]处的元素: 1
[1, 2]处的元素: 6
[0, 1]处的元素: 2
```

可以把第一个索引看作选择行，第二个索引看作选择该行内的列。

G

struct

行\列

0

1

2

0

arr2d[0, 0]
(值: 1)

arr2d[0, 1]
(值: 2)

arr2d[0, 2]
(值: 3)

1

arr2d[1, 0]
(值: 4)

arr2d[1, 1]
(值: 5)

arr2d[1, 2]
(值: 6)

> 二维数组索引的可视化表示。行索引在前，列索引在后。突出显示的单元格 `arr2d[1, 1]` 对应于行索引1和列索引1处的元素。

### 高维数组的索引

这个方法很自然地适用于更高维度的数组。对于N维数组，你在方括号内提供N个由逗号分隔的索引： `array[index_dim0, index_dim1, ..., index_dimN-1]`。

考虑一个三维数组（你可以把它看作是二维数组的层或页）：

```python

arr3d = np.array([[[ 0,  1,  2],
                   [ 3,  4,  5]],

                  [[ 6,  7,  8],
                   [ 9, 10, 11]]])

print("原始三维数组形状:", arr3d.shape)
```

```text
原始三维数组形状: (2, 2, 3)
```

要访问一个元素，我们需要三个索引：层、行和列。

```python

element_102 = arr3d[1, 0, 2]
print(f"[1, 0, 2]处的元素: {element_102}")

element_011 = arr3d[0, 1, 1]
print(f"[0, 1, 1]处的元素: {element_011}")
```

```text
[1, 0, 2]处的元素: 8
[0, 1, 1]处的元素: 4
```

使用 `arr[i]` 或 `arr[i, j]` 等索引来访问单个元素，能得到该位置存储的特定值。这是处理NumPy数组中各个数据点最基本的方法。接下来，我们将在此基础上，学习如何使用切片和其他高级索引技术一次选取多个元素。

获取即时帮助、个性化解释和交互式代码示例。

---

### 一维数组切片

# 一维数组切片

访问单个元素可以提供精确控制。然而，通常需要处理一维数组中的一个序列或一段元素。切片操作可以让你根据索引范围取数组的一部分。如果你用过 Python 列表，会觉得 NumPy 的切片操作很熟悉，但它针对 NumPy 的数组结构做了优化。

### 一维数组切片的基础

对一维数组 `arr` 进行切片的语法是 `arr[start:stop:step]`。我们来分析一下这些组成部分：

- `start`：切片开始的索引（包含）。如果省略，默认是数组的开头（索引 0）。
- `stop`：切片结束的索引（不包含）。此索引处的元素*不*在切片结果中。如果省略，默认是数组的末尾。
- `step`：选择元素之间的间隔。如果省略，默认是 1（即选择连续的元素）。

我们来创建一个简单的数组进行操作：

```python
import numpy as np

arr = np.arange(10)
print(arr)
```

现在，我们来看一些切片示例：

1. **选择从索引 2 开始直到（但不包括）索引 5 的元素：**

   ```python
   slice1 = arr[2:5]
   print(slice1)
   ```

   这选择了索引 2、3 和 4 处的元素。
2. **选择从开头直到（但不包括）索引 4 的元素：**

   ```python
   slice2 = arr[:4]
   print(slice2)
   ```

   这选择了从索引 0 开始直到（但不包括）索引 4 的元素。
3. **选择从索引 5 开始到末尾的元素：**

   ```python
   slice3 = arr[5:]
   print(slice3)
   ```

   这选择了从索引 5 开始直到最后一个元素。
4. **选择整个数组：**

   ```python
   slice4 = arr[:]
   print(slice4)
   ```

### 使用步长值

`step` 值让你能选择不连续的元素：

1. **选择每隔一个元素：**

   ```python
   slice5 = arr[::2]
   print(slice5)
   ```
2. **在指定范围（索引 1 到 7）内选择每隔一个元素：**

   ```python
   slice6 = arr[1:7:2]
   print(slice6)
   ```

### 切片中的负索引

就像访问单个元素一样，你可以在切片中使用负索引。`-1` 指代最后一个元素，`-2` 指代倒数第二个，以此类推。

1. **选择最后 3 个元素：**

   ```python
   slice7 = arr[-3:]
   print(slice7)
   ```
2. **选择从开头直到最后 2 个元素之前的元素：**

   ```python
   slice8 = arr[:-2]
   print(slice8)
   ```
3. **使用负步长反转数组：**

   ```python
   reversed_arr = arr[::-1]
   print(reversed_arr)
   ```

### 切片是视图，不是副本

这是 NumPy 切片的一个重要方面：数组的切片返回的是数组数据的*视图*而不是*副本*。这意味着切片只是观察*相同*底层数据的一种不同方式。修改切片中的元素会影响到原始数组。

我们来看一下实际效果：

```python
print("原始数组：", arr)

arr_slice = arr[5:8]
print("切片：", arr_slice)

arr_slice[1] = 999
print("修改后的切片：", arr_slice)

print("切片修改后的原始数组：", arr)
```

可以看到，修改 `arr_slice` 中索引 1 处的元素（它对应于原始 `arr` 中索引 6 处的元素）也改变了 `arr` 中的值。这种行为的目的是为了提高性能和内存效率，因为它避免了不必要的数据重复，特别是对于数据分析和科学计算中常见的大型数组。

如果你明确需要一个切片的副本，以便修改不会影响原始数据，可以使用 `copy()` 方法：

```python
arr_copy = arr[5:8].copy()
arr_copy[1] = 111

print("原始数组：", arr)

print("复制的切片：", arr_copy)
```

理解切片以及视图与副本的区别，对于有效地处理数据和避免 NumPy 代码中意料之外的副作用非常重要。

获取即时帮助、个性化解释和交互式代码示例。

---

### 二维数组切片

# 二维数组切片

切片是一种强大的操作，适用于一维和多维数组。一维数组用于提取元素序列，而多维数组则允许选择特定的行、列或矩形数据块。这项功能对于处理NumPy中表示的矩阵式数据（如图像或表格）来说非常基本。

对于二维数组（矩阵），切片通过使用逗号分隔每个维度的切片规范，从而扩展了熟悉的 `start:stop:step` 符号。一般语法是：

`array[row_slice, column_slice]`

在这里，`row_slice` 指定要选择的行范围，`column_slice` 指定要选择的列范围。每个切片遵循与一维切片相同的规则：`start` 索引是包含的，`stop` 索引是不包含的，`step` 定义了步长。如果切片的任何部分（`start`、`stop` 或 `step`）被省略，NumPy 会使用默认值（维度开头、维度结尾、步长为1）。

我们使用一个示例二维数组来看看实际效果。

```python
import numpy as np

arr2d = np.array([[ 1,  2,  3,  4,  5],
                  [ 6,  7,  8,  9, 10],
                  [11, 12, 13, 14, 15],
                  [16, 17, 18, 19, 20]])

print("原始二维数组:\n", arr2d)
```

### 选择行

要选择行范围同时保留所有列，你需要为行指定切片，并对列使用 `:`。

```python

first_two_rows = arr2d[0:2, :]
print("\n前两行:\n", first_two_rows)

rows_1_and_2 = arr2d[1:3, :]
print("\n索引 1 和 2 的行:\n", rows_1_and_2)
```

冒号 `:` 充当列维度的通配符，表明所选行应包含所有列。如果你选择整行，也可以省略逗号后的冒号：`arr2d[0:2]` 等同于 `arr2d[0:2, :]`。

### 选择列

类似地，要选择特定列同时保留所有行，你需要对行切片使用 `:` 并指定所需的列范围。

```python

first_three_cols = arr2d[:, 0:3]
print("\n前三列:\n", first_three_cols)

cols_2_and_3 = arr2d[:, 2:4]
print("\n索引 2 和 3 的列:\n", cols_2_and_3)
```

### 选择子数组（矩形区域）

真正的优势在于你结合行和列切片来选择数组的特定矩形子区域。

```python

sub_array_1 = arr2d[0:2, 1:3]
print("\n子数组 (行 0-1, 列 1-2):\n", sub_array_1)

sub_array_2 = arr2d[1:3, 2:5]
print("\n子数组 (行 1-2, 列 2-4):\n", sub_array_2)
```

以下图表展示了对 `arr2d[1:3, 2:4]` 定义的区域进行切片操作，该操作对应于索引为 1 和 2 的行，以及索引为 2 和 3 的列。

G

clusterₐrray

arr2d

clusterₛlice

切片 [1:3, 2:4]

1

1

2

2

3

3

4

4

5

5

6

6

7

7

8

8

9

9

s8

s8

8->s8

10

10

s9

s9

9->s9

11

11

12

12

13

13

14

14

s13

s13

13->s13

15

15

s14

s14

14->s14

16

16

17

17

18

18

19

19

20

20

> 对 `arr2d[1:3, 2:4]` 进行切片的视觉表示。蓝色高亮显示的元素构成了结果 2x2 子数组，其中包含 `[[8, 9], [13, 14]]`。

### 切片是视图

需要记住的一个重要方面是，与一维数组切片一致，二维数组切片是原始数组数据的*视图*，而不是副本。这意味着如果你修改切片中的元素，这些更改将反映在原始数组中。

```python

slice_view = arr2d[0:2, 0:2]
print("\n原始切片视图:\n", slice_view)

slice_view[0, 0] = 99
print("\n修改后的切片视图:\n", slice_view)

print("\n修改切片后的原始数组:\n", arr2d)
```

如果你需要切片的独立副本，并且它独立于原始数组，你必须明确使用 `.copy()` 方法。

```python

slice_copy = arr2d[2:, 2:].copy()
print("\n切片副本:\n", slice_copy)

slice_copy[0, 0] = -1
print("\n修改后的切片副本:\n", slice_copy)

print("\n修改副本后的原始数组:\n", arr2d)
```

熟练掌握二维切片对于在 NumPy 中高效处理表格数据、图像块或任何网格状结构非常重要。它允许你隔离并处理数据的特定部分，而无需手动迭代元素。

获取即时帮助、个性化解释和交互式代码示例。

---

### 布尔索引

# 布尔索引

数组可以通过数字位置（例如 `array[2]`、`array[1, 3]`）或通过对数据范围进行切片（例如 `array[1:5]`）进行索引。然而，有时你需要根据数据的*值*或某些*条件*而不是其位置来选择数据。在这种情况下，布尔索引是一种强大的技术，它允许你使用 True/False 条件来筛选数组。

### 创建布尔数组

布尔索引的基础是布尔数组。你通常通过将比较运算符直接应用于 NumPy 数组来创建这些数组。NumPy 会逐元素执行比较，返回一个形状相同、填充了 `True` 或 `False` 值的新数组。

我们来看一个例子：

```python
import numpy as np

data = np.array([1, 5, 2, 8, 3, 7, 4, 6])
print(f"原始数组: {data}")

is_greater_than_4 = data > 4
print(f"布尔数组 (data > 4): {is_greater_than_4}")
```

执行这段代码会得到以下输出：

```
Original array: [1 5 2 8 3 7 4 6]
Boolean array (data > 4): [False  True False  True False  True False  True]
```

请注意 `is_greater_than_4` 具有与 `data` 相同数量的元素。如果 `data` 中对应元素大于 4，则 `is_greater_than_4` 中的每个元素为 `True`，否则为 `False`。

你可以使用任何标准比较运算符：`>`、`<`、`>=`、`<=`、`==` (等于) 和 `!=` (不等于)。

### 使用布尔数组选择数据

有了布尔数组后，你可以将其直接用作原始数组方括号 `[]` 内的索引。此操作会从原始数组中选择布尔数组中对应值为 `True` 的元素。

```python

selected_data = data[is_greater_than_4]
print(f"选定元素 (data > 4 时): {selected_data}")

selected_directly = data[data < 5]
print(f"选定元素 (data < 5 时): {selected_directly}")
```

输出：

```
Selected elements (where data > 4): [5 8 7 6]
Selected elements (where data < 5): [1 2 3 4]
```

结果是一个新的一维数组，只包含满足条件的元素。无论原始数组的维度如何，这都适用。

我们用二维数组来试试：

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

selected_matrix_elements = matrix[matrix > 5]
print(f"\n原始二维数组:\n{matrix}")
print(f"选定元素 (matrix > 5 时): {selected_matrix_elements}")
```

输出：

```
Original 2D array:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
Selected elements (where matrix > 5): [6 7 8 9]
```

重要提示：请注意，即使将布尔索引应用于二维数组，结果通常也是一个包含所选元素的一维数组。NumPy 会将输出展平，因为 `True` 值可能分散在不同的行和列中，不一定形成一个矩形子数组。

012345672468所有数据data > 4

> 散点图显示了原始数据点（灰色）以及使用条件 `data > 4` 选择的点（蓝色）。

### 组合多个条件

你经常需要同时根据多个条件筛选数据。你可以使用逻辑运算符组合布尔数组：

- `&` (逻辑与): 选择*两个*条件都为 `True` 的元素。
- `|` (逻辑或): 选择*至少一个*条件为 `True` 的元素。
- `~` (逻辑非): 反转布尔数组（将 `True` 变为 `False`，`False` 变为 `True`）。

**重要提示：** 当使用 `&` 或 `|` 组合条件时，你**必须**在每个单独的条件周围使用括号 `()`。这是因为 Python 和 NumPy 中的运算符优先级规则。位运算符 `&` 和 `|` 的优先级高于比较运算符，如 `<` 或 `>`。如果没有括号，Python 可能会尝试评估类似 `data > 3 & data` 的表达式，这会导致错误或意料之外的结果。

```python
data = np.array([1, 5, 2, 8, 3, 7, 4, 6, 9, 0])

selected_and = data[(data > 3) & (data < 8)]
print(f"元素满足 (data > 3) 且 (data < 8): {selected_and}")

selected_or = data[(data % 2 == 0) | (data > 5)]
print(f"元素满足 (data 为偶数) 或 (data > 5): {selected_or}")

selected_not = data[~(data == 5)]
print(f"元素满足 data 不等于 5: {selected_not}")
```

输出：

```
Elements where (data > 3) AND (data < 8): [5 7 4 6]
Elements where (data is even) OR (data > 5): [2 8 7 4 6 9 0]
Elements where data is NOT equal to 5: [1 2 8 3 7 4 6 9 0]
```

### 使用布尔索引修改数据

布尔索引不仅用于选择；它对于修改符合特定条件的数组元素也非常有用。你可以在赋值操作的左侧使用布尔索引，来改变条件为 `True` 的所有元素的值。

```python

arr = np.array([1, -2, 3, -4, 5, -6])
print(f"原始数组: {arr}")

arr[arr < 0] = 0
print(f"将负数替换为 0 后的数组: {arr}")

arr[arr > 3] = 100
print(f"将大于 3 的元素设置为 100 后的数组: {arr}")
```

输出：

```
Original array: [ 1 -2  3 -4  5 -6]
Array after replacing negatives with 0: [1 0 3 0 5 0]
Array after setting elements > 3 to 100: [  1   0   3   0 100   0]
```

这种方法在数据清理和准备中很常见，例如，替换无效的传感器读数或限制异常值。

布尔索引提供了一种灵活易读的方法，可以根据数据本身筛选和处理 NumPy 数组，是高效数据分析流程的重要组成部分。

获取即时帮助、个性化解释和交互式代码示例。

---

### 花式索引

# 花式索引

花式索引提供了一种精确的方法，用于根据索引位置选择数组中的特定元素，尤其当这些位置不连续时。与选择连续元素块的切片不同，也与根据条件筛选的布尔索引不同，花式索引提供了一种灵活的方式来检索单独指定的项目。这种技术使用包含整数索引的NumPy数组（或Python列表）来从另一个数组中挑选元素。

### 使用索引数组选择元素

核心思路很简单：你将一个索引数组或列表传递给要从中选择的数组。

#### 一维数组中的花式索引

我们从一个简单的一维数组开始：

```python
import numpy as np

arr = np.arange(10) * 5
print(f"Original array:\n{arr}")

indices = np.array([1, 5, 8, 2])
print(f"Indices to select: {indices}")

selected_elements = arr[indices]
print(f"Selected elements:\n{selected_elements}")
```

请注意以下几点：

1. 我们在方括号 `[]` 中使用了 `indices`，它是一个包含 `[1, 5, 8, 2]` 的NumPy数组。
2. NumPy 从 `arr` 中获取了这些特定索引位置的元素。
3. 结果数组 `selected_elements` 包含元素 `arr[1]`、`arr[5]`、`arr[8]` 和 `arr[2]`。
4. 结果数组的形状与*索引数组* (`indices`) 的形状匹配，而不是原始数组 (`arr`) 的形状。结果中元素的顺序也与索引数组中的顺序一致。

如果需要，你可以多次使用同一个索引：

```python
indices_repeated = np.array([3, 3, 1, 8, 1])
print(f"Indices with repetition: {indices_repeated}")

repeated_selection = arr[indices_repeated]
print(f"Selection with repeated indices:\n{repeated_selection}")
```

#### 多维数组中的花式索引

花式索引在多维数组中更为通用。你可以传递多个索引数组，根据不同轴上的坐标来选择元素。

考虑一个二维数组：

```python
arr2d = np.array([[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11],
                  [12, 13, 14, 15]])

print(f"Original 2D array:\n{arr2d}")
```

**选择特定行：**
如果你提供一个索引数组，它会选择与这些索引对应的整行：

```python

row_indices = np.array([0, 2])
selected_rows = arr2d[row_indices]

print(f"Selected rows (0 and 2):\n{selected_rows}")
```

**选择特定元素（坐标）：**
要使用坐标选择单个元素，你需要为*每个*维度提供一个索引数组。这些数组通常应具有相同的长度。NumPy 将索引逐个配对：选择的第一个元素在 `(row_indices[0], col_indices[0])`，第二个在 `(row_indices[1], col_indices[1])`，依此类推。

```python

row_indices = np.array([0, 3, 1, 2])

col_indices = np.array([1, 2, 0, 3])

selected_coords = arr2d[row_indices, col_indices]

print(f"Selected elements by coordinates:\n{selected_coords}")
```

这里，选定的元素是 `arr2d[0, 1]`（即1），`arr2d[3, 2]`（即14），`arr2d[1, 0]`（即4）和 `arr2d[2, 3]`（即11）。结果是一个一维数组，因为我们提供了特定的坐标。

**花式索引与切片或基本索引的组合：**
你也可以将花式索引与切片或基本索引混合使用。例如，要选择特定行和特定列，你可以将它们组合起来：

```python

row_indices = np.array([0, 2])
col_indices = np.array([1, 3])

selected_block_1 = arr2d[row_indices][:, col_indices]
print(f"Selected block (Method 1):\n{selected_block_1}")

selected_block_2 = arr2d[np.ix_(row_indices, col_indices)]
print(f"Selected block (Method 2 with np.ix_):\n{selected_block_2}")
```

两种方法都生成了一个子数组，其中包含行 0 和 2 与列 1 和 3 交汇处的元素。对于初学者来说，第一种方法通常更直观，它首先选择行，然后从中间结果中选择列。

你也可以选择特定行和*所有*列，或者从*所有*行中选择特定列：

```python

print(f"Rows 1 and 3, all columns:\n{arr2d[[1, 3]]}")

print(f"All rows, columns 0 and 2:\n{arr2d[:, [0, 2]]}")
```

### 使用花式索引修改数组

就像基本索引和切片一样，花式索引也可用于修改数组元素。你将花式索引表达式放在赋值语句的左侧：

```python
arr = np.arange(10) * 5
print(f"Original array: {arr}")

indices = np.array([1, 5, 8, 2])
arr[indices] = 99
print(f"Array after modifying elements at {indices}: {arr}")

arr[indices] = np.array([-1, -2, -3, -4])
print(f"Array after assigning multiple values: {arr}")
```

这同样适用于多维数组：

```python
print(f"Original 2D array:\n{arr2d}")

row_indices = np.array([0, 3, 1, 2])
col_indices = np.array([1, 2, 0, 3])
arr2d[row_indices, col_indices] = 0

print(f"2D array after setting specific elements to 0:\n{arr2d}")
```

### 重要提示：副本与视图

切片和花式索引之间一个显著的区别是，**花式索引几乎总是返回数据的*副本*，而不是视图**。这意味着对花式索引返回的数组所做的更改*不会*影响原始数组。

```python
arr = np.arange(10)
print(f"Original array: {arr}")

slice_view = arr[2:5]
print(f"Slice view: {slice_view}")
slice_view[0] = 99
print(f"Original array after modifying slice view: {arr}")

indices = np.array([6, 8])
fancy_copy = arr[indices]
print(f"Fancy copy: {fancy_copy}")
fancy_copy[0] = -1
print(f"Original array after modifying fancy copy: {arr}")
```

记住这种副本行为很重要，特别是在你打算修改原始数组数据时。如果你需要使用索引数组修改原始数组，请像前面所示，将花式索引表达式直接放在赋值语句的左侧。

---

### 修改数组子集

# 修改数组子集

各种索引方法能够选择 NumPy 数组的特定元素或部分。这些强大的选择方法被用来*更改*数组中的值。这是在数据分析和机器学习 (machine learning)任务中进行数据清理、转换和处理的一项基本操作。

核心思想很简单：任何您用来选择数据的方法，只要在赋值运算符 (`=`) 的左侧使用，都可以用来为选定元素赋新值。

### 为单个元素赋值

就像您使用方括号 `[]` 和索引来访问元素一样，您也可以使用相同的语法来为该特定位置赋新值。

对于一维数组：

```python
import numpy as np

arr1d = np.arange(10)
print("原始一维数组:", arr1d)

arr1d[5] = 99
print("修改后的一维数组:", arr1d)
```

对于二维数组，您使用 `[行, 列]` 记法：

```python

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("原始二维数组:\n", arr2d)

arr2d[1, 2] = -5
print("修改后的二维数组:\n", arr2d)
```

### 修改数组切片

切片允许您选择一系列元素。您可以为整个切片赋一个单一值，或者赋一个与切片形状匹配的数组值。

**为切片赋标量值：**

当您为切片赋一个单一标量值（例如一个数字）时，NumPy 会使用*广播*机制，用该值填充整个选定的切片。

```python

arr1d = np.arange(10)
print("原始一维数组:", arr1d)

arr1d[2:5] = 100
print("修改后的切片（标量）:", arr1d)
```

对于二维数组，这同样适用：

```python

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("原始二维数组:\n", arr2d)

arr2d[0:2, 0:2] = 0
print("修改后的切片（标量，二维）:\n", arr2d)
```

**为切片赋数组：**

您也可以将另一个数组（或列表）赋给切片，前提是您要赋的数组的形状与您选择的切片的形状匹配。

```python

arr1d = np.arange(10)
print("原始一维数组:", arr1d)

arr1d[5:8] = [55, 66, 77]
print("修改后的切片（数组）:", arr1d)
```

**关于切片与视图的重要提示：** 当您修改一个切片时，通常是直接修改原始数组。这是因为切片通常返回原始数据的*视图*，而不是副本。请注意这一点，因为通过切片视图所做的更改会反映在原始数组中。

```python
arr = np.arange(5)
print("原始数组:", arr)
arr_slice = arr[1:4]
print("切片:", arr_slice)

arr_slice[:] = 99
print("修改后的切片:", arr_slice)
print("切片修改后的原始数组:", arr)
```

请注意，修改 `arr_slice` 也改变了 `arr`。

### 使用布尔索引进行修改

布尔索引对于更改满足特定条件的元素非常有用。您将布尔条件放在方括号内。所有条件评估为 `True` 的元素都将被选中进行赋值。

```python

data = np.array([-1, 5, -3, 0, 8, -2, 4])
print("原始数据:", data)

data[data < 0] = 0
print("替换负数后的数据:", data)

data[data > 5] = 500
print("将大于 5 的值设限后的数据:", data)
```

同样，广播机制也适用。当您赋一个单一标量值（如上面的 `0` 或 `500`）时，它会被赋给所有通过布尔条件选中的元素。

### 使用高级索引进行修改

高级索引使用整数数组来指定您要访问或修改的元素的索引。这使得选择和更改非连续元素成为可能。

```python

arr = np.zeros(10, dtype=int)
print("初始数组:", arr)

indices_to_change = [1, 4, 7, 8]
arr[indices_to_change] = 99
print("高级索引赋值后的数组:", arr)

new_values = [11, 44, 77, 88]
arr[indices_to_change] = new_values
print("赋值序列后的数组:", arr)
```

高级索引也可以用于多维数组：

```python

arr2d = np.arange(12).reshape((3, 4))
print("原始二维数组:\n", arr2d)

rows = np.array([0, 1, 2])
cols = np.array([1, 2, 0])

arr2d[rows, cols] = -1
print("修改后的二维数组（高级索引）:\n", arr2d)
```

主要内容是，相同的灵活选择机制对于赋值操作同样有效。

能够使用索引、切片、布尔条件或索引数组选择数据的子集，然后直接修改这些子集，是 NumPy 中数据操作的根本。在为数据分析或机器学习 (machine learning)模型准备数据时，您会经常使用这些技巧。

获取即时帮助、个性化解释和交互式代码示例。

---

### 动手实践：从数组中选择数据

# 动手实践：从数组中选择数据

NumPy 数组中数据访问和修改的各种选择方法的实际应用。这包括基本索引、切片、布尔索引和花式索引。

首先，请确保已导入 NumPy。如果您正在 Jupyter Notebook 或 Python 脚本中进行操作，请从这一行开始：

```python
import numpy as np
```

### 准备示例数组

我们需要一些数组来操作。让我们创建一个简单的一维数组和一个二维数组。

```python

arr1d = np.arange(10)
print("一维数组:")
print(arr1d)

arr2d = np.array([[ 1,  2,  3,  4],
                  [ 5,  6,  7,  8],
                  [ 9, 10, 11, 12]])
print("\n二维数组:")
print(arr2d)
```

### 基本索引和切片

请记住，索引用于访问单个元素，而切片用于提取子数组。

**练习 1：访问元素**

1. 从 `arr1d` 中获取索引为 3 的元素。
2. 从 `arr2d` 中获取第二行（索引 1）第三列（索引 2）的元素。

```python

element_1d = arr1d[3]
print(f"arr1d 中索引为 3 的元素: {element_1d}")

element_2d = arr2d[1, 2]
print(f"arr2d 中第 1 行第 2 列的元素: {element_2d}")
```

**练习 2：数组切片**

1. 从 `arr1d` 中获取前 5 个元素。
2. 从 `arr1d` 中获取从索引 5 开始的所有元素。
3. 从 `arr2d` 中获取前两行。
4. 从 `arr2d` 中获取前两列。
5. 从 `arr2d` 的右上角提取一个 2x2 的子数组（行 0-1，列 2-3）。

```python

slice1 = arr1d[:5]
print(f"\narr1d 的前 5 个元素: {slice1}")

slice2 = arr1d[5:]
print(f"arr1d 中从索引 5 开始的所有元素: {slice2}")

slice3 = arr2d[:2]
print("\narr2d 的前两行:")
print(slice3)

slice4 = arr2d[:, :2]
print("\narr2d 的前两列:")
print(slice4)

slice5 = arr2d[:2, 2:]
print("\narr2d 的右上角 2x2 子数组:")
print(slice5)
```

### 布尔索引

布尔索引允许基于条件进行选择。

**练习 3：条件选择**

1. 从 `arr1d` 中选择所有大于 5 的元素。
2. 从 `arr2d` 中选择所有第一列（索引 0）的元素大于 4 的行。

```python

bool_mask_1d = arr1d > 5
selected_1d = arr1d[bool_mask_1d]
print(f"\narr1d 中大于 5 的元素: {selected_1d}")

bool_mask_2d = arr2d[:, 0] > 4
selected_rows_2d = arr2d[bool_mask_2d]
print("\narr2d 中第一个元素大于 4 的行:")
print(selected_rows_2d)
```

### 花式索引

花式索引使用索引数组来选择元素。

**练习 4：使用索引数组选择**

1. 从 `arr1d` 中选择索引为 1、3 和 7 的元素。
2. 从 `arr2d` 中选择索引为 0 和 2 的行。
3. 从 `arr2d` 中选择坐标为 (0, 1)、(1, 3) 和 (2, 0) 的元素。

```python

indices_1d = np.array([1, 3, 7])
selected_fancy_1d = arr1d[indices_1d]
print(f"\narr1d 中索引 {indices_1d} 处的元素: {selected_fancy_1d}")

row_indices = np.array([0, 2])
selected_fancy_rows = arr2d[row_indices]
print("\narr2d 中索引 0 和 2 的行:")
print(selected_fancy_rows)

row_coords = np.array([0, 1, 2])
col_coords = np.array([1, 3, 0])
selected_fancy_elements = arr2d[row_coords, col_coords]
print(f"\narr2d 中坐标为 (0,1), (1,3), (2,0) 的元素: {selected_fancy_elements}")
```

### 修改数组子集

您可以使用任何这些索引方法对选定的部分赋值。请记住，切片通常返回*视图*，这意味着修改会影响原始数组，而花式索引通常返回*副本*。

**练习 5：更新数组值**

1. 将 `arr1d` 中索引为 0 的元素更改为 99。
2. 将 `arr1d` 的前 3 个元素更改为 100。
3. 将 `arr1d` 中所有偶数元素更改为 0。
4. 在 `arr2d` 中，将 (1, 1) 处的元素更改为 -5。
5. 在 `arr2d` 中，将整个最后一行（索引 2）更改为只包含值 13。

```python
print("\n修改前的原始 arr1d:", arr1d)
print("修改前的原始 arr2d:\n", arr2d)

arr1d_copy1 = arr1d.copy()
arr1d_copy1[0] = 99
print("\n1. arr1d 将索引 0 设置为 99 后的结果:", arr1d_copy1)

arr1d_copy2 = arr1d.copy()
arr1d_copy2[:3] = 100
print("2. arr1d 将前 3 个元素设置为 100 后的结果:", arr1d_copy2)

arr1d_copy3 = arr1d.copy()
arr1d_copy3[arr1d_copy3 % 2 == 0] = 0
print("3. arr1d 将偶数设置为 0 后的结果:", arr1d_copy3)

arr2d_copy1 = arr2d.copy()
arr2d_copy1[1, 1] = -5
print("\n4. arr2d 将元素 (1,1) 设置为 -5 后的结果:\n", arr2d_copy1)

arr2d_copy2 = arr2d.copy()
arr2d_copy2[2] = 13

print("5. arr2d 将最后一行设置为 13 后的结果:\n", arr2d_copy2)
```

这些练习涵盖了您在 NumPy 数组中操作数据的主要方式。熟练掌握这些选择和修改方法对于为数据分析或模型训练做准备至关重要。您还可以进一步尝试组合使用这些方法，例如，从通过布尔条件选择的行中选择特定列。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 4 Fundamental Numpy Operations

### 基本算术运算

# 基本算术运算

NumPy 在数值计算方面表现出色的主要原因之一，在于它能够对数据执行批量操作，而无需使用 Python `for` 循环。这些操作是*按元素*应用的，这意味着操作是独立地对数组中每个对应的元素执行的。

我们从基本算术开始：加法、减法、乘法和除法。当您在 NumPy 数组上使用标准算术运算符（`+`、`-`、`*`、`/`）时，NumPy 会自动将操作应用到每个元素。

### 数组间的算术运算

如果您有两个相同形状的数组，可以直接在它们之间执行算术运算。操作将按元素应用。

考虑两个简单数组：

```python
import numpy as np

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([10, 20, 30, 40])

print(f"arr1: {arr1}")
print(f"arr2: {arr2}")
```

输出：

```
arr1: [1 2 3 4]
arr2: [10 20 30 40]
```

现在，我们执行一些算术运算：

**加法：**

```python
sum_arr = arr1 + arr2
print(f"arr1 + arr2: {sum_arr}")
```

输出：

```
arr1 + arr2: [11 22 33 44]
```

这里，`arr1` 的第一个元素 (1) 与 `arr2` 的第一个元素 (10) 相加，`arr1` 的第二个元素 (2) 与 `arr2` 的第二个元素 (20) 相加，依此类推。

**减法：**

```python
diff_arr = arr2 - arr1
print(f"arr2 - arr1: {diff_arr}")
```

输出：

```
arr2 - arr1: [ 9 18 27 36]
```

**乘法：**

```python
prod_arr = arr1 * arr2
print(f"arr1 * arr2: {prod_arr}")
```

输出：

```
arr1 * arr2: [ 10  40  90 160]
```

> **注意：** 这是按元素乘法，而非矩阵乘法。对于矩阵乘法，您将使用 `@` 运算符或 `np.dot()` 函数。

**除法：**

```python
div_arr = arr2 / arr1
print(f"arr2 / arr1: {div_arr}")
```

输出：

```
arr2 / arr1: [10. 10. 10. 10.]
```

请注意，即使 `arr1` 和 `arr2` 包含整数，除法结果也是浮点数。NumPy 通常会将除法结果提升为浮点类型以保持精度。

这些操作对于多维数组也类似地起作用，只要它们的形状匹配：

```python
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])

print(f"mat1:\n{mat1}\n")
print(f"mat2:\n{mat2}\n")

prod_mat = mat1 * mat2
print(f"mat1 * mat2:\n{prod_mat}")
```

输出：

```
mat1:
[[1 2]
 [3 4]]

mat2:
[[5 6]
 [7 8]]

mat1 * mat2:
[[ 5 12]
 [21 32]]
```

### 数组与标量间的算术运算

NumPy 也允许您在数组和单个数字（标量）之间执行算术运算。在这种情况下，NumPy 会将操作应用于标量与数组中的*每个元素*之间。这是一种简单的*广播*形式，我们稍后会更正式地介绍这个。

```python
arr = np.array([1, 2, 3, 4])
scalar = 10

print(f"Array: {arr}")
print(f"Scalar: {scalar}\n")

add_scalar = arr + scalar
print(f"arr + scalar: {add_scalar}")

mul_scalar = arr * scalar
print(f"arr * scalar: {mul_scalar}")

pow_scalar = arr ** 2
print(f"arr ** 2: {pow_scalar}")
```

输出：

```
Array: [1 2 3 4]
Scalar: 10

arr + scalar: [11 12 13 14]
arr * scalar: [10 20 30 40]
arr ** 2: [ 1  4  9 16]
```

### 操作会创建新数组

务必记住，对 NumPy 数组执行算术运算会生成包含结果的*新*数组。原始数组保持不变。

```python
arr_a = np.array([5, 10, 15])
arr_b = np.array([1, 2, 3])

print(f"Original arr_a: {arr_a}")
print(f"Original arr_b: {arr_b}\n")

result = arr_a / arr_b

print(f"Result of division: {result}")
print(f"arr_a after division: {arr_a}")
print(f"arr_b after division: {arr_b}")
```

输出：

```
Original arr_a: [ 5 10 15]
Original arr_b: [1 2 3]

Result of division: [5. 5. 5.]
arr_a after division: [ 5 10 15]
arr_b after division: [1 2 3]
```

如果您想就地修改数组，可以使用增广赋值运算符，如 `+=`、`-=`、`*=`、`/=`，尽管这不太常见，并且需要仔细考虑，特别是在数据类型方面。

```python
arr_a = np.array([5.0, 10.0, 15.0])
arr_b = np.array([1.0, 2.0, 3.0])

print(f"arr_a before in-place division: {arr_a}")

arr_a /= arr_b

print(f"arr_a after in-place division: {arr_a}")
```

输出：

```
arr_a before in-place division: [ 5. 10. 15.]
arr_a after in-place division: [5. 5. 5.]
```

这些基本算术运算是 NumPy 数值计算的根本。它们的按元素应用以及与标量一起工作的能力，与在 Python 中编写显式循环相比，提供了一种简洁高效的计算方法。接下来，我们将介绍通用函数 (ufuncs)，它们在此功能的基础上，支持更广范围的数学运算。

获取即时帮助、个性化解释和交互式代码示例。

---

### 通用函数（ufuncs）简介

# 通用函数（ufuncs）简介

基本的算术运算（如加法和乘法）可以在 NumPy 数组上逐元素执行。虽然这些操作很方便，但使用标准 Python 循环在大型数据集上执行它们会相当慢。NumPy 提供了一种高效的方式来执行快速的逐元素操作：通用函数（Universal Functions），通常简称为 ufuncs。

ufunc 本质上是一个以逐元素方式操作 `ndarray` 对象的函数。可以将它们看作是简单函数的向量 (vector)化封装器，能够一次性处理整个数组，从而省去了显式的 Python 循环。这种向量化 (quantization)是 NumPy 高效的一个重要原因。ufunc 是通过编译的 C 代码实现的，这让它们的执行速度远超对应的 Python 代码。

### 为什么要使用 Ufuncs？

- **速度：** 通过 ufuncs 执行操作比使用 Python 循环遍历数组快得多。
- **方便性：** 它们提供简洁的语法来对数组应用函数。
- **广播：** Ufuncs 本身支持广播，允许对形状不同但兼容的数组执行操作（关于广播，本章后续会详细介绍）。

### Ufuncs 的类型

Ufuncs 通常可以根据其接受的输入数组数量进行分类：

1. **一元 Ufuncs：** 操作单个输入数组。例子包括计算每个元素的平方根、指数、对数或三角函数值的函数。
2. **二元 Ufuncs：** 操作两个输入数组。例子包括逐元素加法、减法、乘法、除法、比较运算符和逻辑运算。

### 一元 Ufuncs 示例

我们来看一些常用的一元 ufuncs。考虑以下数组：

```python
import numpy as np

arr = np.arange(1, 6)
print(arr)
```

现在，我们来应用一些一元 ufuncs：

- **平方根（`np.sqrt`）：** 计算每个元素的非负平方根。

  ```python
  sqrt_arr = np.sqrt(arr)
  print(sqrt_arr)
  ```
- **指数（`np.exp`）：** 计算每个元素 xxx 的指数（exe^xex）。

  ```python
  exp_arr = np.exp(arr)
  print(exp_arr)
  ```
- **自然对数（`np.log`）：** 计算每个元素的自然对数（底数为 eee）。

  ```python
  log_arr = np.log(arr)
  print(log_arr)
  ```
- **正弦（`np.sin`）：** 计算每个元素的正弦（假定元素以弧度为单位）。

  ```python
  sin_arr = np.sin(arr)
  print(sin_arr)
  ```

### 二元 Ufuncs 和算术运算符

您实际上已经隐式地使用了二元 ufuncs！标准的算术运算符（`+`、`-`、`*`、`/`、`**`）对应于特定的 ufuncs：

- `arr1 + arr2` 等同于 `np.add(arr1, arr2)`
- `arr1 - arr2` 等同于 `np.subtract(arr1, arr2)`
- `arr1 * arr2` 等同于 `np.multiply(arr1, arr2)`
- `arr1 / arr2` 等同于 `np.divide(arr1, arr2)`
- `arr1 ** arr2` 等同于 `np.power(arr1, arr2)`

我们来看一个 `np.add` 的例子：

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([10, 20, 30])

sum_arr_op = arr1 + arr2
sum_arr_ufunc = np.add(arr1, arr2)

print(f"使用 + 运算符: {sum_arr_op}")

print(f"使用 np.add ufunc: {sum_arr_ufunc}")
```

如您所见，结果是相同的。对于简单的算术运算，使用运算符通常更易读，但了解底层的 ufunc（在此例中是 `np.add`）会很有用。其他二元 ufuncs 包括 `np.maximum`、`np.minimum`、`np.mod`、`np.copysign`、比较函数（如 `np.greater`、`np.less_equal` 等）和逻辑函数（如 `np.logical_and`、`np.logical_or`）。

Ufuncs 是 NumPy 性能和易用性的重要组成部分。它们提供了一个全面的优化函数库，用于对数组进行逐元素操作，为在数据分析和科学计算中进行的许多数值计算提供了支持。理解它们的工作方式对于编写高效的 NumPy 代码来说非常重要。

获取即时帮助、个性化解释和交互式代码示例。

---

### 数学和统计函数

# 数学和统计函数

在处理NumPy数组时，通常需要对数据进行汇总。你通常不需要查看每个单独的值，而是需要一个数字来表示数组的某个属性，例如总和、平均值或最大值与最小值之间的范围。NumPy提供了一系列专门用于这些数学和统计聚合的优化函数。

### 基本聚合函数

我们从一些基础聚合函数开始。这些函数接受一个数组作为输入，并返回一个汇总数组内容的单一值。考虑一个简单的一维数组：

```python
import numpy as np

arr1d = np.arange(1, 10)
print(f"Original array: {arr1d}")
```

我们可以轻松计算其总和、最小值、最大值和平均值：

```python
print(f"Sum: {np.sum(arr1d)}")

print(f"Minimum: {np.min(arr1d)}")

print(f"Maximum: {np.max(arr1d)}")

print(f"Mean: {np.mean(arr1d)}")

print(f"Standard Deviation: {np.std(arr1d)}")

print(f"Variance: {np.var(arr1d)}")
```

请注意，这些函数将整个数组归约为一个单一的标量值。许多这些聚合函数也可以作为数组对象上的方法直接使用，这有时能让代码稍微更易读：

```python
print(f"Sum (method): {arr1d.sum()}")

print(f"Mean (method): {arr1d.mean()}")
```

`np.sum(arr1d)` 和 `arr1d.sum()` 都实现相同的效果。两者之间的选择通常取决于个人偏好或编码风格。使用方法形式（`arr.sum()`）非常常见。

### 沿轴操作

这些函数的实际用处在处理多维数组时变得清晰。聚合可以对整个数组进行（如上所示，得到一个单一值），也可以沿着特定的*轴*进行。

请记住，对于二维数组（如矩阵）：

- `axis=0` 指的是*沿列*执行的操作。压缩行。
- `axis=1` 指的是*沿行*执行的操作。压缩列。

这乍一看可能有些反直觉。把它理解为被“压缩”或聚合的轴。如果你沿着 `axis=0` 求和，你会压缩行以获得每列的总和。如果你沿着 `axis=1` 求和，你会压缩列以获得每行的总和。

让我们通过一个二维数组来看一下：

```python
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print("原始二维数组:")
print(arr2d)

print(f"\nSum of all elements: {arr2d.sum()}")

print(f"Sum along axis 0 (columns): {arr2d.sum(axis=0)}")

print(f"Sum along axis 1 (rows): {arr2d.sum(axis=1)}")

print(f"Mean along axis 0 (columns): {arr2d.mean(axis=0)}")

print(f"Mean along axis 1 (rows): {arr2d.mean(axis=1)}")
```

指定 `axis` 参数 (parameter)是进行计算的基础方法，例如在用NumPy数组表示的成绩矩阵中，查找每个作业的平均分数（跨学生，需要沿 `axis=0` 聚合）或者每个学生的平均分数（跨作业，需要沿 `axis=1` 聚合）。

这是示例中列和（`axis=0`）的可视化效果：

列 0列 1列 2051015

> 条形图显示了沿着 `axis=0` 计算的总和。第一个条形代表第一列的总和（1+4+7=12），第二个条形代表第二列的总和（2+5+8=15），第三个条形代表第三列的总和（3+6+9=18）。

### 其他实用函数

除了基础聚合之外，NumPy 还提供其他有助于数组分析的函数：

- `argmin()` 和 `argmax()`: 这些函数不返回最小值或最大值，而是返回最小值或最大值的*索引*（位置）。这对于查找数据中特定特征出现的位置很有用。
- `cumsum()`: 计算元素的累积和。输出数组中的每个元素都是输入数组中沿指定轴的所有先前元素（包括其自身）到该点的总和。
- `cumprod()`: 类似 `cumsum`，计算累积乘积。

以下是使用一维数组的快速示例：

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

print(f"Array: {arr}")

print(f"Index of minimum (argmin): {np.argmin(arr)}")

print(f"Index of maximum (argmax): {np.argmax(arr)}")

print(f"Cumulative sum (cumsum): {np.cumsum(arr)}")

arr_prod = np.array([1, 2, 3, 4])
print(f"\nArray for product: {arr_prod}")

print(f"Cumulative product (cumprod): {np.cumprod(arr_prod)}")
```

这些函数，像基础聚合函数一样，也支持多维数组的 `axis` 参数 (parameter)，让你可以计算诸如列的累计总和或行的累计总和等。

### 处理缺失值：NaN安全聚合

"实际数据通常包含缺失值。在 NumPy（以及随后的 Pandas）中，缺失的浮点值通常由一个特殊值表示：`np.nan`（非数字）。标准聚合函数通常会传播 `NaN` 值。如果聚合中涉及的某个元素是 `NaN`，结果也通常会是 `NaN`。"

```python
arr_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
print(f"Array with NaN: {arr_nan}")

print(f"Sum with NaN using np.sum: {np.sum(arr_nan)}")

print(f"Mean with NaN using np.mean: {np.mean(arr_nan)}")
```

这种行为可能并非总是合乎预期。你通常希望在计算时忽略缺失值。为此，NumPy 提供了一组 `NaN` 安全函数，通常以 `nan` 为前缀：

- `np.nansum()`
- `np.nanmean()`
- `np.nanmin()`
- `np.nanmax()`
- `np.nanstd()`
- `np.nanvar()`

这些函数在计算结果时，会将 `NaN` 值视为数组中不存在的元素。

```python

print(f"Array with NaN: {arr_nan}")

print(f"Sum ignoring NaN using np.nansum: {np.nansum(arr_nan)}")

print(f"Mean ignoring NaN using np.nanmean: {np.nanmean(arr_nan)}")

print(f"Max ignoring NaN using np.nanmax: {np.nanmax(arr_nan)}")
```

使用这些 `NaN` 安全函数是处理预期存在缺失值的数据集时的重要做法，可确保你的汇总统计数据准确反映可用数据。

NumPy 的数学和统计函数提供了高效的工具，用于汇总数组中的数据。你可以计算整个数组的简单聚合值，如总和和平均值，或者在多维数组中沿着特定轴（行或列）执行这些计算。像 `argmin`、`argmax` 和 `cumsum` 这样的函数提供了额外的方法来分析数组内容。重要的是，NumPy 提供了许多聚合函数的 `NaN` 安全版本，让你可以在计算过程中妥善处理缺失数据。这些功能是使用 NumPy 执行许多数据分析任务的核心。

获取即时帮助、个性化解释和交互式代码示例。

---

### 数组上的逻辑运算

# 数组上的逻辑运算

NumPy 数组支持逻辑比较，从而能够在数据分析工作流程中实现高效的数据筛选、条件检查和决策制定。这些逻辑运算通常按元素应用，与 NumPy 中的算术运算非常相似，并生成布尔值（`True` 或 `False`）的数组。

### 逐元素比较

您可以使用标准 Python 比较运算符将整个数组与标量值进行比较，或者逐元素比较两个数组。NumPy 重载了这些运算符，使其可以在数组上高效工作。

可用的比较运算符有：

- `==` (等于)
- `!=` (不等于)
- `<` (小于)
- `>` (大于)
- `<=` (小于或等于)
- `>=` (大于或等于)

我们来看看它们是如何工作的。

**数组与标量比较**

当您将 NumPy 数组与单个数字（一个标量）进行比较时，NumPy 会在标量与数组的*每个*元素之间执行比较，并返回一个相同形状的新布尔数组。

```python
import numpy as np

ages = np.array([25, 18, 65, 40, 12])

is_adult = ages > 21
print(is_adult)

is_eighteen = ages == 18
print(is_eighteen)
```

结果 `is_adult` 是一个布尔数组，在满足条件 `age > 21` 的地方为 `True`，否则为 `False`。

**比较两个数组**

您也可以比较两个数组，前提是它们具有兼容的形状（要么形状相同，要么根据广播规则兼容的形状，我们稍后会讨论）。比较是逐元素执行的。

```python

scores1 = np.array([85, 92, 78, 88])
scores2 = np.array([85, 90, 80, 88])

equal_scores = scores1 == scores2
print(equal_scores)

lower_scores = scores1 < scores2
print(lower_scores)
```

结果布尔数组中的每个元素都对应于原始数组中相同位置元素的比较结果。

### 组合逻辑条件

通常，您需要同时检查多个条件。例如，您可能希望找到大于下限*且*小于上限的元素。NumPy 为此使用了逐元素的逻辑运算符：

- `&` (逻辑与)
- `|` (逻辑或)
- `~` (逻辑非)

> **重要提示：** 在 NumPy 数组上进行逐元素逻辑运算时，请使用位运算符 `&`、`|` 和 `~`，而*不是* Python 关键字 `and`、`or` 和 `not`。关键字 `and` 和 `or` 评估整个对象的真值，而 `&` 和 `|` 则对数组内的布尔值执行逐元素比较。另外，请记住由于运算符优先级，要在单个比较周围使用括号 `()`。

我们来结合之前 `ages` 数组的条件：

```python

working_age = (ages > 20) & (ages <= 50)
print(working_age)

young_or_senior = (ages < 18) | (ages > 60)
print(young_or_senior)

not_eighteen = ~(ages == 18)

print(not_eighteen)
```

这些组合的布尔数组对于选择满足多个条件的数据非常有用，这是一种通常被称为布尔索引的技巧（您在第三章中遇到过）。

### 检查数组中的条件：`any()` 和 `all()`

有时，您不需要逐元素的结果，而是需要一个概要：*任何*元素是否满足条件，或者*所有*元素都满足条件？NumPy 为此提供了数组的 `any()` 和 `all()` 方法（以及独立函数 `np.any()`、`np.all()`）。

- `any()`: 如果布尔数组中至少有一个元素为 `True`，则返回 `True`。
- `all()`: 仅当布尔数组中*所有*元素都为 `True` 时，才返回 `True`。

```python

results = np.array([True, False, True, True])

print(results.any())

print(results.all())

ages = np.array([25, 18, 65, 40, 12])

print((ages < 20).any())

print((ages >= 18).all())
```

这些函数有助于快速检查和断言您的数据。您也可以在多维数组中沿着特定轴应用 `any()` 和 `all()`，类似于 `sum()` 或 `mean()` 等统计函数。

逻辑运算是数据选择和操作的核心部分。它们生成的布尔数组充当强大的掩码，用于根据复杂条件筛选和修改您的数据集。当您处理更大的数据集时，使用 NumPy 的向量 (vector)化逻辑运算将比编写显式 Python 循环效率显著提高。

获取即时帮助、个性化解释和交互式代码示例。

---

### 广播功能简介

# 广播功能简介

NumPy 可以直接在*相同*大小和形状的数组之间执行操作，例如将两个 3×33 \times 33×3 矩阵相加。但是，当需要对*不同*形状的数组执行操作（例如加法）时，情况会怎样呢？例如，将一个单独的数字（一个标量）加到数组的每个元素上，或者将一个一维数组（一个向量 (vector)）加到二维数组（一个矩阵）的每一行上。

手动操作时，你可能会想到编写循环来执行这些操作。然而，NumPy 提供了一种更高效、更简洁的机制，称为**广播（broadcasting）**。广播描述了 NumPy 用来处理不同形状数组之间算术运算的一套规则，它有效地“拉伸”或复制较小的数组，使其形状与较大的数组匹配，而无需实际使用额外的内存。这使得向量化 (quantization)操作成为可能，它们比显式的 Python 循环快得多。

### 广播的规则

如果数组满足特定的兼容性条件，广播机制就使得在不同形状数组之间进行操作成为可能。NumPy 从末尾（最右边）的维度开始，逐元素比较数组的形状。两个维度兼容的条件是：

1. 它们相等，或者
2. 任一维度为 1。

如果这些条件对于任何维度对都不满足，就会引发 `ValueError: operands could not be broadcast together` 错误。

让我们分解一下 NumPy 如何应用这些规则：

1. **对齐 (alignment)维度：** 如果两个数组的维度数量不同，NumPy 会在较小数组的形状前添加 1，直到它们具有相同数量的维度。例如，将形状为 (3, 4) 的二维数组与形状为 (4,) 的一维数组进行比较时，一维数组的形状会被视为 (1, 4)。
2. **检查兼容性并拉伸：** 然后，NumPy 会从最右边的维度开始，比较每个维度的大小。
   - 如果维度大小匹配，则继续检查下一个维度。

- 如果有一个维度大小是 1，NumPy 会“拉伸”或“广播”该维度，以匹配另一个数组中对应维度的大小。其效果就好像大小为 1 的维度上的数据被复制以匹配更大的大小。
  - 如果维度大小不匹配，并且两个维度都不是 1，则无法进行广播，NumPy 会引发错误。

一旦形状兼容并完成了广播，NumPy 就会执行逐元素操作。

### 广播示例

我们来看一下广播的实际作用。

#### 示例 1：标量与数组

最简单的例子是数组与标量值之间的操作。标量被视为零维数组。

```python
import numpy as np

arr = np.array([1, 2, 3])
scalar = 5

result = arr + scalar
print(f"数组：\n{arr}")
print(f"形状：{arr.shape}\n")
print(f"标量：{scalar}\n")
print(f"结果 (arr + scalar)：\n{result}")
print(f"形状：{result.shape}")
```

在这里，标量 `5` 被有效地广播到数组 `arr` 的所有元素上。遵循规则：

1. `arr` 的形状是 (3,)，`scalar` 实际的形状是 ()。NumPy 为标量添加维度以匹配 `arr` 的维度数量：形状变为 (1,)。等等，标量是 0 维的。数组是 1 维的（形状 (3,)）。标量的形状被视为通过重复其值来匹配任何数组的形状。该操作将标量视为一个与 `arr` 形状相同的数组 `[5, 5, 5]`。

#### 示例 2：一维数组与二维数组

我们考虑将一个一维数组加到二维数组的每一行上。

```python
matrix = np.arange(6).reshape((2, 3))
row_vector = np.array([10, 20, 30])

result = matrix + row_vector
print(f"矩阵 (形状 {matrix.shape})：\n{matrix}\n")
print(f"行向量 (形状 {row_vector.shape})：\n{row_vector}\n")
print(f"结果 (matrix + row_vector) (形状 {result.shape})：\n{result}")
```

我们来追踪一下广播规则：

1. `matrix` 形状：(2, 3)。`row_vector` 形状：(3,)。
2. 对齐 (alignment)维度：NumPy 会在 `row_vector` 的形状前添加 1。它变为 (1, 3)。
3. 从右到左比较维度：
   - 末尾维度：3（来自矩阵）和 3（来自向量 (vector)）。它们匹配。
   - 下一个维度：2（来自矩阵）和 1（来自向量）。有一个是 1，因此兼容。
4. 拉伸：NumPy 将向量的第一个维度从 1 拉伸到 2。`row_vector` 实际上变为 `[[10, 20, 30], [10, 20, 30]]`。
5. 执行逐元素加法。

以下图表说明了此过程：

G

clusterₘatrix

矩阵 (2, 3)

clusterᵥector

向量 (3,) -> (1, 3)

cluster\_broadcastedᵥector

广播后的向量 (2, 3)

clusterᵣesult

结果 (2, 3)

m

0

1

2

3

4

5

r

10

21

32

13

24

35

m->r

+

v

10

20

30

bv

10

20

30

10

20

30

v->bv

广播

bv->r

+

caption
沿轴 0 拉伸

> 一维数组 `[10, 20, 30]`（形状 (3,)）首先被视为形状 (1, 3)，然后沿第一个轴广播（拉伸），以匹配矩阵的形状 (2, 3) 进行逐元素加法。灰色行表示数据的复制。

#### 示例 3：列向量与行向量

广播还可以组合数组以生成更高维度的结果。我们来将一个列向量加到一个行向量上。

```python
col_vector = np.array([[0], [10], [20]])
row_vector = np.array([1, 2, 3])

result = col_vector + row_vector
print(f"列向量 (形状 {col_vector.shape})：\n{col_vector}\n")
print(f"行向量 (形状 {row_vector.shape})：\n{row_vector}\n")
print(f"结果 (col + row) (形状 {result.shape})：\n{result}")
```

我们来追踪一下这个过程：

1. `col_vector` 形状：(3, 1)。`row_vector` 形状：(3,)。
2. 对齐维度：NumPy 将 `row_vector` 视为形状 (1, 3)。
3. 从右到左比较维度 (3, 1) 与 (1, 3)：
   - 末尾维度：1（来自列）和 3（来自行）。有一个是 1，兼容。
   - 下一个维度：3（来自列）和 1（来自行）。有一个是 1，兼容。
4. 拉伸：

- `col_vector` 大小为 1 的维度被拉伸到 3。它变为形状 (3, 3)。
- `row_vector` 大小为 1 的维度被拉伸到 3。它也变为形状 (3, 3)。

5. 对广播后的 (3, 3) 版本执行逐元素加法。

### 广播失败的情况

广播只有在形状根据规则兼容的情况下才有效。如果在任何时候维度大小不同且都不是 1，NumPy 将无法解决这种不明确性，并会引发错误。

```python
arr1 = np.arange(6).reshape((2, 3))
arr2 = np.array([1, 2])

try:
    result = arr1 + arr2
except ValueError as e:
    print(f"错误：{e}")
```

这里，`arr1` 是 (2, 3)，`arr2` 是 (2,)。NumPy 将 `arr2` 视为 (1, 2)。比较 (2, 3) 与 (1, 2)：

- 末尾维度：3 与 2。它们不同，且都不是 1。广播失败。

为了使其正常工作，`arr2` 需要具有形状 (3,) 或 (2, 1) 或 (1, 3)，具体取决于预期的操作。

广播是 NumPy 中一个基本机制，它通过避免显式的 Python 循环，让你能编写更清晰、更简洁、明显更快的代码，用于对兼容但不同形状的数组执行操作。了解其规则对于使用 NumPy 进行高效的数值编程非常重要。

获取即时帮助、个性化解释和交互式代码示例。

---

### 动手实践：对数组进行计算

# 动手实践：对数组进行计算

通过例子展示算术运算、通用函数、统计计算、逻辑运算以及广播在NumPy数组上的应用。请确保已导入 NumPy，通常使用 `import numpy as np`。

### 准备工作

首先，我们创建一些数组来操作。我们将使用简单的数组，以便于理解计算过程。

```python
import numpy as np

temps_loc_a = np.array([15.0, 17.5, 18.0, 16.5, 19.0])
temps_loc_b = np.array([12.0, 14.0, 13.5, 15.0, 16.0])

sensor_readings = np.array([[1.1, 1.2, 1.0, 1.3],
                           [2.0, 2.2, 2.1, 1.9],
                           [0.8, 0.7, 0.9, 0.8]])

print("地点A的温度:", temps_loc_a)
print("地点B的温度:", temps_loc_b)
print("传感器读数:\n", sensor_readings)
```

### 1. 算术运算和通用函数 (ufuncs)

让我们执行一些基本计算。

**a) 元素级加法:** 计算每天两个地点的平均温度。

```python

daily_sum = temps_loc_a + temps_loc_b

daily_avg = daily_sum / 2.0

print("每日总和:", daily_sum)
print("每日平均温度:", daily_avg)
```

请注意，加法 `+` 和除法 `/` 运算是如何逐元素应用的。

**b) 温度转换:** 将地点A的温度从摄氏度转换为华氏度。公式是 F=(C×9/5)+32F = (C \times 9/5) + 32F=(C×9/5)+32。

```python
temps_fahrenheit_a = (temps_loc_a * 9/5) + 32
print("地点A的温度 (华氏度):", temps_fahrenheit_a)
```

同样，乘法 `*` 和加法 `+` 会自动应用于每个元素。

**c) 使用通用函数 (ufunc):** 计算每个传感器读数的平方根（可能用于数据转换）。我们使用 `np.sqrt()`。

```python
sqrt_readings = np.sqrt(sensor_readings)
print("传感器读数的平方根:\n", sqrt_readings)
```

NumPy 的 `np.sqrt` 函数对整个数组进行高效操作。

### 2. 数学和统计函数

现在我们来计算一些汇总统计量。

**a) 总体统计:** 找出地点A记录的总体最低、最高和平均温度。

```python
min_temp_a = np.min(temps_loc_a)
max_temp_a = np.max(temps_loc_a)
avg_temp_a = np.mean(temps_loc_a)
std_dev_temp_a = np.std(temps_loc_a)

print(f"地点A - 最低温度: {min_temp_a:.2f} C")
print(f"地点A - 最高温度: {max_temp_a:.2f} C")
print(f"地点A - 平均温度: {avg_temp_a:.2f} C")
print(f"地点A - 标准差: {std_dev_temp_a:.2f} C")
```

**b) 沿轴统计:** 从 `sensor_readings` 数组中计算每个传感器的平均读数（跨时间）和每个时间点的平均读数（跨传感器）。

请记住：

- `axis=0` 沿行操作（为每列计算统计量）。
- `axis=1` 沿列操作（为每行计算统计量）。

```python

avg_reading_per_timepoint = np.mean(sensor_readings, axis=0)

avg_reading_per_sensor = np.mean(sensor_readings, axis=1)

print("每个时间点的平均读数:", avg_reading_per_timepoint)
print("每个传感器的平均读数:", avg_reading_per_sensor)
```

我们可以使用简单的条形图来可视化每个传感器的平均读数。

传感器 1传感器 2传感器 300.511.52

> 为每个传感器在所有时间点计算的平均电压读数。

### 3. 逻辑运算

让我们找出地点A温度高于17摄氏度的日期。

```python

hot_days_a = temps_loc_a > 17.0
print("地点A温度高于17摄氏度的日期:", hot_days_a)

hot_temps_a = temps_loc_a[hot_days_a]

print("地点A高温日期的温度:", hot_temps_a)

num_hot_days = np.sum(hot_days_a)
print("温度高于17摄氏度的天数:", num_hot_days)
```

比较表达式 `temps_loc_a > 17.0` 返回一个布尔数组 (`[False, True, True, False, True]`)。然后，此数组用作索引，只选择 `temps_loc_a` 中布尔数组对应值为 `True` 的元素。

### 4. 广播

如果形状兼容，广播允许在不同形状的数组之间进行操作。

**a) 添加标量:** 将一个常数调整值（例如 0.5）添加到所有传感器读数。

```python
adjusted_readings = sensor_readings + 0.5
print("调整后的传感器读数:\n", adjusted_readings)
```

在这里，标量 `0.5` 被有效地“拉伸”或广播，以匹配 `sensor_readings` 的形状，然后进行加法运算。

**b) 与一维数组操作:** 假设我们为每个时间点设置了一个基线值（可能是前几天的平均值），并希望查看传感器1当前读数与此基线值的差异。

```python
baseline_per_timepoint = np.array([1.0, 1.1, 0.9, 1.2])
sensor1_readings = sensor_readings[0, :]

difference_sensor1 = sensor1_readings - baseline_per_timepoint
print("传感器1读数:", sensor1_readings)
print("每个时间点的基线值:", baseline_per_timepoint)
print("传感器1的差异:", difference_sensor1)
```

`sensor1_readings` 和 `baseline_per_timepoint` 都是相同大小（4个元素）的一维数组，因此逐元素相减可以直接进行。

**c) 将一维数组广播到二维数组:** 现在，让我们从*所有*传感器读数中减去 `baseline_per_timepoint`。

```python
difference_all_sensors = sensor_readings - baseline_per_timepoint
print("所有传感器与基线的差异:\n", difference_all_sensors)
```

这是如何工作的？`sensor_readings` 的形状是 `(3, 4)`，`baseline_per_timepoint` 的形状是 `(4,)`。NumPy 的广播规则从右到左比较维度：

1. 最后一个维度匹配（都为 4）。
2. `sensor_readings` 的下一个维度是 3，但 `baseline_per_timepoint` 没有更多维度了。
   NumPy 会沿着缺失的维度（本例中为行）“拉伸”或复制 `baseline_per_timepoint`，以有效地创建一个与 `sensor_readings` 匹配的 `(3, 4)` 数组。然后进行逐元素相减。就像 NumPy 执行了这样的减法：

```
[[1.1, 1.2, 1.0, 1.3],      [[1.0, 1.1, 0.9, 1.2],
 [2.0, 2.2, 2.1, 1.9],  -    [1.0, 1.1, 0.9, 1.2],
 [0.8, 0.7, 0.9, 0.8]]       [1.0, 1.1, 0.9, 1.2]]
```

### 总结

在本次动手实践中，你应用了本章介绍的基本数值运算：

- 执行了逐元素的**算术运算**，如加法和乘法。
- 使用了**通用函数** (ufuncs)，如 `np.sqrt`。
- 计算了整个数组和特定轴的**统计指标**（`np.mean`、`np.min`、`np.max`、`np.std`）。
- 应用了**逻辑运算**（`>`）来创建布尔数组，用于数据过滤（布尔索引）。
- 运用**广播**在兼容但形状不同的数组之间执行操作（标量与数组，一维与二维）。

这些操作是 NumPy 数值计算的核心，也是你将遇到的许多数据分析任务的根本。进一步尝试创建自己的数组，并尝试不同的函数和操作。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 5 Introduction To Pandas

### Pandas 是什么？

# Pandas 是什么？

强大的数组对象是 Python 中数值计算的基本构成要素。然而，数据分析通常不仅仅涉及原始数字。我们经常遇到以表格形式组织的数据，类似于电子表格或数据库表，它们带有描述性的行和列标签，不同列中可能包含不同数据类型，并且存在常见的缺失值问题。

**Pandas** 在此发挥作用。Pandas 是一个基于 NumPy 构建的开源 Python 库，专门用于**数据处理和分析**。它提供数据结构和操作，以便高效且直观地处理结构化数据。可以将其视为提供了您在电子表格软件或关系型数据库中可能找到的数据分析功能，但直接集成到您的 Python 环境中。

那么，是什么使得 Pandas 在数据任务中如此有用呢？

- **专门的数据结构：** Pandas 引入了两种主要的数据结构：`Series`（一维带标签数组）和 `DataFrame`（二维带标签数据结构，本质上是一个表格）。这些结构允许您将标签（行的索引，列的名称）与数据关联起来，使得操作比使用普通 NumPy 数组处理表格数据更加直观。我们很快会详细介绍这些结构。
  "\* **处理异构数据：** 与通常要求所有元素具有相同数据类型的 NumPy 数组不同，Pandas `DataFrame` 可以轻松处理不同数据类型的列（例如整数、浮点数、字符串、Python 对象）。这种灵活性符合大多数数据集的特点。"
- **缺失数据管理：** 数据很少是完美的。Pandas 提供方便的函数来检测、移除或替换数据集中的缺失数据点（通常表示为 `NaN`，非数字）。
- **带标签数据操作：** Pandas 的一个显著优势是它能够基于标签而非仅仅位置进行操作。在执行算术运算或组合数据集时，Pandas 会根据其行和列标签自动对齐 (alignment)数据，减少使用无标签数组时常见的错误风险。
- **高效输入/输出：** Pandas 包含工具，可轻松从各种文件格式（例如逗号分隔值 (CSV)、Excel 电子表格、JSON 文件、SQL 数据库等）读取数据，并将处理过的数据写回这些格式。
- **强大的数据处理工具：** 除了基本结构之外，Pandas 提供丰富的功能集，用于选择数据子集、根据条件筛选行、重塑表格、合并和连接多个数据集、对数据进行分组以进行聚合计算，以及处理时间序列数据。

本质上，Pandas 提供加载、清洗、转换、合并和分析结构化数据所需的高层工具。它在底层采用 NumPy 的计算效率，同时提供更具表现力且用户友好的界面，专为数据分析工作流程定制。在您后续学习中，您将看到 `Series` 和 `DataFrames` 如何成为处理数据的主力，然后数据可能会被输入到机器学习 (machine learning)模型或用于生成洞察。

获取即时帮助、个性化解释和交互式代码示例。

---

### Pandas 数据结构：Series

# Pandas 数据结构：Series

虽然像 NumPy 的 `ndarray` 这样强大的数组结构常用于数值计算，但许多数据不仅仅是原始数字；它们带有标签和结构。例如，随时间变化的股票价格、来自不同位置的传感器读数，或者调查受访者的人口统计信息。这就是 Pandas 发挥作用的地方，它用于一维数据的基本数据结构是 `Series`。

可以将 `Series` 想象成电子表格中的单个列，或者 Python 列表或 NumPy 数组的更高级版本。它本质上是一个一维的类数组对象，包含一系列值和一组相关的数据标签数组，称为其 **索引**。

### Series 的结构

Pandas `Series` 有两个主要组成部分：

1. **值：** 这是一系列数据点。在底层，这些值通常存储在 NumPy `ndarray` 中，这使得对其进行操作既快速又高效。Series 中的值通常具有相同的数据类型（如整数、浮点数、字符串或 Python 对象）。
2. **索引：** 这是一组与值对应的标签数组。与主要使用基于整数的索引（0, 1, 2, ...）的 NumPy 数组不同，Pandas Series 有一个*显式* 索引。此索引可以由整数组成，但也可以由字符串、日期或其他 Python 对象构成。如果您在创建 Series 时未指定索引，Pandas 会自动创建一个默认的整数索引，范围从 0 到 N−1N-1N−1，其中 NNN 是值的数量。

这里是一个简单的可视化表示：

G

clusterᵢndex

索引对象

clusterᵥalues

NumPy 数组（通常）

series

索引

值

idxₗabels

'Mon'

'Tue'

'Wed'

'Thu'

'Fri'

series:idx->idxₗabels

标签

val\_data

22.5

23.1

21.9

22.8

23.5

series:val->val\_data

数据

> Pandas Series 将值数组（通常是 NumPy 数组）与一个用于标记 (token)的显式索引对象结合起来。

### 带标签数据的重要性

显式索引是 Pandas Series 的一个重要特性。相比仅使用普通的 NumPy 数组，它提供了几个优势：

- **直观访问：** 您可以使用有意义的标签（如日期或类别名称）访问数据点，而不是仅仅通过整数位置。例如，获取“周三”的温度通常比记住它在索引位置 2 更直观。
- **数据对齐 (alignment)：** 在多个 Series 之间执行操作时，Pandas 会根据索引标签自动对齐数据。这可以防止在使用无序或不同顺序的数据时出现的许多常见错误。
- **灵活性：** 相比简单的整数索引，索引允许更复杂的选择和处理逻辑。

可以将 Series 看作是通过增加这一层有意义的标签来增强 NumPy 数组。它保留了 NumPy 对底层值的计算效率，同时提供了一种更灵活、内容更丰富的结构，适合数据分析。在下一节中，我们将介绍在 Python 中创建这些 `Series` 对象的实用方法。

获取即时帮助、个性化解释和交互式代码示例。

---

### 创建 Series

# 创建 Series

要创建 Pandas Series（一种带标签的一维数组），有几种常见方法可以使用。`pd.Series()` 构造函数是主要工具。

我们需要 Pandas 库，它通常使用别名 `pd` 导入。如果您还打算使用 NumPy 数组（它与 Pandas 配合得很好），则将其导入为 `np`。

```python
import pandas as pd
import numpy as np
```

### 从 Python 列表创建 Series

创建 Series 最直接的方法是使用标准的 Python 列表。如果您不指定索引，Pandas 会自动创建一个从 0 开始的默认整数索引。

```python

data_list = [10, 20, 30, 40, 50]

series_from_list = pd.Series(data_list)

print(series_from_list)
```

**输出：**

```
0    10
1    20
2    30
3    40
4    50
dtype: int64
```

注意输出显示了两列：左侧是索引（0 到 4），右侧是数据值。Pandas 还会推断数据类型（在此例中是 `dtype: int64`，表示 64 位整数）。

### 创建带自定义索引的 Series

虽然默认的整数索引很有用，但 Series 真正的用处在于它能够为索引使用有意义的标签。您可以在创建时使用 `index` 参数 (parameter)提供一个标签列表或数组。索引列表的长度必须与数据列表相同。

```python

data_list = [100, 200, 300, 400]

index_labels = ['alpha', 'beta', 'gamma', 'delta']

series_custom_index = pd.Series(data=data_list, index=index_labels)

print(series_custom_index)
```

**输出：**

```
alpha    100
beta     200
gamma    300
delta    400
dtype: int64
```

现在，索引不再是 0、1、2、3，而是由字符串标签 'alpha'、'beta'、'gamma' 和 'delta' 组成。这使得访问特定数据点更直观，我们将在后面的章节中看到。

### 从 NumPy 数组创建 Series

您可以同样轻松地从 NumPy 数组创建 Series。这非常普遍，因为数据通常源自使用 NumPy 进行的数值计算。这个过程与使用列表相同。

```python

numpy_array = np.array([5.5, 6.6, 7.7, 8.8])

series_from_numpy = pd.Series(numpy_array)

print(series_from_numpy)
```

**输出：**

```
0    5.5
1    6.6
2    7.7
3    8.8
dtype: float64
```

同样，Pandas 会创建默认的整数索引并推断数据类型（在此例中是 float64）。您也可以在从 NumPy 数组创建 Series 时提供自定义索引，就像使用列表一样。

### 从 Python 字典创建 Series

另一种方便的方法是直接从 Python 字典创建 Series。在这种情况下，Pandas 会使用字典键作为索引标签，字典值作为 Series 数据。Series 元素的顺序通常会遵循字典的插入顺序（对于 Python 3.7+）。

```python

data_dict = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}

series_from_dict = pd.Series(data_dict)

print(series_from_dict)
```

**输出：**

```
Ohio      35000
Texas     71000
Oregon    16000
Utah       5000
dtype: int64
```

当您的数据已按键值对形式组织时，这种方法特别有用。

您也可以在从字典创建 Series 时显式指定索引。如果提供的索引标签在字典中不存在对应的键，Pandas 会插入一个 `NaN`（非数字）值，这是 Pandas 表示缺失数据的标准方式。如果字典包含未出现在指定索引中的键，则这些键值对将被忽略。

```python

data_dict = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}

states = ['California', 'Ohio', 'Oregon', 'Texas']

series_explicit_index = pd.Series(data_dict, index=states)

print(series_explicit_index)
```

**输出：**

```
California        NaN
Ohio          35000.0
Oregon        16000.0
Texas         71000.0
dtype: float64
```

注意 'California' 具有 `NaN` 值，因为它不在 `data_dict` 中。此外，原始字典中的 'Utah' 被排除，因为它不在 `states` 索引列表中。`dtype` 变为 `float64` 是因为 `NaN` 被视为浮点值。

这些方法涵盖了实例化 Pandas Series 对象最常见的方式。当您处理数据时，您会经常发现自己从列表、字典或 NumPy 数组等现有数据结构创建 Series，作为您分析流程的第一步。

获取即时帮助、个性化解释和交互式代码示例。

---

### Pandas 数据结构：DataFrame

# Pandas 数据结构：DataFrame

Pandas `DataFrame` 是处理表格数据的核心数据结构，这些数据通常由行和列组成。虽然 Pandas 也提供了用于一维带标签数据的 `Series`，但 `DataFrame` 是管理多维数据集的重要工具，可说是 Pandas 中最主要的数据结构。它直接受到 R 编程语言中数据帧概念的启发。

可以将 `DataFrame` 看作一个普适的二维表格，类似于你在 Microsoft Excel 中使用的电子表格，或 SQL 数据库中的表。它旨在以结构化的方式存放数据，方便处理和分析。

以下是 DataFrame 的主要特点：

1. **二维：** 它同时具有行和列，形成网格状结构。
2. **带标签的轴：** 行和列都有标签。行标签统称为 `index`，列标签称为 `columns`。这使得根据这些标签而非仅仅整数位置来获取数据变得更直接。
3. **异构数据类型：** 与通常只存储单一数据类型的 NumPy 数组不同，DataFrame 中的列可以包含不同的数据类型（例如：整数、浮点数、字符串、布尔值、Python 对象）。
4. **大小可变：** 通常，在 DataFrame 创建后，你可以添加或移除列。添加或移除行也是可行的。
5. **与 Series 的关系：** 你可以将 `DataFrame` 理解为 `Series` 对象的字典或集合，其中每个 `Series` 代表一列。`DataFrame` 中的所有 `Series`（列）共享相同的索引（行标签）。

DataFrame

struct

列 'Name' (对象/字符串)

列 'Age' (整数)

列 'Score' (浮点数)

索引 0

Alice

25

85.5

索引 1

Bob

30

92.1

索引 2

Charlie

22

78.0

> Pandas DataFrame 的视图，显示行索引标签、列标签（可能包含数据类型）和数据网格。

尽管 `DataFrame` 内部使用 NumPy 数组来提升效率，但它提供了更灵活和富有表现力的接口来处理结构化数据。它在操作期间自动处理数据对齐 (alignment)，并提供精巧的方法进行索引、切片、重塑、合并和处理缺失信息。这使其成为数据清理、数据检查和分析任务中不可或缺的工具，这些任务在数据科学和 AI 工作流程中很常见。

接下来的章节将会说明如何从各种数据源创建这些多用途的 `DataFrame` 对象，以及如何开始了解其内容。

获取即时帮助、个性化解释和交互式代码示例。

---

### 创建 DataFrame

# 创建 DataFrame

Pandas `Series` 代表单列数据，而 `DataFrame` 则是您在多数表格数据分析中会使用的主要结构。可以把它看作是 Python 环境中的电子表格或 SQL 表。它是一个二维、大小可变、可能包含不同类型数据的表格结构，带有标签轴（行和列）。

我们来看看构建 `DataFrame` 的常见方法。通常，您可以从现有数据结构（如 Python 字典或 NumPy 数组）创建它们，或者通过读取文件中的数据来创建（这将在下一章中介绍）。

### 从列表或 NumPy 数组的字典创建

最常用的方法之一是使用 Python 字典，其中键代表所需的列名，值是包含每列数据的列表（或 NumPy 数组）。重要的是，用作值的所有列表或数组必须具有相同的长度，因为每个列表对应一列，并且列表在相同位置的元素构成一行。

```python
import pandas as pd
import numpy as np

data = {
    'StudentID': ['S001', 'S002', 'S003', 'S004'],
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Score': [85, 92, 78, 88]
}

df_from_dict_list = pd.DataFrame(data)

print(df_from_dict_list)
```

此代码会生成以下 `DataFrame`：

```
  StudentID     Name  Score
0      S001    Alice     85
1      S002      Bob     92
2      S003  Charlie     78
3      S004    David     88
```

请注意 Pandas 如何自动为行分配了默认的整数索引（0, 1, 2, 3）。字典的键 (`StudentID`、`Name`、`Score`) 成为了列标签。

您可以使用 NumPy 数组作为字典的值来达到同样的效果：

```python

data_np = {
    'StudentID': np.array(['S001', 'S002', 'S003', 'S004']),
    'Name': np.array(['Alice', 'Bob', 'Charlie', 'David']),
    'Score': np.array([85, 92, 78, 88])
}

df_from_dict_np = pd.DataFrame(data_np)

print(df_from_dict_np)
```

### 从字典列表创建

另一种常见模式是使用一个列表，其中每个元素都是一个字典。在这种情况下，每个字典代表结果 `DataFrame` 中的一*行*。字典中的键成为列标签。Pandas 足够灵活，可以处理某些字典可能缺少键的情况；它会用 `NaN`（非数字）填充这些位置，`NaN` 是 Pandas 表示缺失数据的默认标记 (token)。

```python

list_of_dicts = [
    {'StudentID': 'S001', 'Name': 'Alice', 'Score': 85},
    {'StudentID': 'S002', 'Name': 'Bob', 'Age': 21},
    {'StudentID': 'S003', 'Name': 'Charlie', 'Score': 78, 'Age': 22},
    {'StudentID': 'S004', 'Name': 'David', 'Score': 88}
]

df_from_list_dict = pd.DataFrame(list_of_dicts)

print(df_from_list_dict)
```

输出显示了 Pandas 如何处理不一致的键：

```
  StudentID     Name  Score   Age
0      S001    Alice   85.0   NaN
1      S002      Bob    NaN  21.0
2      S003  Charlie   78.0  22.0
3      S004    David   88.0   NaN
```

Pandas 从字典中存在的键推断出所有可能的列名 (`StudentID`、`Name`、`Score`、`Age`)。当特定行（字典）缺少某个键时，会插入 `NaN`。此外，请注意 `Score` 和 `Age` 列被自动分配了浮点数据类型 (`float64`)，因为 `NaN` 在技术上是一个浮点值。

### 从 NumPy 数组创建

如果您的数据已存在为二维 NumPy 数组，您可以直接将其转换为 `DataFrame`。默认情况下，Pandas 会为列和行（索引）都分配整数标签。但是，您可以使用 `columns` 和 `index` 参数 (parameter)明确提供标签。

```python

np_array = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

df_from_np_default = pd.DataFrame(np_array)
print("带有默认标签的 DataFrame:")
print(df_from_np_default)

df_from_np_custom = pd.DataFrame(
    np_array,
    columns=['Col_A', 'Col_B', 'Col_C'],
    index=['Row_X', 'Row_Y', 'Row_Z']
)
print("\n带有自定义标签的 DataFrame:")
print(df_from_np_custom)
```

输出:

```
DataFrame with default labels:
   0  1  2
0  1  2  3
1  4  5  6
2  7  8  9

DataFrame with custom labels:
       Col_A  Col_B  Col_C
Row_X      1      2      3
Row_Y      4      5      6
Row_Z      7      8      9
```

DataFrame\_Structure

cluster\_df

DataFrame

k1
列A

l1
[1, 4, 7]

df

列A

列B

0

1

2

1

4

5

2

7

8

k1->df

 映射到
 列

k2
列B

l2
[2, 5, 8]

k2->df

l1->df

 映射到
 列

l2->df

> 图示展示了字典键和列表如何映射到 Pandas DataFrame 中的列和数据。如果未指定，行索引通常会自动生成。

### 从 Series 字典创建

您也可以从一个字典构建 `DataFrame`，其中值是 Pandas `Series` 对象。这类似于使用列表字典，数据根据 Series 的索引标签进行对齐 (alignment)。

```python

s1 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
s2 = pd.Series([15, 25, 35, 45], index=['a', 'b', 'c', 'd'])

df_from_series = pd.DataFrame({'Col1': s1, 'Col2': s2})

print(df_from_series)
```

输出:

```
   Col1  Col2
a  10.0    15
b  20.0    25
c  30.0    35
d   NaN    45
```

Pandas 会根据索引标签（'a'、'b'、'c'、'd'）自动对齐数据。由于 `s1` 没有索引 'd'，因此在 `Col1` 中为该行引入了 `NaN`。

这些方法提供了灵活的途径来程序化地创建 DataFrame。一旦您有了 DataFrame，下一步通常是检查其属性和内容，这我们将在后续内容中介绍。请记住，直接从 CSV 或 Excel 等文件加载数据也是将数据导入 DataFrame 的一种非常常见的方式，这将在第 6 章中讨论。

获取即时帮助、个性化解释和交互式代码示例。

---

### 检查 DataFrames

# 检查 DataFrames

在进行任何数据操作或分析之前，了解 Pandas DataFrame 的结构和内容是首要步骤。必须了解 DataFrame 的大小、每列的数据类型以及是否存在缺失值。Pandas 提供了多种便捷的属性和方法，可以快速检查 DataFrame 并回答这些基础问题。

### 获取快速预览: head() 和 tail()

通常，您不需要一次性查看整个数据集，特别是当它有成千上万或数百万行时。`head()` 和 `tail()` 方法非常适合快速查看数据的结构和内容。

- `head()`: 返回 DataFrame 的前 *n* 行。默认情况下，它返回前 5 行。
- `tail()`: 返回 DataFrame 的后 *n* 行。默认情况下，它返回后 5 行。

```python

print(df.head())

print(df.head(n=3))

print(df.tail())

print(df.tail(n=2))
```

使用 `head()` 和 `tail()` 对于验证数据是否正确加载以及初步了解列名及其包含的值的类型很有用。

### 检查维度: shape

要准确查明您的 DataFrame 有多少行和列，请使用 `shape` 属性。它不需要括号 `()`，因为它是一个属性（对象的特征）而不是一个方法（一个动作）。

```python

dimensions = df.shape
print(f"DataFrame 维度: {dimensions}")
print(f"行数: {dimensions[0]}")
print(f"列数: {dimensions[1]}")
```

`shape` 属性返回一个元组，其中第一个元素是行数，第二个是列数。了解维度对于理解数据集的规模是基本的。

### 了解数据类型和非空计数: info()

`info()` 方法提供 DataFrame 的简洁概览。这是最有用的检查方法之一。它会告知您：

- 总行数（条目数）。
- 索引的类型。
- 每列的名称。
- 每列中非空（非缺失）值的数量。
- 每列的数据类型 (`dtype`)。
- DataFrame 的大致内存使用情况。

```python

df.info()
```

运行 `df.info()` 可能会产生如下输出：

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
---  ------        --------------  -----
 0   sepal_length  150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  148 non-null    float64
 3   petal_width   150 non-null    float64
 4   species       150 non-null    object
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
```

仔细查看“非空计数”列。将此计数与总条目数（本例中为 150）进行比较，是快速识别包含缺失数据的列（例如 `petal_length` 有 148 个非空值，表示有 2 个缺失值）的方法。它还清楚地显示了 Pandas 为每列推断的数据类型（`float64` 用于带小数的数字，`object` 通常用于字符串）。

### 获取摘要统计信息: describe()

对于包含数值数据的列，`describe()` 方法会计算几种常用摘要统计信息。这为您提供了数据分布的快速量化 (quantization)概览。

```python

summary_stats = df.describe()
print(summary_stats)
```

输出通常包括：

- `count`: 非空值的数量。
- `mean`: 平均值。
- `std`: 标准差，衡量数据的分散程度。
- `min`: 最小值。
- `25%`: 第一四分位数（百分位数）。
- `50%`: 中位数或第二四分位数。
- `75%`: 第三四分位数。
- `max`: 最大值。

```
       sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    148.000000   150.000000
mean       5.843333     3.057333      3.758108     1.199333
std        0.828066     0.435866      1.765298     0.762238
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
```

默认情况下，`describe()` 只包含数值列。您可以修改其行为以包含其他类型：

- `df.describe(include='object')`: 显示 `object` 数据类型（通常是字符串）列的摘要统计信息。这包括计数（`count`）、唯一值的数量（`unique`）、最常出现的值（`top`）及其频率（`freq`）。
- `df.describe(include='all')`: 尝试包含所有列的统计信息，在适当的情况下混合数值和对象摘要。

### 列出列名: columns

要仅获取列的名称，请使用 `columns` 属性。

```python

column_names = df.columns
print(column_names)
```

这会返回一个包含列名的 `Index` 对象。如果您需要检查列名的确切拼写或以编程方式遍历列，这会很有用。

```
Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'], dtype='object')
```

### 列出索引标签: index

类似地，`index` 属性会为您提供行的标签。

```python

row_labels = df.index
print(row_labels)
```

这会返回另一个 `Index` 对象。默认情况下，DataFrame 会获得一个 `RangeIndex`（从 0 开始的整数），但索引也可以是其他类型，例如日期或特定标签，您将在以后遇到。

```
RangeIndex(start=0, stop=150, step=1)
```

这些检查工具（`head`、`tail`、`shape`、`info`、`describe`、`columns`、`index`）是您处理新数据集时的首选工具。它们帮助您快速定位，理解基本结构和内容，并在进行更复杂的数据清理、转换或分析任务之前，识别潜在问题，例如缺失数据或不正确的数据类型。请养成在创建或加载 DataFrame 时使用它们的习惯。

获取即时帮助、个性化解释和交互式代码示例。

---

### 实操：创建和查看 Series/DataFrame

# 实操：创建和查看 Series/DataFrame

本实践旨在巩固您对 Pandas Series 和 DataFrame 的理解。您将通过不同方法创建这些基本数据结构，然后运用多种技术检查它们，从而了解它们的内容和属性。假定您在 Jupyter Notebook 环境中操作。

首先，请确保您已导入 Pandas。通常将其导入并取别名为 `pd`。我们还将导入 NumPy 并取别名为 `np`，因为 Pandas 经常与它配合使用。

```python
import pandas as pd
import numpy as np

print("Pandas 版本:", pd.__version__)
print("NumPy 版本:", np.__version__)
```

### 创建和查看 Pandas Series

Series 类似于带标签的一维数组。我们来创建几个。

**1. 从 Python 列表创建：**
如果您不指定，Pandas 会自动创建一个默认的整数索引。

```python

data_list = [10, 20, 30, 40, 50]
series_from_list = pd.Series(data_list)

print("从列表创建的 Series:")
print(series_from_list)

print("\n第一个元素:", series_from_list[0])
```

**2. 从带有自定义索引的 Python 列表创建：**
您可以通过分配标签来为索引提供含义。

```python

temperatures = [22.5, 24.1, 19.8, 23.0]

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday']

series_with_index = pd.Series(temperatures, index=days)

print("\n带有自定义索引的 Series:")
print(series_with_index)

print("\n星期二的温度:", series_with_index['Tuesday'])
```

**3. 从 Python 字典创建：**
字典自然地将键（成为索引）映射到值。

```python

population_dict = {'California': 39.5, 'Texas': 29.1, 'Florida': 21.5, 'New York': 19.4}
series_from_dict = pd.Series(population_dict)

print("\n从字典创建的 Series:")
print(series_from_dict)

print("\nSeries 的数据类型:", series_from_dict.dtype)
```

**4. 从 NumPy 数组创建：**
您可以轻松地将 NumPy 数组转换为 Series。

```python

np_array = np.array([100, 200, 300, 400])
series_from_numpy = pd.Series(np_array, index=['a', 'b', 'c', 'd'])

print("\n从 NumPy 数组创建的 Series:")
print(series_from_numpy)

print("\n索引:", series_from_numpy.index)
print("值:", series_from_numpy.values)
```

### 创建和查看 Pandas DataFrame

DataFrame 是 Pandas 的主力，表示表格数据。

**1. 从列表字典创建：**
这是创建 DataFrame 的一种非常常见的方法。每个字典键成为列名，与其关联的列表成为该列的数据。所有列表的长度必须相同。

```python

student_data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [20, 21, 19, 22],
    'Major': ['CompSci', 'Physics', 'Math', 'CompSci'],
    'GPA': [3.8, 3.5, 3.9, 3.2]
}

df_students = pd.DataFrame(student_data)

print("从列表字典创建的 DataFrame:")
print(df_students)
```

**2. 从字典列表创建：**
列表中的每个字典代表一行。Pandas 从键推断列名。字典中缺失的键将导致 `NaN`（非数字）值。

```python

sensor_readings = [
    {'sensor': 'A', 'temp': 25.5, 'humidity': 60},
    {'sensor': 'B', 'temp': 26.1},
    {'sensor': 'A', 'temp': 25.8, 'humidity': 62},
    {'sensor': 'C', 'temp': 24.9, 'pressure': 1012}
]

df_sensors = pd.DataFrame(sensor_readings)

print("\n从字典列表创建的 DataFrame:")
print(df_sensors)
```

请注意数据缺失处的 `NaN` 值。Pandas 会自动处理这种情况。

**3. 从二维 NumPy 数组创建：**
您可以从 NumPy 数组创建 DataFrame，并可选择提供列名和索引名。

```python

data_np = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

df_from_np_default = pd.DataFrame(data_np)
print("\n从 NumPy 数组创建的 DataFrame（默认名称）：")
print(df_from_np_default)

df_from_np_custom = pd.DataFrame(data_np,
                                 index=['Row1', 'Row2', 'Row3'],
                                 columns=['ColA', 'ColB', 'ColC'])
print("\n从 NumPy 数组创建的 DataFrame（自定义名称）：")
print(df_from_np_custom)
```

### 检查 DataFrame

现在，我们使用前面创建的 `df_students` DataFrame 来练习检查技术。

```python
print("用于检查的学生 DataFrame:")
print(df_students)

print("\n前 2 行 (head):")
print(df_students.head(2))

print("\n后 2 行 (tail):")
print(df_students.tail(2))

print("\nDataFrame 信息:")
df_students.info()

print("\n统计描述:")
print(df_students.describe())

print("\nDataFrame 形状（行，列）:", df_students.shape)

print("\n列名:", df_students.columns)

print("\n索引:", df_students.index)
```

本次实操课程涵盖了从列表、字典和 NumPy 数组等各种常见数据结构创建 Series 和 DataFrame 的方法。我们还练习了使用 `head()`、`tail()`、`info()` 和 `describe()` 等重要方法，以及 `shape`、`columns` 和 `index` 等属性，以便快速了解数据的结构和内容。这些是您在使用 Pandas 时会持续用到的基本技能。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 6 Loading Saving Data Pandas

### 从CSV文件读取数据

# 从CSV文件读取数据

逗号分隔值（CSV）文件可能是存储和交换表格数据最普遍的格式。可以把它们看作简单的文本文件，每行代表一条数据记录，而行内值则用逗号分隔。由于它们是纯文本格式，人类易于阅读，并广泛兼容于不同的软件应用。

Pandas提供了一个功能强大且灵活的函数`pd.read_csv()`，专门用于将CSV文件数据直接读取到DataFrame中。此函数能自动处理许多复杂情况，同时也提供了多种选项来定制数据加载方式。

### 基本读取

在最简单的情况下，如果你的脚本或Notebook所在目录下有一个名为`my_data.csv`的CSV文件，你可以这样加载它：

```python
import pandas as pd

df = pd.read_csv('my_data.csv')

print(df.head())
```

默认情况下，`pd.read_csv()`假定：

1. 值由逗号（`,`）分隔。
2. 文件的第一行包含列标题。

### 指定文件路径

`pd.read_csv()`的第一个参数 (parameter)是文件路径。这可以是：

- **本地文件路径**：`'data/sales_records.csv'`或`'C:\\Users\\YourName\\Documents\\data.csv'`。
- **URL**：`'https://raw.githubusercontent.com/...'`，直接指向在线CSV文件。

```python

url = 'https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv'
titanic_df = pd.read_csv(url)
print(titanic_df.head())
```

### 处理不同分隔符

尽管逗号是标准的，但数据有时可能使用其他字符作为分隔符（也称为定界符），例如制表符（`\t`）、分号（`;`）或空格。你可以使用`sep`（或`delimiter`）参数 (parameter)指定分隔符。例如，要读取一个以制表符分隔的文件（通常以`.tsv`结尾）：

```python

df_tsv = pd.read_csv('data.tsv', sep='\t')
```

### 处理标题行

Pandas默认假定第一行为标题行（`header=0`）。

- 如果你的文件**没有标题行**，Pandas会错误地将第一行数据用作标题。为避免这种情况，请使用`header=None`。Pandas随后会自动分配默认的整数列名（0, 1, 2, ...）。

  ```python

  df_no_header = pd.read_csv('no_header_data.csv', header=None)
  ```
- 如果标题在**不同行**（例如，第二行，索引为1），你可以指定为：`header=1`。

### 分配列名

如果你的文件缺少标题行（`header=None`）或你想覆盖现有标题，你可以使用`names`参数 (parameter)提供自己的列名。这应该是一个字符串列表。

```python

column_names = ['ID', 'Measurement', 'Timestamp', 'Status']
df_named = pd.read_csv('no_header_data.csv', header=None, names=column_names)
```

注意：如果你提供`names`而*没有*设置`header=None`，并且文件*确实*有标题行，你提供的`names`将覆盖文件中找到的标题（假定`header=0`）。如果你同时设置`header=0`（或保持默认）并提供`names`，文件中的原始标题行将被丢弃并替换为你提供的`names`。如果你想使用`names`并*跳过*原始标题，你可以使用`header=0`和`names=...`，或者如果文件一开始就没有标题行，可以使用`skiprows=1`和`names=...`。

### 指定索引列

通常，你的CSV文件中的某一列包含唯一标识符，你可能想将其用作DataFrame的索引，而不是默认的整数索引（0, 1, 2, ...）。为此，请使用`index_col`参数 (parameter)。你可以通过列名（如果存在标题）或其整数位置（从0开始计数）来指定列。

```python

titanic_df_indexed = pd.read_csv(url, index_col='PassengerId')
print(titanic_df_indexed.head())
```

### 只读取特定列

对于列数较多的大型数据集，你可能只需要一部分。只读取必要的列可以节省内存并加快加载速度。使用`usecols`参数 (parameter)，提供列名或整数位置的列表。

```python

titanic_subset = pd.read_csv(url, usecols=['Survived', 'Pclass', 'Age'])
print(titanic_subset.head())
```

### 限制行数

处理非常大的文件时，你可能只想加载前几行来检查数据结构或测试代码，而不将整个文件加载到内存中。`nrows`参数 (parameter)允许你指定要读取的精确行数（不包括标题行）。

```python

titanic_sample = pd.read_csv(url, nrows=10)
print(titanic_sample)
```

这些参数涵盖了使用Pandas读取CSV文件时最常见的场景。`pd.read_csv()`函数还有更多选项用于处理日期、缺失值、引用规则、注释和编码，使其成为一个非常多功能的工具，用于将数据导入DataFrame。目前，掌握这些基本选项将使你能够有效地加载各种CSV数据。

获取即时帮助、个性化解释和交互式代码示例。

---

### 从Excel文件读取数据

# 从Excel文件读取数据

Microsoft Excel电子表格（旧版本为`.xls`，新版本为`.xlsx`）常用于存储表格数据，尤其是在商业环境中，因此非常普遍。Pandas提供了一个方便的函数`pd.read_excel()`，专门用于直接从这些文件读取数据到DataFrame。

### Excel文件基本读取

最简单地，读取Excel文件与读取CSV文件非常相似。你只需提供文件路径：

```python
import pandas as pd

try:
    df_excel = pd.read_excel('data.xlsx')
    print("Excel文件加载成功：")
    print(df_excel.head())
except FileNotFoundError:
    print("错误：未找到data.xlsx。请确保文件存在。")
```

默认情况下，`pd.read_excel()`会读取Excel工作簿中的*第一个*工作表。它还假定该工作表的第一行包含列标题。

### 处理多个工作表

Excel工作簿通常包含多个工作表，每个工作表可能存储不同的表格或相关信息。`pd.read_excel()`函数允许你使用`sheet_name`参数 (parameter)指定要加载哪个工作表。

- **按工作表名称加载：** 如果你知道工作表的名称（例如，“Sales Data”，“Inventory”），你可以将其作为字符串提供：

  ```python

  sales_df = pd.read_excel('multi_sheet_data.xlsx', sheet_name='Sales Data')
  print("\n已加载 'Sales Data' 工作表：")
  print(sales_df.head())
  ```
- **按工作表索引加载：** 工作表也可以按其位置（索引）引用，第一个工作表从0开始，第二个从1开始，依此类推。

  ```python

  inventory_df = pd.read_excel('multi_sheet_data.xlsx', sheet_name=1)
  print("\n已加载第二个工作表（索引1）：")
  print(inventory_df.head())
  ```
- **加载所有工作表：** 要将所有工作表加载到一个字典中，其中键是工作表名称，值是DataFrames，请设置`sheet_name=None`：

  ```python
  all_sheets = pd.read_excel('multi_sheet_data.xlsx', sheet_name=None)

  print("\n已加载所有工作表：")
  for sheet_name, df_sheet in all_sheets.items():
      print(f"\n--- 工作表：{sheet_name} ---")
      print(df_sheet.head())
  ```

  这会返回一个字典，当你在同一工作簿中需要处理来自多个工作表的数据时，这会很有用。

### 引擎要求

读取Excel文件需要额外的Python库来处理文件格式本身。Pandas为此使用不同的“引擎”。

- 对于较新的`.xlsx`文件（Excel 2007+），通常使用`openpyxl`库。
- 对于较旧的`.xls`文件（Excel 97-2003），传统上使用`xlrd`库，尽管出于安全原因，其在最近版本中已移除对`.xlsx`的支持。

如果你尝试读取Excel文件但未安装所需的引擎，Pandas通常会给出一条提示你安装的信息性错误消息。你通常可以使用pip安装所需的引擎：

```bash

pip install openpyxl

pip install xlrd
```

如果你打算处理现代Excel文件，通常建议安装`openpyxl`。

### 其他常用参数 (parameter)

与`pd.read_csv()`类似，`pd.read_excel()`提供多个参数来调整数据加载方式：

- `header=`: 指定用作列名称的行号（0索引）。默认为0。
- `index_col=`: 指定用作DataFrame索引的列（按名称或0索引位置）。
- `usecols=`: 允许你指定要读取的列，可以通过名称或位置。这可以在你只需要部分数据时节省内存和时间。
- `skiprows=`: 跳过工作表开头指定数量的行。
- `nrows=`: 仅读取工作表中指定数量的行。

```python

inventory_subset = pd.read_excel(
    'multi_sheet_data.xlsx',
    sheet_name='Inventory',
    usecols=['ProductID', 'ProductName', 'Quantity'],
    index_col='ProductID'
)
print("\n已加载“Inventory”工作表的子集，并以 ProductID 为索引：")
print(inventory_subset.head())
```

掌握`pd.read_excel()`对于处理存储在电子表格中数据的工作流程来说是必不可少的。它在处理不同工作表和自定义导入过程方面的灵活性，使其成为Pandas中数据获取的一个有价值的工具。请记住，在开始读取Excel文件之前，要安装必要的引擎（`openpyxl`通常是你需要的）。

获取即时帮助、个性化解释和交互式代码示例。

---

### 从其他格式读取数据

# 从其他格式读取数据

尽管CSV和Excel文件在存储表格数据方面非常常用，但Pandas提供了从各种其他来源读取数据的灵活性，例如JSON文件和SQL数据库。

### 读取JSON数据

JSON（JavaScript对象表示法）是一种轻量级的数据交换格式。它使用人类可读的文本来传输由属性-值对和数组数据类型组成的数据对象。它经常用于Web应用程序和API。

设想您有数据存储在一个名为`data.json`的JSON文件中。JSON文件的结构很大程度上影响Pandas如何读取它。一种常见结构是记录（字典）列表，例如：

```json
[
  {"name": "Alice", "age": 30, "city": "New York"},
  {"name": "Bob", "age": 24, "city": "Los Angeles"},
  {"name": "Charlie", "age": 35, "city": "Chicago"}
]
```

要将其读取到DataFrame中，您可以使用`pd.read_json()`函数：

```python
import pandas as pd

try:
    df_json = pd.read_json('data.json')
    print(df_json)
except FileNotFoundError:
    print("Error: data.json not found. Please create it with the example content.")

json_string = """
[
  {"name": "Alice", "age": 30, "city": "New York"},
  {"name": "Bob", "age": 24, "city": "Los Angeles"},
  {"name": "Charlie", "age": 35, "city": "Chicago"}
]
"""
df_from_string = pd.read_json(json_string)
```

**输出：**

```
      name  age         city
0    Alice   30     New York
1      Bob   24  Los Angeles
2  Charlie   35      Chicago
```

`pd.read_json()`有参数 (parameter)来处理不同的JSON结构。例如，如果您的JSON结构以键作为索引或列，您可能需要指定`orient`参数（例如，`orient='index'`、`orient='columns'`、`orient='records'`）。默认的`orient=None`会尝试推断结构，这对于上面显示的常见记录列表格式通常运行良好。

### 从SQL数据库读取数据

Pandas可以直接与SQL数据库交互，允许您执行查询并将结果加载到DataFrame中。这需要一个名为`SQLAlchemy`的额外库来处理不同SQL数据库类型（如PostgreSQL、MySQL、SQLite等）的数据库连接详情。

首先，您通常使用SQLAlchemy建立连接引擎：

```python

from sqlalchemy import create_engine
import pandas as pd

try:

    engine = create_engine('sqlite:///my_database.db')

    with engine.connect() as connection:
        connection.execute("DROP TABLE IF EXISTS users")
        connection.execute("CREATE TABLE users (name TEXT, age INTEGER, city TEXT)")
        connection.execute("INSERT INTO users (name, age, city) VALUES ('David', 42, 'Boston')")
        connection.execute("INSERT INTO users (name, age, city) VALUES ('Eve', 29, 'Miami')")

    df_sql_table = pd.read_sql_table('users', con=engine)
    print("--- Reading entire table ---")
    print(df_sql_table)

    query = "SELECT name, city FROM users WHERE age > 30"
    df_sql_query = pd.read_sql_query(query, con=engine)
    print("\n--- Reading specific query results ---")
    print(df_sql_query)

except ImportError:
    print("SQLAlchemy is required but not installed. Run: pip install sqlalchemy")
except Exception as e:

    print(f"An error occurred: {e}")
    print("Ensure 'my_database.db' exists or adjust the connection string.")
```

**输出（假设连接和表创建成功）：**

```
--- Reading entire table ---
    name  age    city
0  David   42  Boston
1    Eve   29   Miami

--- Reading specific query results ---
    name    city
0  David  Boston
```

在这些示例中：

1. 我们从`SQLAlchemy`导入`create_engine`。
2. 我们创建一个表示数据库连接的`engine`对象。连接字符串（`'sqlite:///my_database.db'`）告诉SQLAlchemy如何连接（使用SQLite方言，连接到文件`my_database.db`）。对于其他数据库，如PostgreSQL或MySQL，字符串会看起来不同，并包含主机名、用户名、密码等。
3. `pd.read_sql_table('users', con=engine)`将名为`users`的整个表读取到DataFrame中。
4. `pd.read_sql_query(query, con=engine)`通过`engine`对数据库执行指定的SQL `query`，并将结果作为DataFrame返回。

从SQL读取数据功能强大，因为它允许您使用数据库的查询能力，在将数据作为Pandas DataFrame加载到内存之前，对其进行筛选、聚合和选择，这对于大型数据集来说非常高效。请记住，设置数据库连接（`engine`）是一个先决条件，并取决于您正在使用的特定数据库系统。

### 其他格式

Pandas也包含其他格式的功能，尽管可能不如CSV、Excel、JSON或SQL常用：

- `pd.read_html()`：直接从HTML网页读取表格。适用于网页抓取。
- `pd.read_fwf()`：从固定宽度格式的行读取数据。
- `pd.read_clipboard()`：从系统剪贴板读取文本并尝试将其解析为DataFrame（通常对于快速获取数据非常有用）。
- 存在用于HDF5、Parquet、Feather、Stata、SAS、SPSS文件等的功能，满足科学计算和其他特定方面的需求。

虽然我们这里侧重于JSON和SQL，但知道Pandas为许多数据类型提供了统一的接口（`pd.read_...`），使其成为数据加载的多功能工具。每个函数的具体参数 (parameter)因格式的特性而异。始终查阅Pandas文档以了解您打算使用的特定读取函数。

获取即时帮助、个性化解释和交互式代码示例。

---

### 将数据写入 CSV 文件

# 将数据写入 CSV 文件

在 Pandas DataFrame 中处理数据时，保存处理结果或整理后的数据集是常见需求，以便将来使用、分享或作为其他过程的输入。正如 `pd.read_csv()` 是读取 CSV 数据的标准方式一样，将 DataFrame 数据写回到 CSV 文件的相应方法是 `.to_csv()`。

CSV（逗号分隔值）格式因其简单性以及与众多应用程序（包括 Microsoft Excel 或 Google Sheets 等电子表格软件、数据库和其他编程环境）的兼容性，仍然是交换表格数据被广泛使用的标准。

### `to_csv()` 的基本用法

`.to_csv()` 最基本的用法是在您的 DataFrame 对象上调用它，并提供您希望保存数据的文件路径。

假设我们有以下 DataFrame，它可能是在之前的步骤中创建或修改的：

```python
import pandas as pd
import numpy as np

data = {'col_A': [1, 2, 3, 4, 5],
        'col_B': ['apple', 'banana', 'orange', 'grape', 'kiwi'],
        'col_C': [0.1, 0.2, np.nan, 0.4, 0.5]}
df_to_save = pd.DataFrame(data)

print(df_to_save)
```

要将此 DataFrame 保存到当前工作目录中名为 `output_data.csv` 的文件，您只需运行：

```python
df_to_save.to_csv('output_data.csv')
```

如果您用文本编辑器打开 `output_data.csv`，您会看到类似以下内容：

```csv
,col_A,col_B,col_C
0,1,apple,0.1
1,2,banana,0.2
2,3,orange,
3,4,grape,0.4
4,5,kiwi,0.5
```

请注意以下几点：

1. 所有值都由逗号分隔。
2. 列标题（`col_A`、`col_B`、`col_C`）作为第一行包含在内。
3. DataFrame 的索引（0、1、2、3、4）已作为第一列写入，并且它没有列头名称。

通常，您可能不想在输出文件中包含 DataFrame 索引，特别是如果它只是默认的整数索引（0、1、2...），不表示有意义的数据。

### 控制输出：常用参数 (parameter)

`.to_csv()` 方法提供多种参数来自定义输出文件。以下是一些最常用的参数：

#### `index` 参数

正如所见，默认行为是写入 DataFrame 索引。为了避免这种情况，请将 `index` 参数设置为 `False`。这是一个非常普遍的需求。

```python

df_to_save.to_csv('output_data_no_index.csv', index=False)
```

现在，`output_data_no_index.csv` 将会是这样：

```csv
col_A,col_B,col_C
1,apple,0.1
2,banana,0.2
3,orange,
4,grape,0.4
5,kiwi,0.5
```

这种格式通常更整洁，更适合导入到可能生成自己行标识符的其他系统中。

#### `header` 参数

同样地，您有时可能希望省略标题行（即列名）。这可以通过将 `header` 参数设置为 `False` 来完成。

```python

df_to_save.to_csv('output_data_no_header.csv', index=False, header=False)
```

`output_data_no_header.csv` 的内容将是：

```csv
1,apple,0.1
2,banana,0.2
3,orange,
4,grape,0.4
5,kiwi,0.5
```

相比于省略索引，这不太常见，但在特定情况下会有用，例如向已包含标题的现有文件追加数据。

#### `sep` 参数

尽管 CSV 代表逗号分隔值，但数据文件有时会使用不同的字符（分隔符）来分隔字段。常见的替代选项包括制表符（`\t`）、分号（`;`）或竖线（`|`）。`sep` 参数允许您指定分隔符。例如，要创建一个制表符分隔值（TSV）文件：

```python

df_to_save.to_csv('output_data.tsv', index=False, sep='\t')
```

打开 `output_data.tsv` 会显示字段由制表符而不是逗号分隔。

#### `na_rep` 参数

请注意，在默认输出中，`col_C` 中缺失的 `NaN` 值导致 CSV 中出现一个空字段（`3,orange,,`）。您可以使用 `na_rep` 参数为缺失值指定一个自定义的字符串表示。

```python

df_to_save.to_csv('output_data_na_rep.csv', index=False, na_rep='MISSING')
```

文件 `output_data_na_rep.csv` 将包含：

```csv
col_A,col_B,col_C
1,apple,0.1
2,banana,0.2
3,orange,MISSING
4,grape,0.4
5,kiwi,0.5
```

#### `columns` 参数

如果您只想保存 DataFrame 中一部分列，可以向 `columns` 参数提供一个列名列表。

```python

df_to_save.to_csv('output_data_subset.csv', index=False, columns=['col_A', 'col_B'])
```

文件 `output_data_subset.csv` 将只包含这两列的数据：

```csv
col_A,col_B
1,apple
2,banana
3,orange
4,grape
5,kiwi
```

#### `encoding` 参数

文本文件使用特定的字符编码存储。虽然通常默认处理正确（Pandas 通常默认使用 'utf-8'，它具有广泛的兼容性），但有时如果接收系统有要求，或者您的数据包含需要特定编码（如 'latin1' 或 'cp1252'）的特殊字符，您可能需要指定不同的编码。

```python

```

有效保存数据与加载数据同样重要。`.to_csv()` 方法提供灵活的选项，以确保您的 DataFrame 以所需格式导出，准备好用于存储、分享或数据工作流程的后续步骤。请记住，将 `index=False` 通常是一个好的做法，除非您的 DataFrame 索引包含需要明确在输出文件中保留的重要信息。

获取即时帮助、个性化解释和交互式代码示例。

---

### 写入数据到Excel文件

# 写入数据到Excel文件

Pandas 提供了简单的方法将您的 DataFrame 保存为不同格式。虽然 CSV 文件因其简洁性和系统间的互操作性而表现出色，Microsoft Excel 文件（`.xlsx` 或旧版 `.xls`）在商业环境中非常常见，并支持在单个文件中包含多个工作表等功能。

Pandas 使用 `to_excel()` 方法可以简单地将 DataFrame 写入 Excel 文件，该方法与我们之前看到的 `to_csv()` 方法类似。

### 基本用法：将 DataFrame 保存到 Excel 文件

我们从一个基本示例开始。假设我们已经处理了一些数据并将其存储在一个 DataFrame 中：

```python
import pandas as pd

data = {'Product': ['Widget A', 'Widget B', 'Gadget C', 'Widget A'],
        'Region': ['North', 'South', 'North', 'West'],
        'Sales': [150, 200, 120, 180],
        'Profit': [30, 45, 25, 38]}
df = pd.DataFrame(data)

print("原始 DataFrame:")
print(df)

df.to_excel('output_data.xlsx')

print("\nDataFrame 已成功保存到 output_data.xlsx")
```

如果您运行此代码，然后使用 Microsoft Excel、LibreOffice Calc 或 Google Sheets 等电子表格软件打开生成的 `output_data.xlsx` 文件，您会看到您的 DataFrame 以表格形式呈现。

### 使用参数 (parameter)控制输出

`to_excel()` 方法提供了多个参数用于自定义输出文件。以下是一些常用的参数：

1. **`excel_writer`（文件路径）：** 这是第一个参数，用于指定输出 Excel 文件的文件路径和名称（例如，`'data/processed_sales.xlsx'`，`'report.xlsx'`）。
2. **`sheet_name`：** 默认情况下，DataFrame 会写入名为 'Sheet1' 的工作表。您可以使用此参数指定不同的名称。例如，`df.to_excel('report.xlsx', sheet_name='SalesData')`。
3. **`index`：** 与 `to_csv()` 类似，`to_excel()` 默认将 DataFrame 的索引作为 Excel 工作表的第一列写入。通常，此索引只是默认的顺序整数索引（0, 1, 2...），并不是您希望在最终电子表格中显示的有效数据。为避免写入索引，请设置 `index=False`：

   ```python

   df.to_excel('output_data_no_index.xlsx', index=False)
   print("DataFrame 已保存到 output_data_no_index.xlsx，不含索引列。")
   ```

   将 `index=False` 设置为常见做法，尤其是在为不熟悉 Pandas 索引的人导出数据时。
4. **`columns`：** 如果您只想保存 DataFrame 中的特定列，可以向 `columns` 参数提供列名列表：

   ```python

   df.to_excel('output_subset.xlsx', columns=['Product', 'Sales'], index=False)
   print("已将 Product 和 Sales 列保存到 output_subset.xlsx")
   ```
5. **`header`：** 此参数控制是否将列名（标题行）写入文件。它默认为 `True`。如果您不想包含标题行，请设置 `header=False`。
6. **`startrow` 和 `startcol`：** 这些参数允许您指定 DataFrame 在工作表中写入的左上角单元格（0-索引）。例如，`startrow=1, startcol=2` 将从 C2 单元格开始写入 DataFrame（如果适用，包括其标题）。这在向现有工作表添加数据或创建更复杂布局时会很有用。

### 将多个 DataFrame 写入单个 Excel 文件

Excel 文件相对于简单 CSV 的一个重要优点是它们能够包含多个工作表。Pandas 允许您使用 `pd.ExcelWriter` 对象将多个 DataFrame 写入同一个 `.xlsx` 文件中的不同工作表。这通常使用 `with` 语句完成，它能确保文件正确保存和关闭。

您可以这样做：

```python
import pandas as pd

df_sales = pd.DataFrame({
    'Region': ['North', 'South', 'East', 'West'],
    'Sales': [1000, 1500, 1200, 1800]
})

df_inventory = pd.DataFrame({
    'Product ID': ['P101', 'P102', 'P103', 'P104'],
    'Stock': [50, 75, 30, 90],
    'Warehouse': ['WH-A', 'WH-B', 'WH-A', 'WH-C']
})

output_filename = 'multi_sheet_report.xlsx'
with pd.ExcelWriter(output_filename) as writer:
    df_sales.to_excel(writer, sheet_name='销售汇总', index=False)
    df_inventory.to_excel(writer, sheet_name='库存水平', index=False)

print(f"两个 DataFrame 已保存到 {output_filename}")
```

如果您打开 `multi_sheet_report.xlsx`，您会发现两个名为“销售汇总”和“库存水平”的工作表，每个工作表都包含相应 DataFrame 的数据。

### 所需库（引擎）

为写入 Excel 文件，Pandas 依赖于称为“引擎”的外部库。

- 对于现代 `.xlsx` 文件（推荐），您通常需要 `openpyxl` 库。
- 对于旧版 `.xls` 文件，曾使用 `xlwt` 库。

如果您尝试使用 `to_excel()` 但未安装所需引擎，Pandas 通常会引发 `ImportError`，并附带安装说明。您可以使用 pip 安装 `openpyxl`：

```bash
pip install openpyxl
```

或使用 conda：

```bash
conda install openpyxl
```

### 总结

使用 `df.to_excel()` 将 DataFrame 保存到 Excel 文件是一项有用的技能，可用于分享您的分析结果或处理过的数据，尤其是在电子表格是数据查看和操作标准工具的环境中。请记住 `index=False` 选项可以使输出更整洁，并在需要将相关数据整理到单个文件中的多个工作表时，考虑使用 `pd.ExcelWriter`。虽然 Excel 很方便，但请注意，对于非常大的数据集，其写入速度可能比 CSV 或更专业的二进制格式慢。

获取即时帮助、个性化解释和交互式代码示例。

---

### 动手实践：数据集的导入与导出

# 动手实践：数据集的导入与导出

让我们将本章的理念付诸实践。我们将演练从文件读取数据、进行少量修改，然后保存结果的常见情况。这模拟了数据分析中常见的模式：加载、处理、保存。

首先，请确保已导入 Pandas。通常，我们使用别名 `pd`。

```python
import pandas as pd
print(f"Pandas version: {pd.__version__}")
```

### 准备示例数据文件

对于这些练习，我们需要一些数据文件。假设我们有两个文件：

1. `students.csv`：一个逗号分隔值文件。
2. `grades.xlsx`：一个 Microsoft Excel 文件。

如果您在本地运行此代码，可以自行创建这些文件。

**创建 `students.csv`：**
将以下文本保存到与您的脚本或笔记本位于同一目录下的 `students.csv` 文件中：

```csv
StudentID,Name,Major
101,Alice,Physics
102,Bob,Chemistry
103,Charlie,Mathematics
104,David,Physics
```

**创建 `grades.xlsx`：**
您需要安装 `openpyxl` 等库（`pip install openpyxl`）才能处理 `.xlsx` 文件。创建一个名为 `grades.xlsx` 的 Excel 文件，并在名为 `Sheet1` 的工作表中放入以下数据：

| 学生ID | 课程 | 分数 |
| --- | --- | --- |
| 101 | 力学 | 85 |
| 102 | 有机化学 | 92 |
| 101 | 电磁学 | 78 |
| 103 | 微积分 | 95 |
| 102 | 热力学 | 88 |
| 104 | 力学 | 81 |

### 从 CSV 读取数据

现在，让我们使用 `pd.read_csv()` 将 `students.csv` 文件读取到 Pandas DataFrame 中。

```python

students_df = pd.read_csv('students.csv')

print("学生 DataFrame：")
print(students_df)
```

您会注意到 Pandas 正确地将第一行识别为标题，并使用了默认的整数索引（0, 1, 2, 3）。

如果我们的文件使用不同的分隔符（例如制表符）怎么办？或者，如果我们想让 `StudentID` 列作为 DataFrame 的索引怎么办？我们可以在 `read_csv` 中使用参数 (parameter)。

```python

students_df_indexed = pd.read_csv('students.csv', index_col='StudentID')

print("\n带有 StudentID 作为索引的学生 DataFrame：")
print(students_df_indexed)
```

设置 `index_col='StudentID'` 会指示 Pandas 将该列中的值用作 DataFrame 的行标签（即索引）。

### 从 Excel 读取数据

接下来，让我们使用 `pd.read_excel()` 从 `grades.xlsx` 文件加载数据。

```python

try:
    grades_df = pd.read_excel('grades.xlsx')
    print("\n成绩 DataFrame（来自 Excel Sheet1）：")
    print(grades_df)
except ImportError:
    print("\n请安装 'openpyxl' 以读取 Excel 文件：pip install openpyxl")
except FileNotFoundError:
    print("\n请确保 'grades.xlsx' 位于正确的目录中。")
```

默认情况下，`read_excel` 读取第一个工作表。如果我们的数据在不同的工作表上，例如“CourseGrades”，我们会这样指定：

```python

```

您可以为 `sheet_name` 使用工作表名称（字符串）或其基于零的索引（整数）。

### 简单修改

让我们对已加载的 `grades_df` 添加一个新列。假设我们想要一个“Status”列，指示分数在 70 或以上为“Pass”（通过），否则为“Fail”（未通过）。我们可以使用布尔索引和赋值来实现这一点。

```python

if 'grades_df' in locals():

    grades_df['Status'] = 'Fail'

    grades_df.loc[grades_df['Grade'] >= 70, 'Status'] = 'Pass'

    print("\n已添加 Status 列的成绩 DataFrame：")
    print(grades_df)
else:
    print("\n跳过修改，因为 grades_df 未加载。")
```

我们首先创建了该列并分配了一个默认值（'Fail'）。然后，我们使用带有布尔条件（`grades_df['Grade'] >= 70`）的 `.loc` 来选择分数达到及格标准的行，并仅将其“Status”值更新为“Pass”。

### 将数据写入 CSV

现在我们已经修改了 `grades_df`，让我们将其保存回一个新的 CSV 文件，名为 `processed_grades.csv`。

一个常见的要求是*不*将 DataFrame 的索引（0, 1, 2...）写入文件，因为它通常不表示有意义的数据。我们为此使用 `index=False` 参数 (parameter)。

```python

if 'grades_df' in locals():

    grades_df.to_csv('processed_grades.csv', index=False)
    print("\n已将修改后的成绩数据保存到 processed_grades.csv（不含索引）。")

else:
    print("\n跳过 CSV 导出，因为 grades_df 不可用。")
```

如果您打开 `processed_grades.csv`，您会看到 `StudentID`、`Course`、`Grade` 和 `Status` 列，但没有默认的基于 0 的索引列。

### 将数据写入 Excel

同样，我们可以将修改后的 DataFrame 保存到 Excel 文件中。我们将其命名为 `processed_grades.xlsx`，并将数据放入名为“Final Grades”的工作表中。同样，我们通常会希望 `index=False`。

```python

if 'grades_df' in locals():
    try:

        grades_df.to_excel('processed_grades.xlsx', sheet_name='Final Grades', index=False)
        print("\n已将修改后的成绩数据保存到 processed_grades.xlsx（工作表：“Final Grades”，不含索引）。")

    except ImportError:
        print("\n请安装 'openpyxl' 以写入 Excel 文件：pip install openpyxl")
else:
    print("\n跳过 Excel 导出，因为 grades_df 不可用。")
```

这会创建一个 Excel 文件，其中包含我们处理过的数据，并在指定的工作表中组织整齐。

### 总结

在此练习环节中，您使用了 Pandas 的基本输入输出函数：

- `pd.read_csv()`：从文本文件加载数据，并了解 `index_col`。
- `pd.read_excel()`：从电子表格加载数据，并了解 `sheet_name`。
- `.to_csv()`：保存 DataFrame，强调使用 `index=False`。
- `.to_excel()`：保存 DataFrame，使用 `index=False` 和 `sheet_name`。

您还在读取和写入之间执行了一个简单的数据修改步骤，这体现了一个基本的数据处理流程。熟练掌握这些输入和输出操作对于将 Pandas 融入您的数据分析任务非常重要。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 7 Data Selection Indexing Pandas

### 选取列

# 选取列

提取一个或多个列是在处理 DataFrame 时最常见的任务之一。也许您只需要从一个更大的数据集中获取姓名和年龄，或者您想对某个数值列执行计算。Pandas 提供了直接的方法来实现这一点。

让我们从一个示例 DataFrame 开始，以演示这些原理。假设我们有关于个人的数据，包括他们的姓名、年龄、城市和薪资：

```python
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston'],
        'Salary': [70000, 80000, 90000, 100000]}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
```

Output:

```
Original DataFrame:
      Name  Age         City  Salary
0    Alice   25     New York   70000
1      Bob   30  Los Angeles   80000
2  Charlie   35      Chicago   90000
3    David   40      Houston  100000
```

### 选取单个列

选取单个列最直接的方法是使用方括号 `[]`，并在其中放入列名（字符串形式）：

```python

names = df['Name']

print("Selected 'Name' column:")
print(names)
print("\nType of the selected object:", type(names))
```

Output:

```
Selected 'Name' column:
0      Alice
1        Bob
2    Charlie
3      David
Name: Name, dtype: object

Type of the selected object: <class 'pandas.core.series.Series'>
```

请注意，以这种方式选取单个列会返回一个 Pandas Series 对象，而不是 DataFrame。Series 类似于一个一维带标签数组，它保存了该列的数据及其索引。

### 选取多个列

要选取多个列，您再次使用方括号 `[]`。然而，在方括号内，您需要提供一个您想选取的列名*列表*：

```python

subset = df[['Name', 'City']]

print("Selected 'Name' and 'City' columns:")
print(subset)
print("\nType of the selected object:", type(subset))
```

Output:

```
Selected 'Name' and 'City' columns:
      Name         City
0    Alice     New York
1      Bob  Los Angeles
2  Charlie      Chicago
3    David      Houston

Type of the selected object: <class 'pandas.core.frame.DataFrame'>
```

重要说明：当您在方括号 `[['Col1', 'Col2']]` 内使用列表选取多列时，结果是一个新的 DataFrame，它只包含您指定的列，并按您列出的顺序排列。

### 另一种方法：点表示法

Pandas 也允许使用点表示法访问单个列，类似于访问对象的属性，前提是列名是有效的 Python 标识符（例如，没有空格，不以数字开头，不与现有 DataFrame 方法冲突）：

```python

ages_dot = df.Age

print("Selected 'Age' column using dot notation:")
print(ages_dot)
print("\nType of the selected object:", type(ages_dot))
```

Output:

```
Selected 'Age' column using dot notation:
0    25
1    30
2    35
3    40
Name: Age, dtype: int64

Type of the selected object: <class 'pandas.core.series.Series'>
```

与选取单个列的方括号表示法一样，点表示法也返回一个 Series。

### 为何更推荐方括号表示法 (`[]`)？

尽管点表示法在快速访问时可能很方便，但通常推荐使用方括号表示法（`df['Column']` 或 `df[['Col1', 'Col2']]`）来选取列，特别是对于初学者，原因有以下几点：

1. **适用所有列名：** 方括号表示法适用于*任何*列名，包括那些包含空格、特殊字符或数字的列名（例如，`df['First Name']`，`df['Sales 2023']`）。点表示法对此类名称无效。
2. **避免冲突：** 列名可能与现有 DataFrame 方法或属性冲突（例如，如果您有一个名为 `count` 的列，`df.count` 将指代方法，而非您的列）。方括号表示法（`df['count']`）避免了这种歧义。
3. **一致性：** 方括号表示法用于选取单列和多列（通过传递字符串或字符串列表）。它也与我们很快会看到的 `.loc` 和 `.iloc` 索引器密切相关，提供更一致的语法模式。
4. **赋值：** 为列赋值需要使用方括号表示法（例如，`df['New Column'] = values`）。

基于这些原因，尽管您可能会在示例中遇到点表示法，但坚持使用方括号表示法进行列选取有助于编写更清晰、更不易出错的代码。

选取合适的列通常是您为进一步分析或处理选取所需数据的第一步。接下来，我们将了解如何将列选取与行选取结合起来，以实现更强大的数据获取。

获取即时帮助、个性化解释和交互式代码示例。

---

### 使用标签选择行 (.loc)

# 使用标签选择行 (.loc)

尽管按名称选择列很简单，但 Pandas 提供了更强大的机制来选择*行*或行与列的组合。主要的基于标签的选择方法是 `.loc` 访问器。可以将 `.loc` 视为根据数据的“名称”或“标签”进行选择 — 无论是您赋予行的名称（其索引标签）还是列的名称。

这与按数字位置选择不同，我们将在下一节中使用 `.iloc` 来介绍。只要标签保持一致，使用标签可以使您的代码更易读，并且在数据顺序改变时更不容易出错。

基本语法涉及在 `.loc` 之后，将所需的行标签和（可选的）列标签放在方括号 `[]` 中传入：

```python

```

我们来创建一个示例 DataFrame 来展示 `.loc` 如何工作。假设我们有几天的天气数据：

```python
import pandas as pd
import numpy as np

data = {'Temperature': [25, 28, 22, 31, 29],
        'Humidity': [60, 55, 70, 50, 65],
        'WindSpeed': [10, 12, 8, 15, 11]}
index_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
weather_df = pd.DataFrame(data, index=index_labels)

print(weather_df)
```

```text
     Temperature  Humidity  WindSpeed
Mon           25        60         10
Tue           28        55         12
Wed           22        70          8
Thu           31        50         15
Fri           29        65         11
```

### 选择单行

要选择单行，将其索引标签传给 `.loc`。结果将是一个 Pandas Series，其索引是原始 DataFrame 的列名。

```python

wednesday_data = weather_df.loc['Wed']

print(wednesday_data)
print(type(wednesday_data))
```

```text
Temperature    22
Humidity       70
WindSpeed       8
Name: Wed, dtype: int64
<class 'pandas.core.series.Series'>
```

### 选择多行

您可以通过提供一个索引标签列表来选择多个特定行。结果是一个只包含指定行的新 DataFrame。

```python

mon_fri_data = weather_df.loc[['Mon', 'Fri']]

print(mon_fri_data)
print(type(mon_fri_data))
```

```text
     Temperature  Humidity  WindSpeed
Mon           25        60         10
Fri           29        65         11
<class 'pandas.core.frame.DataFrame'>
```

### 选择行范围 (按标签切片)

`.loc` 也支持使用索引标签进行切片。与标准 Python 切片或基于位置的切片（我们将在 `.iloc` 中看到）的一个重要区别是，**使用 `.loc` 进行的基于标签的切片包含起始和结束标签**。

```python

tue_to_thu_data = weather_df.loc['Tue':'Thu']

print(tue_to_thu_data)
```

```text
     Temperature  Humidity  WindSpeed
Tue           28        55         12
Wed           22        70          8
Thu           31        50         15
```

注意，标签为 'Thu' 的行被包含在输出中。这种包含性行为仅在使用标签进行切片时适用。

### 同时选择行和列

当您按标签同时选择行和列时，`.loc` 的真正作用就体现出来了。您首先提供行选择器，然后是一个逗号，再是列选择器。

```python

temp_wed = weather_df.loc['Wed', 'Temperature']
print(f"星期三气温: {temp_wed}\n")

hum_wind_mon_tue = weather_df.loc[['Mon', 'Tue'], ['Humidity', 'WindSpeed']]
print(hum_wind_mon_tue, "\n")

temp_humidity = weather_df.loc[:, ['Temperature', 'Humidity']]
print(temp_humidity, "\n")

subset_slice = weather_df.loc['Wed':'Fri', 'Humidity':'WindSpeed']
print(subset_slice)
```

```text
星期三气温: 22

     Humidity  WindSpeed
Mon        60         10
Tue        55         12

     Temperature  Humidity
Mon           25        60
Tue           28        55
Wed           22        70
Thu           31        50
Fri           29        65

     Humidity  WindSpeed
Wed        70          8
Thu        50         15
Fri        65         11
```

在上面的第三个示例中，行位置 `weather_df.loc[:, ['Temperature', 'Humidity']]` 中使用的冒号 `:` 表示“选择所有行”。同样，您可以在列位置使用 `:` 来选择特定行的所有列。

### `.loc` 与布尔条件一起使用

尽管后面有一个专门讲布尔索引的部分，但知道 `.loc` 可以用于行选择的布尔数组 (Series) 是有益的。您可以创建一个条件，该条件会产生一个由 `True`/`False` 值组成的 Series，然后将这个 Series 传给 `.loc`。它将只返回条件为 `True` 的行。

```python

hot_days = weather_df.loc[weather_df['Temperature'] > 25]

print(hot_days)
```

```text
     Temperature  Humidity  WindSpeed
Tue           28        55         12
Thu           31        50         15
Fri           29        65         11
```

这里，`weather_df['Temperature'] > 25` 创建了一个布尔 Series：

```text
Mon    False
Tue     True
Wed    False
Thu     True
Fri     True
Name: Temperature, dtype: bool
```

将这个 Series 传给 `weather_df.loc[...]` 会有效地选择对应 `True` 值的行。

### 使用不同的索引类型

请记住，`.loc` 总是作用于索引和列的*标签*。如果您的 DataFrame 具有默认的整数索引 (0, 1, 2, ...)，那么 `.loc` 会将这些整数用作*标签*。

```python
df_int_index = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df_int_index)

print("\n标签为 1 的行:")
print(df_int_index.loc[1])
```

```text
   A  B
0  1  4
1  2  5
2  3  6

标签为 1 的行:
A    2
B    5
Name: 1, dtype: int64
```

尽管 `1` 看起来像一个位置，但在此情境下，`.loc` 将其视为该行的*名称*或*标签*。这有时会让人困惑，这也是为什么纯粹基于位置的 `.iloc` 访问器存在的原因，我们将在后面看到。

总之，当您知道所需行和列的名称（标签）时，`.loc` 是您选择数据的工具。它支持按标签选择单个项、项列表和切片（包含末端），使您的选择逻辑清晰。

获取即时帮助、个性化解释和交互式代码示例。

---

### 使用整数位置选择行 (.iloc)

# 使用整数位置选择行 (.iloc)

Pandas 中的数据选择常需要根据数据的确切位置而不是其标签来访问信息。有时，你需要完全根据数据的整数位置来检索数据，而无需考虑 DataFrame 中分配的索引标签或列名。当行或列的顺序已知但其具体标签未知时，或者在处理使用默认整数索引的 DataFrame 时，这种方法尤其有用。为了满足这一特定需求，Pandas 提供了 `.iloc` 访问器。

`.iloc` 访问器的工作方式很像标准的 Python 列表和 NumPy 数组索引。它使用基于 0 的整数位置来选择行和列。请记住，Python 的切片约定适用：起始边界包含在内，而结束边界不包含在内。

### 基本语法

一般语法是 `DataFrame.iloc[row_indexer, column_indexer]`。`row_indexer` 和 `column_indexer` 都接受整数、整数列表、带有整数的切片或布尔数组（尽管使用标准 `[]` 或 `.loc` 进行布尔索引通常更清晰）。如果你只提供一个索引器，则默认为行索引器。

我们来用一个示例 DataFrame 进行说明：

```python
import pandas as pd
import numpy as np

data = {'col_a': [10, 20, 30, 40, 50],
        'col_b': [0.1, 0.2, 0.3, 0.4, 0.5],
        'col_c': ['x', 'y', 'z', 'x', 'y']}

df_example = pd.DataFrame(data, index=['row1', 'row2', 'row3', 'row4', 'row5'])

print("示例 DataFrame:")
print(df_example)
```

```
示例 DataFrame:
      col_a  col_b col_c
row1     10    0.1     x
row2     20    0.2     y
row3     30    0.3     z
row4     40    0.4     x
row5     50    0.5     y
```

### 选择单行

要通过整数位置选择单行，请将该整数传递给 `.iloc`：

```python

first_row = df_example.iloc[0]
print("\n第一行（位置 0）：")
print(first_row)

third_row = df_example.iloc[2]
print("\n第三行（位置 2）：")
print(third_row)
```

```
第一行（位置 0）：
col_a     10
col_b    0.1
col_c      x
Name: row1, dtype: object

第三行（位置 2）：
col_a     30
col_b    0.3
col_c      z
Name: row3, dtype: object
```

注意，尽管我们的索引由字符串（如 `'row1'`、`'row2'` 等）组成，但 `.iloc` 仍根据行基于 0 的整数位置来访问它们。结果是一个 Pandas Series，包含该行的数据，并以原始列名作为其索引。

### 使用列表或切片选择多行

你可以通过提供整数列表来选择多个特定行，或者使用切片表示法选择一系列行。

```python

rows_0_2 = df_example.iloc[[0, 2]]
print("\n位置 0 和 2 的行：")
print(rows_0_2)

rows_1_to_3 = df_example.iloc[1:4]
print("\n从位置 1 到位置 4 的行：")
print(rows_1_to_3)

first_three_rows = df_example.iloc[:3]
print("\n前三行：")
print(first_three_rows)

last_rows = df_example.iloc[3:]
print("\n从位置 3 到末尾的行：")
print(last_rows)
```

```
位置 0 和 2 的行：
      col_a  col_b col_c
row1     10    0.1     x
row3     30    0.3     z

从位置 1 到位置 4 的行：
      col_a  col_b col_c
row2     20    0.2     y
row3     30    0.3     z
row4     40    0.4     x

前三行：
      col_a  col_b col_c
row1     10    0.1     x
row2     20    0.2     y
row3     30    0.3     z

从位置 3 到末尾的行：
      col_a  col_b col_c
row4     40    0.4     x
row5     50    0.5     y
```

正如所料，选择多行会返回一个包含指定行的新 DataFrame。切片行为 `start:end` 包含 `start` 但不包含 `end`，这与 Python 标准一致。

### 同时选择行和列

当你需要根据行和列的位置选择特定元素或子部分时，`.iloc` 的优势明显。你先提供行索引器，然后是列索引器，两者之间用逗号分隔。

```python

element_1_0 = df_example.iloc[1, 0]
print(f"\n行位置 1、列位置 0 的元素：{element_1_0}")

row0_cols01 = df_example.iloc[0, 0:2]
print("\n第一行，前两列：")
print(row0_cols01)

subset = df_example.iloc[0:3, [0, 2]]
print("\n前三行，列 0 和 2：")
print(subset)

last_col = df_example.iloc[:, -1]
print("\n所有行，最后一列：")
print(last_col)
```

```
行位置 1、列位置 0 的元素：20

第一行，前两列：
col_a     10
col_b    0.1
Name: row1, dtype: object

前三行，列 0 和 2：
      col_a col_c
row1     10     x
row2     20     y
row3     30     z

所有行，最后一列：
row1    x
row2    y
row3    z
row4    x
row5    y
Name: col_c, dtype: object
```

使用 `:` 选择所有行或所有列，这与 NumPy 切片相似。负数索引从末尾开始计数（例如，`-1` 是最后一列）。

### 区别：`.iloc` 与 `.loc`

记住它们的区别非常重要：

- `.loc`：基于**标签**选择（索引标签、列名）。使用标签进行切片时，起始和结束标签都包含在内。
- `.iloc`：基于**整数位置**选择（基于 0）。使用整数进行切片时，不包含结束位置。

尝试将标签与 `.iloc` 结合使用，或将整数位置与 `.loc` 结合使用（除非标签**恰好**是整数）将导致错误。理解这种区别对于正确获取所需数据很重要。

熟练使用 `.iloc` 提供了一种根据数据在 DataFrame 结构中的位置来获取数据的精确方法，补充了 `.loc` 提供的基于标签的选择功能。

获取即时帮助、个性化解释和交互式代码示例。

---

### 混合标签和基于位置的索引

# 混合标签和基于位置的索引

在 Pandas 中，可以使用 `.loc` 通过标签选择数据，并使用 `.iloc` 通过整数位置选择数据。这些工具非常有用，但了解它们各自的作用以及为什么 Pandas 中通常避免在单个访问器（如 `.loc` 或 `.iloc`）中直接混合基于标签和基于位置的索引，这一点很重要。

尽管同时通过行标签和列的整数位置（或反之）来指定数据可能看似直观，但 Pandas 有意将这些方法分开，以保持清晰度和可预测性。

### 为什么不建议直接混合使用

主要原因是模糊不清。让我们看看为什么尝试混合使用不会如预期那样起作用：

1. **`.loc` 期望标签：** `.loc` 访问器是*专门*为基于标签的索引而设计的。如果你提供一个整数，例如 `df.loc['row_label', 0]`，`.loc` 会将 `0` 解释为*列标签*，而不是整数位置。只有当你的列实际命名为 `0` 时，这才会起作用。否则，它通常会引发 `KeyError` 错误。唯一的例外是当*索引本身*由整数组成时；在这种情况下，`df.loc[0]` 将选择标签为 `0` 的行。但即便如此，解释也是基于*标签*，而不是位置。
2. **`.iloc` 期望整数：** 相反地，`.iloc` 是*专门*为基于整数位置的索引而设计的。它根据行和列的零起始位置进行操作，无论它们的标签是什么。如果你提供一个标签，例如 `df.iloc[0, 'column_label']`，`.iloc` 将无法理解字符串标签，并将引发 `TypeError` 错误。

### 历史背景：已弃用的 `.ix`

在 Pandas 的旧版本中，有一个名为 `.ix` 的索引器，它试图同时处理基于标签和基于整数的索引，允许混合类型。然而，这导致了一些难以察觉的错误和令人困惑的代码，因为它的行为可能会根据 DataFrame 的索引是否包含整数而改变。例如，如果索引包含整数标签，`df.ix[0]` 可能会通过标签选择；如果索引不包含整数，则会通过位置选择。这种模糊性使得代码更难阅读和调试。

由于这些问题，`.ix` 被弃用。`.loc`（仅限标签）和 `.iloc`（仅限整数）之间的明确分离提供了一种更清晰、更易于维护的数据选择方式。

### 安全地实现“混合”选择结果

那么，当你已知一个轴（行）的标签和另一个轴（列）的整数位置，或者反之，该如何选择数据呢？你不应在一次 `.loc` 或 `.iloc` 调用中混合使用它们。相反，你应该一致地使用一个访问器，并可能转换另一个轴的标识符，或者通过链式操作实现。

让我们使用一个示例 DataFrame：

```python
import pandas as pd
import numpy as np

data = {'Score': [85, 92, 78, 88, 95],
        'Attempts': [1, 3, 2, 3, 1],
        'Grade': ['B', 'A', 'C', 'B', 'A']}
index_labels = ['Student1', 'Student2', 'Student3', 'Student4', 'Student5']
df = pd.DataFrame(data, index=index_labels)

print(df)
```

Output:

```
          Score  Attempts Grade
Student1     85         1     B
Student2     92         3     A
Student3     78         2     C
Student4     88         3     B
Student5     95         1     A
```

#### 场景 1：按行标签和列位置选择

假设你想要获取 `Student3`（标签）在*第二列*（位置 1，即 'Attempts'）的数据。

- **选项 A：使用 `.loc`（将位置转换为标签）**
  首先找到位置 1 处列的*标签*。

  ```python

  col_label = df.columns[1]
  print(f"位置 1 处的列标签: {col_label}")

  value = df.loc['Student3', col_label]
  print(f"Student3，列位置 1 处的值: {value}")
  ```

  Output:

  ```
  位置 1 处的列标签: Attempts
  Student3，列位置 1 处的值: 2
  ```
- **选项 B：使用 `.iloc`（将标签转换为位置）**
  首先找到行标签 `Student3` 的*整数位置*。

  ```python

  row_pos = df.index.get_loc('Student3')
  print(f"Student3 的行位置: {row_pos}")

  value = df.iloc[row_pos, 1]
  print(f"Student3，列位置 1 处的值: {value}")
  ```

  Output:

  ```
  Student3 的行位置: 2
  Student3，列位置 1 处的值: 2
  ```
- **选项 C：链式选择**
  首先按标签选择行，然后从结果 Series 中按位置选择元素。对于大型 DataFrame，这有时效率较低，但通常易于阅读。

  ```python
  value = df.loc['Student3'].iloc[1]
  print(f"使用链式选择的值: {value}")
  ```

  Output:

  ```
  使用链式选择的值: 2
  ```

#### 场景 2：按行位置和列标签选择

现在，假设你想要获取*第四行*（位置 3）和标签为 `'Grade'` 的列的数据。

- **选项 A：使用 `.iloc`（将标签转换为位置）**
  找到列标签 `'Grade'` 的*整数位置*。

  ```python

  col_pos = df.columns.get_loc('Grade')
  print(f"Grade 的列位置: {col_pos}")

  value = df.iloc[3, col_pos]
  print(f"行位置 3，列 Grade 处的值: {value}")
  ```

  Output:

  ```
  Grade 的列位置: 2
  行位置 3，列 Grade 处的值: B
  ```
- **选项 B：使用 `.loc`（将位置转换为标签）**
  找到位置 3 处行的*标签*。

  ```python

  row_label = df.index[3]
  print(f"位置 3 处的行标签: {row_label}")

  value = df.loc[row_label, 'Grade']
  print(f"行位置 3，列 Grade 处的值: {value}")
  ```

  Output:

  ```
  位置 3 处的行标签: Student4
  行位置 3，列 Grade 处的值: B
  ```
- **选项 C：链式选择**
  首先按标签选择列，然后从结果 Series 中按位置选择元素。

  ```python
  value = df['Grade'].iloc[3]
  print(f"使用链式选择的值: {value}")
  ```

  Output:

  ```
  使用链式选择的值: B
  ```

### 总结

尽管混合使用标签和整数索引的想法看起来很方便，但 Pandas 通过 `.loc` 和 `.iloc` 强制分离是有充分理由的，主要为了清晰度和可预测性。当你需要使用标签和位置信息的组合来选择数据时，选择与你的主要标识符匹配的访问器（`.loc` 或 `.iloc`），并使用辅助方法如 `df.columns.get_loc()`、`df.index.get_loc()`、`df.columns[]` 或 `df.index[]` 根据需要转换另一个标识符，或者使用链式选择。这种明确的方法使你的代码更容易理解，并且不易出错。

获取即时帮助、个性化解释和交互式代码示例。

---

### 条件选择 (布尔索引)

# 条件选择 (布尔索引)

为了根据列中的特定 *值* 筛选数据，可以使用一种称为 **布尔索引** 的方法。这种方法允许您选择符合特定条件的行，例如查找所有“年龄”列大于 30 的条目，或者“城市”列是“New York”的条目。虽然通过标签 (`.loc`) 或整数位置 (`.iloc`) 选择数据等方法提供了直接访问，但布尔索引提供了根据数据内容进行查询的灵活性。

其核心是，布尔索引涉及根据条件创建一个布尔序列（一个只包含 `True` 或 `False` 值的序列），然后使用该序列从 DataFrame 中选择行。

### 如何运作：创建布尔序列

让我们从一个简单的 DataFrame 开始。假设我们有一些关于个人的数据，包括他们的年龄：

```python
import pandas as pd
import numpy as np

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
        'Age': [24, 27, 22, 32, 29, 18],
        'City': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Los Angeles', 'Chicago'],
        'Salary': [70000, 80000, 65000, 90000, 85000, 60000]}
df = pd.DataFrame(data)

print(df)
```

输出：

```
      Name  Age         City  Salary
0    Alice   24     New York   70000
1      Bob   27  Los Angeles   80000
2  Charlie   22     New York   65000
3    David   32      Chicago   90000
4      Eve   29  Los Angeles   85000
5    Frank   18      Chicago   60000
```

现在，假设我们想找出哪些人的年龄大于 25 岁。我们可以直接将比较应用于“Age”列：

```python
condition = df['Age'] > 25
print(condition)
```

输出：

```
0    False
1     True
2    False
3     True
4     True
5    False
Name: Age, dtype: bool
```

如您所见，Pandas 对整个“Age”序列逐元素执行比较（`> 25`），生成一个新的布尔值序列。`True` 表示该行满足条件，而 `False` 表示不满足。

### 使用布尔序列进行选择

这个布尔序列就像一个过滤器。您可以将其直接传递到 DataFrame 的方括号 `[]` 中（或 `.loc` 中），以仅选择条件评估为 `True` 的行：

```python

older_than_25 = df[df['Age'] > 25]
print(older_than_25)
```

输出：

```
    Name  Age         City  Salary
1    Bob   27  Los Angeles   80000
3  David   32      Chicago   90000
4    Eve   29  Los Angeles   85000
```

只返回布尔序列中对应 `True` 的行（索引 1、3 和 4）。

这对于其他类型的条件也类似适用，例如字符串比较：

```python

in_new_york = df[df['City'] == 'New York']
print(in_new_york)
```

输出：

```
      Name  Age      City  Salary
0    Alice   24  New York   70000
2  Charlie   22  New York   65000
```

### 组合多个条件

“数据分析通常需要同时根据多个条件进行筛选。您可以使用逻辑运算符组合布尔条件：”

- `&` : 逐元素逻辑与
- `|` : 逐元素逻辑或
- `~` : 逐元素逻辑非

**重要提示：** 组合条件时，您**必须**将每个单独的条件用括号 `()` 括起来，因为这涉及到 Python 的运算符优先级规则。使用 Python 标准的 `and`、`or`、`not` 运算符将导致错误或意外行为，因为它们作用于整个对象的真值，而非逐元素操作。

#### 示例：AND (`&`)

让我们找出居住在“Los Angeles”*且*年龄大于 25 岁的人：

```python

combined_condition_and = (df['City'] == 'Los Angeles') & (df['Age'] > 25)

print(df[combined_condition_and])
```

输出：

```
  Name  Age         City  Salary
1  Bob   27  Los Angeles   80000
4  Eve   29  Los Angeles   85000
```

只有 Bob 和 Eve 满足这两个条件。

#### 示例：OR (`|`)

现在，让我们找出居住在“Chicago”*或*薪水大于 80000 的人：

```python

combined_condition_or = (df['City'] == 'Chicago') | (df['Salary'] > 80000)

print(df[combined_condition_or])
```

输出：

```
    Name  Age         City  Salary
3  David   32      Chicago   90000
4    Eve   29  Los Angeles   85000
5  Frank   18      Chicago   60000
```

David 和 Frank 住在 Chicago。Eve 不住在 Chicago，但她的薪水大于 80000。这三个人都被包括在内。

#### 示例：NOT (`~`)

要选择不满足条件的行，请使用波浪号 `~` 运算符。让我们找出所有不居住在“New York”的人：

```python

not_in_new_york = df[~(df['City'] == 'New York')]
print(not_in_new_york)
```

输出：

```
    Name  Age         City  Salary
1    Bob   27  Los Angeles   80000
3  David   32      Chicago   90000
4    Eve   29  Los Angeles   85000
5  Frank   18      Chicago   60000
```

### 将布尔索引与 `.loc` 结合使用

布尔索引也适用于 `.loc` 访问器。这会特别实用和易读，尤其当您希望根据行条件选择特定列时：

```python

older_names_salaries = df.loc[df['Age'] > 25, ['Name', 'Salary']]
print(older_names_salaries)
```

输出：

```
    Name  Salary
1    Bob   80000
3  David   90000
4    Eve   85000
```

在这里，`.loc` 的第一个参数 (parameter)是选择行的布尔条件，第二个参数指定了为这些行检索的列。

### 记住以下几点

- **布尔序列：** 将条件应用于 DataFrame 列会创建一个布尔序列（`True`/`False`）。
- **筛选：** 在 `[]` 或 `.loc` 中使用此布尔序列来筛选 DataFrame。
- **逻辑运算符：** 使用 `&`（与）、`|`（或）和 `~`（非）组合条件。
- **括号不可少：** 使用 `&` 或 `|` 时，务必将每个单独的条件用括号 `()` 括起来。示例：`(condition1) & (condition2)`。
- **`.loc` 结合：** 布尔数组可以非常有效地与 `.loc` 结合使用，以便根据条件同时选择行和特定列。

条件选择是数据子集划分的一种基本方法。它允许您隔离满足您条件的特定观察值，为更集中的分析和操作任务奠定基础。

获取即时帮助、个性化解释和交互式代码示例。

---

### 设置 DataFrame 索引

# 设置 DataFrame 索引

DataFrame 通常包含可用作每行唯一标识的列，例如产品 ID、用户名称、时间戳或国家代码。尽管存在通过整数位置或列名选择数据的方法，但在存在这些自然标识符时，使用默认的 `0, 1, 2, ...` 数字索引来访问特定行通常不如直接使用这些标识符直观。

Pandas 允许您将一个或多个现有列指定为 DataFrame 的索引。这能让使用 `.loc` 选择数据时更符合逻辑，并且有时能提升连接或查找等操作的性能。

### 使用 `set_index()` 方法

修改索引的主要工具是 `.set_index()` 方法。它的基本用法是指定您希望用作新索引的列（或多个列）。

让我们创建一个代表产品数据的简单 DataFrame：

```python
import pandas as pd

data = {'ProductID': ['P101', 'P102', 'P103', 'P104'],
        'ProductName': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
        'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics'],
        'Price': [1200, 25, 75, 300]}
products_df = pd.DataFrame(data)

print("Original DataFrame:")
print(products_df)
print("\nOriginal Index:")
print(products_df.index)
```

输出：

```
Original DataFrame:
  ProductID ProductName     Category  Price
0      P101      Laptop  Electronics   1200
1      P102       Mouse  Accessories     25
2      P103    Keyboard  Accessories     75
3      P104     Monitor  Electronics    300

Original Index:
RangeIndex(start=0, stop=4, step=1)
```

请注意默认的 `RangeIndex`。现在，我们将 `ProductID` 列设为索引：

```python

products_indexed = products_df.set_index('ProductID')

print("\nDataFrame with ProductID as Index:")
print(products_indexed)
print("\nNew Index:")
print(products_indexed.index)
```

输出：

```
DataFrame with ProductID as Index:
          ProductName     Category  Price
ProductID
P101           Laptop  Electronics   1200
P102            Mouse  Accessories     25
P103         Keyboard  Accessories     75
P104          Monitor  Electronics    300

New Index:
Index(['P101', 'P102', 'P103', 'P104'], dtype='object', name='ProductID')
```

请注意以下几个要点：

1. `ProductID` 列不再是普通数据列；它已成为索引。
2. 索引类型现在是一个包含产品 ID 的 `Index` 对象。该索引也保留了原始列的名称（'ProductID'）。
3. 默认情况下，`.set_index()` 返回一个带有修改后索引的*新* DataFrame。原始的 `products_df` 保持不变。

### 使用新索引选择数据

设置有意义索引的主要好处是使用 `.loc` 进行数据选择时有所改进。现在，您可以使用索引标签（产品 ID）来代替整数位置：

```python

product_info = products_indexed.loc['P103']
print("\nData for P103:")
print(product_info)

selected_products = products_indexed.loc[['P101', 'P104'], ['ProductName', 'Price']]
print("\nSelected data for P101 and P104:")
print(selected_products)
```

输出：

```
Data for P103:
ProductName    Keyboard
Category    Accessories
Price                75
Name: P103, dtype: object

Selected data for P101 and P104:
          ProductName  Price
ProductID
P101           Laptop   1200
P104          Monitor    300
```

这种使用 `.loc` 基于标签的选择方式，与依赖整数位置相比，通常使代码更易读，且更不易出错，尤其是在 DataFrame 之后可能被排序或过滤的情况下。请记住，无论索引标签如何，`.iloc` 仍使用整数位置 (0, 1, 2, ...)。

Index\_Transformation

cluster\_before

setᵢndex('ProductID') 之前

clusterₐfter

setᵢndex('ProductID') 之后

tbl\_before

产品ID

产品名称

类别

价格

0

P101

Laptop

Electronics

1200

1

P102

Mouse

Accessories

25

2

P103

Keyboard

Accessories

75

3

P104

Monitor

Electronics

300

tblₐfter

产品ID

产品名称

类别

价格

P101

Laptop

Electronics

1200

P102

Mouse

Accessories

25

P103

Keyboard

Accessories

75

P104

Monitor

Electronics

300

tbl\_before->tblₐfter

  .setᵢndex('ProductID')

> DataFrame 索引从默认的 RangeIndex 变为使用 'ProductID' 列。

### `inplace` 参数 (parameter)

如果您确定要直接修改原始 DataFrame，可以使用 `inplace=True` 参数。

```python

products_df_copy = products_df.copy()

print("\nDataFrame before inplace modification:")
print(products_df_copy.index)

products_df_copy.set_index('ProductID', inplace=True)

print("\nDataFrame after inplace modification:")
print(products_df_copy.index)
```

输出：

```
DataFrame before inplace modification:
RangeIndex(start=0, stop=4, step=1)

DataFrame after inplace modification:
Index(['P101', 'P102', 'P103', 'P104'], dtype='object', name='ProductID')
```

使用 `inplace=True` 可以为大型 DataFrame 节省内存，因为它避免创建副本。然而，通常认为更好的做法是，尤其对于初学者，避免 `inplace` 操作。将结果赋值给新变量（或重新赋值给同一个变量，例如 `products_df = products_df.set_index('ProductID')`）能使数据流更清晰，更便于调试。

### 设置多级索引（层级索引）

您也可以将多个列设为索引，创建一个 `MultiIndex` 或层级索引。当列值的组合能够唯一标识行时，这会很有用。

我们稍微修改一下示例：

```python
data_multi = {'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics'],
              'ProductID': ['P101', 'P102', 'P103', 'P104', 'P105'],
              'ProductName': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Webcam'],
              'Price': [1200, 25, 75, 300, 50]}
products_multi_df = pd.DataFrame(data_multi)

products_multi_indexed = products_multi_df.set_index(['Category', 'ProductID'])

print("\nDataFrame with MultiIndex:")
print(products_multi_indexed)
print("\nNew MultiIndex:")
print(products_multi_indexed.index)
```

输出：

```
DataFrame with MultiIndex:
                         ProductName  Price
Category    ProductID
Electronics P101              Laptop   1200
Accessories P102               Mouse     25
            P103            Keyboard     75
Electronics P104             Monitor    300
            P105              Webcam     50

New MultiIndex:
MultiIndex([('Electronics', 'P101'),
            ('Accessories', 'P102'),
            ('Accessories', 'P103'),
            ('Electronics', 'P104'),
            ('Electronics', 'P105')],
           names=['Category', 'ProductID'])
```

现在索引有了多个级别。使用 `.loc` 从多级索引 DataFrame 中选择数据通常需要提供一个索引标签元组：

```python

keyboard_info = products_multi_indexed.loc[('Accessories', 'P103')]
print("\nData for ('Accessories', 'P103'):")
print(keyboard_info)

electronics_products = products_multi_indexed.loc['Electronics']
print("\nData for 'Electronics' Category:")
print(electronics_products)
```

输出：

```
Data for ('Accessories', 'P103'):
ProductName    Keyboard
Price                75
Name: (Accessories, P103), dtype: object

Data for 'Electronics' Category:
          ProductName  Price
ProductID
P101           Laptop   1200
P104          Monitor    300
P105           Webcam     50
```

层级索引是一项功能强劲的特点，尽管从多级索引中选择数据可能会有比这些示例更复杂的变体。

设置合适的索引是构建数据，以便在 Pandas 中进行高效分析和选择的必要一步。在下一节中，我们将介绍如何使用 `reset_index()` 来反转此过程。

获取即时帮助、个性化解释和交互式代码示例。

---

### 重置 DataFrame 索引

# 重置 DataFrame 索引

Pandas DataFrame 经常使用自定义索引，这通常是通过使用 `.set_index()` 将一个或多个列提升为索引来建立的。这种方法对于使用 `.loc` 根据有意义的标签选择数据通常很有用。然而，有时您可能需要转换自定义的 DataFrame 索引。这可能涉及将索引标签转回为普通数据列，或许是为了数据处理，或者仅仅是为了恢复默认的整数索引 (0, 1, 2, ...)。Pandas 为此提供了一个直接的方法：`.reset_index()`。

### 使用 `reset_index` 还原索引

`.reset_index()` 方法实质上是 `.set_index()` 的逆操作。它将当前索引级别移入 DataFrame 作为新列。默认情况下，它还会用一个简单的默认整数索引替换现有索引。

我们来看看实际操作。首先，我们将创建一个示例 DataFrame 并将其一列设为索引：

```python
import pandas as pd

data = {'City': ['Austin', 'Dallas', 'Houston', 'San Antonio'],
        'State': ['TX', 'TX', 'TX', 'TX'],
        'Population': [961855, 1304379, 2304580, 1434625]}
df = pd.DataFrame(data)

df_indexed = df.set_index('City')

print("以 'City' 为索引的 DataFrame：")
print(df_indexed)
```

这将输出：

```
以 'City' 为索引的 DataFrame：
            State  Population
City
Austin         TX      961855
Dallas         TX     1304379
Houston        TX     2304580
San Antonio    TX     1434625
```

请注意，'City' 不再是普通列，而是作为左侧的索引标签。

现在，我们使用 `.reset_index()` 将 'City' 索引转回为列：

```python

df_reset = df_indexed.reset_index()

print("\n重置索引后的 DataFrame：")
print(df_reset)
```

输出清楚地显示了变化：

```
重置索引后的 DataFrame：
          City State  Population
0       Austin    TX      961855
1       Dallas    TX     1304379
2      Houston    TX     2304580
3  San Antonio    TX     1434625
```

如您所见，`reset_index()` 执行了两个主要操作：

1. 它将索引 ('City') 移回 DataFrame 作为普通列。
2. 它创建了一个新的默认整数索引 (0, 1, 2, 3)。

### 控制重置操作

`.reset_index()` 方法提供参数 (parameter)来控制其行为，主要是 `drop` 和 `inplace`。

#### `drop` 参数

有时，您可能希望完全丢弃旧索引，而不是将其转为列。如果索引值在重置后是多余的或不再需要，这会很有用。您可以使用 `drop=True` 参数实现这一点。

```python

df_dropped_index = df_indexed.reset_index(drop=True)

print("\n重置并丢弃索引后的 DataFrame：")
print(df_dropped_index)
```

输出：

```
重置并丢弃索引后的 DataFrame：
  State  Population
0    TX      961855
1    TX     1304379
2    TX     2304580
3    TX     1434625
```

这里，'City' 列已消失，我们只剩下 'State' 和 'Population' 列以及新的默认整数索引。如前所述，默认行为对应于 `drop=False`。

#### `inplace` 参数

与许多 Pandas 方法一样，`.reset_index()` 也有一个 `inplace` 参数。

- `inplace=False` (默认值)：此方法返回一个索引已重置的*新* DataFrame，不改变原始 DataFrame。您通常将此新 DataFrame 赋值给一个变量（如我们对 `df_reset` 和 `df_dropped_index` 所做的那样）。
- `inplace=True`：此方法直接修改*原始* DataFrame 并返回 `None`。不会创建新的 DataFrame。

使用 `inplace=True` 可以为非常大的 DataFrame 节省内存，因为它避免了创建副本，但请谨慎使用，因为它会直接修改您的数据。

```python

df_indexed_copy = df_indexed.copy()

print("\n原地重置前的原始 DataFrame：")
print(df_indexed_copy)

result = df_indexed_copy.reset_index(inplace=True)

print("\n原地重置后的 DataFrame：")
print(df_indexed_copy)
print("\n原地重置的返回值：", result)
```

输出：

```
原地重置前的原始 DataFrame：
            State  Population
City
Austin         TX      961855
Dallas         TX     1304379
Houston        TX     2304580
San Antonio    TX     1434625

原地重置后的 DataFrame：
          City State  Population
0       Austin    TX      961855
1       Dallas    TX     1304379
2      Houston    TX     2304580
3  San Antonio    TX     1434625

原地重置的返回值： None
```

正如预期，`df_indexed_copy` 被直接修改，并且该方法返回 `None`。

### 何时重置索引

在数据清理和准备过程中，重置索引是常见的操作。一些常见情况包括：

- **过滤后：** 当您从 DataFrame 过滤行时，剩余的索引值可能会变得不连续（例如，0, 2, 5, 8）。重置索引可获得一个整齐的连续索引。
- **准备合并/连接：** 某些组合操作在默认整数索引下运行更顺畅或更符合预期，特别是当原始索引没有有意义的对齐 (alignment)时。
- **将索引值用作数据：** 您可能需要将之前存储在索引中的信息作为普通列用于计算、绘图或进一步分析。
- **简化 MultiIndex：** 如果您有一个带有分层索引 (MultiIndex) 的 DataFrame，`reset_index()` 可以将索引的一个或多个级别平展回列中，从而简化 DataFrame 结构。

了解如何设置和重置索引为在 Pandas DataFrame 中访问和组织数据提供了很大的灵活性。熟练使用 `.set_index()` 和 `.reset_index()` 可让您为特定的数据处理或分析任务选择最方便的数据表示形式。

获取即时帮助、个性化解释和交互式代码示例。

---

### 动手实践：访问特定数据子集

# 动手实践：访问特定数据子集

访问 Pandas DataFrame 中的特定数据子集可以通过主要方法实现，例如按名称选择列，使用 `.loc` 进行基于标签的索引，使用 `.iloc` 进行基于位置的索引，以及使用布尔条件筛选数据。实际练习将巩固对这些方法的理解，并增强应用这些技术的信心。

首先，确保你已导入 Pandas。通常，我们将其以别名 `pd` 导入。

```python
import pandas as pd
import numpy as np
```

接下来，我们创建一个示例 DataFrame 进行操作。我们将构建一个表示不同类型水果信息的小型数据集。

```python

data = {
    'Fruit': ['Apple', 'Banana', 'Orange', 'Grape', 'Strawberry', 'Blueberry'],
    'Color': ['Red', 'Yellow', 'Orange', 'Purple', 'Red', 'Blue'],
    'Price': [1.2, 0.5, 0.8, 4.5, 3.0, 5.5],
    'Quantity': [100, 150, 80, 200, 120, 90]
}

index_labels = ['F01', 'F02', 'F03', 'F04', 'F05', 'F06']

inventory_df = pd.DataFrame(data, index=index_labels)

print("原始水果库存 DataFrame:")
print(inventory_df)
```

这将输出：

```
Original Fruit Inventory DataFrame:
         Fruit   Color  Price  Quantity
F01      Apple     Red    1.2       100
F02     Banana  Yellow    0.5       150
F03     Orange  Orange    0.8        80
F04      Grape  Purple    4.5       200
F05 Strawberry     Red    3.0       120
F06  Blueberry    Blue    5.5        90
```

现在，让我们练习从 `inventory_df` 中选择数据。

### 任务 1：选择列

选择 `Fruit` 列。请记住，访问单个列会返回一个 Pandas Series。

```python

fruit_column = inventory_df['Fruit']
print("\n选择 'Fruit' 列（返回 Series）：")
print(fruit_column)
```

```
Selecting the 'Fruit' column (returns a Series):
F01         Apple
F02        Banana
F03        Orange
F04         Grape
F05    Strawberry
F06     Blueberry
Name: Fruit, dtype: object
```

现在，同时选择 `Fruit` 和 `Price` 列。访问多个列会返回一个 DataFrame。注意这里使用了双层方括号 `[[]]`。

```python

fruit_price_df = inventory_df[['Fruit', 'Price']]
print("\n选择 'Fruit' 和 'Price' 列（返回 DataFrame）：")
print(fruit_price_df)
```

```
Selecting 'Fruit' and 'Price' columns (returns a DataFrame):
         Fruit  Price
F01      Apple    1.2
F02     Banana    0.5
F03     Orange    0.8
F04      Grape    4.5
F05 Strawberry    3.0
F06  Blueberry    5.5
```

### 任务 2：使用 `.loc` 选择（基于标签）

选择标签为 `F03`（橙子）对应的行。

```python

orange_row = inventory_df.loc['F03']
print("\n使用 .loc 选择标签为 'F03' 的行：")
print(orange_row)
```

```
Selecting row with label 'F03' using .loc:
Fruit       Orange
Color       Orange
Price          0.8
Quantity        80
Name: F03, dtype: object
```

选择标签为 `F05`（草莓）的水果的 `Price`。

```python

strawberry_price = inventory_df.loc['F05', 'Price']
print(f"\n使用 .loc 获取水果 'F05' 的价格：{strawberry_price}")
```

```
Price of fruit 'F05' using .loc: 3.0
```

选择标签从 `F02` 到 `F04`（包含）的行，并且只选择 `Fruit` 和 `Quantity` 列。

```python

subset_loc = inventory_df.loc['F02':'F04', ['Fruit', 'Quantity']]
print("\n使用 .loc 选择 'F02' 到 'F04' 的行和特定列：")
print(subset_loc)
```

```
Selecting rows 'F02' to 'F04' and specific columns using .loc:
      Fruit  Quantity
F02  Banana       150
F03  Orange        80
F04   Grape       200
```

注意 `.loc` 如何在切片中包含结束标签（`F04` 被包含在内）。

### 任务 3：使用 `.iloc` 选择（基于位置）

选择第三行（对应于橙子，索引位置为 2，因为索引从 0 开始）。

```python

third_row_iloc = inventory_df.iloc[2]
print("\n使用 .iloc 选择第三行（索引 2）：")
print(third_row_iloc)
```

```
Selecting the third row (index 2) using .iloc:
Fruit       Orange
Color       Orange
Price          0.8
Quantity        80
Name: F03, dtype: object
```

选择第 4 行（索引 3）和第 2 列（索引 1）交叉处的值。这应该是颜色 'Purple'。

```python

value_iloc = inventory_df.iloc[3, 1]
print(f"\n使用 .iloc 获取行索引 3，列索引 1 处的值：{value_iloc}")
```

```
Value at row index 3, column index 1 using .iloc: Purple
```

选择前三行（索引 0, 1, 2）和前两列（索引 0, 1）。

```python

subset_iloc = inventory_df.iloc[0:3, 0:2]
print("\n使用 .iloc 选择前 3 行和前 2 列：")
print(subset_iloc)
```

```
Selecting first 3 rows and first 2 columns using .iloc:
     Fruit   Color
F01   Apple     Red
F02  Banana  Yellow
F03  Orange  Orange
```

请记住，使用 `.iloc` 进行切片时会排除结束位置，这与标准的 Python 列表切片类似。

### 任务 4：条件选择（布尔索引）

选择所有红色的水果。

```python

red_fruits_condition = inventory_df['Color'] == 'Red'
print("\n'Color' == 'Red' 的布尔 Series：")
print(red_fruits_condition)

red_fruits_df = inventory_df[red_fruits_condition]
print("\n选择颜色为 'Red' 的行：")
print(red_fruits_df)
```

```
Boolean Series for 'Color' == 'Red':
F01     True
F02    False
F03    False
F04    False
F05     True
F06    False
Name: Color, dtype: bool

Selecting rows where Color is 'Red':
         Fruit Color  Price  Quantity
F01      Apple   Red    1.2       100
F05 Strawberry   Red    3.0       120
```

选择所有 `Price` 大于 1.001.001.00 且 `Quantity` 小于 150 的水果。使用 `&` 运算符组合条件。

```python

price_condition = inventory_df['Price'] > 1.0

quantity_condition = inventory_df['Quantity'] < 150

combined_condition = price_condition & quantity_condition

filtered_df = inventory_df[combined_condition]
print("\n选择 Price > 1.0 且 Quantity < 150 的行：")
print(filtered_df)
```

```
Selecting rows where Price > 1.0 AND Quantity < 150:
         Fruit Color  Price  Quantity
F01      Apple   Red    1.2       100
F05 Strawberry   Red    3.0       120
F06  Blueberry  Blue    5.5        90
```

选择所有红色或 `Price` 小于 0.600.600.60 的水果的 `Fruit` 名称和 `Price`。使用 `|` 运算符。

```python

color_condition = inventory_df['Color'] == 'Red'

price_condition_low = inventory_df['Price'] < 0.6

combined_or_condition = color_condition | price_condition_low

filtered_subset = inventory_df.loc[combined_or_condition, ['Fruit', 'Price']]
print("\n选择颜色为 'Red' 或 Price < 0.6 的 'Fruit' 和 'Price'：")
print(filtered_subset)
```

```
Selecting 'Fruit' and 'Price' where Color is 'Red' OR Price < 0.6:
         Fruit  Price
F01      Apple    1.2
F02     Banana    0.5
F05 Strawberry    3.0
```

注意我们如何将布尔索引与 `.loc` 结合使用，以选择筛选后行的特定列。

### 任务 5：设置和重置索引

有时，将其中一列设置为 DataFrame 索引很有用，特别是当它包含唯一标识符时。让我们将 `Fruit` 列设置为索引。

```python

inventory_df_fruit_index = inventory_df.set_index('Fruit')
print("\n以 'Fruit' 为索引的 DataFrame：")
print(inventory_df_fruit_index)
```

```
DataFrame with 'Fruit' as index:
             Color  Price  Quantity
Fruit
Apple          Red    1.2       100
Banana      Yellow    0.5       150
Orange      Orange    0.8        80
Grape       Purple    4.5       200
Strawberry     Red    3.0       120
Blueberry     Blue    5.5        90
```

现在，尝试使用新的索引，通过 `.loc` 选择 'Orange' 行。

```python

orange_data = inventory_df_fruit_index.loc['Orange']
print("\n使用新水果索引选择 'Orange' 数据：")
print(orange_data)
```

```
Selecting 'Orange' data using the new fruit index:
Color       Orange
Price          0.8
Quantity        80
Name: Orange, dtype: object
```

最后，让我们将索引重置回默认的整数索引，将 'Fruit' 索引变回一个普通列。

```python

inventory_df_reset = inventory_df_fruit_index.reset_index()
print("\n重置索引后的 DataFrame：")
print(inventory_df_reset)
```

```
DataFrame after resetting the index:
        Fruit   Color  Price  Quantity
0       Apple     Red    1.2       100
1      Banana  Yellow    0.5       150
2      Orange  Orange    0.8        80
3       Grape  Purple    4.5       200
4  Strawberry     Red    3.0       120
5   Blueberry    Blue    5.5        90
```

你可以看到 `Fruit` 列又回来了，并且添加了一个默认的整数索引（0, 1, 2...）。

本次动手操作提供了在 Pandas 中应用主要数据选择技术的练习。熟练使用 `[]` 进行列选择、`.loc` 进行基于标签的访问、`.iloc` 进行基于位置的访问以及布尔索引进行条件筛选，对于几乎任何数据分析任务都非常重要。你可以使用 `inventory_df` DataFrame 进行更多练习，或者尝试加载你自己的数据集（如前一章所学），并练习选择不同的数据子集。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 8 Basic Data Manipulation Pandas

### 识别缺失数据

# 识别缺失数据

数据集很少是完整的。它们常有空缺，以缺失值表示。在分析数据或训练模型之前，你需要了解这些空缺的位置及可能有多大范围。Pandas 提供了直接的工具来发现缺失数据，传统上用特殊浮点值 NaNNaNNaN（非数字）来标记 (token)。Python 的 `None` 对象在 Pandas 对象中也视为缺失数据。

### 使用 `isnull()` 和 `notnull()` 识别缺失值

Pandas 提供了两种主要方法来识别缺失值：

1. `isnull()`: 返回一个与原对象（Series 或 DataFrame）大小相同的布尔对象，其中 `True` 表示缺失值（NaNNaNNaN 或 `None`），`False` 表示非缺失值。
2. `notnull()`: 与 `isnull()` 相反。它对非缺失值返回 `True`，对缺失值返回 `False`。

我们来看看这些方法的实际运用。首先，我们需要导入 pandas 和 numpy。

```python
import pandas as pd
import numpy as np
```

现在，我们创建一个简单的 Pandas Series，其中包含一些由 `np.nan` 表示的缺失数据：

```python

data_series = pd.Series([1, np.nan, 3.5, np.nan, 7])
print("原始 Series:")
print(data_series)
```

```text
原始 Series:
0    1.0
1    NaN
2    3.5
3    NaN
4    7.0
dtype: float64
```

现在，我们可以使用 `isnull()` 创建一个布尔掩码来识别 NaNNaNNaN 值的位置：

```python

missing_mask = data_series.isnull()
print("\nisnull() 生成的布尔掩码:")
print(missing_mask)
```

```text
isnull() 生成的布尔掩码:
0    False
1     True
2    False
3     True
4    False
dtype: bool
```

可以看到，生成的 Series 在索引 1 和 3 处包含 `True`，这与原始 `data_series` 中的 NaNNaNNaN 值相对应。

反过来，`notnull()` 识别非缺失值：

```python

not_missing_mask = data_series.notnull()
print("\nnotnull() 生成的布尔掩码:")
print(not_missing_mask)
```

```text
notnull() 生成的布尔掩码:
0     True
1    False
2     True
3    False
4     True
dtype: bool
```

这会在数据存在的地方返回 `True`，在数据缺失的地方返回 `False`。

*(注意：你可能还会遇到别名 `isna()`（对应 `isnull()`）和 `notna()`（对应 `notnull()`）。它们执行完全相同的功能。)*

### 识别 DataFrames 中的缺失值

这些方法在 DataFrames 上的作用相似，但它们返回的是一个布尔 DataFrame 而非 Series。

我们来创建一个包含缺失值的 DataFrame：

```python

data = {'col_a': [1, 2, np.nan, 4, 5],
        'col_b': [np.nan, 7, 8, np.nan, 10],
        'col_c': [11, 12, 13, 14, 15],
        'col_d': ['apple', 'banana', 'orange', np.nan, 'grape']}
df = pd.DataFrame(data)

print("原始 DataFrame:")
print(df)
```

```text
原始 DataFrame:
   col_a  col_b  col_c     col_d
0    1.0    NaN     11     apple
1    2.0    7.0     12    banana
2    NaN    8.0     13    orange
3    4.0    NaN     14       NaN
4    5.0   10.0     15     grape
```

将 `isnull()` 应用于此 DataFrame 会得到：

```python

missing_df_mask = df.isnull()
print("\nisnull() 生成的布尔 DataFrame 掩码:")
print(missing_df_mask)
```

```text
isnull() 生成的布尔 DataFrame 掩码:
   col_a  col_b  col_c  col_d
0  False   True  False  False
1  False  False  False  False
2   True  False  False  False
3  False   True  False   True
4  False  False  False  False
```

这个布尔 DataFrame 直接映射了原始 `df` 中缺失值的位置。

### 计数缺失值

虽然看到缺失值的确切位置有用，但你经常需要一个概况。总共有多少缺失值，或者每列有多少？通过对 `isnull()` 的结果求和，你可以轻松做到这一点，因为在数值上下文 (context)中，`True` 被视为 1，`False` 被视为 0。

要计算每列中的缺失值数量：

```python

missing_counts_per_column = df.isnull().sum()
print("\n每列的缺失值数量:")
print(missing_counts_per_column)
```

```text
每列的缺失值数量:
col_a    1
col_b    2
col_c    0
col_d    1
dtype: int64
```

这是一个非常常见的操作。它快速告诉你 `col_a` 有一个缺失值，`col_b` 有两个，`col_c` 没有，`col_d` 有一个。

要获取整个 DataFrame 中缺失值的总数，你可以对结果求和两次：

```python

total_missing_count = df.isnull().sum().sum()
print(f"\nDataFrame 中缺失值的总数: {total_missing_count}")
```

```text
DataFrame 中缺失值的总数: 4
```

识别数据缺失的位置和数量是数据清洗过程中首要的步骤。一旦你使用 `isnull()` 和 `sum()` 等方法识别了这些空缺，你就可以着手决定如何处理它们，这也是接下来几节的重点。

获取即时帮助、个性化解释和交互式代码示例。

---

### 处理缺失数据：删除

# 处理缺失数据：删除

数据集经常含有空缺，在 Pandas 中表示为 `NaN`（非数字）。处理这些空缺的一个直接方法是简单地移除包含它们的行或列。这通常是一个合理的初步操作，特别是当你的数据中只有很小一部分缺失，或者某个特定行或列有太多缺失值以至于缺乏信息时。

Pandas 为此提供了 `dropna()` 方法。我们来了解它的用法。

### 删除含有缺失值的行

默认情况下，如果行中*任何*值是 `NaN`，`dropna()` 会移除整行。

请看这个示例 DataFrame：

```python
import pandas as pd
import numpy as np

data = {'col1': [1, 2, np.nan, 4, 5],
        'col2': [np.nan, 7, 8, 9, 10],
        'col3': [11, 12, 13, 14, np.nan],
        'col4': ['A', 'B', 'C', 'D', 'E']}
df = pd.DataFrame(data)

print("原始 DataFrame：")
print(df)
```

Output:

```
原始 DataFrame：
   col1  col2  col3 col4
0   1.0   NaN  11.0    A
1   2.0   7.0  12.0    B
2   NaN   8.0  13.0    C
3   4.0   9.0  14.0    D
4   5.0  10.0   NaN    E
```

现在，我们使用 `dropna()` 的默认设置：

```python
df_dropped_rows = df.dropna()

print("\n删除包含任何 NaN 的行后的 DataFrame：")
print(df_dropped_rows)
```

Output:

```
删除包含任何 NaN 的行后的 DataFrame：
   col1  col2  col3 col4
1   2.0   7.0  12.0    B
3   4.0   9.0  14.0    D
```

请注意，行 0、2 和 4 被移除了，因为它们都至少包含一个 `NaN` 值。只有行 1 和 3 被保留，它们在所有列中都含有完整数据。

#### 控制行如何被删除

`dropna()` 方法有参数 (parameter)，可以让你更精细地控制：

1. **`how` 参数**：

   - `how='any'`（默认）：如果行中存在*任何* `NaN` 值，则删除该行。
   - `how='all'`：仅当该行中的*所有*值都是 `NaN` 时，才删除该行。

   我们创建一个 DataFrame，其中有一行完全是 `NaN`：

   ```python
   data_with_all_nan = {'col1': [1, np.nan, np.nan, 4],
                        'col2': [np.nan, 7, np.nan, 9],
                        'col3': [11, 12, np.nan, 14]}
   df_all_nan = pd.DataFrame(data_with_all_nan)

   print("\n可能包含全 NaN 行的原始 DataFrame：")
   print(df_all_nan)

   df_dropped_all = df_all_nan.dropna(how='all')
   print("\n删除所有 NaN 行后的 DataFrame：")
   print(df_dropped_all)
   ```

   Output:

   ```
   可能包含全 NaN 行的原始 DataFrame：
      col1  col2  col3
   0   1.0   NaN  11.0
   1   NaN   7.0  12.0
   2   NaN   NaN   NaN
   3   4.0   9.0  14.0

   删除所有 NaN 行后的 DataFrame：
      col1  col2  col3
   0   1.0   NaN  11.0
   1   NaN   7.0  12.0
   3   4.0   9.0  14.0
   ```

   在这种情况下，当使用 `how='all'` 时，只有所有值都是 `NaN` 的行 2 被删除了。
2. **`thresh` 参数**：这允许你指定一行要保留所需的*非缺失*值的最小数量。例如，`thresh=3` 意味着一行只有在至少有 3 个有效（非 `NaN`）值时才会被保留。

   使用我们原始的 `df`：

   ```python

   df_thresh3 = df.dropna(thresh=3)

   print("\n保留至少有 3 个非 NaN 值的行后的 DataFrame：")
   print(df_thresh3)
   ```

   Output:

   ```
   保留至少有 3 个非 NaN 值的行后的 DataFrame：
      col1  col2  col3 col4
   0   1.0   NaN  11.0    A  # 已保留 (3 个非 NaN)
   1   2.0   7.0  12.0    B  # 已保留 (4 个非 NaN)
   2   NaN   8.0  13.0    C  # 已保留 (3 个非 NaN)
   3   4.0   9.0  14.0    D  # 已保留 (4 个非 NaN)
   4   5.0  10.0   NaN    E  # 已保留 (3 个非 NaN)
   ```

   在这里，所有行都被保留了，因为每行都至少有 3 个非缺失值。如果我们增加阈值：

   ```python

   df_thresh4 = df.dropna(thresh=4)

   print("\n保留至少有 4 个非 NaN 值的行后的 DataFrame：")
   print(df_thresh4)
   ```

   Output:

   ```
   保留至少有 4 个非 NaN 值的行后的 DataFrame：
      col1  col2  col3 col4
   1   2.0   7.0  12.0    B
   3   4.0   9.0  14.0    D
   ```

   现在，只保留了行 1 和行 3，因为它们是唯一具有 4 个有效值的行。

### 删除含有缺失值的列

有时，如果列中含有缺失数据，你可能希望移除整个列，特别是当一列有许多 `NaN` 值或对你的分析不重要时。你可以通过将 `axis` 参数 (parameter)设置为 `1`（或 `'columns'`）来实现此操作。

```python

df_dropped_cols = df.dropna(axis=1)

print("\n删除包含任何 NaN 的列后的 DataFrame：")
print(df_dropped_cols)
```

Output:

```
删除包含任何 NaN 的列后的 DataFrame：
  col4
0    A
1    B
2    C
3    D
4    E
```

在我们的示例 `df` 中，列 `col1`、`col2` 和 `col3` 都至少包含一个 `NaN`，因此它们被删除了。只有 `col4` 被保留，因为它没有缺失值。

当应用于列时，`how` 和 `thresh` 参数的用法类似：

- `df.dropna(axis=1, how='all')` 会仅当列中*所有*值都是 `NaN` 时才删除该列。
- `df.dropna(axis=1, thresh=4)` 会仅当列中至少有 4 个非 `NaN` 值时才保留该列。

```python

df_thresh4_cols = df.dropna(axis=1, thresh=4)

print("\n保留至少有 4 个非 NaN 值的列后的 DataFrame：")
print(df_thresh4_cols)
```

Output:

```
保留至少有 4 个非 NaN 值的列后的 DataFrame：
   col1  col2  col3 col4
0   1.0   NaN  11.0    A
1   2.0   7.0  12.0    B
2   NaN   8.0  13.0    C
3   4.0   9.0  14.0    D
4   5.0  10.0   NaN    E
```

在这种情况下，`col1`、`col2` 和 `col3` 各有 4 个非 `NaN` 值（总共 5 行），`col4` 有 5 个。由于所有都达到了 4 的阈值，因此没有列被删除。

### 就地修改 DataFrame

默认情况下，`dropna()` 返回一个*新*的 DataFrame，其中缺失值已被删除，而原始 DataFrame 保持不变。如果你想直接修改原始 DataFrame，可以使用 `inplace=True` 参数 (parameter)。

```python
df_copy = df.copy()

print("\n就地删除前的 DataFrame：")
print(df_copy)

df_copy.dropna(inplace=True)

print("\n就地删除后的 DataFrame：")
print(df_copy)
```

Output:

```
就地删除前的 DataFrame：
   col1  col2  col3 col4
0   1.0   NaN  11.0    A
1   2.0   7.0  12.0    B
2   NaN   8.0  13.0    C
3   4.0   9.0  14.0    D
4   5.0  10.0   NaN    E

就地删除后的 DataFrame：
   col1  col2  col3 col4
1   2.0   7.0  12.0    B
3   4.0   9.0  14.0    D
```

请谨慎使用 `inplace=True`。由于它直接修改你的数据，除非你确定不再需要包含 `NaN` 值的原始数据，否则通常更安全的方法是将结果赋给一个新变量。

### 删除数据时的考量

删除缺失数据很简单，但它有代价：你会丢失信息。

- 删除行意味着丢失这些行中所有其他可能有用的信息。
- 删除列意味着对所有数据点完全丢失该特征。

这种策略通常最适用在以下情况：

- 缺失数据的比例非常小。
- 特定行或列绝大部分数据缺失（例如，超过 50-60% 为 NaN），并且不太可能有用。
- 你计划使用的特定分析技术完全无法处理缺失值。

---

### 处理缺失数据：填充

# 处理缺失数据：填充

使用 `dropna()` 删除含有缺失值的行或列是一种简单的方法，但这通常不是最佳方法。删除数据意味着丢失可能有的重要信息，特别是当缺失值稀疏或数据集较小时。一个常用方法是*填充*缺失值，这也被称为数据补全。

Pandas 中处理此问题的主要工具是 `fillna()` 方法。它提供了灵活选项，用于替换 Series 或 DataFrame 中的 `NaN`（非数字）值。

### `fillna()` 方法

我们从一个包含缺失值的简单 DataFrame 开始：

```python
import pandas as pd
import numpy as np

data = {'col_a': [1, np.nan, 3, 4, np.nan],
        'col_b': [np.nan, 6, 7, 8, 9],
        'col_c': ['apple', 'banana', np.nan, 'orange', 'banana']}
df = pd.DataFrame(data)

print("原始 DataFrame：")
print(df)
print("\n每列的缺失值：")
print(df.isnull().sum())
```

运行此代码将显示我们的 DataFrame 以及每列的缺失值计数：

```
原始 DataFrame：
   col_a  col_b   col_c
0    1.0    NaN   apple
1    NaN    6.0  banana
2    3.0    7.0     NaN
3    4.0    8.0  orange
4    NaN    9.0  banana

每列的缺失值：
col_a    2
col_b    1
col_c    1
dtype: int64
```

### 用一个常量值填充

最基本的策略是用一个固定值替换所有 `NaN` 值。值的选择取决于具体情况。对于数值列，0 是一个常用选择，而对于分类列，您可以使用“未知”或“缺失”之类的占位符。

要用 0 填充整个 DataFrame 中的所有 `NaN`：

```python
df_filled_zero = df.fillna(0)
print("\n用 0 填充 NaN 后的 DataFrame：")
print(df_filled_zero)
```

输出：

```
用 0 填充 NaN 后的 DataFrame：
   col_a  col_b     col_c
0    1.0    0.0     apple
1    0.0    6.0    banana
2    3.0    7.0         0
3    4.0    8.0    orange
4    0.0    9.0    banana
```

请注意，`fillna(0)` 将 `col_c`（一个字符串列）中的 `NaN` 替换为整数 0。这可能不太理想。通常，您会希望对不同列应用不同的填充策略。稍后我们将看到如何做到这一点。

### 用统计度量填充（均值、中位数、众数）

除了常量之外，您还可以使用从该列的可用数据中计算出的统计量（例如均值或中位数）来填充缺失的数值。对于分类数据，通常使用众数（最常出现的值）。

- **均值：** 最适合大致呈正态分布（没有极端异常值）的数值数据。
- **中位数：** 对异常值更具抵抗力，适用于偏斜的数值数据。
- **众数：** 适用于分类数据或离散数值数据。

我们用 `col_a` 的均值填充它，用 `col_b` 的中位数填充它。首先，我们计算这些值（请记住在计算中跳过 `NaN`，Pandas 默认会这样做）：

```python
mean_a = df['col_a'].mean()
median_b = df['col_b'].median()

print(f"\ncol_a 的均值：{mean_a}")
print(f"col_b 的中位数：{median_b}")

df['col_a_filled_mean'] = df['col_a'].fillna(mean_a)

df['col_b_filled_median'] = df['col_b'].fillna(median_b)

print("\n已填充均值/中位数的 DataFrame 列：")
print(df[['col_a', 'col_a_filled_mean', 'col_b', 'col_b_filled_median']])
```

输出：

```
col_a 的均值：2.6666666666666665
col_b 的中位数：7.5

已填充均值/中位数的 DataFrame 列：
   col_a  col_a_filled_mean  col_b  col_b_filled_median
0    1.0           1.000000    NaN                  7.5
1    NaN           2.666667    6.0                  6.0
2    3.0           3.000000    7.0                  7.0
3    4.0           4.000000    8.0                  8.0
4    NaN           2.666667    9.0                  9.0
```

要用 `col_c`（分类）的众数填充它：

```python
mode_c = df['col_c'].mode()[0]
print(f"\ncol_c 的众数：{mode_c}")

df['col_c_filled_mode'] = df['col_c'].fillna(mode_c)

print("\n已填充众数的 DataFrame 列：")
print(df[['col_c', 'col_c_filled_mode']])
```

输出：

```
col_c 的众数：banana

已填充众数的 DataFrame 列：
    col_c col_c_filled_mode
0   apple             apple
1  banana            banana
2     NaN            banana
3  orange            orange
4  banana            banana
```

### 前向填充 (ffill) 和 后向填充 (bfill)

有时，特别是在处理时间序列数据或有序数据时，根据紧邻其前或后的值来填充缺失值是合理的。

- **前向填充 (`ffill`)**：将最后一个有效观测值向前传播以填充空白。
- **后向填充 (`bfill`)**：使用下一个有效观测值来填充空白。

您可以使用 `fillna()` 中的 `method` 参数 (parameter)来指定这些方法：

```python

data_seq = {'value': [10, np.nan, np.nan, 13, np.nan, 15, np.nan, np.nan, np.nan, 20]}
df_seq = pd.DataFrame(data_seq)
print("\n序列 DataFrame：")
print(df_seq)

df_seq['ffill'] = df_seq['value'].fillna(method='ffill')

df_seq['bfill'] = df_seq['value'].fillna(method='bfill')

print("\n前向和后向填充后的 DataFrame：")
print(df_seq)
```

输出：

```
序列 DataFrame：
   value
0   10.0
1    NaN
2    NaN
3   13.0
4    NaN
5   15.0
6    NaN
7    NaN
8    NaN
9   20.0

前向和后向填充后的 DataFrame：
   value  ffill  bfill
0   10.0   10.0   10.0
1    NaN   10.0   13.0
2    NaN   10.0   13.0
3   13.0   13.0   13.0
4    NaN   13.0   15.0
5   15.0   15.0   15.0
6    NaN   15.0   20.0
7    NaN   15.0   20.0
8    NaN   15.0   20.0
9   20.0   20.0   20.0
```

观察 `ffill` 如何将最后见到的值（`10`、`13`、`15`）向前传播，而 `bfill` 则用下一个可用值（`13`、`15`、`20`）填充空白。

### 用不同值填充不同列

您通常需要对不同列应用不同的填充策略。您可以通过向 `fillna()` 传递一个字典来实现此目的，其中键是列名，值是相应的填充值或策略（尽管 `ffill`/`bfill` 等方法以此方式使用时会应用于整个 DataFrame，但更常见的是逐列应用它们，或将字典值用于常量/统计量）。

对于不同的策略，更典型的方法是逐列填充或使用字典填充常量值：

```python

df = pd.DataFrame(data)

fill_values = {'col_a': df['col_a'].mean(),
               'col_b': 0,
               'col_c': 'Unknown'}

df_filled_dict = df.fillna(value=fill_values)

print("\n使用字典填充后的 DataFrame：")
print(df_filled_dict)
```

输出：

```
使用字典填充后的 DataFrame：
      col_a  col_b    col_c
0  1.000000    0.0    apple
1  2.666667    6.0   banana
2  3.000000    7.0  Unknown
3  4.000000    8.0   orange
4  2.666667    9.0   banana
```

### 原地修改 DataFrame

与许多 Pandas 方法一样，`fillna()` 默认返回一个*新*的 DataFrame，其中包含更改，而原始 DataFrame 保持不变。如果您想直接修改原始 DataFrame，可以使用 `inplace=True` 参数 (parameter)：

```python

df = pd.DataFrame(data)

print("\n原始 DataFrame（原地填充前）：")
print(df)

df.fillna(0, inplace=True)

print("\n原始 DataFrame（原地填充后）：")
print(df)
```

输出：

```
原始 DataFrame（原地填充前）：
   col_a  col_b   col_c
0    1.0    NaN   apple
1    NaN    6.0  banana
2    3.0    7.0     NaN
3    4.0    8.0  orange
4    NaN    9.0  banana

原始 DataFrame（原地填充后）：
   col_a  col_b     col_c
0    1.0    0.0     apple
1    0.0    6.0    banana
2    3.0    7.0         0
3    4.0    8.0    orange
4    0.0    9.0    banana
```

虽然 `inplace=True` 看起来方便，但通常建议（尤其是在学习时）将结果赋值给一个新变量，或者重新赋值回原始变量名（`df = df.fillna(...)`）。这使得数据转换步骤更明确，并可以防止复杂代码中出现意外的副作用。

选择正确的填充策略需要理解您的数据和分析目标。缺失是随机的吗？均值、中位数或众数会引入偏差吗？顺序重要吗（暗示使用 ffill/bfill）？仔细考虑这些问题会带来更有效的数据准备。

获取即时帮助、个性化解释和交互式代码示例。

---

### 删除列和行

# 删除列和行

移除不必要或有问题的数据是数据准备中一个常见步骤。您可能需要从DataFrame中删除整行（观测值）或整列（特征）。例如：

- 一列可能包含另一列中已有的冗余信息。
- 一列可能包含过多的缺失值，无法可靠地填充。
- 特定行可能代表您希望从分析中排除的异常值或错误。
- 行或列可能与您尝试回答的具体问题不相关。

Pandas提供了多功能的`drop()`方法来处理行和列的删除。

### 使用索引标签删除行

要删除一行或多行，您需要将要删除的行的索引标签提供给`drop()`方法。通常，您还会使用`index`参数 (parameter)指定要操作的是行。

我们从一个示例DataFrame开始：

```python
import pandas as pd
import numpy as np

data = {'StudentID': ['S101', 'S102', 'S103', 'S104', 'S105', 'S106'],
        'Math': [85, 92, 78, 88, 95, np.nan],
        'Science': [90, 88, 94, 85, 79, np.nan],
        'History': [76, 81, 85, 70, 88, 60],
        'Notes': ['Good', 'Excellent', np.nan, 'Needs Improvement', 'Good', 'Incomplete']}
df = pd.DataFrame(data)
df.set_index('StudentID', inplace=True)

print("原始 DataFrame：")
print(df)
```

这会生成：

```
Original DataFrame:
           Math  Science  History              Notes
StudentID
S101       85.0     90.0       76               Good
S102       92.0     88.0       81          Excellent
S103       78.0     94.0       85                NaN
S104       88.0     85.0       70  Needs Improvement
S105       95.0     79.0       88               Good
S106        NaN      NaN       60         Incomplete
```

假设我们想删除学生`S106`，因为他们的数据不完整。我们可以使用`drop()`方法来完成此操作：

```python

df_dropped_row = df.drop(index='S106')

print("\n删除行 'S106' 后的 DataFrame：")
print(df_dropped_row)
```

Output:

```
DataFrame after dropping row 'S106':
           Math  Science  History              Notes
StudentID
S101       85.0     90.0       76               Good
S102       92.0     88.0       81          Excellent
S103       78.0     94.0       85                NaN
S104       88.0     85.0       70  Needs Improvement
S105       95.0     79.0       88               Good
```

请注意，`df.drop()`默认返回一个*新*的DataFrame，其中指定行已删除。原始DataFrame `df`保持不变。

要删除多行，请传入一个索引标签列表：

```python

df_dropped_rows = df.drop(index=['S103', 'S106'])

print("\n删除行 'S103' 和 'S106' 后的 DataFrame：")
print(df_dropped_rows)
```

Output:

```
DataFrame after dropping rows 'S103' and 'S106':
           Math  Science  History              Notes
StudentID
S101       85.0     90.0       76               Good
S102       92.0     88.0       81          Excellent
S104       88.0     85.0       70  Needs Improvement
S105       95.0     79.0       88               Good
```

### 删除列

删除列非常相似，但您提供的是列名而不是索引标签。指定要操作列的优选方式是使用`columns`参数 (parameter)。

假设我们的分析不需要`Notes`列。我们可以将其删除：

```python

df_dropped_col = df.drop(columns='Notes')

print("\n删除 'Notes' 列后的 DataFrame：")
print(df_dropped_col)
```

Output:

```
DataFrame after dropping the 'Notes' column:
           Math  Science  History
StudentID
S101       85.0     90.0       76
S102       92.0     88.0       81
S103       78.0     94.0       85
S104       88.0     85.0       70
S105       95.0     79.0       88
S106        NaN      NaN       60
```

同样，此操作返回一个新的DataFrame。原始的`df`仍然包含`Notes`列。

要删除多列，请传入一个列名列表：

```python

df_dropped_cols = df.drop(columns=['Math', 'Notes'])

print("\n删除 'Math' 和 'Notes' 列后的 DataFrame：")
print(df_dropped_cols)
```

Output:

```
DataFrame after dropping 'Math' and 'Notes' columns:
           Science  History
StudentID
S101          90.0       76
S102          88.0       81
S103          94.0       85
S104          85.0       70
S105          79.0       88
S106           NaN       60
```

过去，您可能会看到使用`axis=1`来表示删除列的代码（例如，`df.drop('Notes', axis=1)`）。虽然这种方式可行，但使用`columns`参数（例如，`df.drop(columns='Notes')`）通常被认为更具可读性和明确性。同样，`axis=0`对应于删除行（默认），但使用`index`参数更清晰。

### 原地修改 DataFrame

如果您确定要直接修改原始DataFrame而不创建新DataFrame，可以使用`inplace=True`参数 (parameter)。

```python

df_copy = df.copy()
print("\n原始 df_copy (前 3 行)：")
print(df_copy.head(3))

return_value = df_copy.drop(columns='History', inplace=True)

print("\n原地删除 'History' 后 df_copy (前 3 行)：")
print(df_copy.head(3))
print(f"\n当 inplace=True 时返回值为：{return_value}")
```

Output:

```
Original df_copy (first 3 rows):
           Math  Science  History      Notes
StudentID
S101       85.0     90.0       76       Good
S102       92.0     88.0       81  Excellent
S103       78.0     94.0       85        NaN

df_copy after dropping 'History' inplace (first 3 rows):
           Math  Science      Notes
StudentID
S101       85.0     90.0       Good
S102       92.0     88.0  Excellent
S103       78.0     94.0        NaN

Return value when inplace=True: None
```

请注意，当使用`inplace=True`时，`drop`方法会直接修改DataFrame并返回`None`。使用`inplace=True`时请务必小心，因为它会永久改变您的数据。通常情况下，尤其是在学习或操作数据时，使用默认行为（返回修改后的副本）会更安全，这让您能够保留原始数据。

使用`drop()`删除行和列是优化数据集的基本步骤，使您能够专注于与您的分析最相关的数据。

获取即时帮助、个性化解释和交互式代码示例。

---

### 添加新列

# 添加新列

通常，你加载的原始数据不包含分析所需的所有信息。你可能需要根据现有数据计算新值或添加分类标签。Pandas 让向 DataFrame 添加新列变得简单。

添加新列最直接的方法是向尚不存在的列名赋值数据。可以把它想象成向字典添加新条目，但其中的“值”通常是 Series、数组，或一个会被广播到所有行的单个值。

### 赋值常量值

让我们从一个简单的 DataFrame 开始：

```python
import pandas as pd
import numpy as np

data = {'col_A': [10, 20, 30, 40],
        'col_B': [5, 15, 25, 35]}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
```

```text
原始 DataFrame:
   col_A  col_B
0     10      5
1     20     15
2     30     25
3     40     35
```

如果你想添加一个新列，其中每一行都具有相同的值（一个标量值），你可以直接赋值它：

```python

df['source'] = 'dataset_1'

df['version'] = 1.0

print("\nDataFrame after adding constant columns:")
print(df)
```

```text
添加常量列后的 DataFrame:
   col_A  col_B     source  version
0     10      5  dataset_1      1.0
1     20     15  dataset_1      1.0
2     30     25  dataset_1      1.0
3     40     35  dataset_1      1.0
```

Pandas 自动将单个值 'dataset\_1' 和数字 1.0 广播，分别填充新列 'source' 和 'version' 中的所有行。

### 根据现有数据创建列

一种非常常见的操作是根据涉及一个或多个现有列的计算来创建新列。由于 Pandas 操作通常是矢量化 (quantization)的（像 NumPy 一样），你可以高效地执行这些计算。

我们来添加一个列 'col\_C'，它是 'col\_A' 和 'col\_B' 的和：

```python

df['col_C'] = df['col_A'] + df['col_B']

print("\nDataFrame after adding calculated column 'col_C':")
print(df)
```

```text
添加计算列 'col_C' 后的 DataFrame:
   col_A  col_B     source  version  col_C
0     10      5  dataset_1      1.0     15
1     20     15  dataset_1      1.0     35
2     30     25  dataset_1      1.0     55
3     40     35  dataset_1      1.0     75
```

加法操作是针对每一行进行元素级的。你可以使用任何标准算术运算符（+、-、\*、/、%）或 NumPy 等库中更复杂的函数。

例如，我们添加另一个列 'col\_D'，它是 'col\_A' 除以 10 的结果：

```python

df['col_D'] = df['col_A'] / 10

print("\nDataFrame after adding calculated column 'col_D':")
print(df)
```

```text
添加计算列 'col_D' 后的 DataFrame:
   col_A  col_B     source  version  col_C  col_D
0     10      5  dataset_1      1.0     15    1.0
1     20     15  dataset_1      1.0     35    2.0
2     30     25  dataset_1      1.0     55    3.0
3     40     35  dataset_1      1.0     75    4.0
```

### 赋值列表或 NumPy 数组

你也可以通过赋值一个 Python 列表或 NumPy 数组来添加新列。主要要求是列表或数组的长度必须与 DataFrame 中的行数（其索引长度）匹配。

```python

new_values_list = [100, 200, 300, 400]
df['col_E'] = new_values_list

new_values_np = np.array([0.1, 0.2, 0.3, 0.4])
df['col_F'] = new_values_np

print("\nDataFrame after adding columns from list and NumPy array:")
print(df)
```

```text
从列表和 NumPy 数组添加列后的 DataFrame:
   col_A  col_B     source  version  col_C  col_D  col_E  col_F
0     10      5  dataset_1      1.0     15    1.0    100    0.1
1     20     15  dataset_1      1.0     35    2.0    200    0.2
2     30     25  dataset_1      1.0     55    3.0    300    0.3
3     40     35  dataset_1      1.0     75    4.0    400    0.4
```

如果长度不匹配，Pandas 将引发 `ValueError`。

### 赋值 Pandas Series

你也可以赋值一个现有的 Pandas Series 来创建新列。这样做时，Pandas 会根据 Series 和 DataFrame 的 **索引** 来对齐 (alignment)数据。如果索引匹配，值会相应地放置。如果索引不完全对齐，DataFrame 中没有在 Series 中找到匹配索引的行，在新列中将获得一个缺失值 (NaNNaNNaN)。

我们来创建一个索引稍有不同的 Series：

```python
s = pd.Series([500, 600, 700], index=[1, 2, 4])
print("\nSeries 's' to be added:")
print(s)

df['col_G'] = s

print("\nDataFrame after adding column 'col_G' from Series 's':")
print(df)
```

```text
要添加的 Series 's':
1    500
2    600
4    700
dtype: int64

从 Series 's' 添加列 'col_G' 后的 DataFrame:
   col_A  col_B     source  version  col_C  col_D  col_E  col_F  col_G
0     10      5  dataset_1      1.0     15    1.0    100    0.1    NaN
1     20     15  dataset_1      1.0     35    2.0    200    0.2  500.0
2     30     25  dataset_1      1.0     55    3.0    300    0.3  600.0
3     40     35  dataset_1      1.0     75    4.0    400    0.4    NaN
```

请注意，`col_G` 仅在索引 1 和 2 处有值，与 Series `s` 匹配。DataFrame 中的索引 0 和 3 在 `s` 中不存在，因此它们获得了 `NaN`。`s` 中索引 4 的值被忽略了，因为 DataFrame 没有索引 4。这种索引对齐行为是 Pandas 的一个基本特点，可以避免数据错位导致的错误。

能够添加新列，特别是派生列，是数据准备中的重要一步。它让你能够构建新特征、规范化数据，或只是更有效地组织信息以进行分析或建模。

获取即时帮助、个性化解释和交互式代码示例。

---

### 修改现有列

# 修改现有列

修改现有列中的信息是数据准备中一项常见工作。你可能需要根据当前值进行计算、标准化文本格式、更改数据类型，或者执行自定义转换。Pandas为此提供了多种灵活的方法。

### 直接赋值与向量 (vector)化操作

修改列的最简单方式通常是通过直接的算术或逻辑运算。由于 Pandas 构建于 NumPy 之上，这些运算通常是向量化 (quantization)的，这意味着它们会逐元素高效地执行，无需显式循环。

例如，假设你有一个 DataFrame，其中包含摄氏度温度，你想将它们转换为华氏度。

```python
import pandas as pd
import numpy as np

data = {'City': ['London', 'Paris', 'Tokyo'],
        'Temp_C': [12, 15, 8]}
df = pd.DataFrame(data)

print("原始 DataFrame:")
print(df)

df['Temp_F'] = df['Temp_C'] * 9/5 + 32

print("\n已添加华氏度列的 DataFrame:")
print(df)
```

这种向量化方式适用于加法（`+`）、减法（`-`）、乘法（`*`）、除法（`/`）、幂运算（`**`）以及其他标准数学运算。

### 使用 `apply()` 方法执行自定义函数

有时，你需要的转换比简单的算术更复杂。你可能有一个自定义函数，想将其执行到列（Pandas Series）中的每个元素。`apply()` 方法在此处很有用。

假设我们想将温度归类为'Cold'（冷）、'Mild'（温和）或'Warm'（暖）。

```python

def categorize_temp(temp):
  if temp < 10:
    return 'Cold'
  elif 10 <= temp < 20:
    return 'Mild'
  else:
    return 'Warm'

df['Temp_Category'] = df['Temp_C'].apply(categorize_temp)

print("\n带有温度类别的 DataFrame:")
print(df)
```

你也可以将 `apply()` 与 lambda 函数结合使用，进行更简洁的一次性任务：

```python

df['Temp_C_Squared'] = df['Temp_C'].apply(lambda x: x**2)

print("\n带有温度平方的 DataFrame:")
print(df)
```

`apply()` 方法很灵活，但如果存在针对你任务的向量 (vector)化操作或更专门的函数，有时它的速度会比这些方法慢。

### 使用 `replace()` 替换值

如果你需要替换列中的特定值，`replace()` 方法会非常实用。你可以替换单个值，或提供一个字典来同时替换多个值。

```python

data_cat = {'ID': [101, 102, 103, 104],
            'Grade': ['A', 'B', 'C', 'B'],
            'Status': ['Pass', 'Pass', 'Fail', 'Pass']}
df_cat = pd.DataFrame(data_cat)

print("原始分类 DataFrame:")
print(df_cat)

df_cat['Status'] = df_cat['Status'].replace('Fail', 'Did Not Pass')

print("\n替换 'Fail' 后的 DataFrame:")
print(df_cat)

grade_map = {'A': 'Excellent', 'B': 'Good', 'C': 'Fair'}
df_cat['Grade_Desc'] = df_cat['Grade'].replace(grade_map)

print("\n替换多个等级后的 DataFrame:")
print(df_cat)
```

### 使用 `map()` 进行值替换

与 `replace()` 相似，Series 上的 `map()` 方法可以用来根据字典（或另一个 Series，或一个函数）替换每个值。当你希望根据预设的对应关系将*所有*现有值映射到新值时，它会特别有用。在映射字典中找不到的值将变为 NaNNaNNaN。

```python

s = pd.Series(['cat', 'dog', 'rabbit', 'cat'])
print("原始 Series:")
print(s)

animal_sounds = {'cat': 'meow', 'dog': 'bark'}
sounds = s.map(animal_sounds)

print("\n映射后的 Series（未知值变为 NaN）：")
print(sounds)
```

`replace()` 针对要更改的个别值，而 `map()` 常用于基于查找的全面转换。

### 使用 `.str` 进行字符串操作

Pandas 通过包含字符串（对象 dtype）的 Series 上的 `.str` 访问器，提供了一套强大的字符串处理方法。这让你能轻松地对整个列执行向量 (vector)化的常见字符串操作。

```python

data_text = {'Product': ['Apple iPhone 14', 'SAMSUNG GALAXY S23', 'google pixel 7'],
             'Code': ['APL-14', 'SAM-S23', 'GGL-PX7']}
df_text = pd.DataFrame(data_text)

print("原始文本 DataFrame:")
print(df_text)

df_text['Product_Lower'] = df_text['Product'].str.lower()

df_text[['Brand_Code', 'Model_Code']] = df_text['Code'].str.split('-', expand=True)

df_text['Has_Galaxy'] = df_text['Product'].str.contains('GALAXY', case=False)

print("\n字符串操作后的 DataFrame:")
print(df_text)
```

其他常见的 `.str` 方法包括 `startswith()`、`endswith()`、`replace()`、`len()`、`strip()`、`get()` 等等。它们在清理和标准化文本数据方面非常实用。

### 使用 `astype()` 更改列数据类型

有时，数据加载时的数据类型不正确。例如，数字可能被读取为字符串（对象类型），或者在处理缺失值后你可能想将浮点数转换为整数。`astype()` 方法可以更改列的数据类型。

```python

data_types = {'ID': ['101', '102', '103'],
              'Value': [20.5, 15.0, 33.8],
              'Category': ['X', 'Y', 'X']}
df_types = pd.DataFrame(data_types)

print("带有数据类型的原始 DataFrame:")
print(df_types)
print(df_types.dtypes)

df_types['ID'] = df_types['ID'].astype(int)

df_types['Value_Int'] = df_types['Value'].astype(int)

df_types['Category'] = df_types['Category'].astype('category')

print("\n更改数据类型后的 DataFrame:")
print(df_types)
print(df_types.dtypes)
```

更改类型时请务必小心。如果转换不可行（例如，尝试将 'hello' 这样的字符串转换为整数），Pandas 将引发错误。在使用 `astype()` 之前，请确认你的数据适合目标类型。对于涉及潜在错误或缺失值的转换，你可能需要先整理数据，或使用像 `pd.to_numeric(errors='coerce')` 这样的函数，它会将无法转换的值变为 NaNNaNNaN。

修改现有列是数据整理的核心部分。通过结合直接操作、`apply()`、`replace()`、字符串方法和类型转换，你可以很好地调整数据，使其符合分析所需的格式。

获取即时帮助、个性化解释和交互式代码示例。

---

### 重命名列

# 重命名列

在清洗和准备数据时，您会发现原始列名通常不理想。它们可能不够清晰，过长，包含空格或特殊字符导致在代码中使用不便，或者 simply 不符合一致的命名约定。重命名列是让您的DataFrame更易于理解和使用的一个常见且主要步骤。

Pandas 提供了灵活的 `.rename()` 方法，专门用于此目的。它允许您更改列名（和索引标签），而不会改变数据本身。

### 使用 `.rename()` 方法

使用 `.rename()` 最常用的方法是向其 `columns` 参数 (parameter)传递一个字典。这个字典应该将旧列名（键）映射到新列名（值）。

我们从一个示例 DataFrame 开始：

```python
import pandas as pd
import numpy as np

data = {'Student ID': [101, 102, 103, 104],
        'Test Score (Math)': [85, 92, np.nan, 78],
        'Test Score (English)': [76, 88, 95, 80],
        'attendance %': [90, 95, 85, 92]}
df = pd.DataFrame(data)

print("原始 DataFrame:")
print(df)
```

这将输出：

```
Original DataFrame:
   Student ID  Test Score (Math)  Test Score (English)  attendance %
0         101               85.0                    76            90
1         102               92.0                    88            95
2         103                NaN                    95            85
3         104               78.0                    80            92
```

请注意，列名包含空格、括号和 '%' 等符号。我们把它们重命名为更适合编程的形式，使用小写字母和下划线。

```python

rename_map = {
    'Student ID': 'student_id',
    'Test Score (Math)': 'math_score',
    'Test Score (English)': 'english_score',
    'attendance %': 'attendance_pct'
}

df_renamed = df.rename(columns=rename_map)

print("\n重命名后的 DataFrame:")
print(df_renamed)
```

输出显示了新的列名：

```
DataFrame after renaming:
   student_id  math_score  english_score  attendance_pct
0         101        85.0             76              90
1         102        92.0             88              95
2         103         NaN             95              85
3         104        78.0             80              92
```

您不必一次性重命名所有列。如果您只提供部分列的映射，那么只有那些列会被重命名。

### 原地修改 DataFrame

默认情况下，`.rename()` 会返回一个带有更新名称的 *新* DataFrame，同时保持原始 DataFrame 不变。这通常更安全，因为它防止了对原始数据的意外修改。

但是，如果您确定要直接修改 DataFrame，可以使用 `inplace=True` 参数 (parameter)：

```python

df_copy = df.copy()

print("\n原始 DataFrame (副本):")
print(df_copy)

df_copy.rename(columns=rename_map, inplace=True)

print("\n原地重命名后的 DataFrame:")
print(df_copy)
```

Output:

```
Original DataFrame (copy):
   Student ID  Test Score (Math)  Test Score (English)  attendance %
0         101               85.0                    76            90
1         102               92.0                    88            95
2         103                NaN                    95            85
3         104               78.0                    80            92

DataFrame after inplace renaming:
   student_id  math_score  english_score  attendance_pct
0         101        85.0             76              90
1         102        92.0             88              95
2         103         NaN             95              85
3         104        78.0             80              92
```

使用 `inplace=True` 有时能让代码稍微简洁，但请谨慎使用。当对象被直接修改时，追踪更改通常会变得更难，尤其是在较长的分析脚本或 Jupyter notebook 中。

### 重命名索引标签

`.rename()` 方法也可以用于重命名索引标签，使用 `index` 参数 (parameter)。它与 `columns` 参数的工作方式类似，接受一个将旧索引标签映射到新索引标签的字典。

```python

df_indexed = df_renamed.set_index('student_id')
print("\n以 student_id 为索引的 DataFrame:")
print(df_indexed)

index_rename_map = {101: 'S101', 104: 'S104'}
df_index_renamed = df_indexed.rename(index=index_rename_map)

print("\n重命名索引标签后的 DataFrame:")
print(df_index_renamed)
```

Output:

```
DataFrame with student_id as index:
            math_score  english_score  attendance_pct
student_id
101               85.0             76              90
102               92.0             88              95
103                NaN             95              85
104               78.0             80              92

DataFrame after renaming index labels:
            math_score  english_score  attendance_pct
student_id
S101              85.0             76              90
102               92.0             88              95
103                NaN             95              85
S104              78.0             80              92
```

### 替代方法：赋值给 `df.columns`

如果您需要重命名 *所有* 列，并且知道它们新名称的正确顺序，您可以直接将新名称列表赋值给 DataFrame 的 `.columns` 属性。

```python

new_column_names = ['id', 'score_math', 'score_english', 'attendance']

df_copy2 = df.copy()

df_copy2.columns = new_column_names

print("\n赋值给 df.columns 后的 DataFrame:")
print(df_copy2)
```

Output:

```
DataFrame after assigning to df.columns:
    id  score_math  score_english  attendance
0  101        85.0             76          90
1  102        92.0             88          95
2  103         NaN             95          85
3  104        78.0             80          92
```

这种方法更直接，但比 `.rename()` 灵活性更差。您必须为 *所有* 列提供名称，并且列表长度必须与 DataFrame 中的列数完全匹配，否则会报错。它通常更适合于创建 DataFrame 或全面修改列名的情况。对于有针对性的重命名，`.rename()` 通常是更推荐的方法。

重命名列是一种简单而有效的方法，可以提高 DataFrame 的清晰度和可用性，使后续分析步骤更顺畅，并使您的代码更易于阅读。

获取即时帮助、个性化解释和交互式代码示例。

---

### 数据排序

# 数据排序

为了进行有效的数据分析，按有意义的顺序排列数据通常是必要的一步。排序允许您按从低到高、字母顺序或根据自定义标准查看数据，使模式更容易发现，特定条目更容易找到。Pandas 提供了灵活的方法，可以根据索引标签或列中的实际值对数据进行排序。

### 按索引排序

有时，您需要根据行或列标签（即索引）排列数据。如果索引表示时间序列、有序类别，或者只是需要按字母或数字顺序排列，这会特别有用。`sort_index()` 方法可处理这种情况。

我们从一个简单的 DataFrame 开始：

```python
import pandas as pd
import numpy as np

data = {'col_b': [4, 7, 1, 8, 5],
        'col_a': ['apple', 'banana', 'orange', 'apple', 'banana'],
        'col_c': [10.0, np.nan, 20.0, 10.0, 15.0]}
df = pd.DataFrame(data, index=['R3', 'R1', 'R5', 'R2', 'R4'])

print("原始 DataFrame:")
print(df)
```

```text
原始 DataFrame:
    col_b   col_a  col_c
R3      4   apple   10.0
R1      7  banana    NaN
R5      1  orange   20.0
R2      8   apple   10.0
R4      5  banana   15.0
```

请注意，行索引（`R3`、`R1`、`R5` 等）未按字母顺序排列。要根据索引标签对 DataFrame 行进行排序：

```python
df_sorted_by_index = df.sort_index()

print("\n按行索引排序的 DataFrame（升序）：")
print(df_sorted_by_index)
```

```text
按行索引排序的 DataFrame（升序）：
    col_b   col_a  col_c
R1      7  banana    NaN
R2      8   apple   10.0
R3      4   apple   10.0
R4      5  banana   15.0
R5      1  orange   20.0
```

默认情况下，`sort_index()` 按升序排序。要按降序排序，请使用 `ascending=False` 参数 (parameter)：

```python
df_sorted_by_index_desc = df.sort_index(ascending=False)

print("\n按行索引排序的 DataFrame（降序）：")
print(df_sorted_by_index_desc)
```

```text
按行索引排序的 DataFrame（降序）：
    col_b   col_a  col_c
R5      1  orange   20.0
R4      5  banana   15.0
R3      4   apple   10.0
R2      8   apple   10.0
R1      7  banana    NaN
```

您也可以通过列索引（列名）进行排序，通过指定 `axis=1`：

```python
df_sorted_by_columns = df.sort_index(axis=1)

print("\n按列索引排序的 DataFrame（升序）：")
print(df_sorted_by_columns)
```

```text
按列索引排序的 DataFrame（升序）：
     col_a  col_b  col_c
R3   apple      4   10.0
R1  banana      7    NaN
R5  orange      1   20.0
R2   apple      8   10.0
R4  banana      5   15.0
```

与许多 Pandas 操作一样，`sort_index()` 默认情况下会返回一个新的已排序 DataFrame。如果您想直接修改原始 DataFrame，请使用 `inplace=True` 参数。使用 `inplace=True` 时请务必小心，因为它会覆盖您的原始数据结构。

```python
df_copy = df.copy()
df_copy.sort_index(inplace=True)

print("\n按索引就地排序后的原始 DataFrame：")
print(df_copy)
```

```text
按索引就地排序后的原始 DataFrame：
    col_b   col_a  col_c
R1      7  banana    NaN
R2      8   apple   10.0
R3      4   apple   10.0
R4      5  banana   15.0
R5      1  orange   20.0
```

### 按值排序

更常见的情况是，您会希望根据一个或多个列中的值对 DataFrame 进行排序。`sort_values()` 方法用于此目的。`sort_values()` 最重要的参数 (parameter)是 `by`，它指定了用于排序的列名（或列名列表）。

我们根据 `col_b` 中的值对原始 DataFrame `df` 进行排序：

```python
df_sorted_by_col_b = df.sort_values(by='col_b')

print("\n按 'col_b' 排序的 DataFrame（升序）：")
print(df_sorted_by_col_b)
```

```text
按 'col_b' 排序的 DataFrame（升序）：
    col_b   col_a  col_c
R5      1  orange   20.0
R3      4   apple   10.0
R4      5  banana   15.0
R1      7  banana    NaN
R2      8   apple   10.0
```

同样，默认排序顺序是升序。使用 `ascending=False` 可进行降序排序：

```python
df_sorted_by_col_b_desc = df.sort_values(by='col_b', ascending=False)

print("\n按 'col_b' 排序的 DataFrame（降序）：")
print(df_sorted_by_col_b_desc)
```

```text
按 'col_b' 排序的 DataFrame（降序）：
    col_b   col_a  col_c
R2      8   apple   10.0
R1      7  banana    NaN
R4      5  banana   15.0
R3      4   apple   10.0
R5      1  orange   20.0
```

您可以通过向 `by` 参数传递一个列名列表来通过多列进行排序。Pandas 将首先按列表中的第一列排序，然后使用第二列来处理并列情况，以此类推。

我们先按 `col_a`（按字母顺序）排序，然后对于 `col_a` 值相同的行，再按 `col_b`（按数字顺序）排序：

```python
df_sorted_by_multi = df.sort_values(by=['col_a', 'col_b'])

print("\n按 'col_a' 然后按 'col_b' 排序的 DataFrame（升序）：")
print(df_sorted_by_multi)
```

```text
按 'col_a' 然后按 'col_b' 排序的 DataFrame（升序）：
    col_b   col_a  col_c
R3      4   apple   10.0
R2      8   apple   10.0
R4      5  banana   15.0
R1      7  banana    NaN
R5      1  orange   20.0
```

请注意，带有“apple”的行在一起，并按 `col_b` 排序（4 然后 8），带有“banana”的行在一起，并按 `col_b` 排序（5 然后 7）。

当通过多列排序时，您还可以为每列指定不同的排序顺序。将布尔值列表传递给 `ascending` 参数，与传递给 `by` 的列表相对应。

我们按 `col_a` 升序和 `col_b` 降序排序：

```python
df_sorted_by_multi_mixed = df.sort_values(by=['col_a', 'col_b'], ascending=[True, False])

print("\n按 'col_a'（升序）然后按 'col_b'（降序）排序的 DataFrame：")
print(df_sorted_by_multi_mixed)
```

```text
按 'col_a'（升序）然后按 'col_b'（降序）排序的 DataFrame：
    col_b   col_a  col_c
R2      8   apple   10.0
R3      4   apple   10.0
R1      7  banana    NaN
R4      5  banana   15.0
R5      1  orange   20.0
```

现在，对于“apple”，`col_b`=8 的行排在 `col_b`=4 的行之前。对于“banana”，`col_b`=7 的行排在 `col_b`=5 的行之前。

#### 排序时处理缺失值

排序时缺失值 (`NaN`) 会怎样？默认情况下，`sort_values()` 会将 `NaN` 值放置在排序输出的*末尾*，无论排序是升序还是降序。您可以使用 `na_position` 参数控制此行为，该参数接受 `'first'` 或 `'last'`。

我们对 `col_c`（包含 `NaN`）进行排序，并明确将 `NaN` 放在首位：

```python
df_sorted_nan_first = df.sort_values(by='col_c', na_position='first')

print("\n按 'col_c' 排序的 DataFrame，NaN 优先：")
print(df_sorted_nan_first)
```

```text
按 'col_c' 排序的 DataFrame，NaN 优先：
    col_b   col_a  col_c
R1      7  banana    NaN
R3      4   apple   10.0
R2      8   apple   10.0
R4      5  banana   15.0
R5      1  orange   20.0
```

与 `sort_index()` 一样，`sort_values()` 也接受 `inplace=True` 参数以直接修改 DataFrame。

排序是一种基本操作，用于组织和理解数据。无论是按索引标签排列行，还是根据列内容排序，`sort_index()` 和 `sort_values()` 方法都提供了必要的工具，为您的 DataFrames 带来结构。

获取即时帮助、个性化解释和交互式代码示例。

---

### 实践练习：清理和修改数据框

# 实践练习：清理和修改数据框

让我们将本章讨论的技术付诸实践。我们将依照一个常见的流程来操作：找出原始数据集中的问题，清理这些问题，并调整其结构，使其更适合分析。

### 建立示例数据框

我们将建立一个小数据框来表示产品信息，故意包含一些常见问题，例如缺失值和不一致的命名。请确保已导入 Pandas（通常使用 `import pandas as pd`）和 NumPy（通常使用 `import numpy as np`）。

```python
import pandas as pd
import numpy as np

data = {'ProductID': ['P101', 'P102', 'P103', 'P104', 'P105', 'P106'],
        'Category': ['A', 'B', 'A', 'C', 'B', np.nan],
        'Sales': [150, 200, np.nan, 300, 250, 180],
        'Inventory Count': [25, 0, 10, 5, np.nan, 15],
        'Region': ['North', 'South', 'North', 'West', 'South', 'East']}
df_store = pd.DataFrame(data)

print("原始数据框:")
print(df_store)
```

此数据框包含产品信息，包括它们的类别、销售数字、库存数量和区域。请留意 `np.nan` 值，它们表示缺失数据。

### 找出缺失数据

在处理缺失数据之前，我们需要找出它们。`isnull()` 方法会返回一个布尔型数据框，其中 `True` 表示数据缺失。将它与 `sum()` 结合使用，可以得出每列缺失值的数量。

```python

print("\n每列的缺失值数量:")
print(df_store.isnull().sum())
```

输出:

```
Missing values per column:
ProductID          0
Category           1
Sales              1
Inventory Count    1
Region             0
dtype: int64
```

这说明在 `Category`、`Sales` 和 `Inventory Count` 列中各有一个缺失值。

### 处理缺失数据

我们有几种处理 `NaN` 值的方法。

#### 选项1：删除含有缺失值的行

如果我们认为任何含有缺失值的行都无法使用，我们可以使用 `dropna()` 删除这些行。

```python

df_dropped_rows = df_store.dropna()

print("\n删除含 NaN 值的行后的数据框:")
print(df_dropped_rows)
```

请注意，索引为 2、4 和 5 的行（分别对应 `Sales`、`Inventory Count` 和 `Category` 缺失的情况）已被移除。采用这种方法时要小心，因为如果缺失值很普遍，删除行可能导致大量数据丢失。

#### 选项2：填充缺失数据（填充处理）

通常，填充（或估算）缺失值是更好的做法。我们可以用固定值或计算值来填充。

让我们再次使用原始的 `df_store`。我们可能会决定用 'Unknown' 这样的占位符来填充缺失的 `Category`，并用各自列的平均值来填充缺失的 `Sales` 和 `Inventory Count`。

```python

df_filled = df_store.copy()

df_filled['Category'].fillna('Unknown', inplace=True)

mean_sales = df_filled['Sales'].mean()
df_filled['Sales'].fillna(mean_sales, inplace=True)

mean_inventory = df_filled['Inventory Count'].mean()
df_filled['Inventory Count'].fillna(mean_inventory, inplace=True)

print("\n填充 NaN 值后的数据框:")
print(df_filled)
print("\n填充后的缺失值:")
print(df_filled.isnull().sum())
```

`inplace=True` 参数 (parameter)会直接修改数据框。现在，所有缺失值都已替换。用平均值填充是一种常见方法，但最佳策略取决于具体的数据和情境。有时，用中位数或众数填充可能更合适，特别是当数据存在异常值或为分类数据时。

### 删除列或行

有时，整个列或特定的行是不相关或存在问题的。假设在 `df_filled` 数据框中，我们不需要 `Region` 列进行分析。

```python

df_no_region = df_filled.drop(columns=['Region'])

print("\n删除 'Region' 列后的数据框:")
print(df_no_region)
```

我们也可以根据索引标签删除行。假设产品 'P104'（索引 3）已停产，需要移除。

```python

df_dropped_row_3 = df_no_region.drop(index=3)

print("\n删除索引为 3 的行后的数据框:")
print(df_dropped_row_3)
```

### 添加和修改列

我们通常需要根据现有数据建立新列或修改现有列。

#### 添加新列

让我们添加一个 'SalesPerInventory' 列，通过 'Sales' 除以 'Inventory Count' 来计算。如果库存为 0，我们需要处理潜在的除以零问题。

```python

df_to_modify = df_no_region.copy()

df_to_modify['SalesPerInventory'] = df_to_modify['Sales'] / df_to_modify['Inventory Count'].replace(0, np.nan)

df_to_modify['SalesPerInventory'].fillna(0, inplace=True)

print("\n添加 'SalesPerInventory' 列后的数据框:")
print(df_to_modify)
```

#### 修改现有列

假设我们想将 'ProductID' 列转换为大写以保持一致性。

```python

df_to_modify['ProductID'] = df_to_modify['ProductID'].str.upper()

print("\n'ProductID' 转换为大写后的数据框:")
print(df_to_modify)
```

这里，我们使用了 `.str` 访问器，将字符串方法 `upper()` 应用到 'ProductID' 列。

### 重命名列

列名可能不清楚，或者包含不便于编写代码的字符（例如空格）。让我们将 'Inventory Count' 重命名为更简单的名称，例如 'Stock'。

```python

df_renamed = df_to_modify.rename(columns={'Inventory Count': 'Stock'})

print("\n列已重命名的数据框:")
print(df_renamed)
```

### 数据排序

通过排序组织数据可以使其更容易理解。

#### 按值排序

让我们将数据框按 'Sales' 降序排序。

```python

df_sorted_sales = df_renamed.sort_values(by='Sales', ascending=False)

print("\n按 Sales 降序排序的数据框:")
print(df_sorted_sales)
```

我们也可以按多列排序。让我们按 'Category'（升序）排序，然后在每个类别中按 'Stock'（降序）排序。

```python

df_sorted_multi = df_renamed.sort_values(by=['Category', 'Stock'], ascending=[True, False])

print("\n按 Category（升序）和 Stock（降序）排序的数据框:")
print(df_sorted_multi)
```

#### 按索引排序

如果需要，您也可以使用 `sort_index()` 按数据框的索引排序。

```python

df_sorted_index = df_sorted_multi.sort_index()

print("\n按索引恢复排序的数据框:")
print(df_sorted_index)
```

本次实践课详细介绍了本章中涵盖的核心数据清理和修改技术的使用：找出并处理缺失数据、添加、移除和重命名列，以及数据排序。这些是为进行有意义的分析而准备几乎任何数据集的基本步骤。随着您处理更复杂的数据，您将结合使用这些技术并查看更高级的 Pandas 功能，但这些基本知识是必要的。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 9 Grouping Aggregating Data Pandas

### 拆分-应用-组合方法

# 拆分-应用-组合方法

在处理数据集时，你通常需要看的不仅仅是单个行或列。一项常见而重要的工作是计算数据的*部分*的汇总统计量或对其进行操作，这些部分由一个或多个列中的值确定。例如，你可能想找出每个产品类别的平均销售额、每个网站部分的访问总数，或者每个气象站记录的最高温度。

分组分析通常遵循一种被称为“**拆分-应用-组合**”的方法。这是一种有用的思考方式，可以帮助理解Pandas等工具处理分组操作的方式。

### 1. 拆分

第一步是将原始DataFrame拆分成多个小块或组。这种划分是根据一个或多个指定列中的唯一值进行的，这些列通常被称为“分组键”。

想象你有一个包含销售数据的DataFrame，有`ProductCategory`和`SalesAmount`等列。如果你选择按`ProductCategory`分组，Pandas会划分DataFrame的行。所有`ProductCategory`为“Electronics”的行会构成一个组，所有为“Clothing”的行会构成另一个组，依此类推，适用于数据中所有独特的类别。每个部分都包含所有原始列，但只包含与特定键值对应的行。

### 2. 应用

数据被拆分成这些独立的组后，下一步就是对每个组应用一个函数。这个函数可以是：

- **聚合：** 计算每个组的单个汇总统计量（例如，`sum()`、`mean()`、`count()`、`min()`、`max()`）。这会把每个组简化为一个单一值或一组汇总值。
- **转换：** 进行一些针对组的计算，但得到一个与输入组形状相同的对象（例如，对每个组内的数据进行标准化）。
- **筛选：** 根据某些组级别的计算排除整个组（例如，只保留具有足够数据点的组）。

主要之处在于，所选函数在“拆分”阶段生成的每个组上独立运行。如果你计算的是每个类别的平均销售额，“Electronics”的平均值计算与“Clothing”的平均值计算是分开进行的。

### 3. 组合

最后，在“应用”阶段对每个组应用函数所得到的结果被收集并合并到一个新的数据形式中。通常，这种结果形式是一个新的Pandas Series或DataFrame。

这个结果对象的索引通常由“拆分”阶段确定的独特分组键构成。如果你计算了每个产品类别的平均销售额，最终结果很可能是一个Pandas Series，其索引包含独特的产产品类别（“Electronics”、“Clothing”等），而值则是每个组计算出的相应平均销售额。

G

cluster₀

原始 DataFrame

cluster₁

拆分

cluster₂

应用

cluster₃

组合

Original

DataFrame
(行, 列)

Group2

组 B

Original->Group2

GroupN

组 ...

Original->GroupN

Group1

按...分组

Apply1

应用函数
(例如, mean())

Group1->Apply1

Apply2

应用函数
(例如, mean())

Group2->Apply2

ApplyN

应用函数
(例如, mean())

GroupN->ApplyN

Result

结果
(新的 Series/DataFrame)

Apply1->Result

组合结果

Apply2->Result

ApplyN->Result

> “拆分-应用-组合”过程的图示。数据首先根据键被拆分成组，然后一个函数分别应用于每个组，最后将结果组合成一个最终输出。

这种“拆分-应用-组合”方法是一种通用的模式，适用于许多数据分析问题。在Pandas中，`groupby()`方法是支持此过程的主要工具。了解这种三阶段方法提供了一种清晰的思路，来考虑如何高效地进行复杂的组级操作。本章后续部分将向你展示如何使用Pandas `groupby()`来具体操作这个方法。

获取即时帮助、个性化解释和交互式代码示例。

---

### 使用 groupby() 方法对数据进行分组

# 使用 groupby() 方法对数据进行分组

“分-应用-合并”策略是数据分析中一种强大的模式。此过程的第一步是根据某些条件将数据分成组。Pandas 主要通过 `groupby()` 方法实现此目的。

可以将 `groupby()` 理解为告诉 Pandas：“将此 DataFrame 的行分离到不同的‘桶’中，每个‘桶’包含在特定列（或多列）中具有相同值的行。”此操作不会立即改变 DataFrame 或显示已分离的数据。相反，它会创建一个特殊的中间对象，称为 `GroupBy` 对象。该对象包含所有必要的分组信息，并已准备好进行下一步：对每个组应用函数（例如计算总和或平均值）。

### `groupby()` 方法

基本语法很简单：你在 DataFrame 上调用 `groupby()` 方法，并传入你希望作为分组依据的列名（或列名列表）。

让我们用一个例子来说明。假设我们有一个小数据集，记录了不同产品在各个区域的销售情况。

```python
import pandas as pd

data = {'Region': ['North', 'South', 'North', 'South', 'East', 'East', 'North'],
        'Product': ['A', 'A', 'B', 'B', 'A', 'B', 'A'],
        'Sales': [100, 150, 200, 50, 120, 80, 90]}
df = pd.DataFrame(data)

print("原始 DataFrame:")
print(df)
```

Output:

```
原始 DataFrame:
  Region Product  Sales
0  North       A    100
1  South       A    150
2  North       B    200
3  South       B     50
4   East       A    120
5   East       B     80
6  North       A     90
```

现在，让我们按照‘Region’列对这个 DataFrame 进行分组：

```python

grouped_by_region = df.groupby('Region')

print("\ngroupby('Region') 的结果：")
print(grouped_by_region)
```

Output:

```
groupby('Region') 的结果：
<pandas.core.groupby.generic.DataFrameGroupBy object at 0x...>
```

注意，输出本身并不是分组后的数据。它是一个 `DataFrameGroupBy` 对象。这个对象实际上包含多个小型 DataFrame，‘Region’列中的每个唯一值（‘North’、‘South’、‘East’）对应一个。

G

cluster\_df

原始 DataFrame (df)

cluster\_groupby

df.groupby('Region')

cluster\_groups

组（GroupBy 对象内部）

dfₙode

区域

产品

销售额

North

A

100

South

A

150

North

B

200

South

B

50

East

A

120

East

B

80

North

A

90

groupbyₒbj

DataFrameGroupBy 对象
(包含分组信息)

dfₙode->groupbyₒbj

groupby('Region')

groupₙorth

组：North

North

A

100

North

B

200

North

A

90

groupbyₒbj->groupₙorth

groupₛouth

组：South

South

A

150

South

B

50

groupbyₒbj->groupₛouth

groupₑast

组：East

East

A

120

East

B

80

groupbyₒbj->groupₑast

> 将 `groupby('Region')` 应用到 DataFrame 的示意图。该方法创建一个 `GroupBy` 对象，其中包含原始 DataFrame 中每个唯一区域的子集引用。

### GroupBy 对象有何用途？

这个 `GroupBy` 对象是分-应用-合并过程的根本。虽然它本身不直接显示太多，但已准备好进行“应用”步骤。你可以对其执行各种操作，例如：

1. **聚合：** 为每个组独立计算汇总统计量（如平均值、总和、计数）。这是最常见的用法，将在下一节（“应用聚合函数”）中说明。
2. **转换：** 执行组特定的计算，并返回与原始 DataFrame 形状一致的结果。
3. **筛选：** 根据某些组级别计算丢弃整个组。
4. **迭代：** 遍历各个组（我们将在后面的“迭代分组”中看到这一点）。

目前，重要的要点是 `df.groupby('column_name')` 能根据指定列高效地分割 DataFrame，并创建一个 `GroupBy` 对象，为后续对每个组的分析打下基础。

获取即时帮助、个性化解释和交互式代码示例。

---

### 应用聚合函数

# 应用聚合函数

应用聚合函数首先需要通过在 DataFrame 上调用 `groupby()` 方法来创建 `GroupBy` 对象。然后，下一步是对每个组应用一个函数以计算汇总统计量。这一操作是“拆分-应用-合并”模式中的“应用”部分。

Pandas 的 `GroupBy` 对象内置了多个聚合方法，它们与 Series 和 DataFrame 上的对应方法功能非常相似。这些方法会自动独立地对每个组进行操作，然后将结果合并到一个新的 Series 或 DataFrame 中。

让我们看一个简单的数据框，它表示不同产品在各个区域的销售数据：

```python
import pandas as pd
import numpy as np

data = {'Region': ['North', 'South', 'North', 'South', 'East', 'East', 'North'],
        'Product': ['A', 'A', 'B', 'B', 'A', 'C', 'B'],
        'Sales': [100, 150, 200, 50, 120, 80, 180],
        'Quantity': [10, 15, 20, 5, 12, 8, 15]}
df = pd.DataFrame(data)

print("原始数据框:")
print(df)
```

```text
原始数据框:
  Region Product  Sales  Quantity
0  North       A    100        10
1  South       A    150        15
2  North       B    200        20
3  South       B     50         5
4   East       A    120        12
5   East       C     80         8
6  North       B    180        15
```

现在，让我们按“Region”对这些数据进行分组：

```python
grouped_by_region = df.groupby('Region')
```

`grouped_by_region` 对象现在包含已分离的组，但我们尚未进行任何计算。

### 常用聚合函数

您可以直接将常用聚合函数应用于 `GroupBy` 对象。Pandas 会智能地将函数应用于每个组中合适的列（通常是数值列）。

1. **求和 (`.sum()`)**：计算每个组的值的总和。

   ```python

   region_totals = grouped_by_region.sum()
   print("\n每个区域的总销售额和总数量:")
   print(region_totals)
   ```

   ```text
   每个区域的总销售额和总数量:
          Sales  Quantity
   Region
   East     200        20
   North    480        45
   South    200        20
   ```

   请注意，输出是一个新的数据框，其中索引是分组键（“Region”），列是原始数据框中的数值列（“Sales”、“Quantity”），包含每个区域的总和值。非数值的“Product”列自动从求和中排除。
2. **平均值 (`.mean()`)**：计算每个组的平均值。

   ```python

   region_means = grouped_by_region.mean(numeric_only=True)
   print("\n每个区域的平均销售额和平均数量:")
   print(region_means)
   ```

   ```text
   每个区域的平均销售额和平均数量:
             Sales  Quantity
   Region
   East   100.000000  10.000000
   North  160.000000  15.000000
   South  100.000000  10.000000
   ```

   同样，输出索引是“Region”，值代表属于每个区域的行的“Sales”和“Quantity”的平均值。
3. **计数 (`.count()`)**：计算每个组中每列的非空条目数。

   ```python

   region_counts = grouped_by_region.count()
   print("\n每个区域的条目数:")
   print(region_counts)
   ```

   ```text
   每个区域的条目数:
          Product  Sales  Quantity
   Region
   East         2      2         2
   North        3      3         3
   South        2      2         2
   ```

   这里，`count()` 包含“Product”列，因为它计算任何非缺失值，无论数据类型如何。它显示了有多少记录属于每个区域的组。
4. **大小 (`.size()`)**：返回每个组的总行数（包括空值，与 `count()` 不同）。

   ```python

   region_sizes = grouped_by_region.size()
   print("\n每个区域组的大小:")
   print(region_sizes)
   ```

   ```text
   每个区域组的大小:
   Region
   East     2
   North    3
   South    2
   dtype: int64
   ```

   `size()` 的输出是一个 Pandas Series，其中索引是分组键（“Region”），值是属于该组的行数。
5. **最小值 (`.min()`) 和最大值 (`.max()`)**：找出每个组中每个适用列的最小值或最大值。

   ```python

   region_min_sales = grouped_by_region['Sales'].min()
   print("\n每个区域的最小销售额:")
   print(region_min_sales)

   region_max_quantity = grouped_by_region['Quantity'].max()
   print("\n每个区域的最大数量:")
   print(region_max_quantity)
   ```

   ```text
   每个区域的最小销售额:
   Region
   East      80
   North    100
   South     50
   Name: Sales, dtype: int64

   每个区域的最大数量:
   Region
   East     12
   North    20
   South    15
   Name: Quantity, dtype: int64
   ```

   在这些示例中，我们首先从 `GroupBy` 对象中选择一个特定列（`['Sales']` 或 `['Quantity']`），然后再应用聚合。这会得到一个 Series，其中索引是分组键（“Region”），值是该组所选列的最小/最大值。如果您在未首先选择列的情况下直接将 `.min()` 或 `.max()` 应用于 `GroupBy` 对象，它将计算所有适用（通常是数值）列的最小/最大值，类似于 `.sum()` 或 `.mean()`。

### 应用于特定列

正如在 `.min()` 和 `.max()` 中所见的，您可以在分组后将聚合函数应用于特定列。当您只需要某些特征的汇总时，这很有用。

```python

total_sales_per_region = df.groupby('Region')['Sales'].sum()
print("\n每个区域的总销售额（特定列）:")
print(total_sales_per_region)

avg_quantity_per_product = df.groupby('Product')['Quantity'].mean()
print("\n每个产品的平均数量:")
print(avg_quantity_per_product)
```

```text
每个区域的总销售额（特定列）:
Region
East     200
North    480
South    200
Name: Sales, dtype: int64

每个产品的平均数量:
Product
A    12.333333
B    13.333333
C     8.000000
Name: Quantity, dtype: float64
```

在聚合*之前*选择列 (`df.groupby('Region')['Sales'].sum()`) 通常比计算所有列的聚合然后选择您需要的列 (`df.groupby('Region').sum()['Sales']`) 更有效，尤其是在大型数据集上。

这些基本聚合函数（`sum`、`mean`、`count`、`size`、`min`、`max`、`std`、`var`、`median` 等）涵盖了许多常见的数据汇总任务。它们是理解 Pandas 中按组操作的基本要素。在下一节中，我们将学习如何一次应用多个聚合函数。

获取即时帮助、个性化解释和交互式代码示例。

---

### 应用多种聚合操作

# 应用多种聚合操作

分析分组数据时，仅计算平均值或总和等单一汇总统计量通常不够。您可能希望同时查看每个组的几种不同汇总信息。例如，对于每个产品类别，您可能想了解总销售额 *和* 平均销售额。Pandas 提供了灵活的方法，通过 GroupBy 对象的 `agg()` 方法来实现这一点。

让我们从一个表示销售数据的示例 DataFrame 开始：

```python
import pandas as pd
import numpy as np

data = {'Category': ['Electronics', 'Clothing', 'Electronics', 'Clothing', 'Groceries', 'Electronics', 'Groceries'],
        'Product': ['Laptop', 'T-Shirt', 'Mouse', 'Jeans', 'Apples', 'Keyboard', 'Bananas'],
        'Sales': [1200, 25, 20, 50, 5, 75, 3],
        'Quantity': [1, 2, 1, 1, 10, 1, 8]}
df = pd.DataFrame(data)

print("原始 DataFrame:")
print(df)

grouped = df.groupby('Category')
```

运行此代码将显示我们的初始数据：

```
Original DataFrame:
      Category   Product  Sales  Quantity
0  Electronics    Laptop   1200         1
1     Clothing   T-Shirt     25         2
2  Electronics     Mouse     20         1
3     Clothing     Jeans     50         1
4    Groceries    Apples      5        10
5  Electronics  Keyboard     75         1
6    Groceries   Bananas      3         8
```

### 使用列表应用多个函数

应用多个聚合函数最简单的方法是向 `agg()` 方法传递一个函数名称列表（作为字符串）。Pandas 会将列表中的每个函数应用于分组选择中的每个数值列。

```python

multi_agg_list = grouped[['Sales', 'Quantity']].agg(['sum', 'mean'])

print("\n使用列表进行多重聚合:")
print(multi_agg_list)
```

输出显示了每个类别中 'Sales' 和 'Quantity' 列的总和与平均值：

```
Multiple aggregations using a list:
             Sales        Quantity
               sum     mean      sum mean
Category
Clothing        75    37.50        3  1.5
Electronics   1295  431.67        3  1.0
Groceries        8     4.00       18  9.0
```

请注意，结果具有分层列标签（MultiIndex）。顶层表示原始列（'Sales'，'Quantity'），第二层表示聚合函数（'sum'，'mean'）。

### 使用字典为每列指定聚合

如果您想对不同的列应用不同的函数怎么办？例如，您可能希望按类别获取总销售额（`sum`），但获取平均数量（`mean`）。您可以通过向 `agg()` 传递一个字典来实现这一点。字典的键应该是您要聚合的列名，值应该是应用于该特定列的函数（或函数列表）。

```python

multi_agg_dict = grouped.agg({
    'Sales': 'sum',
    'Quantity': 'mean'
})

print("\n使用字典进行多重聚合:")
print(multi_agg_dict)
```

这会得到一个更整洁的输出，没有分层列，因为每个指定的聚合都会产生一个单独的输出列：

```
Multiple aggregations using a dictionary:
             Sales  Quantity
Category
Clothing        75       1.5
Electronics   1295       1.0
Groceries        8       9.0
```

您还可以使用字典中的列表对特定列应用多个函数：

```python

multi_agg_dict_list = grouped.agg({
    'Sales': ['sum', 'mean'],
    'Quantity': 'sum'
})

print("\n使用函数列表的字典聚合:")
print(multi_agg_dict_list)
```

现在输出仅在 'Sales' 列中具有分层列，因为对其应用了多个函数：

```
Dictionary aggregation with a list of functions:
             Sales        Quantity
               sum     mean      sum
Category
Clothing        75    37.50        3
Electronics   1295  431.67        3
Groceries        8     4.00       18
```

### 命名聚合以获得更清晰的输出

虽然字典方法很有用，但管理潜在复杂的层次结构列名可能会变得繁琐。一种更现代且通常更清晰的方法是使用 *命名聚合*。这允许您明确定义输出列的名称。

您向 `agg()` 传递关键字参数 (parameter)，其中关键字是您想要的输出列名。与每个关键字关联的值是一个元组，包含 `(要聚合的列名, 聚合函数)`。

```python

named_agg = grouped.agg(
    Total_Sales = pd.NamedAgg(column='Sales', aggfunc='sum'),
    Avg_Quantity = pd.NamedAgg(column='Quantity', aggfunc='mean'),
    Num_Products = pd.NamedAgg(column='Product', aggfunc='count')
)

print("\n命名聚合以获得更清晰的输出列:")
print(named_agg)
```

这会生成一个 DataFrame，其列名清晰地反映了执行的特定聚合操作：

```
Named aggregations for clearer output columns:
             Total_Sales  Avg_Quantity  Num_Products
Category
Clothing              75           1.5             2
Electronics         1295           1.0             3
Groceries              8           9.0             2
```

当应用多个可能不同的聚合操作时，通常推荐使用这种命名聚合语法（`pd.NamedAgg`），因为它能使代码更易读，并产生可预测的输出结构。

您甚至可以在 `agg()` 中应用使用 `def` 或 lambda 表达式定义的自定义函数，尽管对于标准统计数据，内置函数名（作为字符串）通常已足够且更高效。

能够使用 `agg()` 计算每个组的多个汇总统计量是数据分析中非常常见且实用的模式，它能让您快速从分组数据中生成有洞察力的汇总信息。

获取即时帮助、个性化解释和交互式代码示例。

---

### 按多列分组

# 按多列分组

数据分析常常需要根据类别的组合而不是单一类别来划分数据。例如，您可能希望计算的平均销售额不仅按区域划分，还要按每个区域*内*的产品划分。Pandas 的 `groupby()` 方法通过允许您同时按多列进行分组，使这变得简单易行。

### 为什么要按多列分组？

按单列分组可以提供宽泛类别的汇总。按多列分组则可以进行更详细、分层的分析。它有助于回答以下问题：

- 每个月*和*每个城市的平均降雨量是多少？
- 在每个具体商店\*中，每种产品销售了多少单位？
- 在每个学校内\*，*每个年级*的学生取得的最高分数是多少？

这种多级分组相比单级聚合提供了更细致的见解。

### 语法

要按多列分组，您只需向 `groupby()` 方法传递一个列名列表，而不是单个字符串。

让我们设置一个示例 DataFrame 来进行说明：

```python
import pandas as pd
import numpy as np

data = {'Region': ['North', 'South', 'North', 'South', 'North', 'South', 'North', 'South'],
        'Product': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
        'Sales': [100, 150, 200, 50, 120, 180, 210, 80],
        'Quantity': [10, 15, 20, 5, 12, 18, 21, 8]}
df_sales = pd.DataFrame(data)

print(df_sales)
```

输出：

```
  Region Product  Sales  Quantity
0  North       A    100        10
1  South       A    150        15
2  North       B    200        20
3  South       B     50         5
4  North       A    120        12
5  South       A    180        18
6  North       B    210        21
7  South       B     80         8
```

现在，让我们同时按“Region”和“Product”分组，并计算每个组合的总销售额：

```python

grouped_multi = df_sales.groupby(['Region', 'Product'])

total_sales_multi = grouped_multi['Sales'].sum()

print(total_sales_multi)
```

输出：

```
Region  Product
North   A          220
        B          410
South   A          330
        B          130
Name: Sales, dtype: int64
```

### 理解结果：MultiIndex

请注意聚合结果 (`total_sales_multi`)。索引不再是简单的标签列表。相反，它有两层：“Region”和“Product”。这在 Pandas 中被称为 `MultiIndex`（或分层索引）。它代表了分组列的独特组合。

`df_sales.groupby(['Region', 'Product'])` 创建的 `GroupBy` 对象根据独有的对（'North', 'A'），（'North', 'B'），（'South', 'A'）和（'South', 'B'）来划分原始 DataFrame。然后，对这些划分中每个“Sales”列应用 `sum()` 聚合。

### 聚合多列或使用多个函数

就像单列分组一样，您可以在按多列分组时对多列应用聚合，或使用多个聚合函数。

例如，让我们找出每个区域-产品组合的总销售额和平均数量：

```python

agg_results = grouped_multi.agg(
    Total_Sales=('Sales', 'sum'),
    Average_Quantity=('Quantity', 'mean')
)

print(agg_results)
```

输出：

```
                Total_Sales  Average_Quantity
Region Product
North  A                220              11.0
       B                410              20.5
South  A                330              16.5
       B                130               6.5
```

结果是一个 DataFrame，其索引是 `MultiIndex`（“Region”，“Product”），列则代表指定聚合的结果（“Total\_Sales”，“Average\_Quantity”）。

您也可以对同一列应用多个聚合函数：

```python

sales_stats = grouped_multi['Sales'].agg(['sum', 'mean', 'count'])

print(sales_stats)
```

输出：

```
                sum   mean  count
Region Product
North  A        220  110.0      2
       B        410  205.0      2
South  A        330  165.0      2
       B        130   65.0      2
```

这提供了每个独特（“Region”，“Product”）对的销售额的总和、平均值和计数。

按多列分组是一种有效方法，用于进行详细的类别分析，允许您根据多种因素组合来汇总数据。生成的 `MultiIndex` 结构能有效表示这些分层分组。

获取即时帮助、个性化解释和交互式代码示例。

---

### 遍历分组

# 遍历分组

Pandas 中的强大 `groupby()` 方法可以根据列值将 DataFrame 分割成多个部分。它允许使用“分割-应用-合并”策略应用 `mean()`、`sum()` 或 `count()` 等聚合函数。这对于汇总数据极其有用。

然而，有时仅进行简单的聚合是不够的。你可能需要对每个分组单独执行更复杂的操作，查看每个分组内的数据，或者应用一个不完全符合标准聚合框架的函数。对于这些情况，Pandas 允许你直接遍历由 `groupby()` 创建的分组。

当你对 DataFrame 调用 `.groupby()` 时，它不会立即计算出任何可见的结果，例如聚合。相反，它会返回一个特殊的 `GroupBy` 对象。此对象包含管理不同分组所需的所有信息。你可以将其看作是多个较小 DataFrame 的集合，其中每个 DataFrame 对应你分组所依据的列中一个唯一值（或值组合）。

这个 `GroupBy` 对象是可迭代的，意味着你可以遍历它，就像你在 Python 中遍历列表一样。当你迭代 `GroupBy` 对象时，每次迭代都会生成一个包含两个元素的元组：

1. **分组名称（或名称）：** 这是定义当前分组的列中的唯一值。如果你按单个列分组，这将是一个单个值（例如字符串或数字）。如果你按多个列分组，这将是一个包含该分组值组合的元组。
2. **分组数据：** 这是一个 DataFrame，只包含原始 DataFrame 中属于当前分组的行。

标准的遍历方式是使用 `for` 循环：

```python

for name, group_df in grouped:

    print(f"正在处理分组: {name}")

    print(group_df.head(2))

    print("-" * 20)
```

让我们通过一个例子来说明。假设我们有一个小型 DataFrame，它记录了不同地区不同产品的销售数据：

```python
import pandas as pd

data = {'Region': ['North', 'South', 'North', 'South', 'West', 'North', 'West'],
        'Product': ['A', 'A', 'B', 'B', 'A', 'B', 'B'],
        'Sales': [100, 150, 200, 250, 50, 210, 70]}
sales_df = pd.DataFrame(data)

print("原始 DataFrame:")
print(sales_df)
print("\n")

grouped_by_region = sales_df.groupby('Region')

print("按地区遍历分组:")
for region_name, region_group in grouped_by_region:
    print(f"地区: {region_name}")
    print("该地区的数据:")
    print(region_group)
    print("-" * 30)
```

运行此代码将输出：

```
原始 DataFrame:
  Region Product  Sales
0  North       A    100
1  South       A    150
2  North       B    200
3  South       B    250
4   West       A     50
5  North       B    210
6   West       B     70

按地区遍历分组:
地区: North
该地区的数据:
  Region Product  Sales
0  North       A    100
2  North       B    200
5  North       B    210
------------------------------
地区: South
该地区的数据:
  Region Product  Sales
1  South       A    150
3  South       B    250
------------------------------
地区: West
该地区的数据:
  Region Product  Sales
4   West       A     50
6   West       B     70
------------------------------
```

请注意，每次迭代都提供了 `region_name`（例如 'North'、'South'、'West'）以及一个 `region_group` DataFrame，其中只包含匹配该地区的行。

### 遍历多个分组列

如果你按多个列分组，循环中的 `name` 变量将成为一个元组，其中包含定义该分组的值组合。

```python

grouped_multi = sales_df.groupby(['Region', 'Product'])

print("\n按地区和产品遍历分组:")
for (region_name, product_name), group_data in grouped_multi:
    print(f"分组键: 地区={region_name}, 产品={product_name}")
    print("该分组的数据:")
    print(group_data)
    print("-" * 30)
```

输出将显示由地区和产品的唯一配对定义的分组：

```
按地区和产品遍历分组:
分组键: 地区=North, 产品=A
该分组的数据:
  Region Product  Sales
0  North       A    100
------------------------------
分组键: 地区=North, 产品=B
该分组的数据:
  Region Product  Sales
2  North       B    200
5  North       B    210
------------------------------
分组键: 地区=South, 产品=A
该分组的数据:
  Region Product  Sales
1  South       A    150
------------------------------
分组键: 地区=South, 产品=B
该分组的数据:
  Region Product  Sales
3  South       B    250
------------------------------
分组键: 地区=West, 产品=A
该分组的数据:
  Region Product  Sales
4   West       A     50
------------------------------
分组键: 地区=West, 产品=B
该分组的数据:
  Region Product  Sales
6   West       B     70
------------------------------
```

### 何时使用遍历有益？

虽然标准聚合函数（`.sum()`、`.mean()`、`.agg()`）高效且适用于多种用例，但遍历分组在以下情况时有益：

1. **应用复杂函数：** 你需要对每个分组应用一个函数，但该函数无法通过内置聚合方法或 `agg()` 中的 lambda 函数轻松表达。
2. **分组特定逻辑：** 处理逻辑在不同分组之间差异很大，需要基于分组名称或其数据的条件语句。
3. **生成可视化：** 你希望为每个分组创建单独的图表或可视化。遍历允许你顺序访问每个图表的数据。
4. **调试：** 你需要检查每个分组的具体内容，以了解为什么聚合可能产生意料之外的结果。
5. **详细报告：** 你需要生成一份报告，其中包含数据的每个部分的详细信息或特定计算。

请记住，遍历分组的计算效率可能低于使用 Pandas 内置的聚合函数，特别是对于非常大的数据集。Pandas 用于标准聚合的向量 (vector)化操作是高度优化的。因此，在可能的情况下，优先使用内置方法；当遍历提供的灵活性对你的特定任务是必需时，才使用它。

获取即时帮助、个性化解释和交互式代码示例。

---

### 动手实践：使用 GroupBy 汇总数据

# 动手实践：使用 GroupBy 汇总数据

分组和聚合是数据操作中的基本概念。我们将通过使用示例数据集的多个例子，演示 Pandas `groupby()` 的分拆-应用-组合模式。

### 设置示例数据

首先，确保已导入 Pandas。如果您在 Jupyter Notebook 或 Python 脚本中跟随操作，请从这里开始：

```python
import pandas as pd
import numpy as np
```

现在，我们来创建一个简单的 DataFrame，它表示不同产品类别在各个区域的销售数据。这将是我们练习的数据。

```python

data = {'Category': ['Electronics', 'Clothing', 'Electronics', 'Groceries', 'Clothing', 'Groceries', 'Electronics', 'Clothing'],
        'Region': ['North', 'South', 'North', 'East', 'West', 'East', 'West', 'South'],
        'Sales': [1200, 300, 1500, 250, 450, 200, 1800, 350],
        'Quantity': [2, 3, 1, 5, 4, 4, 3, 4]}
sales_df = pd.DataFrame(data)

print("示例销售数据：")
print(sales_df)
```

这应该输出：

```
Sample Sales Data:
      Category Region  Sales  Quantity
0  Electronics  North   1200         2
1     Clothing  South    300         3
2  Electronics  North   1500         1
3    Groceries   East    250         5
4     Clothing   West    450         4
5    Groceries   East    200         4
6  Electronics   West   1800         3
7     Clothing  South    350         4
```

### 按单列分组

我们的首要任务是找出每个产品`类别`的总销售额。这需要按`类别`拆分 DataFrame，然后对每个组内的`销售额`列应用 `sum()` 函数，最后合并结果。

```python

category_sales_sum = sales_df.groupby('Category')['Sales'].sum()

print("\n每个类别的总销售额：")
print(category_sales_sum)
```

输出：

```
Total Sales per Category:
Category
Clothing        1100
Electronics     4500
Groceries        450
Name: Sales, dtype: int64
```

请注意，结果是一个 Pandas Series，其索引是我们分组的`类别`，值是汇总后的`销售额`。

如果我们想知道每个类别的*平均*销售数量呢？

```python

category_quantity_avg = sales_df.groupby('Category')['Quantity'].mean()

print("\n每个类别的平均数量：")
print(category_quantity_avg)
```

输出：

```
Average Quantity per Category:
Category
Clothing       3.666667
Electronics    2.000000
Groceries      4.500000
Name: Quantity, dtype: float64
```

您也可以类似地应用其他聚合函数，例如 `count()`、`size()`、`min()`、`max()`、`std()`（标准差）等。请记住，`count()` 不包括缺失值 (NaN)，而 `size()` 包括它们。

### 应用多个聚合操作

通常，您会希望一次性计算每个组的多个汇总统计量。`.agg()` 方法非常适合此目的。

我们来计算每个`类别`的总销售额 (`sum`) 和平均数量 (`mean`)。

```python

category_summary = sales_df.groupby('Category').agg(
    Total_Sales=('Sales', 'sum'),
    Average_Quantity=('Quantity', 'mean'),
    Number_of_Sales=('Sales', 'count')
)

print("\n每个类别的汇总统计量：")
print(category_summary)
```

输出：

```
Summary Statistics per Category:
             Total_Sales  Average_Quantity  Number_of_Sales
Category
Clothing            1100          3.666667                3
Electronics         4500          2.000000                3
Groceries            450          4.500000                2
```

在这里，我们向 `.agg()` 传递了一个字典。字典的键（'Total\_Sales'、'Average\_Quantity'、'Number\_of\_Sales'）成为结果 DataFrame 中的新列名。值是元组，其中第一个元素是要聚合的原始列（'Sales' 或 'Quantity'），第二个元素是聚合函数（'sum'、'mean'、'count'）。这提供了一个清晰且有条理的汇总。

或者，如果您想将相同的聚合函数集应用于所有数值列（或选定列），可以传递一个函数列表：

```python

category_sales_min_max = sales_df.groupby('Category')['Sales'].agg(['min', 'max'])

print("\n每个类别的最小和最大销售额：")
print(category_sales_min_max)
```

输出：

```
Min and Max Sales per Category:
             min   max
Category
Clothing     300   450
Electronics 1200  1800
Groceries    200   250
```

### 按多列分组

通过向 `groupby()` 传递列名列表，我们可以创建更细粒度的分组。我们来找出`区域`和`类别`每个组合的总销售额。

```python

region_category_sales = sales_df.groupby(['Region', 'Category'])['Sales'].sum()

print("\n每个区域和类别的总销售额：")
print(region_category_sales)
```

输出：

```
Total Sales per Region and Category:
Region  Category
East    Groceries     450
North   Electronics  2700
South   Clothing      650
West    Clothing      450
        Electronics  1800
Name: Sales, dtype: int64
```

结果是一个带有 `MultiIndex`（区域、类别）的 Series。这个分层索引表示分组的组合。

您也可以在这里应用多个聚合操作：

```python

region_category_summary = sales_df.groupby(['Region', 'Category']).agg(
    Total_Sales=('Sales', 'sum'),
    Average_Quantity=('Quantity', 'mean')
)

print("\n每个区域和类别的汇总：")
print(region_category_summary)
```

输出：

```
Summary per Region and Category:
                     Total_Sales  Average_Quantity
Region Category
East   Groceries             450               4.5
North  Electronics          2700               1.5
South  Clothing              650               3.5
West   Clothing              450               4.0
       Electronics          1800               3.0
```

这给我们一个行上带有 `MultiIndex` 的 DataFrame。

### 分组数据的可视化

聚合数据通常通过可视化更容易理解。我们来使用 Plotly 创建一个简单的条形图，显示每个类别的总销售额。

ClothingElectronicsGroceries01000200030004000各产品类别的总销售额类别总销售额 ($)

> 各产品类别产生的总销售额。在这个小型数据集中，电子产品明显占主导地位。

### 练习任务

现在，请使用 `sales_df` DataFrame 独自尝试以下任务：

1. 找出每个`区域`的平均`销售额`。
2. 统计每个`区域`的销售记录数量 (`size()`)。
3. 找出每个`区域`和`类别`组合中的`最大`销售数量。
4. 计算每个`区域`的`销售额`总和与`数量`总和。使用 `.agg()` 方法。

这些练习巩固了您应用 `groupby()` 和聚合函数，从数据中提取有价值汇总的不同方法。掌握分组是使用 Pandas 进行数据分析的重要一步。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 10 Combining Dataframes Pandas

### 数据合并介绍

# 数据合并介绍

正如我们在本章开篇提到的，数据很少整齐地打包在一个文件中或一张表中。更多时候，您需要的信息分散在多个来源。客户详细信息可能与他们的订单历史分开存储，或者实验结果记录在不同时间段的不同文件中。为了获得完整的数据视图并进行有意义的分析，您需要方法将这些分散的数据整合起来。

请考虑以下常见情况：

- **客户分析：** 设想一个 `DataFrame` 包含客户人口统计信息（如ID、姓名、地点），另一个包含交易记录（如客户ID、购买产品、日期、金额）。要了解哪些人口统计特征与特定购买行为相关联，您需要使用共同的 `客户ID` 将这两个 DataFrame 连接起来。
- **时间序列聚合：** 您可能会收到存储在单个文件中的每日传感器读数（例如，`readings_2023-10-26.csv`、`readings_2023-10-27.csv`）。为了进行周或月度分析，您需要将这些单独的每日 DataFrame 堆叠成一个更大的 `DataFrame`。
- **特征丰富：** 您可能有一个关于产品的主数据集，以及包含附加特征的补充数据集，例如产品类别信息或供应商详情。合并这些数据可以让您的主数据集获得更多背景信息。

数据通常不会整齐地存储在单个文件或表格中，而是分散在多个来源。为了完整地查看和有意义地分析数据，需要能够智能地组合 `DataFrame` 对象的工具。当目标是基于共享信息或结构对齐 (alignment)和集成数据时，简单地执行逐元素加法或逐个添加列通常是不够的。

本章将介绍专门为组合数据集设计的主要 Pandas 函数：

1. **拼接 (`pd.concat`)**：这对于将数据集相互堆叠（追加行）或并排放置（添加列）非常有用。可以将其想象成沿着一个轴将表格粘合在一起。
2. **合并与连接 (`pd.merge`, `.join`)**：这些方法执行数据库样式的连接。它们根据一个或多个共享列（称为键）中的值或基于 DataFrame 索引来组合数据集。当您需要链接来自不同表的相关信息时，例如将客户ID与交易关联起来，这非常必要。

理解何时以及如何使用这些技术是数据准备和分析中的一项基本技能。它使您能够从分散的来源构建统一的数据集，从而进行更全面和有见地的分析。以下章节将详细说明拼接和合并的工作原理，并通过实际示例说明其用法。

获取即时帮助、个性化解释和交互式代码示例。

---

### 连接 DataFrames (pd.concat)

# 连接 DataFrames (pd.concat)

合并 DataFrames 最直接的方法之一是进行**连接**。可以将其想象成将纸张（您的 DataFrames）一张叠在另一张上面（追加行）或并排放置（添加列）。Pandas 提供了 `pd.concat()` 函数来实现此功能。

`pd.concat()` 函数将其主要参数 (parameter)设为 Series 或 DataFrame 对象的列表或序列，并巧妙地将它们拼接在一起。

### 沿行连接（垂直堆叠）

默认情况下，`pd.concat()` 垂直堆叠 DataFrames，这意味着它将第二个 DataFrame 的行追加到第一个的末尾，将第三个追加到第二个的末尾，依此类推。这沿 `axis=0` 进行。

让我们创建两个简单的 DataFrame 进行说明：

```python
import pandas as pd

df1 = pd.DataFrame({'A': ['A0', 'A1'],
                    'B': ['B0', 'B1']},
                   index=[0, 1])

df2 = pd.DataFrame({'A': ['A2', 'A3'],
                    'B': ['B2', 'B3']},
                   index=[2, 3])

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

result_rows = pd.concat([df1, df2])

print("\n连接后的 DataFrame (行):")
print(result_rows)
```

**Output:**

```
DataFrame 1:
    A   B
0  A0  B0
1  A1  B1

DataFrame 2:
    A   B
2  A2  B2
3  A3  B3

Concatenated DataFrame (Rows):
    A   B
0  A0  B0
1  A1  B1
2  A2  B2
3  A3  B3
```

可以看到，`df2` 的行直接追加在 `df1` 的行下方。列自动对齐 (alignment)，因为两个 DataFrame 具有相同的列名（'A' 和 'B'）。

### 行连接期间处理索引

请注意 `result_rows` DataFrame 中的索引：`[0, 1, 2, 3]`。在这种情况下，原始索引是唯一的并且很好地组合在一起。然而，如果原始 DataFrame 具有重叠的索引标签怎么办？

让我们修改 `df2` 使其与 `df1` 具有重叠索引：

```python

df1 = pd.DataFrame({'A': ['A0', 'A1'],
                    'B': ['B0', 'B1']},
                   index=[0, 1])

df2_overlap = pd.DataFrame({'A': ['A2', 'A3'],
                            'B': ['B2', 'B3']},
                           index=[1, 2])

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2 (重叠索引):")
print(df2_overlap)

result_overlap = pd.concat([df1, df2_overlap])

print("\n连接后 (重叠索引):")
print(result_overlap)
print("\n结果的索引:")
print(result_overlap.index)
```

**Output:**

```
DataFrame 1:
    A   B
0  A0  B0
1  A1  B1

DataFrame 2 (Overlapping Index):
    A   B
1  A2  B2
2  A3  B3

Concatenated (Overlapping Index):
    A   B
0  A0  B0
1  A1  B1
1  A2  B2  # 重复索引标签 1
2  A3  B3

Index of the result:
Index([0, 1, 1, 2], dtype='int64')
```

默认情况下，`pd.concat()` 会保留原始索引，即使这会导致重复（例如上面例子中的索引 `1`）。尽管 Pandas 允许重复索引，但它们有时会使通过标签选择数据（`.loc`）变得模糊，或在后续操作中导致意外行为。

如果您不需要保留原始索引，并且更喜欢为结果 DataFrame 提供一个清晰、唯一的索引，您可以使用 `ignore_index=True` 参数 (parameter)：

```python

result_ignore_index = pd.concat([df1, df2_overlap], ignore_index=True)

print("\n连接后 (忽略索引):")
print(result_ignore_index)
print("\n结果的索引 (已忽略):")
print(result_ignore_index.index)
```

**Output:**

```
Concatenated (Ignoring Index):
    A   B
0  A0  B0
1  A1  B1
2  A2  B2
3  A3  B3

Index of the result (Ignored):
RangeIndex(start=0, stop=4, step=1)
```

将 `ignore_index` 设置为 `True` 会丢弃原始索引，并为连接后的 DataFrame 分配一个新的默认整数索引（`RangeIndex`）。

### 沿列连接（水平堆叠）

您也可以通过指定 `axis=1` 将 DataFrames 并排连接。这会根据它们的索引标签对齐 (alignment) DataFrames，并将它们的列放置在彼此旁边。

让我们创建两个索引相同但列不同的 DataFrame：

```python

df1 = pd.DataFrame({'A': ['A0', 'A1'],
                    'B': ['B0', 'B1']},
                   index=[0, 1])

df3 = pd.DataFrame({'C': ['C0', 'C1'],
                    'D': ['D0', 'D1']},
                   index=[0, 1])

print("DataFrame 1:")
print(df1)
print("\nDataFrame 3:")
print(df3)

result_cols = pd.concat([df1, df3], axis=1)

print("\n连接后的 DataFrame (列):")
print(result_cols)
```

**Output:**

```
DataFrame 1:
    A   B
0  A0  B0
1  A1  B1

DataFrame 3:
    C   D
0  C0  D0
1  C1  D1

Concatenated DataFrame (Columns):
    A   B   C   D
0  A0  B0  C0  D0
1  A1  B1  C1  D1
```

在这里，`pd.concat` 使用索引（`0` 和 `1`）对齐行，并将 `df3` 的列放置在 `df1` 的列旁边。

下面是说明沿 `axis=0`（行）和 `axis=1`（列）连接之间区别的图表。

G

cluster₀

Axis=0 (行 - 默认)

cluster₁

Axis=1 (列)

a0

DF1

c0

DF1

DF2

b0

DF2

op0

pd.concat([DF1, DF2])
(axis=0)

op0->c0

结果

a1

DF1

c1

DF1

DF3

b1

DF3

op1

pd.concat([DF1, DF3], axis=1)

op1->c1

结果

> 说明垂直（axis=0）和水平（axis=1）连接的图表。

### 处理对齐 (alignment)和缺失数据（连接逻辑）

当连接在*非*连接轴上不能完美对齐的 DataFrames 时会发生什么？例如，当 DataFrames 具有不同行索引时并排连接（`axis=1`），或当它们具有不同列时垂直堆叠（`axis=0`）。

这由 `join` 参数 (parameter)控制，该参数的工作原理与数据库连接类似：

- `join='outer'` (默认): 取索引（或列）的并集。它保留两个 DataFrame 中的所有标签。如果一个标签存在于一个 DataFrame 中但不存在于另一个中，则缺失值将用 `NaN`（非数字）填充。
- `join='inner'`: 取索引（或列）的交集。它只保留在*两个* DataFrame 中都存在的标签。

让我们通过列连接（`axis=1`）来查看这种情况，其中 DataFrames 具有不同的索引：

```python

print("DataFrame 1:")
print(df1)

df4 = pd.DataFrame({'C': ['C1', 'C2'],
                    'D': ['D1', 'D2']},
                   index=[1, 2])
print("\nDataFrame 4:")
print(df4)

result_outer = pd.concat([df1, df4], axis=1, join='outer')
print("\n连接后的列 (外连接):")
print(result_outer)

result_inner = pd.concat([df1, df4], axis=1, join='inner')
print("\n连接后的列 (内连接):")
print(result_inner)
```

**Output:**

```
DataFrame 1:
    A   B
0  A0  B0
1  A1  B1

DataFrame 4:
    C   D
1  C1  D1
2  C2  D2

Concatenated Columns (Outer Join):
      A    B    C    D
0   A0   B0  NaN  NaN
1   A1   B1   C1   D1
2  NaN  NaN   C2   D2

Concatenated Columns (Inner Join):
    A   B   C   D
1  A1  B1  C1  D1
```

- **外连接：** 结果 DataFrame 的索引为 `0`、`1` 和 `2`（`[0, 1]` 和 `[1, 2]` 的并集）。由于索引 `0` 只存在于 `df1` 中，因此 'C' 和 'D' 列在该行中为 `NaN`。类似地，由于索引 `2` 只存在于 `df4` 中，因此 'A' 和 'B' 列在该行中为 `NaN`。索引 `1` 在两者中都存在，因此所有列都有值。
- **内连接：** 结果 DataFrame 只包含索引 `1`，因为它是 `df1` 和 `df4` 中都存在的唯一索引标签。

同样的 `join` 逻辑也适用于使用具有不同列名的 DataFrames 进行行连接（`axis=0`）的情况。外连接将包含两个 DataFrame 中的所有列，并用 `NaN` 填充缺失值，而内连接将只保留两者共有的列。

使用 `pd.concat` 进行连接是将较小的数据集组合成较大数据集的基本工具。理解如何控制轴、处理索引以及通过连接管理对齐对于正确组合您的数据非常重要。

获取即时帮助、个性化解释和交互式代码示例。

---

### 数据库风格的合并 (pd.merge)

# 数据库风格的合并 (pd.merge)

根据一个或多个共同列或索引中的*值*来组合 DataFrames 是一个常见需求，类似于在SQL等关系型数据库中连接表。虽然使用 `pd.concat` 进行连接对于堆叠结构相似的数据集很有用，但这些基于值的组合操作需要不同的方法。Pandas 提供了 `pd.merge()` 函数来完成这些操作。

`pd.merge()` 允许你根据指定列（通常称为“键”）中的共享值来组合两个 DataFrame 的行。想象一下，这就像根据在另一个表中找到的标识符在一张表中查找信息。

### 核心思想：基于键进行合并

假设我们有两个简单的数据框：一个包含员工信息，另一个包含部门信息。

```python
import pandas as pd

employees = pd.DataFrame({
    'employee_id': [101, 102, 103, 104],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'dept_id': [10, 20, 10, 30]
})

departments = pd.DataFrame({
    'dept_id': [10, 20, 40],
    'dept_name': ['Engineering', 'Marketing', 'Sales']
})

print("员工数据框：")
print(employees)
print("\n部门数据框：")
print(departments)
```

**Output:**

```
Employees DataFrame:
   employee_id     name  dept_id
0          101    Alice       10
1          102      Bob       20
2          103  Charlie       10
3          104    David       30

Departments DataFrame:
   dept_id    dept_name
0       10  Engineering
1       20    Marketing
2       40        Sales
```

两个数据框都共享一个 `dept_id` 列。我们可以使用此列作为“键”来合并这两个数据框，将 `dept_name` 添加到员工信息中。

最简单的合并方法是使用 `on` 参数 (parameter)指定要合并的数据框和要合并的列。

```python

merged_df = pd.merge(employees, departments, on='dept_id')

print("\n合并后的数据框 (默认 - 内连接)：")
print(merged_df)
```

**Output:**

```
Merged DataFrame (Default - Inner Join):
   employee_id     name  dept_id    dept_name
0          101    Alice       10  Engineering
1          103  Charlie       10  Engineering
2          102      Bob       20    Marketing
```

请注意以下几点：

1. `dept_name` 现在根据 `dept_id` 与相应的员工关联起来。
2. **David** (员工ID 104，部门ID 30) 从结果中消失了。为什么？因为 `dept_id` 30 在 `departments` 数据框中不存在。
3. **销售部** (部门ID 40) 也消失了。为什么？因为 `employees` 数据框中没有员工的 `dept_id` 为 40。

默认情况下，`pd.merge()` 执行**内连接**。这表示它只保留那些（在本例中是 `dept_id`）同时存在于*两个*数据框中的行。我们很快会介绍其他连接类型。

### 指定不同的列名

如果两个数据框中的列名不同怎么办？例如，假设 `employees` 数据框中的部门键名为 `department_code` 而不是 `dept_id`。

```python

employees_alt = pd.DataFrame({
    'employee_id': [101, 102, 103, 104],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'department_code': [10, 20, 10, 30]
})

print("员工数据框 (替代版本)：")
print(employees_alt)
print("\n部门数据框：")
print(departments)
```

**Output:**

```
Employees DataFrame (Alternative):
   employee_id     name  department_code
0          101    Alice               10
1          102      Bob               20
2          103  Charlie               10
3          104    David               30

Departments DataFrame:
   dept_id    dept_name
0       10  Engineering
1       20    Marketing
2       40        Sales
```

在这种情况下，你不能直接使用 `on` 参数 (parameter)。相反，你需要使用 `left_on` 和 `right_on` 分别指定左侧 (`employees_alt`) 和右侧 (`departments`) 数据框的列名。

```python

merged_alt_keys = pd.merge(employees_alt, departments,
                           left_on='department_code',
                           right_on='dept_id')

print("\n合并后的数据框 (列名不同)：")
print(merged_alt_keys)
```

**Output:**

```
Merged DataFrame (Different Names):
   employee_id     name  department_code  dept_id    dept_name
0          101    Alice               10       10  Engineering
1          103  Charlie               10       10  Engineering
2          102      Bob               20       20    Marketing
```

结果与之前相同，但请注意，合并后的数据框中包含了 *两个* 列 (`department_code` 和 `dept_id`)。如果其中一个冗余，你可能希望在合并后将其删除。

### 基于多个键进行合并

你还可以基于多个列进行合并。只需向 `on` 参数 (parameter)提供一个列名列表（或者在使用 `left_on` 和 `right_on` 时提供列名列表）。合并操作将只组合那些所有指定列在数据框之间都匹配的行。

```python

df_left = pd.DataFrame({
    'key1': ['A', 'B', 'B', 'C'],
    'key2': [1, 2, 1, 2],
    'left_val': [10, 20, 30, 40]
})

df_right = pd.DataFrame({
    'key1': ['B', 'C', 'C', 'D'],
    'key2': [1, 2, 3, 1],
    'right_val': [100, 200, 300, 400]
})

print("左侧数据框：")
print(df_left)
print("\n右侧数据框：")
print(df_right)

merged_multi = pd.merge(df_left, df_right, on=['key1', 'key2'])

print("\n基于多个键合并后的数据框 (内连接)：")
print(merged_multi)
```

**Output:**

```
Left DataFrame:
  key1  key2  left_val
0    A     1        10
1    B     2        20
2    B     1        30
3    C     2        40

Right DataFrame:
  key1  key2  right_val
0    B     1        100
1    C     2        200
2    C     3        300
3    D     1        400

Merged on Multiple Keys (Inner Join):
  key1  key2  left_val  right_val
0    B     1        30        100
1    C     2        40        200
```

只有那些 `df_left` 和 `df_right` 中 `key1` 和 `key2` *都*匹配的行被保留。

`pd.merge()` 函数是根据共享信息整合来自不同源的数据的基本工具。了解如何指定键是第一步。接下来，我们将介绍你可以执行的不同*类型*的合并。

获取即时帮助、个性化解释和交互式代码示例。

---

### 理解合并类型（连接）

# 理解合并类型（连接）

当您使用 `pd.merge()` 组合 `DataFrame` 对象时，行匹配和最终结果中包含行的方式在很大程度上取决于您执行的合并或连接种类。这种行为由 `pd.merge()` 函数中的 `how` 参数 (parameter)控制。可以将这些合并类型视为决定保留哪些行（根据输入 `DataFrame` 对象中连接键是否匹配）的不同规定。

让我们创建两个简单的 `DataFrame` 对象来准确说明这些不同的连接类型。假设我们有一个包含员工信息的 `DataFrame`，另一个包含他们的项目分配。

```python
import pandas as pd

employees = pd.DataFrame({
    'EmpID': [101, 102, 103, 104],
    'Name': ['Alice', 'Bob', 'Charlie', 'David']
})

projects = pd.DataFrame({
    'EmpID': [103, 104, 105, 106],
    'Project': ['Zeus', 'Apollo', 'Athena', 'Poseidon']
})

print("员工 DataFrame:")
print(employees)
print("\n项目 DataFrame:")
print(projects)
```

运行上述代码，结果如下：

```
Employees DataFrame:
   EmpID     Name
0    101    Alice
1    102      Bob
2    103  Charlie
3    104    David

Projects DataFrame:
   EmpID   Project
0    103      Zeus
1    104    Apollo
2    105    Athena
3    106  Poseidon
```

请注意，`EmpID` 103 和 104 在两个 `DataFrame` 对象中都存在。`EmpID` 101 和 102 只在 `employees` 中，而 `EmpID` 105 和 106 只在 `projects` 中。`EmpID` 列是我们的共同列，常被称为*连接键*，如果未另行指定，`pd.merge()` 将默认使用它。

### 内连接 (`how='inner'`)

内连接是 Pandas 中默认的合并类型。它组合两个 `DataFrame` 对象，只保留那些连接键（我们这里的 `EmpID`）在*左右*两个 `DataFrame` 对象中都存在的行。只在一个 `DataFrame` 对象中存在的键所对应的行将被舍弃。它本质上查找的是键的交集。

```python

inner_join_df = pd.merge(employees, projects, on='EmpID', how='inner')

print("内连接结果:")
print(inner_join_df)
```

输出结果只显示了同时存在于两个表中的员工：

```
Inner Join Result:
   EmpID     Name Project
0    103  Charlie    Zeus
1    104    David  Apollo
```

G

内连接 (交集)
只保留匹配的键 (103, 104)

cluster₀

员工

cluster₁

项目

Emp101

101
Alice

Emp102

102
Bob

Emp103

103
Charlie

Proj103

103
Zeus

Emp104

104
David

Proj104

104
Apollo

Proj105

105
Athena

Proj106

106
Poseidon

> 内连接只保留在员工和项目两个数据集中都存在的共同 `EmpID` 值（103、104）。

### 外连接 (`how='outer'`)

外连接，有时也称为全外连接，保留*两个* `DataFrame` 对象中的*所有*行。如果一个 `DataFrame` 中的某行在另一个 `DataFrame` 中没有匹配的键，另一个 `DataFrame` 中的列将填充 `NaN`（非数字）值。这种连接类型查找的是键的并集。

```python

outer_join_df = pd.merge(employees, projects, on='EmpID', how='outer')

print("外连接结果:")
print(outer_join_df)
```

结果包含了所有员工和所有项目：

```
Outer Join Result:
   EmpID     Name   Project
0  101.0    Alice       NaN
1  102.0      Bob       NaN
2  103.0  Charlie      Zeus
3  104.0    David    Apollo
4  105.0      NaN    Athena
5  106.0      NaN  Poseidon
```

请注意 Alice (101) 和 Bob (102) 的 `Project` 列中显示 `NaN`，因为它们不在 `projects` `DataFrame` 中。同样，分配给 `EmpID` 105 和 106 的项目的 `Name` 列也显示 `NaN`。

G

外连接 (并集)
保留所有键 (101-106)，非匹配项填充NaN

cluster₀

员工

cluster₁

项目

Emp101

101
Alice

Emp102

102
Bob

Emp103

103
Charlie

Proj103

103
Zeus

Emp104

104
David

Proj104

104
Apollo

Proj105

105
Athena

Proj106

106
Poseidon

> 外连接保留两个数据集中所有 `EmpID` 值，当在另一个数据集中找不到相应匹配项时，使用 `NaN` 填充。

### 左连接 (`how='left'`)

左连接保留*左侧* `DataFrame`（即传递给 `pd.merge()` 的第一个 `DataFrame`，我们示例中的 `employees`）中的*所有*行，并包含*右侧* `DataFrame` (`projects`) 中的匹配行。如果左侧 `DataFrame` 中的键在右侧 `DataFrame` 中不存在，右侧 `DataFrame` 的列将填充 `NaN`。只在右侧 `DataFrame` 中存在的键将被舍弃。

```python

left_join_df = pd.merge(employees, projects, on='EmpID', how='left')

print("左连接结果:")
print(left_join_df)
```

这保留了所有员工信息，并在有可用项目信息时将其添加：

```
Left Join Result:
   EmpID     Name Project
0    101    Alice     NaN
1    102      Bob     NaN
2    103  Charlie    Zeus
3    104    David  Apollo
```

来自 `employees` 的所有 `EmpID`（101、102、103、104）都存在。由于 101 和 102 在 `projects` 中没有匹配项，它们的 `Project` 值显示为 `NaN`。来自 `projects` 的 `EmpID` 105 和 106 被排除在外，因为它们不在左侧的 `DataFrame` (`employees`) 中。

G

左连接
保留左侧所有键 (101-104)，
添加匹配的右侧数据，非匹配项填充NaN

cluster₀

员工 (左)

cluster₁

项目 (右)

Emp101

101
Alice

Emp102

102
Bob

Emp103

103
Charlie

Proj103

103
Zeus

Emp104

104
David

Proj104

104
Apollo

Proj105

105
Athena

Proj106

106
Poseidon

> 左连接保留左侧数据集（员工）中的所有 `EmpID` 值，并包含来自右侧数据集（项目）的匹配数据。右侧的非匹配项填充 `NaN`。

### 右连接 (`how='right'`)

右连接是左连接的镜像。它保留*右侧* `DataFrame` (`projects`) 中的*所有*行，并包含来自*左侧* `DataFrame` (`employees`) 的匹配行。如果右侧 `DataFrame` 中的键在左侧 `DataFrame` 中不存在，左侧 `DataFrame` 的列将填充 `NaN`。只在左侧 `DataFrame` 中存在的键将被舍弃。

```python

right_join_df = pd.merge(employees, projects, on='EmpID', how='right')

print("右连接结果:")
print(right_join_df)
```

这保留了所有项目信息，并在有可用员工信息时将其添加：

```
Right Join Result:
   EmpID     Name   Project
0  103.0  Charlie      Zeus
1  104.0    David    Apollo
2  105.0      NaN    Athena
3  106.0      NaN  Poseidon
```

来自 `projects` 的所有 `EmpID`（103、104、105、106）都存在。由于 105 和 106 在 `employees` 中没有匹配项，它们的 `Name` 值显示为 `NaN`。来自 `employees` 的 `EmpID` 101 和 102 被排除在外，因为它们不在右侧的 `DataFrame` (`projects`) 中。

G

右连接
保留右侧所有键 (103-106)，
添加匹配的左侧数据，非匹配项填充NaN

cluster₀

员工 (左)

cluster₁

项目 (右)

Emp101

101
Alice

Emp102

102
Bob

Emp103

103
Charlie

Proj103

103
Zeus

Emp104

104
David

Proj104

104
Apollo

Proj105

105
Athena

Proj106

106
Poseidon

> 右连接保留右侧数据集（项目）中的所有 `EmpID` 值，并包含来自左侧数据集（员工）的匹配数据。左侧的非匹配项填充 `NaN`。

选择正确的合并类型是数据准备过程中的一个重要环节。这完全取决于你在最终组合数据集中所需的信息。你是否只关心同时存在于两个来源中的条目（内连接）？你是否需要一个完整的视图，包括只存在于一个来源中的条目（外连接）？或者你主要想用一个特定数据集的信息来补充另一个数据集（左连接或右连接）？掌握这些连接类型能让你精确地控制数据如何组合。

获取即时帮助、个性化解释和交互式代码示例。

---

### 基于索引的合并

# 基于索引的合并

当组合DataFrame时，连接信息有时存在于其索引中，而非公共列值。Pandas提供了灵活的方法来处理这些场景，包括`pd.merge`函数和专门的`.join`方法。

假设您有两个数据集：一个包含员工详细信息，以员工ID作为索引；另一个包含绩效评估分数，也以相同的员工ID作为索引。要将它们结合起来，您会希望根据它们的索引进行对齐 (alignment)。

### 使用带索引标志的`pd.merge`

`pd.merge`函数通过`left_index`和`right_index`布尔参数 (parameter)支持基于索引的合并。

- `left_index=True`：使用左侧DataFrame的索引作为其连接键。
- `right_index=True`：使用右侧DataFrame的索引作为其连接键。

我们创建两个简单DataFrame来演示：

```python
import pandas as pd

employees = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'],
                          'department': ['HR', 'Engineering', 'Sales']},
                         index=['E101', 'E102', 'E103'])

salaries = pd.DataFrame({'salary': [70000, 85000, 78000]},
                        index=['E101', 'E102', 'E103'])

print("员工信息 DataFrame:")
print(employees)
print("\n薪资信息 DataFrame:")
print(salaries)
```

输出：

```
Employees DataFrame:
           name department
E101    Alice         HR
E102      Bob Engineering
E103  Charlie       Sales

Salaries DataFrame:
      salary
E101   70000
E102   85000
E103   78000
```

要根据它们的共同索引（员工ID）合并这些数据，我们将`left_index`和`right_index`都设为`True`：

```python

employee_data = pd.merge(employees, salaries, left_index=True, right_index=True)

print("\n合并后的DataFrame（使用索引）：")
print(employee_data)
```

输出：

```
Merged DataFrame (using index):
           name department  salary
E101    Alice         HR   70000
E102      Bob Engineering   85000
E103  Charlie       Sales   78000
```

合并操作根据共享的索引值（'E101'、'E102'、'E103'）正确地对齐 (alignment)了行。

### 混合索引和列合并

您也可以混合使用索引和列键。假设薪资信息使用的是名为`emp_id`的列而不是索引：

```python

salaries_col = pd.DataFrame({'emp_id': ['E101', 'E102', 'E104'],
                             'salary': [70000, 85000, 92000]})

print("\n薪资信息 DataFrame（带列）：")
print(salaries_col)

employee_data_mixed = pd.merge(employees, salaries_col,
                               left_index=True, right_on='emp_id',
                               how='left')

print("\n合并后的DataFrame（混合索引/列，左连接）：")
print(employee_data_mixed)
```

输出：

```
Salaries DataFrame (with column):
  emp_id  salary
0   E101   70000
1   E102   85000
2   E104   92000

Merged DataFrame (mixed index/column, left join):
           name department emp_id   salary
E101    Alice         HR   E101  70000.0
E102      Bob Engineering   E102  85000.0
E103  Charlie       Sales    NaN      NaN
```

这里，`left_index=True`告诉`merge`使用`employees` DataFrame的索引，而`right_on='emp_id'`指定`salaries_col`中的`emp_id`列应作为右侧DataFrame的键。我们使用了`how='left'`连接，以保留左侧DataFrame（`employees`）中的所有员工；请注意，'Charlie' (E103) 的`emp_id`和`salary`值为`NaN`，因为`salaries_col`的`emp_id`列中不存在'E103'。同样，`salaries_col`中的'E104'被丢弃了，因为它在左连接过程中未匹配到`employees`中的任何索引。

### 基于索引合并的`.join()`方法

Pandas DataFrames有一个便捷方法`.join()`，专门用于基于索引的合并。它默认执行左连接，但可以通过`how`参数 (parameter)配置为其他连接类型。

对于索引合并，其语法通常更简洁：`left_df.join(right_df, how='...')`

我们用`.join()`重新看第一个例子：

```python

employee_data_join = employees.join(salaries)

print("\n合并后的DataFrame（使用.join()）：")
print(employee_data_join)
```

输出：

```
Merged DataFrame (using .join()):
           name department  salary
E101    Alice         HR   70000
E102      Bob Engineering   85000
E103  Charlie       Sales   78000
```

这产生了与`pd.merge`调用`left_index=True`和`right_index=True`相同的结果（在第一个`merge`示例中隐式使用了默认的内连接，或者显式指定`how='left'`，由于索引完全匹配，在此处效果相同）。

`.join()`方法还可以将一个DataFrame的索引与另一个DataFrame中的列连接起来，如果您将列名传递给`on`参数：

```python

salaries_col_indexed = salaries_col.set_index('emp_id')

print("\n薪资信息 DataFrame（以emp_id为索引）：")
print(salaries_col_indexed)

employee_data_join_indexed = employees.join(salaries_col_indexed, how='inner')

print("\n合并后的DataFrame（使用.join()在索引上，内连接）：")
print(employee_data_join_indexed)

employees_with_id = employees.reset_index().rename(columns={'index': 'emp_id'})
print("\n带有emp_id列的员工信息：")
print(employees_with_id)

print("\n带有emp_id列的薪资信息（原始）：")
print(salaries_col)

employee_data_join_on = employees_with_id.join(salaries_col.set_index('emp_id'), on='emp_id', how='left')

print("\n合并后的DataFrame（使用.join()带'on'参数）：")
print(employee_data_join_on)
```

输出：

```
Salaries DataFrame (indexed by emp_id):
        salary
emp_id
E101     70000
E102     85000
E104     92000

Merged DataFrame (using .join() on indices, inner):
           name department  salary
E101    Alice         HR   70000
E102      Bob Engineering   85000

Employees with emp_id column:
  emp_id     name department
0   E101    Alice         HR
1   E102      Bob Engineering
2   E103  Charlie       Sales

Salaries with emp_id column (original):
  emp_id  salary
0   E101   70000
1   E102   85000
2   E104   92000

Merged DataFrame (using .join() with 'on'):
  emp_id     name department   salary
0   E101    Alice         HR  70000.0
1   E102      Bob Engineering  85000.0
2   E103  Charlie       Sales      NaN
```

**何时使用 `pd.merge` 与 `.join`：**

- 当您需要最大的灵活性时，请使用`pd.merge()`：合并列与列、索引与索引，或者混合列和索引。它是通用的合并函数。
- 将`.join()`用作便捷的简写方式，主要用于基于索引的合并（左侧DataFrame的索引与右侧DataFrame的索引）。它默认执行左连接，这很常见。虽然它*可以*使用`on`关键字进行列连接，但对于基于列或混合的合并，`pd.merge`通常更清晰。

掌握基于索引的合并很重要，以便高效地组合唯一标识符存储为索引标签的数据集，这在时间序列数据和其他结构化数据集中是一种常见模式。

获取即时帮助、个性化解释和交互式代码示例。

---

### 基于索引的合并 (.join)

# 基于索引的合并 (.join)

Pandas提供了 `pd.merge` 方法，可用于组合DataFrame，它基于列或索引，提供了强大的、类似SQL的灵活性。此外，Pandas还提供了 `.join()` 方法，专门为根据索引标签组合DataFrame，或将索引与列合并而优化。此方法是执行特定类型合并的高效方式，尤其是索引上的左连接，这十分普遍。

### 基本的基于索引的合并

默认情况下，`DataFrame.join()` 尝试使用其索引与另一个DataFrame（或多个DataFrame）合并。除非另行指定，它会执行左连接，这意味着它保留了调用DataFrame（“左侧”的那个）的所有行，并根据索引包含“右侧”DataFrame的匹配数据。

我们来设置两个简单的DataFrame以作说明：

```python
import pandas as pd

left_df = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                        'B': ['B0', 'B1', 'B2']},
                       index=['K0', 'K1', 'K2'])

right_df = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                         'D': ['D0', 'D2', 'D3']},
                        index=['K0', 'K2', 'K3'])

print("左侧DataFrame:")
print(left_df)
print("\n右侧DataFrame:")
print(right_df)

joined_df = left_df.join(right_df)

print("\n合并后的DataFrame（索引上的左连接）:")
print(joined_df)
```

**Output:**

```
Left DataFrame:
     A   B
K0  A0  B0
K1  A1  B1
K2  A2  B2

Right DataFrame:
     C   D
K0  C0  D0
K2  C2  D2
K3  C3  D3

Joined DataFrame (left join on index):
     A   B    C    D
K0  A0  B0   C0   D0
K1  A1  B1  NaN  NaN
K2  A2  B2   C2   D2
```

请注意，结果 `joined_df` 保留了 `left_df` 的所有索引标签（`K0`、`K1`、`K2`）。对于 `K0` 和 `K2`，它们存在于两个DataFrame的索引中，`right_df` 的相应值（`C0`、`D0`、`C2`、`D2`）被包含进来。对于仅在 `left_df` 中的 `K1`，`right_df` 的列（`C` 和 `D`）被 `NaN`（非数字）填充，表示缺失数据。来自 `right_df` 的索引 `K3` 未被包含，因为它不在 `left_df` 中，并且我们执行了左连接。

### 列上的索引合并

`.join()` 方法不限于索引对索引的合并。您可以使用 `on` 参数 (parameter)将调用DataFrame（左侧）的索引与传入DataFrame（右侧）中的一列或多列进行合并。

```python

right_df_on_col = pd.DataFrame({'key': ['K0', 'K2', 'K3'],
                                'C': ['C0', 'C2', 'C3'],
                                'D': ['D0', 'D2', 'D3']})

print("左侧DataFrame:")
print(left_df)
print("\n右侧DataFrame:")
print(right_df_on_col)

joined_on_col = left_df.join(right_df_on_col.set_index('key'), on=None)

joined_on_col_idiom = left_df.join(right_df_on_col.set_index('key'))

print("\n合并后的DataFrame（左侧索引与右侧列合并）:")
print(joined_on_col_idiom)
```

**Output:**

```
Left DataFrame:
     A   B
K0  A0  B0
K1  A1  B1
K2  A2  B2

Right DataFrame (with column):
  key   C   D
0  K0  C0  D0
1  K2  C2  D2
2  K3  C3  D3

Joined DataFrame (left index on right column):
     A   B    C    D
K0  A0  B0   C0   D0
K1  A1  B1  NaN  NaN
K2  A2  B2   C2   D2
```

在此示例中，我们首先在 `right_df_on_col` 上使用了 `set_index('key')`，将 `key` 列设为其索引，从而允许与 `left_df` 进行标准的索引对索引合并。这是使用 `.join()` 时一个很常见的模式。结果与第一个示例相同，因为逻辑变得一致：基于匹配的索引值（`left_df` 中的 `K0`、`K1`、`K2` 与 `right_df_on_col` 的新索引）进行合并。

虽然 `left_df.join(other, on='col_in_other')` 是*可能*的，但它通常不如使用 `pd.merge(left_df, other, left_index=True, right_on='col_in_other')` 或上面所示的 `set_index` 方法易读。`.join()` 的主要优点在于其在基于索引的操作方面的简洁性。

### 使用 `how` 指定合并类型

正如 `pd.merge` 一样，`.join()` 方法接受 `how` 参数 (parameter)来控制连接类型。选项有：

- `'left'`: (默认) 保留左侧DataFrame的所有键。
- `'right'`: 保留右侧DataFrame的所有键。
- `'outer'`: 保留两个DataFrame的所有键。
- `'inner'`: 只保留在*两个*DataFrame中都找到的键。

我们来看看这些如何影响使用原始 `left_df` 和 `right_df` 的结果：

```python

inner_join = left_df.join(right_df, how='inner')
print("\n内连接（两个DataFrame中都有的键）:")
print(inner_join)

outer_join = left_df.join(right_df, how='outer')
print("\n外连接（任一DataFrame中有的键）:")
print(outer_join)

right_join = left_df.join(right_df, how='right')
print("\n右连接（右侧DataFrame中有的键）:")
print(right_join)
```

**Output:**

```
Inner Join (keys in both):
     A   B   C   D
K0  A0  B0  C0  D0
K2  A2  B2  C2  D2

Outer Join (keys in either):
      A    B    C    D
K0   A0   B0   C0   D0
K1   A1   B1  NaN  NaN
K2   A2   B2   C2   D2
K3  NaN  NaN   C3   D3

Right Join (keys in right):
      A    B   C   D
K0   A0   B0  C0  D0
K2   A2   B2  C2  D2
K3  NaN  NaN  C3  D3
```

请看差异：

- `inner`：只有 `K0` 和 `K2` 出现，因为它们是 `left_df` 和 `right_df` 中都存在的唯一索引标签。
- `outer`：两个DataFrame中的所有唯一索引标签（`K0`、`K1`、`K2`、`K3`）都存在。原始DataFrame中缺少数据的地方用 `NaN` 填充。
- `right`：`right_df` 的所有索引标签（`K0`、`K2`、`K3`）都被保留。`left_df` 的数据在索引匹配时被包含；否则，使用 `NaN`（如 `K3` 所示）。

这是说明左连接的图示：

G

A

索引

A

B

K0

A0

B0

K1

A1

B1

K2

A2

B2

C

索引

A

B

C

D

K0

A0

B0

C0

D0

K1

A1

B1

NaN

NaN

K2

A2

B2

C2

D2

A:s->C:n

 左侧.合并(右侧)

B

索引

C

D

K0

C0

D0

K2

C2

D2

K3

C3

D3

B:s->C:n

> 表示一个 `left_df.join(right_df)` 操作。左侧表的所有索引键都被保留。右侧表的匹配键会带入其数据；不匹配的键则导致右侧表的列出现 NaN 值。

### 处理重叠的列名

如果您要合并的DataFrame拥有同名列（除了索引或正在合并的列之外），`.join()` 会引发错误。为解决此问题，您可以使用 `lsuffix` 和 `rsuffix` 参数 (parameter)，分别在左侧和右侧DataFrame的重叠列名后添加区分后缀。

```python

right_overlap = pd.DataFrame({'B': ['B_r0', 'B_r2', 'B_r3'],
                              'D': ['D0', 'D2', 'D3']},
                             index=['K0', 'K2', 'K3'])

print("左侧DataFrame:")
print(left_df)
print("\n带重叠的右侧DataFrame:")
print(right_overlap)

joined_overlap_fixed = left_df.join(right_overlap, lsuffix='_left', rsuffix='_right')

print("\n带后缀的合并DataFrame:")
print(joined_overlap_fixed)
```

**Output:**

```
Left DataFrame:
     A   B
K0  A0  B0
K1  A1  B1
K2  A2  B2

Right DataFrame with Overlap:
     B    D
K0  B_r0   D0
K2  B_r2   D2
K3  B_r3   D3

Joined DataFrame with Suffixes:
     A B_left B_right    D
K0  A0     B0    B_r0   D0
K1  A1     B1     NaN  NaN
K2  A2     B2    B_r2   D2
```

正如您所见，`left_df` 中原始的 `B` 列现在变为 `B_left`，而 `right_overlap` 中的 `B` 列在最终结果中现在变为 `B_right`。

### 合并多个DataFrame

您可以通过向 `.join()` 传递一个DataFrame列表，一次性合并两个以上的DataFrame。合并操作根据索引从左到右依次进行。

```python

other_df = pd.DataFrame({'E': ['E1', 'E2', 'E4']}, index=['K1', 'K2', 'K4'])

print("左侧DF:")
print(left_df)
print("\n右侧DF:")
print(right_df)
print("\n其他DF:")
print(other_df)

multi_join = left_df.join([right_df, other_df])
print("\n多DataFrame合并结果（左连接）:")
print(multi_join)

multi_join_outer = left_df.join([right_df, other_df], how='outer')
print("\n多DataFrame合并结果（外连接）:")
print(multi_join_outer)
```

**Output:**

```
Left DF:
     A   B
K0  A0  B0
K1  A1  B1
K2  A2  B2

Right DF:
     C   D
K0  C0  D0
K2  C2  D2
K3  C3  D3

Other DF:
     E
K1  E1
K2  E2
K4  E4

Multi-Join Result (left joins):
     A   B    C    D    E
K0  A0  B0   C0   D0  NaN
K1  A1  B1  NaN  NaN   E1
K2  A2  B2   C2   D2   E2

Multi-Join Result (outer joins):
      A    B    C    D    E
K0   A0   B0   C0   D0  NaN
K1   A1   B1  NaN  NaN   E1
K2   A2   B2   C2   D2   E2
K3  NaN  NaN   C3   D3  NaN
K4  NaN  NaN  NaN  NaN   E4
```

默认的左连接只保留初始DataFrame（`left_df`）的索引。外连接组合所有涉及的DataFrame的所有索引。

总而言之，`DataFrame.join()` 提供了一种简洁的语法，主要用于基于索引的合并。虽然 `pd.merge` 提供了更通用的合并能力，但当您的组合逻辑主要依赖于DataFrame索引时，`.join()` 通常更快、更易读，特别是对于执行左连接的常见情况。请记住，`.join()` 内部使用 `pd.merge`，所以底层逻辑是一致的。

获取即时帮助、个性化解释和交互式代码示例。

---

### 实践操作：组合数据集

# 实践操作：组合数据集

提供使用 `pd.concat`、`pd.merge` 和 `.join()` 方法的实际操作示例。我们将使用小型、清晰的数据集来说明每种技术的工作方式。

首先，确保已导入 Pandas。我们将使用约定俗成的别名 `pd`。

```python
import pandas as pd
import numpy as np
```

### 准备示例 DataFrame

为了演示连接和合并，我们来创建几个简单的 DataFrame。

```python

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2'],
                    'C': ['C0', 'C1', 'C2']},
                   index=[0, 1, 2])

df2 = pd.DataFrame({'A': ['A3', 'A4', 'A5'],
                    'B': ['B3', 'B4', 'B5'],
                    'C': ['C3', 'C4', 'C5']},
                   index=[3, 4, 5])

df3 = pd.DataFrame({'D': ['D0', 'D1', 'D2'],
                    'E': ['E0', 'E1', 'E2'],
                    'F': ['F0', 'F1', 'F2']},
                   index=[0, 1, 2])

left_df = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                        'L_val': ['L0', 'L1', 'L2', 'L3']})

right_df = pd.DataFrame({'key': ['K0', 'K1', 'K4', 'K5'],
                         'R_val': ['R0', 'R1', 'R4', 'R5']})

left_join_df = pd.DataFrame({'L_val': ['L0', 'L1', 'L2']},
                             index=pd.Index(['K0', 'K1', 'K2'], name='key'))

right_join_df = pd.DataFrame({'R_val': ['R0', 'R1', 'R4']},
                              index=pd.Index(['K0', 'K1', 'K4'], name='key'))
```

我们来看看最初的 DataFrame：

```python
print("--- df1 ---")
print(df1)
print("\n--- df2 ---")
print(df2)
print("\n--- df3 ---")
print(df3)
print("\n--- left_df ---")
print(left_df)
print("\n--- right_df ---")
print(right_df)
print("\n--- left_join_df ---")
print(left_join_df)
print("\n--- right_join_df ---")
print(right_join_df)
```

### 实践：使用 `pd.concat` 进行连接

连接类似于将 DataFrame 堆叠或拼接在一起。

#### 垂直连接（堆叠行）

这是 `pd.concat` 的默认行为。它垂直堆叠 DataFrame，按名称对齐 (alignment)列。

```python

vertical_concat = pd.concat([df1, df2])

print("--- 垂直连接（默认）---")
print(vertical_concat)
```

请注意 `df2` 是如何附加在 `df1` 下方的。原始 DataFrame 的索引被保留了。如果索引的唯一性很重要，或者你更喜欢一个整洁的从 0 开始的索引，请使用 `ignore_index=True`。

```python

vertical_concat_new_index = pd.concat([df1, df2], ignore_index=True)

print("\n--- 垂直连接 (ignore_index=True) ---")
print(vertical_concat_new_index)
```

#### 水平连接（添加列）

要根据 DataFrame 的索引将它们并排放置，请使用 `axis=1`。

```python

horizontal_concat = pd.concat([df1, df3], axis=1)

print("--- 水平连接 (axis=1) ---")
print(horizontal_concat)
```

这里，`df1` 和 `df3` 是根据匹配的索引标签（0, 1, 2）组合的。如果索引没有完美对齐，Pandas 将为缺失的匹配项引入 `NaN` 值（这是一种沿着索引的外连接行为）。

### 实践：使用 `pd.merge` 进行合并

合并根据公共列中的值组合 DataFrame，类似于 SQL 连接。

#### 内连接（默认）

内连接只保留键在*两个* DataFrame 中都存在的行。

```python

inner_merge = pd.merge(left_df, right_df, on='key', how='inner')

print("--- 内连接 ---")
print(inner_merge)
```

只有同时存在于 `left_df` 和 `right_df` 中的键 'K0' 和 'K1' 出现在结果中。

#### 外连接

外连接保留*两个* DataFrame 中的所有行。如果一个键在其中一个 DataFrame 中不存在，则会为源自该 DataFrame 的列引入 `NaN` 值。

```python

outer_merge = pd.merge(left_df, right_df, on='key', how='outer')

print("\n--- 外连接 ---")
print(outer_merge)
```

键 'K0'、'K1'、'K2'、'K3'、'K4' 和 'K5' 都存在。请注意，当键在其中一个原始 DataFrame 中缺失时（例如，'K2' 和 'K3' 的 `R_val` 为 `NaN`，'K4' 和 'K5' 的 `L_val` 为 `NaN`），会出现 `NaN` 值。

#### 左连接

左连接保留*左侧* DataFrame 中的所有行，并包含来自右侧 DataFrame 的匹配行。如果左侧 DataFrame 的键在右侧不存在，则右侧 DataFrame 的列将使用 `NaN`。

```python

left_merge = pd.merge(left_df, right_df, on='key', how='left')

print("\n--- 左连接 ---")
print(left_merge)
```

`left_df` 中的所有键（'K0'、'K1'、'K2'、'K3'）都存在。'K2' 和 'K3' 的 `R_val` 为 `NaN`，因为它们不在 `right_df` 中。仅存在于 `right_df` 中的键（'K4'、'K5'）被排除。

#### 右连接

右连接与左连接相反。它保留*右侧* DataFrame 中的所有行，并包含来自左侧的匹配行。

```python

right_merge = pd.merge(left_df, right_df, on='key', how='right')

print("\n--- 右连接 ---")
print(right_merge)
```

`right_df` 中的所有键（'K0'、'K1'、'K4'、'K5'）都存在。'K4' 和 'K5' 的 `L_val` 为 `NaN`。仅存在于 `left_df` 中的键（'K2'、'K3'）被排除。

为了帮助可视化这些连接，可以将键视为集合：

G

cluster₀

内连接
(双方都有的键)

cluster₁

左连接
(左侧所有键)

cluster₂

右连接
(右侧所有键)

cluster₃

外连接
(双方所有键)

a

K0

b

K1

c

K2

d

K3

e

K4

f

K5

g

K0

h

K1

i

K2

j

K3

k

K4

l

K5

m

K0

n

K1

o

K4

p

K5

q

K2

r

K3

s

K0

t

K1

u

K2

v

K3

w

K4

x

K5

> 基于键 'K0'-'K5' 的不同连接类型的图示。蓝色实心圆表示每种连接类型的结果中包含的键，假设 `left_df` 的键是 K0-K3，`right_df` 的键是 K0, K1, K4, K5。

#### 基于不同列名进行合并

如果 DataFrame 中的列具有不同的名称，请使用 `left_on` 和 `right_on`。

```python

right_df_renamed = right_df.rename(columns={'key': 'common_key'})

print("\n--- right_df_renamed ---")
print(right_df_renamed)

merge_diff_names = pd.merge(left_df, right_df_renamed,
                            left_on='key', right_on='common_key',
                            how='inner')

print("\n--- 使用不同名称合并 (left_on, right_on) ---")
print(merge_diff_names)
```

结果中包含两个原始列（`left_df` 中的 `key` 和 `right_df_renamed` 中的 `common_key`）。之后你可能需要删除其中一个。

### 实践：使用 `.join` 进行基于索引的连接

`.join()` 方法是一种基于 DataFrame 索引执行合并的便捷方式。它默认为左连接。

```python

left_index_join = left_join_df.join(right_join_df, how='left')

print("--- 左索引连接（默认 .join()）---")
print(left_index_join)
```

这会保留 `left_join_df` 中的所有索引值（'K0'、'K1'、'K2'），并添加来自 `right_join_df` 的相应 `R_val`。'K2' 的 `R_val` 为 `NaN`，因为它不在 `right_join_df` 的索引中。

你可以使用 `how` 参数 (parameter)指定其他连接类型，就像使用 `pd.merge` 一样。

```python

outer_index_join = left_join_df.join(right_join_df, how='outer')

print("\n--- 外索引连接 (.join(how='outer')) ---")
print(outer_index_join)
```

这包含两个 DataFrame 的所有索引值（'K0'、'K1'、'K2'、'K4'），并用 `NaN` 填充缺失值。

你还可以使用 `.join()` 将 DataFrame 的索引与另一个 DataFrame 中的列进行连接，方法是将连接列*临时*设置为另一个 DataFrame 的索引，或者使用 `.join()` 中的 `on` 参数。

```python

column_index_join = left_df.join(right_join_df, on='key', how='inner')

print("\n--- 将列连接到索引 (right_join_df) ---")
print(column_index_join)
```

本次实践涵盖了在 Pandas 中组合 DataFrame 的主要方法。连接是堆叠数据，而合并和连接是根据共享键（在列或索引中）使用不同的逻辑规则（内、外、左、右）组合数据。请在你自己的数据集上试验这些技术，以巩固你的理解。

获取即时帮助、个性化解释和交互式代码示例。

---
