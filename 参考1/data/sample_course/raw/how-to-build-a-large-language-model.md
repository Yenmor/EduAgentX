# 如何构建大语言模型

## Chapter 1 Introduction Large Scale Language Modeling

### 定义大型语言模型

# 定义大型语言模型

其核心是一个语言模型，它是一个统计工具，用来预测词序列的概率，更确切地说，是预测标记（token）序列的概率（标记可以是词、子词 (subword)或字符，我们将在第五章讨论）。给定一个前面的标记序列，模型会尝试预测最可能的下一个标记。这种基本能力可以数学方式表示为计算概率：

P(tokeni∣token1,token2,...,tokeni−1)P(token\_i | token\_1, token\_2, ..., token\_{i-1})P(tokeni​∣token1​,token2​,...,tokeni−1​)

这种预测能力使语言模型能够生成文本、完成句子、翻译语言以及执行各种其他自然语言处理（NLP）任务。传统的语言模型，包括n-gram和早期的神经网络 (neural network)方法，如循环神经网络（RNN），通常操作的参数 (parameter)量从数千到数亿不等。

那么，是什么让一个语言模型“大型”呢？“大型语言模型”（LLM）这个术语特指基于神经网络的语言模型，其特点是规模非常庞大，无论是在所含参数数量还是训练数据量方面。

### 参数 (parameter)：大小的衡量标准

主要区别在于可学习参数的庞大数量。虽然早期模型如BERT-Large有大约3.4亿个参数，但大型语言模型将这一界限大幅拓宽，通常包含数十亿、数百亿、数千亿甚至数万亿个参数。这些参数（神经网络 (neural network)中的权重 (weight)和偏置 (bias)）对从训练数据中学习到的模式、语法、知识和细节进行编码。这种能力的非常大增长是一个决定性特点。

BERT-LargeGPT-2GPT-3PaLM5125102510025

> 几个知名语言模型的近似参数量，说明了规模差异（注意Y轴是对数刻度）。

### 训练数据：推动规模

相应地，大型语言模型在庞大数据集上进行预训练 (pre-training)，这些数据集通常包含数百TB甚至PB级别的文本数据，抓取自网络、书籍、代码库及其他来源（第6-9章将介绍数据来源和处理）。这与在经过筛选、规模较小、通常以GB计的数据集上训练的较小型模型形成对比。如此规模的数据是有效训练庞大参数 (parameter)量、并让模型接触到广泛语言使用和知识的必要条件。

### 基础架构

早期模型尝试了多种架构，而现代大型语言模型几乎都基于Transformer架构，该架构由Vaswani等人于2017年在论文《Attention Is All You Need》中提出。Transformer的自注意力 (self-attention)机制 (attention mechanism)使模型在进行预测时，能够权衡输入序列中不同标记 (token)的重要性，克服了之前序列架构（如RNN）在处理长距离依赖方面的局限。我们将在第四章详细讨论Transformer。

### 涌现 (emergence)能力：规模效应的结果

也许大型语言模型最令人着迷的一面是，它们涌现出并非明确编程或训练而来的能力，而是作为规模效应的结果而出现。较小型模型通常需要大量针对特定任务的微调 (fine-tuning)，才能在情感分析或问答等下游任务中表现良好。然而，大型语言模型却常表现出卓越的零样本或少样本学习 (few-shot learning)能力。

- **零样本学习 (zero-shot learning)：** 模型能够执行在提示中描述的任务，而无需在训练期间见过该任务的任何具体例子。
- **少样本学习：** 模型能够在提示本身中给出少量（例如1到32个）例子后执行任务（上下文 (context)学习）。

涌现能力的例子包括算术推理 (inference)、复杂指令遵循、训练数据中未明确配对的语言之间的翻译以及代码生成。这些能力在模型规模超过一定阈值时会显得有些突然地出现，这表明数量（规模）可以带来行为上的质变。

考虑一个简化的交互：

```python

context = "The capital of Indonesia is"
next_token_probabilities = llm.predict_next_token_probabilities(context)

print(f"最可能的预测: {llm.get_most_likely_token(next_token_probabilities)}")

context_few_shot = """
Translate English to Indonesian:
sea otter => berang-berang laut
cheese => keju
plush toy => boneka

Translate English to Indonesian:
cloud => ?
"""
next_token_probabilities_few_shot = llm.predict_next_token_probabilities(
    context_few_shot
)

print(f"少样本预测: {llm.get_most_likely_token(
    next_token_probabilities_few_shot
)}")
```

这种少样本能力得益于模型的规模和广泛的预训练 (pre-training)，它使大型语言模型与之前的模型区别开来，之前的模型通常需要在一个专门的英法翻译数据集上进行微调才能获得类似结果。

### 计算需求

最后，定义大型语言模型必然涉及到对其训练所需庞大计算资源的承认。训练过程通常涉及数千个高端GPU或TPU，运行数周或数月，消耗大量能源并产生高昂成本。这种计算规模是与较小型模型的另一个实际区别。

综上所述，大型语言模型由以下几点定义：

1. **庞大规模：** 数十亿或数万亿参数 (parameter)。
2. **广泛训练数据：** 在数TB或数PB的多样化文本上进行预训练 (pre-training)。
3. **Transformer架构：** 通常基于Transformer模型。
4. **涌现 (emergence)能力：** 表现出零样本和少样本学习 (few-shot learning)能力，这些能力并非其明确的训练目标。
5. **高昂计算成本：** 训练需要大规模分布式计算资源。

理解这些特点，为探讨构建、训练和部署这些强大模型所涉及的工程挑战和技术奠定了基础，这也是本课程的重点。

获取即时帮助、个性化解释和交互式代码示例。

---

### 序列建模的历史背景

# 序列建模的历史背景

要理解当今大型语言模型的结构和功能，需要了解为处理序列数据（尤其是文本）而发展起来的技术。从简单的统计方法到复杂的神经网络 (neural network)的演进，表明了捕获语言中语境和依赖关系的持续难题。

### 早期统计模型：N-gram

在深度学习 (deep learning)普及之前，统计方法构成了语言建模的起点。其中最基本的是**N-gram模型**。这些模型基于马尔可夫假设运行：下一个词的概率仅依赖于前 n−1n-1n−1 个词。例如，一个三元模型（n=3n=3n=3）会估算给定前两个词 wi−1w\_{i-1}wi−1​ 和 wi−2w\_{i-2}wi−2​ 时词 wiw\_iwi​ 的概率：

P(wi∣w1,...,wi−1)≈P(wi∣wi−2,wi−1)P(w\_i | w\_1, ..., w\_{i-1}) \approx P(w\_i | w\_{i-2}, w\_{i-1})P(wi​∣w1​,...,wi−1​)≈P(wi​∣wi−2​,wi−1​)

这些概率通常通过计算大型文本语料库中的出现次数来估算：

P(wi∣wi−2,wi−1)=计数(wi−2,wi−1,wi)计数(wi−2,wi−1)P(w\_i | w\_{i-2}, w\_{i-1}) = \frac{\text{计数}(w\_{i-2}, w\_{i-1}, w\_i)}{\text{计数}(w\_{i-2}, w\_{i-1})}P(wi​∣wi−2​,wi−1​)=计数(wi−2​,wi−1​)计数(wi−2​,wi−1​,wi​)​

尽管简单且易于理解，N-gram模型面临明显不足：

1. **稀疏性：** 许多有效的词序列可能从未在训练语料库中出现，导致概率为零。人们开发了平滑技术（例如拉普拉斯平滑、Kneser-Ney平滑）来缓解此问题，但它们不能完全解决问题，特别是当 nnn 较大时。
2. **语境窗口：** N-gram只能考虑一个固定的短语境（n−1n-1n−1个词）。它们从根本上无法捕获长距离依赖关系，这在人类语言中大量存在（例如，跨子句的主谓一致）。增加 nnn 会指数级地加剧稀疏性问题。
3. **缺乏泛化能力：** N-gram将词视为离散单元。它们不能轻易地基于语义相似性进行泛化（例如，知道“cat”与“dog”相似）。

Ngram

...

...

wᵢ-2

wᵢ-2

...->wᵢ-2

 依赖于前面的 n-1 个词

wᵢ-1

wᵢ-1

wᵢ-2->wᵢ-1

 依赖于前面的 n-1 个词

wᵢ

wᵢ

wᵢ-1->wᵢ

 依赖于前面的 n-1 个词

> N-gram模型根据固定窗口内的前置词来预测下一个词。

### 循环神经网络 (neural network)（RNN）的出现

神经网络提供了一种方法来处理N-gram的不足之处。\*\*循环神经网络（RNN）\*\*是专门为序列数据设计的。与前馈网络不同，RNN具有循环连接，使其能够维持一个内部*隐藏状态*（hth\_tht​），该状态理论上可以包含序列中所有先前时间步的信息。

在每个时间步 ttt，RNN接收当前输入 xtx\_txt​ 和前一个隐藏状态 ht−1h\_{t-1}ht−1​，以计算新的隐藏状态 hth\_tht​ 和可能的输出 yty\_tyt​：

ht=tanh⁡(Whhht−1+Wxhxt+bh)h\_t = \tanh(W\_{hh}h\_{t-1} + W\_{xh}x\_t + b\_h)ht​=tanh(Whh​ht−1​+Wxh​xt​+bh​)
yt=Whyht+byy\_t = W\_{hy}h\_t + b\_yyt​=Why​ht​+by​

这里，WhhW\_{hh}Whh​、WxhW\_{xh}Wxh​、WhyW\_{hy}Why​、bhb\_hbh​ 和 byb\_yby​ 是学习得到的参数 (parameter)（权重 (weight)矩阵和偏置 (bias)），它们在所有时间步共享。tanh⁡\tanhtanh 函数是一种常用的激活函数 (activation function)。

RNN

clusterₜ

时间 t

hₜ-1

hₜ₋₁

RNN Cell

RNN
单元

hₜ-1->RNN Cell

xₜ

xₜ

xₜ->RNN Cell

hₜ

hₜ

RNN Cell->hₜ

yₜ

yₜ

RNN Cell->yₜ

...

...

hₜ->...

...->hₜ-1

> RNN单元处理输入 xtx\_txt​ 和前一状态 ht−1h\_{t-1}ht−1​，以生成下一个状态 hth\_tht​ 和输出 yty\_tyt​。状态随着时间传递。

尽管RNN有望建模任意长距离依赖关系，但由于**梯度消失问题**，训练它们变得困难。在时间反向传播 (backpropagation)过程中，梯度在多个时间步相乘可能指数级缩小，阻止与早期时间步相关的权重得到充分更新。这意味着，在实践中，简单RNN难以学习相对短窗口内的依赖关系，效果与N-gram类似。相关的**梯度爆炸问题**（即梯度指数级增长）也可能发生，尽管通常通过梯度裁剪等方式更容易控制。

### 门控RNN：LSTM和GRU

为了解决梯度消失问题，人们设计了更复杂的循环单元。

\*\*长短期记忆（LSTM）\*\*网络引入了门控机制：

1. **遗忘门：** 决定从细胞状态中丢弃哪些信息。
2. **输入门：** 决定哪些新信息要存储到细胞状态中。
3. **输出门：** 决定根据过滤后的细胞状态输出什么。

这些门通过一个独立的*细胞状态*（CtC\_tCt​）控制信息随时间流动，使网络能够选择性地长时间记住或遗忘信息。

```python
import torch
import torch.nn as nn

input_size = 10

hidden_size = 20

num_layers = 2

lstm_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

batch_size = 3
seq_len = 5
example_input = torch.randn(batch_size, seq_len, input_size)

h0 = torch.randn(num_layers, batch_size, hidden_size)
c0 = torch.randn(num_layers, batch_size, hidden_size)

output, (hn, cn) = lstm_layer(example_input, (h0, c0))

print("Output shape:", output.shape)
print("Final hidden state shape:", hn.shape)

print("Final cell state shape:", cn.shape)
```

\*\*门控循环单元（GRU）\*\*提供了一种稍微简化的替代方案，它有两个门（更新门和重置门），并且没有独立的细胞状态。GRU在许多任务上通常能达到与LSTM相近的性能，但参数 (parameter)更少。

与简单RNN相比，LSTM和GRU都明显提升了捕获长距离依赖关系的能力。它们成为许多自然语言处理任务的标准，常用于\*\*序列到序列（Seq2Seq）\*\*架构中。Seq2Seq模型由一个*编码器*RNN和一个*解码器*RNN组成，编码器RNN将输入序列处理成一个上下文 (context)向量 (vector)（通常是最终隐藏状态），解码器RNN则根据该上下文向量生成输出序列。尽管在机器翻译、摘要等任务中取得了成功，Seq2Seq模型仍然面临瓶颈：无论输入序列的长度如何，其全部含义都必须被压缩到一个固定大小的上下文向量中。

### 注意力机制 (attention mechanism)

**注意力机制**被引入以缓解Seq2Seq的瓶颈问题。解码器不再仅仅依赖最终的编码器隐藏状态，而是在生成输出的每一步中被允许“关注”*整个*输入序列的不同部分。它计算当前解码器状态与所有编码器隐藏状态之间的注意力得分，生成一个针对该解码步骤的加权上下文 (context)向量 (vector)。这使得模型在生成相应输出词时，能够专注于相关的输入词，明显提升了性能，特别是对于更长的序列。

Attention

clusterₑncoder

编码器 (RNN/LSTM)

cluster\_decoder

解码器步骤 t

encₕ1

h1

encₕ2

h2

encₕ3

h3

decₛt

sₜ

contextᵥec

上下文
向量 cₜ

decₛt->contextᵥec

计算注意力权重 u03b1

contextᵥec->encₕ1

加权和

contextᵥec->encₕ2

加权和

contextᵥec->encₕ3

加权和

> 注意力机制允许解码器状态 sts\_tst​ 选择性地对编码器隐藏状态（h1,h2,h3h\_1, h\_2, h\_3h1​,h2​,h3​）进行加权，以形成上下文向量 ctc\_tct​。

### Transformer：摆脱循环

尽管注意力机制 (attention mechanism)提升了基于RNN的模型，但循环的序列性仍然是训练效率的瓶颈。处理一个序列需要 O(序列长度)O(\text{序列长度})O(序列长度) 的顺序操作，阻碍了并行化。

在论文“Attention Is All You Need”（Vaswani 等，2017）中提出的**Transformer**架构，通过完全放弃循环机制，彻底改变了序列建模方式。它仅依赖于注意力机制，特别是**自注意力 (self-attention)**。自注意力允许模型在为特定词编码表示时，衡量输入序列中所有其他词的重要性。

Transformer层内的计算（自注意力和前馈网络）可以在序列位置上大部分并行执行。这种并行能力是一项重大进展，使得在比以往使用RNN更大的数据集上训练更大规模的模型成为可能。通过并行化获得的计算效率，加上自注意力在处理复杂依赖关系（包括短距离和长距离）方面的有效性，为构建本课程主要关注的超大型语言模型创造了条件。Transformer架构所带来的扩展特性和计算需求是我们将在后续讨论中多次提及的核心内容。

获取即时帮助、个性化解释和交互式代码示例。

---

### 规模的重要性

# 规模的重要性

谈及大型语言模型时，“大”不仅仅是一个定性描述；它定量指代着庞大的参数 (parameter)数量、海量的训练数据集以及所需的大量计算资源。这种规模并非偶然特性，而是其能力的基本推动力。与早期模型性能提升可能相对较快达到平台期不同，现代大型语言模型展现出与规模增长直接相关的不同现象。

### 缩放定律：可预测的改进

大型语言模型研究中一个重要的发现是*缩放定律*的存在。这些是经验观察结果，表明模型性能（通常通过在独立数据集上的交叉熵损失来衡量）会随着模型大小（参数 (parameter)数量）、数据集大小和训练所用计算量的增加而可预测地提升。

大型语言模型中的性能与规模之间的关系通常被建模为幂律，尤其在对数-对数坐标图上查看时。例如，损失LLL可以近似地与非嵌入 (embedding)参数的数量NNN、数据集大小DDD和计算预算CCC（以FLOPs计）关联如下：

L(N)≈(NcN)αNL(N) \approx \left( \frac{N\_c}{N} \right)^{\alpha\_N}L(N)≈(NNc​​)αN​
L(D)≈(DcD)αDL(D) \approx \left( \frac{D\_c}{D} \right)^{\alpha\_D}L(D)≈(DDc​​)αD​

这里，NcN\_cNc​ 和 DcD\_cDc​ 代表特征尺度，而 αN\alpha\_NαN​ 和 αD\alpha\_DαD​ 是缩放指数（通常是小于1的正值，对于NNN和DDD通常在0.05-0.1左右）。类似的关联也适用于计算量 CCC。这些定律表明，投入更多资源（参数、数据、计算）会在主要训练目标上带来边际递减但持续的改进。

100M10B1T100T10​16​10​18​10​20​10​22​2.833.23.43.63.844.24.44.6模型大小数据集大小计算量

> 验证损失倾向于随着模型大小、数据集大小或计算预算的增加而呈幂律下降。

这些缩放定律对于规划训练过程非常有帮助。它们使研究人员和工程师能够估算在给定预算下可达到的性能提升，反之，也能估算达到目标性能水平所需的资源，从而避免投入昂贵且耗时长的实验。

### 涌现 (emergence)能力

规模最吸引人的方面之一，或许是*涌现能力*的出现。这些能力在较小的模型中不存在或无法衡量，但一旦模型大小、数据量或计算量超过特定阈值，它们就会相对突然地显现。它们不仅仅是现有指标的渐进式改进，而是性质上全新的行为。

实例包括：

- **少样本推理 (inference)：** 能够仅根据提示中提供的少量示例执行新任务，而无需任何梯度更新。较小的模型通常需要大量微调 (fine-tuning)才能执行新任务。
- **思维链推理：** 提示大型模型“一步步思考”可以大幅提升其在需要算术、符号推理或常识逻辑的任务上的表现。这种能力在较小的模型中通常不存在。
- **指令遵循：** 大型模型在遵循自然语言中给出的复杂指令方面变得更好。

这些能力出现的阈值是经验性的且依赖于具体任务，但它们的存在有力地推动了对更大模型的追求。这表明，仅仅扩大现有架构的规模就能产生根本性的新功能。

G

clusterₛmall

较小规模

clusterₘedium

中等规模

clusterₗarge

大规模

Small

基础语言建模
（例如，预测下一个词）

Medium

提升的流畅性
基础任务性能
（例如，微调后的情感分析）

Small->Medium

 参数和数据
 增加

Large

涌现能力
（例如，少样本学习，
 思维链推理，
 复杂指令遵循）

Medium->Large

 参数和数据
 进一步增加

> 说明了规模的增加如何带来更复杂的能力，包括涌现的能力。

### 参数 (parameter)、数据和计算的关系

规模化不仅仅是使某一个方面变大；它涉及平衡三个主要组成部分：模型大小(NNN)、数据集大小(DDD)和训练计算量(CCC)。研究，特别是DeepMind的“Chinchilla”论文（Hoffmann et al.，2022），指出在固定的计算预算下，仅通过最大化模型大小并不能获得最佳性能。相反，存在一个最佳分配方案，即模型大小和数据集大小应大致按比例进行缩放。

以前的模型通常使用相对较小的数据集进行训练，与它们的参数数量相比（计算受限状态）。Chinchilla的研究结果表明，许多大型模型存在显著的训练不足；对于已使用的计算量，通过在更多数据上训练一个较小的模型，本可以提升性能。这表明，数据规模与模型规模同样重要，对于在给定计算范围内达到最佳结果而言。

计算参数数量提供了衡量模型规模的具体方法。对于一个典型的Transformer块，参数主要来自自注意力 (self-attention)投影（Query、Key、Value、Output）和前馈网络层。

```python
import torch
import torch.nn as nn
from math import prod

def count_parameters(model: nn.Module) -> int:
    """计算PyTorch模型中可训练参数的总数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

hidden_dim = 768
ffn_dim = hidden_dim * 4
num_heads = 12
head_dim = hidden_dim // num_heads

qkv_params = 3 * hidden_dim * hidden_dim

attn_output_params = hidden_dim * hidden_dim

ffn1_params = hidden_dim * ffn_dim

ffn2_params = ffn_dim * hidden_dim

approx_params_per_layer = (qkv_params + attn_output_params +\
                           ffn1_params + ffn2_params)

print(f"每个Transformer层的近似参数数量：{approx_params_per_layer:,}")

num_layers = 12

vocab_size = 30522
embedding_params = vocab_size * hidden_dim

total_params_estimate = ((num_layers * approx_params_per_layer) +\
                         embedding_params)
print(f"一个12层模型的总估算参数数量：{total_params_estimate:,}")

large_hidden_dim = 1280
large_ffn_dim = large_hidden_dim * 4
large_num_heads = 16

large_qkv = 3 * large_hidden_dim * large_hidden_dim
large_attn_out = large_hidden_dim * large_hidden_dim
large_ffn1 = large_hidden_dim * large_ffn_dim
large_ffn2 = large_ffn_dim * large_hidden_dim
large_layer_params = (large_qkv + large_attn_out +\
                      large_ffn1 + large_ffn2)
print(f"每个层的大致参数数量 "\
      f"（大型模型）：{large_layer_params:,}")

large_num_layers = 24
large_embedding_params = vocab_size * large_hidden_dim
large_total_params = ((large_num_layers * large_layer_params) +\
                      large_embedding_params)
print(f"一个24层大型模型的总估算参数数量：{large_total_params:,}")
```

此代码片段展示了架构选择（如 `hidden_dim`、`ffn_dim`、`num_layers`）如何直接影响参数数量，而参数数量是衡量规模的主要指标。本课程中讨论的模型通常范围从数亿到数千亿甚至数万亿参数，需要相应增加数据和计算量。

总而言之，规模并非仅仅为了大而大。它根据缩放定律可预测地推动性能提升，使定性上新的涌现 (emergence)能力成为可能，并且需要在模型参数、数据集大小和计算预算之间取得仔细的平衡。理解规模的重要性对于应对构建和训练高效大型语言模型所涉及的工程难题非常重要。

获取即时帮助、个性化解释和交互式代码示例。

---

### 计算方面的挑战概览

# 计算方面的挑战概览

使用数十亿甚至数万亿参数 (parameter)的模型和以TB为单位的数据集进行构建，会带来艰巨的计算难题。这些模型的规模与其能力直接相关，但这种规模在计算资源方面会带来高昂的成本。了解这些难题对理解LLM开发所需的工程工作非常重要。主要的限制集中在两个基本资源上：内存和计算能力。

### 内存限制：模型及其状态的存储

训练大型神经网络 (neural network)需要将多个组件存储在硬件加速器（通常是GPU或TPU）的内存中。对于LLM，这些组件的庞大大小常常超出单个设备的内存容量。我们来分析一下主要的内存占用者：

1. **模型参数 (parameter)：** 这些是模型在训练过程中学习到的权重 (weight)和偏置 (bias)（WWW）。对于大型模型，这通常是最显而易见的内存需求。如果一个模型有NNN个参数并使用32位浮点精度（FP32），它仅存储权重就需要N×4N \times 4N×4字节。例如，一个千亿参数的模型仅参数就需要大约100×109×4=400100 \times 10^9 \times 4 = 400100×109×4=400 GB。即使使用16位精度（FP16或BF16），这仍然达到200 GB，远超典型单个加速器（可能从16GB到80GB）的内存。
2. **梯度：** 在反向传播 (backpropagation)过程中，我们计算损失函数 (loss function)相对于每个参数的梯度（∇W\nabla W∇W）。这些梯度通常具有与参数本身相同的维度和数据类型。因此，存储梯度需要额外的N×4N \times 4N×4字节（对于FP32）或N×2N \times 2N×2字节（对于FP16/BF16）。
3. **优化器状态：** 现代优化器如Adam或AdamW为每个参数维护状态信息，以调整学习率。例如，Adam为每个参数存储一阶矩（动量）和二阶矩（方差）的估算值。如果这些矩以FP32存储，它们各自需要额外的N×4N \times 4N×4字节。这意味着优化器状态可以轻易地使仅参数和梯度所需的内存翻倍或增至三倍（参数1×N1 \times N1×N，梯度1×N1 \times N1×N，Adam状态2×N2 \times N2×N = FP32中与参数相关的总内存为4×N4 \times N4×N）。
4. **激活值：** 也许是内存占用量最大的，尤其是在批处理大小和序列长度较大的情况下，是前向传播过程中产生的中间激活值。为一层计算的每个激活值通常都需要存储，直到在反向传播过程中用于计算梯度。激活值（AAA）所需的内存大致与批处理大小（bbb）、序列长度（sss）、隐藏维度（hhh）和层数（LLL）成比例。特别是对于Transformer，注意力机制 (attention mechanism)的中间结果（例如随序列长度呈平方关系增长的注意力分数矩阵，O(s2)O(s^2)O(s2)）可能特别占用内存。这种内存占用量可能显著超过参数所需的内存，尤其是在训练期间。通常采用激活值检查点（或梯度检查点）等技术，通过在反向传播时重新计算激活值而非全部存储，从而以计算换内存。

这些组件的组合决定了每个加速器所需的总内存。

G

gpuₘemory

GPU内存

参数 (FP16/32)

梯度 (FP16/32)

优化器状态 (FP32)

激活值 (FP16/32)

框架开销

parameters

模型参数
(数十亿)

parameters->gpuₘemory:w

gradients

梯度

gradients->gpuₘemory:w

optimizer

优化器状态
(例如 Adam)

optimizer->gpuₘemory:w

activations

激活值
(批处理大小, 序列长度)

activations->gpuₘemory:w

> LLM训练期间单个加速器内部典型内存占用者的细分说明。激活值和优化器状态所需的内存量常常远超模型参数本身。

由于所需的总内存常常超出单个设备所能提供的，因此将模型及其相关状态分散到多个加速器上的分布式训练策略变得很有必要。

### 计算需求：浮点运算预算

训练LLM是计算密集型任务，需要庞大的浮点运算（FLOPs）。

- **训练所需的浮点运算：** 大部分计算发生在Transformer的前馈网络和自注意力 (self-attention)机制 (attention mechanism)中的矩阵乘法。经验研究，常被称为缩放定律（例如，Kaplan et al., 2020; Hoffmann et al., 2022），提供了计算成本的估算。一个常见的近似值是，在包含DDD个token的数据集上训练一个NNN参数 (parameter)的模型，大约需要C≈6×N×DC \approx 6 \times N \times DC≈6×N×D FLOPs。对于在数万亿个token上训练的千亿参数模型，这会导致计算预算以ZettaFLOPs (102110^{21}1021 FLOPs) 范围或更高来衡量。完成此类计算需要大型加速器集群运行数周或数月。
- **推理 (inference)所需的浮点运算：** 尽管生成单个token比训练快得多，但自回归 (autoregressive)生成（一个接一个地生成token）仍然需要可观的计算量。对于一个NNN参数的模型，生成一个token大约需要2×N2 \times N2×N FLOPs（忽略位置编码 (positional encoding)、层归一化 (normalization)等）。逐个token地生成长序列可能会对延迟敏感。此外，自注意力机制对于长度为sss且隐藏维度为hhh的序列，每层具有O(s2⋅h)O(s^2 \cdot h)O(s2⋅h)的计算复杂度。虽然像键值（KV）缓存这样的技术可以减少推理时过去token的冗余计算，但初始提示的处理仍然涉及相对于提示长度的二次方成本。

### 分布式环境中的通信开销

随着内存和计算需求使得训练必须跨多个加速器（通常是数百或数千个）进行，这些设备之间的通信成为一个决定性因素。

- **数据并行：** 需要跨设备同步梯度或模型参数 (parameter)，通常使用AllReduce操作。
- **模型并行（张量/流水线）：** 需要在负责模型不同部分或计算阶段的设备之间传输激活值和梯度。

互连（例如GPU之间的NVLink或节点之间的InfiniBand/以太网）的效率显著影响整体训练吞吐量 (throughput)。缓慢的通信可能导致加速器在等待数据时空闲，造成限制扩展的瓶颈。

这些计算难题、内存容量限制、庞大的浮点运算需求以及通信开销，促使对专用硬件、软件框架（如支持分布式的PyTorch、DeepSpeed、Megatron-LM）和并行化策略的需求，我们将在本课程中详细研究这些内容。克服这些障碍是构建大型语言模型工程实践的核心部分。

获取即时帮助、个性化解释和交互式代码示例。

---

### 软件与硬件环境

# 软件与硬件环境

构建大型语言模型不只是算法方面的难题；它是一项工程工作，高度依赖特定软件工具与强大硬件基础设施的结合。前面提到的计算和内存需求需要超越单机配置和标准库。接下来我们看看常见的系统组成部分。

### 软件堆栈

LLM开发所使用的软件构成一个分层的堆栈，从基础的深度学习 (deep learning)框架到处理大规模需求的专用库。

#### 深度学习框架

大多数现代LLM开发的核心是灵活高效的深度学习框架。尽管存在多种选择，PyTorch在研发社群中尤其受到重视，因为它有Python风格的接口、动态计算图（便于调试和处理更复杂的控制流），以及丰富的支持库体系。TensorFlow是另一个广泛使用的框架，尤其在生产环境中。

这些框架提供主要构建块：用于梯度计算的自动微分、为加速器优化的张量运算，以及用于构建神经网络 (neural network)层的模块。

```python

import torch

x = torch.randn(128, 768)
w = torch.randn(768, 3072)

output = torch.matmul(x, w)

print(f"Input shape: {x.shape}")
print(f"Weight shape: {w.shape}")
print(f"Output shape: {output.shape}")

import torch.nn as nn

linear_layer = nn.Linear(in_features=768, out_features=3072)
print(f"\nLinear layer: {linear_layer}")
```

#### 分布式训练库

训练具有数十亿或数万亿参数 (parameter)的模型需要将计算和数据分配到多个硬件加速器上。像PyTorch这样的主要框架提供基础的分布式通信原语（`torch.distributed`），但专用库简化了复杂并行策略的实现：

- **DeepSpeed：** 由Microsoft开发，DeepSpeed提供主要侧重于减少内存消耗的优化，从而能训练更大的模型。其ZeRO（零冗余优化器）技术在数据并行工作器间划分优化器状态、梯度乃至模型参数。它还支持流水线并行和高效的混合精度训练。
- **Megatron-LM：** 由NVIDIA开发，Megatron-LM提供高度优化的张量并行（在多个GPU之间拆分单个层）和流水线并行（在多个GPU之间分阶段处理层）实现。它常用于训练最大的模型，在这些情况下，即使是ZeRO技术本身也不够用。
- **完全分片数据并行（FSDP）：** 直接集成到PyTorch中，FSDP提供与DeepSpeed ZeRO Stage 3类似的功能，在数据并行等级之间分片模型参数、梯度和优化器状态。它的目标是为大规模数据并行提供更原生的PyTorch体验。

这些库抽象化了管理数十或数百个设备之间的通信和同步的许多复杂性。

#### 数据处理与分词 (tokenization)

处理数TB的文本数据需要可扩展的工具。像Hugging Face `datasets`这样的库提供高效加载、处理和流式传输大型数据集的方法。对于真正大规模的预处理（清洗、过滤、去重），通常会使用运行在集群上的分布式计算框架，如Apache Spark或Dask。

分词，即将原始文本转换为数字ID的过程，由Hugging Face的`tokenizers`（提供BPE、WordPiece的快速实现）和Google的SentencePiece等库处理。这些工具旨在高效处理大型语料库，并与深度学习框架良好配合。

#### 实验追踪

训练LLM可能需要数天或数周，涉及大量超参数 (hyperparameter)和潜在不稳定的运行。像Weights & Biases、MLflow或TensorBoard这样的工具对于记录指标（损失、学习率、梯度范数）、追踪超参数、保存模型检查点以及可视化结果是必不可少的，有助于调试和复现。

### 硬件平台

软件堆栈运行在专为高性能计算设计的专用硬件之上。

#### 加速器：GPU与TPU

标准CPU不适合深度神经网络 (neural network)所需的大规模并行计算（主要指矩阵乘法）。硬件加速器必不可少：

- **GPU（图形处理单元）：** NVIDIA GPU在LLM训练中占据主导地位。像Ampere（例如A100）和Hopper（例如H100）这样的架构具有：
  - **张量核心：** 专用单元，执行混合精度矩阵乘累加运算，速度远超标准CUDA核心。
  - **高带宽内存（HBM）：** 极快的片上内存（数十或数百GB），对于存储模型参数 (parameter)、激活和梯度很重要。显存 (VRAM)容量通常是决定单个设备可训练的最大模型规模的主要瓶颈。
- **TPU（张量处理单元）：** 谷歌的定制ASIC专为神经网络工作负载设计。它们擅长矩阵运算，并与Google Cloud基础设施紧密结合。TPU通常以大型“pod”形式出现，配有高速互连，使它们非常适合超大规模训练。

Hardware

clusterₙode

计算节点

GPU1

GPU 0
(HBM, 张量核心)

GPU2

GPU 1
(HBM, 张量核心)

GPU1->GPU2

NVLink

CPU

宿主CPU(s)
系统内存

GPU1->CPU

PCIe

GPU3

GPU ...

GPU2->GPU3

NVLink

GPU4

GPU N
(HBM, 张量核心)

GPU3->GPU4

NVLink

GPU4->CPU

PCIe

> 一个典型的计算节点包含多个GPU，它们通过高速NVLink连接，实现节点内快速通信，同时还有宿主CPU和系统内存。

#### 高速互连

当训练涉及多个节点时（每个节点可能包含多个加速器），节点间的通信速度变得非常重要。慢速互连可能导致加速器空闲等待数据，从而成为整个过程的瓶颈。

- **NVLink：** NVIDIA专有的高速直连GPU间互连，提供比标准PCIe显著更高的带宽，对张量并行和单节点内高效数据共享很重要。
- **InfiniBand或高速以太网（例如200/400 Gbps）：** 用于连接多个计算节点的网络技术。低延迟和高带宽对于扩展数据并行（如通过AllReduce进行梯度同步）和跨节点流水线并行是必不可少的。

#### 存储系统

训练需要持续读取大规模数据集。需要快速、可扩展的存储方案，以不成为瓶颈的方式为加速器提供数据。这通常涉及并行文件系统（如Lustre或GPFS）或基于云的对象存储（如AWS S3、Google Cloud Storage），并结合高效的数据加载机制。

### 整体概览

LLM的系统环境包含这些组件的协作。数据被获取并存储，然后使用Spark等工具进行大规模预处理。分词 (tokenization)器 (tokenizer)为模型准备文本。训练期间，分布式库使用PyTorch等深度学习 (deep learning)框架，协调在多个加速器（GPU/TPU）上的执行。高速互连促进设备间必要的通信。实验追踪工具监控整个过程，检查点保存到可靠的存储中。这种配置使得能够开发和训练大型语言模型所需规模的模型。

Workflow

DataStore

数据存储
(S3, GCS, NFS)

Preprocessing

数据预处理
(Spark, Dask, HF Datasets)

DataStore->Preprocessing

原始数据

Frameworks

框架与库
(PyTorch, DeepSpeed,
Megatron-LM, Tokenizers)

Preprocessing->Frameworks

处理后的数据

TrainingCluster

训练集群
(GPU/TPU, 互连)

TrainingCluster->Frameworks

计算资源

Frameworks->TrainingCluster

训练任务

Tracking

实验追踪
(W&B, MLflow)

Frameworks->Tracking

记录指标

Checkpoints

模型检查点
(存储)

Frameworks->Checkpoints

保存状态

Checkpoints->Frameworks

加载状态

> LLM开发工作流程的简化概览，展现了存储、处理、训练硬件、软件库和监控工具之间的配合。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 2 Mathematical Preliminaries Llms

### 线性代数回顾：向量与矩阵

# 线性代数回顾：向量与矩阵

线性代数提供了处理大型语言模型中常见高维数据的数学语言。向量 (vector)和矩阵是我们用于表示神经网络 (neural network)中的输入数据、模型参数 (parameter)和中间激活的主要对象。把握它们的属性和运算方式，对于明白信息在这些模型中如何流动和转变来说，是十分重要的。

### 向量 (vector)：表示数据点

本质上，向量是一个有序的数字列表，通常表示多维空间中的一个点或方向。在大型语言模型的背景下，向量通常表示：

- **词嵌入 (embedding)：** 捕获语义的密集表示，其中每个维度对应一个潜在特征。例如，词汇表 (vocabulary)中的一个词可以映射到一个300维向量。
- **隐藏状态：** 网络层内的中间表示（如RNN隐藏状态或Transformer层输出），捕获截至某一特定点的输入序列的上下文 (context)信息。
- **梯度：** 表示损失函数 (loss function)相对于模型参数 (parameter)或激活值的最陡峭上升方向和大小的向量。

nnn维欧几里得空间中的向量vvv表示为：

v=[v1v2⋮vn]v = \begin{bmatrix} v\_1 \\ v\_2 \\ \vdots \\ v\_n \end{bmatrix}v=​v1​v2​⋮vn​​​

向量加法和标量乘法等基本运算使我们能够组合或缩放这些表示。例如，将输入嵌入添加到位置编码 (positional encoding)向量会组合语义和位置信息。

在PyTorch中，向量表示为一维张量。

```python
import torch

vector_a = torch.tensor([1.0, 2.5, -0.8, 4.0, 0.0])

scaled_vector = 2.0 * vector_a

vector_b = torch.tensor([-0.5, 1.0, 1.2, -2.0, 1.5])
summed_vector = vector_a + vector_b

print(f"原始向量 A: {vector_a}")
print(f"缩放后的向量: {scaled_vector}")
print(f"相加后的向量: {summed_vector}")
print(f"向量维度 (阶): {vector_a.ndim}")
print(f"向量形状: {vector_a.shape}")
```

### 矩阵：表示变换和参数 (parameter)

矩阵是数字的矩形排列，将向量 (vector)扩展到二维。它们在深度学习 (deep learning)中最重要的作用是表示向量空间之间的线性变换。

- **权重 (weight)矩阵：** 线性层（或全连接层）的参数存储在矩阵中。应用一个层涉及将输入向量（或输入向量的矩阵）乘以该层的权重矩阵。
- **注意力分数：** 矩阵可以表示序列中不同位置之间计算出的注意力权重。
- **数据批次：** 输入向量的集合（例如，序列中多个词的嵌入 (embedding)或批次中多个序列）通常排列成矩阵。

一个有mmm行nnn列的矩阵AAA（A∈Rm×nA \in \mathbb{R}^{m \times n}A∈Rm×n）是：

A=[A1,1A1,2⋯A1,nA2,1A2,2⋯A2,n⋮⋮⋱⋮Am,1Am,2⋯Am,n]A = \begin{bmatrix}
A\_{1,1} & A\_{1,2} & \cdots & A\_{1,n} \\
A\_{2,1} & A\_{2,2} & \cdots & A\_{2,n} \\
\vdots & \vdots & \ddots & \vdots \\
A\_{m,1} & A\_{m,2} & \cdots & A\_{m,n}
\end{bmatrix}A=​A1,1​A2,1​⋮Am,1​​A1,2​A2,2​⋮Am,2​​⋯⋯⋱⋯​A1,n​A2,n​⋮Am,n​​​

### 神经网络 (neural network)的核心运算

某些线性代数运算在神经网络计算中无处不在：

#### 矩阵-向量 (vector)乘法

这个运算将矩阵WWW定义的线性变换应用于向量xxx。如果W∈Rm×nW \in \mathbb{R}^{m \times n}W∈Rm×n且x∈Rnx \in \mathbb{R}^nx∈Rn，则结果y=Wxy = Wxy=Wx是一个在mathbbRm\\mathbb{R}^mmathbbRm中的向量。这是神经网络中一个密集层（无偏置 (bias)）内的基本计算，它将一个nnn维输入表示转换为一个mmm维输出表示。

yi=∑j=1nWi,jxjy\_i = \sum\_{j=1}^{n} W\_{i,j} x\_jyi​=j=1∑n​Wi,j​xj​

```python
import torch

W = torch.randn(3, 4)

x = torch.tensor([1.0, 0.5, -1.0, 2.0])

y = torch.matmul(W, x)

print(f"权重矩阵 W (形状 {W.shape}):\n{W}")
print(f"\n输入向量 x (形状 {x.shape}): {x}")
print(f"\n输出向量 y (形状 {y.shape}): {y}")
```

G

cluster₀

输入空间 (R⁴)

cluster₁

输出空间 (R³)

x

x
(4 维)

W

W (3x4)

x->W

 乘法

y

y
(3 维)

W->y

> 矩阵 W 将向量 x 从4维空间变换到3维空间。

#### 矩阵-矩阵乘法

两个矩阵 A∈Rm×nA \in \mathbb{R}^{m \times n}A∈Rm×n 和 B∈Rn×pB \in \mathbb{R}^{n \times p}B∈Rn×p 相乘得到矩阵 C=AB∈Rm×pC = AB \in \mathbb{R}^{m \times p}C=AB∈Rm×p。这在批量处理数据时被广泛使用，其中输入 XXX 可能是一个矩阵，每行是一个输入向量，或者在组合多个线性变换时使用。例如，Transformer的前馈网络中的计算通常涉及多次矩阵乘法。

Ci,k=∑j=1nAi,jBj,kC\_{i,k} = \sum\_{j=1}^{n} A\_{i,j} B\_{j,k}Ci,k​=j=1∑n​Ai,j​Bj,k​

```python
import torch

X = torch.randn(2, 4)

W = torch.randn(3, 4)

Y = torch.matmul(X, W.T)

print(f"输入批次 X (形状 {X.shape}):\n{X}")
print(f"\n权重矩阵 W 转置 (形状 {W.T.shape}):\n{W.T}")
print(f"\n输出批次 Y (形状 {Y.shape}):\n{Y}")
```

#### 逐元素运算（哈达玛积）

这涉及到将两个相同形状的矩阵（或向量）的对应元素相乘。表示为 A⊙BA \odot BA⊙B，结果 CCC 的 Ci,j=Ai,j×Bi,jC\_{i,j} = A\_{i,j} \times B\_{i,j}Ci,j​=Ai,j​×Bi,j​。这与矩阵乘法不同，并出现在各种神经网络组成部分中，例如逐元素应用激活函数 (activation function)或在LSTM或GRU中实现门控机制。

```python
import torch

A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[0.5, 1.], [-1., 2.]])

C = A * B

print(f"矩阵 A:\n{A}")
print(f"矩阵 B:\n{B}")
print(f"逐元素乘积 C:\n{C}")
```

### 点积

一个基本运算是两个向量 (vector) v,w∈Rnv, w \in \mathbb{R}^nv,w∈Rn 之间的点积（或内积）。它计算为 v⋅w=∑i=1nviwiv \cdot w = \sum\_{i=1}^n v\_i w\_iv⋅w=∑i=1n​vi​wi​。在几何上，它与一个向量在另一个向量上的投影相关（v⋅w=∥v∥∥w∥cos⁡θv \cdot w = \|v\| \|w\| \cos \thetav⋅w=∥v∥∥w∥cosθ，其中 θ\thetaθ 是它们之间的夹角）。

如果第一个向量被视为行向量，第二个被视为列向量，则点积在计算上等同于矩阵乘法：vTwv^T wvTw。

vTw=[v1v2⋯vn][w1w2⋮wn]=∑i=1nviwiv^T w = \begin{bmatrix} v\_1 & v\_2 & \cdots & v\_n \end{bmatrix} \begin{bmatrix} w\_1 \\ w\_2 \\ \vdots \\ w\_n \end{bmatrix} = \sum\_{i=1}^n v\_i w\_ivTw=[v1​​v2​​⋯​vn​​]​w1​w2​⋮wn​​​=i=1∑n​vi​wi​

在大型语言模型中，点积是**注意力机制 (attention mechanism)**的核心。缩放点积注意力使用点积来计算查询（QQQ）和键（KKK）向量之间的相关性，以确定对输入序列不同部分的关注程度。

```python
import torch

v = torch.tensor([1.0, 2.0, -1.0])
w = torch.tensor([3.0, -1.0, 0.5])

dot_product_val = torch.dot(v, w)

print(f"向量 v: {v}")
print(f"向量 w: {w}")
print(
    f"点积: {dot_product_val}"
)
```

### 范数：测量向量 (vector)大小

范数是一个函数，它为向量空间中的每个向量赋予一个严格为正的长度或大小（零向量除外，其长度为零）。机器学习 (machine learning)中最常见的范数是：

- **L2范数（欧几里得范数）：** ∥v∥2=∑i=1nvi2\|v\|\_2 = \sqrt{\sum\_{i=1}^n v\_i^2}∥v∥2​=∑i=1n​vi2​​。这对应于到原点的标准欧几里得距离。它常用于正则化 (regularization)（L2正则化或权重 (weight)衰减），以惩罚大的参数 (parameter)值并防止过拟合 (overfitting)。
- **L1范数（曼哈顿范数）：** ∥v∥1=∑i=1n∣vi∣\|v\|\_1 = \sum\_{i=1}^n |v\_i|∥v∥1​=∑i=1n​∣vi​∣。它测量分量的绝对值之和。L1正则化鼓励稀疏性（将一些参数精确地驱动到零）。

范数也用于归一化 (normalization)技术，如层归一化，这通常涉及根据激活值的L2范数对其进行缩放。

```python
import torch

v = torch.tensor([3.0, -4.0, 0.0])

l2_norm = torch.linalg.norm(v, ord=2)
l1_norm = torch.linalg.norm(v, ord=1)

print(f"向量 v: {v}")

print(f"L2 范数: {l2_norm}")
print(f"L1 范数: {l1_norm}")
```

### 维度与张量

尽管我们主要讨论了向量 (vector)（1维）和矩阵（2维），但深度学习 (deep learning)非常依赖于**张量**，它是对更高维度的推广。例如：

- 大型语言模型的输入可能是一个3D张量：`（批次大小，序列长度，嵌入维度）`。
- 卷积层（在某些架构中使用）的权重 (weight)可能是一个4D张量。

跟踪张量形状非常重要，以确保运算兼容。维度不匹配是深度学习代码中常见的错误来源。PyTorch及其他框架提供工具来检查和操作张量形状（`.shape`、`.reshape()`、`.permute()`等）。

本次回顾涵盖了最直接相关的线性代数内容。随着课程进展，特别是在讨论Transformer架构和注意力机制 (attention mechanism)时，矩阵乘法、点积以及管理张量维度的作用将变得越来越明显。牢固掌握这些运算对于有效理解和实现大型语言模型来说是不可或缺的。

获取即时帮助、个性化解释和交互式代码示例。

---

### 微积分回顾：梯度与优化

# 微积分回顾：梯度与优化

微积分，特别是微分学，为优化神经网络 (neural network)（包括大型语言模型）所表示的复杂函数提供了数学工具。训练这些模型涉及使损失函数 (loss function)J(θ)J(\theta)J(θ)最小化，该函数衡量模型在给定当前参数 (parameter)θ\thetaθ的情况下，在训练数据上的表现有多差。基于梯度的优化方法是进行这种最小化的标准方式，它们高度依赖于导数和梯度的观念。

### 导数与偏导数

对于一个单变量函数f(x)f(x)f(x)，导数f′(x)f'(x)f′(x)或dfdx\frac{df}{dx}dxdf​衡量函数输出相对于其输入的瞬时变化率。它告诉我们输入微小变化时输出的变化量。从几何上看，它代表了函数图在点xxx处的切线斜率。

然而，神经网络 (neural network)的损失函数 (loss function)依赖于数百万或数十亿个参数 (parameter)（权重 (weight)和偏置 (bias)），这些参数统称为向量 (vector)θ\thetaθ。因此，我们需要明白当我们在微小调整*一个*特定参数（例如θi\theta\_iθi​），同时保持所有其他参数不变时，损失J(θ)J(\theta)J(θ)是如何变化的。这正是**偏导数**所衡量的，记作∂J∂θi\frac{\partial J}{\partial \theta\_i}∂θi​∂J​。

考虑一个简单的函数f(x,y)=x2yf(x, y) = x^2yf(x,y)=x2y。
对xxx的偏导数将yyy视为常量：
∂f∂x=2xy\frac{\partial f}{\partial x} = 2xy∂x∂f​=2xy
对yyy的偏导数将xxx视为常量：
∂f∂y=x2\frac{\partial f}{\partial y} = x^2∂y∂f​=x2

### 梯度

多变量函数（例如我们的损失函数 (loss function)J(θ)J(\theta)J(θ)）的**梯度**是一个包含其所有偏导数的向量 (vector)。它记作∇J(θ)\nabla J(\theta)∇J(θ)或∇θJ(θ)\nabla\_{\theta} J(\theta)∇θ​J(θ)。如果θ=(θ1,θ2,...,θn)\theta = (\theta\_1, \theta\_2, ..., \theta\_n)θ=(θ1​,θ2​,...,θn​)，那么：

∇J(θ)=(∂J∂θ1,∂J∂θ2,...,∂J∂θn)\nabla J(\theta) = \left( \frac{\partial J}{\partial \theta\_1}, \frac{\partial J}{\partial \theta\_2}, ..., \frac{\partial J}{\partial \theta\_n} \right)∇J(θ)=(∂θ1​∂J​,∂θ2​∂J​,...,∂θn​∂J​)

梯度向量∇J(θ)\nabla J(\theta)∇J(θ)有一个非常重要的特性：它指向函数JJJ在点θ\thetaθ处**最陡峭上升**的方向。相反，负梯度−∇J(θ)-\nabla J(\theta)−∇J(θ)则指向最陡峭*下降*的方向。这是梯度下降 (gradient descent)优化的主要原理。为了使损失最小化，我们希望沿着与梯度相反的方向调整参数 (parameter)θ\thetaθ。

### 链式法则：反向传播 (backpropagation)的驱动力

神经网络 (neural network)本质上是复杂的嵌套函数。一个层的输出成为下一层的输入。例如，预测一个词可能涉及将输入嵌入 (embedding)通过多个Transformer层，每个层执行矩阵乘法并应用激活函数 (activation function)，最终通过softmax函数计算出词汇表 (vocabulary)上的概率分布。

为了计算网络深处参数 (parameter)θ\thetaθ（例如，早期层中的权重 (weight)）相对于最终损失JJJ的梯度，我们需要**链式法则**。链式法则使我们能够计算复合函数的导数。

如果变量zzz依赖于yyy，而yyy又依赖于xxx（即z=f(y)z = f(y)z=f(y)且y=g(x)y = g(x)y=g(x)），链式法则阐明了xxx的变化如何影响zzz：

dzdx=dzdy⋅dydx\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}dxdz​=dydz​⋅dxdy​

在神经网络的背景下，我们考虑一个简化的序列：输入xxx，第1层计算h=f1(x,θ1)h = f\_1(x, \theta\_1)h=f1​(x,θ1​)，第2层计算y=f2(h,θ2)y = f\_2(h, \theta\_2)y=f2​(h,θ2​)，损失为J=L(y)J = L(y)J=L(y)。为了找出损失JJJ如何随第一层中的参数θ1\theta\_1θ1​变化，我们应用链式法则：

∂J∂θ1=∂J∂y⋅∂y∂h⋅∂h∂θ1\frac{\partial J}{\partial \theta\_1} = \frac{\partial J}{\partial y} \cdot \frac{\partial y}{\partial h} \cdot \frac{\partial h}{\partial \theta\_1}∂θ1​∂J​=∂y∂J​⋅∂h∂y​⋅∂θ1​∂h​

**反向传播**本质上是一种高效的算法，用于从最终损失开始，逐层向后遍历网络，递归地应用链式法则，以计算损失对*所有*参数的梯度。

G

x

输入 x

h

隐藏层 h

x->h

theta1

参数 u03b8u2081

theta1->h

y

输出 y

h->y

theta2

参数 u03b8u2082

theta2->y

J

损失 J

y->J

> 两层网络中依赖关系的简化图示。反向传播通过从JJJ向后应用链式法则来计算如∂J∂θ1\frac{\partial J}{\partial \theta\_1}∂θ1​∂J​的梯度。

### 基于梯度的优化：梯度下降 (gradient descent)

一旦我们能够计算梯度∇θJ(θ)\nabla\_{\theta} J(\theta)∇θ​J(θ)，我们就可以使用它迭代更新模型参数 (parameter)以最小化损失。最简单的算法是**梯度下降**。

从参数θ0\theta\_0θ0​的初始估计开始，我们使用以下规则重复更新它们：

θt+1=θt−η∇θJ(θt)\theta\_{t+1} = \theta\_t - \eta \nabla\_{\theta} J(\theta\_t)θt+1​=θt​−η∇θ​J(θt​)

这里：

- θt\theta\_tθt​表示迭代ttt时的参数。
- ∇θJ(θt)\nabla\_{\theta} J(\theta\_t)∇θ​J(θt​)是使用迭代ttt时的参数计算的损失函数 (loss function)梯度。
- η\etaη是**学习率**，一个控制步长的小的正标量超参数 (hyperparameter)。选择合适的学习率对训练成功很重要。如果学习率过大，优化可能会越过最小值或发散；如果过小，训练将会非常缓慢。

这个过程会重复，直到损失收敛到最小值（或至少一个足够低的值），或者达到预设的迭代次数。实际上，我们通常不会计算*整个*数据集的梯度（那样就是批量梯度下降），因为大型语言模型的数据集非常庞大。取而代之的是，我们使用**随机梯度下降（SGD）**或**小批量梯度下降**，在每一步中仅使用一个或一小批训练样本来估计梯度。这会引入噪声，但在计算上效率更高，并且通常会带来更好的泛化能力。

### 实践中的自动微分

现代深度学习 (deep learning)框架如PyTorch提供了**自动微分**（autograd）。这意味着我们定义网络的前向传播（输入如何产生输出），而框架会自动计算反向传播 (backpropagation)所需的梯度，利用链式法则。

这里是一个展示梯度计算的最小PyTorch示例：

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
w = torch.tensor([0.5, -0.1, 0.2], requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)

y = torch.dot(w, x) + b

loss = y.square()

loss.backward()

print(f"w 的梯度: {w.grad}")
print(f"b 的梯度: {b.grad}")

y_val = (0.5 * 1.0 + (-0.1) * 2.0 + 0.2 * 3.0 + 0.1)

grad_w1_manual = 2 * y_val * x[0]

print(f"手动计算的 w_1 梯度: {grad_w1_manual}")
```

这种自动微分能力使工程师能够专注于设计复杂的模型架构和损失函数 (loss function)，而框架则处理优化所需的复杂梯度计算。然而，理解梯度和链式法则的根本原理对于设计高效模型、调试训练问题（如梯度消失或梯度爆炸）以及实现本课程后面将讨论的更高级优化方法仍然非常必要。

获取即时帮助、个性化解释和交互式代码示例。

---

### 概率与统计基础知识

# 概率与统计基础知识

概率与统计提供了处理不确定性的数学语言，这对于语言建模至为根本。本质上，语言模型为词语或符号序列分配概率。理解这些思想有助于解释模型预测，制定训练目标，并分析模型行为。本节回顾支撑大型语言模型的重要概率论观点。

### 概率分布与语言

语言模型关注预测序列中的下一个符号。给定一个包含所有可能符号的词汇表 (vocabulary) VVV，模型对下一个符号的预测可以表示为在 VVV 上的离散概率分布。这表示为词汇表中的每个符号 t∈Vt \in Vt∈V 分配一个概率 P(token=t)P(token=t)P(token=t)，使得所有概率均为非负且总和为一：∑t∈VP(token=t)=1\sum\_{t \in V} P(token=t) = 1∑t∈V​P(token=t)=1。

语言模型根据先前语境生成的下一个符号的分布，通常是**多项分布**。这种分布将二项分布推广到有多个可能结果（词汇表中的符号）的情况。模型输出一个概率向量 (vector)，每个符号对应一个概率，表示该符号是序列中下一个符号的可能性。

例如，如果我们的词汇表仅包含 {`cat`, `dog`, `sat`, `mat`} 且先前语境是 "the cat"，模型可能会输出如下概率：

- P(‘sat‘ | "the cat")=0.7P(\text{`sat` | "the cat"}) = 0.7P(‘sat‘ | "the cat")=0.7
- P(‘mat‘ | "the cat")=0.2P(\text{`mat` | "the cat"}) = 0.2P(‘mat‘ | "the cat")=0.2
- P(‘dog‘ | "the cat")=0.05P(\text{`dog` | "the cat"}) = 0.05P(‘dog‘ | "the cat")=0.05
- P(‘cat‘ | "the cat")=0.05P(\text{`cat` | "the cat"}) = 0.05P(‘cat‘ | "the cat")=0.05

这些概率总和必须为 1.0。

### 链式法则与语言建模

语言建模的一个主要目标是估计整个符号序列 P(w1,w2,…,wn)P(w\_1, w\_2, \dots, w\_n)P(w1​,w2​,…,wn​) 的概率。**概率的链式法则**使得我们能将此联合概率分解为条件概率的乘积：

P(w1,w2,…,wn)=P(w1)P(w2∣w1)P(w3∣w1,w2)…P(wn∣w1,…,wn−1)P(w\_1, w\_2, \dots, w\_n) = P(w\_1) P(w\_2 | w\_1) P(w\_3 | w\_1, w\_2) \dots P(w\_n | w\_1, \dots, w\_{n-1})P(w1​,w2​,…,wn​)=P(w1​)P(w2​∣w1​)P(w3​∣w1​,w2​)…P(wn​∣w1​,…,wn−1​)

这可以更紧凑地写为：

P(w1,…,wn)=∏i=1nP(wi∣w1,…,wi−1)P(w\_1, \dots, w\_n) = \prod\_{i=1}^{n} P(w\_i | w\_1, \dots, w\_{i-1})P(w1​,…,wn​)=i=1∏n​P(wi​∣w1​,…,wi−1​)

其中 P(w1∣w0)P(w\_1 | w\_0)P(w1​∣w0​) 通常简化为 P(w1)P(w\_1)P(w1​)。大型语言模型，特别是像 GPT 系列这样的自回归 (autoregressive)模型，被训练来近似这些条件概率 P(wi∣w1,…,wi−1)P(w\_i | w\_1, \dots, w\_{i-1})P(wi​∣w1​,…,wi−1​)。模型将前面的序列（语境）作为输入，并为下一个符号 wiw\_iwi​ 输出一个在词汇表 (vocabulary)上的概率分布。

大多数基于 Transformer 的语言模型的最后一层是线性变换，接着是**Softmax 函数**。Softmax 函数将原始分数（logits）向量 (vector) z=(z1,z2,…,z∣V∣)z = (z\_1, z\_2, \dots, z\_{|V|})z=(z1​,z2​,…,z∣V∣​) 转换为概率分布 p=(p1,p2,…,p∣V∣)p = (p\_1, p\_2, \dots, p\_{|V|})p=(p1​,p2​,…,p∣V∣​)：

pj=softmax(z)j=ezj∑k=1∣V∣ezkp\_j = \text{softmax}(z)\_j = \frac{e^{z\_j}}{\sum\_{k=1}^{|V|} e^{z\_k}}pj​=softmax(z)j​=∑k=1∣V∣​ezk​ezj​​

每个 pjp\_jpj​ 表示模型估计的词汇表中第 jjj 个符号作为下一个符号的概率。

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([1.0, -0.5, 3.0, 0.0, 1.5])

probabilities = F.softmax(logits, dim=0)

print(f"Logits: {logits}")
print(f"Probabilities: {probabilities}")
print(f"Sum of probabilities: {probabilities.sum()}")
```

### 信息论：熵、交叉熵和 KL 散度

信息论提供了量化 (quantization)信息和比较概率分布的工具，这些工具对于理解和训练语言模型很有用。

#### 熵

**熵**，记为 H(P)H(P)H(P)，表示在一个集合 X\mathcal{X}X 上的离散概率分布 PPP 中，与结果相关的平均不确定性或“惊讶度”。其计算方式为：

H(P)=−∑x∈XP(x)log⁡2P(x)H(P) = - \sum\_{x \in \mathcal{X}} P(x) \log\_2 P(x)H(P)=−x∈X∑​P(x)log2​P(x)

当使用 log⁡2\log\_2log2​ 时，单位通常是比特。高熵的分布更不确定（例如，均匀分布），而低熵的分布则更集中（可预测）。对于语言模型而言，下一个符号预测的熵值越低，表示置信度越高。

#### 交叉熵

**交叉熵**，记为 H(P,Q)H(P, Q)H(P,Q)，衡量在使用为不同分布 QQQ 设计的最优编码时，识别从分布 PPP 中提取事件所需的平均比特数。

H(P,Q)=−∑x∈XP(x)log⁡2Q(x)H(P, Q) = - \sum\_{x \in \mathcal{X}} P(x) \log\_2 Q(x)H(P,Q)=−x∈X∑​P(x)log2​Q(x)

在机器学习 (machine learning)中，交叉熵常被用作损失函数 (loss function)。PPP 代表“真实”分布（通常是一个独热向量 (vector)，其中正确符号的概率为 1，其他为 0），而 QQQ 代表模型的预测概率分布。训练期间最小化交叉熵损失会使模型的预测分布 QQQ 更接近真实分布 PPP。对于单个目标标签 yyy（表示为独热向量）和模型预测 qqq，交叉熵损失简化为 −log⁡2qy-\log\_2 q\_y−log2​qy​，其中 qyq\_yqy​ 是模型分配给正确标签的概率。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

logits = torch.tensor([[1.0, -0.5, 3.0, 0.0, 1.5]])
target = torch.tensor([2])

criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)

print(f"Logits: {logits}")
print(f"Target index: {target}")
print(f"Cross-Entropy Loss: {loss.item()}")

probabilities = F.softmax(logits, dim=1)

manual_loss = -torch.log(probabilities[0, target.item()])
print(f"Manual Calculation: -log(P(target)) = {manual_loss.item()}")
```

#### KL 散度

**Kullback-Leibler (KL) 散度**，记为 DKL(P∣∣Q)D\_{KL}(P || Q)DKL​(P∣∣Q)，衡量两个概率分布 PPP 和 QQQ 之间的差异。它量化了用 QQQ 近似 PPP 时信息损失的量。

DKL(P∣∣Q)=∑x∈XP(x)log⁡2P(x)Q(x)=H(P,Q)−H(P)D\_{KL}(P || Q) = \sum\_{x \in \mathcal{X}} P(x) \log\_2 \frac{P(x)}{Q(x)} = H(P, Q) - H(P)DKL​(P∣∣Q)=x∈X∑​P(x)log2​Q(x)P(x)​=H(P,Q)−H(P)

KL 散度始终非负 (DKL(P∣∣Q)≥0D\_{KL}(P || Q) \ge 0DKL​(P∣∣Q)≥0)，且仅当 P=QP = QP=Q 时为零。它不对称，通常意味着 DKL(P∣∣Q)≠DKL(Q∣∣P)D\_{KL}(P || Q) \neq D\_{KL}(Q || P)DKL​(P∣∣Q)=DKL​(Q∣∣P)。当真实分布 PPP（及其熵 H(P)H(P)H(P)）固定时，最小化真实分布 PPP 与模型预测 QQQ 之间的 KL 散度等同于最小化交叉熵 H(P,Q)H(P, Q)H(P,Q)，这在监督学习 (supervised learning)中通常如此。

### 从模型分布中采样

一旦模型生成了词汇表 (vocabulary)上的概率分布，我们通常需要从该分布中**采样**，尤其是在文本生成等任务中。

#### 基本采样

最简单的方法是根据预测的概率直接采样。概率较高的符号更有可能被选中。

#### 温度采样

为了控制采样过程的随机性，通常在 Softmax 之前对 Logits zzz 应用**温度缩放**：

pj=ezj/T∑k=1∣V∣ezk/Tp\_j = \frac{e^{z\_j / T}}{\sum\_{k=1}^{|V|} e^{z\_k / T}}pj​=∑k=1∣V∣​ezk​/Tezj​/T​

这里，TTT 是温度参数 (parameter)：

- 如果 T→0T \to 0T→0，概率将集中在最有可能的符号上（接近贪婪解码）。
- 如果 T=1T = 1T=1，我们得到原始的 Softmax 概率。
- 如果 T>1T > 1T>1，分布变得更平坦（更均匀），增加了随机性和多样性，但可能会降低连贯性。
- 如果 T→∞T \to \inftyT→∞，分布在词汇表上变得均匀。

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([1.0, -0.5, 3.0, 0.0, 1.5])

temperatures = [0.5, 1.0, 2.0]
sampled_tokens = {}

print(f"Original Logits: {logits}\n")

for T in temperatures:
    scaled_logits = logits / T
    probabilities = F.softmax(scaled_logits, dim=0)
    print(f"Temperature = {T}")
    print(f"Scaled Logits: {scaled_logits.numpy().round(2)}")
    print(f"Probabilities: {probabilities.numpy().round(3)}")

    samples = torch.multinomial(
        probabilities,
        num_samples=10,
        replacement=True
    )
    sampled_list = samples.tolist()
    print(f"Sampled token indices (10 samples): {sampled_list}\n")
    sampled_tokens[T] = probabilities.numpy()
```

符号 0符号 1符号 2符号 3符号 400.20.40.60.81T=0.5T=1.0T=2.0

> 针对大小为 5 的词汇表上的概率分布，这些分布是使用应用于相同初始 logits 的不同温度值（T）计算得出的。较低的温度将概率质量集中在最有可能的符号上（索引 2），而较高的温度则产生更均匀的分布。

#### 其他采样策略

除了温度，**Top-k 采样**（仅从 kkk 个最有可能的符号中采样）和 **Top-p（核）采样**（从累积概率超过阈值 ppp 的最小符号集合中采样）等其他方法常用于平衡生成文本的质量和多样性。

扎实掌握这些概率与统计基础，对于处理大型语言模型来说必不可少。它能帮助您推断模型预测，通过交叉熵等损失函数 (loss function)理解训练流程，并使用采样技术控制文本生成。

获取即时帮助、个性化解释和交互式代码示例。

---

### 数值稳定性考量

# 数值稳定性考量

虽然矩阵乘法或微分等运算的数学定义是精确的，但它们在计算机上的实现涉及有限精度的浮点数（如32位`float`或16位`half`）。这种有限精度可能在深度神经网络 (neural network)训练过程中引出一些不易察觉，有时却又十分显著的问题，尤其是在大型语言模型中常见的那些非常深的架构。若不妥善处理这些数值稳定性问题，它们可能极大地妨碍甚至完全停止训练过程。

## 梯度消失

在反向传播 (backpropagation)过程中，梯度利用链式法则计算，从输出层反向传播通过网络。每一步都涉及到乘以该层操作的局部梯度（包括激活函数 (activation function)导数）和该层的权重 (weight)。设想一个具有许多层的深层网络。如果这些梯度（特别是激活函数导数或权重矩阵）的模值持续小于1，那么梯度信号在反向传播时会呈指数级缩小。

∂L∂W1=∂L∂outN…∂out3∂in3∂in3∂out2∂out2∂in2∂in2∂W1\frac{\partial L}{\partial W\_1} = \frac{\partial L}{\partial \text{out}\_N} \dots \frac{\partial \text{out}\_3}{\partial \text{in}\_3} \frac{\partial \text{in}\_3}{\partial \text{out}\_2} \frac{\partial \text{out}\_2}{\partial \text{in}\_2} \frac{\partial \text{in}\_2}{\partial W\_1}∂W1​∂L​=∂outN​∂L​…∂in3​∂out3​​∂out2​∂in3​​∂in2​∂out2​​∂W1​∂in2​​

如果这个链式乘积中的许多项的模值都小于1，那么对于早期层（如W1W\_1W1​）的最终梯度 ∂L∂W1\frac{\partial L}{\partial W\_1}∂W1​∂L​ 会变得非常小，接近于零。

这种现象被称为**梯度消失问题**。当梯度消失时，网络早期层的权重几乎得不到更新，网络也就无法有效地从这些层的数据中学习到有意义的表示。这在训练早期深层网络时是一个很大的难题，特别是对于使用Sigmoid或Tanh等激活函数的网络，这些函数的导数在输入值较大或较小时会饱和（接近于零）。

```python
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-10, 10, 200)
sigmoid_x = torch.sigmoid(x)
sigmoid_grad = sigmoid_x * (1 - sigmoid_x)

fig, ax1 = plt.subplots()

color = '#4263eb'
ax1.set_xlabel('输入值 (x)')
ax1.set_ylabel('Sigmoid激活', color=color)
ax1.plot(x, sigmoid_x, color=color, label='Sigmoid(x)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle=':')

ax2 = ax1.twinx()

color = '#f03e3e'
ax2.set_ylabel('Sigmoid导数', color=color)
ax2.plot(
    x,
    sigmoid_grad,
    color=color,
    linestyle='--',
    label='d(Sigmoid)/dx'
)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(bottom=0)

fig.tight_layout()

plotly_json = {
  "data": [
    {
      "x": x.tolist()[::10],
      "y": sigmoid_x.tolist()[::10],
      "type": "scatter",
      "mode": "lines",
      "name": "Sigmoid(x)",
      "line": {"color": "#4263eb"}
    },
    {
      "x": x.tolist()[::10],
      "y": sigmoid_grad.tolist()[::10],
      "type": "scatter",
      "mode": "lines",
      "name": "d(Sigmoid)/dx",
      "yaxis": "y2",
      "line": {"color": "#f03e3e", "dash": "dash"}
    }
  ],
  "layout": {
    "xaxis": {"title": "输入值 (x)"},
    "yaxis": {
        "title": "Sigmoid激活",
        "titlefont": {"color": "#4263eb"},
        "tickfont": {"color": "#4263eb"},
        "gridcolor": "#e9ecef"
    },
    "yaxis2": {
        "title": "Sigmoid导数",
        "titlefont": {"color": "#f03e3e"},
        "tickfont": {"color": "#f03e3e"},
        "overlaying": "y",
        "side": "right",
        "range": [0, 0.3],
        "gridcolor": "#e9ecef"
    },
    "legend": {"x": 0.1, "y": 0.9},
    "margin": {"l": 50, "r": 50, "t": 20, "b": 40}
  }
}
```

−10−50500.20.40.60.8100.050.10.150.20.250.3Sigmoid(x)d(Sigmoid)/dx

> Sigmoid函数的导数很小（最大0.25），并且在输入值很大（正或负）时接近于零。在反向传播过程中将许多小数值相乘会导致梯度消失。

## 梯度爆炸

反之，如果链式法则乘积中的项（梯度、权重 (weight)）的模值持续大于1，那么梯度信号在反向传播 (backpropagation)时会呈指数级增长。这会引发**梯度爆炸问题**。

梯度爆炸会导致模型权重（θt+1=θt−η∇θJ(θ)\theta\_{t+1} = \theta\_t - \eta \nabla\_{\theta} J(\theta)θt+1​=θt​−η∇θ​J(θ)）的更新值过大。这些过大的更新可能使优化过程变得不稳定，导致剧烈震荡或完全发散。在实践中，这通常表现为在训练过程中损失函数 (loss function)突然飙升至`NaN`（非数字）或`Inf`（无穷大），因为数值超过了浮点数的表示范围。这对于循环连接或重复多次乘以相同权重矩阵的非常深层网络来说，尤为突出。

## 浮点运算限制

标准的深度学习 (deep learning)通常使用32位浮点数（FP32或`float`）。然而，训练大型语言模型常采用低精度格式，如16位浮点数（FP16或`half`）或BFloat16（BF16），以减少内存消耗并加速计算，尤其是在配备NVIDIA Tensor Cores等专用硬件的设备上。

这些低精度格式的表示范围明显小于FP32，并且精度也较低。

- **FP16**: 表示范围非常有限，容易出现溢出（值变为`Inf`）或下溢（值变为零）。在梯度消失的情况下常见的小梯度，在FP16中很容易变为零，从而停止学习。大梯度或中间激活值可能超出最大可表示值，导致`Inf`或`NaN`。
- **BF16**: 牺牲了精度，但保持了与FP32相似的表示范围。与FP16相比，这使其较不易受到溢出/下溢的影响，但仍可能因精度较低而引入噪声。

使用这些格式需要谨慎处理以保持数值稳定性，我们将在第20章中对此进行更详细的阐述。

## 缓解策略概述

幸运的是，已经发展出多种技术来应对这些稳定性问题，它们构成了训练深度模型的一套标准方法：

1. **细致的初始化：** 适当初始化权重 (weight)有助于从一开始就避免梯度消失或爆炸。Xavier/Glorot和Kaiming初始化（第12章）等技术根据层维度设定初始权重比例。
2. **归一化 (normalization)层：** 诸如批归一化（Batch Normalization）或在Transformer中更为常见的层归一化（Layer Normalization）（第4章）等层会将层内的激活值重新缩放，使其具有零均值和单位方差。这有助于将激活值和梯度保持在合理范围内，从而稳定训练。
3. **梯度裁剪：** 这种技术通过在权重更新步骤之前，限制梯度的最大模值或范数来直接解决梯度爆炸问题。如果梯度范数超过设定的阈值，它会被向下重新缩放。（第17章）。

   ```python
   import torch
   import torch.nn as nn

   param1 = torch.randn(100, 100, requires_grad=True)
   param2 = torch.randn(50, 100, requires_grad=True)
   parameters = [param1, param2]

   if param1.grad is None:
       param1.grad = torch.randn_like(param1) * 100
   if param2.grad is None:
       param2.grad = torch.randn_like(param2) * 50

   total_norm = 0
   for p in parameters:
       if p.grad is not None:
           param_norm = p.grad.data.norm(2)
           total_norm += param_norm.item() ** 2
   total_norm = total_norm ** 0.5
   print(f"Original Gradient Norm: {total_norm:.2f}")

   max_norm = 1.0
   nn.utils.clip_grad_norm_(parameters, max_norm)

   clipped_total_norm = 0
   for p in parameters:
        if p.grad is not None:
           param_norm = p.grad.data.norm(2)
           clipped_total_norm += param_norm.item() ** 2
   clipped_total_norm = clipped_total_norm ** 0.5
   print(f"Clipped Gradient Norm: {clipped_total_norm:.2f}")
   ```
4. **激活函数 (activation function)：** 使用非饱和激活函数，如ReLU（修正线性单元）或其变体（GeLU、SwiGLU），与Sigmoid或Tanh相比，有助于缓解梯度消失问题（第11章）。
5. **混合精度训练技术：** 诸如损失缩放（loss scaling）等方法专用于FP16，以动态调整损失函数 (loss function)的规模，从而在反向传播 (backpropagation)过程中有效放大梯度以防止下溢，然后在权重更新前将梯度重新缩放回来（第20章）。

了解这些潜在的数值难题以及应对它们的策略，对于处理现代大型语言模型的规模和深度特点来说是必不可少的。若不仔细考量数值稳定性，训练这些强大的模型在实际中将无法实现。

获取即时帮助、个性化解释和交互式代码示例。

---

### 本课程中使用的符号表示

# 本课程中使用的符号表示

为了更好地理解本课程内容，我们需要为涉及的数学对象和运算建立一个统一的符号表示。本节确立了各章中保持一致的符号。尽管我们尽可能遵守机器学习 (machine learning)和深度学习 (deep learning)文献中的标准约定，但本课程内的清晰度和一致性是首要考量。熟悉这些符号将有助于您理解描述模型架构、训练算法和评估指标的方程。

### 一般数学约定

- **标量：** 用小写斜体字母表示（例如 a,x,λ,ηa, x, \lambda, \etaa,x,λ,η）。它们通常表示单个数值，例如学习率、正则化 (regularization)参数 (parameter)，或向量 (vector)/矩阵的单个元素。
- **向量：** 用小写粗体字母表示（例如 x,y,w,b\mathbf{x}, \mathbf{y}, \mathbf{w}, \mathbf{b}x,y,w,b）。默认情况下，向量假定为列向量。我们将其维度记为 x∈Rd\mathbf{x} \in \mathbb{R}^dx∈Rd，表示一个包含 ddd 个实数值元素的向量。x\mathbf{x}x 的第 iii 个元素为 xix\_ixi​。
- **矩阵：** 用大写粗体字母表示（例如 X,Y,W\mathbf{X}, \mathbf{Y}, \mathbf{W}X,Y,W）。我们将其维度记为 W∈Rm×n\mathbf{W} \in \mathbb{R}^{m \times n}W∈Rm×n，表示一个包含 mmm 行 nnn 列的矩阵。第 iii 行第 jjj 列的元素为 WijW\_{ij}Wij​ 或 wijw\_{ij}wij​。单位矩阵用 I\mathbf{I}I 表示。
- **张量：** 高阶数组（秩 > 2）有时用大写花体字母表示（例如 T\mathcal{T}T），或者在上下文 (context)清晰表明维度时用大写粗体字母表示（例如一批矩阵）。维度会明确指定，例如 T∈Rd1×d2×⋯×dk\mathcal{T} \in \mathbb{R}^{d\_1 \times d\_2 \times \dots \times d\_k}T∈Rd1​×d2​×⋯×dk​。
- **索引和求和：** 通常用 i,j,ki, j, ki,j,k 来索引元素或维度。ttt 经常用于表示序列中的特定位置或时间步。求和用 ∑\sum∑ 表示。
- **集合：** 用大写花体字母表示（例如数据集用 D\mathcal{D}D，词汇表 (vocabulary)用 V\mathcal{V}V）。集合 S\mathcal{S}S 的大小或基数用 ∣S∣|\mathcal{S}|∣S∣ 表示。
- **函数：** 标准数学函数使用斜体小写字母（例如 f(⋅),g(⋅)f(\cdot), g(\cdot)f(⋅),g(⋅)）。激活函数 (activation function)通常用希腊字母表示（例如 Sigmoid 函数用 σ(⋅)\sigma(\cdot)σ(⋅)，ReLU 变体（如 GeLU）用 ϕ(⋅)\phi(\cdot)ϕ(⋅)）。L(⋅)L(\cdot)L(⋅) 或 J(⋅)J(\cdot)J(⋅) 通常表示损失函数 (loss function)或目标函数。
- **导数和梯度：** 标量函数 JJJ 对向量 w\mathbf{w}w 的梯度表示为 ∇wJ(w)\nabla\_{\mathbf{w}} J(\mathbf{w})∇w​J(w)，如果变量在上下文中明确，则简写为 ∇J\nabla J∇J。偏导数写为 ∂f∂x\frac{\partial f}{\partial x}∂x∂f​。

### 语言模型专用符号

- **序列：** 长度为 TTT 的输入序列通常表示为词元 (token)列表或元组 (x1,x2,…,xT)(x\_1, x\_2, \dots, x\_T)(x1​,x2​,…,xT​) 或其对应的嵌入 (embedding)向量 (vector) (x1,x2,…,xT)(\mathbf{x}\_1, \mathbf{x}\_2, \dots, \mathbf{x}\_T)(x1​,x2​,…,xT​)。TTT 表示序列长度。
- **词汇表 (vocabulary)和分词 (tokenization)：** 唯一词元（单词、子词 (subword)）的集合是词汇表 V\mathcal{V}V。其大小为 ∣V∣|\mathcal{V}|∣V∣。xtx\_txt​ 通常表示位置 ttt 处词元的整数索引。
- **嵌入：**
  - 词元嵌入矩阵：E∈R∣V∣×dmodel\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d\_{model}}E∈R∣V∣×dmodel​，其中 dmodeld\_{model}dmodel​ 是模型的隐藏维度。
  - 词元索引 iii 的嵌入向量：ei\mathbf{e}\_iei​，它是 E\mathbf{E}E 的第 iii 行。
  - 位置 ttt 的位置编码 (positional encoding)向量：pt∈Rdmodel\mathbf{p}\_t \in \mathbb{R}^{d\_{model}}pt​∈Rdmodel​。
  - 位置 ttt 的输入表示：zt=et+pt\mathbf{z}\_t = \mathbf{e}\_t + \mathbf{p}\_tzt​=et​+pt​（或根据模型而异的变体）。
- **Transformer 组成部分：**
  - 序列的查询、键、值矩阵：Q,K,V∈RT×dk\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{T \times d\_k}Q,K,V∈RT×dk​（在单个注意力头内，或投影前为 RT×dmodel\mathbb{R}^{T \times d\_{model}}RT×dmodel​）。
  - 相关联的权重 (weight)矩阵：WQ,WK,WV∈Rdmodel×dk\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V \in \mathbb{R}^{d\_{model} \times d\_k}WQ,WK,WV∈Rdmodel​×dk​（每个头）或 Rdmodel×dmodel\mathbb{R}^{d\_{model} \times d\_{model}}Rdmodel​×dmodel​（整体投影）。
  - 注意力得分矩阵：A∈RT×T\mathbf{A} \in \mathbb{R}^{T \times T}A∈RT×T。
  - 注意力输出：Attention(Q,K,V)=softmax(QKTdk)VAttention(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d\_k}})\mathbf{V}Attention(Q,K,V)=softmax(dk​​QKT​)V。
  - 第 lll 层的隐藏状态：H(l)∈RT×dmodel\mathbf{H}^{(l)} \in \mathbb{R}^{T \times d\_{model}}H(l)∈RT×dmodel​。输入嵌入通常是 H(0)\mathbf{H}^{(0)}H(0)。
  - 前馈网络：FFN(⋅\cdot⋅)。
- **模型参数 (parameter)：** 所有可训练参数（权重和偏置 (bias)）的集合用 θ\thetaθ 表示。
- **数据与训练：**
  - 数据集：D\mathcal{D}D。
  - 训练示例：(x,y)(\mathbf{x}, y)(x,y)，其中 x\mathbf{x}x 是输入，yyy 是目标。
  - 数据批次：B\mathcal{B}B。批次大小 B=∣B∣B = |\mathcal{B}|B=∣B∣。
  - 损失函数 (loss function)：L(θ)\mathcal{L}(\theta)L(θ) 或 J(θ)J(\theta)J(θ)。单示例损失 L(y^,y)L(\hat{y}, y)L(y^​,y)。
- **概率：**
  - 概率分布：P(⋅)P(\cdot)P(⋅)。
  - 条件概率：P(Y∣X)P(Y | X)P(Y∣X)。
  - 模型在给定上下文 (context) c\mathbf{c}c 下对词元 yyy 的预测概率：Pθ(y∣c)P\_{\theta}(y | \mathbf{c})Pθ​(y∣c)。
- **超参数 (hyperparameter)：**
  - 学习率：η\etaη。
  - 层数：LLL。
  - 模型隐藏维度：dmodeld\_{model}dmodel​。
  - FFN 中间维度：dffd\_{ff}dff​。
  - 注意力头数量：hhh。
  - 每个头的维度：dk=dmodel/hd\_k = d\_{model} / hdk​=dmodel​/h。
  - Dropout 概率：pdropp\_{drop}pdrop​。
  - 权重衰减系数：λ\lambdaλ。

### 代码对应关系 (PyTorch 示例)

数学符号与 PyTorch 等框架中的张量运算直接对应。掌握这种对应关系有助于代码实现。

- 一个向量 (vector) x∈Rd\mathbf{x} \in \mathbb{R}^dx∈Rd：

  ```python
  import torch
  d = 128
  x = torch.randn(d)

  x_col = torch.randn(d, 1)
  print(f"向量形状: {x.shape}, 列向量形状: {x_col.shape}")
  ```
- 一个矩阵 W∈Rm×n\mathbf{W} \in \mathbb{R}^{m \times n}W∈Rm×n：

  ```python
  m, n = 64, 128
  W = torch.randn(m, n)
  print(f"矩阵形状: {W.shape}")
  ```
- 矩阵-向量乘法 y=Wx+b\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}y=Wx+b，其中 W∈Rm×n\mathbf{W} \in \mathbb{R}^{m \times n}W∈Rm×n，x∈Rn\mathbf{x} \in \mathbb{R}^nx∈Rn，b∈Rm \mathbf{b} \in \mathbb{R}^mb∈Rm，y∈Rm \mathbf{y} \in \mathbb{R}^my∈Rm：

  ```python

  n = 128
  m = 64
  x = torch.randn(n)
  b = torch.randn(m)
  W = torch.randn(m, n)

  y = W @ x + b

  print(f"输入 x 形状: {x.shape}")
  print(f"权重 W 形状: {W.shape}")
  print(f"偏置 b 形状: {b.shape}")
  print(f"输出 y 形状: {y.shape}")
  ```
- 批处理：通常，第一个维度表示批次大小 BBB。例如，一个序列批次的形状可能是 (B,T,dmodel)(B, T, d\_{model})(B,T,dmodel​)。

这些符号构成了我们后续讨论的根本。后续章节中引入的任何偏差或特定于上下文 (context)的符号都将在局部定义。在学习过程中请随时参考此部分。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 3 Revisiting Sequence Processing Architectures

### 循环神经网络 (RNN) 的基本内容

# 循环神经网络 (RNN) 的基本内容

在 Transformer 出现之前，循环神经网络 (neural network) (RNN) 是处理序列数据（例如文本或时间序列）的标准架构。与独立处理输入的普通前馈网络不同，RNN 具有一种记忆形式，使得序列中先前步骤的信息能够影响当前步骤的处理。这使得它们天生适合处理那些注重上下文 (context)和顺序的任务。

### 核心思想：隐状态和循环

RNN 的核心思想是**隐状态**，在时间步 ttt 通常表示为 hth\_tht​。这个隐状态充当了截至该时间点序列中已见到信息的压缩摘要。在每个时间步 ttt，RNN 接收两个输入：序列中的当前输入元素 xtx\_txt​ 和前一时间步的隐状态 ht−1h\_{t-1}ht−1​。然后，它计算一个新的隐状态 hth\_tht​，并可选地生成一个输出 yty\_tyt​。

想象阅读一个句子：“The cat sat on the \_\_\_”。要预测下一个词，你需要记住“The cat sat on the”。RNN 通过在处理每个词时更新其隐状态来模仿这一点，从而传递相关上下文 (context)。

这个过程包含一个循环：在每个时间步都应用相同的操作和权重 (weight)集，并使用前一个隐状态作为输入。这种共享权重结构使得 RNN 在参数 (parameter)上很高效，因为它们不需要为序列中的每个位置设置单独的参数。

### 数学表达式

我们来看一下在时间步 ttt 简单 RNN 单元内部的计算：

1. **计算新的隐状态 hth\_tht​**：这通常通过使用权重 (weight)矩阵和激活函数 (activation function)（通常是双曲正切函数 tanh⁡\tanhtanh）将当前输入 xtx\_txt​ 和前一个隐状态 ht−1h\_{t-1}ht−1​ 组合起来完成。

   ht=tanh⁡(Wxhxt+Whhht−1+bh)h\_t = \tanh(W\_{xh} x\_t + W\_{hh} h\_{t-1} + b\_h)ht​=tanh(Wxh​xt​+Whh​ht−1​+bh​)

   此处：

   - xtx\_txt​ 是时间步 ttt 的输入向量 (vector)。
   - ht−1h\_{t-1}ht−1​ 是前一时间步的隐状态向量（h0h\_0h0​ 通常初始化为零）。
   - WxhW\_{xh}Wxh​ 是连接输入 xtx\_txt​ 到隐状态的权重矩阵。
   - WhhW\_{hh}Whh​ 是连接前一个隐状态 ht−1h\_{t-1}ht−1​ 到当前隐状态的权重矩阵。
   - bhb\_hbh​ 是隐状态计算的偏置 (bias)向量。
   - tanh⁡\tanhtanh 是双曲正切激活函数，它将值压缩到 -1 和 1 之间。
2. **计算输出 yty\_tyt​（可选）**：根据任务不同，可能会在每个时间步基于当前隐状态生成一个输出。

   yt=Whyht+byy\_t = W\_{hy} h\_t + b\_yyt​=Why​ht​+by​

   此处：

   - hth\_tht​ 是当前隐状态向量。
   - WhyW\_{hy}Why​ 是连接隐状态到输出的权重矩阵。
   - byb\_yby​ 是输出计算的偏置向量。
   - 根据具体应用，可能会对 yty\_tyt​ 应用一个激活函数（如用于分类的 softmax 或用于回归的线性函数）。

重要的是，权重矩阵 (Wxh,Whh,WhyW\_{xh}, W\_{hh}, W\_{hy}Wxh​,Whh​,Why​) 和偏置 (bh,byb\_h, b\_ybh​,by​) 在所有时间步中都是*相同*的。网络学习一个单一的转换函数，并重复应用它。

### 网络在时间上的展开

虽然我们经常用循环来绘制 RNN 单元，但将其在序列长度上“展开”来直观理解很有用。这展示了计算如何从一个时间步流向下一个时间步。

G

clusterₜₘinus₁

t-1

clusterₜ

t

clusterₜₚlus₁

t+1

hₚrev

h(t-1)

rnnₚrev

RNN 单元

hₚrev->rnnₚrev

xₚrev

x(t-1)

xₚrev->rnnₚrev

h\_curr

h(t)

rnnₙext

RNN 单元

h\_curr->rnnₙext

x\_curr

x(t)

rnn\_curr

RNN 单元

x\_curr->rnn\_curr

rnn\_curr->h\_curr

y\_curr

y(t)

rnn\_curr->y\_curr

hₙext

h(t+1)

xₙext

x(t+1)

xₙext->rnnₙext

rnnₙext->hₙext

> 一个在三个时间步上展开的 RNN。相同的 RNN 单元（表示共享权重 (weight) Wxh,Whh,WhyW\_{xh}, W\_{hh}, W\_{hy}Wxh​,Whh​,Why​）处理输入 xtx\_txt​ 和前一个隐状态 ht−1h\_{t-1}ht−1​，以生成当前隐状态 hth\_tht​ 和输出 yty\_tyt​。

### 简单的 PyTorch 实现

PyTorch 为 RNN 提供了方便的模块。以下是定义和使用单层 RNN 的一个基本示例：

```python
import torch
import torch.nn as nn

input_size = 10
hidden_size = 20
sequence_length = 5
batch_size = 3

rnn_layer = nn.RNN(input_size, hidden_size, batch_first=True)

input_sequence = torch.randn(batch_size, sequence_length, input_size)

initial_hidden_state = torch.zeros(1, batch_size, hidden_size)

output, final_hidden_state = rnn_layer(input_sequence, initial_hidden_state)

print("Input shape:", input_sequence.shape)

print("Output shape:", output.shape)

print("Final hidden state shape:", final_hidden_state.shape)

last_time_step_output = output[:, -1, :]
print("Last time step hidden state from output shape:",
      last_time_step_output.shape)

print(
    "最终隐状态和最后一个输出步是否相等？",
    torch.allclose(
        final_hidden_state.squeeze(0),
        last_time_step_output
    )
)
```

这种简单的结构使得 RNN 能够对序列依赖进行建模。然而，正如我们将在下一节看到的，基本的 RNN 在学习序列中相距较远元素之间的关系时存在困难。这个局限性促成了 LSTM 和 GRU 等更复杂的架构的出现。

获取即时帮助、个性化解释和交互式代码示例。

---

### 简单RNN的局限性

# 简单RNN的局限性

尽管简单的循环神经网络 (neural network)（RNN）通过维护一个隐藏状态来处理序列，但在处理更长序列时会遇到显著困难。这些局限性是开发LSTM、GRU等更复杂架构，并最终催生Transformer的主要推动力。其主要问题源于训练时梯度在网络中经过多个时间步的传播方式。

### 梯度消失问题

训练RNN通常涉及时间反向传播 (backpropagation)（BPTT）。该算法将RNN沿序列长度展开并应用标准反向传播。为了计算损失函数 (loss function)相对于序列早期参数 (parameter)（例如，影响隐藏状态 hkh\_khk​ 的权重 (weight)）的梯度，链式法则要求将对应循环状态转换的多个雅可比矩阵相乘。

设 LLL 为序列上的总损失。损失相对于早期隐藏状态 hkh\_khk​ 的梯度取决于所有后续状态 hth\_tht​ （t>kt > kt>k）的梯度：

∂L∂hk=∑t=k+1T∂Lt∂ht∂ht∂hk\frac{\partial L}{\partial h\_k} = \sum\_{t=k+1}^{T} \frac{\partial L\_t}{\partial h\_t} \frac{\partial h\_t}{\partial h\_k}∂hk​∂L​=t=k+1∑T​∂ht​∂Lt​​∂hk​∂ht​​

其中 TTT 是序列长度，LtL\_tLt​ 是时间步 ttt 的损失。项 ∂ht∂hk\frac{\partial h\_t}{\partial h\_k}∂hk​∂ht​​ 表示 hkh\_khk​ 对 hth\_tht​ 的影响。这种影响通过链式计算状态转换的雅可比矩阵得到：

∂ht∂hk=∂ht∂ht−1∂ht−1∂ht−2…∂hk+1∂hk\frac{\partial h\_t}{\partial h\_k} = \frac{\partial h\_t}{\partial h\_{t-1}} \frac{\partial h\_{t-1}}{\partial h\_{t-2}} \dots \frac{\partial h\_{k+1}}{\partial h\_k}∂hk​∂ht​​=∂ht−1​∂ht​​∂ht−2​∂ht−1​​…∂hk​∂hk+1​​

每个雅可比矩阵 ∂hi∂hi−1\frac{\partial h\_i}{\partial h\_{i-1}}∂hi−1​∂hi​​ 都依赖于循环权重矩阵 WhhW\_{hh}Whh​ 和激活函数 (activation function)的导数（例如，tanh⁡′\tanh'tanh′）。如果这些雅可比矩阵的幅值（特别是奇异值）持续小于1，它们的乘积会随着距离 t−kt-kt−k 的增加呈指数级减小。

BPTT

cluster₀

前向传播（时间步）

cluster₁

梯度流（BPTT）

xₖ

xₖ

hₖ

hₖ

xₖ->hₖ

lossₖ

Lossₖ

hₖ->lossₖ

hₖplus1

hₖ₊₁

hₖ->hₖplus1

 Wₕh

gradₕₖ

∂L/∂hₖ

lossₖ->gradₕₖ

lossₖplus1

Lossₖ₊₁

hₖplus1->lossₖplus1

h\_dots

...

hₖplus1->h\_dots

 Wₕh

xₖplus1

xₖ₊₁

xₖplus1->hₖplus1

gradₕₖplus1

gradₕₖplus1

lossₖplus1->gradₕₖplus1

loss\_dots

...

h\_dots->loss\_dots

h\_T

h\_T

h\_dots->h\_T

 Wₕh

x\_dots

...

x\_dots->h\_dots

gradₕ\_dots

gradₕ\_dots

loss\_dots->gradₕ\_dots

loss\_T

Loss\_T

h\_T->loss\_T

x\_T

x\_T

x\_T->h\_T

grad\_L\_T

grad\_L\_T

loss\_T->grad\_L\_T

gradₕ\_T

∂L/∂h\_T

grad\_L\_T->gradₕ\_T

gradₕ\_T->gradₕ\_dots

 × (∂h\_T/∂h...)

labelₘult
雅可比矩阵的
重复乘法

gradₕ\_dots->gradₕₖplus1

 × (...)

gradₕₖplus1->gradₕₖ

 × (∂hₖ₊₁/∂hₖ)

> 时间反向传播（BPTT）示意图。后续时间步（如 TTT）的梯度必须通过循环连接（受 WhhW\_{hh}Whh​ 影响）反向流动，以更新影响早期状态（如 hkh\_khk​）的参数。在此反向传播过程中，雅可比矩阵的重复相乘是梯度消失/爆炸问题的根源。

这种现象被称为**梯度消失问题**。结果是，来自后续时间步的误差信号变得过于微弱，无法有效更新负责获取早期信息的权重。实际上，RNN难以学习数据中的长距离依赖关系。模型很难连接序列中相隔多个时间步的事件或信息。

### 梯度爆炸问题

反之，如果雅可比矩阵 ∂hi∂hi−1\frac{\partial h\_i}{\partial h\_{i-1}}∂hi−1​∂hi​​ 的幅值持续大于1，它们的乘积会呈指数级增大。这会导致**梯度爆炸问题**。

当梯度爆炸时，权重 (weight)更新会变得过大，可能导致模型的参数 (parameter)发散至无穷大或NaN（非数值）。这会使训练过程不稳定，常导致损失函数 (loss function)突然急剧增加或训练完全失败。

梯度爆炸问题通常更容易发现，有时可以通过**梯度裁剪**（当梯度范数超过一定阈值时将其按比例缩小）等方法来减轻，但裁剪更多是一种权宜之计，而非根本解决方法。它能阻止灾难性发散，但并未从根本上解决在长时序上传播有效梯度信号的问题，这正是梯度消失问题所针对的主要局限性。

```python
import torch
import torch.nn as nn

max_grad_norm = 1.0

nn.utils.clip_grad_norm_(params, max_grad_norm)
```

> PyTorch中演示梯度裁剪的简单示例。计算梯度后，如果总范数超过 `max_grad_norm`，`clip_grad_norm_` 会按比例缩小梯度。

### 对性能的影响

这些梯度问题严重限制了简单RNN在需要对长序列建模或获取跨越多个时间步的依赖关系的任务上的实际效果。例如：

- **机器翻译：** 翻译长句时，其含义可能依赖于相隔较远的词语。
- **文档摘要：** 理解长篇文章的主旨需要整合跨段落的信息。
- **语言建模：** 预测下一个词通常依赖于文本中很早之前建立的上下文 (context)。

无法可靠学习这些长距离依赖关系意味着，与旨在解决这些梯度传播问题的模型相比，简单RNN通常表现不佳。这促成了LSTM和GRU的出现，它们包含了专门设计用于更好控制信息和梯度随时间流动的门控机制，我们将在后续章节中看到这一点。

获取即时帮助、个性化解释和交互式代码示例。

---

### 长短期记忆（LSTM）网络

# 长短期记忆（LSTM）网络

简单循环神经网络 (neural network) (RNN)（RNNs）在学习长序列依赖关系时面临困难。反向传播 (backpropagation)过程中涉及时间上的重复乘法，导致梯度要么趋于零（消失），要么不可控地增长（爆炸）。梯度消失问题尤其棘手，因为它阻碍了网络学习序列中相距较远事件之间的关联。

长短期记忆（LSTM）网络专门用于解决这些问题。它由Hochreiter和Schmidhuber于1997年提出，比简单RNN包含更复杂的内部结构，其特点是具有一个专门的*细胞状态*和*门*，用于调节信息流动。这种架构使LSTM能够长时间保持信息，有效捕获长距离依赖关系。

### 核心思想：细胞状态和门

可以将标准RNN的隐藏状态hth\_tht​看作一种工作记忆，它在每个时间步都会被覆盖。LSTM引入了一个额外组件，即*细胞状态* CtC\_tCt​，它作用类似于传送带或记忆高速公路。信息可以在这条高速公路上相对不变地流动，从而更容易长时间保留上下文 (context)。

LSTM的主要创新在于它们能够使用称为*门*的结构，选择性地从细胞状态中添加或移除信息。门由一个S形（sigmoid）神经网络 (neural network)层和一个逐点乘法操作组成。S形层输出0到1之间的数字，描述每个组件应允许通过多少信息。0表示“不让任何信息通过”，而1表示“让所有信息通过”。

一个LSTM细胞通常有三个主要门：

1. **遗忘门：** 决定从细胞状态中丢弃哪些信息。
2. **输入门：** 决定将哪些新信息存储到细胞状态中。
3. **输出门：** 决定细胞状态的哪一部分作为输出。

我们来考察这些组件在单个时间步ttt如何协同工作，给定输入xtx\_txt​、前一个隐藏状态ht−1h\_{t-1}ht−1​和前一个细胞状态Ct−1C\_{t-1}Ct−1​。

### 遗忘门 (ftf\_tft​)

第一步是决定从前一个细胞状态Ct−1C\_{t-1}Ct−1​中丢弃哪些信息。这个决定由遗忘门层做出。它查看ht−1h\_{t-1}ht−1​和xtx\_txt​，并为细胞状态Ct−1C\_{t-1}Ct−1​中的每个数字输出一个0到1之间的数值。

ft=σ(Wf[ht−1,xt]+bf)f\_t = \sigma(W\_f [h\_{t-1}, x\_t] + b\_f)ft​=σ(Wf​[ht−1​,xt​]+bf​)

这里，σ\sigmaσ是S形激活函数 (activation function)，WfW\_fWf​表示权重 (weight)，bfb\_fbf​表示遗忘门的偏置 (bias)。表示法[ht−1,xt][h\_{t-1}, x\_t][ht−1​,xt​]表明前一个隐藏状态和当前输入是拼接在一起的。

### 输入门 (iti\_tit​) 和候选细胞状态 (C~t\tilde{C}\_tC~t​)

接下来，我们需要决定将哪些新信息存储到细胞状态中。这包含两部分。首先，一个*输入门层*（另一个S形层）决定我们要更新哪些值。

it=σ(Wi[ht−1,xt]+bi)i\_t = \sigma(W\_i [h\_{t-1}, x\_t] + b\_i)it​=σ(Wi​[ht−1​,xt​]+bi​)

其次，一个`tanh`层创建了一个新的候选值向量 (vector)C~t\tilde{C}\_tC~t​，这些值可以被添加到状态中。

C~t=tanh⁡(WC[ht−1,xt]+bC)\tilde{C}\_t = \tanh(W\_C [h\_{t-1}, x\_t] + b\_C)C~t​=tanh(WC​[ht−1​,xt​]+bC​)

### 更新细胞状态 (CtC\_tCt​)

现在，我们将旧的细胞状态Ct−1C\_{t-1}Ct−1​更新为新的细胞状态CtC\_tCt​。我们将旧状态乘以ftf\_tft​，遗忘我们之前决定遗忘的信息。然后我们加上it∗C~ti\_t \* \tilde{C}\_tit​∗C~t​。这是新的候选信息，根据我们决定更新每个状态值的程度进行缩放。

Ct=ft∗Ct−1+it∗C~tC\_t = f\_t \* C\_{t-1} + i\_t \* \tilde{C}\_tCt​=ft​∗Ct−1​+it​∗C~t​

这里使用加法非常重要。与简单RNN中的重复乘法不同，这种加法作用使得梯度更容易在时间上向后流动，而不会那么快消失。

### 输出门 (oto\_tot​) 和隐藏状态 (hth\_tht​)

最后，我们需要决定输出什么。这个输出将基于我们的细胞状态，但会是一个过滤后的版本。首先，我们运行一个S形层，它决定了我们将输出细胞状态的哪些部分。

ot=σ(Wo[ht−1,xt]+bo)o\_t = \sigma(W\_o [h\_{t-1}, x\_t] + b\_o)ot​=σ(Wo​[ht−1​,xt​]+bo​)

然后，我们将细胞状态通过`tanh`函数（将值推到-1到1之间），并将其乘以S形门的输出，这样我们只输出我们决定输出的部分。这个最终输出就是隐藏状态hth\_tht​。

ht=ot∗tanh⁡(Ct)h\_t = o\_t \* \tanh(C\_t)ht​=ot​∗tanh(Ct​)

隐藏状态hth\_tht​随后与新的细胞状态CtC\_tCt​一起传递给下一个时间步。

LSTM\_Cell

clusterᵢnput

输入 (t-1)

cluster\_current

当前输入 (t)

cluster\_gates

门与更新 (t)

ht₁

h(t-1)

forget\_gate

f
t
=σ

ht₁->forget\_gate

input\_gate

i
t
=σ

ht₁->input\_gate

candidate\_gate

Č
t
=tanh

ht₁->candidate\_gate

output\_gate

o
t
=σ

ht₁->output\_gate

Ct₁

C(t-1)

pointₘul1

\*

Ct₁->pointₘul1

Ct₁->pointₘul1

xt

x(t)

xt->forget\_gate

xt->input\_gate

xt->candidate\_gate

xt->output\_gate

forget\_gate->pointₘul1

pointₘul2

\*

input\_gate->pointₘul2

candidate\_gate->pointₘul2

pointₘul3

\*

output\_gate->pointₘul3

pointₐdd

+

pointₘul1->pointₐdd

遗忘

pointₘul2->pointₐdd

输入

tanh\_Ct

tanh

pointₐdd->tanh\_Ct

C
t

Ct

C(t)

pointₐdd->Ct

ht

h(t)

pointₘul3->ht

tanh\_Ct->pointₘul3

> LSTM细胞在时间步ttt的信息流动简化视图。细胞状态CtC\_tCt​充当传送带，由遗忘门和输入门修改。输出门过滤细胞状态以生成隐藏状态hth\_tht​。

### 在PyTorch中的实现

PyTorch等深度学习 (deep learning)框架提供了LSTM层的有效实现。使用`torch.nn.LSTM`可以抽象掉门的计算细节。

```python
import torch
import torch.nn as nn

input_size = 10
hidden_size = 20
num_layers = 1
batch_size = 5
seq_len = 7

lstm_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

input_seq = torch.randn(batch_size, seq_len, input_size)

h0 = torch.randn(num_layers, batch_size, hidden_size)
c0 = torch.randn(num_layers, batch_size, hidden_size)

output, (hn, cn) = lstm_layer(input_seq, (h0, c0))

print("输入形状:", input_seq.shape)
print("输出形状（所有隐藏状态）:", output.shape)
print("最终隐藏状态形状 (hn):", hn.shape)
print("最终细胞状态形状 (cn):", cn.shape)
```

请注意，隐藏状态和细胞状态以元组`(hn, cn)`的形式返回。这反映了LSTM在整个序列处理过程中维护的两个不同内部状态。`output`张量提供每个时间步的隐藏状态hth\_tht​，这在序列到序列模型中通常很有用。

通过使用门来控制信息流动以及细胞状态的加法更新机制，LSTM有效缓解了梯度消失问题，与简单RNN相比，能够学习跨越更长时间范围的依赖关系。这种能力使它们在基于注意力模型兴起之前，成为许多自然语言处理任务中的主导架构。虽然我们接下来将研究的门控循环单元 (GRU)（GRUs）提供了一种稍微简单的门控机制，但LSTM引入的核心原则仍然具有影响力。

获取即时帮助、个性化解释和交互式代码示例。

---

### 门控循环单元 (GRU)

# 门控循环单元 (GRU)

长短期记忆（LSTM）网络以有效解决梯度消失问题并捕获长距离依赖关系而闻名。然而，这些网络通常由于其架构（包含三个独立的门：输入、遗忘、输出以及一个独立的细胞状态）而引入了相当大的计算复杂性。2014年，Cho等人提出了一种名为门控循环单元（GRU）的变体，它在许多任务上实现了相似的表现，但结构更简单。GRU将遗忘门和输入门合并成一个“更新门”，并合并了细胞状态和隐藏状态。

让我们看看GRU单元的结构。与其他RNN一样，它接收当前输入 xtx\_txt​ 和前一个隐藏状态 ht−1h\_{t-1}ht−1​ 来生成下一个隐藏状态 hth\_tht​。其作用通过两个门来实现：重置门 (rtr\_trt​) 和更新门 (ztz\_tzt​)。

### 重置门

重置门决定了如何将新输入与前一个隐藏状态结合。具体来说，它控制在计算候选隐藏状态时，前一个隐藏状态 (ht−1h\_{t-1}ht−1​) 应该“遗忘”多少。其计算方式如下：

rt=σ(Wrxt+Urht−1+br)r\_t = \sigma(W\_r x\_t + U\_r h\_{t-1} + b\_r)rt​=σ(Wr​xt​+Ur​ht−1​+br​)

这里，WrW\_rWr​、UrU\_rUr​ 和 brb\_rbr​ 是重置门的可学习权重 (weight)矩阵和偏置 (bias)向量 (vector)。Sigmoid函数 σ\sigmaσ 将输出压缩在0到1之间。值接近0表示前一个隐藏状态大部分被忽略，而值接近1表示它大部分被保留。

### 更新门

更新门决定了前一个隐藏状态 (ht−1h\_{t-1}ht−1​) 有多少应该传递到新的隐藏状态 (hth\_tht​)，以及有多少*新的*候选隐藏状态应该被使用。这个门基本结合了LSTM的遗忘门和输入门的作用。它的计算方式与重置门类似：

zt=σ(Wzxt+Uzht−1+bz)z\_t = \sigma(W\_z x\_t + U\_z h\_{t-1} + b\_z)zt​=σ(Wz​xt​+Uz​ht−1​+bz​)

同样，WzW\_zWz​、UzU\_zUz​ 和 bzb\_zbz​ 是可学习参数 (parameter)，σ\sigmaσ 是Sigmoid函数。ztz\_tzt​ 的值接近1表示前一个状态 ht−1h\_{t-1}ht−1​ 大部分被保留，而值接近0表示新的候选状态被主要使用。

### 候选隐藏状态

在计算最终隐藏状态之前，GRU会计算一个候选隐藏状态 (h~t\tilde{h}\_th~t​)。这个计算受到重置门的影响，重置门决定了前一个隐藏状态 ht−1h\_{t-1}ht−1​ 的贡献程度：

h~t=tanh⁡(Whxt+Uh(rt⊙ht−1)+bh)\tilde{h}\_t = \tanh(W\_h x\_t + U\_h (r\_t \odot h\_{t-1}) + b\_h)h~t​=tanh(Wh​xt​+Uh​(rt​⊙ht−1​)+bh​)

这里，⊙\odot⊙ 表示按元素相乘（哈达玛积）。如果重置门 rtr\_trt​ 的值接近0，那么 ht−1h\_{t-1}ht−1​ 的贡献将被有效清除，使候选状态主要基于当前输入 xtx\_txt​。WhW\_hWh​、UhU\_hUh​ 和 bhb\_hbh​ 是另一组可学习的权重 (weight)和偏置 (bias)。tanh⁡\tanhtanh 函数有助于调节网络中的值，通常将其压缩在-1到1之间。

### 最终隐藏状态

最后，更新门 ztz\_tzt​ 在前一个隐藏状态 ht−1h\_{t-1}ht−1​ 和候选隐藏状态 h~t\tilde{h}\_th~t​ 之间进行调节，以生成当前时间步的最终隐藏状态 hth\_tht​：

ht=(1−zt)⊙ht−1+zt⊙h~th\_t = (1 - z\_t) \odot h\_{t-1} + z\_t \odot \tilde{h}\_tht​=(1−zt​)⊙ht−1​+zt​⊙h~t​

这个方程的作用类似于加权平均。如果 ztz\_tzt​ 接近1，候选状态 h~t\tilde{h}\_th~t​ 贡献更多，从而有效地用新信息更新隐藏状态。如果 ztz\_tzt​ 接近0，前一个隐藏状态 ht−1h\_{t-1}ht−1​ 被保留更多，允许信息在多个时间步中不变地传递。这个机制是GRU如何维护长距离依赖关系的方式。

GRU\_Cell

clusterᵢnput

输入

cluster\_gates

门

cluster\_candidate

候选状态

clusterₒutput

输出

xt

xₜ

rt

重置门 (rₜ)
σ(...)

xt->rt

Wᵣ

zt

更新门 (zₜ)
σ(...)

xt->zt

W\_z

htₜilde

候选 (h̃ₜ)
tanh(...)

xt->htₜilde

Wₕ

htm1

hₜ₋₁

htm1->rt

Uᵣ

htm1->zt

U\_z

htm1->htₜilde

Uₕ \*

ht

hₜ

htm1->ht

(1 - zₜ) ⊙

rt->htₜilde

⊙

zt->ht

⊙

htₜilde->ht

(1-zₜ)⊙

> GRU单元内信息流的简化视图。xtx\_txt​ 是输入，ht−1h\_{t-1}ht−1​ 是前一个隐藏状态。重置门 (rtr\_trt​) 影响候选状态 (h~t\tilde{h}\_th~t​)，更新门 (ztz\_tzt​) 将候选状态与前一个状态结合，生成最终隐藏状态 hth\_tht​。

### GRU与LSTM的比较

GRU通常被视为LSTM的更精简替代方案。

- **更少的门：** GRU有两个门（重置、更新），而LSTM有三个（输入、遗忘、输出）。
- **无独立细胞状态：** GRU不像LSTM那样维护独立的细胞状态 (ctc\_tct​)；它们直接修改隐藏状态 (hth\_tht​)。
- **效率：** 参数 (parameter)和操作的减少通常意味着GRU比同等大小的LSTM训练速度稍快，并且内存需求更少。

在实践中，LSTM和GRU之间的选择通常取决于具体的数据集和任务。两者在所有场景下都没有哪个能持续胜过另一个，尽管GRU因其相对简单和相近的表现而获得欢迎。

以下是一个PyTorch代码片段，呈现了单个GRU步骤的核心计算（假设输入为`x_t`、`h_tm1`以及预定义的权重 (weight)/偏置 (bias)张量）：

```python
import torch
import torch.nn.functional as F

batch_size = 1
input_size = 10
hidden_size = 20

x_t = torch.randn(batch_size, input_size)
h_tm1 = torch.randn(batch_size, hidden_size)

W_r = torch.randn(input_size, hidden_size)
U_r = torch.randn(hidden_size, hidden_size)
b_r = torch.randn(hidden_size)

W_z = torch.randn(input_size, hidden_size)
U_z = torch.randn(hidden_size, hidden_size)
b_z = torch.randn(hidden_size)

W_h = torch.randn(input_size, hidden_size)
U_h = torch.randn(hidden_size, hidden_size)
b_h = torch.randn(hidden_size)

r_t = torch.sigmoid(x_t @ W_r + h_tm1 @ U_r + b_r)

z_t = torch.sigmoid(x_t @ W_z + h_tm1 @ U_z + b_z)

h_tilde_t = torch.tanh(x_t @ W_h + (r_t * h_tm1) @ U_h + b_h)

h_t = (1 - z_t) * h_tm1 + z_t * h_tilde_t

print("前一个隐藏状态的形状:", h_tm1.shape)
print("当前隐藏状态的形状:", h_t.shape)
```

这种简化结构虽然有效，但仍然依赖于顺序处理。时间步 ttt 的计算依赖于时间步 t−1t-1t−1 的结果。这种固有的顺序依赖性限制了训练期间的并行化，并且在处理非常长的序列时仍然是一个瓶颈，这为Transformer中使用的非循环注意力机制 (attention mechanism)奠定了基础。

获取即时帮助、个性化解释和交互式代码示例。

---

### 基于RNN的序列到序列模型

# 基于RNN的序列到序列模型

“循环神经网络 (neural network)（RNN）、长短期记忆网络（LSTM）和门控循环单元（GRU）按元素处理序列，但许多问题需要将一个长度的输入序列映射到可能不同长度的输出序列。例如，机器翻译（将法语句子翻译成英语）或文本摘要（将长文章压缩成几句话）。输入和输出的长度通常没有直接关系。标准RNN架构通常为每个输入产生一个输出，因此不直接适用于这些任务。”

为解决此问题，序列到序列（seq2seq）框架应运而生，主要采用长短期记忆网络（LSTM）或门控循环单元（GRU）等循环架构。其核心思想是使用两个独立的循环神经网络：一个处理输入序列（编码器），另一个生成输出序列（解码器）。

### 编码器-解码器架构

seq2seq模型包含两个主要组成部分：

1. **编码器**：这个循环神经网络 (neural network) (RNN)逐个读取输入序列的令牌（例如，单词或子词 (subword)）。它的目标不是在每一步都生成输出，而是将整个输入序列的信息压缩成一个固定大小的向量 (vector)表示。这个向量常被称为“上下文 (context)向量”或“思维向量”，通常由编码器RNN的最终隐藏状态（以及长短期记忆网络 (LSTM)的细胞状态）表示。
2. **解码器**：这个循环神经网络将编码器生成的上下文向量作为其初始隐藏状态。然后，它逐个令牌地生成输出序列。在每一步ttt，解码器接收上下文向量、它自己的前一个隐藏状态ht−1h\_{t-1}ht−1​，以及前面生成的输出令牌yt−1y\_{t-1}yt−1​作为输入，以生成下一个输出令牌yty\_tyt​并将其隐藏状态更新为hth\_tht​。生成过程通常以一个特殊的序列开始符`<SOS>`令牌开始，并持续到生成序列结束符`<EOS>`令牌或达到最大长度为止。

G

clusterₑncoder

编码器

cluster\_decoder

解码器

EncInput

输入序列
(X1, X2, ..., Xn)

EncoderRNN

RNN / LSTM / GRU

EncInput->EncoderRNN

Context

上下文向量
(最终隐藏状态)

EncoderRNN->Context

 处理输入

DecoderRNN

RNN / LSTM / GRU

Context->DecoderRNN

 初始隐藏状态

DecoderRNN->DecoderRNN

 上一个输出令牌
 上一个隐藏状态

DecOutput

输出序列
(Y1, Y2, ..., Ym)

DecoderRNN->DecOutput

 生成输出

StartToken

<SOS>

StartToken->DecoderRNN

 初始输入

> 基于RNN的序列到序列模型的高层结构。编码器处理输入以生成上下文向量，该向量用于初始化解码器以生成输出序列。

### 信息流与上下文 (context)向量 (vector)

编码器处理输入序列X=(x1,x2,...,xn)X = (x\_1, x\_2, ..., x\_n)X=(x1​,x2​,...,xn​)并输出一个上下文向量ccc。这个向量ccc旨在总结整个输入序列。

c=编码器(x1,x2,...,xn)c = \text{编码器}(x\_1, x\_2, ..., x\_n)c=编码器(x1​,x2​,...,xn​)

通常，对于长短期记忆网络 (LSTM)， ccc 将是最终的隐藏状态 hnh\_nhn​ 和细胞状态 CnC\_nCn​。

解码器用这个上下文进行初始化（例如，h0dec=hnench\_0^{\text{dec}} = h\_n^{\text{enc}}h0dec​=hnenc​，C0dec=CnencC\_0^{\text{dec}} = C\_n^{\text{enc}}C0dec​=Cnenc​）。然后，它一次生成一个令牌，产生输出序列Y=(y1,y2,...,ym)Y = (y\_1, y\_2, ..., y\_m)Y=(y1​,y2​,...,ym​)。下一个令牌yty\_tyt​的概率取决于上下文ccc、前一个令牌yt−1y\_{t-1}yt−1​以及解码器当前的隐藏状态htdech\_t^{\text{dec}}htdec​：

P(yt∣y1,...,yt−1,c)=解码器(yt−1,ht−1dec,c)P(y\_t | y\_1, ..., y\_{t-1}, c) = \text{解码器}(y\_{t-1}, h\_{t-1}^{\text{dec}}, c)P(yt​∣y1​,...,yt−1​,c)=解码器(yt−1​,ht−1dec​,c)

解码器的第一个输入通常是一个特殊的`<SOS>`令牌（y0=<SOS>y\_0 = \text{<SOS>}y0​=<SOS>）。

### PyTorch实现概述

我们来概述使用PyTorch `nn.LSTM`的简化编码器和解码器模块。

```python
import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True
        )

    def forward(self, input_seq):

        embedded = self.embedding(input_seq)

        outputs, (hidden, cell) = self.lstm(embedded)

        return hidden, cell

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True
        )
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_token, hidden, cell):

        embedded = self.embedding(input_token)

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        output = output.squeeze(1)
        output = self.out(output)

        return output, hidden, cell
```

### 局限性与后续步骤

使用循环神经网络 (neural network) (RNN)的标准编码器-解码器架构在许多任务中被证明是有效的。然而，它依赖于将*整个*输入序列压缩成一个单一的固定大小的上下文 (context)向量 (vector)。这会产生一个信息瓶颈，尤其对于长输入序列而言问题更突出。当模型生成输出序列的末尾时，它很难记住长输入开头部分的细节。

这一局限性是注意力机制 (attention mechanism)发展的重要推动力。注意力机制让解码器在输出生成过程的每一步，能够选择性地关注输入序列的不同部分，而不是仅仅依赖于单一的上下文向量。这种回顾源输入相关部分的能力大幅提升了机器翻译等任务的性能，并为我们将在下一章中介绍的Transformer架构打下了基础。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 4 Transformer Architecture

### 通过注意力机制解决循环问题

# 通过注意力机制解决循环问题

循环神经网络 (neural network)（RNN），包括长短期记忆网络（LSTM）和门控循环单元（GRU），都逐步处理序列。每个隐藏状态 hth\_tht​ 都是根据输入 xtx\_txt​ 和前一个隐藏状态 ht−1h\_{t-1}ht−1​ 计算得出。这种顺序处理方式，虽然在建模序列时很直观，但在处理大型语言模型所需的规模时，会带来显著的难题。

### 循环的瓶颈

核心局限直接源于这种顺序处理：

1. **并行化受限：** hth\_tht​ 的计算*必须*等待 ht−1h\_{t-1}ht−1​ 计算完成后才能进行。这种依赖性阻止了在单个训练示例中沿着时间维度（序列长度）进行计算的并行化。尽管可以在批次中的不同序列之间进行并行化，但每个序列*内部*的处理本质上仍是顺序的。对于很长的序列，这会成为主要的计算瓶颈，从而限制训练速度。想象一下处理一个有数千个词的文档；计算必须逐词进行。
2. **难以处理长距离依赖：** 序列早期部分的信息必须通过整个循环连接链才能影响后续部分的计算。尽管LSTM和GRU设计了门控机制以缓解梯度消失问题，但在很长的距离上保持准确的信息仍然很困难。两个相距较远的词元 (token) xix\_ixi​ 和 xjx\_jxj​ 之间的信息流动路径长度与 ∣i−j∣|i - j|∣i−j∣ 成正比。这意味着梯度在长路径上仍然可能消失或爆炸，使得模型难以学习序列中相距较远的词之间的关系。

考虑一个简化的RNN更新过程：

```python
import torch

input_seq = torch.randn(1, 5, 10)

h_prev = torch.zeros(1, 20)

rnn_cell = lambda input_t, h_prev: torch.tanh(
    input_t @ torch.randn(10, 20) + h_prev @ torch.randn(20, 20)
)

hidden_states = []

for t in range(input_seq.shape[1]):
    input_t = input_seq[:, t, :]
    h_t = rnn_cell(input_t, h_prev)
    hidden_states.append(h_t)
    h_prev = h_t
```

这个循环突显了顺序依赖性。我们无法在计算 `t=2` 对应的 `h_t` 之前计算 `t=3` 对应的 `h_t`。

### 引入注意力机制 (attention mechanism)：直接连接

Transformer架构通过完全去除循环，从根本上打破了这种顺序链条。它不再逐步传递信息，而是使用**注意力机制**。注意力机制的核心思想是让模型在处理序列中的一个元素（例如一个词）时，能够直接查看并从序列中的*所有*其他元素中获取信息。

想象您正在翻译句子“The cat sat on the mat”。当处理词语“sat”时，注意力机制使模型能够直接评估“The”、“cat”、“on”、“the”和“mat”的相关性，以理解“sat”的语境。它计算一组注意力分数，表示序列中其他每个词对当前词的重要性。然后，这些分数用于创建其他词表示的加权和，为“sat”提供一个语境感知的表示。

重要的是，这种“查看”过程不依赖于序列中词之间的距离。模型可以在“The”和“mat”之间建立直接连接，就像在“on”和“the”之间一样容易。序列中任意两个词元 (token)之间的路径长度变为常数，实际上是 O(1)O(1)O(1)，因为注意力机制直接计算它们之间的两两交互。这极大地简化了长距离依赖的学习，与RNN中序列长度为 nnn 时 O(n)O(n)O(n) 的路径长度相比。

### 实现并行化

通过消除顺序依赖性（hth\_tht​ 依赖于 ht−1h\_{t-1}ht−1​），Transformer实现了跨序列长度的大规模并行化。理论上，生成每个词元 (token)表示所需的计算可以同时进行。尽管单个词元的计算*内部*存在依赖性（例如，在应用注意力分数之前计算它们），但层内整个序列表示的总体计算可以比在RNN中更有效地并行化。这一特性对训练当前流行的大型模型非常重要。

从循环到注意力机制 (attention mechanism)的转变可以被形象化为从链式结构转向全连接图（在同一层内），其中每个词元都可以直接与其他所有词元交互。

G

clusterᵣnn

RNN处理

clusterₐttention

注意力机制

RNN\_X1

输入1

RNN\_H1

隐藏状态1

RNN\_X1->RNN\_H1

t=1

RNN\_H2

隐藏状态2

RNN\_H1->RNN\_H2

h(t-1)

RNN\_H3

隐藏状态3

RNN\_H2->RNN\_H3

h(t-1)

RNN\_X2

输入2

RNN\_X2->RNN\_H2

t=2

RNN\_X3

输入3

RNN\_X3->RNN\_H3

t=3

Att\_X1

输入1

Att\_Out1

输出1

Att\_X1->Att\_Out1

Att\_Out2

输出2

Att\_X1->Att\_Out2

Att\_Out3

输出3

Att\_X1->Att\_Out3

Att\_X2

输入2

Att\_X2->Att\_Out1

Att\_X2->Att\_Out2

Att\_X2->Att\_Out3

Att\_X3

输入3

Att\_X3->Att\_Out1

Att\_X3->Att\_Out2

Att\_X3->Att\_Out3

> 信息流比较。在RNN中（左图），信息顺序流动。在像Transformer这样的基于注意力机制的模型中（右图），每个输出可以同时直接关注所有输入。

这种从顺序处理到可并行注意力机制的根本性转变是Transformer架构在大型序列建模中如此成功的主要原因。以下章节将查看实现这一原理的特定机制，例如缩放点积注意力（Scaled Dot-Product Attention）和多头注意力（Multi-Head Attention）。

获取即时帮助、个性化解释和交互式代码示例。

---

### 缩放点积注意力

# 缩放点积注意力

Transformer 架构使用**注意力**作为其核心机制，取代了传统的序列处理方法。注意力机制 (attention mechanism)不依赖于逐步传递的隐藏状态，而是让模型在处理特定部分时，能够直接衡量输入序列不同部分的权重 (weight)。此项操作的基本单元是**缩放点积注意力**。

想象一下，你正在句子“The river bank was eroding.”（河岸正在被侵蚀）中理解“bank”这个词的含义。为了辨别“bank”的含义，你会很自然地更关注“river”（河流），而不是“eroding”（侵蚀）或“was”（是）。自注意力 (self-attention)机制将这种直觉形式化。对于序列中的每个元素（例如，每个词的嵌入 (embedding)向量 (vector)），我们希望计算一个表示。这个表示结合了来自其他元素的信息，并根据它们的关联性进行加权。

为了做到这一点，缩放点积注意力基于从输入序列嵌入中得到的三个输入进行操作：

1. **查询 (QQQ)**：一组向量，表示当前正在获取信息的元素。可以把它想象成元素在问：“序列的哪些部分与我有关联？”
2. **键 (KKK)**：一组与序列中每个元素关联的对应向量。它们就像元素的标签或标识符。通过将它们与查询进行比较来确定关联性。
3. **值 (VVV)**：另一组与每个元素关联的向量。这些向量包含元素的实际信息或表示。一旦关联性确定（通过查询-键交互），关联元素的“值”向量就会被聚合。

在实际操作中，QQQ、KKK和VVV通常是通过带有学习权重的独立线性层对输入嵌入（或前一层的输出）进行投影来生成的。设输入序列嵌入的维度为 dmodeld\_{model}dmodel​，键和查询向量的维度为 dkd\_kdk​。值向量的维度为 dvd\_vdv​（通常 dk=dvd\_k = d\_vdk​=dv​，但并非一定如此）。

计算过程分为几个步骤：

### 1. 计算相似度得分

第一步是衡量每个查询与所有键之间的兼容性或相似度。这通过点积完成。对于单个查询 qqq 和所有键 KKK，我们对每个键 kik\_iki​ 计算 q⋅kiq \cdot k\_iq⋅ki​。对于完整的矩阵 QQQ 和 KKK，这可以有效地通过矩阵乘法计算：

Scores=QKTScores = Q K^TScores=QKT

结果矩阵包含原始得分；ScoresijScores\_{ij}Scoresij​ 表示查询 iii 和键 jjj 之间的相似度。点积越高，表示查询和键之间的关联性越大。

### 2. 缩放得分

点积的数值可能变得很大，特别是对于较大的维度 dkd\_kdk​。输入到softmax函数（下一步）中的较大值可能导致梯度极小，从而阻碍学习。为解决此问题，得分会除以键维度的平方根，即 dk\sqrt{d\_k}dk​​ 进行缩放：

缩放得分=QKTdk缩放得分 = \frac{Q K^T}{\sqrt{d\_k}}缩放得分=dk​​QKT​

这种缩放有助于稳定梯度并使训练更具可靠性。选择 dk\sqrt{d\_k}dk​​ 是基于以下假设：QQQ 和 KKK 的分量是均值为零、方差为一的独立随机变量。在此假设下，点积 q⋅k=∑i=1dkqikiq \cdot k = \sum\_{i=1}^{d\_k} q\_i k\_iq⋅k=∑i=1dk​​qi​ki​ 的均值为 0，方差为 dkd\_kdk​。通过 dk\sqrt{d\_k}dk​​ 进行缩放可将方差恢复到 1，使 softmax 的输入保持在合理的范围内。

### 3. 计算注意力权重 (weight) (Softmax)

为了将缩放后的得分转换为表示注意力权重的概率分布，对缩放得分矩阵的每一行应用 softmax 函数：

注意力权重=softmax(QKTdk)注意力权重 = softmax(\frac{Q K^T}{\sqrt{d\_k}})注意力权重=softmax(dk​​QKT​)

`注意力权重`矩阵的每行现在总和为 1，并且每个元素 WeightsijWeights\_{ij}Weightsij​ 表明查询 iii 应该对值 jjj 投入多少注意力。

### 4. 计算加权值

最后，注意力权重 (weight)用于计算值向量 (vector)的加权和。这意味着将 `注意力权重` 矩阵乘以 `值` 矩阵 VVV：

输出=注意力权重⋅V输出 = 注意力权重 \cdot V输出=注意力权重⋅V

得到的 `输出` 矩阵包含注意力加权表示。每行 OutputiOutput\_iOutputi​ 是一个向量，它是 VVV 中所有值向量的加权组合，其中权重由查询 iii 与所有键的相似度决定。这个输出向量有效地融入了整个序列的上下文 (context)，并根据关联性进行了加权。

因此，缩放点积注意力的完整公式为：

注意力(Q,K,V)=softmax(QKTdk)V注意力(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d\_k}})V注意力(Q,K,V)=softmax(dk​​QKT​)V

我们来绘制数据流图：

G

clusterᵢnput

输入嵌入

clusterₚrojections

线性投影

clusterₐttention

缩放点积注意力

Input

输入
(序列长度 x dₘodel)

Q

查询 (Q)

Input->Q

 Wq

K

键 (K)

Input->K

 Wk

V

值 (V)

Input->V

 Wv

MatMul\_QK

矩阵乘法(Q, Kᵀ)

Q->MatMul\_QK

K->MatMul\_QK

MatMul\_AttnV

矩阵乘法(权重, V)

V->MatMul\_AttnV

Scale

缩放 ( / √dk )

MatMul\_QK->Scale

Softmax

Softmax

Scale->Softmax

Softmax->MatMul\_AttnV

权重

Output

输出
(序列长度 x dᵥ)

MatMul\_AttnV->Output

> 缩放点积注意力机制 (attention mechanism)的数据流图。输入嵌入 (embedding)被投影到Q、K、V矩阵中，然后经过矩阵乘法、缩放和softmax处理，生成加权输出表示。

以下是使用 PyTorch 实现的核心计算的简化示例：

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    计算缩放点积注意力。

    参数：
        query: 查询张量 (批量, 查询序列长度, 键维度)
        key: 键张量 (批量, 键值序列长度, 键维度)
        value: 值张量 (批量, 键值序列长度, 值维度)
        mask: 可选的掩码张量 (批量, 1, 查询序列长度, 键值序列长度)

    返回：
        输出张量 (批量, 查询序列长度, 值维度),
        注意力权重 (批量, 查询序列长度, 键值序列长度)
    """
    dim_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1))

    scaled_scores = scores / math.sqrt(dim_k)

    if mask is not None:

        scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scaled_scores, dim=-1)

    output = torch.matmul(attention_weights, value)

    return output, attention_weights

batch_size = 1
seq_len_q = 3
seq_len_kv = 5
dim_k = 8
dim_v = 10

q = torch.randn(batch_size, seq_len_q, dim_k)
k = torch.randn(batch_size, seq_len_kv, dim_k)
v = torch.randn(batch_size, seq_len_kv, dim_v)

output, weights = scaled_dot_product_attention(q, k, v)

print("输出形状:", output.shape)
print("注意力权重形状:", weights.shape)
```

在这段代码中：

- 我们执行矩阵乘法 QKTQ K^TQKT。
- 我们通过 `math.sqrt(dim_k)` 对结果进行缩放。
- 代码中包含一个可选的 `mask` 参数 (parameter)。掩码在Transformer中非常重要，例如，它们可以防止模型关注填充标记 (token)，或者在解码器中防止关注未来的标记（前瞻掩码）。我们通过将掩码位置设置为一个非常大的负数，在softmax *之前* 应用掩码，以确保它们在softmax后得到接近零的概率。
- `F.softmax` 计算注意力权重。
- 最后，我们将权重乘以 `值` 张量以获得输出。

这种机制构成了Transformer能够捕获序列中不限距离的依赖关系的核心能力，与RNN中固有的序列瓶颈相比，这是一个显著的优势。然而，单次注意力计算可能只关注一种类型的关系。为了同时捕获多种关系，Transformer采用了*多头注意力 (multi-head attention)*，我们将在接下来进行审视。

获取即时帮助、个性化解释和交互式代码示例。

---

### 多头注意力机制

# 多头注意力机制

缩放点积注意力机制 (attention mechanism)允许模型在处理某个特定标记 (token)时衡量不同标记的重要性，但它只使用一组学到的查询(QQQ)、键(KKK)和值(VVV)投影。这可能会限制模型同时识别多种关系或关注输入不同方面的能力。例如，一种注意力模式可能需要用来获取句法依赖关系，而另一种则侧重于长距离的语义相似性。

多头注意力 (multi-head attention)通过并行运行缩放点积注意力机制多次来解决这个问题，每次运行都使用自己学到的线性投影。每个并行运行被称为一个“注意力头”。这使得模型能够同时关注来自不同表示子空间、处于不同位置的信息。

### 机制

不同于执行一个使用维度为 dmodeld\_{model}dmodel​ 的键、值和查询的单一注意力函数，多头注意力 (multi-head attention)首先使用每个头不同的、学到的线性投影对查询、键和值进行 hhh 次线性投影。假设输入的查询、键和值是矩阵 QQQ、KKK和VVV（在自注意力 (self-attention)层中，它们通常是同一个张量）。对于每个头 i∈{1,...,h}i \in \{1, ..., h\}i∈{1,...,h}，我们计算：

头i=注意力(QWiQ,KWiK,VWiV)\text{头}\_i = \text{注意力}(QW^Q\_i, KW^K\_i, VW^V\_i)头i​=注意力(QWiQ​,KWiK​,VWiV​)

这里的投影是参数 (parameter)矩阵：
WiQ∈Rdmodel×dkW^Q\_i \in \mathbb{R}^{d\_{model} \times d\_k}WiQ​∈Rdmodel​×dk​
WiK∈Rdmodel×dkW^K\_i \in \mathbb{R}^{d\_{model} \times d\_k}WiK​∈Rdmodel​×dk​
WiV∈Rdmodel×dvW^V\_i \in \mathbb{R}^{d\_{model} \times d\_v}WiV​∈Rdmodel​×dv​

这里的 注意力\text{注意力}注意力 函数是前一节描述的缩放点积注意力。通常，每个头的维度设置为 dk=dv=dmodel/hd\_k = d\_v = d\_{model} / hdk​=dv​=dmodel​/h。这种划分确保了总计算成本与具有完整维度的单头注意力相似。

在并行计算每个头的注意力输出后，它们的输出（每个维度为 dvd\_vdv​）被拼接在一起：

拼接(头1,头2,...,头h)∈R序列长度×(h⋅dv)\text{拼接}(\text{头}\_1, \text{头}\_2, ..., \text{头}\_h) \in \mathbb{R}^{\text{序列长度} \times (h \cdot d\_v)}拼接(头1​,头2​,...,头h​)∈R序列长度×(h⋅dv​)

由于我们选择 dv=dmodel/hd\_v = d\_{model} / hdv​=dmodel​/h，拼接后的维度是 h⋅dv=dmodelh \cdot d\_v = d\_{model}h⋅dv​=dmodel​。这个拼接后的输出会经过一个最终的线性投影，其参数为 WO∈Rhdv×dmodelW^O \in \mathbb{R}^{h d\_v \times d\_{model}}WO∈Rhdv​×dmodel​（或 Rdmodel×dmodel\mathbb{R}^{d\_{model} \times d\_{model}}Rdmodel​×dmodel​），以生成多头注意力层的最终输出：

多头(Q,K,V)=拼接(头1,...,头h)WO\text{多头}(Q, K, V) = \text{拼接}(\text{头}\_1, ..., \text{头}\_h) W^O多头(Q,K,V)=拼接(头1​,...,头h​)WO

整个过程可以如下方所示：

G

Q

查询 (Q)

Proj1

线性投影
(W^Q₁, W^K₁, W^V₁)

Q->Proj1

ProjH

线性投影
(W^Qₕ, W^Kₕ, W^Vₕ)

Q->ProjH

K

键 (K)

K->Proj1

K->ProjH

V

值 (V)

V->Proj1

V->ProjH

Attn1

缩放点积
注意力 (头₁)

Proj1->Attn1

Dots
...

Concat

拼接

Attn1->Concat

AttnH

缩放点积
注意力 (头ₕ)

ProjH->AttnH

AttnH->Concat

FinalProj

线性投影
(Wᴼ)

Concat->FinalProj

Output

多头
输出

FinalProj->Output

> 输入的查询、键和值针对每个注意力头独立地进行线性投影。并行缩放点积注意力机制 (attention mechanism)的输出被拼接起来，然后经过一个最终的线性投影。

### PyTorch 中的实现概述

让我们看一下使用 PyTorch 的一个简化实现概述，以突出主要步骤。我们假设输入的张量 `query`、`key` 和 `value` 的形状为 `(batch_size, seq_len, d_model)`。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, (
            "d_model 必须能被 num_heads 整除"
        )

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(
        self, Q, K, V, mask=None
    ):

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )

        if mask is not None:

            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, V)

        return output

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(
            batch_size, -1, self.num_heads, self.d_k
        ).transpose(1, 2)
        K = K.view(
            batch_size, -1, self.num_heads, self.d_k
        ).transpose(1, 2)
        V = V.view(
            batch_size, -1, self.num_heads, self.d_k
        ).transpose(1, 2)

        attn_output = self.scaled_dot_product_attention(
            Q, K, V, mask
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.W_o(attn_output)

        return output
```

这个概述演示了输入的 Q、K、V 张量如何被投影并重塑，以实现跨头的并行计算。`transpose` 操作对于将头维度与批次维度分组非常重要，从而在 `scaled_dot_product_attention` 函数中实现高效的批量矩阵乘法。最后，输出被重塑回原形，并经过最终的输出投影 WOW^OWO。

### 优势

使用多个注意力头具有多项优势：

1. **多种表示：** 它使得模型能够同时处理来自不同表示子空间的信息。每个头可能学会关注不同类型的输入特征或关系（例如，短距离依赖与长距离依赖、句法信息与语义信息）。
2. **提升模型容量：** 将输入投影到多个低维子空间（hhh 次，维度为 dkd\_kdk​）然后将它们组合起来，相比于在完整 dmodeld\_{model}dmodel​ 维度上操作的单个注意力头，这赋予了注意力层更强的表示能力。
3. **稳定性：** 组合多个头所固有的平均效应可以使学习过程更稳定，并且对任何单个注意力头的输出不那么敏感。

多头注意力 (multi-head attention)不仅是原始 Transformer 中的核心组成部分，也几乎是所有后续大型语言模型中的核心组成部分。它提供了一种强大且计算上可管理的方式来增强基本的注意力机制 (attention mechanism)。这些头、残差连接和归一化 (normalization)层（接下来会提及）之间的配合构成了 Transformer 模块的核心。

获取即时帮助、个性化解释和交互式代码示例。

---

### 位置编码方法

# 位置编码方法

自注意力 (self-attention)机制 (attention mechanism)，正如我们所见，同时处理所有输入词元 (token)。尽管它在捕获与距离无关的依赖关系方面表现出色，但这种并行处理也有一个不足之处：标准的自注意力操作是排列不变的。如果你打乱输入词元，注意力输出（在添加位置信息之前）将只是原始输出的一个打乱版本。它本身不具备序列顺序的知识。“猫坐在垫子上”和“垫子坐在猫上”在自注意力层看来是完全一样的。显然，对于语言建模和大多数序列任务，顺序是必不可少的。我们需要一种方法将每个词元的位置信息融入模型中。这通过位置编码 (positional encoding)来实现。

位置编码通过创建一个向量 (vector)来实现，该向量表示词元在序列中的位置。这个向量随后被添加到对应词元的输入嵌入 (embedding)中。这个整合了的嵌入，现在既包含语义信息（来自词元嵌入）又包含位置信息，被输入到Transformer堆栈中。

最终输入嵌入=词元嵌入+位置编码\text{最终输入嵌入} = \text{词元嵌入} + \text{位置编码}最终输入嵌入=词元嵌入+位置编码

有多种方法可以生成这些位置编码向量。

### 学习到的位置嵌入 (embedding)

或许最直接的方法是像学习词元 (token)嵌入一样学习位置编码 (positional encoding)。我们可以定义一个最大序列长度，比如LmaxL\_{max}Lmax​，并创建一个大小为(Lmax,dmodel)(L\_{max}, d\_{\text{model}})(Lmax​,dmodel​)的嵌入矩阵，其中dmodeld\_{\text{model}}dmodel​是模型嵌入的维度。对于位置pospospos（其中0≤pos<Lmax0 \le pos < L\_{max}0≤pos<Lmax​）处的词元，我们只需在这个嵌入矩阵中查找第pospospos个向量 (vector)并将其添加到词元的嵌入中。

在PyTorch中，这可以使用`nn.Embedding`来实现：

```python
import torch
import torch.nn as nn

max_seq_len = 512
d_model = 768

positional_embedding_table = nn.Embedding(max_seq_len, d_model)

seq_len = 100
positions = torch.arange(0, seq_len,
                         dtype=torch.long).unsqueeze(0)
learned_pe = positional_embedding_table(positions)

print(f"学习到的位置嵌入的形状: {learned_pe.shape}")
```

这种方法简单，并让模型能够学习表示特定任务和数据位置的最佳方式。然而，它也有不足之处：

1. **参数 (parameter)数量：** 它会为模型增加Lmax×dmodelL\_{max} \times d\_{\text{model}}Lmax​×dmodel​个参数。对于非常大的模型和长序列来说，这会是很大的开销。
2. **外推能力：** 它通常无法很好地外推到训练时未见过的、长于LmaxL\_{max}Lmax​的序列长度。模型尚未学习到超过此限制的位置嵌入。

### 固定正弦位置编码 (positional encoding)

原始的Transformer论文（Vaswani等人，2017）提出了一种固定的、非学习的位置编码方法，该方法使用了不同频率的正弦和余弦函数。这样做的出发点是使用一个确定性函数，它可能使得模型更容易关注相对位置，因为对于任何固定偏移量kkk， PEpos+kPE\_{pos+k}PEpos+k​可以表示为PEposPE\_{pos}PEpos​的线性函数。它还避免了学习到的嵌入 (embedding)所带来的额外参数 (parameter)，并且可能对未见过的序列长度有更好的泛化能力。

位置编码PEPEPE的公式，对于位置pospospos和维度索引iii处的词元 (token)定义为：

PE(pos,2i)=sin⁡(pos/100002i/dmodel)PE\_{(pos, 2i)} = \sin(pos / 10000^{2i/d\_{\text{model}}})PE(pos,2i)​=sin(pos/100002i/dmodel​)
PE(pos,2i+1)=cos⁡(pos/100002i/dmodel)PE\_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d\_{\text{model}}})PE(pos,2i+1)​=cos(pos/100002i/dmodel​)

这里：

- pospospos是词元在序列中的位置（0, 1, 2, ...）。
- iii是嵌入向量 (vector)中的维度索引（0≤i<dmodel/20 \le i < d\_{\text{model}}/20≤i<dmodel​/2）。我们对偶数索引（2i2i2i）计算正弦，对奇数索引（2i+12i+12i+1）计算余弦。
- dmodeld\_{\text{model}}dmodel​是嵌入的维度。

位置编码的每个维度都对应一个正弦曲线。波长形成一个从2π2\pi2π到10000⋅2π10000 \cdot 2\pi10000⋅2π的等比数列。这种选择使得模型可能学习关注相对位置，因为相对位置信息编码在相位差中。

我们用PyTorch来实现它：

```python
import torch
import math
import matplotlib.pyplot as plt

def get_sinusoidal_positional_encoding(seq_len, d_model):
    """计算正弦位置编码。"""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(
        0, seq_len, dtype=torch.float
    ).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2).float()
        * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

seq_len = 100
d_model = 128
fixed_pe = get_sinusoidal_positional_encoding(seq_len, d_model)

print(f"固定位置嵌入的形状: {fixed_pe.shape}")

plt.figure(figsize=(10, 5))

for i in range(0, 8, 2):
    plt.plot(fixed_pe[:, i].numpy(), label=f'维度 {i} (sin)')

for i in range(1, 9, 2):
     plt.plot(
         fixed_pe[:, i].numpy(),
         label=f'维度 {i} (cos)',
         linestyle='--'
     )
plt.ylabel("值")
plt.xlabel("位置")
plt.title("正弦位置编码（前8个维度）")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
```

02468−1−0.500.51维度 0 (sin)维度 1 (cos)维度 2 (sin)维度 3 (cos)维度 4 (sin)维度 5 (cos)维度 6 (sin)维度 7 (cos)

> 前10个位置的正弦位置编码的前8个维度。请注意，较低维度（较小的iii）比高维度变化更快（频率更高）。

尽管正弦编码在原始Transformer和BERT等模型中有效且被广泛采用，但它们是固定的。它们可能不是所有类型序列模式的最佳表示。

### 选择合适的方法

学习到的位置编码 (positional encoding)和固定的正弦位置编码都是常见的起始点。

- **学习式：** 最初实现起来更简单，如果使用足够的数据和适当的最大长度，可能更具适应性。常用于BERT等模型。
- **固定正弦式：** 没有额外参数 (parameter)，潜在地具有更好的长度泛化能力，确定性。用于原始Transformer以及GPT-2/3等模型。

实际应用中，选择可能取决于具体的应用、模型大小和序列长度需求。此外也值得注意的是，这个领域已经发展出了更高级的方法。绝对位置编码，无论是学习式的还是固定的，主要编码词元 (token)相对于序列起点的绝对位置。然而，对注意力而言，词元间的*相对*位置往往才是最重要的。例如相对位置编码和旋转位置嵌入 (embedding)（RoPE）等方法，直接将相对距离信息融入注意力机制 (attention mechanism)本身。这些更高级的方法在第13章中有介绍。

目前，理解学习到的位置编码和正弦编码，为Transformer如何融入序列顺序信息提供了必要的前提，从而克服了核心自注意力 (self-attention)机制的排列不变性。这种位置数据注入是一个简单但必不可少的要素，使得Transformer在处理序列数据上获得成功。

获取即时帮助、个性化解释和交互式代码示例。

---

### 编码器与解码器堆叠

# 编码器与解码器堆叠

编码器和解码器堆叠是Transformer架构的主要组成部分，它们通过组合注意力机制 (attention mechanism)和位置编码 (positional encoding)构建而成。原始的Transformer模型为编码器和解码器都提出了N=6个相同层的堆叠，尽管现代的大型语言模型通常采用更多的层。

## 编码器堆叠

编码器的作用是处理输入序列并生成一系列包含丰富上下文 (context)信息的表示。编码器堆叠中的每个层接收一系列嵌入 (embedding)向量 (vector)（或来自上一层的输出）并对其进行变换。一个单独的编码器层由两个主要子层构成：

1. **多头自注意力 (self-attention):** 该机制允许输入序列中的每个位置关注*上一层输出*序列中的所有位置（包括自身）。它根据序列不同部分之间的关联程度计算加权表示。
2. **逐位置前馈网络 (FFN):** 这是一个简单、全连接的前馈网络，独立应用于每个位置。它通常由两个线性变换和一个非线性激活函数 (activation function)（通常是ReLU或GeLU）组成。其目的是进一步处理注意力机制 (attention mechanism)生成的表示。

重要的是，在两个子层周围都使用了残差连接，随后是层归一化 (normalization)。如果 xxx 是子层（例如，多头注意力 (multi-head attention)或FFN）的输入，并且 子层(x)\text{子层}(x)子层(x) 是子层自身实现的函数，则子层块的输出是 层归一化(x+子层(x))\text{层归一化}(x + \text{子层}(x))层归一化(x+子层(x))。这种结构通过促进梯度流动和稳定激活，有助于训练更深的模型。

EncoderLayer

Input

输入 (来自上一层或嵌入)

MHA

多头
自注意力

Input->MHA

AddNorm1

加和归一化

Input->AddNorm1

+

MHA->AddNorm1

FFN

逐位置
前馈网络

AddNorm1->FFN

AddNorm2

加和归一化

AddNorm1->AddNorm2

+

FFN->AddNorm2

Output

输出 (到下一层或解码器)

AddNorm2->Output

> 单个Transformer编码器层内部的流程。

一个编码器层的输出作为下一个相同编码器层的输入。这种堆叠方式使得模型能够逐步构建输入序列的更复杂表示，在不同层面捕获依赖关系。

以下是编码器层结构的一个简化PyTorch表示：

```python
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):

        attn_output, _ = self.self_attn(
            src,
            src,
            src,
            key_padding_mask=src_mask,
            need_weights=False
        )

        src = self.norm1(src + self.dropout(attn_output))

        ff_output = self.feed_forward(src)

        src = self.norm2(src + self.dropout(ff_output))

        return src

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src
```

整个编码器堆叠的最终输出是一系列向量，每个输入令牌对应一个，旨在捕获该令牌在序列中的上下文含义。此输出通常由解码器使用。

## 解码器堆叠

解码器的作用通常是基于编码后的输入序列和迄今已生成的令牌，一次生成一个令牌的输出序列。与编码器类似，解码器也由N个相同的层堆叠构成。然而，每个解码器层有*三个*主要子层：

1. **带掩码的多头自注意力 (self-attention):** 这与编码器的自注意力运作方式相似，但有一个重要区别：它包含掩码机制。该掩码确保在预测位置 iii 的输出令牌时，自注意力机制 (attention mechanism)只能关注解码器输入序列中位置小于或等于 iii 的部分。它不能“向前看”未来的令牌，从而保留了生成所需的自回归 (autoregressive)属性。
2. **多头编码器-解码器注意力:** 这是解码器与编码器输出交互的地方。查询 (QQQ) 来自前一个解码器子层（即带掩码的自注意力）的输出。键 (KKK) 和值 (VVV) 来自*编码器堆叠*的最终输出。这使得解码器中的每个位置都能关注输入序列中的所有位置，整合来自源端的关联上下文 (context)。
3. **逐位置前馈网络 (FFN):** 结构与编码器层中使用的相同，在编码器-解码器注意力之后独立应用于每个位置。

与编码器类似，在这三个子层之后都应用了残差连接和层归一化 (normalization)： 层归一化(x+子层(x))\text{层归一化}(x + \text{子层}(x))层归一化(x+子层(x))。

DecoderLayer

Input

输入 (来自上一层或目标嵌入)

MaskedMHA

带掩码多头
自注意力

Input->MaskedMHA

AddNorm1

加和归一化

Input->AddNorm1

+

EncoderOutput

编码器输出

EncDecMHA

编码器-解码器
多头注意力

EncoderOutput->EncDecMHA

K, V

MaskedMHA->AddNorm1

AddNorm1->EncDecMHA

Q

AddNorm2

加和归一化

AddNorm1->AddNorm2

+

EncDecMHA->AddNorm2

FFN

逐位置
前馈网络

AddNorm2->FFN

AddNorm3

加和归一化

AddNorm2->AddNorm3

+

FFN->AddNorm3

Output

输出 (到下一层或最终线性层)

AddNorm3->Output

> 单个Transformer解码器层内部的流程，突出了三个子层和输入。

解码器层的一个简化PyTorch结构如下：

```python
import torch
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.encoder_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):

        attn_output, _ = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            need_weights=False
        )

        tgt = self.norm1(tgt + self.dropout(attn_output))

        attn_output, _ = self.encoder_attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_mask,
            need_weights=False
        )

        tgt = self.norm2(tgt + self.dropout(attn_output))

        ff_output = self.feed_forward(tgt)

        tgt = self.norm3(tgt + self.dropout(ff_output))

        return tgt

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return tgt
```

## 最终输出生成

输入通过整个解码器堆叠后，产生的向量 (vector)序列表示预测的输出令牌。为了将这些向量转换为目标词汇表 (vocabulary)上的概率，通常应用一个最终线性层，然后是softmax函数。线性层将解码器输出向量（维度为 dmodeld\_{model}dmodel​）投射到词汇表大小 (VVV) 的空间。softmax函数将这些得分（logits）转换为概率，表示词汇表中每个词成为序列中下一个令牌的可能性。

P(yi∣y<i,x)=softmax(线性(解码器输出i))P(y\_i | y\_{<i}, x) = \text{softmax}(\text{线性}(\text{解码器输出}\_i))P(yi​∣y<i​,x)=softmax(线性(解码器输出i​))

编码器堆叠（处理输入）与解码器堆叠（根据输入和之前输出生成输出）的组合构成了完整的Transformer架构，能够处理各种序列到序列的任务。理解这些堆叠如何由注意力层、前馈层以及归一化 (normalization)和残差连接构成，对于构建和修改这些高效模型非常重要。

获取即时帮助、个性化解释和交互式代码示例。

---

### 层归一化与残差连接的作用

# 层归一化与残差连接的作用

构建多层神经网络 (neural network)时，像Transformer中堆叠多层自注意力 (self-attention)子层和前馈子层那样，训练过程中会面临挑战。主要是，梯度在通过多层向后传播时可能消失（变得非常小）或爆炸（变得非常大），这使得模型难以有效学习。此外，中间层的激活分布在训练期间可能变化（这种现象有时与内部协变量偏移有关），使学习过程变得复杂。在Transformer架构中，残差连接和层归一化 (normalization)是两种简单但非常有效的方法，它们用于解决这些问题，并使多层模型得以训练。

### 残差连接

残差连接，也称作跳跃连接，提供了一条备用通路供梯度流经网络。不是简单地将子层的输出传递给下一层，而是我们将子层的*输入*加到其*输出*上。

如果一个子层由函数 SubLayer(⋅)\text{SubLayer}(\cdot)SubLayer(⋅) 表示，其输入为 xxx，那么*带有*残差连接的块的输出是：

输出=x+子层(x)\text{输出} = x + \text{子层}(x)输出=x+子层(x)

这种结构使得网络能够轻松地学习恒等函数，如果某个特定子层没有益处；子层的输出可以简单地趋近于零。更重要的是，在反向传播 (backpropagation)时，梯度可以直接通过加法操作从输出流回到输入 xxx。这绕过了子层内部的变换，提供了一条“捷径”，有助于防止梯度信号在穿过多层时过度减弱。

G

x

输入 (x)

sublayer

子层(x)

x->sublayer

add

+

x->add

sublayer->add

output

输出

add->output

> 残差连接将输入 `x` 加到 `子层` 的输出上。

在PyTorch中，这实现起来很简单。假设 `sublayer` 是一个模块（如多头注意力 (multi-head attention)或前馈网络），`x` 是输入张量：

```python
import torch
import torch.nn as nn

class ResidualConnection(nn.Module):
    def __init__(self, sublayer):
        super().__init__()
        self.sublayer = sublayer

    def forward(self, x):
        """
        对任意子层应用残差连接。
        """

        return x + self.sublayer(x)
```

### 层归一化 (normalization)

归一化技术有助于稳定训练过程，通过控制激活的分布。尽管批量归一化在计算机视觉中很常见，但它是在批次维度上归一化统计量（均值和方差）。这对于序列模型来说可能存在问题，因为批次内的序列长度可能不同，并且它引入了批次元素之间不总是期望的依赖关系。

层归一化（LayerNorm）提供了一种替代方法。它独立地*跨特征*对每个数据点（例如，序列中的每个词元 (token)）的输入进行归一化。它根据单个训练样本中单个层内所有神经元的求和输入，计算用于归一化的均值和方差。

给定输入向量 (vector) xxx（表示单个词元位置在所有特征上的激活，dmodeld\_{model}dmodel​），层归一化计算归一化输出 hhh 如下：

1. 计算跨特征维度（dmodeld\_{model}dmodel​）的均值（μ\muμ）和方差（σ2\sigma^2σ2）：

   μ=1dmodel∑i=1dmodelxi\mu = \frac{1}{d\_{model}} \sum\_{i=1}^{d\_{model}} x\_iμ=dmodel​1​i=1∑dmodel​​xi​
   σ2=1dmodel∑i=1dmodel(xi−μ)2\sigma^2 = \frac{1}{d\_{model}} \sum\_{i=1}^{d\_{model}} (x\_i - \mu)^2σ2=dmodel​1​i=1∑dmodel​​(xi​−μ)2
2. 归一化输入 xxx：

   x^i=xi−μσ2+ϵ\hat{x}\_i = \frac{x\_i - \mu}{\sqrt{\sigma^2 + \epsilon}}x^i​=σ2+ϵ​xi​−μ​

   其中 ϵ\epsilonϵ 是为数值稳定性添加的一个小常数。
3. 使用可学习参数 (parameter) γ\gammaγ（伽马，缩放）和 β\betaβ（贝塔，平移）来缩放和移动归一化后的输出，这些参数与 xxx 具有相同的维度：

   hi=γix^i+βih\_i = \gamma\_i \hat{x}\_i + \beta\_ihi​=γi​x^i​+βi​

这些可学习参数 γ\gammaγ 和 β\betaβ 使得网络能够自适应地缩放和移动归一化后的激活，如果这对网络来说最优，甚至可能恢复原始激活。层归一化有助于稳定隐藏状态的动态，减少对初始化尺度的敏感性，甚至可以提供轻微的正则化 (regularization)作用。

在PyTorch中，`torch.nn.LayerNorm` 实现了这一点：

```python
import torch
import torch.nn as nn

batch_size = 4
seq_len = 10
d_model = 512
epsilon = 1e-5

input_tensor = torch.randn(batch_size, seq_len, d_model)

layer_norm = nn.LayerNorm(d_model, eps=epsilon)

normalized_output = layer_norm(input_tensor)

print("输入形状:", input_tensor.shape)
print("输出形状:", normalized_output.shape)

print(
    "\n归一化输出的均值（示例 0，词元 0）:",
    normalized_output[0, 0, :].mean().item()
)
print(
    "归一化输出的标准差（示例 0，词元 0）:",
    normalized_output[0, 0, :].std().item()
)

print("\nLayerNorm 可学习伽马（权重）:", layer_norm.weight.shape)
print("LayerNorm 可学习贝塔（偏置）:", layer_norm.bias.shape)
```

### 结合残差连接和层归一化 (normalization)

在Transformer架构中，层归一化和残差连接通常围绕每个子层（多头注意力 (multi-head attention)和逐位置前馈网络）一起应用。原始论文《Attention Is All You Need》中描述的标准结构是在残差加法*之后*应用归一化（Post-LN）：

输出=LayerNorm(x+子层(x))\text{输出} = \text{LayerNorm}(x + \text{子层}(x))输出=LayerNorm(x+子层(x))

然而，随后的研究和实践经常发现，在残差分支中，将层归一化应用在子层*之前*（Pre-LN）可以带来更稳定的训练，特别是对于非常深的Transformer模型：

输出=x+子层(LayerNorm(x))\text{输出} = x + \text{子层}(\text{LayerNorm}(x))输出=x+子层(LayerNorm(x))

我们将在第11章讨论扩展定律和架构选择时，进一步探讨Pre-LN与Post-LN的影响。目前，请认识到这种结合（通常被描述为“添加与归一化”步骤）是基础的。

G

clusterₚostₗn

Post-LN 结构

clusterₚreₗn

Pre-LN 结构

xₚost

输入 (x)

sublayerₚost

子层(x)

xₚost->sublayerₚost

addₚost

+

xₚost->addₚost

sublayerₚost->addₚost

normₚost

LayerNorm

addₚost->normₚost

outputₚost

输出

normₚost->outputₚost

xₚre

输入 (x)

normₚre

LayerNorm

xₚre->normₚre

addₚre

+

xₚre->addₚre

sublayerₚre

子层(...)

normₚre->sublayerₚre

sublayerₚre->addₚre

outputₚre

输出

addₚre->outputₚre

> 残差块中Post-LN（加法后归一化）和Pre-LN（子层前归一化）结构的比较。

以下是Transformer编码器层通常如何使用Pre-LN方法结合这些组件的示例：

```python
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    """
    实现一个带有Pre-LN的Transformer编码器层。
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        norm_x = self.norm1(x)

        attn_output = self.self_attn(norm_x, norm_x, norm_x, mask)

        x = x + self.dropout1(attn_output)

        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)

        x = x + self.dropout2(ff_output)

        return x
```

总而言之，残差连接有助于梯度流动和信息在深层网络中的传播，而层归一化则稳定了激活分布。它们的结合使用是成功训练深层Transformer模型的重要因素，构成了编码器和解码器层的核心部分。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 5 Tokenization Large Vocabularies

### 子词分词的必要性

# 子词分词的必要性

将原始文本转换为适合神经网络 (neural network)的格式，具体来说是数值ID序列，这是非常基础的第一步。此过程称为分词 (tokenization)。一种直接的方法可能是根据空格和标点符号分割文本，并为训练语料库中遇到的每个不同词汇分配一个唯一的整数ID。

考虑一个简单的句子：“LLMs learn representations.”
一个词级别分词器 (tokenizer)可能产生：`["LLMs", "learn", "representations", "."]`
如果我们的词汇表 (vocabulary)将“LLMs”映射到5，“learn”映射到123，“representations”映射到456，而“.”映射到7，则数值序列将是`[5, 123, 456, 7]`。

这种词级别方法虽然简单直观，但在处理训练大型语言模型所需的海量数据集时，会迅速出现问题。一些重要问题：

### 词汇表 (vocabulary)规模急剧增长

主要问题是所需词汇表的庞大规模。网络规模的文本语料库不仅包含标准字典词汇，还包括名称、地点、技术术语、代码片段、俚语、拼写错误以及形态变体（例如，“run”、“runs”、“running”、“ran”）。从此类数据构建的词级别词汇表很容易膨胀到包含数百万甚至数千万个独特的类型。

这带来了严峻的实际挑战：

1. **内存占用：** 模型的输入嵌入 (embedding)层将每个词汇ID映射到一个稠密向量 (vector)（例如，1024维或更多）。1000万词的词汇表需要一个 10,000,000×dmodel10,000,000 \times d\_{model}10,000,000×dmodel​ 参数 (parameter)的嵌入矩阵。即使 dmodel=1024d\_{model}=1024dmodel​=1024 规模不大，仅此一层也需要大约41GB的内存来以标准32位浮点格式存储权重 (weight)。

   ```python
   import torch
   import torch.nn as nn

   d_model = 1024

   word_vocab_size = 10_000_000

   word_memory_bytes = word_vocab_size * d_model * 4
   word_memory_gb = word_memory_bytes / (1024**3)
   print(
       f"Word Embedding Memory Estimate (Vocab={word_vocab_size:,}, "
       f"d_model={d_model}): {word_memory_gb:.2f} GB"
   )

   subword_vocab_size = 50_000
   subword_embeddings = nn.Embedding(subword_vocab_size, d_model)
   subword_memory_bytes = subword_vocab_size * d_model * 4
   subword_memory_gb = subword_memory_bytes / (1024**3)
   print(
       f"Subword Embedding Memory Estimate (Vocab={subword_vocab_size:,}, "
       f"d_model={d_model}): {subword_memory_gb:.2f} GB"
   )
   ```
2. **计算成本：** 典型语言模型的最后一层通常涉及对整个词汇表进行softmax计算，以预测下一个词元 (token)。此操作的复杂度与词汇表大小成比例，即 O(∣V∣)O(|V|)O(∣V∣)。数百万词的词汇表使得这一最后步骤在训练和推理 (inference)过程中计算成本高昂。
3. **参数效率低下：** 语言遵循齐普夫定律等分布，其中少数词出现频率很高，而绝大多数词极其罕见（即“长尾”现象）。词级别模型甚至为在大型语料库中只出现一两次的词分配唯一的嵌入向量和参数，这是对模型能力的低效配置。

### 词汇外（OOV）词汇

无论训练语料库有多大，你在推理 (inference)或评估期间都将不可避免地遇到训练数据中不存在的词。这些是词汇外（OOV）词汇。词级别分词 (tokenization)器 (tokenizer)必须有一种处理这些词的策略，通常是将所有未知词映射到一个特殊的单一词元 (token)，通常表示为 `<UNK>` 或 `[UNK]`。

考虑句子：“We analyzed the giga-scale dataset using GloVe embeddings.”
如果“giga-scale”不在训练词汇表 (vocabulary)中，词分词器可能产生：`["We", "analyzed", "the", "<UNK>", "dataset", "using", "GloVe", "embeddings", "."]`

用 `<UNK>` 替换词汇会导致大量信息丢失。模型没有关于特定未知词的信息，这阻碍了其理解或生成包含新术语、名称、拼写错误或领域特定词汇文本的能力。当模型应用于与其训练数据不同的范围时，OOV词汇的频率会增加。

~1M+~50k~300~5%<0.1%WordSubwordCharacter100100010k100k1M0246词汇表规模OOV率 (%)分词：词汇表规模与OOV率对比

> 大型文本语料库上不同分词级别的典型词汇表规模（对数刻度）和词汇外（OOV）率的对比。词级别分词产生大型词汇表和较高的OOV率。字符级别没有OOV但单元最小。子词 (subword)方法提供了一种平衡。

### 形态复杂性

语言通过形态学构成词汇，将词根与前缀、后缀和屈折变化结合。例如，“train”、“trainer”、“training”、“retrain”、“trained”都共享词根“train”。词级别分词 (tokenization)器 (tokenizer)将每个词视为一个完全独立的单元，拥有自己的ID和嵌入 (embedding)向量 (vector)。这是低效的，因为它未能借助这些相关形式之间固有的关联。模型必须完全根据共现统计从头学习“training”和“trained”之间的关联，而不是识别共享的底层词素。这个问题在形态丰富的语言（如土耳其语、芬兰语或德语）中尤其突出，但在英语中也很重要。

### 子词 (subword)解决方案

为应对这些挑战，现代大型语言模型几乎普遍采用**子词分词 (tokenization)**算法。核心思想是将词汇分解为更小、频繁出现的单元。这些单元可能是词素（如“train”、“ing”、“er”），常见的字符序列，甚至是用于最罕见部分的单个字符。

例如，“tokenization”这个词可能被分解为子词单元，如 `["token", "##ization"]`，其中“##”表示词内连续部分的开始（此标记 (token)法有所不同）。像“giga-scale”这样的罕见词可能变为 `["giga", "-", "scale"]` 或者 `["g", "##iga", "-", "s", "##cale"]`，具体取决于学习到的词汇表 (vocabulary)。

这种方法巧妙地缓解了词级别分词的问题：

- **可控的词汇表规模：** 词汇表不再包含数百万词汇，而是由数万（例如3万到10万）个子词单元组成。这大大降低了嵌入 (embedding)层和softmax层的内存和计算开销。
- **消除OOV：** 任何新词或罕见词几乎总可以通过组合已知的子词单元来构成。如果一个词包含真正新颖的字符序列，它可以被进一步分解，甚至分解为单个字符，这些字符必定在词汇表中。 `<UNK>` 词元 (token)变得很大程度上不再必要。
- **处理形态学：** 形态相关的词汇自然共享常见的子词单元（例如，“training”和“trainer”中的“train”）。这使得模型能够潜在地共享表示，并更有效地学习词形之间的语义关联 (semantic relationship)。序列 `["train", "##ing"]` 通过共享的“train”子词元，固有地与 `["train", "##ed"]` 相关联。

通过在子词级别操作，我们实现了平衡：词汇表保持可管理规模，OOV问题几乎被消除，模型获得了更好地理解词汇结构的能力。那么问题就来了：我们如何确定给定语料库的最佳子词单元集？字节对编码（BPE）、WordPiece和Unigram语言模型分割等技术，常在SentencePiece等框架中使用，提供了构建这些子词词汇表的数据驱动方法，我们将在后续章节中介绍这些内容。

获取即时帮助、个性化解释和交互式代码示例。

---

### 字节对编码 (BPE) 算法

# 字节对编码 (BPE) 算法

标准的词级分词 (tokenization)方法在处理用于训练大型语言模型的庞大且多样的文本语料库时遇到不少难题。主要问题在于处理词汇表 (vocabulary)外 (OOV) 词汇以及词汇表可能非常庞大。字节对编码 (BPE) 为应对这些难题提供了一种有效的数据驱动型方法。BPE 最初是一种数据压缩算法，后来成功调整用于文本分词，构建了子词 (subword)单元词汇表。

BPE 的核心思想十分简洁：它以训练语料库中所有单个字符构成的词汇表为起始，迭代合并出现频率最高的相邻符号对，形成新的、更长的子词符号。这个过程会持续预设的合并次数，有效地控制最终词汇表的大小。

### BPE 训练过程

让我们仔细看看 BPE 算法如何从语料库中学习其词汇表 (vocabulary)和合并规则。

1. **初始化**：

   - 设定目标最终词汇表大小 VtargetV\_{target}Vtarget​。
   - 将训练语料库中的每个词表示为字符序列，并添加一个特殊的词尾符号（通常是 `</w>` 或 `</w>`) 以区分词 (tokenization)边界。例如，“lower” 变成 `l`、`o`、`w`、`e`、`r`、`</w>`。
   - 初始词汇表包含语料库中所有独有的字符。
2. **迭代**：重复以下步骤，直至达到目标词汇表大小 VtargetV\_{target}Vtarget​ 或已执行预设数量的合并操作：

   - **统计词对**：在语料库当前表示中，找出所有相邻符号对（字符或已合并的子词 (subword)），并统计它们的频率。
   - **找出最频繁词对**：选择出现频率最高的词对 (s1,s2)(s\_1, s\_2)(s1​,s2​)。
   - **合并**：创建一个新符号 s12s\_{12}s12​ 来表示合并后的词对。将 s12s\_{12}s12​ 添加到词汇表。
   - **更新语料库表示**：将语料库表示中所有相邻词对 (s1,s2)(s\_1, s\_2)(s1​,s2​) 的出现替换为新符号 s12s\_{12}s12​。记录此合并操作（例如：“将 s1s\_1s1​ 和 s2s\_2s2​ 合并为 s12s\_{12}s12​”）。

### 一个小示例

假设有一个小语料库，词频如下：`{'low': 5, 'lower': 2, 'newest': 6, 'widest': 3}`。

**步骤 0: 初始化**

- 将词表示为字符序列 + `</w>`：
  - `low`：`l o w </w>` (出现 5 次)
  - `lower`：`l o w e r </w>` (出现 2 次)
  - `newest`：`n e w e s t </w>` (出现 6 次)
  - `widest`：`w i d e s t </w>` (出现 3 次)
- 初始词汇表 (vocabulary)：`{l, o, w, </w>, e, r, n, s, t, i, d}`
- 目标词汇表大小：为演示目的，我们进行几次合并。

**步骤 1: 统计词对并合并**

- 统计整个语料库中的相邻词对（考虑频率）：
  - `(l, o)`：5 + 2 = 7
  - `(o, w)`：5 + 2 = 7
  - `(w, </w>)`：5
  - `(w, e)`：2 + 6 = 8
  - `(e, r)`：2
  - `(r, </w>)`：2
  - `(n, e)`：6
  - `(e, w)`：6
  - `(e, s)`：6 + 3 = 9 <- 最频繁
  - `(s, t)`：6 + 3 = 9 <- 频率相同（我们选择 `(e, s)`）
  - `(t, </w>)`：6 + 3 = 9 <- 频率相同
  - `(w, i)`：3
  - `(i, d)`：3
  - `(d, e)`：3
- 将 `(e, s)` 合并成新符号 `es`。词汇表大小增加。
- 更新语料库：
  - `l o w </w>` (5)
  - `l o w e r </w>` (2)
  - `n e w es t </w>` (6)
  - `w i d es t </w>` (3)
- 记录合并：`e + s -> es`

**步骤 2: 统计词对并合并**

- 在更新后的语料库中重新计算词对频率：
  - `(l, o)`：7
  - `(o, w)`：7
  - `(w, </w>)`：5
  - `(w, e)`：2 + 6 = 8
  - `(e, r)`：2
  - `(r, </w>)`：2
  - `(n, e)`：6
  - `(e, w)`：6
  - `(w, es)`：6
  - `(es, t)`：6 + 3 = 9 <- 最频繁（或与 `(t, </w>)` 频率相同）
  - `(t, </w>)`：6 + 3 = 9
  - `(w, i)`：3
  - `(i, d)`：3
  - `(d, es)`：3
- 将 `(es, t)` 合并成新符号 `est`。
- 更新语料库：
  - `l o w </w>` (5)
  - `l o w e r </w>` (2)
  - `n e w est </w>` (6)
  - `w i d est </w>` (3)
- 记录合并：`es + t -> est`

**步骤 3: 统计词对并合并**

- 重新计算频率：
  - ...
  - `(est, </w>)`：6 + 3 = 9 <- 最频繁（如果 `est` 在第 2 步中未被选中，则与 `(t, </w>)` 频率相同）
  - ...
- 将 `(est, </w>)` 合并成 `est</w>`。
- 更新语料库：
  - `l o w </w>` (5)
  - `l o w e r </w>` (2)
  - `n e w est</w>` (6)
  - `w i d est</w>` (3)
- 记录合并：`est + </w> -> est</w>`

此过程持续进行。如果我们执行更多合并，像 `(l, o)`、`(o, w)`、`(w, e)` 这样的词对也可能被合并，潜在地形成 `low` 或 `we`。最终词汇表将包含单个字符和常见的多个字符子词 (subword)，例如 `es`、`est`、`est</w>` 等，这些都是根据它们在训练数据中的频率形成的。

### 使用已学习的 BPE 分词 (tokenization)新文本

一旦从训练语料库中学习到 BPE 词汇表 (vocabulary)和有序的合并操作列表，对新文本进行分词涉及以下步骤：

1. 将输入词拆分为字符序列。添加词尾符号 `</w>`。
2. 迭代地应用在训练期间学习到的合并操作，*顺序相同*。对于每个学习到的合并 (s1,s2)→s12(s\_1, s\_2) \rightarrow s\_{12}(s1​,s2​)→s12​，在当前序列中找出所有相邻的 (s1,s2)(s\_1, s\_2)(s1​,s2​) 出现，并用 s12s\_{12}s12​ 替换。
3. 继续此过程，直到没有更多学习到的合并可以应用于序列。
4. 生成的符号序列（字符和子词 (subword)）即为分词表示。每个符号对应最终词汇表中的一个 ID。

例如，如果我们学习到合并规则 `e + s -> es`，然后 `es + t -> est`，对“tests”进行分词的步骤如下：

1. 初始：`t`、`e`、`s`、`t`、`s`、`</w>`
2. 应用 `e + s -> es`：`t`、`es`、`t`、`s`、`</w>`
3. 应用 `es + t -> est`：不存在相邻的 `es`、`t` 词对。
4. 最终分词结果：`t`、`es`、`t`、`s`、`</w>`

如果在 `est` *之后* 也学习到 `t + s -> ts`，那么它现在可能适用：`t`、`es`、`ts`、`</w>`。合并的顺序很重要。

### 处理未知词

BPE 的一个显著优点是其固有的能力，可以处理训练期间未曾出现的词汇（OOV 词汇）。由于初始词汇表 (vocabulary)包含所有单个字符，任何词汇在必要时都可以分解为字符序列。如果词汇的部分对应于学习到的子词 (subword)，则会使用这些子词；否则，它会退回到单个字符。例如，如果“huggingface”不在训练数据中，但“hugg”、“ing”、“face”是已学习的子词（或可以通过合并形成），它可能会被分词 (tokenization)为 `hugg`、`ing`、`face`、`</w>`。如果不是，它可能会变成 `h`、`u`、`g`、`g`、`i`、`n`、`g`、`f`、`a`、`c`、`e`、`</w>`。从需要专用 `[UNK]` 标记 (token)的意义上讲，BPE 不存在真正的“未知”标记，尽管为了其他目的可能仍会包含一个。

### 实现说明

- **字节级 BPE**：一种常见变体在字节而非字符上操作。这特别有用，因为它保证了固定的初始词汇表 (vocabulary)大小（256 字节），并且自然地处理所有 Unicode 字符，无需特殊处理重音或未见字符。其逻辑保持不变，只是合并频繁的字节对。
- **库**：高效地实现 BPE 需要仔细处理数据结构和计数。幸运的是，Hugging Face 的 `tokenizers` 等库提供了优化的实现。训练 BPE 分词 (tokenization)器 (tokenizer)可能如下所示（伪代码）：

```python

corpus = ["low low low low low", "lower lower", ...]
word_counts = get_word_counts(corpus)
vocab = initialize_with_characters(word_counts)
splits = initial_split_words(word_counts)
num_merges = target_vocab_size - len(vocab)
merges = {}

for i in range(num_merges):
  pair_counts = count_adjacent_pairs(splits)
  if not pair_counts:
    break
  most_frequent_pair = find_most_frequent(pair_counts)
  new_token = merge_pair(most_frequent_pair)
  vocab.add(new_token)
  merges[most_frequent_pair] = new_token
  splits = apply_merge_to_splits(splits, most_frequent_pair, new_token)
```

实际使用中，通常会利用成熟的库。下面是如何在 PyTorch 环境中使用 Hugging Face `transformers` 库的预训练 (pre-training) BPE 分词器（如 GPT-2 的）的示例：

```python
import torch
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "lower newest widest"
encoded_input = tokenizer(text, return_tensors='pt')

print("输入文本:", text)
print("分词ID:", encoded_input['input_ids'])

print(
    "解码分词:",
    tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
)

text_oov = "supercalifragilisticexpialidocious"
encoded_oov = tokenizer(text_oov, return_tensors='pt')
print("\n类OOV文本:", text_oov)
print("分词ID:", encoded_oov['input_ids'])
print(
    "解码分词:",
    tokenizer.convert_ids_to_tokens(encoded_oov['input_ids'][0])
)
```

这段代码演示了训练后的 BPE 分词器如何将文本分段为由数字 ID 表示的子词 (subword)单元，以便输入到如 Transformer 这样的模型。它还展示了 OOV 处理，即一个未曾见过的词被分解为可识别的片段。

### 词汇表 (vocabulary)大小权衡

BPE 训练期间执行的合并操作次数直接决定了最终词汇表的大小。这带来一个权衡：

- **更大的词汇表**：可以用单个分词 (tokenization)表示常用词和语素，从而使常见文本的分词序列更短。但是，这会增加模型的嵌入 (embedding)层大小（∣V∣×dmodel|V| \times d\_{model}∣V∣×dmodel​），并可能包含非常稀有的子词 (subword)。
- **更小的词汇表**：由于词汇更常被分解为更小的单元，导致分词序列更长。这会增加序列处理期间的计算成本，但会减小嵌入层大小，并通过将罕见或形态复杂的词汇从更小片段组合起来，可能提供更好的泛化能力。

选择合适的词汇表大小是一个经验性的过程，通常受模型规模、训练数据特点以及下游任务表现的指导。BPE 提供了控制这种平衡的机制。

总之，BPE 是一种功能强大且广泛使用的子词分词技术。通过学习合并大型语料库中频繁出现的字符或字节对，它构建了一个词汇表，能有效处理大量文本，避免 OOV 问题，并允许控制词汇表大小和序列长度之间的平衡。

获取即时帮助、个性化解释和交互式代码示例。

---

### WordPiece 分词

# WordPiece 分词

WordPiece 词元 (token)化是一种构建大型词汇表 (vocabulary)的方法，由 Google 开发，并特别用于 BERT（来自 Transformer 的双向编码器表示）模型及其变体。此方法通过迭代合并单元来构建词汇表。与字节对编码（BPE）合并最常出现的词元对不同，WordPiece 的合并准则基于最大化训练数据的似然性，而非原始频率计数。

该过程与 BPE 类似：用训练数据中所有单个字符初始化词汇表。然后，迭代地考虑合并相邻词元。主要区别在于选择合并哪个对。WordPiece 选择一对（比如 AAA 和 BBB），使得将它们合并成单个词元 ABABAB 会导致训练语料库的似然性增加幅度最大，这假设词元上使用单一词语言模型。

### 似然性最大化准则

设想训练语料库是从我们当前词汇表 (vocabulary) VVV 生成的一系列词元 (token)。语料库的似然性 LLL 是序列中每个词元出现概率的乘积：

L=∏token∈CorpusP(token)L = \prod\_{token \in Corpus} P(token)L=token∈Corpus∏​P(token)

词元 P(token)P(token)P(token) 的概率通常估算为其在语料库中的频率除以词元总数：

P(token)=count(token)∑t∈Vcount(t)P(token) = \frac{count(token)}{\sum\_{t \in V} count(t)}P(token)=∑t∈V​count(t)count(token)​

当我们考虑将两个词元 AAA 和 BBB 合并成一个新词元 ABABAB 时，我们实际上会修改所有 AAA 和 BBB 相邻出现位置的词元序列。这会改变 AAA、BBB 和 ABABAB 的计数，以及语料库中词元的总数（因为每次合并都会使词元计数减少一）。WordPiece 会评估语料库当前分词 (tokenization)中所有可能的相邻对的这种变化。选择合并后对修改后的语料库产生最高似然性 LLL 的对 (A,B)(A, B)(A,B)，并将其添加到词汇表以进行下一次迭代。

实际操作中，计算完整的似然性变化可能很复杂。通常，会使用近似值或与似然性相关的分数。一种常见的考虑方式是选择能使如下分数最大化的合并 (A,B)→AB(A, B) \to AB(A,B)→AB：

score(A,B)=count(AB)count(A)×count(B)score(A, B) = \frac{count(AB)}{count(A) \times count(B)}score(A,B)=count(A)×count(B)count(AB)​

虽然这并非精确的似然性变化，但它反映了相似的直觉：相对于它们各自的频率，将经常一起出现的对进行合并是有益的。确切的评分函数在不同实现之间可能有所不同。

### WordPiece 的实践：前缀和实现

与许多 BPE 实现一样，WordPiece 通常通过给表示单词延续部分的词元 (token)添加一个特殊前缀（通常是 `##`）来处理单词内部的子词 (subword)。最初的单词划分可能发生在空白处，然后 WordPiece 分词 (tokenization)会在这些初始单词单元内部进行。

例如，单词“hugging”最初可能会被拆分为字符：`h`、`u`、`g`、`g`、`i`、`n`、`g`。通过基于似然性最大化的迭代合并，词汇表 (vocabulary)最终可能包含 `hug` 和 `##ging`。单词“hugging”随后将被分词为 `['hug', '##ging']`。`##` 表示 `ging` 无空格地连接到前一个词元。这使得模型能够区分一个词元是单词的开头部分还是中间部分，并实现明确的逆分词。

我们来看看如何在 PyTorch 中通过 Hugging Face `transformers` 库使用预训练 (pre-training)的 WordPiece 分词器 (tokenizer)，特别是 BERT 所用的分词器。

```python

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence = "WordPiece maximizes likelihood."

tokens = tokenizer.tokenize(sentence)

input_ids = tokenizer.convert_tokens_to_ids(tokens)

encoded_input = tokenizer(sentence)

print(f"原始句子: {sentence}")
print(f"词元: {tokens}")
print(f"词元 ID: {input_ids}")
print(f"完整编码输入（含特殊词元）:")
print(f"{encoded_input['input_ids']}")
print(f"解码后的完整输入: {tokenizer.decode(encoded_input['input_ids'])}")
```

运行此代码会产生如下所示的输出：

```
原始句子: WordPiece maximizes likelihood.
词元: ['word', '##piece', 'max', '##imi', '##zes', 'like', '##li', '##hood', '.']
词元 ID: [2773, 19352, 4011, 24027, 16464, 2066, 2135, 12731, 1012]
完整编码输入（含特殊词元）:
[101, 2773, 19352, 4011, 24027, 16464, 2066, 2135, 12731, 1012, 102]
解码后的完整输入: [CLS] wordpiece maximizes likelihood. [SEP]
```

请注意“WordPiece”如何被拆分为 `word` 和 `##piece`，“maximizes”被拆分为 `max`、`##imi` 和 `##zes`，“likelihood”被拆分为 `like`、`##li` 和 `##hood`。`##` 前缀清楚地标记 (token)了不作为单词开头的子词单元。最终的 `encoded_input['input_ids']` 在开头包含特殊词元 ID `[CLS]` (101)，在结尾包含 `[SEP]` (102)，这些是 BERT 风格模型的标准要求。

### WordPiece 与 BPE 的比较

WordPiece 和 BPE 之间的主要区别在于词元 (token)合并的准则：BPE 使用频率，而 WordPiece 使用语料库似然性最大化（或相关分数）。这种差异可能导致最终词汇表 (vocabulary)以及单词分段方式的不同。WordPiece 可能偏向于创建统计上更可能的单元的合并，即使它们在计数上并非严格地最频繁的对。

Failed to render diagram

> BPE 和 WordPiece 之间合并准则的不同。

实际应用中，这两种方法都能有效地创建子词 (subword)词汇表，从而很好地处理大型文本语料库，缓解 OOV 问题并保持词汇量在可管理范围内。它们之间的选择通常取决于它们最初开发的特定模型架构（例如，BERT 使用 WordPiece，GPT-2 使用 BPE）或在特定任务和数据集上的实际表现。实现方式随处可见，尤其是在 Hugging Face 的 `tokenizers` 库等框架中，这些框架通常会为最终用户抽象掉细微差别。然而，了解其基本机制有助于理解原始文本如何转换为大型语言模型处理的数字序列。

获取即时帮助、个性化解释和交互式代码示例。

---

### SentencePiece 实现

# SentencePiece 实现

尽管像字节对编码（BPE）和WordPiece这样的算法比简单的词语切分有许多优点，但它们通常对预分词 (tokenization)文本进行操作，这些文本一般通过空格切分。这一预处理步骤可能存在问题。它带来了对词语边界的假定，而这些假定并非适用于所有语言（例如中文、日文、泰语），并且可能永久丢失有关空格变化的有用信息。此外，管理单独的预分词脚本会增加数据处理流程的复杂度。

由Google开发的SentencePiece提供了一个统一的框架，解决了这些局限。它直接处理原始文本序列，将输入视为Unicode字符流。这避免了对特定语言预分词器 (tokenizer)的需求，使其成为多语言模型的一种适应性强的工具。

### 直接处理原始文本

SentencePiece的主要区别在于它不假定空格表示词语边界。相反，它将空格视为任何其他字符。在构建词表时，SentencePiece通常会明确地编码空格，通常在应用子词 (subword)算法（如BPE或Unigram）之前，将其替换为像 `▁`（U+2581，一个下八分之一块）这样的元符号。

考虑文本 "Hello world."。

- 依赖空格的BPE可能会首先将其切分为 `["Hello", "world."]`，然后对每个部分应用BPE。
- SentencePiece将其视为 `"Hello world."`。它可能会学习像 `He`、`llo`、`_world`、`.` 这样的分词 (tokenization)单元（其中 `_` 表示编码的空格）。这使得它能够从分词序列精确地重构原始字符串，包括空格。

这种方法使得SentencePiece与语言无关。它不需要知道词语的开始或结束位置；它直接从数据中学习常用的字符序列。

### 支持BPE和Unigram模型

SentencePiece并非单一算法，而是一个实现多种子词 (subword)分词 (tokenization)策略的框架。主要有两种：

1. **BPE：** 它包含字节对编码算法的一个实现，与我们之前讨论的相似，但经过调整以直接处理Unicode字符流，并且通常包含明确的空格处理。
2. **Unigram语言模型：** 这是一种不同的方法。它从一个包含大量可能子词（例如，训练数据中的所有子字符串）的初始词表开始，然后根据Unigram语言模型，迭代地移除对训练语料库总体似然贡献最小的分词单元，直到达到预设的词表大小。在分词过程中，它会根据学习到的词表分词单元的Unigram概率，找到使概率最大的切分。虽然它功能强大，但在许多大型Transformer模型中，BPE仍然更常见。

在训练SentencePiece模型时，您通常选择所需的算法（`--model_type=bpe` 或 `--model_type=unigram`）。

### 规范化与可逆性

SentencePiece将其文本规范化功能直接整合到其流程中。它可以应用标准的Unicode规范化形式（如NFKC），并支持通过正则表达式定义的自定义规范化规则。这确保了在分词 (tokenization)开始前文本表示的一致性。

重要的一点是，SentencePiece分词被设计为可逆的。因为它直接处理原始Unicode字符流并明确处理空格，您几乎总是可以将ID序列解码回精确的原始（规范化）文本字符串。这比那些在预分词过程中丢弃空格信息的现有分词器 (tokenizer)有很大的优势。

### 训练和使用SentencePiece模型

`sentencepiece` 库提供了命令行工具和Python绑定，用于训练和使用模型。

**1. 训练：**
您通常从原始文本文件训练模型。假设您有一个文件 `corpus.txt`。

```bash

spm_train --input=corpus.txt --model_prefix=my_sp_model --vocab_size=16000 --model_type=bpe --character_coverage=1.0 --normalization_rule_name=nmt_nfkc_cf
```

- `--input`：您的原始训练文本数据路径。
- `--model_prefix`：输出文件（`my_sp_model.model` 和 `my_sp_model.vocab`）的基本名称。
- `--vocab_size`：目标词表大小 ∣V∣|V|∣V∣。
- `--model_type`：要使用的算法（`bpe` 或 `unigram`）。
- `--character_coverage`：目标是用基本单字符分词 (tokenization)单元覆盖至少这个比例的输入字符。对于处理多样的文字系统很重要。
- `--normalization_rule_name`：预定义的规范化规则（例如，`nmt_nfkc_cf` 应用NFKC规范化和大小写折叠）。

这会生成 `my_sp_model.model`（包含词表和合并规则/概率）和 `my_sp_model.vocab`（一个人类可读的词表列表）。

**2. 在Python中使用（PyTorch环境下）：**
训练完成后，您加载 `.model` 文件，并将其用于编码和解码。

```python
import sentencepiece as spm
import torch

sp = spm.SentencePieceProcessor()
sp.load('my_sp_model.model')

text = "SentencePiece is useful for LLMs."

ids = sp.encode_as_ids(text)
print(f"原始文本: {text}")
print(f"编码ID: {ids}")

tensor_ids = torch.tensor(ids, dtype=torch.long)
print(f"PyTorch张量: {tensor_ids}")

decoded_text = sp.decode_ids(ids)
print(f"解码文本: {decoded_text}")

pieces = sp.encode_as_pieces(text)
print(f"编码片段: {pieces}")

vocab_size = sp.get_piece_size()
bos_id = sp.bos_id()
eos_id = sp.eos_id()
pad_id = sp.pad_id()
unk_id = sp.unk_id()

print(f"词表大小: {vocab_size}")
print(f"BOS ID: {bos_id}, EOS ID: {eos_id}, PAD ID: {pad_id}, UNK ID: {unk_id}")
```

在此示例中，`sp.encode_as_ids` 直接将原始字符串转换为整数列表。`sp.decode_ids` 执行逆向操作。通过 `encode_as_pieces` 获取片段的能力有助于理解分词过程。SentencePiece还定义了特殊分词单元的标准ID，如序列开始符（`BOS`）、序列结束符（`EOS`）、填充符（`PAD`）和未知符（`UNK`），这些对于为Transformer等模型准备输入批次很重要。

### 大型模型的优点

SentencePiece的设计为构建大型语言模型带来了几个好处：

- **无需预分词 (tokenization)：** 简化了数据处理流程，避免了特定语言的脚本。
- **语言无关：** 能够有效地处理具有不同书写系统和词语切分规则的多种语言。
- **无损重构：** 通常可以完美地重构原始规范化文本，保留空格。
- **高效率：** 用C++实现以提高速度，并提供Python绑定以便于使用。
- **集成规范化：** 一致地处理文本规范化。

通过将文本视为原始序列并采用BPE或Unigram等数据驱动的方法，SentencePiece提供了一种灵活的分词方法，非常适合现代大型语言模型开发中使用的多样和海量数据集。它避开了依赖空格的分词器 (tokenizer)的限制，并为准备文本数据提供了一个统一的解决方案。

获取即时帮助、个性化解释和交互式代码示例。

---

### 处理特殊分词

# 处理特殊分词

虽然BPE或WordPiece之类的子词 (subword)分词 (tokenization)算法显著减少了词汇表 (vocabulary)外（OOV）词的问题并控制了词汇量，但它们本身不提供许多基于Transformer模型所需的结构信息。为了处理序列分类、句子对比较或仅仅是定义序列边界等任务，我们引入了专门的*特殊分词*。这些分词被视为独立的词汇项，通常在训练分词器 (tokenizer)或配置预训练 (pre-training)分词器时明确预留和添加。

我们来了解最常见的特殊分词及其作用：

### 填充分词 (tokenization)：`[PAD]`

机器学习 (machine learning)模型通常以批次方式处理数据以提高效率。然而，批次内的序列长度常常不同。为形成深度学习 (deep learning)框架所需的矩形张量，较短的序列需要被填充到与批次中最长序列相同的长度。`[PAD]` 分词就是为此目的而设。

例如：考虑对两个句子进行分词：“Hello world” 和 “Tokenization example”。
如果分词为ID，它们可能看起来像 `[101, 7592, 2088, 102]` 和 `[101, 19204, 7404, 2742, 102]`。
为了以最大长度6进行批处理，假设 `[PAD]` 的ID为0，则批次会是：

```
[[101, 7592, 2088,  102,    0,    0],
 [101, 19204, 7404, 2742,  102,    0]]
```

模型在计算时忽略这些填充分词非常重要，特别是在注意力机制 (attention mechanism)中。这通过使用一个*注意力掩码*来实现，注意力掩码是一个二进制张量，表示哪些分词应该被关注（1）以及哪些应该被忽略（0）。对于上面的例子，掩码会是：

```
[[1, 1, 1, 1, 0, 0],
 [1, 1, 1, 1, 1, 0]]
```

### 未知分词 (tokenization)：`[UNK]`

即使使用子词 (subword)分词，也可能存在在分词器 (tokenizer)训练期间未曾遇到的稀有字符或字符序列（例如，错别字、不支持的符号、真正的新词），这些序列无法分解成已知子词。`[UNK]` 分词表示这些未知实体。尽管子词方法旨在最大程度减少 `[UNK]` 分词的出现频率，但有备用机制是必要的。推理 (inference)时 `[UNK]` 分词的高频率通常表明分词词汇表 (vocabulary)与输入数据分布之间存在不匹配，这可能会降低模型性能。

### 分类分词 (tokenization)：`[CLS]`

一些Transformer架构，特别是BERT，在每个输入序列前都加上一个特殊的 `[CLS]` 分词。与该分词对应的最终隐藏状态通常用作整个序列的聚合表示。此表示随后可以输入到分类器头部，用于情感分析或主题分类等序列级任务。尽管存在其他池化策略（例如，分词嵌入 (embedding)的平均池化），但使用 `[CLS]` 分词的输出是一种常见做法。

### 分隔分词 (tokenization)：`[SEP]`

为处理涉及多个文本片段的任务（例如，输入为 `[问题, 上下文]` 的问答，或输入为 `[前提, 假设]` 的自然语言推理 (inference)），会使用分隔分词 `[SEP]`。它明确标记 (token)了馈送给模型的单个输入序列中不同片段之间的边界。

BERT在句子对任务中的输入格式示例：
`[CLS] Sentence A tokens [SEP] Sentence B tokens [SEP]`

请注意，有些模型可能只使用 `[SEP]` 一次，而另一些则可能在每个片段之后使用它，包括最后一个片段。

### 掩码分词 (tokenization)：`[MASK]`

此分词是掩码语言建模（MLM）预训练 (pre-training)目标特有的，该目标由BERT推广。在预训练期间，一部分输入分词会被随机替换为 `[MASK]` 分词。模型的任务是根据周围的上下文 (context)预测被掩码的原始分词。这使得模型能够学习到丰富的双向表示。除非任务明确涉及预测掩码范围，否则 `[MASK]` 分词通常不用于微调 (fine-tuning)或标准推理 (inference)。

### 其他可能的分词 (tokenization)：`[BOS]`、`[EOS]`

像GPT这样的自回归 (autoregressive)模型经常使用序列开始（`[BOS]` 或 `<s>`）和序列结束（`[EOS]` 或 `</s>`）分词。`[BOS]` 表示生成开始，而 `[EOS]` 表示序列完成。尽管它们在标记 (token)边界方面与 `[CLS]` 和 `[SEP]` 相似，但它们的主要作用是在每次生成一个分词的文本生成环境中。

### 集成与使用

特殊分词 (tokenization)被添加到分词器 (tokenizer)的词汇表 (vocabulary)中并分配唯一的ID，就像常规子词 (subword)分词一样。Hugging Face `transformers` 等库管理此过程。当您加载预训练 (pre-training)分词器时，它会附带其预训练期间使用的特殊分词配置。

```python
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

sentence1 = "This is sentence one."
sentence2 = "This is another sentence."

encoded_input = tokenizer(
    sentence1,
    sentence2,
    padding=True,
    truncation=True,
    return_tensors='pt'
)

print("分词ID：")
print(encoded_input['input_ids'])

print("\n注意力掩码：")
print(encoded_input['attention_mask'])

print("\n解码后的分词：")

print(
    tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
)

print(f"\n[CLS] ID: {tokenizer.cls_token_id}")
print(f"[SEP] ID: {tokenizer.sep_token_id}")
print(f"[PAD] ID: {tokenizer.pad_token_id}")
print(f"[UNK] ID: {tokenizer.unk_token_id}")
print(f"[MASK] ID: {tokenizer.mask_token_id}")
```

执行此代码将显示分词器如何自动在开头添加 `[CLS]`，在句子之间和末尾添加 `[SEP]`，并填充较短的序列（如果此特定非填充示例中需要填充的话，但在此处两个句子可能编码为相同长度，因此不需要）。`attention_mask` 正确地识别了非填充分词。

```text
分词ID：
tensor([[  101,  2023,  2003,  6251,  2028,  1012,   102,  2023,  2003,  2178,
          6251,  1012,   102]])

注意力掩码：
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

解码后的分词：
['[CLS]', 'this', 'is', 'sentence', 'one', '.', '[SEP]', 'this', 'is', 'another', 'sentence', '.', '[SEP]']

[CLS] ID: 101
[SEP] ID: 102
[PAD] ID: 0
[UNK] ID: 100
[MASK] ID: 103
```

理解并正确管理这些特殊分词是根本。它们为模型正确解释各种预训练和下游任务的输入序列提供了必要的结构提示。特殊分词的具体集合及其使用模式通常取决于模型架构和其训练目标。请务必查阅您所用模型和分词器的文档。

获取即时帮助、个性化解释和交互式代码示例。

---

### 词汇量大小选择的权衡

# 词汇量大小选择的权衡

为您的子词 (subword)分词 (tokenization)器 (tokenizer)选择词汇量大小，通常表示为 ∣V∣|V|∣V∣，是一个主要的超参数 (parameter) (hyperparameter)选择，直接影响模型表现、内存占用和计算成本。与词汇表 (vocabulary)可能自然增长的简单分词方法不同，BPE 或 WordPiece 等子词算法要求您预先定义一个目标词汇量大小。这一决定需要平衡多个相互影响的因素。

### 对序列长度和信息粒度的影响

**较小的词汇量大小**会使分词 (tokenization)器 (tokenizer)更频繁地将单词拆分为更小的子词 (subword)单元。

- **优点：** 它能通过常见子词组合来妥善处理稀有词和词汇表 (vocabulary)外词（OOV）。这能提高对拼写错误和新词的鲁棒性。
- **缺点：** 这会导致相同输入文本的标记 (token)序列更长。由于Transformer中注意力机制 (attention mechanism)的计算成本与序列长度呈平方关系（O(L2)O(L^2)O(L2)），更长的序列会显著增加训练和推理 (inference)时间以及内存使用。

反之，**更大的词汇量大小**允许分词器将更多常见词或频繁出现的子词序列表示为单个标记。

- **优点：** 这会产生更短的标记序列，从而减轻注意力机制的计算负担。如果常见词保持完整，单个标记嵌入 (embedding)中可能会包含更多语义信息。
- **缺点：** 如果词汇表对于目标领域不够大或不够多样，会增加出现OOV词的风险。它还会增加模型嵌入层的大小。

以“tokenization”（分词）一词为例。

- 对于较小的词汇量（例如，∣V∣=10,000|V|=10,000∣V∣=10,000），它可能被分词为 `['tok', 'en', 'ization']`（3个标记）。
- 对于更大的词汇量（例如，∣V∣=50,000|V|=50,000∣V∣=50,000），如果在训练时包含“tokenization”，它可能变为 `['tokenization']`（1个标记）。
- 对于中等词汇量（例如，∣V∣=30,000|V|=30,000∣V∣=30,000），可能为 `['token', 'ization']`（2个标记）。

更长的序列直接影响训练期间激活所需的内存，特别是注意力得分矩阵（L×LL \times LL×L）。

20k40k60k80k100k900100011001200

> 词汇量越大，对于固定语料库，通常会使平均序列长度变短，从而降低注意力计算成本。

### 对模型大小和内存的影响

词汇量大小直接决定了输入嵌入 (embedding)矩阵和输出投影层（通常是绑定的）的大小。嵌入矩阵的维度为 ∣V∣×dmodel|V| \times d\_{model}∣V∣×dmodel​，其中 dmodeld\_{model}dmodel​ 是模型的隐藏维度。

嵌入矩阵大小=∣V∣×dmodel×每参数字节数\text{嵌入矩阵大小} = |V| \times d\_{model} \times \text{每参数字节数}嵌入矩阵大小=∣V∣×dmodel​×每参数字节数

更大的 ∣V∣|V|∣V∣ 会导致嵌入矩阵按比例增大。对于dmodeld\_{model}dmodel​ 为数千的大型模型，将 ∣V∣|V|∣V∣ 从30,000增加到100,000，仅在嵌入层就可能增加数十亿参数 (parameter)和数千兆字节的模型内存占用。

```python
import torch
import torch.nn as nn

d_model = 4096

vocab_size_small = 32000
vocab_size_large = 128000

embedding_small = nn.Embedding(vocab_size_small, d_model)
embedding_large = nn.Embedding(vocab_size_large, d_model)

params_small = sum(p.numel() for p in embedding_small.parameters())
params_large = sum(p.numel() for p in embedding_large.parameters())

mem_small_gb = params_small * 4 / (1024**3)
mem_large_gb = params_large * 4 / (1024**3)

print(
    f"Vocab Size: {vocab_size_small}, "
    f"Embedding Params: {params_small:,}, "
    f"Memory: {mem_small_gb:.2f} GB"
)
print(
    f"Vocab Size: {vocab_size_large}, "
    f"Embedding Params: {params_large:,}, "
    f"Memory: {mem_large_gb:.2f} GB"
)
```

此计算仅包含嵌入层。最终输出层将隐藏状态投影回词汇维度以预测下一个标记 (token)，其大小通常也相似（dmodel×∣V∣d\_{model} \times |V|dmodel​×∣V∣），这会进一步放大 ∣V∣|V|∣V∣ 对模型大小的影响。

### 对计算（Softmax）的影响

在训练和推理 (inference)期间，模型通常使用应用于输出logits的softmax函数计算整个词汇表 (vocabulary)的概率。此最终softmax层的计算成本与 ∣V∣|V|∣V∣ 呈线性关系。

Softmax 成本∝L×dmodel×∣V∣\text{Softmax 成本} \propto L \times d\_{model} \times |V|Softmax 成本∝L×dmodel​×∣V∣

虽然对于长序列，注意力机制 (attention mechanism)（O(L2×dmodel)O(L^2 \times d\_{model})O(L2×dmodel​)）通常占据主导地位，但softmax计算（O(L×dmodel×∣V∣)O(L \times d\_{model} \times |V|)O(L×dmodel​×∣V∣)）可能成为一个重要瓶颈，特别是在词汇量非常大或在推理过程中批量大小可能较小且延迟很关键时。更大的 ∣V∣|V|∣V∣ 直接增加了这项成本。

### 寻求平衡：常见做法

没有单一的“最佳”词汇量大小；它取决于具体的任务、语言、数据集大小、模型架构和可用的计算资源。然而，存在一些常见的观察和做法：

1. **典型范围：** 对于许多英语模型，词汇量大小通常在30,000到100,000之间。32,000、50,000和64,000左右的大小很常见，通常是为了计算便利而选择（例如，可被2的幂或硬件向量 (vector)单元整除）。多语言模型可能需要更大的词汇量（例如，100,000到256,000）以充分覆盖多种字符集和词形。
2. **收益递减：** 增加 ∣V∣|V|∣V∣ 在序列长度缩减和困惑度改进方面，在某个点后会产生递减收益。从3万到5万的收益可能很明显，而从10万到15万的收益可能微不足道，但会带来显著的内存成本。
3. **数据特点：** 最佳大小受训练数据的多样性和规模影响。更多样的数据（多种语言、领域、代码）可能会从更大的词汇量中获益。
4. **模型大小：** 对于较小的模型，由于词汇量增大而导致的模型大小按比例增长更明显。对于非常大的模型（数千亿参数 (parameter)），嵌入 (embedding)层可能只占总参数的一小部分，如果更大的词汇量能带来性能改善，可能使其更易于接受。
5. **计算预算：** 最终，硬件限制（GPU内存）和训练时间约束通常对 ∣V∣|V|∣V∣ 设定了实际的上限。

G

V\_Small

较小的 |V|

SeqLen

序列长度 (L)

V\_Small->SeqLen

增加

MemEmbed

嵌入内存

V\_Small->MemEmbed

减少

OOV

OOV 鲁棒性

V\_Small->OOV

增加

V\_Large

较大的 |V|

V\_Large->SeqLen

减少

V\_Large->MemEmbed

增加

V\_Large->OOV

减少

MemAttn

注意力内存 (O(L²))

SeqLen->MemAttn

增加

CompSoftmax

Softmax 计算

SeqLen->CompSoftmax

影响 (通过 L)

MemEmbed->CompSoftmax

影响 (通过 |V|)

> 选择较小或较大词汇量大小（∣V∣|V|∣V∣）所涉及的权衡。

在实际操作中，选择 ∣V∣|V|∣V∣ 通常涉及一些经验性测试，或采用文献中针对类似模型规模和数据集报告的大小。您需要权衡更短序列和潜在更好表示常见词（更大的 ∣V∣|V|∣V∣）带来的益处，与增加的内存使用、较慢的softmax计算以及潜在更差的稀有词处理（较小的 ∣V∣|V|∣V∣）所产生的成本。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 6 Sourcing Acquiring Massive Text Datasets

### 确定潜在数据来源

# 确定潜在数据来源

任何强大大型语言模型的根本在于其预训练 (pre-training)期间所使用的数据。如前所述，我们说的不是兆字节或吉字节；我们处理的是数太字节，有时甚至是数拍字节的文本数据。找出这类庞大数量数据的来源是实际的第一步。数据的构成在很大程度上决定了最终模型的能力、偏见和整体表现。因此，了解不同数据来源的特点、优点和缺点非常必要。

让我们分析一下大型语言模型预训练常用的文本数据主要类别。

### 互联网

互联网是目前最大且最多样化的文本数据来源。它包含几乎所有能想到的主题信息，以无数种风格和语言书写。

- **数据量庞大：** 网络爬取可以获取数百太字节的原始HTML内容。像Common Crawl这样的项目提供了可公开访问的网络快照，构成了许多广泛使用的大型语言模型数据集（例如C4、OSCAR）的根本。我们将在下一节详细讨论Common Crawl数据的处理。
- **多样性：** 网络文本涵盖新闻、博客、论坛、评论、百科文章等等。这种多样性有助于模型学习广泛的语言模式和知识。
- **挑战：** 原始网络数据出了名的嘈杂。它包含大量“样板内容”（导航菜单、广告、法律免责声明）、重复内容、低质量写作、机器翻译文本，以及可能有害或带有偏见的语言。提取核心文本内容和进行质量过滤是第7章中介绍的重要工程任务。有针对性的网络抓取可以补充广泛的爬取，但这需要在道德和网站服务条款方面仔细执行（本章稍后会讨论）。

### 图书

数字化图书提供了高质量、长篇文本的来源。

- **质量与连贯性：** 图书通常经过编辑过程，因此语言语法正确、结构良好。它们使模型接触到叙事流畅、复杂句式以及连贯的论证或故事，这些在普通网络文本中较难找到。
- **来源：** 像古腾堡项目这样的公共领域藏书很有价值。BookCorpus等其他数据集过去也曾被使用，但由于版权限制，获取大量、多样化且合法合规的图书数据集仍然存在挑战。
- **挑战：** 版权是主要障碍。即使对于可获取的图书，光学字符识别（OCR）错误也可能引入噪声，并且移除页码、页眉和页脚等格式化痕迹需要特定的预处理步骤。

### 代码库

对于旨在理解或生成计算机代码的模型，在预训练 (pre-training)数据中包含源代码是不可或缺的。

- **结构与逻辑：** 代码是高度结构化的文本，具有正式的语法和语义。对代码进行训练有助于模型学习逻辑推理 (inference)和算法模式。
- **来源：** GitHub等平台上的公共代码库是主要来源。像“The Stack”这样的聚合数据集提供了大量来自多种语言的许可宽松的代码集合。
- **挑战：** 处理各种编程语言及其特定语法是必要的。过滤掉非代码元素（例如，代码库中混杂的构建文件、文档、问题讨论）也很重要。软件许可证差异很大，需要仔细过滤以符合使用条款。

### 学术论文与科学文献

像arXiv、PubMed Central和机构知识库这样的来源包含大量的科学知识。

- **专业知识：** 这些数据充分覆盖了特定的科学和技术方面。
- **正式语言：** 学术写作通常正式且精确，使模型接触到专业词汇和复杂的推理 (inference)结构。
- **挑战：** 从PDF文件中提取干净文本是一个重大障碍，因为PDF文件通常具有复杂的多栏布局、图表、表格和公式。付费墙和限制性许可限制了对许多已发表文献的访问。

### 对话数据

来自社交媒体、论坛和其他互动平台的文本记录了非正式的语言使用。

- **非正式风格与对话：** 这些数据反映了人们的自然交流方式，包括俚语、缩写和对话轮次。它对于构建聊天机器人或用于互动应用程序的模型很有价值。
- **来源：** 像Reddit这样的平台已被使用（例如Pushshift Reddit数据集），但API变更、服务条款以及用户隐私方面的道德考量是主要因素。
- **挑战：** 高度噪声、重复内容、毒性言论和个人身份信息（PII）很常见。数据可能碎片化，缺乏更广泛的背景。道德获取和严格的过滤/匿名化极其重要。

### 新闻文章

新闻语料库提供了访问时事和事实报道风格的途径。

- **时事信息：** 新闻文本使模型了解最新事件、人物和地点。
- **事实性风格：** 新闻写作通常目标是客观和事实呈现（尽管偏见仍然是一个问题）。
- **来源：** 存在大量新闻档案，有时可通过API或特定数据集（例如RealNews）访问。
- **挑战：** 付费墙限制了对许多付费新闻来源的访问。检测和减轻观点偏见很困难。新闻的快速变化意味着数据可能很快过时，并且类似事件经常在不同来源中重复报道。

### 专门语料库

在这些通用类别之外，还存在许多特定领域的数据集。

- **专业知识：** 法律文件（法院裁决、合同）、医学文献（研究论文、临床笔记——如果经过匿名化和符合道德规范的来源）或财务报告可以被纳入以构建具有专业知识的模型。
- **挑战：** 这些数据集通常小于网络规模的语料库。由于隐私、商业敏感性或知识产权，访问可能会受到限制。处理可能需要特定领域的知识。

### 组合数据来源

单一来源的数据不足以满足需求。最先进的大型语言模型几乎总是基于对这些类别中多个数据源的精心策划的*混合*进行训练。具体的组合是一个重要的设计选择，影响着模型的优点和缺点。例如，一个主要在网络文本上训练的模型可能擅长日常对话，但在正式推理 (inference)方面可能表现不佳，而一个侧重于学术论文的模型则可能表现出相反的特点。C4、The Pile或精炼的专有数据集所使用的具体比例通常是根据下游任务的表现进行调整的。我们将在第9章进一步讨论数据混合策略。

60%15%10%5%5%5%网络 (已过滤)图书代码学术新闻对话

> 这是一个示例分布，说明了通用大型语言模型的预训练 (pre-training)数据集中不同数据来源可能如何加权。实际比例在不同模型之间差异很大。

访问和处理这些多样化的来源通常需要使用为大型数据集设计的库和工具。例如，Hugging Face的`datasets`库提供了对许多预处理数据集的便捷访问，包括大型网络爬取数据的子集或特定语料库。

```python

from datasets import load_dataset

try:

    oscar_subset = load_dataset(
        "oscar",
        "unshuffled_deduplicated_en",
        split='train[:1%]',
        streaming=True
    )

    print("OSCAR子集的前5个示例：")
    count = 0
    for example in oscar_subset:
        print(f"示例 {count + 1}：")

        print(example['text'][:200] + "...")
        print("-" * 20)
        count += 1
        if count >= 5:
            break

except Exception as e:
    print(
        "尝试加载数据集时发生错误： "
        f"{e}"
    )
    print(
        "请确保您有互联网连接并且 'datasets' 库已安装。"
    )
    print("某些数据集可能需要特定的配置或权限。")
```

此代码片段说明了如何以编程方式访问大型标准化数据集的一部分。虽然方便，但请记住，使用现有数据集意味着要依赖其创建者所做的预处理选择。构建独特、高质量的数据集通常需要回到原始来源（如Common Crawl存档或直接网络抓取），并实施自定义的清洗和过滤流程，我们将在后续部分和第7章中介绍这些内容。

获取即时帮助、个性化解释和交互式代码示例。

---

### 使用 Common Crawl 数据

# 使用 Common Crawl 数据

获取足够大的数据集用于大型语言模型（LLM）预训练 (pre-training)是一项重要的工程难题。为此，Common Crawl (CC) 语料库是最大且常用的资源之一。它通过大规模网络爬取操作收集，代表了公共网络很大一部分的快照。Common Crawl 公开提供数PB的原始网页数据、元数据和提取的文本，通常每隔一两个月发布新的爬取数据集。其规模之大使其成为训练先进语言模型不可或缺的资源，尽管其利用存在一定困难。

### 理解 Common Crawl 档案

Common Crawl 是一个开放的网络爬取数据仓库。可以将其视为一个庞大的图书馆，其中包含多年来收集的数十亿网页副本。这些数据存储在亚马逊云服务（AWS）的简单存储服务（S3）上，所有人均可访问，尽管在“请求者付费”模式下通常会产生数据传输费用。

每次爬取的数据主要分为三种格式：

1. **WARC (Web ARChive)：** 这些文件包含每个被爬取 URL 的原始响应数据，包括完整的 HTTP 头和未经处理的 HTML 内容。WARC 文件内容全面但非常大，需要大量的解析工作才能提取有用文本。
2. **WAT (WARC Archive Transformation)：** 这些文件存储从 WARC 记录中提取的元数据。这包括 HTTP 头、页面上检测到的链接以及在爬取处理过程中获取的其他元数据。WAT 文件对于分析网络结构或特定的元数据点很有用，但它们不直接包含页面的主要文本内容。
3. **WET (WARC Encapsulated Text)：** 这些文件包含从 WARC 文件中存储的 HTML 内容中提取的纯文本。Common Crawl 应用启发式方法去除 HTML 标签、导航栏、广告和其他非内容元素。WET 文件明显小于 WARC 文件，通常是为 LLM 预训练 (pre-training)收集文本的最实用起点。但是，请注意文本提取的质量可能有所不同，有时重要的上下文 (context)或格式可能会丢失。

### 访问 AWS S3 上的 Common Crawl 数据

访问批量数据最直接的方式是通过 AWS S3。每次爬取都有一个唯一标识符（例如，`CC-MAIN-2023-50`），其数据组织在相应的 S3 存储桶中（例如，`s3://commoncrawl/`）。您可以使用标准 AWS 工具，如 AWS 命令行界面 (CLI) 或 SDK（例如 Python 的 `boto3`），来列出和下载文件。

```bash

aws s3 ls s3://commoncrawl/crawl-data/CC-MAIN-2023-50/segments/1700495631538.95/wet/ --request-payer requester
```

请记住，`--request-payer requester` 标志是必需的，因为 Common Crawl 存储桶被配置为由*请求*数据的人支付下载带宽费用。下载数 TB 或 PB 数据可能会产生可观的费用。

### 处理 WET 文件以提取文本

由于 WET 文件包含提取的纯文本，它们通常是最方便的处理格式。一个典型的 WET 文件（`.wet.gz`）是一个 gzip 压缩档案，其中包含多个文本记录，每个记录对应一个网页。每个记录都以提供元数据（如目标 URI 和内容长度）的 WARC 头开始，后跟提取的纯文本内容。

以下是一个使用 `warcio` 库（它简化了 WARC 格式文件，包括 WET 的读取）在 WET 文件中遍历记录的 Python 代码片段：

```python
import gzip
from warcio.archiveiterator import ArchiveIterator
import io

try:
    with open(wet_file_path, 'rb') as stream:

        with gzip.open(stream, 'rb') as compressed_stream:

            bytes_stream = io.BytesIO(compressed_stream.read())

            for record in ArchiveIterator(bytes_stream):

                if record.rec_type == 'conversion':

                    uri = record.rec_headers.get_header('WARC-Target-URI')

                    content_bytes = record.content_stream().read()
                    try:
                        content_text = content_bytes.decode('utf-8')

                        if len(content_text.strip()) > 100:
                            print(f"URI: {uri}")
                            print(f"Content Length: {len(content_text)}")

                    except UnicodeDecodeError:

                        print(f"Skipping record for URI {uri} "
                              f"because of decoding error.")
                        continue

except FileNotFoundError:
    print(f"Error: File not found at {wet_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
```

> 使用 Python 和 `warcio` 库处理 WET 文件的简要概述。

处理整个 Common Crawl 数据集需要分布式计算框架，如 Apache Spark 或 Dask，并在可访问 S3 的集群上运行。这些框架允许您在多台机器上并行化 WET 文件的读取、解析和过滤，考虑到 PB 级别的数据规模，这是必不可少的。我们将在第 7 章讨论如何构建此类可扩展的数据处理流程。

### 使用 CC-Index 服务器

仅仅为了查找来自特定网站或页面类型的数据而下载整个数 TB 的数据段是效率低下的。Common Crawl 提供了一个索引服务（CC-Index），允许您查询爬取的元数据，而无需首先下载大型 WARC/WET 文件。

该索引包含关于每个被爬取 URL 的信息，包括其 MIME 类型、时间戳、在相应 WARC 文件中的位置以及状态码。您可以使用 `cdx-toolkit` 等工具或通过直接向 CDX API 端点发出 HTTP 请求来查询此索引。

例如，您可以查询索引以查找特定域名（`example.com`）下爬取的所有页面，或被识别为特定语言的所有页面。这使得数据获取更具针对性。一旦您从索引查询中获得了 WARC 文件位置（文件名和偏移量），您就可以执行范围 HTTP GET 请求，仅获取您需要的特定 WARC 记录，如果您只需要部分数据，这将显著减少下载量。

```bash

```

### 数据质量和过滤考量

尽管 Common Crawl 提供了庞大的规模，但原始数据，即使是 WET 格式，本质上也是嘈杂的，需要大量清理。常见问题包括：

- **样板内容：** 导航菜单、页脚、侧边栏、Cookie 同意表单。
- **广告：** 与主要内容混合的广告文本。
- **低质量内容：** 垃圾邮件、机器生成文本、语法差的论坛帖子。
- **标记 (token)残余：** HTML/JavaScript 代码未完全去除。
- **语言污染：** 非目标语言的文本。
- **重复：** 多次爬取到的近似相同或完全相同的页面。

G

Raw

原始 WET 文件

LangID

语言识别

Raw->LangID

识别语言

Quality

质量过滤

LangID->Quality

过滤低质量内容

Dedupe

去重

Quality->Dedupe

删除重复项

Clean

干净文本语料库

Dedupe->Clean

存储结果

> 典型的过滤流程应用于 Common Crawl WET 数据，然后才用于 LLM 训练。

解决这些问题通常需要构建一个多阶段流程，包括语言识别、基于启发式的质量过滤（例如，根据文本长度、符号比例、停用词频率）和近似重复检测技术（如 MinHash）。这些重要的预处理步骤是第 7 章的重点。

总之，Common Crawl 是获取 LLM 预训练 (pre-training)所需规模的网络数据的基础资源。访问和处理其 WET 文件提供了获取大量文本的直接途径，但这需要操作 AWS S3、管理成本，并实施分布式处理和清理流程，以处理网络内容的噪声和多样性。理解如何有效使用 Common Crawl 是 LLM 开发数据获取阶段的重要一步。

获取即时帮助、个性化解释和交互式代码示例。

---

### 规模化网页抓取技术

# 规模化网页抓取技术

像 Common Crawl 这样的大型精选数据集提供了很好的起点。然而，在某些情况下，您需要比现有归档数据更具针对性或更新的数据。网页抓取，即从网站自动提取数据的过程，成为一种必要手段。将此过程扩展到收集适用于大型语言模型预训练 (pre-training)的数TB文本，会带来相当大的工程挑战。

本节侧重于构建和运行能够高效且负责任地获取海量数据的网页爬虫所需的技术和注意事项。

### 异步操作以提高I/O效率

网页抓取本质上是一个I/O密集型任务。您的爬虫大部分时间都在等待网络响应。使用标准同步代码（如流行的 `requests` 库）意味着每个请求都会阻塞执行，直到它完成。这会严重限制吞吐量 (throughput)。

为解决此问题，异步编程必不可少。Python 的 `asyncio` 库，结合 `aiohttp` 等 HTTP 客户端，允许您的爬虫同时发起多个网络请求。当一个请求等待响应时，程序可以开始其他请求或处理已完成的请求，从而大幅提升整体速度。

这是一个使用 `aiohttp` 同时获取多个URL的例子：

```python
import asyncio
import aiohttp
import time

async def fetch(session, url):
    """异步获取单个URL。"""
    print(f"正在获取: {url}")
    try:

        async with session.get(url, timeout=10) as response:

            if response.status == 200:

                print(
                    f"成功获取: {url} "
                    f"(状态: {response.status})"
                )
                return await response.text()
            else:
                print(
                    f"失败: {url} (状态: {response.status})"
                )
                return None
    except asyncio.TimeoutError:
        print(f"获取 {url} 超时")
        return None
    except aiohttp.ClientError as e:
        print(f"获取 {url} 时发生客户端错误: {e}")
        return None

async def main(urls):
    """管理多个URL的并发获取。"""

    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]

        results = await asyncio.gather(*tasks)

        successful_fetches = [r for r in results if r is not None]
        print(
            f"\n获取完成。 "
            f"成功: {len(successful_fetches)}/{len(urls)}"
        )

target_urls = [
    "https://example.com",
    "https://httpbin.org/get",
    "https://httpbin.org/delay/1",
    "https://nonexistent-domain-xyz123.org",
    "https://httpbin.org/status/404"
]

start_time = time.time()

if __name__ == "__main__":
    asyncio.run(main(target_urls))
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")
```

运行此代码通常比最长的单个请求所需时间略长（此示例中的1秒延迟），而不是所有请求时间的总和，这体现了并发的好处。

### 分布式爬虫架构

即使有异步操作，单台机器在带宽、CPU和内存方面也有局限。要达到大型语言模型所需的抓取规模（可能数十亿页面），需要分布式架构。常见组件包括：

1. **URL 前沿/队列：** 一个管理待爬取URL列表的系统。这需要处理可能数十亿条目，优先处理URL，并防止重复爬取最近访问过的页面。常用消息队列如 RabbitMQ、Kafka，甚至是 Redis 列表。工作节点从队列中获取URL批次。
2. **爬虫工作节点：** 多个进程或机器，每个都运行一个异步爬虫实例（如 `aiohttp` 示例，通常使用 Scrapy 等框架进行增强）。它们从队列中获取URL，下载页面，可能提取新URL的链接（将其添加回队列），并处理/存储内容。
3. **内容存储：** 用于原始HTML或提取文本的可伸缩存储系统。云对象存储（如 AWS S3、Google Cloud Storage）或分布式文件系统（HDFS）是合适的选择（第8章会更详细讨论）。
4. **元数据存储：** 通常需要用于追踪已访问URL、爬取时间戳、`robots.txt` 规则，以及可能用于去重化处理的页面校验和。数据库（SQL或NoSQL）或专用键值存储可以实现此目的。

G

cluster\_queue

URL 管理

cluster\_workers

爬取工作节点

clusterₛtorage

数据存储

URL\_Queue

URL 队列
(例如, Kafka, Redis)

Worker1

爬虫工作节点 1
(异步获取器)

URL\_Queue->Worker1

 URL 批次

Worker2

爬虫工作节点 2
(异步获取器)

URL\_Queue->Worker2

 URL 批次

WorkerN

爬虫工作节点 N
(异步获取器)

URL\_Queue->WorkerN

 URL 批次

Metadata\_DB

元数据数据库
(已访问URL, 时间戳)

Worker1->URL\_Queue

 发现新URL

Worker1->Metadata\_DB

 更新状态

Worker1->Metadata\_DB

 检查是否已访问

Data\_Store

原始/处理过的数据
(例如, S3, HDFS)

Worker1->Data\_Store

 存储内容

Worker2->URL\_Queue

 发现新URL

Worker2->Metadata\_DB

 更新状态

Worker2->Data\_Store

 存储内容

WorkerN->URL\_Queue

 发现新URL

WorkerN->Metadata\_DB

 更新状态

WorkerN->Data\_Store

 存储内容

> 一个分布式爬取系统的简化图。工作节点从中央队列获取URL，处理页面，存储内容，更新元数据，并可能发现新的URL并将其添加回队列。

### 实现负责任的爬取

激进的抓取可能导致网站过载，影响其对合法用户的可用性，并可能导致您的IP地址被封锁。大规模爬取*必须*负责任地进行。

#### robots.txt

大多数网站提供一个 `/robots.txt` 文件，概述了对自动化代理（机器人/爬虫）的规则。这些规则指定了网站的哪些部分不应被访问（`Disallow`），有时会建议一个推荐的爬取延迟。遵守 `robots.txt` 是道德抓取的一个基本方面。

Python 的 `urllib.robotparser` 可以提供帮助：

```python
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin

USER_AGENT = "MyLLMDataCrawler/1.0 (+http://mycrawlerinfo.example.com)"

def can_fetch_url(robot_parser, url):
    """检查URL是否被robots.txt允许，供我们的用户代理使用。"""
    try:
        return robot_parser.can_fetch(USER_AGENT, url)
    except Exception as e:

        print(f"检查 {url} 的robots.txt权限时出错: {e}")
        return False

robots_url_cache = {}

async def check_and_fetch(session, url):

    base_url = urljoin(url, '/')
    robots_url = urljoin(base_url, 'robots.txt')

    parser = robots_url_cache.get(base_url)
    if not parser:
        print(f"正在获取 {base_url} 的robots.txt")
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:

            import requests
            response = requests.get(
                robots_url,
                timeout=5,
                headers={'User-Agent': USER_AGENT}
            )
            if response.status_code == 200:
                 parser.parse(response.text.splitlines())
            else:
                 print(
                     f"未找到 {base_url} 的有效robots.txt "
                     f"(状态: {response.status_code})"
                 )

            robots_url_cache[base_url] = parser

        except Exception as e:
            print(f"获取或解析 {base_url} 的robots.txt失败: {e}")
            robots_url_cache[base_url] = None
            return None

    if parser and can_fetch_url(parser, url):
        print(f"允许获取: {url}")

        return await fetch(session, url)
    elif parser:
        print(f"被robots.txt禁止: {url}")
        return None
    else:

        print(f"无法验证 {url} 的robots.txt权限，跳过。")
        return None
```

#### 频率限制

即使 `robots.txt` 允许，每秒向单个服务器发送数百或数千个请求也可能使其不堪重负。对每个域名实施频率限制。

- **固定延迟：** 在对同一域名的请求之间添加一个简单的 `await asyncio.sleep(delay_seconds)`。
- **`Crawl-delay` 指令：** 有些 `robots.txt` 文件会建议一个延迟。如果存在，请遵守。
- **自适应延迟：** 监测服务器响应时间或错误率（如 HTTP 429 “请求过多”，503 “服务不可用”）。如果出现问题，自动增加延迟。
- **并发连接限制：** 限制对单个域名/IP地址的并发连接数。

一个简单的按域名延迟机制：

```python
import asyncio
import time

last_access_times = {}

MIN_DELAY_PER_DOMAIN = 1.0

async def rate_limited_fetch(session, url):

    domain = urlparse(url).netloc

    last_access = last_access_times.get(domain, 0)
    now = time.monotonic()
    elapsed = now - last_access

    if elapsed < MIN_DELAY_PER_DOMAIN:
        wait_time = MIN_DELAY_PER_DOMAIN - elapsed
        print(f"对 {domain} 进行频率限制: 等待 {wait_time:.2f} 秒")
        await asyncio.sleep(wait_time)

    last_access_times[domain] = time.monotonic()

    result = await fetch(session, url)
    return result
```

#### User-Agent 标识

始终设置清晰的 `User-Agent` 字符串，以标识您的爬虫，理想情况下提供联系方式（例如，带有信息或电子邮件地址的URL）。这有助于网站管理员了解流量来源，并在您的爬虫导致问题时与您联系。避免使用通用浏览器用户代理。

示例：`User-Agent: LLMBuilderBot/0.1 (+http://www.my-llm-project.org/crawler-info)`

### 处理规模挑战

除了礼貌爬取之外，大规模抓取还会带来其他技术障碍：

- **动态内容（JavaScript）：** 许多现代网站在初始HTML页面加载后使用JavaScript加载内容。简单的HTTP请求无法捕获这些数据。需要像 `Playwright` 或 `Selenium` 这样控制真实浏览器引擎的工具。然而，它们比简单的HTTP客户端资源消耗明显更高（CPU、内存），会减缓爬取速度并增加成本。仅在必要时选择性使用它们。
- **爬虫陷阱：** 网站可能有意或无意地创建“爬虫陷阱”——动态生成链接的无限循环（例如，日历中无休止的“下个月”链接），可以困住一个缺乏经验的爬虫。实施诸如每个域的最大爬取深度、URL模式过滤或监控从单个页面/域发现的URL数量等机制。
- **会话管理与登录：** 访问需要登录的内容需要处理认证（cookie、令牌），这增加了复杂性并可能违反服务条款。除非明确允许，大规模预训练 (pre-training)数据收集通常会避免这种情况。
- **错误处理：** 网络故障、服务器错误（HTTP 5xx）、客户端错误（HTTP 4xx）和超时在大规模操作中很常见。对超时或503等瞬时错误实施重试逻辑，通常采用指数退避（每次失败后等待时间逐渐延长）。对于持久性错误（如 404 Not Found），在几次重试后放弃。

构建一个可伸缩的网页抓取器是一项重要的软件工程任务。虽然库和框架提供了构建模块，但仔细的设计，考虑并发、分布式、存储、礼貌性和错误处理，对于成功收集训练大型语言模型所需的庞大数据是必要的。请记住，数据质量也十分重要；抓取的原始输出通常需要大量清洗和过滤，如第7章（“数据清洗和预处理流程”）所述。

获取即时帮助、个性化解释和交互式代码示例。

---

### 使用开放许可数据集

# 使用开放许可数据集

虽然自行构建大规模网络爬虫或处理Common Crawl等原始档案能够提供最大程度的控制，但这需要大量的工程投入和基础设施。幸运的是，研究社群和各类组织已经整理并发布了许多大型文本数据集，这些数据集均采用开放许可，为获取预训练 (pre-training)数据提供了有用的捷径或补充。

使用这些现有数据集可以节省大量与抓取、初步清洗和格式化相关的时间和资源。它们通常附带详细说明其来源和预处理步骤的文档，不过仍需仔细核查。

### 理解开放许可

在使用任何数据集之前，了解其许可条款非常重要。“开放”并非意味着“不受限制地免费用于任何用途”。许可规定了数据如何使用、修改和分发。常见例子包括：

- **知识共享 (CC)：** 常见的变体包括CC0（公共领域奉献）、CC-BY（要求署名）和CC-BY-SA（要求署名并以相同许可共享修改）。
- **开放数据公共领域 (ODC)：** 像ODbL（开放数据库许可）或ODC-By（类似于CC-BY）这样的许可专门用于数据库。
- **自定义许可：** 某些数据集可能拥有由发布组织制定的独有许可。

务必仔细审查数据集*及其各个组成部分*所附带的特定许可。有些汇编数据混合了来自不同原始许可来源的数据。不遵守许可条款可能导致法律问题，尤其是在商业环境中。

### 值得关注的开放许可数据集

几个大型文本语料库已成为LLM预训练 (pre-training)的标准资源。以下是一些著名例子：

1. **The Pile：** 由EleutherAI开发，The Pile是一个825 GiB的英文文本数据集，由22个不同的小型数据集汇编而成。其目的是创建一个广泛且多样的语料库，适用于通用语言模型。来源包含学术论文（PubMed Central、arXiv）、网络文本（Common Crawl子集）、书籍（Project Gutenberg、Books3）、代码（GitHub）、对话（Stack Exchange）等。虽然该汇编旨在采用宽松许可，但底层许可因来源而异，用户需要针对其具体用例检查合规性。
2. **C4 (Colossal Cleaned Common Crawl)：** C4最初是为T5模型创建的，它来自Common Crawl网络档案。它经过了大量的筛选和清洗，包括删除样板文本、使用黑名单排除冒犯性语言、文档去重以及主要保留英文文本。生成的数据集约为750GB，并以ODC-By许可发布。它对清洗的侧重使其成为一个受欢迎的起点，尽管清洗过程本身可能会过滤掉某些有用的文本类型，或反映清洗启发式方法的偏见。
3. **ROOTS (Responsible Open-science Open-collaboration Text Sources)：** 这个1.6TB的多语言语料库是为训练BLOOM模型而创建的。它汇集了来自59种语言的498个数据源。在文档化来源和许可方面投入了大量精力，以负责任的方式进行。它为训练多语言模型提供了宝贵的资源。
4. **其他来源：** 除了这些大型汇编，还有许多更小或更专业的开放数据集存在：

   - **维基百科：** 定期发布的数据快照可用，提供多种语言的高质量百科全书文本。常被包含在像The Pile这样的大型数据集中。
   - **古腾堡计划：** 提供数万本公共领域书籍。
   - **arXiv：** 一个科学预印本库，主要涉及物理、数学、计算机科学等。是技术语言的很好来源。
   - **GitHub：** 虽然有效过滤具有挑战性，但它是各种编程语言代码的大量来源。常被处理并包含在像The Pile这样的数据集中。

### 以编程方式访问数据集

Hugging Face的`datasets`等库简化了访问和使用许多流行的开放数据集，包括The Pile、C4和ROOTS。该库处理下载、缓存、处理和流式传输，与PyTorch等深度学习 (deep learning)框架结合得很好。

以下是使用`datasets`加载和查看C4数据集一小部分的简单示例：

```python
import torch
from datasets import load_dataset

try:
    c4_dataset = load_dataset(
        'allenai/c4',
        'en',
        split='train',
        streaming=True
    )
except Exception as e:
    print(
        f"加载数据集出错。您可能需要通过"
        f"`huggingface-cli login`登录。错误：{e}"
    )

    c4_dataset = None

if c4_dataset:

    sample_size = 5
    sampled_data = list(c4_dataset.take(sample_size))

    print(f"从C4中抽取了{len(sampled_data)}个示例：")
    for i, example in enumerate(sampled_data):
        print(f"\n--- 示例 {i+1} ---")
        print(f"URL: {example.get('url', 'N/A')}")

        text_snippet = example.get('text', '')[:300]
        print(f"文本片段: {text_snippet}...")
        print(f"时间戳: {example.get('timestamp', 'N/A')}")
```

对于多TB级别的数据集，强烈推荐使用`streaming=True`。它允许您迭代数据，而无需在本地下载和存储整个数据集，而是按需获取数据块。`datasets`库提供了强大的映射函数（`.map()`），可以在流式传输期间即时应用分词 (tokenization)和其他预处理步骤。

### 使用开放数据集时的注意事项

虽然方便，但使用现有数据集需要仔细考虑：

- **许可合规性：** 仔细检查您预期用途的许可条款（研究用途还是商业用途，分发要求）。对于像The Pile这样的汇编，请核实您所依赖的特定组件的许可。
- **数据质量和偏见：** 创建者应用的预处理步骤（例如C4的清洗）会引入其自身的偏见，并且可能与您的需求不完全一致。务必检查数据并考虑是否需要额外的过滤或清洗。请注意，这些数据集反映了其原始来源中固有的偏见。
- **数据集重叠：** 许多大型数据集都来源于Common Crawl或维基百科等常见来源。在比较使用不同语料库训练的模型时，或构建评估集以避免数据污染时，这种重叠值得注意。
- **任务适用性：** 数据集的组成会影响生成模型的能力。确保所选数据集与您的LLM的目标领域或任务相符。您可能需要组合多个开放数据集或使用自定义数据进行补充，以达到所需的性能特征。这与第九章讨论的数据混合策略直接关联。

总而言之，开放许可数据集是LLM开发的重要资源。与从头开始相比，它们提供了大量文本数据，并可能降低工程开销。然而，负责任的使用需要仔细关注许可、数据质量、潜在偏见以及对预期应用的适用性。像Hugging Face的`datasets`这样的库使得访问和处理这些资源的实际操作变得更加容易。

获取即时帮助、个性化解释和交互式代码示例。

---

### 数据获取的法律考量

# 数据获取的法律考量

获取数TB的文本数据是一项工程难题，而处理围绕这些数据的法律问题同样重要，尤其是考虑到LLM预训练 (pre-training)所涉及的规模。不当使用数据可能导致巨大的法律风险、声誉损害，并可能使训练中的大量投入作废。作为这些模型的工程师，了解基本的法律界限并非要成为律师，而是要实行负责任的数据获取做法，并识别潜在的危险信号。

### 版权与合理使用考量

大多数创意作品，包括网站文本、书籍和文章，在创作时即自动受到版权保护。这赋予创作者复制、分发和创建衍生作品的独占权利。训练大型语言模型（LLM）可以说涉及复制（将数据复制到训练集中）并可能创建衍生作品（模型的输出）。

未经许可使用受版权保护的材料构成侵权，除非适用例外情况。在美国，最相关的例外是“合理使用”。合理使用会考量以下因素：

1. 使用目的和性质（例如，商业用途对比非营利教育用途）。
2. 受版权保护作品的性质。
3. 使用部分的数量和实质性。
4. 使用对受版权保护作品潜在市场的影响。

合理使用原则在LLM训练中的应用，目前是激烈法律争论和活跃诉讼的焦点。支持合理使用的观点常围绕训练过程（使用文本提取统计模式而非展示原始内容）以及AI进步的公共利益。反对观点则强调复制的数据量以及LLM可能生成与原始作品竞争的输出。作为一名工程师，您应假定使用受版权保护的数据存在风险，并且合理使用抗辩并非万无一失。记录您的数据来源及其包含的理由是很重要的。

### 数据许可

除了默认的版权保护外，数据常在特定许可下共享。理解这些许可非常必要。

- **宽松的开放许可：** 通常优先选择如知识共享零（CC0）、CC BY、MIT或Apache 2.0等许可。
  - `CC0`：公共领域奉献，限制最少。
  - `CC BY`：要求署名。
  - `CC BY-SA`：要求署名，并要求衍生作品（可能包括使用该数据训练的模型，但这有争议）以相同或兼容的许可（ShareAlike，即“相同方式共享”）发布。
  - `MIT/Apache 2.0`：常用于代码，但有时也适用于数据集，通常宽松，但要求保留许可文本和版权声明。
- **限制性许可：** 某些数据集可能带有禁止商业使用（`CC BY-NC`）、限制衍生作品（`CC BY-ND`）的许可，或带有特定限制的自定义许可。通过私人协议或数据供应商获取的数据也将有合同限制。
- **默示许可：** 有时，数据在没有明确许可的情况下公开发布。其法律地位可能不明确。

维护数据来源并追踪每个进入训练语料库的数据片段所关联的许可，是一项重要的工程任务。这些元数据对于合规审计和风险管理是必要的。简单的追踪可能涉及将许可信息与数据集标识符一起存储。

G

clusterₛources

数据来源

cluster\_corpus

训练语料库

WebCrawl

网络爬取
(混合/未知)

Dataset

聚合数据集
(需追踪许可)

WebCrawl->Dataset

 已过滤

BooksCorpus

图书语料库
(有版权?)

BooksCorpus->Dataset

 已抽样

GitHub

GitHub 代码
(MIT, Apache, GPL...)

GitHub->Dataset

 已处理

Wikipedia

维基百科
(CC BY-SA)

Wikipedia->Dataset

 已包含

> 简化流程图，展示不同许可类型的数据源如何汇入中央训练语料库，并强调追踪的必要性。

### 网络爬取实践

获取网络数据需要技术和法律上的细致考量。

- `robots.txt`：该文件为网络爬虫提供指令。尽管其并非在所有地方都具有法律约束力，但忽视它（尤其是`Disallow`指令）通常被认为是糟糕的做法，可能违反服务条款，并增加法律风险。遵守`robots.txt`是负责任爬取行为的常规部分。

  您可以使用Python的`urllib.robotparser`来检查`robots.txt`：

  ```python
  import urllib.robotparser
  import urllib.request

  robots_url = 'https://example.com/robots.txt'

  user_agent = 'MyLLMDataCrawler/1.0 (+http://mycrawlerinfo.example.com)'

  url_to_check = 'https://example.com/private/data'

  rp = urllib.robotparser.RobotFileParser()

  try:

      with urllib.request.urlopen(robots_url, timeout=10) as response:

          content = response.read().decode('utf-8')
          rp.parse(content.splitlines())

          if rp.can_fetch(user_agent, url_to_check):
              print(f"允许爬取 {url_to_check}")

          else:
              print(f"robots.txt 禁止爬取 {url_to_check}")

  except urllib.error.URLError as e:
      print(f"访问 robots.txt 时出错 {robots_url}: {e}")
  except Exception as e:
      print(f"解析 robots.txt 时出错: {e}")
  ```
- **服务条款 (ToS)：** 网站通常有服务条款页面，列明允许的使用方式。许多网站明确禁止自动化爬取。违反服务条款可能被视为违约，在某些司法管辖区（例如美国《计算机欺诈和滥用法案》- CFAA 下），可能被视为未经授权的访问，尽管CFAA应用于公共网站爬取的情况在法律上也有争议。始终审查主要数据源的服务条款。
- **服务器负载：** 激进的爬取可能使网站服务器过载，可能导致拒绝服务。这不道德，并可能导致IP封禁或法律投诉。实行礼貌的爬取做法：遵守`robots.txt`中的`Crawl-delay`指令，使用适当的用户代理，并实行速率限制。

### 个人可识别信息 (PII)

大型数据集，尤其是网络爬取数据，常包含个人可识别信息（姓名、电子邮件地址、电话号码、财务详情、健康信息）。使用个人可识别信息进行训练会带来显著的隐私风险，并可能违反欧洲的GDPR（通用数据保护条例）或CCPA（加州消费者隐私法案）等法规。

尽管第7章讨论了数据清洗技术，包括使用正则表达式或命名实体识别（NER）模型进行个人可识别信息清除，但大规模实现完美的个人可识别信息移除极其困难。模型记忆并可能“反刍”在训练期间学到的个人可识别信息的风险是一个严重问题。通过仔细选择来源和过滤来最大限度地减少个人可识别信息的摄入是必要的步骤。

### 数据溯源与审慎

最终，负责任的数据获取涉及周密的规划和文档记录。保持清晰的记录，包括：

- 数据集的每个部分来自何处。
- 获取时所依据的许可或法律基础。
- 应用的任何预处理步骤（特别是过滤或个人可识别信息清除）。

这种溯源信息对于确保合规性、调试模型行为（例如，追踪偏差或毒性来源）以及回应潜在的法律询问具有极高价值。大型语言模型训练数据的法律环境是变化的；请咨询熟悉相关司法管辖区数据隐私和知识产权法的法律专家。您作为工程师的职责是构建能够遵守法律要求的系统，并通过严谨的数据处理实践来促进风险管理。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 7 Data Cleaning Preprocessing Pipelines

### 质量筛选策略

# 质量筛选策略

从网络等来源收集的原始文本数据，除了有用信息外，还包含大量噪声。文档可能被截断，主要包含导航链接，由机器生成日志组成，包含大量样板文本，或者只是与目标语言或领域不相关。将此类低质量数据直接送入LLM预训练 (pre-training)，可能会降低性能，浪费计算资源，并可能引入不良偏见或安全问题。因此，实施有效的质量筛选策略是预处理流程中的主要一步。

这些策略从简单的启发式方法到更复杂的基于模型的方法不等。通常，多种方法结合能提供最佳效果，形成一个多阶段的筛选过程。

### 基于启发式方法的筛选

启发式方法是根据对典型低质量文本的观察得出的经验法则。它们通常计算成本较低，并且能有效去除明显的噪声样本。

- **文档长度：** 极短的文档通常缺少有意义的内容（例如，仅仅是标题、页眉/页脚片段或错误消息）。反之，极长的文档可能表明解析错误、文件串联或非散文内容。常见做法是设置最小和最大字符数或词数阈值。

  ```python

  MIN_CHARS = 100
  MAX_CHARS = 100000

  def filter_by_length(text):
      char_count = len(text)

      word_count = len(text.split())
      if char_count < MIN_CHARS or char_count > MAX_CHARS:
          return False

      return True

  document = "This is a sample document..."
  if filter_by_length(document):

      pass
  ```

  具体阈值（如 `MIN_CHARS`、`MAX_CHARS`）是超参数 (parameter) (hyperparameter)，通常需要根据数据集特性和后续目标进行调整。
- **符号和标点符号比例：** 主要由非字母数字字符组成的文本通常表示源代码、配置文件、表格或损坏的文本。计算字母字符与总字符的比例，或特定符号（如 `#`、`{`、`}`）的比例，有助于识别此类内容。

  ```python

  MIN_ALPHA_RATIO = 0.7

  def filter_by_alpha_ratio(text):
      if not text:
          return False
      alpha_chars = sum(c.isalpha() for c in text)
      total_chars = len(text)
      ratio = alpha_chars / total_chars
      return ratio >= MIN_ALPHA_RATIO

  code_snippet = "if (x > 0) { y = x * 2; } // 示例代码"
  prose = "This sentence contains mostly alphabetic characters."

  print(f"Code snippet passes filter: {filter_by_alpha_ratio(code_snippet)}")
  print(f"Prose passes filter: {filter_by_alpha_ratio(prose)}")
  ```
- **“不良词汇”或样板短语的存在：** 创建常见于低质量网络内容的特定词汇或短语列表（例如，“javascript is required”、“terms of use”、错误消息、像“lorem ipsum”这样的占位符文本），允许直接进行筛选。正则表达式可用于匹配与样板相关的模式，如长串导航链接或版权声明。必须注意不要让这些列表过于严格，因为它们可能会无意中过滤掉有效内容。
- **重复内容：** 包含过度重复（例如，同一行或段落重复多次）的文档通常是低质量的。这可以通过查找重复的行、段落，甚至长n-gram来检测。计算文档的压缩比（使用gzip等标准算法）也可以作为一种近似方法；高度重复的文档压缩效果极好。

### 统计和基于模型的筛选

这些方法通常运用文本的统计属性或预训练 (pre-training)模型来评估质量。

- **语言识别：** 对于单语或特定多语模型，识别每个文档的语言非常必要。`fastText` 或 `pycld3` 等库提供预训练的语言识别模型。不匹配目标语言的文档或模型分配较低置信度分数的文档可以被筛选掉。

  ```python

  ```

# 使用语言识别库的示例

```
# import language_identifier_lib

# model = language_identifier_lib.load_model()
TARGET_LANG = 'en'
MIN_CONFIDENCE = 0.90

def filter_by_language(text):
    # detected_lang, confidence = model.predict(text)
    # return detected_lang == TARGET_LANG and confidence >= MIN_CONFIDENCE
    # 仅为说明而进行的模拟实现：
    if "esto es espa\u00f1ol" in text:
         return False # 模拟非目标语言
    if "confidence low" in text:
         return False # 模拟低置信度
    return True # 否则假定为英语

spanish_text = "esto es espa\u00f1ol"
low_confidence_text = "mixed signals confidence low maybe"
english_text = "This is primarily English text."

print(f"Spanish text passes: {filter_by_language(spanish_text)}")
print(f"Low confidence text passes: {filter_by_language(low_confidence_text)}")
print(f"English text passes: {filter_by_language(english_text)}")
# 预期输出：
# Spanish text passes: False
# Low confidence text passes: False
# English text passes: True
```
```

- **困惑度筛选：** 在高质量参考语料库（例如维基百科）上训练的语言模型，会为其训练数据相似的文本分配较低的困惑度（或较高的概率）。我们可以使用这样的模型对原始数据集中的文档进行评分。超过特定困惑度阈值的文档可能与参考语料库不相似，并且潜在地质量较低。这需要访问合适的质量参考语言模型，并且可能计算密集。

  ```python

  ```

# 使用预训练模型进行困惑度评分的示例

```
import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# 假设模型和分词器已加载
# model = AutoModelForCausalLM.from_pretrained("gpt2") # 示例
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
PERPLEXITY_THRESHOLD = 50 # 示例阈值

def calculate_perplexity(text, model, tokenizer):
    # 占位符：实际实现包括编码、通过模型、计算损失并转换为困惑度。
    # inputs = tokenizer(text, return_tensors="pt")
    # with torch.no_grad():
    #     outputs = model(**inputs, labels=inputs["input_ids"])
    # loss = outputs.loss
    # perplexity = torch.exp(loss)
    # return perplexity.item()
    # 模拟实现：
    if "gibberish" in text:
        return 100.0
    else:
        return 20.0
```

# 示例用法

```
# high_quality_text = "This is well-formed text."
# low_quality_text = "asdf qwer gibberish zxcv."
# if calculate_perplexity(document, model, tokenizer) < PERPLEXITY_THRESHOLD:
#     # 保留文档
#     pass
```
```

- **基于分类器的筛选：** 如果有区分高质量和低质量文档的小型标注数据集可用或可以创建，则可以训练一个分类模型。特征可以从简单的TF-IDF向量 (vector)到预训练模型的嵌入 (embedding)向量不等。训练好的分类器随后会预测大型数据集中每个文档的质量分数，从而允许根据此分数进行筛选。这需要标注工作，但可以捕获比单独使用启发式方法更不明显的质量指标。

### 在流程中结合策略

没有单一的筛选策略是完美的。有效的数据清洗通常涉及在流程中将多个筛选器串联起来。一种常见的方法是首先使用计算成本较低的启发式方法（长度、语言识别）来快速去除明显噪声，随后对剩余数据采用困惑度或基于分类器筛选等更耗资源的方法。

G

RawData

原始文本数据

LenFilter

长度筛选器

RawData->LenFilter

LangFilter

语言识别筛选器

LenFilter->LangFilter

BoilerplateFilter

样板文本筛选器

LangFilter->BoilerplateFilter

QualityModel

模型筛选器
(困惑度/分类器)

BoilerplateFilter->QualityModel

CleanData

清洗后的数据

QualityModel->CleanData

> 文本数据的典型多阶段筛选流程。

### 注意事项

- **偏见放大：** 筛选规则，特别是那些基于启发式方法或在可能存在偏见的参考语料库上训练的模型，可能会无意中删除来自特定人群、领域或方言的有效文本，从而放大最终数据集中的偏见。分析被丢弃的数据以了解可能引入的偏见非常重要。
- **阈值调整：** 设置合适的长度、比例、置信度分数或困惑度阈值通常取决于数据集，需要仔细实验。评估不同阈值对后续模型性能的影响，并分析每个阶段被筛选掉的文档类型。
- **计算成本：** 虽然启发式方法成本低，但将大型语言模型或复杂分类器应用于数TB或数PB数据进行筛选，需要大量计算资源。权衡筛选的严格性与处理时间/成本。

成功运用这些质量筛选策略，是准备构建高性能大型语言模型所需的大量（但通常杂乱）数据集的不可或缺的环节。它为后续的规范化和去重等预处理步骤做好了铺垫。

获取即时帮助、个性化解释和交互式代码示例。

---

### 文本标准化方法

# 文本标准化方法

在过滤掉原始文本中明显的质量问题后，为大型语言模型准备数据的下一步通常是文本标准化。标准化的目的是通过减少那些不会大幅改变意义，但可能导致数据稀疏或模型不一致的变体来规范文本。可以将其看作是消除表面差异，以更统一地呈现基础内容。这个过程通常在分词 (tokenization)之前进行，因为标准化选择直接影响分词器 (tokenizer)如何拆分文本并构建其词汇表 (vocabulary)。

应用一致的标准化对于降低模型需要处理的复杂性很重要。没有标准化，像“U.S.A.”、“USA”和“usa”这样的变体可能会被视为不同实体，从而分裂模型的理解。然而，标准化是一种平衡之举；过度激进的标准化会消除有意义的区别。

### 常用标准化步骤

让我们看看LLM数据处理流程中经常使用的一些标准文本标准化方法。

#### 大小写折叠

将文本转换为单一大小写（通常是小写）是最简单和最常见的标准化步骤之一。它确保像“Apple”（公司名）和“apple”（水果）这样的词最初被同等对待，从而减少有效词汇量大小，并避免模型需要为句子开头或标题中使用的大写变体学习单独的表示。

```python
import torch

text = "The Quick Brown Fox Jumps Over The Lazy Dog."
lower_text = text.lower()
print(f"原始: {text}")
print(f"小写: {lower_text}")
```

虽然大小写折叠通常有益，但它也有缺点。它会消除潜在有用的区别，例如区分公司名“Apple”和水果“apple”，或者区分缩写词“IT”（信息技术）和单词“it”。对于在海量、多样化数据集上训练的通用LLM，词汇量减少的好处通常超过了大小写信息丢失的坏处，因为模型无论如何都可能从语境中推断出这些信息。然而，对于特定用途或任务，保留大小写可能是必要的。

#### Unicode 规范化

从各种来源抓取的文本数据通常包含以多种方式在 Unicode 中表示的字符。例如，像“é”这样的带重音字符可以表示为单个预组字符（U+00E9），也可以表示为基础字符“e”后跟一个组合尖音符（U+0065 U+0301）。尽管视觉上相同，但如果未经标准化，这些不同的字节序列可能会被后续处理视为不同实体。

Python 中的 `unicodedata` 模块提供了标准化的形式：

- **NFC (规范化形式组合):** 尽可能组合字符。'e' + '´' 变为 'é'。这通常是网页内容的默认选择。
- **NFD (规范化形式分解):** 将字符分解为基础字符和组合标记 (token)。'é' 变为 'e' + '´'。
- **NFKC (规范化形式兼容性组合):** 进行兼容性分解，将兼容性字符替换为其规范等效项，然后进行规范组合 (NFC)。这种方式更激进，可能改变视觉外观或含义（例如，连字如 'ﬁ' 变为 'f' + 'i'，分数 '½' 变为 '1/2'）。
- **NFKD (规范化形式兼容性分解):** 进行兼容性分解，然后进行规范分解 (NFD)。

对于 LLM 训练，**NFC** 是一个安全基线，它确保了规范表示，而不会大幅改变内容。**NFKC** 有时用于更强的标准化，通过合并视觉变体可能进一步简化词汇，但需要谨慎，因为它在少数情况下会改变语义。

```python
import unicodedata

char_nfc = 'é'
char_nfd = 'e\u0301'

print(f"NFC 字符串: '{char_nfc}', 长度: {len(char_nfc)}")
print(f"NFD 字符串: '{char_nfd}', 长度: {len(char_nfd)}")
print(f"它们是否相等? {char_nfc == char_nfd}")

normalized_nfc_1 = unicodedata.normalize('NFC', char_nfc)
normalized_nfc_2 = unicodedata.normalize('NFC', char_nfd)

print(
    f"规范化 NFC 1: '{normalized_nfc_1}', 长度: {len(normalized_nfc_1)}"
)
print(
    f"规范化 NFC 2: '{normalized_nfc_2}', 长度: {len(normalized_nfc_2)}"
)
print(
    f"规范化形式是否相等? {normalized_nfc_1 == normalized_nfc_2}"
)

ligature = 'ﬁ'
normalized_nfkc = unicodedata.normalize('NFKC', ligature)
print(f"原始连字: '{ligature}'")
print(f"NFKC 规范化后: '{normalized_nfkc}'")
```

选择正确的 Unicode 规范化形式取决于数据的特性和所需的标准化程度。NFC 通常建议作为起始点。

#### 处理变音符号（移除重音）

有时，会移除重音或其他变音符号，以进一步简化文本，将像 'é'、'ê'、'è' 这样的字符都映射到 'e'。这比大小写折叠或标准 Unicode 规范化更激进。

```python
import unicodedata

def remove_accents(input_str):

    nfkd_form = unicodedata.normalize('NFD', input_str)

    return "".join([
        c for c in nfkd_form
        if not unicodedata.category(c) == 'Mn'
    ])

text_with_accents = "El niño juega al fútbol en el café."
text_without_accents = remove_accents(text_with_accents)

print(f"原始: {text_with_accents}")
print(f"重音符号已移除: {text_without_accents}")
```

尽管这简化了文本，但对于重音在语音学上有重要意义（区分含义）的语言，这可能会有问题，例如法语（'pêche' - 桃子 vs. 'péché' - 罪）或西班牙语（'año' - 年 vs. 'ano' - 肛门）。对于包含此类语言的多语言模型或数据集，通常不建议移除重音，因为它会丢弃重要的语言信息。它的使用仅在目标应用明确需要对重音不敏感的匹配，或者源数据在重音方面质量极差时才合理。

#### 空白符标准化

原始文本中常见不一致的空白符（多余的空格、制表符、换行符）。空白符标准化包括：

1. 清除行或整个文档的首尾空白符。
2. 将多个连续的空白符（空格、制表符、换行符）替换为单个空格。
3. 有时，标准化换行符（例如，将 `\r\n` 或 `\r` 转换为 `\n`）。

```python
import re

text_with_bad_whitespace = "  This text \t has  extra \n\n whitespace.  "

stripped_text = text_with_bad_whitespace.strip()

normalized_whitespace_text = re.sub(r'\s+', ' ', stripped_text)

print(f"原始: '{text_with_bad_whitespace}'")
print(f"规范化后: '{normalized_whitespace_text}'")
```

这一步确保间距一致，有助于分词 (tokenization)，并避免模型基于任意空白符变体学习到虚假模式。

### 在LLM环境中的考虑事项

- **对分词 (tokenization)的影响:** 标准化直接影响分词器 (tokenizer)的输入。小写化减少所需词汇量。Unicode 规范化避免视觉上相同的字符获得不同的词元 (token)。过度激进的标准化（例如删除所有标点符号或重音）可能使子词 (subword)分词器更难找到有意义的单元，或者错误地合并单词。
- **信息丢失:** 总要权衡标准化的好处与有用信息潜在的丢失。大小写、标点符号和变音符号可以承载语义权重 (weight)。对于大型通用模型，通常最好执行最小限度的标准化（例如 NFC、基本空白符清理、可能小写化），并让模型通过接触大量数据和语境来学习处理变体。
- **语言依赖性:** 标准化规则可能因语言而异。例如，大小写或重音的重要性在不同语言中差异很大。多语言数据集需要仔细考量，可能需要对每种语言应用不同的规则，或选择最小化、与语言无关的标准化。
- **流程集成:** 这些标准化步骤通常在初始清理（如HTML标签移除）和质量过滤之后，但在分词之前执行。它们通常作为大型数据处理流程的一部分来实现，处理数TB文本时可能会使用分布式计算框架来实现可扩展性。

深思熟虑地应用这些文本标准化方法有助于创建更干净、更一致的数据集，这反过来有助于更稳定的训练和可能更好的大型语言模型性能。核心在于一致性以及对所涉权衡做出明智选择。

获取即时帮助、个性化解释和交互式代码示例。

---

### 处理冗余内容与标记删除

---

### 近似重复和精确重复检测

# 近似重复和精确重复检测

从Common Crawl等来源抓取的海量数据集不可避免地包含冗余内容。某些重复可能反映自然语言模式，但过多的重复，无论是精确的还是近似的，都可能对大型语言模型（LLM）的训练产生负面影响。这会浪费计算资源多次处理相同信息，可能使模型对数据分布的理解产生偏差，并可能放大重复内容中存在的偏见。因此，移除重复项是数据预处理中一个标准且重要的步骤。

### 精确重复检测

识别字节级别完全相同的文件相对简单。最常见的方法是为每个文件的内容计算一个加密哈希值（如SHA-256或MD5）。具有相同哈希值的文件被认为是精确重复项。

```python
import hashlib
import os

def get_document_hash(doc_content):
  """计算文件内容的SHA-256哈希值。"""

  if isinstance(doc_content, str):
    doc_content = doc_content.encode('utf-8')
  return hashlib.sha256(doc_content).hexdigest()

hashes_seen = set()
duplicates_to_remove = []
data_directory = "/path/to/your/text/files"

for filename in os.listdir(data_directory):
    filepath = os.path.join(data_directory, filename)
    if os.path.isfile(filepath):
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
                doc_hash = get_document_hash(content)

                if doc_hash in hashes_seen:
                    duplicates_to_remove.append(filepath)

                else:
                    hashes_seen.add(doc_hash)
        except Exception as e:
            print(
                f"处理 {filepath} 时出错: "
                f"{e}"
            )

print(
    f"根据内容哈希，识别出 {len(duplicates_to_remove)} 个精确重复项。"
)
```

这种基于哈希的方法虽然对相同内容有效，但也有局限性。首先，为数十亿或数万亿个文件存储和比较哈希值需要大量的内存或分布式存储及查找基础设施。其次，更重要的是，它无法识别*几乎*相同的文件，这些文件可能仅在格式、时间戳、用户名、广告或措辞上略有不同。这些*近似重复项*仍然代表冗余信息。

### 近似重复检测

处理近似重复项需要从精确哈希转向测量内容*相似性*的方法。挑战在于如何高效地完成这项工作，而无需比较每个文件对，因为这在计算上是不可行的（对于 nnn 个文件，复杂度为 O(n2)O(n^2)O(n2)）。MinHash结合局部敏感哈希（LSH）等技术提供了一种概率性的、可扩展的解决方案。

#### 分片：将文件表示为集合

第一步是将每个文件转换为适合相似性比较的表示形式。一种常用技术是*分片*（shingling），它将文件分解为一组重叠的标记 (token)或字符序列，称为片（shingles）或k-grams。

例如，考虑句子“the quick brown fox”。使用字符3-gram（长度为 k=3k=3k=3 的片）：
`{"the", "he ", "e q", " qu", "qui", "uic", "ick", "ck ", "k b", " br", "bro", "row", "own", "wn ", "n f", " fo", "fox"}`

选择片的类型（字符 vs. 单词）和长度（kkk）会影响粒度。字符片对细微的单词变化更具弹性，而单词片则捕获更多语义结构。字符片的典型 kkk 值可能是5到10。

```python
def get_character_shingles(text, k=5):
  """为字符串生成一组字符k-gram（片）。"""
  text = text.lower()
  shingles = set()
  if len(text) < k:
      return shingles
  for i in range(len(text) - k + 1):
    shingles.add(text[i:i+k])
  return shingles

doc1 = "The quick brown fox jumps over the lazy dog."
doc2 = "The quick brown fox jumped over the lazy dogs."
doc3 = "A completely different sentence."

shingles1 = get_character_shingles(doc1, k=5)
shingles2 = get_character_shingles(doc2, k=5)
shingles3 = get_character_shingles(doc3, k=5)

print(f"文件1的片（示例）：{list(shingles1)[:5]}")
print(f"文件2的片（示例）：{list(shingles2)[:5]}")
```

#### Jaccard相似度

一旦文件被表示为片集（S1S\_1S1​, S2S\_2S2​），它们的相似性就可以使用Jaccard相似系数来量化 (quantization)：

J(S1,S2)=∣S1∩S2∣∣S1∪S2∣J(S\_1, S\_2) = \frac{|S\_1 \cap S\_2|}{|S\_1 \cup S\_2|}J(S1​,S2​)=∣S1​∪S2​∣∣S1​∩S2​∣​

这衡量了集合交集的大小除以它们并集的大小。值为1表示集合完全相同，而0表示它们没有共同元素。直接计算这仍然需要比较片集，而片集可能非常大。

```python

intersection_size = len(shingles1.intersection(shingles2))
union_size = len(shingles1.union(shingles2))
jaccard_1_2 = intersection_size / union_size if union_size > 0 else 0

intersection_size_1_3 = len(shingles1.intersection(shingles3))
union_size_1_3 = len(shingles1.union(shingles3))
jaccard_1_3 = intersection_size_1_3 / union_size_1_3 if union_size_1_3 > 0 else 0

print(f"Jaccard(文件1, 文件2): {jaccard_1_2:.4f}")
print(f"Jaccard(文件1, 文件3): {jaccard_1_3:.4f}")
```

#### MinHash：估算Jaccard相似度

MinHash是一种巧妙的技术，它允许我们估算Jaccard相似度，而无需明确计算潜在巨大片集的交集和并集。它依赖于对片进行哈希处理并观察最小哈希值。

其核心思想是：如果我们对集合 S1S\_1S1​ 和 S2S\_2S2​ 中的所有片应用一个随机哈希函数 hhh，那么从 S1S\_1S1​ 获得的最小哈希值与从 S2S\_2S2​ 获得的最小哈希值相同的概率等于它们的Jaccard相似度。

P(min⁡(h(S1))=min⁡(h(S2)))=J(S1,S2)P(\min(h(S\_1)) = \min(h(S\_2))) = J(S\_1, S\_2)P(min(h(S1​))=min(h(S2​)))=J(S1​,S2​)

为什么会这样？考虑组合集 S1∪S2S\_1 \cup S\_2S1​∪S2​。如果我们从这个并集中随机选择一个元素 xxx，那么 xxx 也属于交集 S1∩S2S\_1 \cap S\_2S1​∩S2​ 的概率正好是 J(S1,S2)J(S\_1, S\_2)J(S1​,S2​)。应用随机哈希函数并选择最小哈希值，就像根据哈希排序随机采样一个元素。对于并集 S1∪S2S\_1 \cup S\_2S1​∪S2​ 生成最小哈希值的片必须属于 S1S\_1S1​ 或 S2S\_2S2​（或两者）。只有当那个特定的、哈希值最小的片存在于它们的交集中时，它才会为两个集合产生*相同*的最小值。

为了得到可靠的估计，我们不只使用一个哈希函数。相反，我们使用 mmm 个不同的哈希函数（h1,h2,...,hmh\_1, h\_2, ..., h\_mh1​,h2​,...,hm​）。对于每个文件，我们计算其片通过这 mmm 个函数获得的最小哈希值。这为每个文件生成一个由 mmm 个最小哈希值组成的“签名”向量 (vector)。

Signature(S)=[min⁡(h1(S)),min⁡(h2(S)),...,min⁡(hm(S))]\text{Signature}(S) = [\min(h\_1(S)), \min(h\_2(S)), ..., \min(h\_m(S))]Signature(S)=[min(h1​(S)),min(h2​(S)),...,min(hm​(S))]

两个文件 S1S\_1S1​ 和 S2S\_2S2​ 之间的Jaccard相似度估计值，就是它们签名向量中最小哈希值匹配位置的比例。

像 `datasketch` 这样的库提供了高效的实现。

```python

from datasketch import MinHash, MinHashLSH

num_perm = 128

m1 = MinHash(num_perm=num_perm)
for shingle in shingles1:
  m1.update(shingle.encode('utf8'))

m2 = MinHash(num_perm=num_perm)
for shingle in shingles2:
  m2.update(shingle.encode('utf8'))

m3 = MinHash(num_perm=num_perm)
for shingle in shingles3:
  m3.update(shingle.encode('utf8'))

print(f"估计的 Jaccard(文件1, 文件2): {m1.jaccard(m2):.4f}")
print(f"估计的 Jaccard(文件1, 文件3): {m1.jaccard(m3):.4f}")
```

#### 局部敏感哈希（LSH）：高效查找相似签名

即使有了MinHash签名，比较所有签名对（O(n2)O(n^2)O(n2) 次比较）对于数十亿个文件来说也太慢了。局部敏感哈希（LSH）是一种用于高效查找可能相似的*候选*文件对的技术，它显著减少了所需的明确比较次数。

LSH的工作原理是对MinHash签名本身进行哈希处理，使得相似的签名更有可能哈希到同一个桶中。该策略涉及将MinHash签名（长度为 mmm）分成 bbb 个带（band），每个带包含 rrr 行（因此 m=b×rm = b \times rm=b×r）。

对于每个带，我们对其包含的签名部分进行哈希处理。如果两个文件在至少一个带中哈希到*同一个桶*，它们就被认为是候选对。

- 如果两个文件具有高的Jaccard相似度，它们的MinHash签名很可能在许多位置匹配。极有可能至少有一个完整的带会完全匹配，从而使它们在该带中哈希到同一个桶并成为候选对。
- 如果两个文件具有低的Jaccard相似度，它们的MinHash签名很可能在许多位置不同。任何单个带内的所有 rrr 个值都不太可能完全匹配，这使得它们在任何带中哈希到同一个桶的可能性很小。

通过调整参数 (parameter) bbb 和 rrr，我们可以平衡找到真正近似重复项的概率（召回率）与被标记为候选的非相似对的数量（精确率）。增加 bbb（更多带，每个带的行数 rrr 更少）会使LSH标准更严格，减少误报，但可能遗漏一些真阳性。减少 bbb（更少带，每个带的行数 rrr 更多）会使文件更容易成为候选，提高召回率，但也会增加需要验证的候选对数量。

```python

threshold = 0.8

lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

lsh.insert("doc1", m1)
lsh.insert("doc2", m2)
lsh.insert("doc3", m3)

result_doc1 = lsh.query(m1)
print(f"文件1的近似候选项: {result_doc1}")

result_doc3 = lsh.query(m3)
print(f"文件3的近似候选项: {result_doc3}")
```

### 实现和扩展

在大型语言模型预训练 (pre-training)数据集所需的规模下实现近似重复检测涉及多个考量：

1. **参数 (parameter)调整：** 片长度（kkk）、哈希函数数量（mmm）以及LSH带/行（b,rb, rb,r）的选择取决于数据集的特点和所需的相似度阈值。这些通常需要在数据样本上进行经验调整。常用值包括字符的 k=5..9k=5..9k=5..9， m=128..512m=128..512m=128..512，以及选择 b,rb, rb,r 以达到特定的Jaccard阈值（例如0.8或0.85）。
2. **分布式处理：** 处理TB或PB级数据需要Apache Spark或Dask等分布式框架。工作流程通常包括：
   - 并行读取分区中的文件。
   - 在分区内为每个文件生成片和MinHash签名。
   - 在分区内应用LSH分带和哈希，或使用分布式哈希表/连接。
   - 收集候选对。
   - （如果需要）对候选对执行最终的Jaccard相似度检查。
   - 根据选定的策略（例如，保留最旧的、保留最短的、随机选择）筛选出已识别的重复项。
3. **内存管理：** MinHash签名紧凑，但LSH索引仍可能变得很大。可能需要高效的数据结构以及基于磁盘的LSH实现。

有效移除精确和近似重复项可以确保最终训练数据集更加多样化，减少冗余，并可能减轻重复内容中存在的偏见的放大。这会带来更高效的训练，并且通常会使模型具有更好的泛化能力。

获取即时帮助、个性化解释和交互式代码示例。

---

### 语种识别与过滤

# 语种识别与过滤

在处理大量文本语料库时，特别是那些从网络抓取的大型语料库（如Common Crawl），你不可避免地会遇到多种语言的文档。尽管存在多语言模型，但目标通常是训练一个专注于一种或特定几种语言的模型。即使是对于多语言模型，了解语言分布对于数据抽样和评估也很重要。因此，识别每个文档的语言并进行相应过滤是预处理流程中一个常规且必要的步骤。

这个流程有助于确保模型主要在相关数据上进行训练，从而提升目标语言的性能，并减少由不需要的语言或字符引入的干扰。它还避免了分词 (tokenization)器 (tokenizer)（为一种语言优化）对其他语言文本处理不当，从而导致低效表示的情况。

### 语种识别工具

有一些可用的库用于自动语言检测。它们通常依赖统计方法，通过分析字符n-gram或其他特征来预测最有可能的语言。一些常用的选项包括：

1. **fastText:** 由Facebook AI研究开发，`fastText` 提供预训练 (pre-training)模型，速度很快且通常准确，特别是在较长的文本片段上。它支持多种语言。它的速度使其适用于处理大量数据集。安装通常需要C++编译工具。
2. **langdetect:** 一个流行的纯Python库，基于谷歌的语言检测库。它通常易于安装和使用，但对于非常大的数据集，可能比`fastText`或`pycld2`等编译型替代方案慢。其准确性在短文本或有噪声的文本上有时不太可靠。
3. **pycld2:** 谷歌Compact Language Detector 2 (CLD2)的Python绑定，以其速度和准确性著称，在识别单个文档中的多种语言方面特别有效。安装可能需要C++编译器。
4. **langid.py:** 另一个纯Python选项，旨在平衡速度和准确性，支持大量语言。

选择工具通常需要权衡准确性要求、处理速度、集成便捷性（例如，纯Python与C++依赖）以及你需要识别的具体语言。对于大规模LLM数据准备，`fastText`和`pycld2`因其性能而常被选择。

让我们来看一个使用`fastText`的基本示例。首先，你需要安装它（`pip install fasttext`）并从`fastText`网站下载一个预训练的语言识别模型（例如`lid.176.bin`）。

```python
import fasttext

try:
    model_path = 'lid.176.bin'
    model = fasttext.load_model(model_path)
except ValueError as e:
    print(f"模型加载错误: {e}")
    print("请确保模型文件 'lid.176.bin' 已下载并可访问。")

    model = None

def identify_language(text_document):
    """
    使用fastText识别文本文档的语言。

    参数:
        text_document (str): 输入文本。

    返回:
        tuple: 包含预测的语言代码
               （例如 '__label__en'）和置信度分数，
               如果模型失败或文本为空，则返回 (None, 0.0)。
    """
    if not model or not text_document:
        return None, 0.0

    processed_text = text_document.replace('\n', ' ')

    predictions = model.predict(processed_text, k=1)

    if predictions and predictions[0]:
        language_code = predictions[0][0]
        confidence = predictions[1][0]
        return language_code, confidence
    else:
        return None, 0.0

text_en = "This is an example of English text."
text_fr = "Ceci est un exemple de texte en fran\u00e7ais."
text_es = "Este es un ejemplo de texto en espa\u00f1ol."
text_short = "Ok"
text_mixed = "This text mixes English and un peu de fran\u00e7ais."

for text in [text_en, text_fr, text_es, text_short, text_mixed]:
    lang_code, score = identify_language(text)
    if lang_code:

        lang = lang_code.replace('__label__', '')
        print(f"文本: '{text[:30]}...' -> "
              f"语言: {lang}, 置信度: {score:.4f}")
    else:
        print(f"文本: '{text[:30]}...' -> "
              f"未能识别语言。")
```

运行这段代码（假设你已经有了`fastText`模型）将输出每个示例的预测语言（如`en`、`fr`、`es`）和置信度分数。请注意，非常短或混合语言的文本可能会产生较低的置信度分数或潜在的错误分类。

### 集成到流程中

在以Apache Spark或Dask等工具进行大规模数据处理流程中（如在可扩展预处理流程部分中提及），语言识别成为应用于每个文档的转换步骤。

```python

def filter_by_language(
    document,
    target_languages=['en'],
    min_confidence=0.85
):
    """
    检查文档识别出的语言是否在目标列表中
    并且满足置信度阈值。
    """
    language_code, confidence = identify_language(document)
    if language_code:
        lang = language_code.replace('__label__', '')
        if lang in target_languages and confidence >= min_confidence:
            return True
    return False

filtered_documents_rdd = documents_rdd.filter(
    lambda doc: filter_by_language(
        doc,
        target_languages=['en', 'de'],
        min_confidence=0.80
    )
)
```

这段代码展示了如何将 `identify_language` 函数集成到分布式数据流程中，以根据语言和置信度过滤文档。

### 过滤策略和置信度阈值

这一步骤的一个重要方面是决定*如何*进行过滤。常见策略包括：

- **保留单一目标语种:** 只保留被识别为主要目标语言（例如，英语）且具有足够置信度的文档。
- **保留多个目标语种:** 保留被识别为属于预定义目标语言集合（例如，英语、德语、法语）的文档。
- **处理不确定性:** 决定如何处理语言检测置信度低的文档。选项包括丢弃它们、标记 (token)它们以供人工审核（如果可行），或者如果丢失潜在相关数据的成本很高，则可能保留它们。
- **多语种文档:** 一些工具（如`pycld2`）可以识别文档中的多种语言。如果主要语言与你的目标匹配，或者文本的很大一部分是目标语言，你可能会保留这类文档。

置信度阈值的选择也很重要。高阈值（例如 > 0.95）会提高精度（保留的错误标记文档较少），但可能会降低召回率（如果模型置信度不高，可能会丢弃更多正确文档）。较低的阈值会增加召回率，但可能会引入更多来自错误分类文档的噪声。这个阈值通常需要根据检测工具的质量和特定数据集进行调整。

G

RawData

原始文本
文档

LangID

语种
识别

RawData->LangID

Filter

过滤逻辑
(例如，lang=='en'
& score > 0.85)

LangID->Filter

TargetData

目标语言
数据集 (例如，英语)

Filter->TargetData

通过

OtherData

其他语种
(丢弃/存档)

Filter->OtherData

不通过

> 流程图，说明了数据预处理流程中的语种识别和过滤阶段。

英语德语法语西班牙语其他0102030405060阶段过滤前过滤后 (保留英语)语言分布检测到的语言文档百分比

> 数据集中语言分布示例，显示过滤英语文档前后的情况。

### 注意事项和限制

- **短文本准确性:** 大多数语言识别工具在非常短的文本（例如，单个词、短语）上表现不佳，因为没有足够的统计证据。
- **噪声:** 格式不佳的文本、过多的标记 (token)或OCR错误会干扰准确的语言检测。这表明在初次清理步骤（如移除标记）之后执行语言识别的重要性。
- **语码转换:** 包含大量多种语言的文档很难明确分类。
- **计算成本:** 对数十亿或数万亿个token进行语言识别需要高效的工具和大量计算资源。对文档进行抽样进行语言识别可能是一个办法，但可能会遗漏一些不同语言的片段。
- **模型偏差:** 语言识别模型本身可能会表现出偏见，在某些语言上的表现优于其他语言，这可能会导致过滤后的数据集出现偏差。

有效地识别和过滤语言是构建高质量数据集的重要一步，该数据集是针对你计划构建的大型语言模型的具体要求而定制的。它减少了噪声，提高了训练效率，并有助于确保模型学习与目标语言相关的模式。

获取即时帮助、个性化解释和交互式代码示例。

---

### 构建可扩展的预处理流水线

# 构建可扩展的预处理流水线

处理PB级文本数据，不仅仅需要巧妙的过滤算法；它还需要一套能够应对信息庞大体量和高速率的基础设施。对于Common Crawl或大型内部语料库这类数据集，在单台机器上顺序执行标准的数据预处理步骤，包括清洗、过滤、规范化和去重，是不可行的。因此，处理流水线必须明确为扩展性设计，采用分布式计算方法。

这些可扩展的流水线通常将复杂的预处理任务分解为一系列独立的阶段，在机器集群上执行。Apache Spark和Dask等工具常用于此目的，它们为分布式数据结构和计算提供抽象功能，并提供长时间运行任务不可或缺的容错机制。

## 选择合适的框架

Apache Spark凭借其弹性分布式数据集（RDD）和更高级的DataFrame API（PySpark），擅长大规模批处理。它管理数据分布、在工作节点间调度任务以及处理节点故障。Dask提供类似功能，侧重于与Python生态系统（NumPy、pandas、scikit-learn）原生集成，并提供更灵活的调度选项。

让我们考虑一个简单的过滤任务：删除短于特定长度的文档。在PySpark中，这可能看起来像这样：

```python

from pyspark.sql.functions import length

MIN_LENGTH = 100

filtered_df = raw_documents_df.filter(length(raw_documents_df.text) >= MIN_LENGTH)
```

Dask使用其DataFrame API提供类似的使用感受：

```python

import dask.dataframe as dd

MIN_LENGTH = 100

filtered_ddf = ddf[ddf['text'].str.len() >= MIN_LENGTH]
```

这些示例说明了如何在分布式数据结构上表达操作，从而使得框架能够在集群中并行执行。

## 设计流水线阶段

典型的预处理流水线不仅仅是单个脚本，而是操作的有向无环图（DAG）。每个阶段执行特定的转换，通常读取前一阶段处理过的数据，并为下一阶段写入结果。

考虑一个处理网络爬取数据的流水线：

1. **加载原始数据：** 读取原始文件（例如，来自Common Crawl的WARC文件，存储在HDFS或S3中）。
2. **初始提取与清理：** 提取纯文本，移除样板HTML/JavaScript，执行基本的Unicode规范化。
3. **语言识别：** 识别每份文档的主要语言。
4. **语言过滤：** 仅保留被识别为目标语言的文档。
5. **质量过滤：** 应用启发式或基于模型的过滤器，以删除低质量内容（例如，基于长度、停用词比例、小型模型的困惑度分数）。
6. **去重：** 使用MinHash LSH等技术识别并删除近似重复的文档。
7. **最终规范化：** 应用所有剩余的规范化步骤（例如，特定字符替换）。
8. **分词 (tokenization)（可选）：** 有时分词作为最后的预处理步骤执行，存储分词后的数据以便在训练期间更快地加载。否则，它由训练数据加载器即时完成。
9. **保存处理后的数据：** 将清理、过滤后的数据存储为Parquet等高效格式。

这可以如下图所示：

G

Load

加载原始数据
（例如，WARC）

Extract

提取文本和
基本清理

Load->Extract

LangID

语言识别

Extract->LangID

LangFilter

语言过滤

LangID->LangFilter

QualityFilter

质量过滤

LangFilter->QualityFilter

Dedupe

去重
(MinHash LSH)

QualityFilter->Dedupe

FinalNorm

最终规范化

Dedupe->FinalNorm

Save

保存处理后的数据
（例如，Parquet）

FinalNorm->Save

> 一个典型的数据预处理流水线，说明了从原始数据加载到保存处理后的文本的顺序阶段。

## 可扩展地实现阶段

让我们看看一些主要阶段如何在分布式环境中实现：

- **过滤和规范化：** 这些任务通常“易于并行化”。每个文档通常可以独立处理。分布式框架使用map操作高效处理此任务。您可以定义Python函数（Spark中的用户定义函数或UDF，或Dask中通过`map_partitions`应用的函数），封装对单个文档或一批文档进行清洗、规范化或应用质量启发式方法的逻辑。

  ```python

  from pyspark.sql.functions import udf
  from pyspark.sql.types import StringType
  import unicodedata

  def normalize_text(text):
      if text is None:
          return None

      return unicodedata.normalize('NFKC', text)

  normalize_udf = udf(normalize_text, StringType())

  normalized_df = filtered_df.withColumn("normalized_text", normalize_udf(filtered_df.text))
  ```
- **去重：** 使用文档内容的哈希比较来查找完全重复项相对简单，然后进行分布式分组或去重操作即可。然而，检测*近似重复项*需要更精巧的技术，例如MinHash结合局部敏感哈希（LSH）。

  分布式去重的MinHash LSH核心思路是：

  1. **分片：** 将每份文档表示为一组短字符序列（分片，例如5-gram）。
  2. **MinHashing：** 为每份文档的分片集计算固定数量的哈希值（MinHash签名）。具有相似分片集的文档很可能具有相似的MinHash签名。
  3. **LSH分带：** 将签名组件分组到带中。如果文档在至少一个完整带上匹配，则认为它们是潜在的重复项。这显着减少了所需的比较次数。
  4. **候选配对：** 生成在至少一个LSH带中匹配的文档对。
  5. **验证：** 计算候选对的精确相似度（例如，Jaccard相似度），并根据阈值进行过滤。

  高效地实现这一点通常涉及Spark中的多个MapReduce风格阶段或Dask中的等效操作，同时仔细管理数据混洗。存在诸如`spark-deduplication`之类的库，或者您可以利用Spark的MLlib LSH功能来实现，或使用Dask Bags/DataFrames构建自定义逻辑。
- **语言识别：** `pycld2`、`langdetect`等库或`fastText`的模型可以集成。由于这些通常按文档操作，因此它们非常适合map操作。然而，在每个工作任务上加载语言识别模型可能效率不高。策略包括广播模型（如果足够小）或使用`mapPartitions`为每个分区/工作器加载一次模型。

  ```python

  import dask.dataframe as dd

  def identify_language_partition(partition_df):

      langs = []
      for text in partition_df['text']:
          if text and isinstance(text, str) and len(text.strip()) > 20:

              pred = fasttext_model.predict(text.replace('\n', ' '))
              lang = pred[0][0].replace('__label__', '') if pred[0] else 'und'
          else:
              lang = 'und'
          langs.append(lang)
      partition_df['language'] = langs
      return partition_df

  lang_identified_ddf = filtered_ddf.map_partitions(
      identify_language_partition,
      meta=filtered_ddf._meta.assign(language=str)
  )
  ```

## 优化和效率考量

构建真正可扩展的流水线不仅仅是使用分布式框架。

- **中间存储：** 避免将中间结果写入纯文本文件或CSV。使用Apache Parquet或Apache Arrow等列式格式。这些格式提供高效的压缩、编码，并支持谓词下推（允许阶段只读取必要的数据）。
- **分区：** 数据在节点和任务间如何拆分（分区）至关重要。确保数据合理地良好分区，通常通过哈希相关键或使用文件块边界。分区不佳可能导致“拖延”任务，从而拖慢整个阶段（数据倾斜）。重新分区可能是必要的，但这会引入网络混洗成本。
- **缓存：** 对于多次重用的中间DataFrame或RDD（例如，用于质量过滤和去重的已清理数据集），将其缓存到集群内存或磁盘中（Spark中的`.cache()`或`.persist()`，Dask中的`.persist()`）可以显着加快后续阶段的速度。
- **资源管理：** 调整执行器/工作器数量、每个工作器的核心数和内存分配对性能很重要。这通常需要根据特定的集群硬件和处理任务的性质（CPU密集型与I/O密集型）进行试验。
- **批处理：** 在应用复杂函数（例如运行模型进行质量过滤或语言识别）时，在每个任务内部批量处理数据，而不是逐条处理，以分摊函数调用开销或模型加载时间。

## 整合（流水线）

这是一个显示链式操作的简化PySpark结构：

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, length, col
from pyspark.sql.types import StringType, BooleanType

def complex_quality_filter(text):

    if text is None: return False
    has_enough_words = len(text.split()) > 10

    return has_enough_words

quality_filter_udf = udf(complex_quality_filter, BooleanType())

def normalize_text(text):

    if text is None: return None
    return text.lower().strip()

normalize_udf = udf(normalize_text, StringType())

input_path = "s3://my-bucket/data/extracted_text/"
df = spark.read.parquet(input_path)

df_filtered_basic = df.filter(length(col("text")) > 50)

df_filtered_quality = df_filtered_basic.filter(quality_filter_udf(col("text")))

df_normalized = df_filtered_quality.withColumn("processed_text", normalize_udf(col("text"))) \
                                   .select("doc_id", "processed_text")

output_path = "s3://my-bucket/data/processed_text/"
df_normalized.write.mode("overwrite").parquet(output_path)
```

构建这些流水线需要仔细规划和反复迭代。从少量数据样本开始开发和调试各个阶段，然后再扩展到集群上的完整数据集。Spark或Dask提供的监控工具对于找出瓶颈和优化大型数据集上的性能非常有帮助。目标是创建一个可重复且高效的过程，将原始文本转换为大型语言模型的高质量数据源。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 8 Building Managing Large Scale Datasets

### 数据存储格式（文本、Arrow、Parquet）

# 数据存储格式（文本、Arrow、Parquet）

大型语言模型需要海量文本语料库，这些语料库在训练期间必须高效存储并快速访问。此类语料通常会经过抓取、清洗和预处理。选择合适的存储格式是一个根本性决定，它会影响存储成本、I/O性能以及数据加载流程的整体效率。虽然简单格式容易开始使用，但随着数据集扩展到TB或PB级别，专门格式的益处将越来越明显。

我们来考察常见选择：纯文本文件、Apache Arrow和Apache Parquet。

### 纯文本格式（例如，`.txt`、`.jsonl`）

最直接的方法通常是将文本数据存储在纯文本文件中。每行可能代表一个文档，或者您可以使用JSON Lines（`.jsonl`）等格式，其中每行都是一个表示数据记录的有效JSON对象（例如，包含文本和元数据）。

**优点：**

- **简单性：** 易于手动创建、检查和调试。标准文本处理工具（`grep`、`sed`、`awk`）可直接使用。
- **普适性：** 几乎任何编程语言或工具无需特殊库即可读取。

**缺点：**

- **低效：** 文本解析（特别是复杂的JSON）在数据加载期间可能占用大量CPU资源。
- **压缩：** 可以应用Gzip或Zstandard等标准压缩，但由于列内数据相似性，列式格式通常能实现更好的压缩比。
- **无模式/类型：** 格式本身不强制执行数据类型或结构，如果解析假设被违反，可能导致后续错误。
- **随机访问慢：** 访问特定记录需要从头开始扫描和解析，或维护单独的索引。

一个典型的`.jsonl`文件可能如下所示：

```json
{
    "text": "The quick brown fox jumps over the lazy dog.",
    "source": "wikipedia",
    "id": "doc_001"
}
{
    "text": "Large language models require immense amounts of text data.",
    "source": "common_crawl",
    "id": "cc_abc"
}
{
    "text": "...",
    "source": "...",
    "id": "..."
}
```

在Python中读取这个文件很简单，但对于非常大的文件来说可能很慢，特别是当反序列化复杂的JSON时：

```python
import json
import gzip

def read_jsonl_gz(filepath):
    """从一个gzipped JSON Lines文件中读取记录。"""
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:

                print(f"正在跳过格式错误的行：{line.strip()}")
```

对于较小的数据集或初步原型来说尚可接受，但逐行解析文本的CPU开销会成为瓶颈，当向多个GPU提供高速消耗数据的LLM训练时。

### Apache Arrow

Apache Arrow是一个内存列式数据格式标准。它旨在提高分析查询性能，并以最小的序列化/反序列化开销（通常是零拷贝读取）在系统和语言之间高效交换数据。

**理念：**

- **列式布局：** 数据按列而非按行组织。特定字段（例如“text”字段）的所有值在内存中连续存储。
- **零拷贝读取：** 进程通常可以直接访问内存中的Arrow数据结构而无需复制，这显著加快了数据访问速度，尤其是在进程间通信中（例如数据加载进程和主训练进程之间）。
- **语言无关：** 许多语言（Python、Java、C++、Rust等）都有官方库，使其非常适合涉及不同技术的流程。
- **丰富的数据类型：** 支持全面的数据类型集，包括嵌套结构、时间戳和数值类型。

**优点：**

- **快速读取/扫描：** 列式布局对缓存友好，并支持向量 (vector)化操作（使用SIMD指令），使顺序读取速度非常快。
- **高效互操作性：** 非常适合在进程之间（例如使用共享内存）或库之间传递数据（例如Pandas DataFrames可以几乎即时地转换为Arrow表或从Arrow表转换）。
- **内存效率：** 对于相同数据，通常比Python对象表示更节省内存。

**缺点：**

- **内存侧重：** 尽管Arrow有文件格式（`.arrow`或`.feather`），但它主要针对内存表示进行优化。Parquet通常更适合磁盘上的持久性、压缩存储。
- **写入开销：** 构建Arrow数组有时比写入纯文本的开销更高。
- **可读性差：** 二进制格式，需要特定工具或库来检查。

Arrow在Hugging Face `datasets`等库的*底层*被大量使用。当您使用`datasets`加载或映射数据集时，它通常在内部使用Arrow表进行缓存和快速访问。

```python
import pyarrow as pa
import pyarrow.feather as feather
import time

texts = pa.array([d['text'] for d in data], type=pa.string())
ids = pa.array([d['id'] for d in data], type=pa.string())

table = pa.Table.from_arrays([texts, ids], names=['text', 'id'])

feather.write_feather(table, 'my_dataset.arrow')

start_time = time.time()
read_table = feather.read_table('my_dataset.arrow')
end_time = time.time()
print(f"Arrow读取时间：{end_time - start_time:.4f} 秒")

text_column = read_table['text']
```

当您流程的多个阶段都使用Arrow时，才能体现其真正的效用，从而避免昂贵的序列化步骤。在Spark上运行的数据预处理任务可以直接将Arrow文件输出到云存储，然后PyTorch `DataLoader`可以使用`pyarrow`高效读取这些文件。

### Apache Parquet

Apache Parquet是一个广泛采用的列式*存储*格式，针对大规模数据仓库和分析进行了优化，对于在磁盘上存储LLM数据集也非常有效。

**理念：**

- **磁盘列式存储：** 和Arrow一样，它逐列存储数据，但专为磁盘持久化设计（例如HDFS、S3、本地文件系统）。
- **压缩与编码：** 通过应用适合每列数据类型和特点的编码（如字典编码、行程长度编码），然后进行通用压缩（Snappy、Gzip、Zstd、Brotli），实现了出色的压缩比。
- **模式演进：** 支持模式演进，允许您稍后添加列而无需重写旧数据。
- **谓词下推：** 存储系统通常可以在将数据加载到内存*之前*，直接在Parquet文件内部根据查询谓词过滤行（尽管这对于分析查询比顺序训练读取更相关）。

**优点：**

- **存储效率：** 与文本格式相比，显著节省磁盘空间，降低存储成本和网络传输时间。
- **快速顺序扫描：** 高效读取训练所需的列，跳过不相关的列。如果存在其他元数据，只读取“text”列比读取整个行快得多。
- **生态系统整合：** 在大数据框架（Spark、Dask、Presto、Hive）和Python库（`pandas`、`pyarrow`）中都有出色的支持。

**缺点：**

- **写入复杂性：** 由于编码和压缩，写入Parquet文件比纯文本涉及更多的计算开销。
- **不可读性：** 二进制格式，需要特定工具。
- **行级操作：** 与基于行的格式相比，对于需要一次性访问整行的操作效率较低（但由于I/O节省，通常总体上仍然更快）。

Parquet通常是最终处理数据集的首选格式，该数据集将在训练期间重复读取。

```python
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

pq.write_table(table, 'my_dataset.parquet', compression='snappy')

start_time = time.time()

read_parquet_table = pq.read_table('my_dataset.parquet', columns=['text'])
end_time = time.time()
print(
    f"Parquet读取时间（仅文本列）："
    f"{end_time - start_time:.4f} 秒"
)
```

### 对比与建议

以下是权衡的汇总：

| 特点 | 文本 (.txt, .jsonl) | Apache Arrow | Apache Parquet |
| --- | --- | --- | --- |
| **类型** | 基于行 | 列式（内存中） | 列式（磁盘上） |
| **主要用途** | 简单性，调试 | 快速内存分析，进程间通信 | 高效存储，分析 |
| **可读性** | 是 | 否 | 否 |
| **压缩** | 外部（Gzip, Zstd） | 有限（进程间通信通常是原始数据） | 优秀（内置） |
| **读取速度** | 慢（解析开销） | 非常快（零拷贝） | 快（列式扫描） |
| **写入速度** | 快 | 中等 | 中等-慢（编码） |
| **CPU使用（读取）** | 高（解析） | 低 | 低-中等（解码） |
| **随机访问** | 差 | 中等（内存中） | 中等（行组） |
| **生态系统** | 通用 | 增长中（Pandas, Spark） | 优秀（大数据） |

> 大型文本数据集常见数据存储格式的比较。

**对LLM数据集的建议：**

1. **中间处理：** 在复杂预处理流程的各阶段之间，使用Apache Arrow进行高效数据传输，尤其是在使用Spark或Dask等工具，或在Python进程间传递数据时。Hugging Face `datasets`等库大量使用Arrow进行缓存和内存映射。
2. **训练的最终存储：** 将最终的、大型的、已清洗的数据集以Apache Parquet文件格式存储在分布式文件系统（如S3或HDFS）上。其压缩功能显著降低存储成本，其列式特性允许在训练期间高效读取仅必要的列（通常仅为文本标记 (token)），从而最大限度地提高I/O吞吐量 (throughput)。使用Snappy或Zstandard等高效压缩方式。
3. **小型数据集/调试：** 对于小型实验或当人工可读性对于调试特定数据点非常重要时，纯文本或JSON Lines是可接受的。

为存储在云存储或HDFS上的大规模训练数据选择Parquet，并通过使用Arrow进行内存表示的库（如`datasets`或使用`pyarrow.parquet`的自定义`DataLoader`）进行读取，为将数据馈送给您的分布式训练任务提供了高效的根本。这最大限度地减少了I/O瓶颈和存储占用，使您能够将计算资源集中在模型训练本身上。

获取即时帮助、个性化解释和交互式代码示例。

---

### 分布式文件系统 (HDFS, S3)

# 分布式文件系统 (HDFS, S3)

处理以太字节甚至拍字节衡量的数据集时，将它们存储在单台机器的文件系统上会变得不切实际，甚至不可能。庞大的数据量超出了本地磁盘容量，而在分布式训练期间，多个机器（工作节点）需要并发读取这些数据，这会造成一个很大的瓶颈。分布式文件系统（DFS）和云对象存储提供了处理这些挑战所需的基础设施，它们具有可扩展性、高可靠性和并行访问能力。

这些系统与本地文件系统根本不同，它们将数据分布在多个物理存储节点上，同时向用户或应用程序呈现统一的视图。这种分布实现了横向扩展；随着数据增长，你可以增加更多存储节点，而不会受到单台机器容量的限制。

### LLM分布式存储的核心要点

两个主要特点使分布式存储适合大规模数据管理：

1. **可扩展性和性能：** 分布式系统可以存储比任何单个节点多得多的数据。它们通常设计为高吞吐量 (throughput)，允许许多客户端（例如你的训练工作节点）并行读取数据块。这对于在训练期间持续为昂贵的GPU提供数据供给至关重要。特别是云对象存储，按需提供几乎无限的可扩展性。
2. **可靠性和容错性：** 存储拍字节的数据会增加遇到硬件故障（磁盘故障、节点中断）的可能性。分布式系统通过在多个节点上复制数据块（HDFS常见）或使用纠删码技术（对象存储常见）来降低这种风险。如果一个节点发生故障，数据仍然可以从其他副本重建或取回，从而数据持久性并防止因存储故障导致的训练中断。

### Hadoop分布式文件系统 (HDFS)

HDFS起源于Apache Hadoop生态系统，专为存储具有流式数据访问模式的超大文件而设计。它采用主/从架构：

- **NameNode（名称节点）：** 管理文件系统命名空间（元数据，如目录结构、文件权限和数据块位置）。它是中心协调器。
- **DataNodes（数据节点）：** 在本地磁盘上存储实际数据块，并向NameNode报告。

HDFS通常会将每个数据块（例如，128MB或256MB的块）在不同的DataNodes上复制三份以实现容错。

**优点：**

- 对大型顺序读取具有出色的吞吐量 (throughput)，非常适合处理大型数据文件。
- 成熟的生态系统集成，特别是与Apache Spark进行数据预处理。
- 可以使用商用硬件在本地部署。

**缺点：**

- NameNode可能成为性能瓶颈或单点故障（尽管存在高可用性配置）。
- 对随机访问或小文件工作负载的优化较少。
- 需要主动的集群管理和维护。

当LLM数据预处理流程严重依赖现有Hadoop/Spark基础设施，或者组织管理大型本地集群时，HDFS常被使用。

### 云对象存储 (AWS S3, GCS, Azure Blob Storage)

像Amazon S3（简单存储服务）、Google Cloud Storage（GCS）和Azure Blob Storage这样的云对象存储服务已经成为在云中存储大型数据集的事实标准。它们与传统文件系统的运作方式不同：

- **扁平命名空间：** 数据作为对象存储在容器中（S3/GCS中称为“桶”，Azure Blob Storage中称为“容器”）。对象通过唯一键识别（通常类似文件路径，例如`my-data/processed/part-0001.arrow`）。
- **API驱动：** 主要通过HTTP API（GET, PUT, LIST, DELETE）进行交互。
- **高持久性和可用性：** 这些是托管服务，旨在提供极高的持久性（例如，99.999999999%或“十一个九”）和可用性，将底层硬件管理和复制抽象化。

G

cluster\_compute

计算节点 (GPU)

clusterₛtorage

分布式存储

gpu1

工作节点 1

dfs

云对象存储（例如S3）
或 HDFS

gpu1->dfs

读取()

gpu2

工作节点 2

gpu2->dfs

读取()

gpu3

工作节点 N

gpu3->dfs

读取()

> 多个计算工作节点从中心分布式存储系统并发访问数据。

**优点：**

- 大规模可扩展性和弹性（按使用量付费）。
- 由云服务提供商管理的高持久性和可用性。
- 与自管理HDFS相比，运维负担更小。
- 与基于云的计算实例和训练平台直接集成。
- 支持数据分层以优化成本（例如，将旧数据移动到更便宜、不常访问的层级）。

**缺点：**

- 性能可能受计算与存储之间网络延迟的影响。
- 最终一致性模型（尽管强一致性越来越普遍）在某些读写后场景中可能需要仔细处理。
- 如果将大量数据移出云，数据传输成本（出站流量）可能是一个需要考虑的因素。

对于大多数基于云的LLM开发，对象存储是首选，因为它的可扩展性、易用性以及与更广泛的云生态系统的集成。

### 训练期间访问分布式存储

训练框架需要有效的方法从这些分布式系统中读取数据。在训练前简单地将数太字节的数据下载到每个工作节点的本地磁盘是不切实际的。相反，数据通常在训练期间直接从分布式存储进行流式传输。

像`fsspec`（文件系统规范）这样的库为各种存储后端提供了统一的Python接口，包括本地文件、HDFS、S3、GCS和Azure Blob Storage。PyTorch的`DataLoader`可以与利用这些库的自定义`Dataset`实现结合使用。

这里有一个使用`torch.utils.data.IterableDataset`从存储在S3类桶中的多个文本文件逐行流式传输数据的例子：

```python
import torch
import s3fs
import gzip
from torch.utils.data import IterableDataset, DataLoader

class S3StreamingTextDataset(IterableDataset):
    def __init__(self, bucket_name, prefix, shuffle_files=True):
        super().__init__()
        self.fs = s3fs.S3FileSystem()
        self.bucket = bucket_name
        self.file_paths = self.fs.glob(f"{bucket_name}/{prefix}/*.txt.gz")
        self.shuffle_files = shuffle_files
        print(f"在"
              f"s3://{bucket_name}/{prefix}" f"中找到了 {len(self.file_paths)} 个文件")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files_to_process = self.file_paths

        if worker_info is not None:

            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            files_to_process = [
                f for i, f in enumerate(self.file_paths)
                if i % num_workers == worker_id
            ]

        if self.shuffle_files:

             import random
             random.shuffle(files_to_process)

        for file_path in files_to_process:
            print(f"工作节点 {worker_info.id if worker_info else 0}: "
                  f"正在处理 {file_path}")
            try:

                with self.fs.open(file_path, 'rb') as f_remote:
                    with gzip.open(f_remote, 'rt', encoding='utf-8') as f_gz:
                        for line in f_gz:

                            yield line.strip()
            except Exception as e:
                print(f"工作节点 {worker_info.id if worker_info else 0}: "
                      f"处理 {file_path} 时出错: {e}")

                continue
```

这个例子展示了流式传输：每个工作进程（`DataLoader`中的`num_workers`）被分配S3中文件的一个子集，并直接从对象存储逐行读取，同时进行解压。这避免了将整个数据集下载到本地，并允许在多个工作节点和文件之间进行并行处理。更完善的实现可能会以更大的块读取数据，使用像Apache Arrow或Parquet（在上一节中讨论过）这样的优化文件格式，并实施更强大的混洗策略。

选择合适的分布式存储系统是重要的第一步。对于现在开始的大多数项目，特别是那些利用云计算的项目，S3、GCS或Azure Blob等托管对象存储提供了可扩展性、持久性和易用性的吸引人组合，并且与专为流式访问设计的现代分布式训练框架和数据加载器很好地集成。

获取即时帮助、个性化解释和交互式代码示例。

---

### 数据索引用于高效检索

# 数据索引用于高效检索

当处理TB或PB级别的数据集时，这些数据集可能分散存储在分布式文件系统或对象存储中的数千个文件中，顺序扫描整个数据集以查找特定记录或子集会变得非常缓慢且成本高昂。想象一下，您可能只需要检索来自特定网站的文档，或者在预处理阶段被标记 (token)为高质量的文档；读取每一个字节的数据显然不切实际。这就是数据索引成为一种必不可少的技术，用于大规模数据的高效管理。

在此背景下，索引的作用类似于关系数据库中的索引或书本末尾的索引。它是一种辅助数据结构，能够根据特定条件快速查找数据记录，无需彻底扫描主要数据源。对于LLM数据集，索引通常涉及创建从元数据属性或记录标识符到相应数据物理位置的映射。

### 为什么索引对LLM数据集很重要

有效的索引直接支持LLM开发生命周期中的多项重要操作：

1. **有针对性的数据检索：** 您可能需要检查训练期间导致问题的特定示例，或者检索日志中提及的与唯一ID关联的所有文档。索引能让您快速定位（文件路径、字节偏移）这些特定记录的位置。
2. **高效子集划分和过滤：** 预训练 (pre-training)通常涉及组合来自不同源（网页文本、代码、书籍）的数据。您可能希望训练一个模型版本，只使用代码数据，或者过滤掉在预处理期间（第7章）识别出的低于特定质量阈值的文档。基于`source`、`quality_score`或`language`等元数据构建的索引，能让您在不进行完整数据集扫描的情况下，识别出相关文件或记录。
3. **促进复杂的采样策略：** 如第9章所讨论的，训练效果可能取决于数据源的组合。采样策略可能需要提高某些数据源的权重 (weight)，或根据文档属性实施课程学习。索引使得数据加载器能够高效地识别和检索符合这些复杂采样要求的批次。例如，如果您可以通过索引快速找到每个源的记录，那么检索一个由60%网页数据、30%书籍和10%代码组成的批次会简单得多。
4. **数据集分析和查看：** 在训练之前或期间，您可能需要分析数据集的组成。诸如“有多少文档超过2048个token？”或“Common Crawl子集中的语言分布如何？”这类查询，如果长度或语言等元数据被索引，查询速度会显著加快。

### 常见的索引策略

可以采用多种策略，通常组合使用，取决于数据格式、存储系统和访问模式。

#### 元数据索引

这可能是最常用的方法。它涉及创建辅助文件或结构，将元数据值映射到拥有这些值的数据记录列表。

- **结构：** 这可以简单到是序列化的Python字典、JSON文件，或者更结构化的格式，例如专门用于索引的Parquet文件。例如，您可能有一个将数据源映射到包含该源数据文件的索引：

  ```python

  ```

# 索引结构（例如，存储为JSON或pickle文件）

```
metadata_index = {
    "common_crawl__file_ABC": {
        "source": "Common Crawl",
        "language": "en",
        "quality_score": 0.85,
        "token_count": 1500,
        "location": {"file": "/path/to/data/part-001.parquet", "row_group": 5}
    },
    "book_corpus_xyz": {
        "source": "Book Corpus",
        "language": "en",
        "quality_score": 0.95,
        "token_count": 35000,
        "location": {"file": "/path/to/data/part-099.parquet", "row_group": 12}
    },
    # ... 数百万或数十亿更多条目
}

# 使用示例：查找高质量的Common Crawl文档
cc_high_quality_docs = []
for doc_id, metadata in metadata_index.items():
    if metadata["source"] == "Common Crawl" and metadata["quality_score"] > 0.9:
        cc_high_quality_docs.append(doc_id) # 或者直接存储位置

print(f"找到 {len(cc_high_quality_docs)} 个高质量的Common Crawl文档。")
```
```

- **字段：** 常见的索引元数据字段包括文档ID、源数据集、语言、质量分数、文档长度（或token计数）以及主题分类（如果可用）。

#### 偏移量索引

当数据未以整齐的可按行寻址的格式存储时（例如单个文本文件或不可分割容器中的记录），偏移量索引变得重要。该索引存储大型文件中每个逻辑记录的精确起始字节位置和长度。

- **结构：** 通常，这是一个列表或表格，其中每个条目包含 `(记录ID, 文件路径, 起始字节, 字节数)`。
- **使用场景：** 允许在大型文件中直接查找（例如，文档串联的多千兆字节纯文本文件），以读取特定记录而无需读取前面的数据。这通常与元数据索引结合使用；元数据索引可能指向一个记录ID，偏移量索引则将该ID转换为文件位置和字节范围。

G

clusterᵢndex

偏移量索引

cluster\_data

数据文件（例如，在S3/HDFS上）

index

索引条目

记录1:
part-001, 偏移量 0, 长度 512

记录2:
part-001, 偏移量 512, 长度 1024

...

记录100:
part-002, 偏移量 0, 长度 768

...

file1

part-001.bin

记录1

记录2

...

index:s->file1:n

index:s->file1:n

file2

part-002.bin

记录100

记录101

...

index:s->file2:n

> 一个偏移量索引，将记录ID映射到大型数据文件中其精确的字节位置（偏移量和长度）。

#### 发挥数据格式的特点

某些数据格式，例如Apache Parquet，在文件和行组级别内置了对元数据和索引的支持。Parquet在行组内存储列的统计信息（最小值/最大值）。查询引擎可以使用这些元数据来跳过读取整个行组，如果过滤条件不可能匹配该组内的数据（谓词下推）。虽然这不能完全取代所有用例中专门的记录级索引，但借助这些特点可以在数据加载期间显著加快基于索引列的过滤速度。

### 实施考量

- **粒度：** 索引每个单独的文档能提供最精细的控制，但会导致最大的索引大小。在文件或块级别进行索引则更粗略，但索引更小。选择取决于访问模式；对于单个文档的随机抽样，通常需要文档级索引。
- **存储：** 索引可以与分布式文件系统中的数据文件一起存储，或由单独的元数据服务或数据库管理。将它们与数据一起存储简化了部署，但可能需要自定义加载逻辑。
- **格式：** 简单的格式，如序列化的字典或JSON文件，易于创建，但在大规模情况下可能变得难以管理。使用高效的二进制格式如MessagePack，或将索引本身存储为可查询的格式（如Parquet），可以更具扩展性。数据库（SQL或NoSQL）提供了强大的查询能力，但增加了操作开销。
- **创建：** 索引构建通常集成到数据预处理管道的最后阶段（第7章）。如果数据集非常庞大，这个过程本身可能非常耗费资源，并且可能需要分布式处理框架（如Spark或Dask）。
- **一致性：** 对于静态预训练 (pre-training)数据集，索引只需构建一次。在涉及连续训练（第28章）和不断演进的数据集场景中，更新索引的策略变得重要。

### 将索引与数据加载器结合

训练期间，这些索引的主要使用者是数据加载器。在PyTorch等框架中，自定义的`Dataset`实现可以使用索引高效检索特定项目或实施复杂的采样。

```python
import torch
from torch.utils.data import Dataset
import pickle

class IndexedTextDataset(Dataset):
    def __init__(self, index_path, doc_ids_path):
        print(f"正在从 {index_path} 加载索引...")
        with open(index_path, 'rb') as f:
            self.metadata_index = pickle.load(f)
        print(f"正在从 {doc_ids_path} 加载文档ID列表...")
        with open(doc_ids_path, 'rb') as f:
            self.doc_ids = pickle.load(f)
        print("索引和文档列表加载完毕。")

    def __len__(self):
        return len(self.doc_ids)

    def __getitem__(self, idx):

        doc_id = self.doc_ids[idx]

        try:
            metadata = self.metadata_index[doc_id]
            location = metadata['location']
            file_path = location['file']
            offset = location['offset']
            length = location['length']
        except KeyError:

            print(f"警告：在索引中未找到文档ID {doc_id}。")

            return {"text": "", "doc_id": doc_id, "error": True}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.seek(offset)
                text_content = f.read(length)

            return {"text": text_content, "doc_id": doc_id}
        except Exception as e:
            print(f"从 {file_path} 的偏移量 {offset} 读取文档 {doc_id} 时出错: {e}")
            return {"text": "", "doc_id": doc_id, "error": True}
```

这个示例展示了`Dataset`如何接收索引`idx`（从0到`len(self)-1`），将其映射到预采样的`doc_id`，在主索引中查找该`doc_id`的位置，然后从正确的文件在特定偏移量处执行有针对性的读取。这避免了扫描不相关的数据。

### 权衡

实施数据索引涉及权衡：

- **存储成本：** 索引会消耗额外的存储空间，对于非常大的数据集和细粒度索引，可能会增加数千兆字节甚至数太字节。
- **预处理时间：** 构建索引会增加初始数据准备阶段的计算时间。
- **复杂性：** 设计、构建和维护索引系统增加了数据管道的复杂性，相比简单的顺序读取。

然而，收益通常超过成本，特别是对于大规模LLM训练：

- **查询性能：** 显著减少了访问特定记录或子集的时间。
- **效率：** 通过只读取必要数据，降低了训练期间的I/O成本。
- **灵活性：** 使得复杂的采样和过滤策略成为可能，这些策略在完整扫描下是不切实际的。

获取单条记录按来源过滤 (1%)12510251002510002510k无索引（完整扫描）有索引

> 比较有索引和无索引时数据访问操作所需的时间，突出了索引为有针对性检索和过滤提供的显著加速。请注意时间轴上的对数刻度。

总之，数据索引是一种基础方法，用于管理LLM训练所需的海量数据集。它在存储PB级数据与高效访问训练、分析和评估所需的特定数据片段之间架起了一座桥梁，构成可扩展数据处理管道的一个重要组成部分。

获取即时帮助、个性化解释和交互式代码示例。

---

### 数据集版本管理与复现性

# 数据集版本管理与复现性

训练大型语言模型是一个资源消耗大的过程，通常需要在昂贵的硬件集群上运行数周或数月。因此，复现性不仅是一个科学上的理想，更是工程上的必然要求。如果一次训练运行产生了意想不到的结果，或者你需要回顾几个月前某个特定的模型检查点，你必须能够精确重构该次运行所使用的确切数据集状态。仅仅存储数TB的数据是不够的；你需要数据集版本管理的方法。

与Git等系统能很好地处理代码不同，数据集由于其庞大的体积，带来了独有的版本管理难题。在版本控制系统中直接存储多TB数据集的多个完整副本是不切实际且成本过高的。此外，数据集的“版本”不仅仅是原始文件；它包括用于创建最终训练就绪数据的特定预处理代码、筛选参数 (parameter)、分词 (tokenization)设置和抽样策略。

### 为何要进行数据集版本管理？

有效的数据集版本管理在LLM开发背景下提供了几项重要益处：

1. **复现性：** 主要目标。能够重新创建特定训练运行所使用的确切数据，可以让你可靠地复现结果、排查问题（如第24章讨论的损失峰值），并公平地比较不同的实验。
2. **追踪数据演变：** LLM数据集很少是静态的。你可能会引入新的数据源、改进清洗启发式规则，或者修复预处理流程中的错误。版本管理追踪这些变化，让你能够理解数据漂移或修改如何随时间影响模型性能。
3. **协作：** 大型项目通常涉及多位工程师进行数据准备工作。版本管理提供了一个清晰的参考点，确保每个人都使用预期的版本数据来完成他们的任务。
4. **审计和合规性：** 在某些情况下，为了审计或法规目的，可能需要追溯用于训练特定模型版本的确切数据。

### 数据集版本的组成部分

当我们谈论LLM数据集的版本管理时，我们通常需要追踪几个相互关联的组成部分：

- **处理后的数据文件：** 实际的分词 (tokenization)数据文件，通常是分片的，存放在分布式文件系统或对象存储中。
- **预处理代码：** 用于清洗、筛选、归一化 (normalization)和分词的代码的特定版本（例如，Git提交哈希）。
- **配置：** 预处理期间使用的参数 (parameter)，如质量阈值、去重设置、词汇量大小和特殊标记 (token)定义。
- **源数据标识符：** 用于原始数据源的指针或标识符（例如，Common Crawl快照ID、网络抓取日期、源数据集名称和版本）。
- **元数据：** 处理后的数据文件的校验和或哈希值，以确保数据完整性；数据集统计信息（文档数量、标记数量）；以及可能的数据划分信息（训练集/验证集）。

### 数据集版本管理策略

考虑到数据集的规模，版本管理通常涉及管理元数据和指针，而不是直接复制数据本身。以下是一些常见策略：

#### 1. 命名约定和清单文件

一种基本方法是为目录或存储前缀采用规范的命名约定，其中包含版本标识符、时间戳或相关的配置哈希值。例如，处理后的数据集可能位于`s3://my-llm-datasets/processed/v2.1_vocab32k_cc-2023-03/`这样的路径中。

作为补充，你可以创建清单文件（例如，JSON或YAML格式），列出属于特定版本的所有数据文件及其校验和与重要元数据。这个清单文件很小，可以与预处理代码一起轻松地使用Git进行追踪。

```python
import hashlib
import json
import os
from glob import glob

def calculate_sha256(filepath):
  """计算文件的SHA256哈希值。"""
  sha256_hash = hashlib.sha256()
  with open(filepath, "rb") as f:

    for byte_block in iter(lambda: f.read(4096), b""):
      sha256_hash.update(byte_block)
  return sha256_hash.hexdigest()

def create_dataset_manifest(data_dir, manifest_path, version_info):
  """为目录中的文件创建清单文件。"""
  manifest = {
      "version_info": version_info,
      "files": []
  }

  data_files = sorted(glob(os.path.join(data_dir, "*.arrow")))

  print(f"正在为 {len(data_files)} 个文件在 {data_dir} 中生成清单...")

  for filepath in data_files:
    filename = os.path.basename(filepath)
    checksum = calculate_sha256(filepath)
    manifest["files"].append({
        "filename": filename,
        "sha256": checksum,
        "size_bytes": os.path.getsize(filepath)
    })

  with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

  print(f"清单已保存至 {manifest_path}")

dataset_directory = "/path/to/processed_data/v2.1_vocab32k_cc-2023-03"
output_manifest = "/path/to/repo/dataset_manifests/v2.1.json"
git_commit_hash = "a1b2c3d4e5f6"

version_metadata = {
    "dataset_version": "2.1",
    "preprocessing_code_commit": git_commit_hash,
    "source_info": "Common Crawl 2023年3月快照",
    "tokenizer_vocab_size": 32000
}

os.makedirs(os.path.dirname(output_manifest), exist_ok=True)

print("说明性清单生成设置已完成。")
print(f"将处理的文件位于: {dataset_directory}")
print(f"将清单保存至: {output_manifest}")
print(f"相关元数据: {version_metadata}")
```

这个清单（`v2.1.json`）随后可以提交到Git。要使用这个数据集版本，你的训练流程会读取清单，（可选地）验证校验和，并从其存储位置加载列出的文件。

#### 2. 专用数据版本控制工具

DVC（数据版本控制）、Pachyderm和LakeFS等工具是专门设计用于结合Git处理大型数据文件的。它们的操作原理是在Git中存储元数据和指针，而实际数据则存放在外部存储中（如S3、GCS、HDFS，甚至是本地驱动器）。

例如，DVC通过创建小的`.dvc`元文件来工作，这些文件包含有关实际数据文件的信息，包括它们的哈希值和存储位置。这些元文件会提交到Git。

一个典型的工作流程可能如下所示：

1. **追踪数据：** `dvc add s3://my-llm-datasets/processed/v2.1_vocab32k_cc-2023-03`
   - 此命令分析数据、计算哈希值、创建一个`.dvc`文件（例如，`v2.1_vocab32k_cc-2023-03.dvc`），如果数据尚未存在，还可能将其上传到已配置的DVC远程存储。
2. **提交元数据到Git：** `git add v2.1_vocab32k_cc-2023-03.dvc .gitignore; git commit -m "Add dataset v2.1"`
   - 这个小的`.dvc`文件被添加到Git，将这个特定的数据版本与代码库版本关联起来。
3. **获取数据：** `dvc pull v2.1_vocab32k_cc-2023-03.dvc`（或者直接 `dvc pull`）
   - 在另一台机器上或稍后，检出Git提交并运行`dvc pull`会从远程存储下载`.dvc`文件中列出的相应数据文件。

这些工具通常提供超越基本版本管理的功能，例如数据管道和实验追踪集成。

G

clusterₗocal

本地工作区

clusterᵣemote

远程存储

GitRepo

Git仓库
(代码 + .dvc文件)

LocalData

本地数据缓存
(可选)

GitRepo->LocalData

 dvc pull/checkout

RemoteStore

S3 / GCS / 等
(实际大型数据文件)

GitRepo->RemoteStore

 .dvc文件指向

LocalData->RemoteStore

 dvc push/pull

> DVC如何将Git追踪与大型文件存储分离的概述。

#### 3. 云存储版本管理功能

许多云存储服务（如Amazon S3或Google Cloud Storage）提供内置的对象版本管理功能。启用此功能后，当对象被覆盖或删除时，会自动保留其先前版本。尽管启用简单，但这种方法通常缺少清单文件或DVC等专用工具所提供的与代码版本和预处理步骤的明确关联。它主要作为备份和恢复机制，而非用于复杂机器学习 (machine learning)流程的完整成熟数据集版本管理系统。在没有额外追踪机制的情况下，更难识别究竟哪一组对象版本对应于特定的训练运行。

### 关联数据集、代码和实验

真正的复现性需要将版本化数据集与用于训练的特定代码提交以及生成的模型产物和指标关联起来。MLflow、Weights & Biases、Comet ML等实验追踪平台在此处非常有价值。在记录实验时，你应该包含：

- 训练代码的Git提交哈希。
- 所用数据集版本的标识符（例如，清单文件的路径、DVC标签或存储前缀）。
- 该次运行使用的超参数 (parameter) (hyperparameter)。
- 输出模型检查点路径。
- 评估指标。

这创建了一个完整的、可审计的记录，将输入（代码、数据、配置）与输出（模型、指标）连接起来。

总而言之，管理用于LLM训练的庞大数据集不仅仅是存储问题。实施清晰的数据集版本管理方法，无论是通过规范命名和清单还是通过专用工具，对于复现性、调试、协作以及构建可靠的大型语言模型都十分根本。它确保你对数据准备和训练计算的大量投入能够带来可理解和可重复的结果。

获取即时帮助、个性化解释和交互式代码示例。

---

### 用于训练的流式数据加载器

# 用于训练的流式数据加载器

一旦海量文本数据集被清理、处理并存储，通常是在分布式文件系统或云存储中，重点转向如何将这些数据高效地送入分布式训练环境。将多TB级的数据集完全加载到内存中是不可能的。即使在训练开始前将它们完全从磁盘读取也常常不切实际。在这种情况下，流式数据加载器变得极其重要。

流式数据加载器不是预先加载整个数据集（映射式数据集），而是在训练过程中，直接从存储中动态地逐个样本或逐块读取数据。这种方法能保持较低的内存占用，并允许训练几乎立即开始，但它也带来了新的复杂性，尤其是在I/O性能、数据混洗以及分布式环境中的协调方面。

### 对速度的要求：避免I/O瓶颈

现代GPU处理数据的速度非常快。如果数据加载器无法以比GPU消耗更快的速度提供训练批次，昂贵的加速器就会处于闲置状态，浪费计算资源，并大大延长训练时间。这就是I/O瓶颈问题。

流式数据加载器必须设计成：

1. **高效读取：** 对所选存储格式采用优化的读取模式（例如，对HDFS上的文件进行顺序读取，对S3等对象存储进行并行请求）。使用Apache Arrow或Parquet等二进制格式（在“数据存储格式”中介绍）通常比即时解析原始文本文件快得多。
2. **并行加载：** 使用多个CPU工作进程并行获取、解码和预处理数据，使数据加载与GPU计算重叠进行。
3. **预取数据：** 维护一个已准备好的批次队列，供GPU使用，这样当前批次处理完成后，下一个批次即可立即使用。

PyTorch的`torch.utils.data.DataLoader`内置支持并行工作进程（`num_workers`）和预取（`prefetch_factor`），这些对于流式处理来说必不可少。然而，数据如何逐样本获取和处理的核心逻辑位于`IterableDataset`内部。

### PyTorch中的可迭代式数据集

PyTorch区分映射式数据集（实现`__getitem__`和`__len__`）和可迭代式数据集（实现`__iter__`）。对于流式处理大型数据集，`IterableDataset`是自然而然的选择。

`IterableDataset`的`__iter__`方法负责一次生成一个已处理的样本。当与`DataLoader`和多个工作进程（`num_workers > 0`）一起使用时，PyTorch会处理工作负载的分配。每个工作进程都会获得自己的迭代器实例，通常配置为处理数据分片的不同子集。

下面是一个为流式处理设计的`IterableDataset`基本结构：

```python
import torch
import os
import random
from torch.utils.data import IterableDataset, DataLoader

class StreamingTextDataset(IterableDataset):
    def __init__(self, data_dir, shard_pattern="shard_*.txt",
                 shuffle_shards=True):
        super().__init__()
        self.data_dir = data_dir
        self.shard_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".txt")
        ])
        self.shuffle_shards = shuffle_shards

    def _iter_shard(self, shard_file):

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        print(f"Worker {worker_id}: Processing {shard_file}")
        try:
            with open(shard_file, 'r', encoding='utf-8') as f:
                for line in f:

                    processed_sample = line.strip()
                    if processed_sample:
                         yield processed_sample
        except Exception as e:
            print(f"处理分片 {shard_file} 时出错：{e}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        current_shards = self.shard_files

        if self.shuffle_shards:

             g = torch.Generator()

             seed = int(torch.randint(0, 10000, (1,)).item())
             g.manual_seed(seed)
             shard_indices = torch.randperm(
                 len(current_shards), generator=g
             ).tolist()
             current_shards = [self.shard_files[i] for i in shard_indices]

        if worker_info is None:

            worker_id = 0
            num_workers = 1
            shards_for_worker = current_shards
        else:

            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            shards_for_worker = [
                s for i, s in enumerate(current_shards)
                if i % num_workers == worker_id
            ]

        print(
            f"Worker {worker_id}/{num_workers}: "
            f"Assigned {len(shards_for_worker)} shards."
        )

        for shard_file in shards_for_worker:
             yield from self._iter_shard(shard_file)
```

在这个示例中：

1. `__init__` 确定数据分片（例如，单个文件）。
2. `__iter__` 为每个工作进程调用。它根据`worker_info`确定此特定工作进程应处理哪些分片。它还可以选择打乱分片的*顺序*。
3. `_iter_shard` 处理从单个分片读取并生成样本。

### 大规模混洗：混洗缓冲区

真正的随机混洗需要加载整个数据集，这是不可能的。流式加载器通常使用**近似混洗**技术。

- **分片级混洗：** 如代码所示，打乱分片的处理顺序可以提供粗粒度的随机性。一个分片内的所有样本仍然是连续的。
- **混洗缓冲区：** 更有效的方法是维护一个内存中的缓冲区。加载器将样本读入缓冲区，并在生成样本时，从缓冲区中随机选择一个并用从存储中读取的下一个样本替换它。

G

cluster₀

数据存储（分片）

Shard1

分片 1

Reader

读取器
（填充缓冲区）

Shard1->Reader

Shard2

分片 2

Shard2->Reader

ShardN

分片 N

ShardN->Reader

Buffer

混洗缓冲区
（大小 M）

Reader->Buffer

填充

Sampler

采样器
（生成样本）

Buffer->Sampler

随机样本

Sampler->Buffer

替换

Trainer

训练过程

Sampler->Trainer

生成批次

> 混洗缓冲区按顺序读取数据，但从固定大小的内存缓冲区中随机生成样本，提供比仅混洗分片更好的局部混洗效果。

混洗缓冲区的大小（MMM）是一个权衡：更大的缓冲区提供更好的随机性，但每个工作进程消耗更多内存。

### 专用于流式处理的库

尽管可以使用原始PyTorch `IterableDataset`构建流式加载器，但有几个库提供预构建的、高度优化的解决方案，专门用于大规模深度学习 (deep learning)：

- **`webdataset`：** 非常适合以文件序列形式存储的数据，特别是`.tar`归档文件。它提供了灵活的管道用于解码、增强和混洗数据流。它广泛用于多模态 (multimodal)数据集，但也适用于文本。
- **MosaicML `StreamingDataset`：** 为云原生训练而设计。它以其自己的优化格式（MDS）将数据存储在对象存储（如S3）上。它自动处理高效的分片、混洗（使用分片混洗和混洗缓冲区思想），以及在工作进程和节点间的顺畅恢复。它首先要求将数据集转换为MDS格式。
- **Hugging Face `datasets`：** 这个常用的库现在包含了流式处理能力（`load_dataset(..., streaming=True)`）。它允许直接从Hugging Face Hub或本地存储迭代处理大型数据集，而无需先下载所有内容。它与它们的分词 (tokenization)器 (tokenizer)和模型结合良好。

使用这些库可以大大简化开发，并且由于对I/O、缓存和混洗的专门优化，通常能提供更好的性能。

### 恢复与状态管理

长时间的LLM训练任务不可避免地会遇到中断（硬件故障、抢占）。流式数据加载器必须支持**恢复**，这意味着它可以在数据流中精确地从上次中断的位置重新开始。

这要求将数据加载器的状态与模型权重 (weight)和优化器状态一同进行检查点保存（参见第19章：“检查点与容错”）。状态通常包含：

1. 每个工作进程正在处理的最后一个分片的标识符。
2. 在该分片内的确切位置（字节偏移或样本索引）。
3. 每个工作进程的混洗缓冲区内容和状态。
4. 用于混洗的随机数生成器状态。

像`StreamingDataset`这样的库会自动管理此状态。如果构建自定义加载器，则需要在检查点保存时明确地从所有工作进程收集此状态，并在恢复时将其还原。

### 性能调优

- **`num_workers`：** 设置此参数 (parameter)以利用多个CPU核心进行数据加载。最佳值取决于CPU、存储速度和数据处理的复杂程度。可以从每个GPU可用的物理CPU核心数量开始试验。
- **`prefetch_factor`：** 控制每个工作进程预加载的批次数量。通常将值设置为`2`是一个不错的起始点。
- **数据格式：** 二进制格式（Arrow、Parquet、MDS）通常比文本或JSON格式的读取和解析速度更快。压缩方式的选择也重要（例如，Zstandard速度快）。
- **存储位置：** 从快速的本地存储（如NVMe SSD）读取数据比通过网络从云对象存储读取更快，尽管像`StreamingDataset`这样的库已针对后者进行了优化。
- **混洗缓冲区大小：** 平衡内存使用与混洗质量。
- **批次大小：** 较大的批次能提高GPU利用率，但需要更多内存，并可能影响模型收敛（参见第17章：“LLM的优化算法”）。

有效管理和流式传输数据是训练大型语言模型的重要基础设施组成部分。通过选择合适的存储格式、运用分布式文件系统，并实现高效、可恢复的流式数据加载器，可以确保您的GPU持续获得数据，从而实现成功的大规模训练。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 9 Data Sampling Strategies Training

### 数据配比的重要性

# 数据配比的重要性

大型语言模型的训练通常需要精心收集、清洗和准备的数PB文本数据。然而，如果不加结构地将所有这些数据投入训练过程，不太可能得到最佳结果。预训练 (pre-training)所用的具体数据源*组合*，通常被称为*数据配比*，显著影响模型的各项能力、所涵盖的知识范围，甚至其固有的偏见。这不仅仅是数据的总量；来自不同来源的*比例*影响深远。

可以将预训练数据集视为模型的初级教育。就像人类的能力由学习经历塑造一样，大语言模型 (LLM)的能力由其所用数据塑造。一个主要在网络文本（如Common Crawl）上训练的模型将建立强大的通用语言理解能力和广泛的知识，但在专业方面可能缺乏有深度技能。反之，一个主要在源代码上训练的模型将擅长编程任务，但可能难以处理散文或会话互动。

### 塑造模型能力

不同数据源培养不同的技能：

1. **网络文本（例如 Common Crawl, C4）：** 提供广泛的语言、主题和常识推理 (inference)覆盖。对通用会话能力和理解多种文本风格很重要。然而，质量可能差异很大，包含噪声、观点和错误信息。
2. **书籍（例如 Project Gutenberg, Books3）：** 提供结构化的长篇文本，通常具有更高的语法质量和叙事连贯性。对于学习长距离依赖和更正式的语言风格很重要。
3. **代码库（例如 GitHub）：** 教会模型编程语言、逻辑、结构和常见编码模式。直接影响编码辅助能力。
4. **科学文献（例如 arXiv, PubMed Central）：** 传授技术方面的知识、复杂的推理和正式的科学写作风格。提升在专业问答和推理任务上的表现。
5. **维基百科：** 提供广泛主题的百科全书式事实知识，通常以中立、信息丰富的风格撰写。有助于模型立足于事实信息。
6. **对话数据（例如 论坛, 特定对话数据集）：** 教会会话流程、轮流发言和适合交互式应用的回复风格。

配比决定了这些技能的平衡。例如，增加训练配比中的代码比例，可能提升模型在编码基准测试上的表现，但如果数据总量不变或代码数据替代了高质量散文，这可能会导致纯语言任务上的表现略有下降。

设想一个万亿级token预训练 (pre-training)运行的两种可能配比：

- **配比 A（侧重通用能力）：** 70% 网络文本, 10% 书籍, 5% 代码, 10% 维基百科, 5% 其他。
- **配比 B（侧重代码能力）：** 40% 网络文本, 10% 书籍, 40% 代码, 5% 维基百科, 5% 其他。

模型A可能更好地进行通用对话并拥有更广泛的知识。模型B在生成和理解代码方面几乎肯定更强，但与模型A相比，在创意写作或新闻文章总结方面可能不太擅长。

70%10%10%5%5%40%40%10%5%5%配比 A配比 B

> 两种数据配比的比较，强调通用知识（A）与编码能力（B）的异同。

### 对偏见和安全的影响

数据配比也是引入或减少社会偏见的主要途径。如果训练数据主要反映特定群体的观点、人口特征或语言模式，产生的模型很可能继承这些特点。例如，一个严重偏向西方文化的文本数据集可能会产生一个难以理解或生成反映其他文化背景的文本的模型。同样，如果某些数据源（如未过滤的网络文本）中存在有害或有毒语言，模型可能学会复制它。

因此，精心策划和组成数据配比是负责任AI发展的重要部分。这不仅包括选择多样化的数据源，还要考虑过滤策略（第7章有所涉及），并可能调整配比，以降低已知含有较多问题内容的数据源的权重 (weight)。

### 通用性与专业性的权衡

设计数据配比需要权衡构建一个能力广泛的通用模型与一个高度专业的专家模型。

- **多样化配比：** 在多种数据类型上训练有助于提升通用性。模型学习语言和推理 (inference)中常见的深层模式，使其能够适应许多后续任务，即使是那些与训练数据不同的任务（少样本或零样本学习 (zero-shot learning)）。但其在任一方面可能不如专业模型那样有深度。
- **窄范围配比：** 将训练集中在特定方面（例如法律文件、医学文本）会产生在该方面具有深度知识和熟练程度的模型。这对抗专业应用很有价值，但会限制模型在其特定用途之外的适用性。如果排除了这些数据类型，它可能在常识推理或基本对话方面表现不佳。

理想的配比通常取决于大语言模型 (LLM)的预期应用。旨在广泛应用的基座模型（如 GPT-3/4、Llama、Claude）通常使用高度多样化的配比，而专为特定行业设计的模型可能会采用更有针对性的构成。像“The Pile”这样的数据集明确以多样性为目标构建，结合了许多不同的文本源，以鼓励通用化。

了解数据配比的深远影响是在训练期间策略性地选择数据的第一步。它允许您有意识地塑造模型的能力特征，平衡知识的广度与深度、专业技能，并减少潜在偏见，最终形成一个更有效、更可靠的大语言模型。以下章节将详细介绍实施这些采样策略的具体方法。

获取即时帮助、个性化解释和交互式代码示例。

---

### 来源权重策略

# 来源权重策略

训练大型语言模型通常涉及从网络爬取数据（例如 Common Crawl）、书籍、代码库和专门语料库等来源收集多样化的文本数据。在此过程中，一个主要考虑因素是确定每种数据来源在训练中的比例贡献。简单地将所有可用数据拼接并均匀打乱，看起来可能是一个直接的方法，但很少能产生最佳结果。这是因为不同的数据来源在质量、相关性和语言风格方面本身就存在差异。来源加权提供了对训练数据组合构成精确的控制，使得模型能够优先学习某些类型的文本而非其他。

主要思路是为每个数据来源分配一个特定的权重 (weight)或概率。在训练期间，当组建一个数据批次时，样本会根据这些预设权重从不同来源中抽样。与权重较低的来源相比，被分配较高权重的来源将随着时间为训练过程提供更多样本。

### 定义抽样概率

假设你有 kkk 个不同的数据来源，S1,S2,…,SkS\_1, S\_2, \dots, S\_kS1​,S2​,…,Sk​。每个来源 SiS\_iSi​ 包含 NiN\_iNi​ 个文档或样本。我们希望定义在任意给定步骤从来源 SiS\_iSi​ 抽样一个样本的概率 pip\_ipi​。一个常见做法是使这个概率与来源的大小和分配的权重 (weight) wiw\_iwi​ 成比例：

pi∝wi⋅Nip\_i \propto w\_i \cdot N\_ipi​∝wi​⋅Ni​

或者，或许更直接的控制方式是，你可以直接为每个来源定义所需的比例（或概率）pip\_ipi​，使得 ∑i=1kpi=1\sum\_{i=1}^{k} p\_i = 1∑i=1k​pi​=1。这时，权重 wiw\_iwi​ 就简单地成为这些目标比例 pip\_ipi​。

例如，你可能会决定采用以下组合方式：

- 网络文本 (来源 1): p1=0.60p\_1 = 0.60p1​=0.60 (60%)
- 书籍 (来源 2): p2=0.15p\_2 = 0.15p2​=0.15 (15%)
- 代码 (来源 3): p3=0.15p\_3 = 0.15p3​=0.15 (15%)
- 维基百科 (来源 4): p4=0.10p\_4 = 0.10p4​=0.10 (10%)

这意味着，平均而言，在每个训练周期（或大量步骤中），模型看到的样本中有 60% 将来自网络文本数据集，15% 来自书籍，以此类推。

网络文本书籍代码维基百科0204060

> 大型语言模型预训练 (pre-training)期间从不同数据来源抽样的比例示例。

### 设定权重 (weight)的方法和考量

选择合适的权重通常受多种因素影响，需要仔细考虑并进行多次试验：

1. **数据质量：** 维基百科或同行评审文章等高质量、精心整理的来源，相对于其大小，可能会比嘈杂的网络爬取数据获得更高的权重。这能指导模型学习更多事实性强、结构良好的语言。
2. **领域特化：** 如果你正在构建一个旨在用于特定领域（例如医疗聊天机器人或代码生成助手）的模型，你可能会增加相关领域特定数据集（例如 PubMed 摘要或代码库）的权重。
3. **数据量与影响：** 如果大型数据集（如 Common Crawl）按其原始大小比例抽样，它们很容易在训练组合中占据主导地位。分配明确的权重可以避免这种情况，确保较小、高质量或领域特定数据集仍能对模型的学习产生重要影响。
4. **防止数据污染：** 如果在特定基准上进行评估，你可能会降低已知包含这些基准节选的数据来源的权重，或将其排除，以确保公平评估。
5. **期望的模型能力：** 如果对话文本的流畅性很重要，你可能会包含并合理加权包含对话或论坛讨论的数据集。如果事实基础很重要，结构化文本和百科全书来源可能会获得更高权重。

### 实现方面的考量

实现来源加权通常涉及修改数据加载过程。数据加载器需要了解不同的来源及其相关权重 (weight)，而不是从单一大型数据集中均匀抽样。

在 PyTorch 中，你可以通过创建自定义的 `IterableDataset` 或使用能够适应多个底层数据集加权抽样的采样器来实现这一点。一个简化示例可能如下所示：

```python
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader

class DummyDataset(IterableDataset):
    def __init__(self, source_name, size):
        self.source_name = source_name
        self.size = size

    def __iter__(self):
        for i in range(self.size):

            yield torch.randn(512), self.source_name
            if i % 1000 == 0 and i > 0 :
                 print(f"Yielded {i} from {self.source_name}")

sources = {
    "web": {"dataset": DummyDataset("web", 1_000_000), "weight": 0.60},
    "books": {"dataset": DummyDataset("books", 200_000), "weight": 0.15},
    "code": {"dataset": DummyDataset("code", 300_000), "weight": 0.15},
    "wiki": {"dataset": DummyDataset("wiki", 100_000), "weight": 0.10},
}

source_names = list(sources.keys())
source_weights = np.array([sources[name]["weight"] for name in source_names])

source_iters = {name: iter(sources[name]["dataset"]) for name in source_names}

class WeightedSourceSampler(IterableDataset):
    def __init__(self, source_names, source_weights, source_iters):
        self.source_names = source_names
        self.source_weights = source_weights
        self.source_iters = source_iters

    def __iter__(self):
        while True:

            chosen_source_name = np.random.choice(
                self.source_names, p=self.source_weights
            )

            try:

                item = next(self.source_iters[chosen_source_name])
                yield item
            except StopIteration:

                print(f"正在重新启动 {chosen_source_name} 的迭代器")
                self.source_iters[chosen_source_name] = iter(
                    sources[chosen_source_name]["dataset"]
                )

                try:
                   item = next(self.source_iters[chosen_source_name])
                   yield item
                except StopIteration:
                   print(
                       f"警告：{chosen_source_name} 的迭代器在重置后立即耗尽。"
                   )

                   continue

weighted_sampler_dataset = WeightedSourceSampler(
    source_names, source_weights, source_iters
)

data_loader = DataLoader(weighted_sampler_dataset, batch_size=4, num_workers=0)

print("正在获取一个批次...")
batch = next(iter(data_loader))

print(f"批次数据形状：{batch[0].shape}")
print(f"批次来源标识符：{batch[1]}")

print("\n正在获取更多批次...")
for i in range(5):
     batch = next(iter(data_loader))
     print(f"批次 {i+1} 来源：{batch[1]}")
```

> PyTorch 实现草图，用于根据预设权重从多个数据来源进行抽样。

本示例演示了基本原理。大型语言模型的生产级数据加载器通常会涉及更复杂的机制，以提高效率、处理分布式训练以及管理无法完全载入内存的超大型数据集，可能还会使用流式传输技术。

### 权衡与考量

来源加权虽然功能强大，但也带来了复杂性：

- **调优难度：** 找到最佳权重 (weight)通常需要大量试验，并在下游任务上进行评估，这可能会带来高昂的计算成本。
- **偏差放大：** 对某些来源过度加权可能导致模型采纳该数据中存在的偏见。反之，对重要但较小的数据集加权不足可能会阻碍模型在特定方面的表现。
- **动态加权：** 正如稍后在课程学习和退火的背景下所讨论的，权重可能不会在整个训练过程中固定不变。它们可能根据训练阶段或模型表现动态变化。

来源加权是一种基本方法，用于管理大型语言模型预训练 (pre-training)所用大规模数据集的构成。通过仔细考虑不同数据来源的质量、相关性和数据量，并分配适当的权重，工程师可以更好地指导学习过程，并塑造最终模型的能力。

获取即时帮助、个性化解释和交互式代码示例。

---

### 基于温度的采样

# 基于温度的采样

尽管为不同数据来源分配固定权重 (weight)可以控制整体混合比例，但有时更具动态性的方法会更有益。您可能希望调整分配的权重在训练过程中对实际采样概率的影响强度。这就是基于温度的采样发挥作用的地方，它的名称和原理来源于统计物理，并常用于控制生成模型输出的随机性。在数据采样中，温度允许您调节从来源权重得到的概率分布的“锐度”。

假设您有多个数据来源，每个来源都被分配一个分数或对数权重 wiw\_iwi​，以反映其预估的重要度或质量。不进行温度缩放时，从来源 iii 采样的概率可能由基于这些分数的标准 Softmax 函数确定。温度 TTT 在 Softmax 计算之前对这些分数引入一个缩放因子。

给定温度 TTT 时，选择数据来源 iii 的概率 P(i∣T)P(i | T)P(i∣T) 的计算方式如下：

P(i∣T)=exp⁡(wi/T)∑jexp⁡(wj/T)P(i | T) = \frac{\exp(w\_i / T)}{\sum\_j \exp(w\_j / T)}P(i∣T)=∑j​exp(wj​/T)exp(wi​/T)​

这里，wiw\_iwi​ 是来源 iii 的分数或对数权重，求和范围是所有可用来源 jjj。温度 TTT 是一个正值，它控制着这个概率分布的形态：

- **高温 (T>1T > 1T>1，特别是当 T→∞T \to \inftyT→∞ 时)：** 随着 TTT 增加，wi/Tw\_i / Twi​/T 项对所有 iii 都趋近于零。因此，exp⁡(wi/T)\exp(w\_i / T)exp(wi​/T) 趋近于 exp⁡(0)=1\exp(0) = 1exp(0)=1。无论其原始权重 wiw\_iwi​ 如何，概率 P(i∣T)P(i | T)P(i∣T) 在所有来源上变得几乎一致。高温鼓励从所有来源广泛采样，在训练的早期阶段增加多样性，或确保即使低权重的来源也能偶尔被采样。
- **单位温度 (T=1T = 1T=1)：** 这会恢复基于原始分数 wiw\_iwi​ 的标准 Softmax 概率分布。概率与 exp⁡(wi)\exp(w\_i)exp(wi​) 成正比。
- **低温 (0<T<10 < T < 10<T<1，特别是当 T→0+T \to 0^+T→0+ 时)：** 随着 TTT 从正方向趋近于零，分数之间的差异被放大。与其他来源相比，分数最高的来源 wmaxw\_{max}wmax​ 的 exp⁡(wi/T)\exp(w\_i / T)exp(wi​/T) 项变得极其大。概率分布明显集中，几乎完全集中在分数最高的来源上。低温迫使采样强烈偏向权重最高的来源，有效地“运用”那些被认为最重要的来源。

这种机制提供了一种平滑的插值方式，可以在均匀采样 (T→∞T \to \inftyT→∞) 和基于最高权重的贪婪采样 (T→0+T \to 0^+T→0+) 之间进行调节。

### 实际使用与退火

基于温度的采样与退火策略结合使用时特别有用。您可以从较高温度 (T>1T > 1T>1) 开始训练，以确保模型能看到来自所有可用来源的各种数据，促进广度学习。随着训练的进行，您可以逐渐降低温度（将其退火至 T=1T=1T=1 甚至更低）。这种转变将在后期将训练重心移向更高质量或更相关的数据来源，帮助模型基于优先考虑的数据混合提升能力。

考虑一个包含三个数据来源的场景：网页文本（分数 w1=2.0w\_1 = 2.0w1​=2.0）、书籍（分数 w2=3.0w\_2 = 3.0w2​=3.0）和代码（分数 w3=1.0w\_3 = 1.0w3​=1.0）。让我们看看采样概率如何随温度变化。

```python
import torch
import torch.nn.functional as F

scores = torch.tensor([2.0, 3.0, 1.0])

temperatures = [0.5, 1.0, 2.0, 10.0]
source_names = ["网页文本", "书籍", "代码"]

print("来源分数:", dict(zip(source_names, scores.tolist())))
print("-" * 30)

for T in temperatures:

    probs = F.softmax(scores / T, dim=0)
    print(f"温度 T = {T:.1f}")
    for name, prob in zip(source_names, probs.tolist()):
        print(f"  P({name}): {prob:.4f}")
    print("-" * 30)

T_sample = 1.0
sampling_probs = F.softmax(scores / T_sample, dim=0)

num_samples = 5
sampled_indices = torch.multinomial(
    sampling_probs,
    num_samples=num_samples,
    replacement=True
)

print(f"\n采样示例 (T={T_sample:.1f}):")
sampled_sources = [source_names[i] for i in sampled_indices]
print(f"采样到的来源索引: {sampled_indices.tolist()}")
print(f"采样到的来源名称: {sampled_sources}")
```

运行此代码会产生类似于以下输出：

```
------------------------------
温度 T = 0.5
  P(网页文本): 0.1173
  P(书籍): 0.8681
------------------------------
温度 T = 1.0
  P(网页文本): 0.2419
  P(书籍): 0.6577
------------------------------
温度 T = 2.0
  P(网页文本): 0.3067
  P(书籍): 0.4487
------------------------------
温度 T = 10.0
  P(网页文本): 0.3293
  P(书籍): 0.3433
------------------------------

采样示例 (T=1.0):
采样到的来源索引: [1, 1, 1, 0, 1]
采样到的来源名称: ['书籍', '书籍', '书籍', '网页文本', '书籍']
```

请注意，在 T=0.5T=0.5T=0.5 时，“书籍”（最高分数）的概率占主导地位（0.8681）。在 T=1.0T=1.0T=1.0 时，“书籍”仍然被偏好（0.6577），但其他来源也有合理的机会。随着 TTT 增加到 2.02.02.0 和 10.010.010.0，概率变得更接近，趋近于均匀分布（1/3≈0.33331/3 \approx 0.33331/3≈0.3333）。

网页文本书籍代码00.20.40.60.81温度T=0.5T=1.0T=2.0T=10.0

> 在不同温度下，从三个来源（“网页文本”、“书籍”、“代码”，分数分别为 2.0、3.0、1.0）采样的概率。较低温度使分布更集中于分数最高的来源（“书籍”），而较高温度则使其趋于均匀。

### 实现时的考量

- **权重 (weight)的理解：** 公式 P(i∣T)∝exp⁡(wi/T)P(i | T) \propto \exp(w\_i / T)P(i∣T)∝exp(wi​/T) 假设 wiw\_iwi​ 是分数或对数权重，且数值越大越好。如果您的权重 wiw\_iwi​ 直接表示比例（总和为 1），您可以使用 P(i∣T)∝wi1/TP(i | T) \propto w\_i^{1/T}P(i∣T)∝wi1/T​ 并进行归一化 (normalization)。请确保您的实现与您的权重定义相符。当处理神经网络 (neural network)输出或分数时，指数形式很常见。
- **退火策略：** 温度降低的速度（即退火策略）是另一个超参数 (parameter) (hyperparameter)。常见选择包括线性衰减、指数衰减或在训练过程中进行阶梯式降低。
- **集成：** 这种采样逻辑需要集成到您的数据加载流程中。通常，您会为您的 `Dataset` 实现一个自定义采样器，它会为每个批次或样本，首先使用基于温度的采样选择一个来源，然后从该来源的数据集中取出一个项。

基于温度的采样提供了一个灵活的调节手段，可以在整个训练过程中动态控制数据混合，使得能够制定平衡广泛学习和集中运用高价值数据来源的策略。

获取即时帮助、个性化解释和交互式代码示例。

---

### 课程学习简介

# 课程学习简介

虽然从多样化的数据混合中随机打乱和采样是一种常见做法，但它从一开始就将所有数据点视为信息量均等。思考一下人类是如何学习的：我们通常从更简单的知识点开始，逐步提升至更复杂的知识。我们不会在理解基础算术之前直接跳到高等微积分。课程学习（CL）将类似原则应用于训练机器学习 (machine learning)模型，包括大型语言模型。课程学习不是以纯随机顺序从整个数据集中呈现数据点，而是引入一种结构，通常在训练过程中从“简单”示例转向“困难”示例。

基本思想是，从简单示例开始可以帮助模型构建根本表示，并避免在训练早期陷入不良局部最小值。这种初步的立足点可能使模型后续更容易从更复杂或带噪声的数据中学习。在大型语言模型预训练 (pre-training)的背景下，“简单”与“困难”的定义可以有多种形式。

### 定义课程

对于大型语言模型而言，什么构成“简单”或“困难”的示例？这并非总是直截了当，但常见方法包括：

1. **序列长度：** 较短序列通常被认为更简单。课程可能从主要在较短文档或片段上训练开始，并随着训练进行逐步增加批次中允许的最大序列长度。这有助于模型首先学习局部依赖，然后处理长距离依赖。
2. **词汇或句法复杂性：** 使用更简单词汇或更基本句型的示例可以在早期优先处理。像Flesch-Kincaid可读性分数、解析树深度或词汇稀有度等指标潜在地可用于对示例进行排序，尽管这会增加大量预处理开销。
3. **数据质量或来源：** 如同在数据混合中讨论的，一些来源（例如，精心整理的百科全书、书籍）通常比其他来源（例如，原始网页抓取数据）更干净、更有结构。课程可能涉及主要从高质量来源开始训练，随后逐步引入更多噪声或更多样化的数据。这与数据源加权密切相关，但具有有意的时间推进。
4. **困惑度评分：** 可以使用一个更小、已有的语言模型对训练示例的困惑度进行评分。困惑度较低的示例（即，较简单模型更容易预测的那些）可以先引入。

### 课程学习的实现

实现课程学习需要修改数据加载或采样过程。采样器需要了解训练进度（例如，当前周期或步骤），并根据定义的课程计划选择数据，而不是从数据集中均匀采样。

一个简单方法可能涉及根据难度指标（如序列长度）将数据分桶，并控制在不同训练阶段从哪些桶中进行主动采样。

考虑一个通过自定义PyTorch `Sampler`实现的基础的基于长度的课程。此示例体现了核心逻辑，并非生产级实现。

```python
import torch
from torch.utils.data import Sampler
import numpy as np

class LengthBasedCurriculumSampler(Sampler):
    def __init__(self,
                 data_lengths,
                 batch_size,
                 start_percentile=0.1,
                 end_percentile=1.0,
                 total_steps=10000):
        """
        根据训练中序列长度百分位数递增的方式采样批次。

        Args:
            data_lengths (list or np.array): 每个数据样本的长度列表。
            batch_size (int): 每个批次的大小。
            start_percentile (float): 初始长度百分位阈值 (0.0 到 1.0)。
            end_percentile (float): 最终长度百分位阈值 (0.0 到 1.0)。
            total_steps (int): 课程学习进行的总训练步数。
        """
        self.data_lengths = np.array(data_lengths)
        self.indices = np.argsort(self.data_lengths)
        self.sorted_lengths = self.data_lengths[self.indices]
        self.batch_size = batch_size
        self.start_percentile = start_percentile
        self.end_percentile = end_percentile
        self.total_steps = total_steps
        self.current_step = 0

        self.num_samples = len(data_lengths)

        self.start_idx = int(self.start_percentile * self.num_samples)
        self.final_max_idx = int(self.end_percentile * self.num_samples)

    def get_current_max_index(self):

        progress = min(1.0, self.current_step / self.total_steps)
        increase = progress * (self.final_max_idx - self.start_idx)
        current_max_idx = int(self.start_idx + increase)

        return max(self.start_idx, current_max_idx)

    def __iter__(self):
        current_max_idx = self.get_current_max_index()

        eligible_indices = self.indices[:current_max_idx]

        if len(eligible_indices) < self.batch_size:

            eligible_indices = np.random.choice(
                eligible_indices, size=self.batch_size, replace=True
            )
        else:

            np.random.shuffle(eligible_indices)

        num_batches = 0
        for i in range(0, len(eligible_indices), self.batch_size):
            batch_indices = eligible_indices[i : i + self.batch_size]

            if len(batch_indices) == self.batch_size:
                yield batch_indices.tolist()
                num_batches += 1

        self.current_step += num_batches

    def __len__(self):

        current_max_idx = self.get_current_max_index()
        num_eligible = len(self.indices[:current_max_idx])
        return num_eligible // self.batch_size
```

此采样器在初始化时一次性按长度排序数据。在每次迭代（通常对应一个周期）中，它根据当前的训练进度（`current_step`）确定允许的最大数据索引。然后它打乱并生成仅包含达到该长度百分位数的数据点的批次。`get_current_max_index`函数定义了课程的节奏。

### 优点与挑战

课程学习的潜在优点包括：

- **更快收敛：** 通过最初专注于更简单的示例，模型可能在早期阶段更快收敛。
- **泛化能力提升：** 一些研究表明，课程学习可以使模型泛化能力更强，潜在地通过引导学习过程形成更有意义的表示。
- **训练稳定性：** 逐步引入复杂性可能有助于稳定训练，特别是对于复杂架构或有难度的数据集。

然而，课程学习也带来挑战：

- **定义“难度”：** 如前所述，自动且有效地量化 (quantization)难度并非易事。不佳的衡量标准可能导致有害的课程。
- **节奏函数：** 确定最优计划（如何快速增加难度）需要调整，并且可能依赖于数据集或模型。过慢的计划可能浪费计算资源，而过快的计划则可能抵消优势。
- **实现复杂性：** 与简单随机打乱相比，集成课程学习增加了数据加载和训练管道的复杂性。
- **遗忘风险：** 如果模型仅在训练后期看到困难示例，它可能会“遗忘”从早期呈现的简单示例中学到的模式（尽管残差连接和大型模型容量通常能减轻这种情况）。

虽然基于细粒度难度指标的显式复杂课程并非总是训练大型语言模型的默认设置（对于大型语言模型，复杂的*数据混合加权*常因其可扩展性和经验成功而被偏好），但课程学习的核心思想常指导这些混合如何设计和潜在排序。例如，一种多阶段训练过程，其中模型首先在更干净的数据上训练，然后再接触完整、带噪声的数据集，可视为一种粗粒度课程形式。了解课程学习的原则提供另一种工具，用于优化大型语言模型训练这一要求较高的过程。

获取即时帮助、个性化解释和交互式代码示例。

---

### 数据步进与退火调度

# 数据步进与退火调度

虽然静态数据混合为训练提供了一个固定的方案，但大型语言模型可能会从更具动态性的数据呈现方式中获益，这就像课程指导学习一样。我们不是在整个训练过程中都以相同比例馈送数据源，而是可以随着时间动态调整混合比例或采样过程本身。这就是数据步进和退火调度的主要思想。这些方法旨在通过控制数据集不同部分的*何时*以及*如何*被着重处理来改善学习过程。

### 数据步进：控制接触数据的速度

数据步进是指在训练过程中控制模型接触不同子集或类型数据的速度。可以将其视为根据模型的学习进展来管理信息流。例如，你可能主要从高质量、精心整理的语料库开始训练，以打下扎实的语言理解基础。随着训练的进行，模型能力增强，你可以逐渐增加来自噪声较大的数据源（如网页抓取数据）或特定类别数据（如代码库）的比例。

步进可以基于多种标准：

1. **数据源质量：** 从干净数据开始，逐渐引入噪声更大或更复杂的数据。
2. **类别特定性：** 首先使用通用文本，然后根据需要逐步加入特定类别文本。
3. **数据难度：** 如果能够估计数据难度（例如，根据文本复杂性或想法的稀有程度），可以逐步引入从简单到困难的例子，尽管在大规模情况下界定难度是一项困难。

实施数据步进通常需要创建一个调度函数，该函数根据当前的训练步骤或周期修改与不同数据源相关的采样权重 (weight)或概率。

```python
import numpy as np

data_sources = {
    "curated_corpus": {"weight": 0.7, "path": "/path/to/curated"},
    "web_crawl": {"weight": 0.3, "path": "/path/to/web"},

}

total_steps = 1_000_000

def get_pacing_weights(current_step, total_steps, sources):
    """根据训练进度计算动态权重。"""
    progress = current_step / total_steps
    new_weights = {}

    curated_weight = max(0.1, 0.7 - 0.6 * progress)
    web_weight = 1.0 - curated_weight

    new_weights["curated_corpus"] = curated_weight
    new_weights["web_crawl"] = web_weight

    total_weight = sum(new_weights.values())
    normalized_weights = {
        k: v / total_weight
        for k, v in new_weights.items()
    }

    return normalized_weights
```

这段代码展示了一个简单的线性步进调度。根据期望的学习进程，可以设计更复杂的函数（例如，指数型、阶梯型）。重要的一点是，随着训练的进行，动态调整从每个数据源采样的可能性。

### 数据采样的退火调度

退火，在数据采样的语境下，通常指逐渐改变一个参数 (parameter)，该参数控制数据源采样分布的*形态*或*随机性*。这与之前讨论的基于温度的采样紧密联系。

回顾一下，温度 TTT 修改了从权重 (weight) wiw\_iwi​ 推导出的概率 pip\_ipi​：

pi=exp⁡(wi/T)∑jexp⁡(wj/T)p\_i = \frac{\exp(w\_i / T)}{\sum\_j \exp(w\_j / T)}pi​=∑j​exp(wj​/T)exp(wi​/T)​

退火调度可能涉及从较高的温度 T>1T > 1T>1 开始，并在训练过程中逐渐将其降低至 T=1T = 1T=1（甚至略低）。

- **高温（训练早期）：** 当 T>1T > 1T>1 时，概率 pip\_ipi​ 变得更均匀。这促进了多样性采样，意味着模型在不同数据源之间进行更均匀的采样，即使是那些初始权重较低的数据源。这在早期可能有利，可以防止模型过快地专注于主要的训练数据。
- **低温（训练后期）：** 随着 TTT 趋向于 1 降低，采样概率变得更接近由原始权重 wiw\_iwi​ 定义的分布。如果 T<1T < 1T<1，分布会变得更尖锐，进一步突出高权重的数据源。这使得模型在训练结束时，能将其学习成果集中于被认为更重要或具有代表性的数据上。

退火调度可以应用于温度参数，也可以直接应用于权重本身（例如，逐渐增加高权重和低权重之间的差异）。常见的调度包括温度参数的线性衰减、余弦衰减或指数衰减，或者应用于权重的类似调制因子。

00.2M0.4M0.6M0.8M1M11.21.41.61.82线性衰减 T余弦衰减 T

> 示例温度退火调度，涵盖 100 万训练步，从 T=2.0 开始，使用线性和余弦函数衰减至 T=1.0。

### 步进与退火的结合

数据步进和退火并非互斥。你可以将它们结合起来，创建更巧妙的数据馈送方法。例如：

- 使用**步进调度**随时间逐渐改变不同数据源的*目标权重 (weight)*（例如，增加特定类别数据的权重）。
- 同时，在采样温度上使用**退火调度**，以控制基于这些目标权重的采样*随机性*（例如，从高温开始以促进多样性采样，后期降低温度以集中处理高权重数据源）。

这种组合可以精细地控制模型在不同训练阶段看到*什么*数据以及它*多严格地*遵循预设的混合比例。

### 实现考量

实施动态采样调度会增加训练流程的复杂性。

- **进度跟踪：** 训练循环需要准确跟踪当前步骤或周期，以计算调度的当前状态。
- **采样器/加载器配合：** 数据加载机制必须足够灵活，能够动态更新其采样概率或源权重 (weight)，可能在每一步或每几千步进行。这可能涉及自定义采样器逻辑。像 `torch.utils.data.WeightedRandomSampler` 这样的框架可以进行调整，或者可能需要管理多个数据集的自定义迭代器。
- **监控：** 使用动态调度时，密切监控训练稳定性（损失曲线、梯度范数）和评估指标。数据分布的突然变化有时会导致暂时的不稳定。
- **调整：** 找到最佳的步进或退火调度通常需要通过实验。理想的调度形状（线性、余弦、阶梯）及其持续时间很大程度上取决于数据集、模型大小和训练目标。

虽然比静态采样更复杂，但数据步进和退火调度为大型语言模型的学习过程提供了有效的工具。通过随时间精心控制数据输入，你有可能加快收敛速度，增强健壮性，并更好地塑造模型的最终能力以满足特定需求。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 10 Implementing Transformer From Scratch

### 设置项目环境

# 设置项目环境

在编写 Transformer 架构的代码之前，建立一个稳定且可用的开发环境十分重要。这可以确保代码示例按预期运行，并有助于整理将要构建的各种组成部分。必要的工具和项目结构在此进行设置。

我们的实现将依赖于机器学习 (machine learning)和深度学习 (deep learning)中广泛使用的标准 Python 库。我们假设您已安装 Python 3（建议使用 3.8 或更高版本）。我们将使用的主要深度学习框架是 PyTorch，它以其 Python 风格的接口以及在研究和开发中的灵活性而闻名。我们还将使用 NumPy 进行可能的数值运算，尽管 PyTorch 的张量通常就足够了。

### 所需库

1. **PyTorch：** 核心深度学习 (deep learning)框架。我们需要一个支持 Transformer 所需操作的版本。最重要的是，请确保安装与您的硬件兼容的版本，特别是如果您计划使用 NVIDIA GPU 进行加速（对于这种规模的模型，强烈推荐这样做）。
2. **NumPy：** Python 中科学计算的基本软件包。PyTorch 与 NumPy 数组集成良好。

### 安装

您可以使用 Python 的包安装器 `pip` 来安装这些库。如果您使用 NVIDIA GPU，通常最好遵循 PyTorch 官方网站 (pytorch.org) 上的特定安装说明，以获取正确的 CUDA 版本。

A typical installation command for a CPU-only setup or after setting up CUDA might look like this:

```bash
pip install torch numpy
```

对于 GPU 支持，请查阅 PyTorch 网站，以获取针对您的特定 CUDA 版本（例如 CUDA 11.8 或 12.1）定制的命令。它通常看起来像：

```bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy
```

最好在专用的虚拟环境（使用 `venv` 或 `conda`）中工作，以避免项目依赖项之间的冲突。

### 验证安装

库安装完成后，您可以验证 PyTorch 是否正确安装，并检查它是否能检测到您的 GPU（如果适用）。打开 Python 解释器或创建一个包含以下内容的简单脚本 (`check_env.py`)：

```python
import torch
import numpy as np

print(f"PyTorch 版本: {torch.__version__}")
print(f"NumPy 版本: {np.__version__}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU 可用: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU 不可用，使用 CPU。")

tensor = torch.rand(3, 3, device=device)
print("\n在设备上创建的示例张量:")
print(tensor)
print(f"张量位于: {tensor.device}")
```

从您的终端运行此脚本：`python check_env.py`。您应该会看到打印出的库版本，以及一条消息，指明是否检测到 GPU 并将其用于简单的张量操作。看到与示例类似的输出，表示您的基本环境已就绪。

### 项目结构

在实现 Transformer 时，保持代码的组织性很有帮助。本章的最小结构可能如下所示：

```
transformer_from_scratch/
├── check_env.py
├── transformer_components.py  # 我们将在此处添加注意力、FFN 层
├── transformer_model.py       # 我们将在此处组装完整模型
└── (Optional) notebooks/      # 用于使用 Jupyter 进行试验（可选）
```

这种结构将环境检查、构建块（如注意力机制 (attention mechanism)）和最终组装的模型分离到不同的文件中，提高了模块化程度。您可以选择直接在 Python 脚本（`.py` 文件）中工作，或者使用 Jupyter Notebook（`.ipynb` 文件）进行更具交互性的开发，可以将它们放在 `notebooks/` 目录中。

环境设置并验证完成后，我们就可以开始实现 Transformer 架构的核心组成部分了，下一节将从缩放点积注意力机制开始。

获取即时帮助、个性化解释和交互式代码示例。

---

### 实现缩放点积注意力

# 实现缩放点积注意力

缩放点积注意力是 Transformer 注意力机制 (attention mechanism)的基本组成部分。这种机制允许模型在处理特定元素时，衡量输入序列不同部分的重要性。它不依赖循环，而是根据从输入中获得的查询、键和值之间的相互作用来计算注意力分数。

缩放点积注意力的核心计算定义为：

注意力(Q,K,V)=softmax(QKTdk)V\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V注意力(Q,K,V)=softmax(dk​​QKT​)V

让我们分析一下组成部分和实现步骤：

1. **查询 (Q)、键 (K)、值 (V) 矩阵：** 这些矩阵通常是输入嵌入 (embedding)的投影。对于给定的输入序列元素（表示为一个向量 (vector)），我们生成：

   - 一个 `查询` 向量：表示当前元素正在寻找信息。
   - 一个 `键` 向量：表示提供信息的元素，用于计算与查询的兼容性。
   - 一个 `值` 向量：表示提供信息的元素的实际内容。
     如果我们有一个序列批次，QQQ、KKK 和 VVV 变为矩阵，其中每行对应序列中的一个元素。它们的维度通常为 [batch\_size,seq\_len,dmodel][batch\\_size, seq\\_len, d\_{model}][batch\_size,seq\_len,dmodel​]；对于单个注意力头进行投影后，QQQ 和 KKK 的维度为 [batch\_size,seq\_len,dk][batch\\_size, seq\\_len, d\_k][batch\_size,seq\_len,dk​]，而 VVV 的维度为 [batch\_size,seq\_len,dv][batch\\_size, seq\\_len, d\_v][batch\_size,seq\_len,dv​]。通常，dk=dvd\_k = d\_vdk​=dv​。
2. **计算点积 (QKTQK^TQKT)：** 第一步是计算查询矩阵 QQQ 和键矩阵 KTK^TKT 的转置之间的点积。此操作计算每个查询应关注每个键的程度。更高的点积表示查询和键之间具有更高的相关性或兼容性。结果矩阵通常称为 `分数` 或 `能量`，其维度为 [batch\_size,seq\_lenq,seq\_lenk][batch\\_size, seq\\_len\_q, seq\\_len\_k][batch\_size,seq\_lenq​,seq\_lenk​]，其中 seq\_lenqseq\\_len\_qseq\_lenq​ 是查询的序列长度，seq\_lenkseq\\_len\_kseq\_lenk​ 是键的序列长度（在自注意力 (self-attention)中它们通常是相同的）。
3. **缩放 (...dk\frac{...}{\sqrt{d\_k}}dk​​...​)：** 然后，通过除以键向量维度 dkd\_kdk​ 的平方根来缩放分数。这种缩放对于稳定训练过程很重要。没有它，对于较大的 dkd\_kdk​ 值，点积的幅度可能会变得非常大。softmax 函数的输入过大可能导致梯度极小，从而使学习变得困难。缩放确保 softmax 输入的方差保持合理。
4. **应用掩码（可选）：** 在许多情况下，我们需要阻止关注某些位置。这通过在 softmax 步骤*之前*进行掩码来实现。

   - **填充掩码：** 如果输入序列在批次内被填充到相同长度，我们不希望模型关注这些填充标记 (token)。掩码识别填充位置（通常用 `True` 或 `1`）。我们会在这些位置的分数上添加一个大的负数（如 -1e9 或负无穷）。
   - **前瞻掩码：** 在解码器的自注意力层中，一个标记应该只关注之前的位置和自身，而不是未来的位置。这种掩码通过掩盖未来的位置来实现这一点。
     掩码确保随后的 softmax 操作为被掩盖的位置分配接近零的概率。
5. **应用 Softmax：** softmax 函数按行应用于缩放（并可能被掩盖）后的分数。这会将分数转换为概率分布，其中每个值表示一个查询分配给一个键的注意力权重 (weight)。每个查询的权重总和为 1。结果矩阵通常称为 `注意力权重`，其维度为 [batch\_size,seq\_lenq,seq\_lenk][batch\\_size, seq\\_len\_q, seq\\_len\_k][batch\_size,seq\_lenq​,seq\_lenk​]。
6. **乘以值 (...V...V...V)：** 最后，注意力权重矩阵乘以值矩阵 VVV。这一步计算值向量的加权和，其中权重由注意力概率决定。获得更高注意力权重的元素对输出的贡献更大。缩放点积注意力层的输出维度为 [batch\_size,seq\_lenq,dv][batch\\_size, seq\\_len\_q, d\_v][batch\_size,seq\_lenq​,dv​]。

### PyTorch 实现

让我们将这些步骤转换为 PyTorch 函数。我们假设输入 `query`、`key` 和 `value` 是 3D 张量，表示序列批次，可能已经为特定的注意力头进行了投影。

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    query: torch.Tensor,
    torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算缩放点积注意力。

    参数：
        query: 查询张量；形状为 (batch_size, num_heads, seq_len_q, d_k)
               或 (batch_size, seq_len_q, d_k) (如果为单头)。
        key: 键张量；形状为 (batch_size, num_heads, seq_len_k, d_k)
             或 (batch_size, seq_len_k, d_k) (如果为单头)。
        value: 值张量；形状为 (batch_size, num_heads, seq_len_v, d_v)
               或 (batch_size, seq_len_v, d_v) (如果为单头)。
               注意：seq_len_k 和 seq_len_v 必须相同。
        mask: 可选的掩码张量；形状应可广播到
              (batch_size, num_heads, seq_len_q, seq_len_k)。
              `True` 或 `1` 的位置将被掩盖（设为 -inf）。

    返回：
        包含以下内容的元组：
        - output：注意力输出张量；
                  形状为 (batch_size, num_heads, seq_len_q, d_v)
                  或 (batch_size, seq_len_q, d_v) (如果为单头)。
        - attention_weights：注意力权重张量；
                             形状为 (batch_size, num_heads, seq_len_q, seq_len_k)
                             或 (batch_size, seq_len_q, seq_len_k) (如果为单头)。
    """

    d_k = query.size(-1)
    scores = (torch.matmul(query, key.transpose(-2, -1))
              / math.sqrt(d_k))

    if mask is not None:

        scores = scores.masked_fill(mask == True, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)

    if torch.isnan(attention_weights).any():
        print("警告：在注意力权重中检测到 NaN。 "
              "请检查掩码或输入数据。")

    output = torch.matmul(attention_weights, value)

    return output, attention_weights

batch_size = 2
seq_len_q = 5
seq_len_k = 7
d_k = 64
d_v = 128

query_tensor = torch.randn(batch_size, seq_len_q, d_k)
key_tensor = torch.randn(batch_size, seq_len_k, d_k)
value_tensor = torch.randn(batch_size, seq_len_k, d_v)

padding_mask = torch.zeros(batch_size, 1, seq_len_k, dtype=torch.bool)
padding_mask[:, :, -2:] = True

output_tensor, attention_weights_tensor = scaled_dot_product_attention(
    query_tensor,
    key_tensor,
    value_tensor,
    mask=padding_mask
)

print("输出形状:", output_tensor.shape)
print("注意力权重形状:", attention_weights_tensor.shape)

print("批次 0 中第一个查询的注意力权重 "
      "(最后两个键已被掩盖)：")
print(attention_weights_tensor[0, 0, :])
```

此函数封装了核心逻辑。请注意，掩码需要应用在 softmax *之前*。使用一个大的负数配合 `masked_fill` 有效地阻止了被掩盖的位置在 softmax 归一化 (normalization)后对加权和做出贡献。该函数返回最终的加权输出和注意力权重 (weight)本身，这有助于分析或可视化（正如我们将在第 23 章中看到的）。

这个基本组成部分现在将在多头注意力 (multi-head attention)机制 (attention mechanism)中使用，我们接下来将实现它。多头注意力将并行地多次运行此缩放点积注意力，并使用查询、键和值的不同学习到的投影。

获取即时帮助、个性化解释和交互式代码示例。

---

### 构建多头注意力层

# 构建多头注意力层

缩放点积注意力机制 (attention mechanism)允许模型关注序列的不同部分。多头注意力 (multi-head attention)机制，由“Attention Is All You Need”论文提出，进一步提升了此功能。多头注意力不是使用dmodeld\_{model}dmodel​维度的键、值和查询执行单个注意力函数，而是通过不同的、学习到的线性投影，将查询、键和值分别投影到dkd\_kdk​、dkd\_kdk​和dvd\_vdv​维度hhh次。接着对这些投影后的版本并行执行注意力操作。其输出被拼接起来并再次投影，得到最终结果。

基本思路是，每个“头”可以同时学习关注序列中不同类型的信息或关联。比如，一个头可能关注句法依赖，而另一个追踪指代关系。通过并行运行这些注意力机制并结合它们的输出，模型获得对输入更丰富、多层面的理解。

数学上，多头注意力定义为：

多头(Q,K,V)=拼接(头1,…,头h)WO\text{多头}(Q, K, V) = \text{拼接}(\text{头}\_1, \dots, \text{头}\_h)W^O多头(Q,K,V)=拼接(头1​,…,头h​)WO

其中每个头i\text{头}\_i头i​计算为：

头i=注意力(QWiQ,KWiK,VWiV)\text{头}\_i = \text{注意力}(QW\_i^Q, KW\_i^K, VW\_i^V)头i​=注意力(QWiQ​,KWiK​,VWiV​)

此处，Q,K,VQ, K, VQ,K,V是输入的查询、键和值矩阵。投影矩阵WiQ∈Rdmodel×dkW\_i^Q \in \mathbb{R}^{d\_{model} \times d\_k}WiQ​∈Rdmodel​×dk​、WiK∈Rdmodel×dkW\_i^K \in \mathbb{R}^{d\_{model} \times d\_k}WiK​∈Rdmodel​×dk​和WiV∈Rdmodel×dvW\_i^V \in \mathbb{R}^{d\_{model} \times d\_v}WiV​∈Rdmodel​×dv​是第iii个头的参数 (parameter)矩阵，而WO∈Rhdv×dmodelW^O \in \mathbb{R}^{hd\_v \times d\_{model}}WO∈Rhdv​×dmodel​是输出投影矩阵。在原版Transformer论文和许多常见实现中，维度设定为dk=dv=dmodel/hd\_k = d\_v = d\_{model} / hdk​=dv​=dmodel​/h。这使得计算成本与使用dmodeld\_{model}dmodel​维度键和值的单头注意力相似。

我们来在PyTorch中实现它。我们将创建一个`MultiHeadAttention`模块，它接收嵌入 (embedding)维度（`embed_dim`）、头数量（`num_heads`）以及可选的dropout概率作为输入。

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """ 实现多头注意力机制。 """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            "Embedding dimension must be divisible by number of heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):

        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def forward(self, query, key, value, mask=None):
        """
        多头注意力的前向传播。
        Args:
            query (torch.Tensor): 查询张量，
                                  形状 (batch_size, seq_len_q, embed_dim)
            (torch.Tensor): 张量，
                                形状 (batch_size, seq_len_k, embed_dim)
            value (torch.Tensor): 值张量，
                                  形状 (batch_size, seq_len_v, embed_dim)
                                注意：通常seq_len_k == seq_len_v。
            mask (torch.Tensor, optional): 掩码张量，用于阻止
                                           对某些位置的注意力。
                                           形状 (batch_size, 1, seq_len_q,
                                                  seq_len_k) 或类似
                                           可广播形状。
        Returns:
            torch.Tensor: 输出张量，
                          形状 (batch_size, seq_len_q, embed_dim)
            torch.Tensor: 注意力权重，
                          形状 (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size, seq_len_q, _ = query.size()

        _, seq_len_k, _ = key.size()
        _, seq_len_v, _ = value.size()
        assert seq_len_k == seq_len_v

        qkv = self.qkv_proj(query)

        k_proj = self.qkv_proj(key)
        v_proj = self.qkv_proj(value)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size,
                   seq_len_q,
                   self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch_size,
                   seq_len_k,
                   self.num_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size,
                   seq_len_v,
                   self.num_heads,
                   self.head_dim).transpose(1, 2)

        if mask is not None:

             if mask.dim() == 3:

                 mask = mask.unsqueeze(1)

             elif mask.dim() == 2:

                 mask = mask.unsqueeze(0).unsqueeze(0)

        attn_output, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.embed_dim
        )

        output = self.out_proj(attn_output)

        output = self.dropout(output)

        return output, attn_weights
```

在此实现中：

- 我们初始化了`embed_dim`、`num_heads`并计算了`head_dim`。重要的是，`embed_dim`必须能被`num_heads`整除。
- 我们使用一个单一的`nn.Linear`层（`qkv_proj`）来同时投影输入的查询、键和值张量以提高效率。然后我们将结果分割为`q`、`k`和`v`。另一种方案是为`q`、`k`和`v`定义单独的`nn.Linear`层。
- `_reset_parameters`方法处理权重 (weight)初始化，采用Xavier均匀初始化，这是Transformer层中的常见做法。
- 在`forward`方法中：
  - 我们使用`qkv_proj`投影输入。
  - 投影后的张量`q`、`k`、`v`被重塑以分离出各个头。维度变为`(batch_size, num_heads, seq_len, head_dim)`。`transpose(1, 2)`操作重新排列维度，使得头维度位于序列长度维度之前，这通常是注意力实现或优化过的核所期望的。
  - 我们对重塑后的`q`、`k`、`v`和可选的`mask`调用我们的`scaled_dot_product_attention`函数（我们假定此函数已在其他地方定义，例如上一节的代码或某个工具文件中）。掩码处理确保它在各个头之间正确广播。
  - 注意力头（`attn_output`）的输出被拼接回来。我们通过先转置回来（`transpose(1, 2)`），然后使用`contiguous().view()`将`num_heads`和`head_dim`维度合并回原始的`embed_dim`来实现。调用`contiguous()`是必要的，因为`transpose`可能返回一个非连续的张量，而`view`不能直接在其上操作。
  - 最后，拼接后的输出通过最终的线性层（`out_proj`）和一个可选的dropout层。

这个`MultiHeadAttention`模块封装了Transformer论文中描述的核心逻辑。它接收查询、键和值输入（在自注意力 (self-attention)层中它们通常是相同的张量），并生成一个与查询形状相同的输出张量，以及用于后续分析的注意力权重。这个模块将是我们接下来构建更大编码器和解码器层的构成单元。

G

clusterᵢnput

输入张量

clusterₘha

多头注意力模块

Query

查询
(bs, seq\_q, embed\_dim)

Proj

线性投影
(Q, K, V) \* h

Query->Proj

Q

Key

键
(bs, seqₖ, embed\_dim)

Key->Proj

K

Value

值
(bs, seqᵥ, embed\_dim)

Value->Proj

V

Split

分离头
(bs, h, seq, head\_dim)

Proj->Split

SDPA

缩放点积注意力
(并行)

Split->SDPA

Q, K, V (每个头)

Concat

拼接头
(bs, seq\_q, embed\_dim)

SDPA->Concat

注意力输出 (每个头)

AttnWeights

注意力权重
(bs, h, seq\_q, seqₖ)

SDPA->AttnWeights

FinalProj

最终线性投影

Concat->FinalProj

Output

输出
(bs, seq\_q, embed\_dim)

FinalProj->Output

流程图，显示了多头注意力模块内的步骤，从输入的查询、键、值张量到最终输出和注意力权重。`h`表示头的数量，`bs`代表批次大小。

获取即时帮助、个性化解释和交互式代码示例。

---

### 实现位置感知前馈网络

# 实现位置感知前馈网络

位置感知前馈网络 (FFN) 是每个 Transformer 模块中的一个组件。这个网络在序列的每个位置独立且相同地应用。尽管自注意力 (self-attention)层允许标记 (token)之间相互作用，FFN 则是单独处理每个标记的表示，为模型提供额外的非线性变换能力。

可以将其看作是在注意力层完成上下文 (context)混合之后，增加进一步的计算深度。它帮助模型为每个位置从注意力输出中提取的特征学习更复杂的函数。

### 前馈网络的结构

FFN 通常是一个简单的两层全连接网络。对于特定位置的输入表示 xxx，其变换定义为：

FFN(x)=Linear2(Activation(Linear1(x)))+x\text{FFN}(x) = \text{Linear}\_2(\text{Activation}(\text{Linear}\_1(x))) + xFFN(x)=Linear2​(Activation(Linear1​(x)))+x

等一下，上面的等式包含残差连接。让我们先拆解核心 FFN 部分。最常见的结构包含：

1. 一个线性变换，将维度从 dmodeld\_{model}dmodel​ 扩展到内部维度，该内部维度通常记作 dffd\_{ff}dff​。
2. 一个逐元素应用的非线性激活函数 (activation function)。原始 Transformer 论文中使用了 ReLU，但其他激活函数如 GeLU（高斯误差线性单元）或 SwiGLU 在现代大型语言模型中已变得流行。我们将在第 11 章讨论这些替代方法。为了本次实现简单起见，我们将使用 ReLU。
3. 第二个线性变换，将维度从 dffd\_{ff}dff​ 投射回 dmodeld\_{model}dmodel​。

FFN 操作本身的公式（在模块结构中处理的残差连接之前）是：

FFN(x)=max⁡(0,xW1+b1)W2+b2\text{FFN}(x) = \max(0, xW\_1 + b\_1)W\_2 + b\_2FFN(x)=max(0,xW1​+b1​)W2​+b2​

这里：

- xxx 是特定位置的输入向量 (vector)（前一层的输出，通常是多头注意力 (multi-head attention)后接层归一化 (normalization)）。
- W1W\_1W1​ 和 b1b\_1b1​ 是第一个线性层的权重 (weight)矩阵和偏置 (bias)（输出维度为 dffd\_{ff}dff​）。
- W2W\_2W2​ 和 b2b\_2b2​ 是第二个线性层的权重矩阵和偏置（输出维度为 dmodeld\_{model}dmodel​）。
- max⁡(0,⋅)\max(0, \cdot)max(0,⋅) 表示 ReLU 激活函数。

内部维度 dffd\_{ff}dff​ 通常大于 dmodeld\_{model}dmodel​。一个常见的选择是 dff=4×dmodeld\_{ff} = 4 \times d\_{model}dff​=4×dmodel​，正如原始论文“Attention Is All You Need”中使用的那样。这种扩展使得模型在投射回原始模型维度之前，能够学习到更丰富的表示。

### PyTorch 实现

让我们将这个 FFN 组件实现为一个 PyTorch `nn.Module`。我们将包含 Dropout，它通常在 FFN 中的第二个线性层之后应用，或作为残差连接步骤的一部分。

```python
import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    """实现位置感知前馈网络（FFN）模块。"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        初始化 PositionWiseFeedForward 模块。

        参数:
            d_model (int): 输入和输出特征的维度。
            d_ff (int): 内部层的维度。
            dropout (float): Dropout 概率。默认值为 0.1。
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FFN 模块的前向传播。

        参数:
            x (torch.Tensor): 形状为 (batch_size, seq_len, d_model) 的输入张量。

        返回:
            torch.Tensor: 形状为 (batch_size, seq_len, d_model) 的输出张量。
        """

        x = self.linear1(x)
        x = self.activation(x)

        x = self.linear2(x)
        x = self.dropout(x)
        return x
```

让我们使用一些示例维度来测试这个模块：

```python

d_model = 512
d_ff = 2048
dropout_rate = 0.1
batch_size = 4
seq_len = 10

input_tensor = torch.randn(batch_size, seq_len, d_model)

ffn_layer = PositionWiseFeedForward(d_model, d_ff, dropout_rate)

output_tensor = ffn_layer(input_tensor)

print(f"输入形状: {input_tensor.shape}")
print(f"输出形状: {output_tensor.shape}")

assert output_tensor.shape == (batch_size, seq_len, d_model)
```

这段代码定义了 `PositionWiseFeedForward` 类。`__init__` 方法设置了两个线性层（`self.linear1`、`self.linear2`）、ReLU 激活函数 (activation function)（`self.activation`）和 Dropout 层（`self.dropout`）。`forward` 方法定义了计算流程：输入 xxx 经过第一个线性层，然后是 ReLU 激活，接着是第二个线性层，最后是 Dropout。

请注意，`linear1`、`activation` 和 `linear2` 这些操作在每个序列位置的表示上是独立应用的。这些层在一次前向传播中在不同位置共享权重 (weight)，但位置 `i` 的计算不直接依赖于 *这个* FFN 模块中位置 `j` 的计算（与注意力机制 (attention mechanism)不同）。

这个 FFN 模块是一个基本构成单元，我们将在后续章节将其整合到更大的编码器和解码器层中。它的作用是提供非线性处理能力，这种能力在多头注意力 (multi-head attention)子层完成上下文 (context)聚合之后，在所有序列位置上均匀应用。

获取即时帮助、个性化解释和交互式代码示例。

---

### 构建编码器层和解码器层

# 构建编码器层和解码器层

多头注意力 (multi-head attention)机制 (attention mechanism)和位置前馈网络这两个主要组成部分可以组装成Transformer的标准构成单元：编码器层和解码器层。这些层不仅包含注意力子网络和前馈子网络，还包含了残差连接和层归一化 (normalization)，这些对于有效训练深度Transformer模型是必不可少的。

## 编码器层

编码器层处理输入序列，通过自注意力 (self-attention)和前馈网络细化其表示。每个编码器层包含两个主要的子层：

1. **多头自注意力：** 允许输入序列中的每个位置关注所有位置（包括自身），以获取背景信息。
2. **位置前馈网络 (FFN)：** 对每个位置的表示独立地应用相同的变换。

重要地，在*每个*子层之后都会应用一个残差连接，接着是层归一化 (normalization)。这种结构有助于梯度流动并稳定激活，从而避免深度层堆叠中出现梯度消失或梯度爆炸等问题。

单个编码器层的数据流可以如下所示：

G

Input

输入嵌入
(来自上一层或输入)

MHA

多头
自注意力

Input->MHA

AddNorm1

加法与归一化

Input->AddNorm1

MHA->AddNorm1

FFN

位置
前馈网络

AddNorm1->FFN

AddNorm2

加法与归一化

AddNorm1->AddNorm2

FFN->AddNorm2

Output

输出表示

AddNorm2->Output

> Transformer编码器层内的数据流。虚线表示残差连接。

让我们在PyTorch中实现它。我们假设 `MultiHeadAttention` 和 `PositionwiseFeedForward` 是基于前面章节的实现而定义好的类。

```python
import torch
import torch.nn as nn
import copy

class EncoderLayer(nn.Module):
    """
    表示Transformer编码器的一层。

    它由一个多头自注意力机制，后接一个
    位置全连接前馈网络组成。残差连接
    和层归一化在每个子层之后应用。
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        参数：
            d_model: 输入和输出的维度
                     （嵌入维度）。
            num_heads: 注意力头的数量。
            d_ff: 前馈网络内部层的维度。
            dropout: dropout比率。
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        将输入通过编码器层。

        参数：
            x: 层的输入张量 (batch_size, seq_len, d_model)。
            mask: 自注意力机制的掩码（可选）。
                  通常用于忽略填充标记。
                  形状为 (batch_size, 1, seq_len) 或
                  (batch_size, seq_len, seq_len)。

        返回：
            层的输出张量 (batch_size, seq_len, d_model)。
        """

        attn_output, _ = self.self_attn(query=x, key=x, value=x, mask=mask)

        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)

        x = self.norm2(x + self.dropout(ff_output))

        return x
```

在 `forward` 方法中，请注意这种模式：输入 `x` 在经过层归一化 (`self.norm1` 或 `self.norm2`) 之前，会被添加到子层（经过dropout后）的输出。这是“加法与归一化”步骤，对于训练的稳定性很重要。如果需要，`mask` 参数 (parameter)会传递给自注意力层，以避免关注填充标记 (token)。

## 解码器层

解码器层与编码器层有相似之处，但增加了一个子层，用于处理来自编码器输出的信息。每个解码器层包含三个主要的子层：

1. **带掩码的多头自注意力 (self-attention)：** 运作方式类似于编码器的自注意力，但增加了一个掩码，以避免位置关注后续位置。这确保了位置 iii 的预测只依赖于位置小于 iii 的已知输出，从而在生成过程中保持所需的自回归 (autoregressive)特性。
2. **多头交叉注意力（编码器-解码器注意力）：** 这一层允许解码器输入序列中的每个位置关注*编码器输出序列*中的*所有*位置。查询来自解码器的带掩码自注意力输出，而键和值来自编码器堆栈的最终输出。这是解码器引入输入序列信息的方式。
3. **位置前馈网络 (FFN)：** 结构上与编码器层中使用的相同。

与编码器类似，这三个子层中的每一个都跟随一个残差连接和层归一化 (normalization)。

以下是解码器层的数据流：

G

DecoderInput

输入嵌入
(来自上一解码器层或目标输入)

MaskedMHA

带掩码多头
自注意力

DecoderInput->MaskedMHA

AddNorm1

加法与归一化

DecoderInput->AddNorm1

EncoderOutput

编码器输出
(记忆)

CrossMHA

多头
交叉注意力

EncoderOutput->CrossMHA

键, 值

MaskedMHA->AddNorm1

AddNorm1->CrossMHA

查询

AddNorm2

加法与归一化

AddNorm1->AddNorm2

CrossMHA->AddNorm2

FFN

位置
前馈网络

AddNorm2->FFN

AddNorm3

加法与归一化

AddNorm2->AddNorm3

FFN->AddNorm3

Output

输出表示

AddNorm3->Output

> Transformer解码器层内的数据流。虚线表示残差连接。编码器输出（记忆）为交叉注意力子层提供键和值。

现在，让我们在PyTorch中实现 `DecoderLayer`。

```python
import torch
import torch.nn as nn
import copy

class DecoderLayer(nn.Module):
    """
    表示Transformer解码器的一层。

    它由带掩码的自注意力、交叉注意力（关注
    编码器输出）和一个位置前馈网络组成。残差
    连接和层归一化在每个子层之后应用。
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        参数：
            d_model: 输入和输出的维度。
            num_heads: 注意力头的数量。
            d_ff: 前馈网络内部层的维度。
            dropout: dropout比率。
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                src_mask: torch.Tensor | None = None,
                tgt_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        将输入通过解码器层。

        参数：
            x: 解码器层的输入张量
               (batch_size, tgt_seq_len, d_model)。
            memory: 编码器堆栈的输出张量
                    (batch_size, src_seq_len, d_model)。
            src_mask: 交叉注意力（编码器-解码器注意力）层的掩码，用于忽略
                      编码器输出中的填充标记（可选）。
                      形状为 (batch_size, 1, src_seq_len)。
            tgt_mask: 带掩码自注意力层的掩码，结合了
                      前瞻掩码和目标填充掩码（可选）。
                      形状为 (batch_size, tgt_seq_len, tgt_seq_len)。

        返回：
            层的输出张量
            (batch_size, tgt_seq_len, d_model)。
        """

        self_attn_output, _ = self.self_attn(query=x,
                                             x,
                                             value=x,
                                             mask=tgt_mask)

        x = self.norm1(x + self.dropout(self_attn_output))

        cross_attn_output, _ = self.cross_attn(query=x,
                                               key=memory,
                                               value=memory,
                                               mask=src_mask)

        x = self.norm2(x + self.dropout(cross_attn_output))

        ff_output = self.feed_forward(x)

        x = self.norm3(x + self.dropout(ff_output))

        return x
```

`DecoderLayer` 实现中的要点：

1. **带掩码自注意力：** `self_attn` 层接收 `tgt_mask`。这个掩码通常是填充掩码（如果目标序列有填充）和前瞻掩码（一个下三角矩阵）的组合，以确保因果关系。
2. **交叉注意力：** `cross_attn` 层使用第一个加法与归一化块 (`x`) 的输出作为其 `query`。重要地，`key` 和 `value` 来自 `memory` 参数 (parameter)，它代表了编码器堆栈的最终输出。这里的 `src_mask` 用于忽略原始源序列（编码器输入）中的填充标记 (token)。
3. **三个子层：** 请注意与三个子层对应的三个不同的“加法与归一化”步骤。

定义了这些 `EncoderLayer` 和 `DecoderLayer` 类之后，我们就有了构建完整编码器和解码器堆栈所需的基本组成部分。这只需简单地创建这些层的多个副本，并将一层的输出作为下一层的输入。这种堆叠使得模型能够学习输入和目标序列越来越复杂的表示。我们将在下一节中组装这些堆栈。

获取即时帮助、个性化解释和交互式代码示例。

---

### 组装完整的Transformer模型

# 组装完整的Transformer模型

为了构建完整的 Transformer 模型，`EncoderLayer` 和 `DecoderLayer` 作为独立的组成部分被使用。构建过程包括堆叠这些层，添加必需的嵌入 (embedding)层，加入位置编码 (positional encoding)，以及定义最终的输出投影。

回顾一下，Transformer 架构采用编码器-解码器结构。编码器处理输入序列并生成富含上下文 (context)信息的表示（通常称为 `memory`）。解码器随后将此 `memory` 与目标序列（训练期间）或之前生成的标记 (token)（推理 (inference)期间）一同使用，以生成输出序列。

我们来定义一个用于完整 Transformer 的 PyTorch `nn.Module`。

```python
import torch
import torch.nn as nn
import math

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                 key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2, _ = self.multihead_attn(
            tgt, memory, memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数：
            x: 张量，形状为 [batch_size, seq_len, embedding_dim]
        """

        pe_for_seq = self.pe[:x.size(1), :].permute(1, 0, 2)
        x = x + pe_for_seq
        return self.dropout(x)

class TransformerModel(nn.Module):
    """
    完整的 Transformer 模型实现。
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int, nhead: int, num_encoder_layers: int,
                 num_decoder_layers: int, dim_feedforward: int,
                 dropout: float = 0.1, max_len: int = 5000):
        """
        参数：
            src_vocab_size: 源词汇表大小。
            tgt_vocab_size: 目标词汇表大小。
            d_model: 嵌入和模型层的维度。
            nhead: 注意力头数量。
            num_encoder_layers: 堆叠编码器层的数量。
            num_decoder_layers: 堆叠解码器层的数量。
            dim_feedforward: 前馈网络隐藏层的维度。
            dropout: Dropout 比率。
            max_len: 位置编码的最大序列长度。
        """
        super().__init__()
        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """初始化 Transformer 模型中的参数。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None,
               src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        将源序列通过编码器堆栈。

        参数：
            src: 源序列张量（batch_size, src_seq_len）。
            src_mask: 源序列注意力掩码（src_seq_len, src_seq_len）。
                      如果需要，可防止注意力集中于未来位置（编码器通常不需要）。
            src_key_padding_mask: 源序列中填充标记的掩码（batch_size, src_seq_len）。

        返回：
            编码器输出张量（batch_size, src_seq_len, d_model）。
        """

        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask=src_mask,
                           src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: torch.Tensor = None,
               memory_mask: torch.Tensor = None,
               tgt_key_padding_mask: torch.Tensor = None,
               memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        将目标序列和编码器 memory 通过解码器堆栈。

        参数：
            tgt: 目标序列张量（batch_size, tgt_seq_len）。
            memory: 编码器输出张量（batch_size, src_seq_len, d_model）。
            tgt_mask: 目标序列自注意力掩码（tgt_seq_len, tgt_seq_len）。
                      防止注意力集中于未来位置。
            memory_mask: 编码器-解码器注意力掩码（tgt_seq_len, src_seq_len）。
                         通常不需要，除非需要特定的交叉注意力掩码。
            tgt_key_padding_mask: 目标序列中填充标记的掩码（batch_size, tgt_seq_len）。
            memory_key_padding_mask: 源序列中填充标记的掩码，用于编码器-解码器注意力（batch_size, src_seq_len）。

        返回：
            解码器输出张量（batch_size, tgt_seq_len, d_model）。
        """

        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        output = tgt_emb
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
        return output

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor = None,
                tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Transformer 模型的完整前向传播。

        参数：
            src: 源序列张量（batch_size, src_seq_len）。
            tgt: 目标序列张量（batch_size, tgt_seq_len）。
            src_mask: 源序列注意力掩码。
            tgt_mask: 目标序列自注意力掩码。
            memory_mask: 编码器-解码器注意力掩码。
            src_key_padding_mask: 源序列的填充掩码。
            tgt_key_padding_mask: 目标序列的填充掩码。
            memory_key_padding_mask: 用于交叉注意力的源序列填充掩码。

        返回：
            输出对数张量（batch_size, tgt_seq_len, tgt_vocab_size）。
        """
        memory = self.encode(src, src_mask, src_key_padding_mask)
        decoder_output = self.decode(tgt, memory, tgt_mask, memory_mask,
                                     tgt_key_padding_mask,
                                     memory_key_padding_mask)
        logits = self.generator(decoder_output)
        return logits
```

### 集成各部分

1. **初始化 (`__init__`)**:

   - 我们为源词汇表 (vocabulary)和目标词汇表初始化独立的嵌入 (embedding)层（`nn.Embedding`）。这些嵌入的大小为 `d_model`。
   - 创建一个单独的 `PositionalEncoding` 模块。由于其计算与词汇表无关，因此可以共享。请注意，在添加位置编码 (positional encoding)之前，令牌嵌入会乘以 `sqrt(d_model)`，这与原始论文中保持一致。
   - 我们使用 `nn.ModuleList` 来存放 `EncoderLayer` 和 `DecoderLayer` 实例的堆叠。这确保了层被正确注册为子模块。
   - 一个最终的 `nn.Linear` 层，`self.generator`，将解码器堆栈的 `d_model` 维度输出投影到目标词汇表（`tgt_vocab_size`）上的对数。
   - 应用了使用 Xavier 均匀初始化进行权重 (weight)初始化（`_reset_parameters`），这是 Transformer 的常见做法。
2. **编码 (`encode`)**:

   - 此方法包含编码器逻辑。它接收源序列（`src`），应用嵌入和位置编码，然后按顺序将结果通过堆栈中的每个 `EncoderLayer`。
   - 它接受掩码（`src_mask`，`src_key_padding_mask`），这些掩码会被传递到每个 `EncoderLayer` 内部的底层注意力机制 (attention mechanism)。`src_key_padding_mask` 对于防止对填充标记 (token)进行注意力计算很重要。
   - 编码器堆栈的最终输出（`memory`）表示已处理的源序列上下文 (context)。
3. **解码 (`decode`)**:

   - 此方法处理解码器逻辑。它接收目标序列（`tgt`）、来自编码器的 `memory` 以及各种掩码。
   - 与编码器类似，它对目标序列应用嵌入和位置编码。
   - 结果按顺序通过每个 `DecoderLayer`。每个 `DecoderLayer` 对目标序列执行自注意力 (self-attention)（使用 `tgt_mask` 和 `tgt_key_padding_mask`），并与编码器 `memory` 执行交叉注意力（使用 `memory_mask` 和 `memory_key_padding_mask`）。`tgt_mask` 在此尤其重要，用于阻止解码器在训练期间关注未来的标记（因果掩码）。
   - 最终输出是解码器对目标序列的表示，以源序列为条件。
4. **前向传播 (`forward`)**:

   - 主要的 `forward` 方法组织整个过程。
   - 它首先调用 `encode` 来获取 `memory`。
   - 然后，它使用目标序列和 `memory` 调用 `decode`。
   - 最后，它将 `generator` 线性层应用于解码器输出，以生成目标序列中每个位置的最终对数。这些对数随后可在训练期间与损失函数 (loss function)（如交叉熵）一同使用，或在推理 (inference)期间进行进一步处理（例如，使用 argmax 或采样）。

### 数据流图

以下图表展示了组装后的 Transformer 模型中的高层数据流：

G

clusterᵢnput

输入

clusterₑmbedding

嵌入与位置编码

clusterₑncoder

编码器堆栈

cluster\_decoder

解码器堆栈

clusterₒutput

输出投影

src

源标记

srcₑmb

源嵌入
+ 位置编码

src->srcₑmb

tgt

目标标记

tgtₑmb

目标嵌入
+ 位置编码

tgt->tgtₑmb

encoder

N x 编码器层

srcₑmb->encoder

decoder

N x 解码器层

tgtₑmb->decoder

memory

记忆 (上下文)

encoder->memory

memory->decoder

generator

线性 + Softmax
(隐含)

decoder->generator

output

输出概率

generator->output

> Transformer 模型中的高层数据流，展示了从输入标记 (token)经过嵌入 (embedding)、编码器、解码器和最终输出投影的路径。

这个完整的 `TransformerModel` 类提供了一个功能实现，基于之前开发的组件。它包含核心架构，可随时集成到训练循环中。后续章节将在此基础上进行构建，研究如何扩展此架构，在大规模数据集上高效训练它，并针对各种任务优化它。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 11 Scaling Transformers Architectural Choices

### 神经网络语言模型的缩放定律

# 神经网络语言模型的缩放定律

在扩展Transformer模型时，一个自然而然的问题出现了：我们如何最佳地投入计算资源？我们应该优先增加层数、拓宽隐藏维度，还是在更多数据上训练？仅仅在没有明确策略的情况下增加参数 (parameter)可能导致收益递减或昂贵计算周期的低效利用。幸运的是，经验研究揭示了模型表现、模型大小、数据集大小以及训练中使用的计算量之间可预测的关系。这些关系通常被称为“缩放定律”。

理解这些缩放定律提供了一个有价值的框架，用于对模型架构和训练方案做出明智的决定。它们使我们能够估计扩大训练过程不同方面所带来的预期性能提升，并协助优化计算预算的分配。

### 缩放定律的经验基础

开创性工作，值得注意的是OpenAI的Kaplan等人（2020），证明了语言模型的表现（通常通过在保留测试集上的交叉熵损失来衡量），随着我们扩展模型大小（非嵌入 (embedding)参数 (parameter)数量，NNN）、数据集大小（训练标记 (token)数量，DDD）和计算量（总浮点运算，FLOPsFLOPsFLOPs, CCC）而可预测地提高。

核心发现是，当其他因素不是瓶颈时，测试损失LLL通常与NNN、DDD和CCC遵循幂律关系。对于模型大小NNN和数据集大小DDD，这种关系通常可以建模为：

L(N,D)≈L∞+(NcN)αN+(DcD)αDL(N, D) \approx L\_{\infty} + \left(\frac{N\_c}{N}\right)^{\alpha\_N} + \left(\frac{D\_c}{D}\right)^{\alpha\_D}L(N,D)≈L∞​+(NNc​​)αN​+(DDc​​)αD​

其中：

- L(N,D)L(N, D)L(N,D)是预测的交叉熵损失。
- L∞L\_{\infty}L∞​表示一个不可约的损失下限，可能与语言固有的熵或建模方法的局限性有关。
- NNN是模型中非嵌入参数的数量。
- DDD是训练数据集中标记的数量。
- NcN\_cNc​、DcD\_cDc​、αN\alpha\_NαN​和αD\alpha\_DαD​是模型架构和训练设置的特定常数，通过将公式拟合到不同小规模训练运行的结果中来凭经验确定。通常，αN\alpha\_NαN​和αD\alpha\_DαD​小于1，表示随着NNN或DDD的增加，收益递减。

这个公式表明，损失主要受对应于限制因素的项的影响：如果模型太小（N≪NcN \ll N\_cN≪Nc​），增加NNN会显著有益；如果数据集太小（D≪DcD \ll D\_cD≪Dc​），增加DDD更有益。

类似地，损失通常随计算预算CCC呈幂律关系：

L(C)≈L∞+(CcC)αCL(C) \approx L\_{\infty} + \left(\frac{C\_c}{C}\right)^{\alpha\_C}L(C)≈L∞​+(CCc​​)αC​

其中CcC\_cCc​和αC\alpha\_CαC​也是凭经验确定的常数。这些关系通常在几个数量级内成立，使其对推断有用。

10M25100M251B2510B25100B4567891234567891023观测损失幂律拟合

> 一幅对数-对数图，展示测试损失如何随模型大小增加而降低，通常在几个数量级内遵循可预测的幂律。

### 计算最优缩放与Chinchilla研究结果

对缩放定律理解的一项重要改进来自于DeepMind的Hoffmann等人（2022），被称为“Chinchilla”研究。他们进行了仔细的分析，以确定固定计算预算CCC在模型大小NNN和数据大小DDD之间的最佳分配。

训练密集型Transformer模型的计算量、模型大小和数据大小之间的近似关系通常估算为：

C≈6×N×DC \approx 6 \times N \times DC≈6×N×D

这表明计算成本大致与参数 (parameter)数量乘以处理的标记 (token)数量成正比。鉴于固定的计算预算CCC，Chinchilla研究建议，为了获得最佳表现（最低损失），NNN和DDD都应该大致按照计算预算的平方根比例进行缩放。也就是说，如果计算预算翻倍，您应该尝试将*模型大小*和*数据集大小*都增加大约2≈1.4\sqrt{2} \approx 1.42​≈1.4倍。

这一发现很重要，因为它表明在这项研究之前训练的许多大型语言模型（如GPT-3或Gopher）相对于其训练数据可能“参数过多”了。对于所使用的计算预算，通过在更多数据上训练更小的模型，可能已经取得了更好的表现。Chinchilla模型本身，根据这些“计算最优”原则进行训练，以更少的参数但更多的训练数据，取得了比其一些大型同期模型更出色的成果。

让我们用一个简化的计算来说明。假设我们拟合了一个缩放定律，发现当N≈kCN \approx k \sqrt{C}N≈kC​和D≈C6N≈C6kC=C6kD \approx \frac{C}{6N} \approx \frac{C}{6k\sqrt{C}} = \frac{\sqrt{C}}{6k}D≈6NC​≈6kC​C​=6kC​​（其中kkk为某个常数）时，会获得最佳表现。如果我们有一个计算预算C1C\_1C1​并找到了最优的N1,D1N\_1, D\_1N1​,D1​，那么对于更大的预算C2=4C1C\_2 = 4 C\_1C2​=4C1​，最优参数将近似为N2≈k4C1=2N1N\_2 \approx k \sqrt{4C\_1} = 2 N\_1N2​≈k4C1​​=2N1​和D2≈4C16k=2D1D\_2 \approx \frac{\sqrt{4C\_1}}{6k} = 2 D\_1D2​≈6k4C1​​​=2D1​。我们将模型和数据都与C\sqrt{C}C​成比例地缩放。

### 训练大型语言模型的实际意义

这些缩放定律对构建大型模型的工程师有几点直接影响：

1. **资源分配：** 它们为是否投资更多计算硬件（以增加NNN和DDD）、更多数据获取和清洗（以增加DDD），或者可能改变常数（Nc,Dc,αN,αDN\_c, D\_c, \alpha\_N, \alpha\_DNc​,Dc​,αN​,αD​）的架构改进提供了量化 (quantization)指导。
2. **性能预测：** 通过运行小规模实验来估计缩放参数 (parameter)（αN,αD\alpha\_N, \alpha\_DαN​,αD​等），团队可以在投入完整训练运行所需的大量资源之前，预测更大模型的预期表现。这降低了风险并有助于证明计算请求的合理性。
3. **预算优化：** 对于固定预算CCC，这些定律有助于确定NNN和DDD之间的最佳平衡，以最大化预期表现。这避免了模型因其大小或训练数据而严重受限的情况。
4. **基准测试：** 缩放定律为评估新架构或训练技术提供了基准。如果一种新方法产生的表现显著优于标准Transformer的既定缩放定律所预测的，则表明其是真正的改进。

### 计算需求估计

我们可以使用C≈6NDC \approx 6NDC≈6ND公式来估计训练的计算成本。例如，在1万亿个标记 (token)（D=1×1012D=1 \times 10^{12}D=1×1012）上训练一个70亿参数 (parameter)模型（N=7×109N=7 \times 10^9N=7×109）大约需要：

```python
import math

N = 7e9

D = 1e12

C_flops = 6 * N * D

petaflops = 1e15
seconds_per_day = 86400

C_petaflop_days = C_flops / (petaflops * seconds_per_day)

print(f"估算计算量: {C_flops:.2e} FLOPs")
print(f"估算计算量: {C_petaflop_days:.2f} Petaflop-天")
```

这一计算凸显了所涉及的巨大计算规模。一个能够持续10 Petaflops的集群，进行这样的运行大约需要49天，这还不考虑开销和潜在的低效率。这强调了为何由缩放定律指导的有效资源分配如此重要。

### 注意事项与考量

尽管它们功能强大，但仍需记住这些缩放定律的背景：

- **经验性：** 它们是观察到的趋势，而非普适定律。它们严重依赖于特定的架构（例如Transformer）、优化器、数据分布以及其他训练细节。从用于推导这些定律的实验范围之外进行过度推断会带来风险。
- **数据质量：** 这些定律主要建模数据的*数量*（DDD）。然而，数据质量、多样性和相关性也是影响最终模型表现的非常重要的因素，这些因素并未在这些简单公式中明确体现。在大量低质量数据上进行训练可能比在少量但更干净的数据上训练产生更差的结果。
- **不可约损失：** L∞L\_{\infty}L∞​项暗示了一个性能限制。随着模型变得非常大并在大量数据集上训练，当它们接近这个极限时，改进可能会放缓。
- **计算估算：** 6ND6ND6ND公式是一个粗略的近似值。实际的FLOPs可能会因具体的实现、序列长度和硬件效率而异。更精确的计算可能会单独考虑前向和反向传播 (backpropagation)。

总而言之，缩放定律提供了一个宝贵的定量视角，用于审视构建更大、更强大语言模型 (LLM)的过程。它们将缩放从猜测游戏转变为更可预测的工程学科，从而更有效地利用计算资源，并提供了一个评估该领域进展的框架。在接下来的章节中，当我们讨论具体的架构选择时，请记住这些缩放原则，因为它们经常是大型Transformer设计决策的依据。

获取即时帮助、个性化解释和交互式代码示例。

---

### 深度与宽度取舍

# 深度与宽度取舍

在扩展Transformer模型时，可以调整的两个主要架构方面是其深度（层数）和宽度（隐藏表示与前馈网络的尺寸）。扩展法则表明，仅仅增加总参数 (parameter)量并不能说明全部。这些参数在深度和宽度之间的分配，很大程度上影响着模型的行为、训练动态和计算效率。没有一个普遍的最优比例；最佳选择通常取决于具体的任务、数据集、可用计算资源以及所需的推理 (inference)特点。

下面我们来审视这项取舍中涉及的因素。

### 增加模型深度（更多层）

添加更多层（增加编码器或解码器块的数量 NNN），能使模型学习输入数据更复杂、分层级的表示。每层都在前一层执行的转换基础上进行构建，从而实现一个更深的处理管道。

- **潜在优势：**

  - **分层特征学习：** 更深的模型可能更适合捕捉语言中复杂组合结构，其中特征是逐层构建的。
  - **参数 (parameter)效率（可能）：** 对于某些任务，与较浅、较宽的模型相比，在更深、更窄的模型中可能需要更少的总参数才能达到某个性能水平，尽管并非总是如此。
- **潜在缺点：**

  - **优化挑战：** 非常深的神经网络 (neural network)由于潜在的梯度消失或梯度爆炸，可能更难训练，尽管残差连接和层归一化 (normalization)（特别是稍后讨论的Pre-LN）等技术很大程度上缓解了这些问题。
  - **顺序计算：** 前向和反向传播 (backpropagation)涉及顺序处理每层。增加深度会直接增加这个顺序路径的长度，与增加宽度相比，这可能会减慢训练迭代速度，前提是存在并行硬件。
  - **更长的训练时间：** 即使每次迭代没有慢很多，更深的模型也可能需要更多的训练步骤或周期才能有效收敛。

### 增加模型宽度（更大维度）

增加宽度通常涉及扩大核心嵌入 (embedding)维度（dmodeld\_{model}dmodel​），并且通常会按比例增加位置前馈网络中的中间维度（dffd\_{ff}dff​，通常设置为 4×dmodel4 \times d\_{model}4×dmodel​）。

- **潜在优势：**

  - **每层容量增加：** 每层有更大的容量来学习复杂转换，并在其表示中存储信息。
  - **并行性：** 在较宽的层内进行的计算（例如，FFN中的大型矩阵乘法）通常可以在单个加速器（GPU/TPU）内的计算单元之间更有效地并行化。
  - **可能更快的收敛速度（每步）：** 较宽的模型有时在训练步骤方面可能更快地学习某些模式，尽管每一步需要更长时间并使用更多内存。
- **潜在缺点：**

  - **内存占用：** 模型内存需求，特别是训练期间的激活，随宽度显著增加。前馈层通常主导参数 (parameter)数量和内存使用，随 dmodeld\_{model}dmodel​ 呈二次方增长（因为 dffd\_{ff}dff​ 通常与 dmodeld\_{model}dmodel​ 成比例）。注意力机制 (attention mechanism)也随 dmodeld\_{model}dmodel​ 扩展。这会很快成为瓶颈，即使对于中等宽度，也需要更先进的分布式训练策略（如张量并行）。
  - **计算成本：** 每层的计算成本，尤其是在FFN中（O(dmodel×dff)≈O(dmodel2)O(d\_{model} \times d\_{ff}) \approx O(d\_{model}^2)O(dmodel​×dff​)≈O(dmodel2​)），随宽度大幅增加。自注意力 (self-attention)机制中的计算也随 dmodeld\_{model}dmodel​ 扩展。
  - **注意力复杂度：** 虽然自注意力的二次方复杂度主要与序列长度（nnn）相关，但常数因子涉及 dmodeld\_{model}dmodel​（O(n2⋅dmodel)O(n^2 \cdot d\_{model})O(n2⋅dmodel​)），因此宽度也增加了这方面的成本。

### 寻找平衡：来自扩展法则的认识

对扩展法则的实证研究，例如 Kaplan 等人 (2020) 的工作，表明在固定计算预算下，为了达到最佳性能，模型大小、数据集大小和训练计算应根据幂律同步扩展。在扩展模型大小时，研究显示，同时增加深度和宽度比仅仅大幅扩展一个维度能带来更好的结果。

例如，像 GPT-3 这样的架构使用较多层数（例如96）*和* 较大的隐藏维度尺寸（例如12288）。精确的比例通常会随着模型的增长而演变；比较 GPT-2 和 GPT-3 显示，深度和宽度都有增加，但并非一定成比例。

The decision often comes down to:

1. **计算预算：** 更宽的模型对内存容量和每层计算要求更高，可能需要更复杂的并行技术（张量并行）。更深的模型则对顺序计算时间要求更高。
2. **推理 (inference)延迟：** 更深的模型由于其顺序性可能导致更高的延迟，而更宽的模型则可能由于每层的计算需求而导致更高的延迟。
3. **训练稳定性：** 尽管存在稳定深层模型的技术，但过深的深度有时仍可能带来难题。
4. **任务要求：** 某些任务可能本质上更受益于分层处理（深度），而另一些任务可能更受益于每一步的更宽表示（宽度）。

### 示例：在PyTorch中配置深度和宽度

在使用PyTorch这样的框架时，深度和宽度通常在模型初始化期间由特定参数 (parameter)控制。考虑 `nn.TransformerEncoder`：

```python
import torch
import torch.nn as nn

vocab_size = 30000
d_model = 512
nhead = 8
num_encoder_layers = 6
dim_feedforward = 2048
dropout = 0.1

embedding = nn.Embedding(vocab_size, d_model)

encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    batch_first=True
)

transformer_encoder = nn.TransformerEncoder(
    encoder_layer=encoder_layer,
    num_layers=num_encoder_layers
)
```

在此代码片段中：

- `num_encoder_layers` 直接控制**深度**。
- `d_model`、`nhead` 和 `dim_feedforward` 控制**宽度**。

对这些值进行实验对于扩展Transformer模型非常重要。您可以比较一个 `num_encoder_layers=12, d_model=768` 的模型与一个 `num_encoder_layers=24, d_model=512` 的模型，同时保持总参数量或计算成本大致相当，以便通过实验理解这些取舍。

### 可视化取舍

考虑两个参数 (parameter)量大致相似但架构不同的模型：

- **模型 A（深而窄）：** 较多层，较小的隐藏尺寸。
- **模型 B（浅而宽）：** 较少层，较大的隐藏尺寸。

每层内存顺序计算路径00.511.522.5模型 A（深/窄）模型 B（浅/宽）

> 此图表说明，与参数量相似但较浅、较宽的模型（B）相比，一个更深、更窄的模型（A）可能每层内存需求更低，但顺序计算路径更长，而模型 B 则表现出相反的趋势。

总而言之，在深度和宽度之间做出选择是一种权衡。虽然更深的模型提供了进行复杂分层学习的可能性，但它们可能增加顺序计算时间并带来优化难题。更宽的模型每层提供更大的容量，并且可能在内部更好地并行化，但每层会带来显著更高的内存和计算成本。现代大型模型通常会同时扩展这两个维度，这受到扩展法则研究的实证结果指导，并受限于可用的硬件和训练基础设施。通常需要仔细实验才能找到最适合您特定目标的配置。

获取即时帮助、个性化解释和交互式代码示例。

---

### 激活函数选择 (ReLU, GeLU, SwiGLU)

# 激活函数选择 (ReLU, GeLU, SwiGLU)

每个Transformer模块中的前馈网络 (FFN) 子层在改变注意力机制 (attention mechanism)学习到的表示方面发挥着重要作用。标准FFN由两个线性变换和一个非线性激活函数 (activation function)组成：

FFN(x)=max(0,xW1+b1)W2+b2FFN(x) = max(0, xW\_1 + b\_1)W\_2 + b\_2FFN(x)=max(0,xW1​+b1​)W2​+b2​

其中，xxx 是注意力子层的输入，W1W\_1W1​、b1b\_1b1​、W2W\_2W2​ 和 b2b\_2b2​ 是可学习参数 (parameter)，所示的激活函数是ReLU。这种非线性是必不可少的；没有它，两个线性层将合并为一个线性变换，从而限制模型的表达能力。

随着模型规模扩大，这种激活函数的选择不仅仅是一个小细节。它会影响梯度流动、训练稳定性、计算成本，并最终影响模型的最终表现。下面我们检视大型Transformer中常见的选项：ReLU、GeLU和SwiGLU。

### 修正线性单元 (ReLU)

修正线性单元，即ReLU，定义为 ReLU(x)=max(0,x)ReLU(x) = max(0, x)ReLU(x)=max(0,x)，是深度学习 (deep learning)的基本激活函数 (activation function)。它的主要优点是简单性和计算效率。它避免了在深度网络中经常与sigmoid或tanh函数出现的梯度消失问题。

```python
import torch
import torch.nn as nn

d_model = 512
d_ff = 2048
relu_ffn = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.ReLU(),
    nn.Linear(d_ff, d_model)
)

x = torch.randn(16, 128, d_model)
output = relu_ffn(x)
print("Output shape:", output.shape)
```

然而，ReLU并非没有缺点。主要问题是“死亡ReLU”问题：如果神经元的输入持续低于零，它们可能变得不活跃，导致它们的权重 (weight)停止更新，因为在该区域梯度为零。虽然像仔细初始化和降低学习率等技术可以减轻此问题，但它仍然需要考虑，特别是在非常深的神经网络 (neural network)中。此外，它在 x=0x=0x=0 处的非平滑性质有时会阻碍优化，相较于更平滑的替代品。

### 高斯误差线性单元 (GeLU)

高斯误差线性单元 (GeLU) 作为ReLU的一种更平滑的替代品被提出，并随着BERT和GPT系列模型而获得广泛应用。它根据输入值进行加权，但这种加权是随机的，结合了标准高斯累积分布函数 (Φ(x)\Phi(x)Φ(x))。

GeLU(x)=x⋅Φ(x)GeLU(x) = x \cdot \Phi(x)GeLU(x)=x⋅Φ(x)

由于计算精确高斯累积分布函数可能较慢，因此常使用近似方法：

GeLU(x)≈0.5x(1+tanh⁡[2/π(x+0.044715x3)])GeLU(x) \approx 0.5x \left(1 + \tanh\left[\sqrt{2/\pi}(x + 0.044715x^3)\right]\right)GeLU(x)≈0.5x(1+tanh[2/π​(x+0.044715x3)])

直观理解是GeLU提供比ReLU更平滑的曲线，可能使得优化更容易，梯度流动更好。经验上，它在Transformer模型中通常表现优于ReLU。

```python
import torch
import torch.nn as nn

d_model = 512
d_ff = 2048
gelu_ffn = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.GELU(),
    nn.Linear(d_ff, d_model)
)

x = torch.randn(16, 128, d_model)
output = gelu_ffn(x)
print("Output shape:", output.shape)
```

GeLU的计算量略大于ReLU，但受到硬件加速器的良好支持。它在许多基本大型语言模型中的成功使其在多年来成为一个标准选择。

### Swish门控线性单元 (SwiGLU)

近期，FFN层中涉及门控机制的变体表现出良好性能。一个流行变体是SwiGLU，它在PaLM论文中提出，并用于Llama等模型。

核心理念是将Swish激活函数 (activation function) (Swish(x)=x⋅σ(x)Swish(x) = x \cdot \sigma(x)Swish(x)=x⋅σ(x)，其中 σ\sigmaσ 是sigmoid函数) 与门控机制结合。SwiGLU通常不使用单个线性层来扩展维度，而是使用两个线性层，它们的输出进行逐元素相乘。其中一个输出通过Swish函数，作为另一个的门。

SwiGLU(x,W,V,b,c)=(xW+b)⊗Swish(xV+c)SwiGLU(x, W, V, b, c) = (xW + b) \otimes Swish(xV + c)SwiGLU(x,W,V,b,c)=(xW+b)⊗Swish(xV+c)

其中，xxx 是输入，WWW、VVV、bbb 和 ccc 是可学习参数 (parameter)，⊗\otimes⊗ 表示逐元素相乘。Swish函数定义为：

Swish(x)=x⋅σ(βx)Swish(x) = x \cdot \sigma(\beta x)Swish(x)=x⋅σ(βx)
通常，β\betaβ 设置为1或设为可学习参数。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, bias=True):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)

        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):

        hidden1 = self.w1(x)
        hidden2 = self.w2(x)

        gated_hidden = F.silu(hidden1) * hidden2

        output = self.w3(gated_hidden)
        return output

d_model = 512

d_ff_swiglu = 1024

swiglu_ffn = SwiGLUFFN(d_model, d_ff_swiglu)

x = torch.randn(16, 128, d_model)
output = swiglu_ffn(x)
print("Output shape:", output.shape)
```

关于SwiGLU（以及GeGLU等类似门控激活函数）一个细微但重要的点是其对参数数量的影响。为了使SwiGLU实现中使用的 `hidden_dim`（如上文的 `d_ff_swiglu`）与中间维度为 dffd\_{ff}dff​ 的标准ReLU/GeLU FFN保持相似的参数数量，该 `hidden_dim` 通常设置为约 23dff\frac{2}{3} d\_{ff}32​dff​。这是因为SwiGLU使用两个线性投影 (WWW 和 VVV) 来达到中间维度，有效地分摊了标准FFN中通常由一个更大矩阵 (W1W\_1W1​) 处理的容量。尽管如此，SwiGLU在大型模型中经常被发现能够得到更好的困惑度分数和下游表现，相比GeLU或ReLU，这表明门控机制有好处。

### 比较与选择

−3−2−1012300.511.522.53ReLUGeLU (近似)Swish (β=1)

> ReLU、GeLU（近似）和Swish激活函数 (activation function)的比较。请注意从ReLU到GeLU再到Swish，平滑度逐渐增加。

选择合适的激活函数涉及权衡：

- **ReLU：** 计算最快，形式最简单。有神经元“死亡”的风险，非平滑。目前在最先进的大型模型中较少见，但仍然可用。
- **GeLU：** 性能和计算成本之间有良好平衡。比ReLU更平滑，在许多知名大型语言模型中经验上表现出色。一个可靠的默认选择。
- **SwiGLU（及其他门控变体）：** 通常在近期大型模型中获得最佳性能（例如，更低的困惑度）。引入了门控，可能对信息流动提供更好的控制。需要在隐藏维度方面仔细实施，以有效管理参数 (parameter)数量。由于门控乘法，计算成本可能略高于GeLU。

在扩展Transformer模型时，从ReLU转向GeLU或SwiGLU是一种旨在提升表现的常见架构变化。尽管FFN实现较复杂，SwiGLU带来的性能提升促使其被多个近期大型模型采用。与许多架构选择一样，最佳选择可能取决于特定的模型大小、数据集和计算预算，通常需要经验验证。

获取即时帮助、个性化解释和交互式代码示例。

---

### 规范化层放置位置（前置LN vs. 后置LN）

# 规范化层放置位置（前置LN vs. 后置LN）

随着Transformer模型规模的扩大，增加深度会带来潜在的训练难题。尽管残差连接旨在缓解梯度消失，但规范化层的具体放置位置对训练时的表现有重要影响，尤其是在非常深的神经网络 (neural network)中。层规范化（LN）本身通过在特征维度上规范化层的输入来稳定训练，确保其具有零均值和单位方差。问题在于，这种规范化在Transformer模块中应该相对于子层（例如自注意力 (self-attention)或前馈网络）和残差连接发生*在何处*。

两种主要方法是后置层规范化（Post-Layer Normalization，简称Post-LN）和前置层规范化（Pre-Layer Normalization，简称Pre-LN）。

### 后置层规范化（原始Transformer）

原始论文《Attention Is All You Need》提出了Post-LN配置。在这种设置下，子层的输出会与输入（残差连接）相加，*然后*应用层规范化。

数据流如下所示：

`output = LayerNorm(x + Sublayer(x))`

G

xᵢn

输入 (x)

sublayer

子层（注意力/FFN）

xᵢn->sublayer

add

+

xᵢn->add

 残差

sublayer->add

ln

层规范化

add->ln

xₒut

输出

ln->xₒut

> Post-LN Transformer模块中的数据流。规范化发生在残差相加之后。

尽管对于中等深度的模型有效，但随着层数显著增加（例如，在12-24层中），Post-LN会遇到稳定性问题。主要问题在于，残差通路的直接输出（图中的`x`）在规范化*之前*被添加到转换（`Sublayer(x)`）的输出上。如果子层输出的幅度变化很大或逐层增大，这些相加可能导致进入下一个LayerNorm的激活值产生很大的方差。这可能导致网络深处出现梯度爆炸或梯度消失，通常需要仔细的学习率预热策略（在训练开始时逐渐增加学习率）以及精确的超参数 (parameter) (hyperparameter)调整来防止发散。训练深层Post-LN模型可能会比较困难。

### 前置层规范化

为了解决Post-LN在非常深的模型中的稳定性问题，提出了前置层规范化（最早在GPT-2等模型中受到关注，尽管早期也有类似变体）。在这里，层规范化应用于输入，*在*其通过子层*之前*。残差连接随后将未规范化的输入`x`添加到子层的输出上。

数据流变为：

`output = x + Sublayer(LayerNorm(x))`

G

xᵢn

输入 (x)

ln

层规范化

xᵢn->ln

add

+

xᵢn->add

 残差

sublayer

子层（注意力/FFN）

ln->sublayer

sublayer->add

xₒut

输出

add->xₒut

> Pre-LN Transformer模块中的数据流。规范化发生在子层之前。

这种看似微小的改变对训练稳定性有重要影响。通过对每个子层的输入进行规范化，Pre-LN确保了注意力网络和前馈网络处理的激活值具有一致的尺度（零均值，单位方差），无论网络深度如何。反向传播 (backpropagation)通过网络的梯度也通常表现更好，因为规范化步骤有效地“重置”了每个残差块输入的尺度。这使得训练深层Transformer模型更加稳定，通常允许更高的学习率，并减少对非常长的预热期（warmup periods）的严格要求（尽管预热通常仍然有益）。

这里是一个简化的PyTorch示例，说明了Transformer模块中的结构差异：

```python
import torch
import torch.nn as nn

class PostLNBlock(nn.Module):
    def __init__(self,
                 d_model,
                 sublayer,
                 dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.dropout(self.sublayer(x))
        x = residual + x
        x = self.norm(x)
        return x

class PreLNBlock(nn.Module):
    def __init__(self,
                 d_model,
                 sublayer,
                 dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.norm(x)
        x = self.dropout(self.sublayer(x))
        x = residual + x
        return x

d_model = 512

dummy_sublayer = nn.Linear(d_model, d_model)

post_ln_block = PostLNBlock(d_model, dummy_sublayer)
pre_ln_block = PreLNBlock(d_model, dummy_sublayer)

input_tensor = torch.randn(
    32, 10, d_model
)

output_post = post_ln_block(input_tensor)
output_pre = pre_ln_block(input_tensor)

print("Post-LN 输出形状：", output_post.shape)
print("Pre-LN 输出形状：", output_pre.shape)
```

### 权衡与建议

主要的权衡是**稳定性与潜在的最高性能**。

- **Pre-LN：** 提供了显著更好的训练稳定性，尤其对于具有数十或数百层的模型。它对学习率和预热时长等超参数 (parameter) (hyperparameter)选择不太敏感。这种稳固性使其成为当前大多数大规模LLM训练的首选。梯度在整个训练过程中表现更好。
- **Post-LN：** 在训练成功时，有时可以获得略好一些的最终性能指标（例如，更低的困惑度）。然而，实现成功的训练通常需要更细致地调整学习率计划、预热步数和初始化。在大规模训练中，它通常被认为更难稳定训练。

考虑到训练大型语言模型的巨大计算成本和时长，**Pre-LN**提供的改善的稳定性和稳固性通常胜过Post-LN可能带来的微小性能提升。在数周的训练过程中，由于不稳定性而出现意外发散是一种代价高昂的挫折。因此，在构建大型、深层Transformer模型时，Pre-LN架构是推荐且被广泛采用的标准。

获取即时帮助、个性化解释和交互式代码示例。

---

### 稀疏注意力机制简介

# 稀疏注意力机制简介

随着Transformer模型的规模扩大，标准自注意力 (self-attention)机制 (attention mechanism)的计算需求很快成为一个主要瓶颈。自注意力计算序列中所有token之间的成对交互。如果序列长度为NNN，计算注意力分数矩阵(QKTQ K^TQKT)的复杂度是O(N2d)O(N^2 d)O(N2d)，存储此矩阵和中间激活所需的内存也是O(N2)O(N^2)O(N2)，其中ddd是模型维度。对于几百甚至几千个token的序列来说尚可管理，但这种二次方扩展阻止了将标准Transformer应用于很长的序列，例如整个文档、视为补丁序列的高分辨率图像或扩展的音频流。处理长度为64,000的序列，其注意力计算所需的计算量将是长度为512的序列的1600万倍以上。

稀疏注意力机制提供了一种实用的办法，它通过修改自注意力层来仅计算所有可能的成对交互中的一个子集，有效地用稀疏矩阵替换了稠密的N×NN \times NN×N注意力矩阵。其主要假设是，对于许多任务，一个token不需要来自*所有*其他token的信息；相反，相关上下文 (context)可能是局部的，或者特定的全局token可能充当信息枢纽。通过选择哪些token对进行交互，我们的目标是将计算复杂度从O(N2)O(N^2)O(N2)降低到更易于处理的程度，通常是O(Nlog⁡N)O(N \log N)O(NlogN)甚至O(N)O(N)O(N)，同时保留了模型大部分的表达能力。

### 常见的稀疏模式

存在几种策略用于定义哪些token对应该相互关注。选择通常取决于对数据性质和任务的假设。

1. **滑动窗口（局部）注意力**：这是最简单的模式之一。每个token只关注固定数量www的前后token（其局部窗口）。复杂度变为O(N⋅w)O(N \cdot w)O(N⋅w)，如果www是常数，则相对于NNN是线性的。当局部上下文 (context)最重要时，例如在因果语言建模或图像处理中，这种模式很有效。
2. **膨胀滑动窗口**：为了捕获更长距离的依赖关系，同时不过度增加窗口大小www，可以引入膨胀机制。一个token可能会关注其窗口内距离为1、2、4、8等的邻居，类似于膨胀卷积。这使得感受野可以随层数呈指数增长，同时保持计算的线性。
3. **全局注意力**：某些token可能需要访问整个序列上下文，或作为信息的集成点。在此模式中，少量预先选择的token（例如，BERT类模型中的`[CLS]` token，或根据任务指定为重要的token）关注所有其他token，并且所有其他token也关注这些全局token。这通常与其他模式结合使用，例如滑动窗口注意力。
4. **随机注意力**：每个token除了关注其局部窗口外，还会关注固定数量的随机选择的token。这有助于信息以概率方式在序列中传播。
5. **分解注意力**：这涉及将完整注意力分解为多个成本较低的步骤。例如，注意力可能首先在固定的token块内计算，然后第二个注意力步骤可能发生在这些块的摘要表示之间。

### 示例：Longformer模式

Longformer架构提供了一个著名的例子，它结合了其中的几个思路。它主要使用滑动窗口注意力机制 (attention mechanism)。然而，为了实现信息在整个序列中的流动，它增加了全局注意力。根据任务确定的特定token（例如，用于分类的`[CLS]` token，用于问答的问题token）被允许全局关注，并且所有token都关注它们。

G

clusterₗocal

局部窗口（大小 3）

cluster\_global

全局注意力（Token 0）

a0

a1

a0->a1

a2

a0->a2

a3

a0->a3

a1->a2

a1->a3

a4

a1->a4

g0

G

a1->g0

a2->a3

a2->a4

a5

a2->a5

a2->g0

a3->a4

a3->a5

a6

a3->a6

a3->g0

a4->a5

a4->a6

a7

a4->a7

a4->g0

a5->a6

a5->a7

a5->g0

a6->a7

a6->g0

a7->g0

g0->a1

g0->a2

g0->a3

g0->a4

g0->a5

g0->a6

g0->a7

> 组合注意力模式的简化示意图。蓝色节点代表具有局部（滑动窗口）注意力的token。黄色节点（G）具有全局注意力，与所有其他token交互（橙色边）。局部连接显示为灰色。

这种组合使Longformer能够处理数千个token长度的序列，同时保持局部上下文 (context)感知和全局信息整合的能力，所有这些都伴随着计算复杂度随序列长度NNN线性增长的特点。

### 实现上的挑战

高效实现稀疏注意力通常不仅仅是在softmax之前应用一个掩码。标准的深度学习 (deep learning)库实现针对稠密矩阵乘法进行了高度优化。通过稀疏性获得性能提升通常需要：

- **定制核心（Kernel）**：使用专门的GPU核心（例如，用CUDA编写或使用Triton等库），可以高效地计算稀疏模式上的注意力，避免对零值项进行计算和内存访问。
- **块稀疏格式**：以适合硬件加速的格式表示稀疏注意力矩阵，例如块稀疏矩阵。OpenAI的Triton等库为此提供了工具。
- **管理稀疏模式**：代码需要高效地生成或检索定义每个token稀疏连接的索引。

这是一个在PyTorch中高度简化的草图，展示了如何为一个滑动窗口模式创建稀疏掩码。请注意，这*不*代表一个高效的实现，但它说明了掩码的原理。

```python
import torch
import torch.nn.functional as F

def simple_sliding_window_mask(sequence_length, window_size):
    """
创建滑动窗口注意力掩码。
    注意：仅用于说明，对于大型序列效率不高。
    """
    mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool)
    half_window = window_size // 2
    for i in range(sequence_length):

        start = max(0, i - half_window)
        end = min(sequence_length, i + half_window + 1)

        mask[i, :start] = 0
        mask[i, end:] = 0

    return mask

seq_len = 10
window = 3
attention_scores = torch.randn(1, seq_len, seq_len)

sparse_mask = simple_sliding_window_mask(seq_len, window)

attention_scores.masked_fill_(~sparse_mask.unsqueeze(0), float('-inf'))

attention_probs = F.softmax(attention_scores, dim=-1)

print("注意力掩码（True=允许）：\n", sparse_mask)
print(
    "\n掩码后的注意力概率（第0行）：\n",
    attention_probs[0, 0].detach().numpy().round(2)
)
```

"这段代码生成一个布尔掩码，其中`True`表示允许的注意力连接。然后，此掩码用于在softmax之前将不允许连接的分数设置为负无穷，以确保它们获得零概率。稀疏注意力实现绕过了完整稠密矩阵的创建，直接计算稀疏交互。"

稀疏注意力是一个活跃的研究方面，各种模式和高效实现不断出现。尽管与标准注意力相比它们增加了复杂性，但它们是促使Transformer架构应用于涉及超长序列问题的关键因素，拓展了大型模型能够处理的界限。权衡在于计算效率和通过限制token交互可能导致的信息损失之间。评估这种权衡通常需要在目标任务上进行经验性测试。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 12 Initialization Techniques Deep Networks

### 恰当初始化的重要性

# 恰当初始化的重要性

训练深度神经网络 (neural network)，特别是现代LLM中使用的巨型Transformer架构，需要仔细关注诸多细节。模型参数 (parameter)的初始状态，特别是其权重 (weight)，是影响训练过程的一个主要方面。简单地将权重初始化为零或从朴素分布中抽取的微小随机数，常导致严重的训练难题。恰当的初始化不只是一种启发式方法；它是使深度网络能够有效学习的基本要求。

核心问题源于信号（包括前向传播时的激活和反向传播 (backpropagation)时的梯度）如何在网络层间传递。设想单层内的计算，通常涉及矩阵乘法后接非线性激活函数 (activation function)。当多层这样堆叠时，这些操作会依序重复。

### 梯度消失

在反向传播 (backpropagation)过程中，损失函数 (loss function)对早期层中某个权重 (weight)的梯度，是使用链式法则计算的，即乘以所有后续层的梯度。如果这些梯度（或层变换的雅可比矩阵）的平均大小小于1，梯度信号在反向传播时会指数级减小。

∂L∂Wl∝(∏k=l+1NJk)∂L∂aN\frac{\partial L}{\partial W\_l} \propto \left( \prod\_{k=l+1}^{N} J\_k \right) \frac{\partial L}{\partial a\_N}∂Wl​∂L​∝(k=l+1∏N​Jk​)∂aN​∂L​

这里，JkJ\_kJk​ 表示第 kkk 层的雅可比矩阵。如果平均而言 ∥Jk∥<1\|J\_k\| < 1∥Jk​∥<1，乘积项会随着层数 N−lN-lN−l 的增加而迅速减小。

这种现象被称为**梯度消失问题**，意味着早期层中的权重得到极其微小的更新。因此，这些层学习非常缓慢，有时甚至完全不学。这实际上阻止了网络学习依赖于其全部深度参数 (parameter)配合调整的复杂表示。历史上，这是训练深度网络的一大障碍，尤其是那些使用如sigmoid或tanh等激活函数 (activation function)的网络，这些函数在饱和区域的导数小于1。

### 梯度爆炸

反之，如果这些梯度（或雅可比矩阵）的平均大小大于1，梯度信号在反向通过各层时会指数级增长。

∥∂L∂Wl∥→∞as N−l→∞if ∥Jk∥>1\| \frac{\partial L}{\partial W\_l} \| \rightarrow \infty \quad \text{as } N-l \rightarrow \infty \quad \text{if } \|J\_k\| > 1∥∂Wl​∂L​∥→∞as N−l→∞if ∥Jk​∥>1

这种**梯度爆炸问题**导致权重 (weight)更新过大。过大的更新会使优化过程变得不稳定，可能导致越过最佳点或剧烈震荡。在极端情况下，梯度变得如此之大，以至于导致数值溢出（表示为`NaN`或`Inf`值），使训练过程停滞。尽管梯度裁剪等方法（将在第17章讨论）可以在训练期间缓解梯度爆炸，但恰当的初始化旨在从一开始就防止它们频繁出现。

让我们看一个简化的模拟。设想一个具有10层的非常简单的线性网络。我们将权重初始化为略小或略大的值，并观察一个模拟梯度反向传播 (backpropagation)后的大小。

```python
import torch
import torch.nn as nn
import math

def check_grad_magnitude(init_scale, num_layers=10):
    """
    模拟通过线性层的反向梯度大小。
    """

    layer_dim = 100
    layers = []
    for _ in range(num_layers):
        layer = nn.Linear(layer_dim, layer_dim, bias=False)

        nn.init.normal_(layer.weight, mean=0.0, std=init_scale)
        layers.append(layer)

    network = nn.Sequential(*layers)

    x = torch.randn(1, layer_dim)

    output_grad = torch.ones(1, layer_dim)

    current_grad = output_grad

    with torch.no_grad():
        for i in range(num_layers - 1, -1, -1):

            current_grad = current_grad @ layers[i].weight

    input_grad_norm = torch.norm(current_grad)
    return input_grad_norm.item()

num_layers = 10
small_init_scale = 0.05
large_init_scale = 0.5
ideal_init_scale = math.sqrt(1.0 / 100)

grad_norm_small = check_grad_magnitude(small_init_scale, num_layers)
grad_norm_large = check_grad_magnitude(large_init_scale, num_layers)
grad_norm_ideal = check_grad_magnitude(ideal_init_scale, num_layers)

print(
    f"Initialization Scale: {small_init_scale:.3f}, "
    f"Final Gradient Norm: {grad_norm_small:.4e}"
)
print(
    f"Initialization Scale: {ideal_init_scale:.3f}, "
    f"Final Gradient Norm: {grad_norm_ideal:.4e}"
)
print(
    f"Initialization Scale: {large_init_scale:.3f}, "
    f"Final Gradient Norm: {grad_norm_large:.4e}"
)
```

上述简单模拟（不含归一化 (normalization)和非线性，它们会使情况更复杂）说明了权重尺度如何直接影响经过多层反向传播后的梯度大小。较小的初始尺度会导致梯度范数消失，而较大的尺度则会导致其爆炸。一个“理想”的尺度（例如Xavier/Glorot初始化中提示的 1/fan\_in1/\sqrt{\text{fan\\_in}}1/fan\_in​）有助于使梯度范数保持在其原始大小附近。

### 前向传播的稳定性

初始化也影响前向传播。如果权重 (weight)过大，线性层的输出会大幅增长，可能将激活函数 (activation function)的输入推到饱和区域（例如，对于sigmoid或tanh），使得梯度接近零。这再次阻碍了学习。反之，如果权重过小，激活值可能会逐层减弱，导致表示趋近于零，并降低网络的有效容量。

因此，我们将要讨论的Xavier和Kaiming初始化等原则性权重初始化策略的目标，是根据层维度谨慎设置权重的初始尺度。目的是确保前向传播中的激活和反向传播 (backpropagation)中的梯度在整个网络中保持合理的方差，从而防止信号消失或爆炸，进而促进更快、更稳定的训练收敛。这对本课程中非常重要的深度Transformer模型尤为重要。

获取即时帮助、个性化解释和交互式代码示例。

---

### Xavier（Glorot）初始化

# Xavier（Glorot）初始化

如前所述，初始化深度网络的权重 (weight)需要仔细考虑。简单地从标准正态或均匀分布中抽取权重通常会导致训练过程不稳定。梯度在反向传播 (backpropagation)时可能呈指数级缩小（梯度消失），或呈指数级增大（梯度爆炸），阻碍模型有效学习。

Xavier 初始化方法由 Xavier Glorot 和 Yoshua Bengio 于 2010 年提出，旨在解决深度网络训练中梯度消失或爆炸等常见挑战，其原理是使激活值和梯度的方差在各层之间大致保持不变。保持方差一致有助于确保信号（前向传播时的激活值和反向传播时的梯度）在穿过网络时不会消失或爆炸。

### 核心思想：方差保持

考虑神经网络 (neural network)中的一个线性层，执行操作 y=Wx+by = Wx + by=Wx+b，其中 WWW 是权重 (weight)矩阵，xxx 是输入向量 (vector)，bbb 是偏置 (bias)向量，yyy 是激活函数 (activation function)前的输出。Xavier 初始化分析了方差如何通过该层传播。

核心假设是：

1. 输入 xxx 是独立同分布 (i.i.d.) 的，均值为零，方差为 Var(x)Var(x)Var(x)。
2. 权重 WWW 是独立同分布 (i.i.d.) 的，均值为零，方差为 Var(W)Var(W)Var(W)。
3. 权重和输入相互独立。
4. 激活函数 fff 关于零对称（例如 tanhtanhtanh），并且其在零附近的导数近似为 1 (f′(0)≈1f'(0) \approx 1f′(0)≈1)。

在这些假设下，层输出 yiy\_iyi​（对于单个神经元 iii）的方差可以与输入方差和权重方差相关联：

Var(yi)=ninVar(Wij)Var(xj)Var(y\_i) = n\_{in} Var(W\_{ij}) Var(x\_j)Var(yi​)=nin​Var(Wij​)Var(xj​)

这里，ninn\_{in}nin​ 表示神经元的输入连接数（“扇入”）。为了使输出方差 Var(yi)Var(y\_i)Var(yi​) 等于输入方差 Var(xj)Var(x\_j)Var(xj​)，我们需要：

ninVar(Wij)=1n\_{in} Var(W\_{ij}) = 1nin​Var(Wij​)=1
Var(Wij)=1ninVar(W\_{ij}) = \frac{1}{n\_{in}}Var(Wij​)=nin​1​

这个条件确保了激活值的方差在前向传播过程中不会显著改变。

同样，考虑到反向传播 (backpropagation)，我们分析梯度的方差。梯度相对于前一层激活值的方差取决于扇出 noutn\_{out}nout​（当前层输出连接到的神经元数量）。为了保持梯度方差，条件变为：

noutVar(Wij)=1n\_{out} Var(W\_{ij}) = 1nout​Var(Wij​)=1
Var(Wij)=1noutVar(W\_{ij}) = \frac{1}{n\_{out}}Var(Wij​)=nout​1​

Xavier 初始化通过对这些约束进行平均，在这两个条件（前向传播时保持激活方差和反向传播时保持梯度方差）之间寻求一种折衷：

Var(Wij)=2nin+noutVar(W\_{ij}) = \frac{2}{n\_{in} + n\_{out}}Var(Wij​)=nin​+nout​2​

此方差构成了初始化策略的依据。

### Xavier 初始化公式

权重 (weight)通常从均匀分布或正态分布中抽取，并根据此推导出的方差进行缩放。

1. **Xavier 均匀初始化：** 权重从均匀分布 U[−a,a]U[-a, a]U[−a,a] 中采样，其中 aaa 的选择使得方差为 2nin+nout\frac{2}{n\_{in} + n\_{out}}nin​+nout​2​。U[−a,a]U[-a, a]U[−a,a] 的方差是 (2a)212=a23\frac{(2a)^2}{12} = \frac{a^2}{3}12(2a)2​=3a2​。将其设置为目标方差：
   a23=2nin+nout\frac{a^2}{3} = \frac{2}{n\_{in} + n\_{out}}3a2​=nin​+nout​2​
   a2=6nin+nouta^2 = \frac{6}{n\_{in} + n\_{out}}a2=nin​+nout​6​
   a=6nin+nouta = \sqrt{\frac{6}{n\_{in} + n\_{out}}}a=nin​+nout​6​​

   因此，权重使用以下方式初始化：
   W∼U[−6nin+nout,6nin+nout]W \sim U\left[-\sqrt{\frac{6}{n\_{in} + n\_{out}}}, \sqrt{\frac{6}{n\_{in} + n\_{out}}}\right]W∼U[−nin​+nout​6​​,nin​+nout​6​​]
2. **Xavier 正态初始化：** 权重从均值为 0、目标方差为 σ2\sigma^2σ2 的正态分布 N(0,σ2)N(0, \sigma^2)N(0,σ2) 中采样：
   σ2=2nin+nout\sigma^2 = \frac{2}{n\_{in} + n\_{out}}σ2=nin​+nout​2​

   因此，权重使用以下方式初始化：
   W∼N(0,2nin+nout)W \sim N\left(0, \sqrt{\frac{2}{n\_{in} + n\_{out}}}\right)W∼N(0,nin​+nout​2​​)

在两种情况下，ninn\_{in}nin​ 是层的输入数量（扇入），noutn\_{out}nout​ 是层的输出数量（扇出）。偏置 (bias)通常初始化为零。

### 适用性和实现

Xavier 初始化适用于后面跟着在零附近大致线性且对称的激活函数 (activation function)（如 tanhtanhtanh 和逻辑 Sigmoid）的层。假设 f′(0)≈1f'(0) \approx 1f′(0)≈1 对于这些函数是比较成立的。然而，它不太适合整流线性单元 (ReLU) 及其变体，这些函数是非对称的，并且对负输入导数为 0。这一局限促成了 Kaiming 初始化的发展，我们接下来会讨论。

在 PyTorch 中，您可以使用 `torch.nn.init` 模块轻松应用 Xavier 初始化。

```python
import torch
import torch.nn as nn

fan_in, fan_out = 512, 256

linear_layer = nn.Linear(fan_in, fan_out, bias=True)

nn.init.xavier_uniform_(linear_layer.weight)

if linear_layer.bias is not None:
    nn.init.constant_(linear_layer.bias, 0)

print(f"层: Linear({fan_in}, {fan_out})")
print("初始化: Xavier 均匀")
print(f"权重均值: {linear_layer.weight.mean():.4f}, 标准差: {linear_layer.weight.std():.4f}")

theoretical_std_uniform = (2.0 / (fan_in + fan_out))**0.5
print(f"理论标准差（基于均匀）: {theoretical_std_uniform:.4f}\n")

linear_layer_norm = nn.Linear(fan_in, fan_out, bias=True)

nn.init.xavier_normal_(linear_layer_norm.weight)

if linear_layer_norm.bias is not None:
    nn.init.constant_(linear_layer_norm.bias, 0)

print("初始化: Xavier 正态")
print(
    f"权重均值: {linear_layer_norm.weight.mean():.4f}, "
    f"标准差: {linear_layer_norm.weight.std():.4f}"
)

theoretical_std_normal = (2.0 / (fan_in + fan_out))**0.5
print(f"理论标准差（正态）: {theoretical_std_normal:.4f}")
```

> 示例输出显示了 Xavier 初始化后的权重 (weight)统计信息。初始化权重的实际标准差应接近从扇入和扇出推导出的理论值。

Xavier 初始化提供了一种合理的方法来设置初始权重，有助于深度网络中更稳定的信号传播，特别是使用对称激活函数的网络。尽管它比朴素的随机初始化有了重要的改进，但其假设并非完全符合所有现代网络架构，尤其是那些高度依赖 ReLU 激活的网络。这引出了我们的下一种方法，Kaiming 初始化，它专门为此类情况而设计。

获取即时帮助、个性化解释和交互式代码示例。

---

### Kaiming (何) 初始化

# Kaiming (何) 初始化

Xavier初始化旨在平衡使用tanhtanhtanh等对称激活函数 (activation function)的层的方差。然而，随着修正线性单元（ReLU）激活函数的普及，情况发生了显著变化。ReLU函数定义为f(x)=max(0,x)f(x) = max(0, x)f(x)=max(0,x)，引入了非对称性：它对所有负输入输出零。这意味着，平均而言，从ReLU单元输出的激活值可能有一半为零。Xavier初始化的原理是为对称激活函数设计的，不能完全处理这种行为。因此，将Xavier初始化与ReLU结合使用时，信号在前向传播过程中仍可能导致方差逐渐减小，这可能会减慢训练速度，或在非常深的神经网络 (neural network)中导致梯度消失。

认识到这种不匹配，何凯明等人G在其论文《Exploring Rectifiers: Surpassing Human-Level Performance on ImageNet Classification》中提出了一种专门为ReLU及其变体设计的初始化方案。其核心思想是在计算初始权重 (weight)的合适方差时，明确考虑ReLU引入的非线性。

### 推导 Kaiming 方差

让我们考虑一个线性层 y=Wx+by = Wx + by=Wx+b。假设输入 xxx 的均值为零，并且权重 (weight) WWW 独立初始化且均值为零，单个神经元 iii 的输出 yiy\_iyi​（激活前）的方差由下式给出：

Var(yi)=∑j=1ninVar(Wijxj)Var(y\_i) = \sum\_{j=1}^{n\_{in}} Var(W\_{ij} x\_j)Var(yi​)=j=1∑nin​​Var(Wij​xj​)

假设 WijW\_{ij}Wij​ 和 xjx\_jxj​ 相互独立，且 E[xj]=0E[x\_j]=0E[xj​]=0：

Var(Wijxj)=E[Wij2xj2]−(E[Wijxj])2Var(W\_{ij} x\_j) = E[W\_{ij}^2 x\_j^2] - (E[W\_{ij} x\_j])^2Var(Wij​xj​)=E[Wij2​xj2​]−(E[Wij​xj​])2
Var(Wijxj)=E[Wij2]E[xj2]−(E[Wij]E[xj])2Var(W\_{ij} x\_j) = E[W\_{ij}^2] E[x\_j^2] - (E[W\_{ij}] E[x\_j])^2Var(Wij​xj​)=E[Wij2​]E[xj2​]−(E[Wij​]E[xj​])2
Var(Wijxj)=Var(Wij)Var(xj)Var(W\_{ij} x\_j) = Var(W\_{ij}) Var(x\_j)Var(Wij​xj​)=Var(Wij​)Var(xj​)

因此，对 ninn\_{in}nin​ 个输入（即扇入）求和：

Var(yi)=ninVar(W)Var(x)Var(y\_i) = n\_{in} Var(W) Var(x)Var(yi​)=nin​Var(W)Var(x)

现在，令 z=f(y)z = f(y)z=f(y) 为应用ReLU激活函数 (activation function) fff 后的输出。何凯明等人的观点是ReLU如何影响方差。如果 yyy 是均值为零的线性层的输出，它会对称地分布在零附近。ReLU将负值设为零。如果我们假设 xxx 来自之前的ReLU层，则方差计算需要调整。然而，如果我们将注意力放在通过当前层并应用ReLU激活函数 fff 的前向传播上，我们有 zi=max(0,yi)z\_i = max(0, y\_i)zi​=max(0,yi​)。如果 yiy\_iyi​ 的均值为零且对称，则 E[zi2]=12E[yi2]E[z\_i^2] = \frac{1}{2} E[y\_i^2]E[zi2​]=21​E[yi2​]。由于 E[yi]=0E[y\_i]=0E[yi​]=0，所以 E[yi2]=Var(yi)E[y\_i^2] = Var(y\_i)E[yi2​]=Var(yi​)。因此，Var(zi)≈E[zi2]=12Var(yi)Var(z\_i) \approx E[z\_i^2] = \frac{1}{2} Var(y\_i)Var(zi​)≈E[zi2​]=21​Var(yi​)。

替换 Var(yi)Var(y\_i)Var(yi​) 的表达式：

Var(zi)≈12ninVar(W)Var(x)Var(z\_i) \approx \frac{1}{2} n\_{in} Var(W) Var(x)Var(zi​)≈21​nin​Var(W)Var(x)

为了保持信号传播的稳定性，我们希望激活函数输出的方差 (Var(zi)Var(z\_i)Var(zi​)) 大致等于层输入的方差 (Var(x)Var(x)Var(x))。这要求：

Var(zi)=Var(x)  ⟹  1≈12ninVar(W)Var(z\_i) = Var(x) \implies 1 \approx \frac{1}{2} n\_{in} Var(W)Var(zi​)=Var(x)⟹1≈21​nin​Var(W)

解出所需的权重 WWW 的方差：

Var(W)=2ninVar(W) = \frac{2}{n\_{in}}Var(W)=nin​2​

这是Kaiming初始化针对ReLU激活函数在考虑前向传播（扇入模式）时的基本结果。类似地，考虑反向传播 (backpropagation)（梯度流）的推导会得出 Var(W)=2noutVar(W) = \frac{2}{n\_{out}}Var(W)=nout​2​。

### Kaiming 初始化公式

基于此推导出的方差，我们可以使用正态分布或均匀分布来初始化权重 (weight)。

1. **Kaiming 正态初始化：** 权重从正态分布 N(0,σ2)\mathcal{N}(0, \sigma^2)N(0,σ2) 中采样，其中标准差 σ\sigmaσ 为：

   - 扇入模式：σ=2nin\sigma = \sqrt{\frac{2}{n\_{in}}}σ=nin​2​​
   - 扇出模式：σ=2nout\sigma = \sqrt{\frac{2}{n\_{out}}}σ=nout​2​​
2. **Kaiming 均匀初始化：** 权重从均匀分布 U(−bound,bound)\mathcal{U}(-bound, bound)U(−bound,bound) 中采样，其中边界值根据所需的方差计算得出：

   - U(−b,b)\mathcal{U}(-b, b)U(−b,b) 的方差为 b23\frac{b^2}{3}3b2​。
   - 将 b23=2nmode\frac{b^2}{3} = \frac{2}{n\_{mode}}3b2​=nmode​2​（其中 nmoden\_{mode}nmode​ 为 ninn\_{in}nin​ 或 noutn\_{out}nout​）设定，得到 b2=6nmodeb^2 = \frac{6}{n\_{mode}}b2=nmode​6​。
   - 因此，bound=6nmodebound = \sqrt{\frac{6}{n\_{mode}}}bound=nmode​6​​。
   - 扇入模式：bound=6ninbound = \sqrt{\frac{6}{n\_{in}}}bound=nin​6​​
   - 扇出模式：bound=6noutbound = \sqrt{\frac{6}{n\_{out}}}bound=nout​6​​

“扇入”模式通常更受青睐，因为它在前向传播过程中保持方差。

### PyTorch 中的实现

PyTorch 在 `torch.nn.init` 模块中提供了方便的 Kaiming 初始化函数。

```python
import torch
import torch.nn as nn
import math

fan_in = 2048
fan_out = 8192

linear_layer = nn.Linear(fan_in, fan_out, bias=False)

nn.init.kaiming_normal_(
    linear_layer.weight,
    mode='fan_in',
    nonlinearity='relu'
)

print("Kaiming 正态初始化权重（形状，样本）：")
print(linear_layer.weight.data.shape)
print(linear_layer.weight.data[0, :5])
actual_var_normal = linear_layer.weight.data.var()
expected_var = 2.0 / fan_in
print(f"\n方差（正态）：{actual_var_normal:.6f}")
print(f"预期方差 (2 / fan_in)：{expected_var:.6f}")

linear_layer_uniform = nn.Linear(fan_in, fan_out, bias=False)
nn.init.kaiming_uniform_(
    linear_layer_uniform.weight,
    mode='fan_in',
    nonlinearity='relu'
)

print("\nKaiming 均匀初始化权重（形状，样本）：")
print(linear_layer_uniform.weight.data.shape)
print(linear_layer_uniform.weight.data[0, :5])
actual_var_uniform = linear_layer_uniform.weight.data.var()

print(f"\n方差（均匀）：{actual_var_uniform:.6f}")
print(f"预期方差 (2 / fan_in)：{expected_var:.6f}")

bias_tensor = torch.zeros(fan_out)
print("\n偏置初始化（示例）：")
print(bias_tensor[:5])
```

在代码中，`nonlinearity='relu'` 告诉函数使用与ReLU相关的增益因子，即 2\sqrt{2}2​。这个因子直接来自推导中需要 Var(W)=2/ninVar(W) = 2/n\_{in}Var(W)=2/nin​。如果您使用带有负斜率 `a` 的 Leaky ReLU，您将设置 `nonlinearity='leaky_relu'`，并可能调整 `a` 参数 (parameter)，它会相应地调整增益计算。`mode='fan_in'` 确保方差计算使用输入特征的数量 (ninn\_{in}nin​)。

### Transformer 中的适用性

Kaiming初始化是Transformer中位置前馈网络（FFN）内权重 (weight)矩阵的通用选择，因为这些网络通常使用ReLU或其近似函数，如GeLU或SwiGLU。尽管GeLU和SwiGLU并非严格意义上的ReLU，但Kaiming初始化通常是一个良好的起始点。对于嵌入 (embedding)层和注意力机制 (attention mechanism)中的线性投影，可能会采用不同的策略（通常更接近Xavier正态初始化或简单的缩放标准正态分布），但对于由类ReLU函数激活的核心FFN层，Kaiming初始化对于实现深度堆叠网络的训练非常重要。

通过专门针对ReLU激活函数 (activation function)的特性，Kaiming初始化为主要使用这些单元的深层网络提供了一种方法。它在阻止信号方差迅速衰减方面起到重要作用，从而有助于大型模型（如现代Transformer）的稳定高效训练。

获取即时帮助、个性化解释和交互式代码示例。

---

### Transformer组件中的初始化

# Transformer组件中的初始化

Xavier和Kaiming初始化的基本原理是深度学习 (deep learning)中必不可少的。本文讨论这些技术如何具体应用于标准Transformer模型中不同类型的层。确保每个组件得到恰当的初始化，对于保持信号传播的稳定并使基于梯度的学习在这些深层架构中得以有效进行具有主要作用。

### 初始化嵌入 (embedding)层

Transformer模型以嵌入层开始，这些层将输入令牌ID及其位置转换为密集的向量 (vector)表示。

- **令牌嵌入（`nn.Embedding`）：** 此层将离散令牌ID映射到连续向量。一个普遍的做法是将这些嵌入权重 (weight)从一个均值为0且标准差相对较小的正态分布中初始化，例如N(0,0.02)\mathcal{N}(0, 0.02)N(0,0.02)。这种方法在嵌入中提供了初始多样性，同时避免引入过大的值，这些值可能在训练早期造成不稳定。虽然它不严格遵循Xavier/Kaiming的扇入/扇出逻辑（因为输入是稀疏的），但这种经验方法在实际中运行良好。有些实现可能会调整Xavier初始化，将嵌入维度视为输出大小。
- **位置嵌入：** 如果使用学习到的位置嵌入（也通常是一个`nn.Embedding`层），类似的初始化策略适用，通常从N(0,σ2)\mathcal{N}(0, \sigma^2)N(0,σ2)中抽取权重，其中σ\sigmaσ较小。对于正弦位置编码 (positional encoding)，不需要初始化，因为它们是固定的、确定的值。

以下是在PyTorch中初始化嵌入层的方法：

```python
import torch
import torch.nn as nn

vocab_size = 30000
hidden_dim = 768
max_position_embeddings = 512

token_embedding_layer = nn.Embedding(vocab_size, hidden_dim)
nn.init.normal_(token_embedding_layer.weight, mean=0.0, std=0.02)

positional_embedding_layer = nn.Embedding(
    max_position_embeddings, hidden_dim
)
nn.init.normal_(positional_embedding_layer.weight, mean=0.0, std=0.02)

print("令牌嵌入层权重形状:", token_embedding_layer.weight.shape)
print(
    "位置嵌入层权重形状:",
    positional_embedding_layer.weight.shape
)
```

### 初始化注意力机制 (attention mechanism)层

Transformer的核心在于其自注意力 (self-attention)和交叉注意力机制。这些机制涉及多个线性投影层：

- **查询（Q）、值（V）投影（`nn.Linear`）：** 这些层将输入嵌入 (embedding)投影到Q、K和V空间。由于它们是标准线性变换，且通常后接受益于受控方差的操作（如缩放点积注意力），Kaiming初始化通常是合适的选择，特别是Kaiming均匀或正态初始化。这有助于在初始投影步骤中保持方差。
- **输出投影（`nn.Linear`）：** 在计算注意力输出后（通常来自多个注意力头），另一个线性层将连接后的输出投影回模型的隐藏维度。与Q/K/V投影类似，Kaiming初始化是此层的常见且合理默认方法。

考虑多头注意力 (multi-head attention)块中的一个线性投影层：

```python

hidden_dim = 768
projection_layer = nn.Linear(hidden_dim, hidden_dim * 3)

nn.init.kaiming_uniform_(
    projection_layer.weight,
    a=0,
    mode='fan_in',
    nonlinearity='leaky_relu'
)

if projection_layer.bias is not None:
    nn.init.zeros_(projection_layer.bias)

print("投影层权重形状:", projection_layer.weight.shape)

output_projection = nn.Linear(hidden_dim, hidden_dim)

nn.init.kaiming_normal_(
    output_projection.weight,
    mode='fan_out',
    nonlinearity='relu'
)
if output_projection.bias is not None:
    nn.init.zeros_(output_projection.bias)

print("输出投影权重形状:", output_projection.weight.shape)
```

请注意，`mode`（`fan_in`或`fan_out`）和`nonlinearity`的选择可以根据理论思考或实验结果进行调整。`fan_in`在前向传播中保持方差，而`fan_out`在反向传播 (backpropagation)中保持方差。对于ReLU类激活（包括GeLU、SwiGLU），指定`nonlinearity='relu'`或`nonlinearity='leaky_relu'`可以适当调整缩放因子。

### 初始化逐位置前馈网络（FFN）

每个Transformer块都含有一个FFN，它通常由两个线性层组成，中间带有一个非线性激活函数 (activation function)（例如，ReLU、GeLU、SwiGLU）。

- **第一个线性层（`nn.Linear`）：** 此层通常扩展维度。在此处强烈推荐使用Kaiming初始化，并与所用的非线性函数匹配（例如，对于ReLU/GeLU使用`nonlinearity='relu'`）。这可以确保激活函数后输出的方差保持受控。
- **第二个线性层（`nn.Linear`）：** 此层将维度投影回模型的隐藏大小。虽然可以使用Kaiming初始化，但一些研究和实现（例如遵循GPT-2/3做法的）建议为此层使用较小的方差进行初始化。其原因通常与FFN块后的残差连接有关。通过缩小贡献给残差分支的层的初始化比例，它有助于训练的稳定，特别是在非常深的神经网络 (neural network)中。这种缩放可能与1/N1/\sqrt{N}1/N​成比例，其中NNN是残差块或层的数量。

```python
hidden_dim = 768
ffn_intermediate_dim = hidden_dim * 4
num_layers = 12

ffn_layer1 = nn.Linear(hidden_dim, ffn_intermediate_dim)

nn.init.kaiming_uniform_(
    ffn_layer1.weight,
    a=0,
    mode='fan_in',
    nonlinearity='leaky_relu'
)
if ffn_layer1.bias is not None:
    nn.init.zeros_(ffn_layer1.bias)

ffn_layer2 = nn.Linear(ffn_intermediate_dim, hidden_dim)

residual_scaling_factor = 2 * num_layers
std_dev = 0.02 / (residual_scaling_factor**0.5)
nn.init.normal_(ffn_layer2.weight, mean=0.0, std=std_dev)

if ffn_layer2.bias is not None:
    nn.init.zeros_(ffn_layer2.bias)

print("FFN层1权重形状:", ffn_layer1.weight.shape)
print("FFN层2权重形状:", ffn_layer2.weight.shape)
```

### 初始化层归一化 (normalization)参数 (parameter)

层归一化（`nn.LayerNorm`）具有可学习的仿射参数：增益（γ\gammaγ）和偏置 (bias)（β\betaβ）。标准做法是将γ\gammaγ初始化为1，将β\betaβ初始化为0。这使得层归一化在初始时对归一化输出表现得接近恒等变换，从而允许网络在训练过程中根据需要学习偏差。

```python
layer_norm = nn.LayerNorm(hidden_dim)

print("层归一化Gamma（权重）形状:", layer_norm.weight.shape)
print("层归一化Beta（偏置）形状:", layer_norm.bias.shape)
```

### 初始化最终输出层

仅解码器或编码器-解码器Transformer的最终层通常将隐藏状态投影到词汇大小，通常后接一个softmax以进行概率分布。

- **输出到词汇的投影（`nn.Linear`）：** 此层将最终隐藏维度映射到词汇的大小。与第二个FFN层类似，直接应用标准的Xavier或Kaiming初始化可能会导致初始输出过大，可能在训练早期产生过度自信的预测和不稳定。一个普遍的做法是为此层使用较小的标准差进行初始化，类似于令牌嵌入 (embedding)（例如，N(0,0.02)\mathcal{N}(0, 0.02)N(0,0.02)），或者可能将其权重 (weight)与令牌嵌入矩阵绑定（权重绑定），尽管如果它们没有绑定，初始化仍然适用。

```python
output_projection_vocab = nn.Linear(hidden_dim, vocab_size)

nn.init.normal_(output_projection_vocab.weight, mean=0.0, std=0.02)
if output_projection_vocab.bias is not None:
    nn.init.zeros_(output_projection_vocab.bias)

print("最终输出投影形状:", output_projection_vocab.weight.shape)
```

总而言之，虽然Xavier和Kaiming等普遍原理提供了很好的起点，但有效初始化一个深度Transformer通常需要对每个组件审慎地应用这些方法，有时还会根据架构选择以及从大型模型训练动态中观察到的情况进行经验性调整（例如，对嵌入或特定残差层使用较小的标准差）。细致的初始化为更稳定和高效的训练奠定了基础。

获取即时帮助、个性化解释和交互式代码示例。

---

### 末尾层的小初始化

# 末尾层的小初始化

Kaiming和Xavier等有章法的初始化方法为深度网络中的大多数层提供了很好的起始点，但通常会特别考虑最终层，特别是将最后隐藏状态映射到语言模型中词汇表 (vocabulary)逻辑值的输出投影层。这一层的初始化能对初始损失值和训练过程的稳定性产生很大影响，尤其是在最开始的几步。

### 末尾层不同初始化的理由

输出层直接计算词汇表 (vocabulary)中每个词元 (token)的Softmax前逻辑值。如果这些初始逻辑值有较大的方差，可能会出现一些情况：

1. **初始损失过大：** 较大的正或负逻辑值可能导致Softmax函数后的概率极高或极低，进而可能引发非常大的初始交叉熵损失。这会使得训练刚开始时就发生梯度爆炸。
2. **Softmax饱和：** 如果初始逻辑值非常大，Softmax函数可能会饱和，这意味着许多词汇项的反向传播 (backpropagation)梯度会接近零。这会减缓学习速度。
3. **训练动态：** 一些研究表明，将最终层初始化为使得初始输出分布更接近均匀（或至少具有较低方差）的状态，可以促成更稳定且有时更快的收敛。直观来说，模型在开始时对任何特定词元的“信心”较少，这使得梯度在早期能更有效地流动。

考虑标准的Kaiming或Xavier初始化。这些方法旨在保持网络层中的方差。然而，对于将隐藏维度 dmodeld\_{model}dmodel​ 映射到大型词汇表 VVV 的最终层，产生的尺度可能仍然过大，不利于稳定的初始训练，特别是在考虑到 VVV 的规模时。

### 常见做法

一种常见做法是，将最终线性层的权重 (weight)矩阵初始化为一个较小的标准差，与Kaiming或Xavier初始化通常建议的值相比。偏置 (bias)项通常初始化为零。

例如，不必使用Kaiming均匀/正态初始化中默认的标准差，而是可以手动指定一个较小的标准差，通常与网络深度成反比缩放，或基于经验结果。一些架构或代码库采用特定的较小标准差，例如 0.020.020.02，或将标准Kaiming/Xavier标准差乘以一个系数（例如 0.50.50.5）。

另一种观点来自一些架构，其中最终层是残差连接通路的一部分。在像GPT-2/3这样的模型中，对残差流有贡献的层有时会根据层数来缩放其初始化值，以避免残差信号增长过大。虽然这通常适用于残差块内的中间前馈或注意力输出投影，但如果观察到稳定性问题，类似的原则也可能应用于最终输出投影。核心思路是控制初始输出的大小。

### 实现示例

让我们看看如何在PyTorch模型中对最终线性层应用自定义的较小初始化。假设你的模型有一个名为 `lm_head` 的最终层，它是 `torch.nn.Linear` 的一个实例。

```python
import torch
import torch.nn as nn
import math

class SimpleTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                 dim_feedforward):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                       num_layers)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):

            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')

            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):

            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        if hasattr(self, 'lm_head') and module is self.lm_head:

            std_dev = 0.01

            print(
                f"对lm_head应用特殊初始化，std={std_dev}"
            )
            nn.init.normal_(module.weight, mean=0.0, std=std_dev)
            if module.bias is not None:

                nn.init.zeros_(module.bias)

    def forward(self, src):

        embedded = self.embedding(src) * math.sqrt(self.d_model)

        output = self.transformer_encoder(embedded)
        logits = self.lm_head(output)
        return logits

vocab_size = 10000
d_model = 512
nhead = 8
num_layers = 6
dim_feedforward = 2048

model = SimpleTransformerLM(
    vocab_size, d_model, nhead, num_layers, dim_feedforward
)
```

在这个示例中，`_init_weights` 函数被递归应用于所有模块。它首先对普通线性层应用标准Kaiming正态初始化，并对嵌入 (embedding)层应用正态初始化。然后，它特别检查当前正在初始化的模块是否为 `lm_head` 实例。如果是，它会使用一个小得多的标准差（本例中 `std=0.01`）的正态分布来覆盖标准初始化。

为最终层选择精确的标准差或缩放因子通常涉及一些经验性调整，依据是特定的模型架构、词汇表 (vocabulary)大小和初始训练观察结果。监测初始损失值和梯度范数有助于判断所选初始化是否合适。目标是让训练在一个稳定状态下开始，使得梯度包含有用信息但不过大。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 13 Positional Encoding Variations

### 绝对位置编码的局限性

# 绝对位置编码的局限性

标准Transformer架构依赖于向输入嵌入 (embedding)中加入位置信息，因为自注意力 (self-attention)机制 (attention mechanism)本身是置换不变的。两种最常见的方法是学习得到的绝对位置嵌入和固定正弦位置编码 (positional encoding)。尽管这些方法有效，但它们存在局限，尤其是在处理很长序列时，或者当词元 (token)之间精确的相对位置关系很重要时。理解这些不足之处为研究更先进的位置编码技术提供了理由。

### 对未见过序列长度的外推

当模型遇到比训练期间更长的序列时，会出现一个重要挑战。

- **学习得到的绝对嵌入 (embedding)：** 如果模型用学习得到的位置嵌入进行训练，最大序列长度比如是512个词元 (token)，那么它根本没有为位置513及之后定义嵌入。在推断时将此类模型用于1024个词元的序列，通常会导致不可预测的行为或性能大幅下降，因为模型遇到了没有位置表示的输入。尽管存在诸如扩展学习嵌入的技术，但它们通常是启发式的，可能无法可靠地工作。
- **正弦绝对编码：** 由数学函数（不同频率的正弦和余弦波）定义的固定正弦编码，理论上可以为任何位置生成编码。

  PE(位置,2i)=sin⁡(位置/100002i/dmodel)PE\_{(位置, 2i)} = \sin(位置 / 10000^{2i / d\_{model}})PE(位置,2i)​=sin(位置/100002i/dmodel​)
  PE(位置,2i+1)=cos⁡(位置/100002i/dmodel)PE\_{(位置, 2i+1)} = \cos(位置 / 10000^{2i / d\_{model}})PE(位置,2i+1)​=cos(位置/100002i/dmodel​)

  这里，pospospos 是位置，iii 是 dmodeld\_{model}dmodel​ 维嵌入中的维度索引。因为这是一个确定性函数，我们可以计算任何 pospospos 的 PE(pos,⋅)PE\_{(pos, \cdot)}PE(pos,⋅)​。然而，实际问题依然存在。尽管在数学上已定义，模型本身可能没有学会在其训练分布之外很远距离或绝对位置上有效理解位置信息。正弦模式在非常大的位置上可能会变得不那么清晰或可能出现混叠，使得模型难以准确区分远距离的词元。模型的泛化能力依赖于从它*已经*见过的位置信息中学习模式，将这些模式外推到大得多的尺度上并不能得到保证。

考虑在某个位置范围内两个维度的正弦值。尽管这些值是唯一的，但模型在比训练时更大的尺度上对这些模式的区分能力可能会下降。

0500100015002000−0.500.51维度 4维度 5

> 两个维度在递增位置上的正弦值示例。尽管在数学上是唯一的，泛化能力取决于模型能否在其训练数据的尺度上理解这些可能不明显的差异。

### 相对位置的隐式表示

自注意力 (self-attention)计算词元 (token)间的交互，基于它们的查询 (QQQ)、键 (KKK) 和值 (VVV) 表示。使用绝对位置编码 (positional encoding) (PPP) 时，这些表示通常是通过将位置编码添加到词元嵌入 (embedding) (EEE) 中形成的。对于位置 iii 和 jjj，注意力得分的计算涉及如下项：

得分(i,j)∝(Ei+Pi)TWQTWK(Ej+Pj)\text{得分}(i, j) \propto (E\_i + P\_i)^T W\_Q^T W\_K (E\_j + P\_j)得分(i,j)∝(Ei​+Pi​)TWQT​WK​(Ej​+Pj​)

这里，WQW\_QWQ​ 和 WKW\_KWK​ 是查询和键的权重 (weight)矩阵。注意绝对位置 PiP\_iPi​ 和 PjP\_jPj​ 影响得分，但相对位置 i−ji-ji−j 没有被明确编码。模型必须*学习*理解 PiP\_iPi​ 和 PjP\_jPj​ 的组合，以了解词元 iii 和 jjj 之间的相对距离和方向。尽管模型在一定程度上能够隐式地学习这一点，但这可能不是捕获严重依赖相对位置的关系（例如句法依赖或局部词语交互）最直接或最有效的方式。

对于许多语言现象，两个词之间的关系更多地取决于它们之间的距离，而不是它们在序列中的绝对位置。例如，形容词和名词之间的语法关系通常取决于它们是相邻的或附近的，无论它们出现在长文档的开头还是中间。一种直接包含相对距离的编码，可能使注意力机制 (attention mechanism)更容易捕获这些局部依赖关系。

考虑以下简化的类似PyTorch的伪代码，说明了添加了绝对编码的标准注意力是如何工作的：

```python
import torch
import torch.nn.functional as F

batch_size = 1
seq_len = 5
embed_dim = 8

token_embed = torch.randn(batch_size, seq_len, embed_dim)

pos_enc = torch.randn(batch_size, seq_len, embed_dim)

input_embed = token_embed + pos_enc

query = input_embed
key = input_embed

attn_scores = torch.matmul(query, key.transpose(-2, -1))

scale_factor = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
scaled_attn_scores = attn_scores / scale_factor

attn_weights = F.softmax(scaled_attn_scores, dim=-1)

print("注意力权重形状:", attn_weights.shape)
```

这种隐式处理要求模型分配能力，从绝对位置信号中解开相对位置关系。

### 固定编码的不灵活性

尽管正弦编码提供数学外推能力，但它们固定的性质意味着它们不适应训练数据或下游任务的特定特征。正弦形式对位置关系施加了特定的结构，这可能并非总是最佳的。学习得到的嵌入 (embedding)通过在训练期间进行适应提供更大的灵活性，但如前所述，它们在泛化到更长序列方面表现不佳。固定编码的外推能力与学习嵌入的适应性之间的这种权衡，突显了绝对位置方法的一个核心局限性。

这些关于序列长度外推、隐式相对位置表示以及固定与学习之间权衡的局限性，促进了对替代位置编码 (positional encoding)方法的研究。以下章节将介绍诸如相对位置编码和旋转位置嵌入（RoPE）等技术，这些技术旨在通过将相对位置信息更直接地融入注意力机制 (attention mechanism)或修改位置信息的集成方式来解决这些不足。

获取即时帮助、个性化解释和交互式代码示例。

---

### 相对位置编码的原理

# 相对位置编码的原理

虽然绝对位置编码 (positional encoding)，如正弦或学习得到的嵌入 (embedding)，为Transformer模型提供了必要的序列顺序信息，但它们独立处理每个位置。位置 iii 的编码，表示为 PiP\_iPi​，在生成时没有直接参考任何其他位置 jjj。这种方法存在局限。例如，如果一个模型在长度为512的序列上训练，在推理 (inference)时遇到长度为1024的序列，这些绝对编码的泛化能力如何，并不立即明了。此外，注意力机制 (attention mechanism)天然地计算标记 (token)对之间的关系（查询 iii 注意到键 jjj）。如果注意力计算中包含的位置信息能明确反映位置 iii 和 jjj 之间的*相对*距离或联系，而非仅仅它们的绝对位置，那可能会更自然。

这一观察促成了**相对位置编码**方案的提出。其核心思想是修改注意力机制或其输入，以便相对位置差（通常是 i−ji-ji−j）影响标记 iii 和标记 jjj 之间的注意力得分。相对方案的目标是，不再仅仅将 PiP\_iPi​ 添加到标记 iii 的输入嵌入中，而是将查询和键位置之间的偏移信息直接注入到它们的交互计算里。

考虑标准的缩放点积注意力：

注意力(Q,K,V)=softmax(QKTdk)V\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V注意力(Q,K,V)=softmax(dk​​QKT​)V

这里，查询 iii（QQQ 的第 iii 行）和键 jjj（KKK 的第 jjj 行）之间的交互由点积 qi⋅kjq\_i \cdot k\_jqi​⋅kj​ 捕获。绝对位置编码通常在初始嵌入被投影到 QQQ 和 KKK *之前*添加。相对位置编码旨在根据相对位置 i−ji-ji−j 来修改这种交互。

这种相对信息通常可以通过两种主要方式来纳入：

1. **修改注意力得分：** 可以计算基于相对位置 i−ji-ji−j 的信息，并将其作为偏置 (bias)项直接添加到 qi⋅kjq\_i \cdot k\_jqi​⋅kj​ 的点积中，*在* softmax 操作*之前*。
2. **修改查询/键向量 (vector)：** 查询向量 qiq\_iqi​ 或键向量 kjk\_jkj​（或两者）可以以位置相关的方式进行修改，使得它们的点积隐式地包含相对位置信息。

让我们将差异可视化：

G

clusterₐbs

绝对位置编码

clusterᵣel

相对位置编码

Origin

原点 (0)

Posᵢₐbs

位置 i

Origin->Posᵢₐbs

 Pᵢ

Posⱼₐbs

位置 j

Origin->Posⱼₐbs

 Pⱼ

Posᵢᵣel

位置 i

Posⱼᵣel

位置 j

Posⱼᵣel->Posᵢᵣel

 Rᵢj (取决于 i-j)

> 绝对编码定义了相对于固定原点的位置，而相对编码则侧重于位置对之间的有向关系。

为何要采用这种方法？相对编码具有多项潜在优势：

- **更好的泛化能力：** 由于相对距离通常落在比绝对位置更小的范围内（例如，大多数依赖关系可能在 +/- 50 个标记内），模型可能学到对训练中未见过的序列长度有更好泛化能力的表示。
- **平移不变性：** 理想情况下，位置 iii 注意位置 jjj 的方式应仅取决于偏移量 i−ji-ji−j。对于偏移量 +2 学到的交互模式，无论是在位置 (5, 3) 还是 (105, 103) 之间，都应可重复使用。相对编码比绝对编码能更自然地实现这一点。
- **直接建模关系：** 注意力*就是*关于成对关系的。将相对位置信息直接注入注意力计算中，与此机制非常契合。

考虑修改注意力得分的一种简化方式是，计算每个查询位置 iii 和位置 jjj 之间的相对偏移。然后，此偏移量可用于查找表示该特定相对距离的学习嵌入。

```python
import torch
import torch.nn as nn

seq_len = 10
max_relative_distance = 4

query_pos = torch.arange(seq_len, dtype=torch.long)

key_pos = torch.arange(seq_len, dtype=torch.long)

relative_pos = query_pos.unsqueeze(1) - key_pos.unsqueeze(0)
print(f"L=5时的原始相对位置 (i-j):\n{relative_pos[:5, :5]}")

clipped_relative_pos = torch.clamp(
    relative_pos,
    -max_relative_distance,
    max_relative_distance
)
relative_pos_indices = clipped_relative_pos + max_relative_distance
print(f"\nL=5时裁剪后的相对索引:\n{relative_pos_indices[:5, :5]}")

num_relative_embeddings = 2 * max_relative_distance + 1

num_heads = 8
relative_embedding = nn.Embedding(num_relative_embeddings, num_heads)

relative_position_bias = relative_embedding(relative_pos_indices)

bias_for_scores = relative_position_bias.permute(2, 0, 1)

print(f"\n查找索引的形状: {relative_pos_indices.shape}")
print(f"相对嵌入表输出的形状: {relative_position_bias.shape}")
print(f"用于注意力得分的置换偏置的形状: {bias_for_scores.shape}")
```

此代码段说明了核心流程：计算相对索引，裁剪它们以管理嵌入表大小，查找对应的嵌入，并准备将其添加到注意力得分中。

当然，这是一种简化视角。高效且实际的实现需要仔细处理嵌入查找、层间嵌入的潜在共享，以及将其自然地整合到Transformer块中。接下来的部分将审视具体、成熟的方法，如Shaw等人的方法、Transformer-XL的相对编码和旋转位置嵌入（RoPE），这些方法在实践中应用了这些思想。

获取即时帮助、个性化解释和交互式代码示例。

---

### Shaw 等人的相对位置实现

# Shaw 等人的相对位置实现

虽然绝对位置编码 (positional encoding)，例如正弦编码或学习型嵌入 (embedding)，为 Transformer 提供了必要的序列顺序信息，但它们独立处理每个位置。这可能会限制它们对训练中未见过的序列长度的泛化能力，并且没有明确地根据标记 (token)之间的距离来表示它们的关系。Shaw 等人（2018）在“带相对位置表示的自注意力 (self-attention)”中提出的方法，通过将相对距离直接融入注意力机制 (attention mechanism)本身，提供了一种替代方案。

核心思想是修改注意力分数计算和潜在的价值向量 (vector)聚合方式，使其对交互标记之间的相对偏移量敏感。这种方法不是仅将位置信息添加到初始嵌入中，而是引入了可学习的嵌入，这些嵌入表示不同的相对距离。

### 用相对嵌入 (embedding)修改注意力分数

回顾位置 iii 的查询向量 (vector) (qi=xiWQq\_i = x\_i W^Qqi​=xi​WQ) 和位置 jjj 的键向量 (kj=xjWKk\_j = x\_j W^Kkj​=xj​WK) 之间的标准缩放点积注意力分数：

分数(qi,kj)=qikjTdk\text{分数}(q\_i, k\_j) = \frac{q\_i k\_j^T}{\sqrt{d\_k}}分数(qi​,kj​)=dk​​qi​kjT​​

Shaw 等人引入了可学习的键相对位置嵌入，表示为 aKa^KaK。这些嵌入捕捉了从位置 iii 到位置 jjj 的关系。具体来说，aijKa^K\_{ij}aijK​ 表示相对距离 j−ij-ij−i 的嵌入。然后修改注意力分数计算，使其包含一个纳入了此相对位置信息的项：

eij=(xiWQ)(xjWK)T+(xiWQ)(aijK)Tdke\_{ij} = \frac{(x\_i W^Q)(x\_j W^K)^T + (x\_i W^Q)(a^K\_{ij})^T}{\sqrt{d\_k}}eij​=dk​​(xi​WQ)(xj​WK)T+(xi​WQ)(aijK​)T​

我们来分解这个修改后的分数 eije\_{ij}eij​：

1. **基于内容的项：** (xiWQ)(xjWK)T(x\_i W^Q)(x\_j W^K)^T(xi​WQ)(xj​WK)T 是位置 iii 的查询和位置 jjj 的键之间的标准点积。它根据标记 (token)表示本身衡量兼容性。
2. **基于位置的项：** (xiWQ)(aijK)T(x\_i W^Q)(a^K\_{ij})^T(xi​WQ)(aijK​)T 引入了基于相对位置的偏差。查询 xiWQx\_i W^Qxi​WQ 与对应于偏移量 j−ij-ij−i 的学习型嵌入 aijKa^K\_{ij}aijK​ 发生作用。此项使得模型能够纯粹根据位置 iii 和位置 jjj 的相对距离，而不考虑位置 jjj 的实际内容，来学习位置 iii 应该对位置 jjj 给予多少注意力。

最终的注意力权重 (weight) αij\alpha\_{ij}αij​ 是通过对这些修改后的分数 eije\_{ij}eij​ 应用 softmax 函数获得的。

ShawRelativeAttention

Xi

查询 (xᵢ)

ContentScore

内容分数
(xᵢ W^Q)(xⱼ W^K)ᵀ

Xi->ContentScore

W^Q

RelativeScore

相对分数
(xᵢ W^Q)(a^Kᵢⱼ)ᵀ

Xi->RelativeScore

W^Q

Xj

(xⱼ)

Xj->ContentScore

W^K

Posᵢj

相对位置 (j-i)

Emb\_K

相对嵌入 (a\_K)

Posᵢj->Emb\_K

查找

Emb\_K->RelativeScore

Eᵢj

注意力分数
(eᵢj)

ContentScore->Eᵢj

+

RelativeScore->Eᵢj

+

> Shaw 等人方法中的注意力分数计算，同时包含了内容相似性以及源自相对位置嵌入的偏差。

### 裁剪相对距离

在非常长的序列中，为每个可能的相对距离计算和存储独特的嵌入 (embedding)将是低效且可能不必要的。相距很远的标记 (token)之间的关系可能信息量较少或遵循一般模式。因此，裁剪考虑的最大相对距离是常见做法。

选择一个最大距离 kkk。任何 ∣j−i∣>k|j-i| > k∣j−i∣>k 的相对距离 j−ij-ij−i 都将被裁剪到 −k-k−k 或 kkk。例如，如果 k=8k=8k=8，相对距离 j−i=10j-i = 10j−i=10 将被视为 888，而 j−i=−12j-i = -12j−i=−12 将被视为 −8-8−8。这意味着模型只需要学习范围 [−k,k][-k, k][−k,k] 内的相对距离嵌入，从而得到 2k+12k+12k+1 个独特的相对位置嵌入。

### 实现考量

在实际的 PyTorch 实现中，你通常会：

1. **定义相对位置嵌入 (embedding)层：** 创建一个 `nn.Embedding` 层来存储可学习的嵌入 aKa^KaK。其大小将是 `(2 * max_relative_position + 1, head_dim)`。

   ```python
   import torch
   import torch.nn as nn

   max_relative_position = 8
   head_dim = 64

   num_relative_embeddings = 2 * max_relative_position + 1

   relative_key_embeddings = nn.Embedding(num_relative_embeddings, head_dim)
   ```
2. **计算相对位置：** 对于给定的序列长度 `seq_len`，计算所有查询 (iii) 和键 (jjj) 位置之间的相对位置矩阵。

   ```python
   seq_len = 512
   range_vec = torch.arange(seq_len)
   relative_pos_matrix = range_vec[None, :] - range_vec[:, None]
   ```
3. **裁剪并将位置映射到索引：** 裁剪相对位置并将它们转换为适合嵌入查找的非负索引。

   ```python
   clipped_relative_pos = torch.clamp(relative_pos_matrix,
                                      -max_relative_position,
                                      max_relative_position)

   embedding_indices = clipped_relative_pos + max_relative_position
   ```
4. **查找嵌入：** 检索相应的 aijKa^K\_{ij}aijK​ 嵌入。

   ```python

   rel_key_embeds = relative_key_embeddings(embedding_indices)
   ```
5. **计算相对注意力项：** 计算 (xiWQ)(aijK)T(x\_i W^Q)(a^K\_{ij})^T(xi​WQ)(aijK​)T 项。这需要仔细的张量操作，以便对每个查询 qiq\_iqi​ 和所有 jjj 对应的相对键嵌入 aijKa^K\_{ij}aijK​ 执行点积。

   ```python

   relative_logits = torch.zeros(q.shape[0], q.shape[1], seq_len, seq_len, device=q.device)
   ```

   *注意：高效计算相对注意力项的具体实现可能很复杂，通常涉及库实现或原始论文中详细说明的特定重塑和矩阵乘法策略。目标是计算每个查询 qiq\_iqi​ 与所有相关 aijKa^K\_{ij}aijK​ 向量 (vector)之间的点积。*
6. **组合分数：** 在缩放并应用 softmax 之前，将 `relative_logits` 添加到 `content_logits`。

   ```python

   combined_logits = content_logits + relative_logits
   attention_scores = combined_logits / (head_dim ** 0.5)
   attention_weights = F.softmax(attention_scores, dim=-1)
   ```

### 修改价值聚合（不常见）

Shaw 等人还提出，在聚合价值向量 (vector) vj=xjWVv\_j = x\_j W^Vvj​=xj​WV 时，添加一个类似的相对位置嵌入 (embedding)项 aijVa^V\_{ij}aijV​。位置 iii 的输出 ziz\_izi​ 将变为：

zi=∑jαij(xjWV+aijV)z\_i = \sum\_j \alpha\_{ij} (x\_j W^V + a^V\_{ij})zi​=j∑​αij​(xj​WV+aijV​)

与重要的嵌入修改相比，这项修改提及较少，但它遵循相同的原则：使输出对被注意力关注标记 (token)的相对位置敏感。它需要一个单独的嵌入层 `relative_value_embeddings`。

### 优点和缺点

**优点：**

- **更好的泛化能力：** 通过关注相对距离，与可能无法很好外推的绝对位置编码 (positional encoding)相比，模型可能对训练中未遇到的序列长度表现出更好的泛化能力。
- **明确的相对信息：** 将成对的位置关系直接编码到注意力机制 (attention mechanism)中。

**缺点：**

- **增加复杂性：** 修改了核心注意力计算，增加了计算步骤，并可能增加延迟。
- **更多参数 (parameter)：** 为相对位置嵌入 (embedding)引入了新的可学习参数 (aKa^KaK 和潜在的 aVa^VaV)。
- **实现细节：** 高效计算相对注意力项需要仔细的张量操作。

此方法是在 Transformer 注意力机制中直接考虑相对位置的一个重要进展。虽然有效，但它是多种方法之一，后续章节将讨论其他方案，例如 Transformer-XL 中使用的相对编码方案和旋转位置嵌入 (RoPE)，它们通过不同的机制实现类似的目标。

获取即时帮助、个性化解释和交互式代码示例。

---

### Transformer-XL 相对位置编码

# Transformer-XL 相对位置编码

绝对位置编码 (positional encoding)，无论是正弦型还是学习型，虽然能为Transformer提供必要的序列顺序信息，但它们存在局限。它们通常预设一个最大序列长度，并且其处理训练数据中未见的更长序列的能力可能受限。此外，它们没有明确表示 token 之间的*相对*距离，而这对注意力机制 (attention mechanism)来说可能是更合适的工作方式。

戴等人 (2019) 提出的 Transformer-XL 架构，提出了一种新颖的相对位置编码方案，旨在专门解决这些问题，尤其是在使用段级别循环机制处理更长序列时（尽管编码方法本身独立来看也很有价值）。Transformer-XL 没有将位置信息加到词嵌入 (embedding)中，而是将相对位置信息直接注入到注意力分数计算中。

### Transformer-XL 相对编码的核心思想

核心思想是修改位置 iii 的查询 (query) 与位置 jjj 的键 (key) 之间注意力分数的计算方式。在标准 Transformer 中，分数取决于查询 qiq\_iqi​ 和键 kjk\_jkj​，两者可能都包含添加到各自嵌入 (embedding)中的绝对位置信息。

Transformer-XL 重新制定了注意力分数计算，使其明确依赖于相对距离 (i−j)(i-j)(i−j)。它通过进行两项主要修改来做到这一点：

1. **键的相对位置嵌入：** 它不使用键向量 (vector)的绝对位置嵌入 pjp\_jpj​，而是使用表示查询和键位置之间偏移的相对位置嵌入 Ri−jR\_{i-j}Ri−j​。这些嵌入 Ri−jR\_{i-j}Ri−j​ 通常是固定的正弦编码，与原始 Transformer 类似，但它们编码的是相对距离而非绝对位置。重要的是，当考虑距离 kkk 个位置的键（即 j=i−kj = i-kj=i−k）时，所有查询位置 iii 都使用相同的相对嵌入 RkR\_kRk​。这使得模型能够处理未见的相对距离。
2. **查询交互的分解：** 查询向量 qiq\_iqi​ 与内容和位置属性的交互方式不同。标准点积 qiTkjq\_i^T k\_jqiT​kj​ 被分解为多个项，这些项使用专门的可训练参数 (parameter)将内容-内容交互、内容-位置交互和位置-位置交互分开。

### 数学公式

设 qi=WQxiq\_i = W\_Q x\_iqi​=WQ​xi​ 是位置 iii 处 token xix\_ixi​ 的查询向量 (vector)， kj=WKxjk\_j = W\_K x\_jkj​=WK​xj​ 是位置 jjj 处 token xjx\_jxj​ 的基于内容的键向量。设 Ri−jR\_{i-j}Ri−j​ 是相对位置 i−ji-ji−j 的正弦嵌入 (embedding)向量。在标准 Transformer 中，softmax 内部的核心项大约是 qiT(kj+pj)q\_i^T (k\_j + p\_j)qiT​(kj​+pj​)。

Transformer-XL 用更精密的计算来替代，得到注意力分数 Ai,jA\_{i,j}Ai,j​：

Ai,jrel=qiTWKxj⏟(a) 基于内容的+qiTWRRi−j⏟(b) 内容-位置+uTWKxj⏟(c) 全局内容偏置+vTWRRi−j⏟(d) 全局位置偏置A\_{i,j}^{rel} = \underbrace{q\_i^T W\_K x\_j}\_{\text{(a) 基于内容的}} + \underbrace{q\_i^T W\_R R\_{i-j}}\_{\text{(b) 内容-位置}} + \underbrace{u^T W\_K x\_j}\_{\text{(c) 全局内容偏置}} + \underbrace{v^T W\_R R\_{i-j}}\_{\text{(d) 全局位置偏置}}Ai,jrel​=(a) 基于内容的qiT​WK​xj​​​+(b) 内容-位置qiT​WR​Ri−j​​​+(c) 全局内容偏置uTWK​xj​​​+(d) 全局位置偏置vTWR​Ri−j​​​

此处：

- WQ,WKW\_Q, W\_KWQ​,WK​ 是查询和基于内容的键的常用权重 (weight)矩阵。
- WRW\_RWR​ 是一个应用于相对位置嵌入 Ri−jR\_{i-j}Ri−j​ 的*新*可训练权重矩阵。
- uuu 和 vvv 是*新*可训练参数 (parameter)向量。
- Ri−jR\_{i-j}Ri−j​ 是一个固定的（通常是正弦型）编码，表示相对距离 i−ji-ji−j。它不依赖于特定的内容 xix\_ixi​ 或 xjx\_jxj​。

我们来逐项分析：

- **(a) 基于内容的寻址：** 这是衡量查询内容 qiq\_iqi​ 与内容 kj=WKxjk\_j = W\_K x\_jkj​=WK​xj​ 之间相似度的标准项。
- **(b) 内容-位置寻址：** 此项描述了查询内容 qiq\_iqi​ 如何与键的*相对位置* i−ji-ji−j 相关联。查询与转换后的相对位置嵌入 WRRi−jW\_R R\_{i-j}WR​Ri−j​ 进行交互。
- **(c) 全局内容偏置 (bias)：** 此项仅基于键的内容 xjx\_jxj​ 添加偏置。可训练向量 uuu 学习对某些类型的键内容的全局偏好，与查询 qiq\_iqi​ 无关。
- **(d) 全局位置偏置：** 此项仅基于*相对位置* i−ji-ji−j 添加偏置。可训练向量 vvv 学习对某些相对距离的全局偏好，与查询 qiq\_iqi​ 无关。

最终的注意力权重通过对这些分数 Ai,jrelA\_{i,j}^{rel}Ai,jrel​ 应用 softmax 获得（通常在乘以 1dk\frac{1}{\sqrt{d\_k}}dk​​1​ 缩放后）。

### 实现考量

生成相对位置编码 (positional encoding) Ri−jR\_{i-j}Ri−j​ 通常涉及为一个最大预期相对距离（例如，从 −L-L−L 到 +L+L+L，其中 LLL 是上下文 (context)长度或段长度）创建标准正弦编码。在计算位置 iii 处查询的注意力时，您会查找每个位置 j=i−kj=i-kj=i−k 处键的相应编码 RkR\_{k}Rk​。

引入可训练向量 (vector) uuu 和 vvv 以及单独的投影矩阵 WRW\_RWR​ 增加了参数 (parameter)，与标准 Transformer 注意力相比，但能够对相对位置重要性进行更精细的模型构建。

计算相对注意力分数的简化 PyTorch 风格纲要可能如下所示（侧重于分数计算，省略多头及其他细节）：

```python
import torch
import torch.nn as nn
import math

class RelativeSinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe_full = torch.cat([
            pe.flip(0)[:-1, :],
            pe
        ], dim=0)
        self.register_buffer('pe', pe_full)
        self.max_len = max_len

    def forward(self, seq_len_q, seq_len_k):

        relative_indices = torch.arange(
            seq_len_k - 1, -seq_len_q, -1, dtype=torch.long
        )

        buffer_indices = relative_indices + self.max_len - 1
        relative_encodings = self.pe[buffer_indices]

        start_idx = self.max_len - seq_len_k
        end_idx = start_idx + seq_len_q + seq_len_k - 1

        rel_enc = self.pe[start_idx:end_idx]

        pass

class TransformerXLRelativeAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_r = nn.Linear(d_model, d_model)

        self.u = nn.Parameter(torch.Tensor(self.nhead, self.d_head))
        self.v = nn.Parameter(torch.Tensor(self.nhead, self.d_head))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

        self.dropout = nn.Dropout(0.1)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, query_embed, key_embed, value_embed, R_ij,
                mask=None):

        batch_size, seq_len_q, _ = query_embed.size()
        seq_len_k = key_embed.size(1)

        Q = self.W_q(query_embed).view(
            batch_size, seq_len_q, self.nhead, self.d_head
        )
        K = self.W_k(key_embed).view(
            batch_size, seq_len_k, self.nhead, self.d_head
        )
        V = self.W_v(value_embed).view(
            batch_size, seq_len_k, self.nhead, self.d_head
        )

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        AC = torch.matmul(Q + self.u.unsqueeze(0).unsqueeze(2),
                          K.transpose(-2, -1))

        scores = AC * self.scale

        if mask is not None:

             scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )

        return output, attn_weights
```

*注意：* 项 (b) 和 (d) 的实际实现需要仔细的张量操作（通常涉及偏斜矩阵），以便高效地执行相对位置计算，而无需在每一步为每对 (i,j)(i,j)(i,j) 明确构建完整的 Ri−jR\_{i-j}Ri−j​ 矩阵。上面的代码简化了这一部分。

### Transformer-XL 相对编码的优点

1. **对更长序列的处理能力：** 由于使用相对位置，模型不与训练期间学习到的特定绝对位置绑定。它能够处理比训练中见过的更长的序列，因为相对偏移量仍然有意义。
2. **明确的相对信息：** 它直接编码 token 之间的距离，这比单独的绝对位置能为注意力提供更多信息。
3. **支持段级别循环：** 这项编码方案是 Transformer-XL 主要成就的构成部分：逐段处理序列，同时重用先前段的隐藏状态。相对编码确保了跨段边界的位置一致性。

与 Shaw 等人的方法（在主要查询-键点积*之后*添加相对位置偏置 (bias)）相比，Transformer-XL 通过让查询直接与相对位置嵌入 (embedding) (qiTWRRi−jq\_i^T W\_R R\_{i-j}qiT​WR​Ri−j​) 交互并整合全局位置偏置 (vTWRRi−jv^T W\_R R\_{i-j}vTWR​Ri−j​)，将相对位置信息更紧密地融入到分数计算中。

总而言之，Transformer-XL 相对位置编码 (positional encoding)提供了替代绝对编码的方案。通过侧重于相对距离并分解注意力分数计算，它为序列长度提供了更佳的通用能力，并为旨在处理极长上下文 (context)的架构提供了支持。其实现需要对相对位置嵌入和额外的可训练参数 (parameter)进行仔细处理，但对长序列建模的益处是显著的。

获取即时帮助、个性化解释和交互式代码示例。

---

### 旋转位置编码 (RoPE)

# 旋转位置编码 (RoPE)

旋转位置编码 (positional encoding)（RoPE）提供了一种独特方式，将位置信息融入到 Transformer 架构中。不同于通过添加位置向量 (vector)的绝对位置编码，或通常直接修改注意力分数计算的相对位置编码（如 Shaw 等人的方法或 Transformer-XL），RoPE 在注意力分数计算 *之前*，对查询 (QQQ) 和键 (KKK) 向量应用依赖于位置的旋转。这种方式通过旋转变换巧妙地表示了相对位置信息。

其核心思想源于一个发现：两个分别旋转了 α\alphaα 和 β\betaβ 角度的向量之间的点积，取决于它们的原始点积以及角度差 (α−β\alpha - \betaα−β)。RoPE 运用此特性，设计出依赖于令牌绝对位置的旋转矩阵。

### 数学表达

设位于位置 mmm 的查询向量 (vector)为 qmq\_mqm​，位于位置 nnn 的键向量为 knk\_nkn​。RoPE 旨在变换这些向量，使得它们的内积 qm′⋅kn′q'\_m \cdot k'\_nqm′​⋅kn′​ 主要依赖于原始向量 qm,knq\_m, k\_nqm​,kn​ 和它们的相对位置 m−nm-nm−n。

实现此目的的方式是将嵌入 (embedding)维度 ddd 视为若干维度对，并对每对应用二维旋转。对于向量 xxx 和位置 mmm，变换 f(x,m)f(x, m)f(x,m) 会施加一个旋转。设查询和键向量的维度为 ddd。我们可以将向量分成 d/2d/2d/2 个大小为 2 的块。对于第 iii 个块（对应维度 2i−12i-12i−1 和 2i2i2i），旋转矩阵 Rm,iR\_{m, i}Rm,i​ 定义为：

Rm,i=(cos⁡(mθi)−sin⁡(mθi)sin⁡(mθi)cos⁡(mθi))R\_{m, i} = \begin{pmatrix} \cos(m \theta\_i) & -\sin(m \theta\_i) \\ \sin(m \theta\_i) & \cos(m \theta\_i) \end{pmatrix}Rm,i​=(cos(mθi​)sin(mθi​)​−sin(mθi​)cos(mθi​)​)

这里，θi\theta\_iθi​ 是一个频率项，它依赖于块索引 iii。一个常见选择是 θi=base−2i/d\theta\_i = \text{base}^{-2i/d}θi​=base−2i/d，其中 base\text{base}base 是一个较大的数字（例如 10000），确保频率在不同维度上有所变化。这类似于正弦绝对位置编码 (positional encoding)中的频率选择。

RoPE 变换随后按块应用于查询 qmq\_mqm​ 和键 knk\_nkn​：

qm′=f(qm,m)=Rmqmkn′=f(kn,n)=Rnknq'\_m = f(q\_m, m) = R\_m q\_m \\
k'\_n = f(k\_n, n) = R\_n k\_nqm′​=f(qm​,m)=Rm​qm​kn′​=f(kn​,n)=Rn​kn​

其中 RmR\_mRm​ 和 RnR\_nRn​ 分别表示由 Rm,iR\_{m,i}Rm,i​ 和 Rn,iR\_{n,i}Rn,i​ 块形成的块对角矩阵。

值得关注的属性是，旋转后的查询向量和键向量之间的内积本身就捕捉了相对位置信息：

(qm′)Tkn′=(Rmqm)T(Rnkn)=qmTRmTRnkn(q'\_m)^T k'\_n = (R\_m q\_m)^T (R\_n k\_n) = q\_m^T R\_m^T R\_n k\_n(qm′​)Tkn′​=(Rm​qm​)T(Rn​kn​)=qmT​RmT​Rn​kn​

由于 RmR\_mRm​ 是一个旋转矩阵，其转置是其逆，RmT=Rm−1=R−mR\_m^T = R\_m^{-1} = R\_{-m}RmT​=Rm−1​=R−m​。因此，RmTRn=R−mRn=Rn−mR\_m^T R\_n = R\_{-m} R\_n = R\_{n-m}RmT​Rn​=R−m​Rn​=Rn−m​。内积变为：

(qm′)Tkn′=qmTRn−mkn(q'\_m)^T k'\_n = q\_m^T R\_{n-m} k\_n(qm′​)Tkn′​=qmT​Rn−m​kn​

这种最终形式表明，位于位置 mmm 的查询与位于位置 nnn 的键之间的作用，明确依赖于它们的相对位置 n−mn-mn−m（或等效地，m−nm-nm−n，因为 Rn−mR\_{n-m}Rn−m​ 包含了此差异）以及原始查询和键向量。

另一种方法是，使用复数提供了一种简洁的视角。将每个二维块 [x2i−1,x2i][x\_{2i-1}, x\_{2i}][x2i−1​,x2i​] 表示为复数 xi=x2i−1+jx2ix\_i = x\_{2i-1} + j x\_{2i}xi​=x2i−1​+jx2i​，则旋转 mθim\theta\_imθi​ 等价于乘以 ejmθie^{j m \theta\_i}ejmθi​。旋转后的查询和键分量为 qm,i′=qm,iejmθiq'\_{m,i} = q\_{m,i} e^{j m \theta\_i}qm,i′​=qm,i​ejmθi​ 和 kn,i′=kn,iejnθik'\_{n,i} = k\_{n,i} e^{j n \theta\_i}kn,i′​=kn,i​ejnθi​。它们对注意力分数的贡献涉及其乘积的实部（考虑到复数点积中有一个被共轭）：

实部(qm,i′kn,i′‾)=实部((qm,iejmθi)(kn,iejnθi‾))=实部(qm,ikn,i‾ejmθie−jnθi)=实部(qm,ikn,i‾ej(m−n)θi)\text{实部}(q'\_{m,i} \overline{k'\_{n,i}}) = \text{实部}((q\_{m,i} e^{j m \theta\_i}) (\overline{k\_{n,i} e^{j n \theta\_i}})) \\
= \text{实部}(q\_{m,i} \overline{k\_{n,i}} e^{j m \theta\_i} e^{-j n \theta\_i}) \\
= \text{实部}(q\_{m,i} \overline{k\_{n,i}} e^{j (m-n) \theta\_i})实部(qm,i′​kn,i′​​)=实部((qm,i​ejmθi​)(kn,i​ejnθi​​))=实部(qm,i​kn,i​​ejmθi​e−jnθi​)=实部(qm,i​kn,i​​ej(m−n)θi​)

对所有块 iii 求和，再次显示了对相对位置 m−nm-nm−n 的依赖。

### 实现

实际应用中，RoPE 应用于多头注意力 (multi-head attention)机制 (attention mechanism)内的查询和键投影，*在*计算注意力分数之前。这通常涉及预先计算所有所需位置和维度的余弦和正弦值。

让我们看一个 PyTorch 实现代码片段。假设 `q` 和 `k` 是形状为 `(batch_size, seq_len, num_heads, head_dim)` 的张量。我们还需要预先计算好的 `cos_cached` 和 `sin_cached` 张量，通常形状为 `(max_seq_len, head_dim // 2)`。

```python
import torch

def rotate_half(x):
    """将输入张量的隐藏维度旋转一半。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos_cached, sin_cached):
    """
    将旋转位置编码应用于查询和键张量。

    参数：
        q (torch.Tensor): 查询张量，形状 (bs, seq_len, num_heads, head_dim)
        k (torch.Tensor): 键张量，形状 (bs, seq_len, num_heads, head_dim)
        cos_cached (torch.Tensor): 预先计算的余弦值，
            形状 (seq_len, head_dim // 2)
        sin_cached (torch.Tensor): 预先计算的正弦值，
            形状 (seq_len, head_dim // 2)

    返回：
        Tuple[torch.Tensor, torch.Tensor]: 旋转后的查询和键张量。
    """

    cos = cos_cached[:q.shape[1], ...].unsqueeze(1)

    sin = sin_cached[:q.shape[1], ...].unsqueeze(1)

    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)

    q_reshaped = q.float().reshape(*q.shape[:-1], -1, 2)
    k_reshaped = k.float().reshape(*k.shape[:-1], -1, 2)

    cos = cos[..., :q.shape[-1] // 2].unsqueeze(-1)

    sin = sin[..., :q.shape[-1] // 2].unsqueeze(-1)

    q_out1 = q_reshaped[..., 0:1] * cos - q_reshaped[..., 1:2] * sin
    q_out2 = q_reshaped[..., 1:2] * cos + q_reshaped[..., 0:1] * sin
    q_rot = torch.cat((q_out1, q_out2), dim=-1).flatten(start_dim=-2)

    k_out1 = k_reshaped[..., 0:1] * cos - k_reshaped[..., 1:2] * sin
    k_out2 = k_reshaped[..., 1:2] * cos + k_reshaped[..., 0:1] * sin
    k_rot = torch.cat((k_out1, k_out2), dim=-1).flatten(start_dim=-2)

    return q_rot.type_as(q), k_rot.type_as(k)
```

`apply_rotary_pos_emb` 函数接收查询、键和预先计算的余弦/正弦值（源自位置索引和频率）。它重塑最后一个维度以处理维度对，应用旋转逻辑，并返回修改后的查询和键张量。这些旋转后的张量随后用于标准的缩放点积注意力计算。

### 优点与考量

RoPE 在现代大型语言模型中得到广泛应用，缘于其多项优点：

1. **有效的相对位置编码 (positional encoding)**：如数学推导所示，它直接将相对位置信息编码到查询-键的作用中。
2. **良好的长度外推能力**：旋转的正弦特性使得 RoPE 有助于更好地泛化到训练中未见的序列长度，相比之下，绝对位置编码可能难以处理超出范围的位置。
3. **无需学习参数 (parameter)**：不同于可学习的绝对嵌入 (embedding)或某些相对偏置 (bias)方案，RoPE 本身不引入与位置相关的额外可学习参数，有助于简化训练过程。
4. **实现效率**：尽管需要特定的张量操作，它避免直接修改核心注意力分数计算，能整洁地融入标准的 Transformer 块结构。

与其他方法比较：

- **对比绝对嵌入**：RoPE 避免添加向量 (vector)，而是通过旋转乘法式地修改查询和键。它本身侧重于相对位置。
- **对比相对位置偏置（例如 T5、Shaw 等）**：这些方法通常根据相对距离直接向注意力分数添加偏置。RoPE 修改注意力分数计算的输入（Q,KQ, KQ,K）。效果可能相似，但机制不同。RoPE 由于其连续旋转特性，可能提供更好的外推能力。
- **对比 Transformer-XL**：Transformer-XL 采用了一种更复杂的相对编码方案，直接集成到注意力分数计算中。RoPE 可以被视为一种可能更简单的替代方案，用于实现相对位置感知。

频率计算中 `base` 超参数 (hyperparameter)的选择 (θi=base−2i/d\theta\_i = \text{base}^{-2i/d}θi​=base−2i/d) 可能影响性能和外推能力，需要仔细调整。尽管它在数学上精妙且在 Llama 和 PaLM 等模型中取得了实际成效，但理解它与其他模型组件的关联以及在超长序列上的表现，仍是一个活跃的研究方向。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 14 Advanced Architectural Modifications

### 参数高效微调的需求

# 参数高效微调的需求

大型语言模型（LLMs），特别是基于Transformer架构的模型，在预训练 (pre-training)阶段需要大量的计算资源和时间投入。一旦在海量文本数据上完成预训练，这些模型便蕴含了通用语言知识和推理 (inference)能力。一种常见且有效的方式，可以将这些预训练知识运用到特定应用中，便是通过*微调 (fine-tuning)*。

标准微调是指将预训练模型进一步在一个较小、面向特定任务的数据集（例如情感分析、问答、摘要）上进行训练。在此过程中，模型的*所有*参数 (parameter)或大部分参数通过梯度下降 (gradient descent)进行更新，以优化其在目标任务上的性能。尽管这种方法有效，但随着模型规模的持续增大，它也带来了一些重大挑战：

### 完全微调 (fine-tuning)的负担

1. **计算成本：** 微调通常比预训练 (pre-training)需要更少的数据和迭代次数，但它仍然涉及通过整个网络进行梯度反向传播 (backpropagation)，而这个网络通常包含数十亿乃至数万亿参数 (parameter)。对于GPT-3（1750亿参数）或PaLM（5400亿参数）这样的模型，更新所有权重 (weight)需要大量的GPU资源（例如，多张高端GPU，如A100或H100）和相当长的训练时间，每个任务通常需要数小时甚至数天。如果一个组织需要针对数十甚至数百个不同任务调整基础LLM，那么累计计算开销将变得非常庞大。
2. **存储和内存开销：** 每个经过完全微调的模型本质上都是原始大型模型的一个独立副本，只是权重略有调整。为每个下游任务存储一个单独的数十亿参数模型会导致庞大的存储需求。例如，一个以FP16精度存储的1750亿参数模型大约需要350GB的磁盘空间。并发部署多个此类模型以服务不同的应用程序会使问题更加严重，需要海量昂贵的高带宽GPU内存（HBM）。

   考虑一个场景，你需要专门的模型用于客户支持工单分类、内部文档摘要和代码生成辅助，而这些模型都源自相同的基础LLM。完全微调将产生三个独立的、大型模型检查点。
3. **部署复杂性：** 管理众多大型独立模型的部署生命周期（更新、监控、服务）在操作上很复杂。为每个特定任务的模型推出更新或扩展推理 (inference)端点需要大量的\_基础设施管理，并可能导致资源碎片化。
4. **灾难性遗忘：** 当在特定任务上进行微调时更新所有参数，模型有丢失部分预训练阶段获得的通用知识的风险。这种现象被称为灾难性遗忘，它会降低模型在刚被微调的任务之外的其他任务上的性能。虽然正则化 (regularization)或仔细选择学习率等技术可以在一定程度上减轻这种影响，但这仍然是一个问题，尤其是在多个任务上进行序列微调时。

### 参数 (parameter)高效的理由

这些挑战共同促使人们寻找更高效的适应方法。理想情况下，我们希望技术能够使预训练 (pre-training)的LLM针对下游任务专业化，同时只修改其一小部分参数。这种方法通常被称为*参数高效微调 (fine-tuning)（PEFT）*。

PEFT的核心思想是冻结预训练模型的大部分权重 (weight)，并引入少量新的、可训练参数，或只修改现有参数的极小部分。如果成功，这将带来显著优势：

- **计算量减少：** 训练涉及更新的参数少得多，大幅减少了适应所需的计算需求（GPU时间、能耗）。
- **存储需求极小：** 无需为每个任务存储完整的模型副本，只需保存一小组修改或新增的参数。这通常将每个任务的存储开销从数百千兆字节减少到兆字节。
- **部署简化：** 只需加载单个大型基础模型副本，不同的微小“任务向量 (vector)”（训练好的PEFT参数）可以被替换或并发使用来服务多个任务，极大地简化了服务基础设施。
- **减轻遗忘：** 由于原始模型的大部分保持不变，PEFT方法在预训练期间学到的通用能力方面通常表现出更少的灾难性遗忘。

设想一个使用PEFT方法的场景。与其更新基础模型 `M` 的所有 `N` 个参数，我们引入一小组 `k` 个新参数，其中 k≪Nk \ll Nk≪N。

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    "gpt2-xl"
)
N = sum(p.numel() for p in base_model.parameters())
print(f"基础模型参数量：{N:,}")

for p in base_model.parameters():
    p.requires_grad = False

class SimpleAdapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.activation = nn.ReLU()

    def forward(self, x):

        return self.up_proj(self.activation(self.down_proj(x)))

adapter_params = []
```

> 标准微调与参数高效微调（PEFT）之间的参数量和存储影响比较。PEFT旨在使 `k` 远小于 `N`。

“后续章节将考察特定的PEFT技术，例如适配器模块和与专家混合（Mixture-of-Experts）相关的方法，详细说明它们如何实现这种效率，同时保持在下游任务上的强大性能。这些技术对于使大型模型在工程实践中更具实用性和适应性正变得日益重要。”

获取即时帮助、个性化解释和交互式代码示例。

---

### Transformer的适配器模块

# Transformer的适配器模块

为特定下游任务微调 (fine-tuning)庞大的预训练 (pre-training)语言模型（如Transformer）带来了显著的计算和存储难题。为每个新任务重新训练整个模型（可能包含数十亿参数 (parameter)）通常不切实际。参数高效微调（PEFT）方法旨在通过仅使用少量新增或修改的参数来适应模型，从而应对这一问题。适配器模块是PEFT技术中最早且最有影响力的策略之一。

适配器的核心理念是在保持预训练Transformer原有权重 (weight)不变的前提下，向其现有架构中注入小型、可训练的模块。在微调过程中，仅更新这些新增适配器模块的参数。与完全微调相比，这极大地减少了可训练参数的数量，通常能降低几个数量级。

### 适配器架构

适配器模块通常由一个瓶颈结构构成，旨在将输入维度映射到一个小得多的中间维度，然后再将其映射回原始维度。该结构包含：

1. 下投影线性层。
2. 非线性激活函数 (activation function)（例如，GeLU、ReLU）。
3. 上投影线性层。
4. 一个残差连接，将适配器的输出加到其接收到的原始输入上。

瓶颈维度（即中间层的大小）是一个重要的超参数 (parameter) (hyperparameter)。它控制着适配器中的参数数量，并影响参数效率与任务性能之间的平衡。更小的瓶颈维度会带来更少的参数，但可能会限制适配器学习特定任务特征的能力。

G

input
层输入（隐状态）

downₚroj

下投影
线性层 (dₘodel -> bottleneck\_dim)

input->downₚroj

add

+

input->add

nonₗinearity

非线性处理
(例如 GeLU)

downₚroj->nonₗinearity

upₚroj

上投影
线性层 (bottleneck\_dim -> dₘodel)

nonₗinearity->upₚroj

upₚroj->add

output
层输出

add->output

> 适配器模块的基本结构，展示了下投影、非线性处理、上投影和残差连接。

### 在Transformer块中的位置

适配器通常被插入到每个Transformer块中，通常位于多头注意力 (multi-head attention)（MHA）子层和前馈网络（FFN）子层之后，但在该子层的最终残差连接和层归一化 (normalization)之前。

G

clusterₜransformer\_block

Transformer 块

input

来自上一层的输入

mha

多头注意力

input->mha

add1

+

input->add1

mha->add1

adapter1

适配器

add1->adapter1

addₐdapt1

+

add1->addₐdapt1

ln1

层归一化

ffn

前馈网络

ln1->ffn

add2

+

ln1->add2

adapter1->addₐdapt1

addₐdapt1->ln1

ffn->add2

adapter2

适配器

add2->adapter2

addₐdapt2

+

add2->addₐdapt2

ln2

层归一化

output

输出到下一层

ln2->output

adapter2->addₐdapt2

addₐdapt2->ln2

> 适配器模块在标准Transformer块中的位置，通常位于MHA和FFN子层之后。请注意适配器自身周围的残差连接。

### 实现范例

以下是适配器模块的简化PyTorch实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck_dim, dropout=0.1):
        super().__init__()
        self.down_project = nn.Linear(d_model, bottleneck_dim)
        self.non_linear = nn.GELU()
        self.up_project = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):

        adapter_input = x
        x = self.down_project(x)
        x = self.non_linear(x)
        x = self.up_project(x)
        x = self.dropout(x)

        output = adapter_input + x
        return output
```

请注意 `up_project` 层的初始化策略。将其权重 (weight)和偏置 (bias)初始化为零，可确保在微调 (fine-tuning)开始时，适配器模块基本上表现为恒等函数（输出 = 输入 + 0），从而保持原始模型的行为。这有助于稳定微调过程的初期。

### 使用适配器训练

使用适配器进行微调 (fine-tuning)的过程包括以下步骤：

1. **加载预训练 (pre-training)模型：** 从预训练的Transformer模型开始。
2. **冻结基础模型：** 将原始Transformer模型所有参数 (parameter)的 `requires_grad` 设置为 `False`。这会阻止它们在训练期间被更新。
3. **注入适配器：** 在模型架构中所需位置添加适配器模块。
4. **训练适配器：** 在目标任务的数据集上训练模型。只有新增适配器模块中的参数（以及可能的层归一化 (normalization)参数或最终的分类头部）将 `requires_grad` 设置为 `True` 并接收梯度更新。

由于只有总参数的一小部分（通常为0.5%到5%）被训练，因此优化器状态和梯度的内存需求大大减少，并且与更新整个模型相比，训练过程也快得多。

### 优点与考量

使用适配器有以下几点优势：

- **效率：** 显著减少微调 (fine-tuning)所需的计算资源（GPU时间、内存）。
- **存储：** 无需为每个任务存储一个完整的数十亿参数 (parameter)模型，只需保存少量适配器权重 (weight)。基础模型在所有任务之间共享。
- **模块化：** 不同任务的适配器可以方便地进行插拔。
- **性能：** 研究表明，适配器在许多任务上能够取得与完全微调非常接近的性能，尤其是在数据充足的情况下。

然而，也有一些需要考量的因素：

- **超参数 (hyperparameter)调整：** 瓶颈维度和放置策略会影响性能。
- **潜在性能差距：** 尽管通常接近，但在某些复杂任务或数据量较少的情况下，适配器的性能可能略微落后于完全微调。
- **推理 (inference)延迟：** 添加适配器会引入额外的计算（每个适配器包含两个线性层和一个非线性处理），这可能会使推理延迟相较于原始模型略有增加。然而，这种增加通常很小。

总之，适配器模块为将大型预训练 (pre-training)语言模型应用于各种下游任务提供了一种实用且有效的方法，避免了与完全微调相关的过高成本。它们是参数高效微调这一发展中方向的一项基本技术。

获取即时帮助、个性化解释和交互式代码示例。

---

### 专家混合模型 (MoE) 简介

# 专家混合模型 (MoE) 简介

扩展大型语言模型通常需要在模型容量与计算成本之间做出权衡。标准Transformer模型随着规模增大，参数 (parameter)密度也越高。在一次前向传播中，每个参数都参与了对每个输入token的计算。尽管增加参数通常能提升性能（正如缩放法则所示），但计算需求（以每秒浮点运算次数，即FLOPs衡量）也会成比例增长。当目标是达到万亿级参数的模型时，这构成了一个显著的障碍。

专家混合模型 (MoE) 提供了一种基于条件计算的不同扩展方法。不同于每个Transformer块中单一的前馈网络（FFN）层，MoE层将其替换为多个并行的FFN“专家”网络。对于每个输入token，一个路由机制（通常称为“门控网络”）会动态选择这些专家中的一个小子集（例如，前1个或前2个）来处理该token。

G

clusterₘoe

MoE 层

Input Token

输入令牌

Router (Gating Network)

路由器 (门控网络)

Input Token->Router (Gating Network)

Output

输出

Expert 1 (FFN)

专家 1 (FFN)

Router (Gating Network)->Expert 1 (FFN)

 选择

Expert 2 (FFN)

专家 2 (FFN)

Router (Gating Network)->Expert 2 (FFN)

 选择

Expert N (FFN)

专家 N (FFN)

Router (Gating Network)->Expert N (FFN)

Expert 1 (FFN)->Output

Expert 2 (FFN)->Output

Expert N (FFN)->Output

> MoE层的简化视图。输入token由路由器引导至可用专家网络中的一个小子集。只有被选中的专家对该token执行计算。

主要思想是**稀疏性**。尽管MoE层中的参数总数（路由器和所有专家的参数总和）可以显著多于标准FFN层，但用于处理单个token的*实际使用*参数数量保持相对较小，这由激活的专家数量决定。

让我们考虑一个典型的Transformer块中的FFN层。如果隐藏维度是 dmodeld\_{model}dmodel​ 且FFN内部维度是 dffd\_{ff}dff​，则计算量大约为 2×dmodel×dff2 \times d\_{model} \times d\_{ff}2×dmodel​×dff​ FLOPs。

在一个有 NNN 个专家的MoE层中，每个专家都与原始FFN具有相同维度，并且路由器选择前 kkk 个专家（其中 kkk 远小于 NNN，通常 k=1k=1k=1 或 k=2k=2k=2），每个token的计算成本大致为：

MoE层FLOPs≈路由器FLOPs+k×(2×dmodel×dff)\text{MoE层FLOPs} \approx \text{路由器FLOPs} + k \times (2 \times d\_{model} \times d\_{ff})MoE层FLOPs≈路由器FLOPs+k×(2×dmodel​×dff​)

路由器的计算量与专家相比通常可以忽略不计。如果 k=1k=1k=1，每个token的FLOPs与原始的密集型FFN层相似，尽管总参数数量大约增加了 NNN 倍。这使得构建具有潜在万亿级参数的模型成为可能，同时在训练和推理 (inference)期间保持可控的计算预算。

这里是PyTorch中的结构示例：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """一个简单的前馈网络专家"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class MoELayer(nn.Module):
"""专家混合层"""
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(num_experts)]
        )

        self.gating_network = nn.Linear(d_model, num_experts)

    def forward(self, x):

        batch_size, seq_len, d_model = x.shape

        x_reshaped = x.view(-1, d_model)

        router_logits = self.gating_network(x_reshaped)

        routing_weights = F.softmax(router_logits, dim=1)

        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.top_k, dim=1
        )

        top_k_weights_norm = (
            top_k_weights / top_k_weights.sum(dim=1, keepdim=True)
        )

        final_output = torch.zeros_like(x_reshaped)

        for i in range(batch_size * seq_len):
            token_input = x_reshaped[i]
            for k in range(self.top_k):
                expert_idx = top_k_indices[i, k].item()
                expert_weight = top_k_weights_norm[i, k]

                expert_output = self.experts[expert_idx](token_input)

                final_output[i] += expert_weight * expert_output

        return final_output.view(
            batch_size, seq_len, d_model
        )
```

*注：上述`forward`方法为清晰起见使用了简单的循环。实际实现依赖于优化操作，以高效地处理跨批次和设备的稀疏路由，避免对token进行显式循环。*

MoE方法的主要益处是这种参数数量与计算量的解耦。它使得模型能够显著增加其容量，可能带来更佳的性能和知识表示，而无需按比例增加处理每个token所需的FLOPs。此外，专家可能能够专注于处理不同类型的输入或语言现象，尽管验证和控制这种专业化仍然是活跃的研究方向。

然而，MoE带来了一系列自身挑战，特别是在训练动态和实现复杂性方面。确保路由器有效地将token分发给专家（负载均衡）以及管理分布式环境中的通信开销，是我们将在后续章节中讨论的重要考量。

获取即时帮助、个性化解释和交互式代码示例。

---

### MoE 中的路由机制

# MoE 中的路由机制

MoE（专家混合）层的核心思想是在不按比例增加处理每个令牌的计算成本的情况下，显著增加模型的参数 (parameter)数量。这是通过在MoE层内设置多个“专家”网络（通常是简单的前馈网络），但对于每个输入令牌，只激活其中一小部分（通常是一个或两个）来实现的。实现这种条件计算的重要组成部分是**路由机制**，通常称为**门控网络**。

### 门控网络

门控网络充当MoE层的交通管制员。它的职责是查看每个传入的令牌表示，并决定哪个专家应处理它。

通常，门控网络本身是一个相对简单的神经网络 (neural network)。常见的设计包括：

1. **输入：** 来自前一层的令牌隐藏状态表示 xxx（例如，自注意力 (self-attention)子层的输出）。
2. **变换：** 将带有权重 (weight) WgW\_gWg​ 的线性层应用于输入令牌表示：h=xWgh = x W\_gh=xWg​。
3. **输出分数（Logits）：** 结果 hhh 是一个维度等于专家数量 NNN 的向量 (vector)。这些是每个专家的初始分数或“logits”。
4. **概率（可选但常见）：** 通常对这些 logits 应用 softmax 函数，以生成专家上的概率分布 ppp：p=softmax(h)p = \text{softmax}(h)p=softmax(h)。此向量 ppp 包含概率 pip\_ipi​，使得 ∑i=1Npi=1\sum\_{i=1}^{N} p\_i = 1∑i=1N​pi​=1，表示门将令牌分配给每个专家 iii 的置信度。

以下是一个简单门控网络的 PyTorch 代码片段：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGatingNetwork(nn.Module):
    def __init__(self, model_dim: int, num_experts: int):
        super().__init__()

        self.layer = nn.Linear(model_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        logits = self.layer(x)

        gate_probabilities = F.softmax(logits, dim=-1)

        return gate_probabilities, logits
```

### Top-k 门控

尽管门控网络会生成所有专家的分数或概率，但激活所有专家将违背条件计算的目的。最普遍的路由策略是 **Top-k 门控**，其中只选择得分最高（根据门 logits）的 kkk 个专家来处理令牌。在实践中，kkk 通常非常小，通常只有1或2。

- **k=1：** 每个令牌仅由单个最佳专家处理，这最大化了稀疏性，但可能限制了模型组合来自多个专业化路径的信息的能力。
- **k=2：** 每个令牌由排名前两位的专家处理。这已成为一种受欢迎的选择（例如，在 Google 的 Switch Transformer 变体和 Mixtral 模型中），因为它提供了良好的平衡。它允许令牌受益于两位专家的“知识”，可能提高表示质量，同时将计算开销保持在远低于密集模型的水平。

**计算输出：**
如果令牌 xxx 被路由到排名前 kkk 的专家 Ei1,Ei2,...,EikE\_{i\_1}, E\_{i\_2}, ..., E\_{i\_k}Ei1​​,Ei2​​,...,Eik​​，并带有相应的门控概率（或归一化 (normalization)分数）pi1,pi2,...,pikp\_{i\_1}, p\_{i\_2}, ..., p\_{i\_k}pi1​​,pi2​​,...,pik​​，则该令牌在 MoE 层的最终输出 yyy 计算为这些选定专家输出的加权和：

y=∑j=1kpij⋅Eij(x)y = \sum\_{j=1}^{k} p\_{i\_j} \cdot E\_{i\_j}(x)y=j=1∑k​pij​​⋅Eij​​(x)

此和中使用的概率 pijp\_{i\_j}pij​​ 通常来自门的 softmax 输出，但仅对选定的 Top-k 专家进行重新归一化。

我们来展示一下 PyTorch 中的 Top-k 选择逻辑：

```python

num_experts = len(experts)
k = 2

gate_probabilities, gate_logits = gating_network(x_flat)

top_k_weights, top_k_indices = torch.topk(gate_probabilities, k, dim=-1)

normalized_weights = top_k_weights / torch.sum(
    top_k_weights, dim=-1, keepdim=True
)

final_output = torch.zeros_like(x_flat)

for i in range(num_experts):

    expert_mask = (top_k_indices == i)

    token_indices, _ = torch.nonzero(expert_mask, as_tuple=True)

    if token_indices.numel() > 0:

        weights_for_expert = normalized_weights[expert_mask]

        inputs_for_expert = x_flat[token_indices]

        expert_output = experts[i](inputs_for_expert)

        weighted_output = expert_output * weights_for_expert.unsqueeze(-1)

        final_output.index_add_(0, token_indices, weighted_output)
```

“*注意：* 上述循环效率非常低。实际实现会使用优化的 scatter/gather 操作或专用核来路由令牌和聚合输出，而无需显式循环，特别是在分布式环境中。”

### 带噪声的 Top-k 门控

为了可能改进负载平衡并引入一种正则化 (regularization)形式，一些 MoE 实现使用了**带噪声的 Top-k 门控**。思路很简单：在应用 softmax 并选择 Top-k 专家*之前*，向门 logits 添加随机噪声（通常是高斯噪声）。

hnoisy=h+噪声h\_{\text{noisy}} = h + \text{噪声}hnoisy​=h+噪声
p=softmax(hnoisy)p = \text{softmax}(h\_{\text{noisy}})p=softmax(hnoisy​)

噪声通常由可学习权重 (weight)或固定超参数 (parameter) (hyperparameter)进行缩放。这种噪声注入可以防止门始终依赖于少数几个相同的专家，鼓励训练期间的尝试，有时会带来更好的泛化能力和更均衡的专家利用。

```python

if self.training:
    noise = torch.randn_like(gate_logits) * noise_std_dev

    noisy_logits = gate_logits + noise
else:
    noisy_logits = gate_logits

gate_probabilities = F.softmax(noisy_logits, dim=-1)
top_k_weights, top_k_indices = torch.topk(gate_probabilities, k, dim=-1)
```

### 路由考量

实现有效的路由机制涉及处理几个实际挑战：

1. **专家容量：** 在并行处理设置（如 GPU）中，当工作负载均衡时，计算效率最高。如果门控网络在处理批次中将明显更多的令牌路由到一个专家，而不是其他专家，那么该专家将成为瓶颈。为缓解这种情况，通常会引入**专家容量**这一思想。它定义了专家每批次可以处理的最大令牌数量，该数量根据令牌总数和专家数量计算，并加上一个缓冲（容量因子）。

   - **容量因子：** 大于 1.0 的值（例如 1.25）允许一定程度的不平衡。容量 = `ceil( (num_tokens / num_experts) * capacity_factor )`。
   - **令牌丢弃：** 如果分配给某个专家的令牌数量超出其容量，多余的令牌可能会被“丢弃”，这意味着它们在 MoE 层的输出变为零（或者输入表示未变地通过）。这是不希望的，但有时为了系统效率是必要的。
2. **负载平衡损失：** 为了明确鼓励门控网络在专家之间均匀分配令牌，训练期间通常会在主模型损失中添加一个**辅助负载平衡损失**。一种常见的表述旨在最小化分配给每个专家的令牌比例以及分配给每个专家的路由概率质量比例的变化。

   - 令 NNN 为专家数量，BBB 为批次中的令牌数量。
   - 将 fif\_ifi​ 定义为分配给专家 iii 的令牌比例。
   - 将 PiP\_iPi​ 定义为门控网络在批次中为所有令牌分配给专家 iii 的平均概率。
   - 一个典型的负载平衡损失（简化形式）可能如下所示：
     L平衡=α⋅N∑i=1Nfi⋅PiL\_{\text{平衡}} = \alpha \cdot N \sum\_{i=1}^{N} f\_i \cdot P\_iL平衡​=α⋅Ni=1∑N​fi​⋅Pi​
     其中 α\alphaα 是控制此损失项强度的超参数 (parameter) (hyperparameter)。最小化此损失鼓励 fif\_ifi​ 和 PiP\_iPi​ 都接近 1/N1/N1/N，从而促进负载均衡。

   ```python

   num_tokens, num_experts = gate_probabilities.shape

   if k == 1:
     expert_counts = torch.bincount(top_k_indices.squeeze(), minlength=num_experts)
     f_i = expert_counts.float() / num_tokens
   else:

     f_i = calculate_fraction_dispatched(top_k_indices, num_experts, num_tokens)

   p_i = torch.mean(gate_probabilities, dim=0)

   load_balancing_loss = alpha * num_experts * torch.sum(f_i * p_i)

   total_loss = main_task_loss + load_balancing_loss
   ```
3. **稀疏路由与软路由：** Top-k 门控是一种稀疏路由形式。存在其他方案（“软路由”），其中每个专家处理每个令牌，但输出由来自门控的完整概率分布加权。虽然可能更简单，但软路由失去了 MoE 的计算优势，因为所有专家都对所有令牌处于活动状态，这使得它在大规模效率提升方面较不常见。

路由机制的设计和调整，包括 kkk 的选择、噪声的使用、容量因子以及负载平衡损失系数，是构建和训练有效 MoE 模型的重要方面。这些选择直接影响模型性能、训练稳定性和计算效率。像 DeepSpeed 这样的框架提供了抽象和优化来管理这些复杂性，尤其是在专家可能位于不同硬件设备上的分布式训练场景中。

获取即时帮助、个性化解释和交互式代码示例。

---

### MoE 层中的负载均衡

# MoE 层中的负载均衡

虽然路由机制（例如 top-k 门控）会将输入令牌导向专家混合 (MoE) 层中的特定专家，但它们本身不保证计算负载在所有专家之间均匀分配。通常，如果没有明确干预，门控网络可能会偏好一小部分专家，从而导致严重的负载不均。这种不均带来了以下几个问题：

1. **计算效率低下：** 如果只有少数专家持续被选择，与其余专家相关的参数 (parameter)和计算资源就会未充分利用。这会抵消 MoE 旨在提供的一些效率提升。
2. **训练不稳定：** 通过大量使用的专家回传的梯度可能会主导训练过程，可能导致不稳定或收敛缓慢。
3. **模型质量下降：** 未充分利用的专家可能无法接收到足够的训练信号以变得专业或有效，从而限制了模型的整体能力和表现。

因此，在训练 MoE 模型时，实施促进负载均衡的机制是一种标准做法。最常见的方法是向主要任务损失（例如，语言建模的交叉熵损失）添加一个辅助损失项。此辅助损失会惩罚不平衡，指导门控网络更均匀地分配令牌。

### 辅助负载均衡损失

辅助损失的目标是激励路由器为每个专家分配大致相同数量的令牌。Switch Transformer 论文和后续工作中介绍的一种被广泛采用的公式，旨在最小化每个专家处理的令牌数量的变异。

令 NNN 为专家数量， BBB 为当前批次（或微批次）中的令牌数量。对于每个专家 i∈{1,…,N}i \in \{1, \dots, N\}i∈{1,…,N}，我们可以定义两个量：

- fif\_ifi​: 路由到专家 iii 的批次内令牌分数。这通过对批次中所有令牌的专家 iii 路由决策（硬路由如 top-k 为 0 或 1）求和，再除以 BBB 来计算。
  fi=1B∑x∈BatchI(为令牌 x 选择的专家 i)f\_i = \frac{1}{B} \sum\_{x \in \text{Batch}} \mathbb{I}(\text{为令牌 } x \text{ 选择的专家 } i)fi​=B1​x∈Batch∑​I(为令牌 x 选择的专家 i)
- PiP\_iPi​: 门控网络分配给专家 iii 的总路由概率质量的分数，对批次中的令牌进行平均。如果 g(x)ig(x)\_ig(x)i​ 是给定令牌 xxx 时门控网络输出的专家 iii 概率，那么：
  Pi=1B∑x∈Batchg(x)iP\_i = \frac{1}{B} \sum\_{x \in \text{Batch}} g(x)\_iPi​=B1​x∈Batch∑​g(x)i​

辅助负载均衡损失 LbalanceL\_{balance}Lbalance​ 通常计算为这两个向量 (vector)的点积，并乘以专家数量 NNN 和一个可调超参数 (parameter) (hyperparameter) α\alphaα：

Lbalance=α⋅N⋅∑i=1Nfi⋅PiL\_{balance} = \alpha \cdot N \cdot \sum\_{i=1}^{N} f\_i \cdot P\_iLbalance​=α⋅N⋅i=1∑N​fi​⋅Pi​

用于反向传播 (backpropagation)的总损失是主要任务损失 LtaskL\_{task}Ltask​ 和均衡损失之和：

Ltotal=Ltask+LbalanceL\_{total} = L\_{task} + L\_{balance}Ltotal​=Ltask​+Lbalance​

**直观理解：** 最小化 LbalanceL\_{balance}Lbalance​ 促使所有专家的 fif\_ifi​ 和 PiP\_iPi​ 都接近 1/N1/N1/N。如果某个专家接收了大量令牌份额（fif\_ifi​ 较高），损失就会增加。类似地，如果门控网络分配给某个专家高概率（PiP\_iPi​ 较高），损失也会增加。当实际分配 (fif\_ifi​) 和路由器的置信度 (PiP\_iPi​) 均匀分布时，损失最小化。超参数 α\alphaα 控制这种均衡激励相对于主要任务目标的强度；典型值通常很小（例如 0.01）。

下面是一个 PyTorch 代码片段，说明了计算方法，假设 `gating_outputs` 包含来自门控网络的概率，而 `indices` 包含每个令牌选择的专家索引：

```python
import torch
import torch.nn.functional as F

num_experts = 8
num_tokens = 1024
k = 2
gating_outputs = torch.randn(num_tokens, num_experts).softmax(dim=-1)

indices = torch.topk(gating_outputs, k, dim=-1).indices

expert_mask = F.one_hot(indices, num_classes=num_experts).sum(dim=1)

tokens_per_expert = expert_mask.sum(dim=0)
f_i = tokens_per_expert / num_tokens

P_i = gating_outputs.mean(dim=0)

load_balance_loss = num_experts * torch.sum(f_i * P_i)

alpha = 0.01

print(f"负载均衡损失项: {load_balance_loss.item():.4f}")
print(f"每个专家的令牌分布: {f_i.detach().numpy()}")
print(f"每个专家的平均概率: {P_i.detach().numpy()}")
```

Expert 1Expert 2Expert 3Expert 4Expert 5Expert 6Expert 7Expert 800.050.10.150.20.25不均衡负载理想均衡负载

> 示例可视化，比较专家间不均衡的令牌分布与理想的完全均衡状态。辅助损失旨在将分布推向均衡状态。

### 容量因子

另一个常与辅助损失结合使用的机制是**容量因子** (CCC)。这限制了任何单个专家在一个批次内可以处理的令牌数量。每个专家的容量通常设置为：

容量=C×每批次令牌数专家数量\text{容量} = C \times \frac{\text{每批次令牌数}}{\text{专家数量}}容量=C×专家数量每批次令牌数​

容量因子 CCC 通常略大于 1（例如 1.25 或 1.5）。如果路由机制分配给某个专家的令牌数量超过其允许的容量，则超额令牌被视为“丢弃”或“溢出”。这些丢弃的令牌不参与该 MoE 层的计算（包括前向和反向传播 (backpropagation)），实际上像通过了一个恒等函数一样被处理。

虽然丢弃令牌可能看起来有害，但使用容量因子提供了对抗严重不平衡的硬性约束。它防止单个专家过载，即使辅助损失尚未完全纠正路由器的偏好。然而，将 CCC 设置得过低可能导致过多的令牌丢弃，从而阻碍学习。在强制均衡和保留所有信息之间存在权衡。在训练期间监控丢弃令牌的百分比对于调整 CCC 很重要。

### 与分布式训练的交互

负载均衡在分布式设置中尤为重要，尤其是在使用**专家并行**时，即不同专家位于不同的计算设备（例如 GPU）上。如果负载不均衡，承载受偏好专家的设备会成为瓶颈，而拥有未充分利用专家的设备则处于空闲状态，导致扩展性差和资源浪费。辅助损失和容量因子共同作用，以确保计算在分布式硬件上更均匀地分布。

总之，实现专家间负载的均衡分配对于高效且稳定的 MoE 模型训练来说，是必要条件。辅助负载均衡损失与仔细调整的容量因子相结合，提供了有效机制来鼓励门控网络更均匀地使用所有专家，从而最大化条件计算的益处。调整辅助损失系数 α\alphaα 和容量因子 CCC 是成功训练大型 MoE 模型的重要方面。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 15 Distributed Training Strategies

### 动机：为什么要进行分布式训练？

# 动机：为什么要进行分布式训练？

目前训练巨型语言模型，这些模型通常包含数百亿甚至数千亿参数 (parameter)，带来了严峻的工程挑战，仅靠单个加速设备（如GPU或TPU）是无法应对的。

虽然较小的深度学习 (deep learning)模型可以轻松地在一块GPU内存中运行，但大型语言模型（LLMs）的规模使得这变得不再可行。尝试在单个设备上加载和训练这类巨型模型，会迅速遭遇硬件方面的基本限制，这些限制涉及内存容量和计算吞吐量 (throughput)。

我们来分析一下为什么分布式训练过程变得绝对必要。

### 内存限制：模型及其附属数据的存放问题

最直接的挑战是内存。一块高端加速卡，例如NVIDIA A100或H100 GPU，通常提供约40GB或80GB的高带宽内存（HBM）。这看起来容量很大，但LLM在训练期间的内存占用包括以下几个主要部分：

1. **模型参数 (parameter)：** 这些是模型学习到的权重 (weight)和偏置 (bias)。存储它们需要大量空间，尤其是在使用标准32位浮点精度（FP32）时。即使使用FP16或BFloat16 (BF16)等16位格式进行混合精度训练，仅参数存储就可能超出单个GPU的内存容量。例如，一个包含700亿参数的模型，若使用BF16（每个参数2字节）存储，则需要140GB内存，这已经远远超过了80GB GPU的容量。

   ```python

   num_parameters = 70_000_000_000
   bytes_per_parameter_mixed_precision = 2

   param_memory_gb = (num_parameters * bytes_per_parameter_mixed_precision) / (1024**3)
   print(f"Approx. parameter memory (BF16/FP16): {param_memory_gb:.2f} GB")
   ```
2. **梯度：** 在反向传播 (backpropagation)过程中，会为每个参数计算梯度。这些梯度通常与参数本身具有相同的维度和精度要求。因此，在我们700亿参数的混合精度示例中，仅存储梯度就需要额外140GB内存。
3. **优化器状态：** Adam或AdamW等现代优化器会维护内部状态，以调整每个参数的学习率。AdamW是LLM训练中常用的优化器，它为每个参数存储两种状态：一阶矩（动量）和二阶矩（方差）。为了保持稳定性，即使在混合精度训练中，这些状态也常以更高精度（FP32，4字节）保存。这意味着优化器状态还需要额外70 亿×2 个状态×4 字节/状态=56070 \text{ 亿} \times 2 \text{ 个状态} \times 4 \text{ 字节/状态} = 56070 亿×2 个状态×4 字节/状态=560 GB。

   ```python

   bytes_per_state_fp32 = 4
   num_states_per_param_adamw = 2

   optimizer_memory_gb = (num_parameters * num_states_per_param_adamw *
                      bytes_per_state_fp32) / (1024**3)
   print(f"Approx. AdamW optimizer state memory (FP32): {optimizer_memory_gb:.2f} GB")
   ```
4. **激活值：** 在正向传播过程中，每一层的中间输出（激活值）必须存储起来，以便在反向传播计算梯度时使用。这些激活值的大小取决于批处理大小、序列长度和模型隐藏维度。对于Transformer模型，自注意力 (self-attention)机制 (attention mechanism)尤其占用内存，在其朴素实现中可能需要O(批处理大小×序列长度2×隐藏维度)O(批处理大小 \times 序列长度^2 \times 隐藏维度)O(批处理大小×序列长度2×隐藏维度)的内存。虽然激活检查点（在反向传播时重新计算激活值而非存储它们）等技术可以减少内存占用，但仍然需要大量的激活值内存，通常根据配置不同，可能达到数十或数百GB。

仅将参数、梯度和优化器状态加起来，对于我们700亿参数的示例，总计为140+140+560=840140 + 140 + 560 = 840140+140+560=840 GB。这已经超过了单个80GB GPU容量的十倍，而且我们甚至还没有完全计入激活值或cuDNN等库所需的额外工作内存。

参数 (BF16)梯度 (BF16)优化器状态 (AdamW)单块GPU HBM (典型)0100200300400500内存估算：约700亿参数模型 vs. 单块GPU

> 训练约700亿参数模型时的内存估算细分，与高端GPU典型的80GB HBM内存对比。激活值和工作内存还会增加额外的需求。

显然，内存需求使得单个设备无法满足要求。

### 计算能力限制：使训练变为可能

除了内存问题，训练LLM所需的极高的计算成本（以浮点运算，即FLOPs衡量）非常高昂。对一个包含数十亿参数 (parameter)的模型进行一次前向和反向传播 (backpropagation)，涉及数万亿次计算。

- **FLOPs需求：** 训练涉及在多个训练周期中处理海量数据集（数万亿个token）。所需的总计算量会随模型大小和数据集大小的增长而大幅增加。经验性的缩放法则表明，要获得好的性能，需要大量的计算预算，这通常以PetaFLOP-天（1 PetaFLOP = 101510^{15}1015 FLOPs）来衡量。
- **训练时间：** 即使是最快的单个加速器，每秒也只能执行有限数量的FLOPs（例如，稠密运算可达数百TeraFLOPs）。在单个设备上完成一次完整的LLM预训练 (pre-training)，需要ExaFLOPs（101810^{18}1018 FLOPs）或更多的计算量，这将耗时数月、数年甚至数十年，使其完全不切实际。研究和开发需要更快的迭代周期。
- **注意力机制 (attention mechanism)的复杂度：** 自注意力 (self-attention)机制是Transformer模型的根本组成部分，具有O(序列长度2×隐藏维度)O(序列长度^2 \times 隐藏维度)O(序列长度2×隐藏维度)的计算复杂度。随着模型在更长上下文 (context)窗口（数千个token）上进行训练，注意力计算的成本变得越来越主导，进一步减慢了单个设备上的训练速度。

### 必由之路：分布式训练

面对这些内存和计算瓶颈，一个有效的方法是将训练负载**分布式**到多个互联加速器组成的集群上。通过在多个并行工作的设备上划分模型的参数 (parameter)、数据或计算图，我们可以做到：

1. 汇集内存容量以容纳整个模型状态。
2. 汇集计算能力，大幅缩短训练时间，从数年降至数周或数天。

接下来的章节会介绍为实现这种分布而发展起来的主要策略：数据并行、张量并行和流水线并行，以及它们的组合和相关的通信考量。理解这些技术对于任何参与构建和训练大型语言模型的人员来说都非常重要。

获取即时帮助、个性化解释和交互式代码示例。

---

### 数据并行 (DP)

# 数据并行 (DP)

数据并行（DP）是分配深度学习 (deep learning)模型（包括大型语言模型）训练计算负担的最直接和常用策略。在单个设备上训练这些大型模型通常会遇到计算时间和内存限制的瓶颈。数据并行通过在多个设备上同时处理数据的不同部分，直接解决计算时间方面的问题。

核心原理很简单：将整个模型复制到每个可用的处理单元（如GPU）上，然后将每个全局数据批次分成更小的部分，称为微批次。每个设备独立处理其分配到的微批次。

### 数据并行如何工作

让我们分析使用KKK个设备进行数据并行训练的一个迭代中的典型步骤：

1. **模型复制：** 训练开始前，整个模型（参数 (parameter)、缓冲区）的相同副本被加载到KKK个设备中的每一个上。
2. **数据分片：** 全局训练数据批次被分割成KKK个微批次。每个设备接收一个微批次。例如，如果全局批次大小为BBB，我们有KKK个设备，每个设备通常处理大小为B/KB/KB/K的微批次。
3. **前向传播：** 每个设备使用其本地模型副本和分配到的微批次执行前向传播。这会计算该部分数据的损失。
4. **反向传播 (backpropagation)：** 每个设备通过反向传播计算损失相对于其本地模型参数的梯度。此时，每个设备iii持有一组梯度gig\_igi​，仅根据其微批次计算得出。
5. **梯度同步：** 这是数据并行的定义性步骤。在每个设备上独立计算的梯度必须结合起来，以代表*整个*全局批次的梯度。这通常通过一种称为`AllReduce`的集体通信操作实现。`AllReduce`操作将所有KKK个设备中的梯度gig\_igi​求和（或平均），并将得到的同步梯度gsyncg\_{sync}gsync​分发回每个设备。通常使用平均：
   gsync=1K∑i=1Kgig\_{sync} = \frac{1}{K} \sum\_{i=1}^{K} g\_igsync​=K1​i=1∑K​gi​
6. **优化器步骤：** 每个设备使用*同步*梯度gsyncg\_{sync}gsync​，通过所选优化器（例如AdamW）更新其本地的模型参数副本。因为所有设备初始参数相同，并基于同步梯度应用相同的更新，所以在优化器步骤之后，模型副本保持相同。

这个循环针对训练数据集中的每个批次重复。

DP\_Flow

cluster\_data

数据输入

cluster\_workers

并行处理（K个设备）

GlobalBatch

全局批次

Splitter

分割数据

GlobalBatch->Splitter

Device1

设备 1
微批次 1
模型副本
前向/反向
(本地梯度 g₁)

Splitter->Device1

微批次 1

DeviceK

设备 K
微批次 K
模型副本
前向/反向
(本地梯度 gₖ)

Splitter->DeviceK

微批次 K

Dots
...

AllReduce

AllReduce
(同步梯度)

Device1->AllReduce

g₁

DeviceK->AllReduce

gₖ

Optimizer

优化器步骤
(更新模型)

AllReduce->Optimizer

gₛync

Optimizer->Device1

更新权重

Optimizer->DeviceK

更新权重

> 数据并行工作流程：全局批次被分割，在持有模型副本的设备上并行处理，梯度通过AllReduce同步，并将同步更新应用到每个副本。

### 数据并行的优点

- **简洁性：** PyTorch等框架（如`DistributedDataParallel`）提供了高级抽象，使得数据并行的实现相对简单，通常只需要对标准单设备训练代码进行少量修改。
- **吞吐量 (throughput)增加：** 通过并行处理数据，数据并行大幅减少了每个训练步骤的时间，从而能够更快地遍历大型数据集或使用更大的有效批次大小，这有时可以改进模型收敛和泛化能力。
- **广泛适用性：** 数据并行本身不需要对模型架构进行根本性改变。它对大多数标准网络设计来说是“开箱即用”的。

### 数据并行的局限性

尽管有其优点，数据并行存在一个显著的局限性，特别是对于本课程关注的*大型*语言模型：

- **内存限制：** 主要缺点是*每个*设备都必须保存模型的参数 (parameter)、梯度、优化器状态的完整副本，以及其微批次在前向传播期间计算的激活值。对于具有数十亿或数万亿参数的模型，所需的内存通常会超出即使是最大可用加速器（GPU/TPU）的容量。这从根本上限制了仅使用数据并行可以训练的最大模型大小。
- **通信开销：** `AllReduce`操作需要在所有相关设备之间通信梯度。传输的数据量与模型参数的大小成比例。随着设备数量（KKK）的增加，或者如果设备之间的互联带宽有限，这个同步步骤可能成为一个显著瓶颈，从而减弱并行计算带来的加速效果。`AllReduce`所需的时间有时会占用大部分计算时间，特别是对于较小的模型或每个设备计算速度较快的情况。

### 使用PyTorch DistributedDataParallel (DDP) 的实现概述

PyTorch 的 `torch.nn.parallel.DistributedDataParallel` 模块是在多进程设置中实现数据并行的标准方式，通常优于旧的 `torch.nn.DataParallel`，因为它具有更好的性能和灵活性。以下是一个概要：

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os

def setup(rank, world_size):
    """初始化进程组。"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """销毁进程组。"""
    dist.destroy_process_group()

def train_worker(rank, world_size, model_args, data_args, train_args):
    """单个工作进程的主要训练函数。"""
    print(f"在 {rank} 进程上运行DDP训练。")
    setup(rank, world_size)

    model = MyLargeModel(**model_args).to(rank)

    ddp_model = DDP(model, device_ids=[rank])

    dataset = YourDataset(**data_args)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    dataloader = DataLoader(
        dataset,
        batch_size=train_args['micro_batch_size'],
        sampler=sampler,
        pin_memory=True,
        num_workers=4
    )

    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=train_args['learning_rate']
    )

    ddp_model.train()
    for epoch in range(train_args['num_epochs']):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            inputs = batch['input_ids'].to(rank)
            labels = batch['labels'].to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(inputs, labels=labels)
            loss = outputs.loss

            loss.backward()

            optimizer.step()

            if rank == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

    cleanup()

if __name__ == '__main__':

    world_size = torch.cuda.device_count()
    model_args = {'vocab_size': 50257, 'hidden_size': 768, 'num_layers': 12}
    data_args = {'data_path': '/path/to/data'}
    train_args = {
        'micro_batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 3
    }

    mp.spawn(
        train_worker,
        args=(world_size, model_args, data_args, train_args),
        nprocs=world_size,
        join=True
    )
```

在这个概述中：

- `setup` 初始化分布式环境。每个进程获得一个从0到`world_size - 1`的唯一`rank`。
- `DistributedSampler` 与 `DataLoader` 一起使用，以确保每个进程获得不重叠的数据分区。
- 主要变化是包装模型：`ddp_model = DDP(model, device_ids=[rank])`。DDP会自动拦截反向传播 (backpropagation)，对梯度执行`AllReduce`操作，并确保在调用`optimizer.step()`之前所有进程都拥有平均后的梯度。

数据并行是扩展训练吞吐量 (throughput)的一种基本技术。然而，其内存限制使得有必要考虑张量并行和流水线并行等其他策略，尤其是在处理现代大型语言模型的庞大规模时。通常，最有效的方法是将数据并行与其他这些技术结合使用，我们接下来将研究这些技术。

获取即时帮助、个性化解释和交互式代码示例。

---

### 张量并行 (TP)

# 张量并行 (TP)

数据并行（DP）通过复制模型和处理不同的数据块来有效使用多个设备，但这要求每个设备承载整个模型。对于规模很大的模型，即使是单个副本也可能超出单个加速器的内存容量。此外，模型中的一些操作，例如前馈网络或注意力机制 (attention mechanism)中的大型线性变换，可能成为计算瓶颈。在这种情况下，张量并行（TP），有时也称为层内模型并行，是必不可少的。

张量并行不是分割数据或层序列，而是在特定操作*内部*，将实际张量（权重 (weight)、激活、梯度）拆分到多个设备上。这使得对这些大张量的计算能够并行进行，分散单个层的内存占用和计算负担。

### 线性层的拆分

张量并行最直接的应用是在大型线性层中，它们是Transformer中多层感知机（MLP）和注意力块的重要组成部分。假设一个线性变换 Y=XAY = XAY=XA，其中XXX是输入激活，AAA是权重 (weight)矩阵。我们可以将这种矩阵乘法并行化，例如在两个设备上，主要有两种方式：列并行和行并行。

#### 列并行

在列并行中，权重矩阵AAA垂直（沿其列）拆分到设备上。如果我们有两个设备，我们将AAA拆分为 A=[A1,A2]A = [A\_1, A\_2]A=[A1​,A2​]。输入XXX通常会被广播或提供给两个设备。每个设备随后计算一部分输出：

- 设备1计算：Y1=XA1Y\_1 = X A\_1Y1​=XA1​
- 设备2计算：Y2=XA2Y\_2 = X A\_2Y2​=XA2​

最终输出YYY通过沿列维度拼接结果获得：Y=[Y1,Y2]Y = [Y\_1, Y\_2]Y=[Y1​,Y2​]。

G

cluster₀

GPU 1

cluster₁

GPU 2

X

输入X

XA1

X \* A1

X->XA1

XA2

X \* A2

X->XA2

Y

输出Y = [Y1, Y2]

XA1->Y

Y1

A1

A1

A1->XA1

XA2->Y

Y2

A2

A2

A2->XA2

> Y=XAY = XAY=XA 的列并行。权重矩阵AAA被拆分为A1A\_1A1​和A2A\_2A2​。输入XXX在不同的GPU上与每个部分相乘。结果Y1Y\_1Y1​和Y2Y\_2Y2​被拼接起来形成最终输出YYY。

从数学角度看，前向传播涉及 XA=X[A1,A2]=[XA1,XA2]X A = X [A\_1, A\_2] = [X A\_1, X A\_2]XA=X[A1​,A2​]=[XA1​,XA2​]。反向传播 (backpropagation)需要计算相对于XXX和AAA的梯度。梯度 ∂L∂A\frac{\partial L}{\partial A}∂A∂L​ 是自然地划分的（∂L∂A1\frac{\partial L}{\partial A\_1}∂A1​∂L​，∂L∂A2\frac{\partial L}{\partial A\_2}∂A2​∂L​）。然而，计算梯度 ∂L∂X\frac{\partial L}{\partial X}∂X∂L​ 需要汇总来自两条路径的贡献：∂L∂X=∂L∂Y1A1T+∂L∂Y2A2T\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y\_1} A\_1^T + \frac{\partial L}{\partial Y\_2} A\_2^T∂X∂L​=∂Y1​∂L​A1T​+∂Y2​∂L​A2T​。这种求和通常通过在持有A1A\_1A1​和A2A\_2A2​的设备之间使用all-reduce通信操作来完成。

#### 行并行

在行并行中，权重矩阵AAA水平（沿其行）拆分。对于两个设备，A=[A1A2]A = \begin{bmatrix} A\_1 \\ A\_2 \end{bmatrix}A=[A1​A2​​]。输入XXX也被认为是沿其列划分的（通常因为它是前一个列并行层的输出）。为了在这种独立的视角下简化，我们假设输入XXX在两个设备上都可用。每个设备根据其权重切片计算部分结果：

- 设备1计算：Y1=XA1Y\_1 = X A\_1Y1​=XA1​（注意：这个公式略有简化。实际上，XXX会被分区，例如X=[X1,X2]X=[X\_1, X\_2]X=[X1​,X2​]，设备1计算X1A1X\_1 A\_1X1​A1​，设备2计算X2A2X\_2 A\_2X2​A2​。我们将在后面展示与列并行结合使用的常见模式）。
- 设备2计算：Y2=XA2Y\_2 = X A\_2Y2​=XA2​

与列并行不同，最终输出YYY是部分结果的*和*：Y=Y1+Y2Y = Y\_1 + Y\_2Y=Y1​+Y2​。

G

cluster₀

GPU 1

cluster₁

GPU 2

X

输入X

XA1

X \* A1

X->XA1

XA2

X \* A2

X->XA2

AllReduce

AllReduce (+)

XA1->AllReduce

Y1

A1

A1

A1->XA1

XA2->AllReduce

Y2

A2

A2

A2->XA2

Y

输出Y

AllReduce->Y

> Y=XAY = XAY=XA 的行并行。权重矩阵AAA被拆分为A1A\_1A1​和A2A\_2A2​（行方向）。部分结果Y1Y\_1Y1​和Y2Y\_2Y2​在不同的GPU上计算。一个all-reduce操作将这些结果求和以生成最终输出YYY。

从数学角度看，如果XXX被视为一个单独的块，Y=X[A1A2]Y = X \begin{bmatrix} A\_1 \\ A\_2 \end{bmatrix}Y=X[A1​A2​​]并非标准的矩阵乘法表示。在这种情况下（例如，紧随一个列拆分层之后）的实际操作是Y=X1A1+X2A2Y = X\_1 A\_1 + X\_2 A\_2Y=X1​A1​+X2​A2​，其中X=[X1,X2]X=[X\_1, X\_2]X=[X1​,X2​]。前向传播需要all-reduce操作以在设备之间执行此求和。反向传播中∂L∂X\frac{\partial L}{\partial X}∂X∂L​的计算涉及基于已分区AAA的分区梯度：∂L∂X1=∂L∂YA1T\frac{\partial L}{\partial X\_1} = \frac{\partial L}{\partial Y} A\_1^T∂X1​∂L​=∂Y∂L​A1T​和∂L∂X2=∂L∂YA2T\frac{\partial L}{\partial X\_2} = \frac{\partial L}{\partial Y} A\_2^T∂X2​∂L​=∂Y∂L​A2T​。梯度∂L∂A\frac{\partial L}{\partial A}∂A∂L​直接在每个设备上为其对应的权重切片计算。

### Transformer中的张量并行

张量并行通常有策略地应用于Transformer块内部，以平衡计算并减少通信开销。Megatron-LM等框架推广的一种常见模式是在MLP块中结合了列并行和行并行，并对注意力机制 (attention mechanism)应用了类似的分区。

**MLP块：** 标准的Transformer MLP块计算 Y=Activation(XA)B+XY = \text{Activation}(XA)B + XY=Activation(XA)B+X （包含残差连接）。张量并行按以下方式应用：

1. **第一线性层 (XAXAXA)：** 对矩阵AAA使用列并行。将AAA拆分为A=[A1,A2]A = [A\_1, A\_2]A=[A1​,A2​]。在各自设备上计算Z1=XA1Z\_1 = XA\_1Z1​=XA1​和Z2=XA2Z\_2 = XA\_2Z2​=XA2​。输出是[Z1,Z2][Z\_1, Z\_2][Z1​,Z2​]。此步骤在反向传播 (backpropagation)的梯度计算中涉及all-reduce，但如果XXX已在两个设备上可用，前向传播则不需要通信。
2. **激活：** 将激活函数 (activation function)（例如，GeLU、SwiGLU）逐元素应用于分区输出：G1=Activation(Z1)G\_1 = \text{Activation}(Z\_1)G1​=Activation(Z1​)，G2=Activation(Z2)G\_2 = \text{Activation}(Z\_2)G2​=Activation(Z2​)。无需通信。
3. **第二线性层 (GBGBGB)：** 对矩阵BBB使用行并行。将BBB拆分为B=[B1B2]B = \begin{bmatrix} B\_1 \\ B\_2 \end{bmatrix}B=[B1​B2​​]。设备1计算Y1=G1B1Y\_1 = G\_1 B\_1Y1​=G1​B1​，设备2计算Y2=G2B2Y\_2 = G\_2 B\_2Y2​=G2​B2​。
4. **求和：** 线性变换的最终输出是Y=Y1+Y2Y = Y\_1 + Y\_2Y=Y1​+Y2​。这个求和在前向传播中通过all-reduce操作执行。反向传播中的梯度计算对于输入G1,G2G\_1, G\_2G1​,G2​不需要all-reduce。

这种组合巧妙地安排了并行，使得前向传播所需的通信（第二线性层后的all-reduce）和反向传播所需的通信（第一线性层后∇X\nabla X∇X的all-reduce）不会不必要地重叠，从而优化了流程。

**注意力块：** 张量并行也可应用于自注意力 (self-attention)机制。

1. **Q、K、V投影：** 用于将输入投影到查询（queries）、键（keys）和值（values）的权重 (weight)矩阵WQW\_QWQ​、WKW\_KWK​和WVW\_VWV​通常使用列并行进行拆分，类似于MLP的第一线性层。Q=XWQQ = XW\_QQ=XWQ​、K=XWKK = XW\_KK=XWK​、V=XWVV = XW\_VV=XWV​在设备之间并行计算，从而得到分区后的Q=[Q1,Q2]Q = [Q\_1, Q\_2]Q=[Q1​,Q2​]、K=[K1,K2]K = [K\_1, K\_2]K=[K1​,K2​]、V=[V1,V2]V = [V\_1, V\_2]V=[V1​,V2​]。
2. **注意力分数计算：** 注意力分数S=softmax(QKTdk)S = \text{softmax}(\frac{QK^T}{\sqrt{d\_k}})S=softmax(dk​​QKT​)涉及分区Q和K张量之间的矩阵乘法。这需要细致的实现和通信（例如，根据具体的拆分策略可能需要all-gather操作）来在每个设备上计算完整的注意力矩阵或其必要部分。
3. **值聚合：** 输出O=SVO = SVO=SV涉及将注意力分数SSS与分区的值张量VVV相乘。
4. **输出投影：** 注意力块中的最终线性层（OWOO W\_OOWO​）通常使用行并行进行并行化，类似于MLP的第二线性层。这在前向传播中需要一个all-reduce来合并结果。

注意力块内部的实现细节可能比较复杂，包含优化的核函数和通信模式，以高效处理序列长度和头维度。

### 通信成本

张量并行一个主要的缺点是与数据并行相比，它增加了通信开销。虽然数据并行通常每个训练步骤（对于梯度）涉及一次all-reduce，但张量并行在每个Transformer块的前向和反向传播 (backpropagation)*内部*引入了通信。

- **列并行：** 在反向传播期间，需要通信（all-reduce）来计算关于输入XXX的梯度。
- **行并行：** 在前向传播期间，需要通信（all-reduce）来求和输出。
- **注意力：** 根据实现情况，可能涉及额外的通信，例如all-gather。

这些通信操作（如all-reduce）涉及在所有参与张量并行组的设备之间同步和交换数据。交换的数据量取决于激活或梯度的大小。这种通信成本随张量并行组中设备数量的增加而变化，如果设备间互连带宽（例如NVLink、InfiniBand）相对于计算速度不足，可能成为性能瓶颈。

### 优点与考虑

- **支持更大模型：** 当模型层本身对于单个设备的内存而言过大时，张量并行是十分必要的。
- **补充其他策略：** 它可以与数据并行（DP）和流水线并行（PP）结合，形成混合并行方法（例如，在流水线阶段内部使用TP，然后使用DP进行复制）。
- **潜在的重叠：** 精心设计的实现可能会将通信与计算重叠，以隐藏延迟。

然而，请注意以下几点：

- **实现复杂度：** 需要细致的代码修改，或者依赖于NVIDIA的Megatron-LM或微软的DeepSpeed等专业库，这些库提供了Transformer中张量并行的优化实现。
- **通信瓶颈：** 性能对计算集群的互连带宽和拓扑结构高度敏感。如果通信成为主导因素，将张量并行扩展到大量设备可能导致回报递减。

这是一个PyTorch风格的代码片段，展示了为列并行拆分权重 (weight)矩阵的*思路*（这高度简化，省略了通信和梯度处理）：

```python
import torch
import torch.nn as nn

class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, world_size, rank):
        super().__init__()
        self.input_size = input_size

        self.output_size_per_partition = output_size // world_size
        self.world_size = world_size
        self.rank = rank

        self.weight = nn.Parameter(
            torch.randn(self.output_size_per_partition, self.input_size)
        )

        self.bias = nn.Parameter(
            torch.randn(self.output_size_per_partition)
        )

    def forward(self, x):

        output_partition = nn.functional.linear(x, self.weight, self.bias)

        return output_partition
```

总结来说，张量并行是一种有效的技术，用于将单个大层的计算和内存分散到多个设备上。虽然它带来了较大的通信开销，但它常常是一个重要的构成部分，与数据并行和流水线并行一起，用于训练突破单加速器能力限制的先进大型语言模型。Megatron-LM和DeepSpeed等框架大大简化了复杂性，提供了优化的构建模块，以有效实现张量并行。

获取即时帮助、个性化解释和交互式代码示例。

---

### 流水线并行 (PP)

# 流水线并行 (PP)

**流水线并行 (PP)** 通过垂直划分*整个模型*来分担大型语言模型的计算负荷。这种方法将连续的层分配给不同设备，形成类似装配线的处理流水线。与复制模型的数据并行或拆分层内单一操作的张量并行不同，流水线并行提供了一种独特的模型分发策略。

设想一个包含许多层的大型Transformer模型。与试图将所有层放入一个设备或在层内拆分复杂的矩阵乘法不同，流水线并行将例如第1-12层分配给GPU 0，第13-24层分配给GPU 1，第25-36层分配给GPU 2，以此类推。在单个设备上执行的每组层被称为一个**阶段**或**分区**。

### 流水线处理的机制

在流水线并行设置中，数据批次首先被分解成更小的**微批次**。这对于使流水线阶段有效发挥作用很重要，我们很快就会看到这一点。

此过程如下进行：

1. **前向传播：** 微批次1进入阶段0（GPU 0）。在阶段0的层处理完毕后，产生的激活值被发送到阶段1（GPU 1）。阶段1处理激活值并将其输出发送到阶段2，这个过程一直持续到最后一个阶段计算出微批次1的输出。重要的是，一旦阶段0完成微批次1的处理，它就可以立即开始处理微批次2。
2. **反向传播 (backpropagation)：** 一旦为一个微批次计算出损失（在它通过所有阶段后），反向传播就开始了。梯度从最后一个阶段开始计算并反向传播。阶段N计算梯度并将所需的梯度信息发送回阶段N-1。阶段N-1使用这些信息，计算自己的梯度，然后将梯度发送回阶段N-2，以此类推，直到梯度到达阶段0。

这种流程使得不同设备可以同时处理不同的微批次，实现了模型深度方向的并行计算。

G

cluster₀

GPU 0 (阶段 0)

cluster₁

GPU 1 (阶段 1)

cluster₂

GPU 2 (阶段 2)

cluster₃

GPU 3 (阶段 3)

layers0

第1-12层

layers1

第13-24层

layers0->layers1

前向 (激活值)

layers1->layers0

反向 (梯度)

layers2

第25-36层

layers1->layers2

前向 (激活值)

layers2->layers1

反向 (梯度)

layers3

第37-48层

layers2->layers3

前向 (激活值)

layers3->layers2

反向 (梯度)

Output

输出/损失

layers3->Output

前向

Input

微批次

Input->layers0

前向

Output->layers3

反向 (梯度)

> 一个四阶段流水线，显示了GPU上的前向（fwd）激活流和反向（bwd）梯度流。

### 流水线气泡

流水线并行中的一个重要问题是**流水线气泡**或**空闲时间**。在处理批次开始时，只有阶段0是活跃的。阶段1必须等待阶段0完成第一个微批次，阶段2必须等待阶段1，以此类推。同样，在反向传播 (backpropagation)期间，初始阶段会因等待来自后续阶段的梯度而变得空闲。这种启动和结束期间会导致硬件利用率不足。

这个气泡的大小取决于流水线阶段数 (SSS) 和微批次数 (MMM)。如果每个微批次通过一个阶段大约需要相同的时间 (ttt)，那么简单的前向-后向顺序调度，其前向传播的总时间近似为 T≈(S+M−1)×tT \approx (S + M - 1) \times tT≈(S+M−1)×t，反向传播也类似。总有效工作量是 S×M×tS \times M \times tS×M×t。效率（设备忙碌时间所占比例）大约是 SM/(S(S+M−1))SM / (S(S+M-1))SM/(S(S+M−1))，对于较大的 SSS，这可以简化为 M/(M+S−1)M/(M+S-1)M/(M+S−1)。气泡部分（空闲时间）近似为 (S−1)/(M+S−1)(S-1)/(M+S-1)(S−1)/(M+S−1)。

为了减少气泡，我们需要相对于阶段数(SSS)增加微批次数(MMM)。然而，增加MMM意味着更小的微批次，这可能无法充分发挥每个GPU的计算能力，并且还会增加所有正在处理的微批次所需的总激活内存。

### 流水线调度

为了减轻气泡问题，已经开发出各种调度策略，超越了简单的“所有前向，然后所有反向”方法（通常与GPipe相关）。一种常见且有效的策略是\*\*1F1B（一次前向，一次反向）\*\*调度，它通过PipeDream等框架而广为人知。

在1F1B调度中，阶段轮流执行即将到来的微批次的前向传播和已完成微批次的反向传播 (backpropagation)。一旦一个阶段完成微批次 `i` 的前向传播，它可能会立即执行微批次 `i-k` 的反向传播（其中 `k` 与阶段数有关），前提是下一阶段的梯度已可用。这使得设备更忙碌，与简单调度相比，显著减少了空闲时间气泡。

NaiveSchedule

clusterₙaive

简单调度（类似GPipe）

T0

时间 0

T1

1

S0\_T0

前向 0

T2

2

T3

3

T4

4

T5

5

T6

6

T7

7

T8

8

S0ₗabel

GPU 0

S0\_T1

前向 1

S1\_T0

空闲

S0\_T2

前向 2

S0\_T3

前向 3

S0\_T4

空闲

S0\_T5

反向 3

S0\_T6

反向 2

S0\_T7

反向 1

S0\_T8

反向 0

S1ₗabel

GPU 1

S1\_T1

前向 0

S2\_T0

空闲

S1\_T2

前向 1

S1\_T3

前向 2

S1\_T4

前向 3

S1\_T5

空闲

S1\_T6

反向 3

S1\_T7

反向 2

S1\_T8

反向 1

S2ₗabel

GPU 2

S2\_T1

空闲

S2\_T2

前向 0

S2\_T3

前向 1

S2\_T4

前向 2

S2\_T5

前向 3

S2\_T6

空闲

S2\_T7

反向 3

S2\_T8

反向 2

InterleavedSchedule

cluster₁f1b

交错式1F1B调度

T10

时间 0

T11

1

S10\_T0

前向 0

T12

2

T13

3

T14

4

T15

5

T16

6

T17

7

T18

8

S10ₗabel

GPU 0

S10\_T1

前向 1

S11\_T0

空闲

S10\_T2

前向 2

S10\_T3

反向 0

S10\_T4

前向 3

S10\_T5

反向 1

S10\_T6

前向 4

S10\_T7

反向 2

S10\_T8

反向 3

S11ₗabel

GPU 1

S11\_T1

前向 0

S12\_T0

空闲

S11\_T2

前向 1

S11\_T3

反向 0

S11\_T4

前向 2

S11\_T5

反向 1

S11\_T6

前向 3

S11\_T7

反向 2

S11\_T8

前向 4

S12ₗabel

GPU 2

S12\_T1

空闲

S12\_T2

前向 0

S12\_T3

前向 1

S12\_T4

反向 0

S12\_T5

前向 2

S12\_T6

反向 1

S12\_T7

前向 3

S12\_T8

反向 2

> 简单调度与交错式1F1B调度在3个阶段和多个微批次下的GPU时间利用率比较。蓝色（Fwd）代表前向传播，红色（Bwd）代表反向传播，灰色（Idle）代表气泡时间。1F1B显著减少了空闲时间。

### 实现方面

有效实现流水线并行需要仔细考虑几个因素：

- **负载均衡：** 各阶段的计算成本（时间）应大致相等。如果一个阶段耗时比其他阶段长很多，它就会成为瓶颈，其他阶段将空闲等待。由于Transformer模型中注意力块和前馈块的计算差异，平衡层间负载可能不易实现。
- **通信：** 通信发生在相邻阶段之间，主要是前向传输激活值和反向传输梯度。这些传输的大小取决于隐藏维度和序列长度。虽然可能不如张量并行所需的层内通信频繁，但这些传输仍然可能较大。
- **微批次大小：** 如前所述，这个参数 (parameter)(MMM)直接影响气泡大小和内存使用。更大的MMM会减少气泡，但会增加存储所有处理中微批次激活值所需的内存。最佳的MMM取决于模型大小、阶段数和可用设备内存。
- **框架支持：** 实现高效调度、通信和梯度处理是复杂的。DeepSpeed和Megatron-LM等库提供了流水线并行的实现，通常与其他并行策略结合使用。

这是一个高度简化的PyTorch代码片段，用来说明阶段和数据传递的思路（实际实现要复杂得多）：

```python
import torch
import torch.nn as nn

class PipelineStage(nn.Module):
    def __init__(self, layers, stage_id):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.stage_id = stage_id
        self.device = get_device_for_stage(stage_id)
        self.to(self.device)

    def forward(self, x):

        if x is not None:
             x = x.to(self.device)

        for layer in self.layers:
            x = layer(x)
        return x

my_stage_id = get_my_stage_id()
num_stages = get_num_stages()

layers_per_stage = len(model_layers) // num_stages

start_layer = my_stage_id * layers_per_stage
if my_stage_id < num_stages - 1:
    end_layer = (my_stage_id + 1) * layers_per_stage
else:
    end_layer = len(model_layers)
my_layers = model_layers[start_layer:end_layer]

pipeline_module = PipelineStage(my_layers, my_stage_id)

def training_step(micro_batch_data):
    activations = None
    if my_stage_id == 0:
        activations = micro_batch_data
    else:

        activations = recv_tensor(from_stage_id=my_stage_id - 1)

    output_activations = pipeline_module(activations)

    if my_stage_id < num_stages - 1:

        send_tensor(output_activations, to_stage_id=my_stage_id + 1)

    else:

        loss = compute_loss(output_activations, target_labels)

        loss.backward()

        grad_to_send = output_activations.grad

    return loss
```

> 注意：此代码仅为说明用途。实际实现需要复杂的调度逻辑（如1F1B）、处理激活值检查点或重新计算、微批次间的梯度累积以及通信原语。

### 流水线并行的优劣

**优点：**

- **模型深度可扩展性：** 能够训练模型深度超出单个设备内存限制的模型，即使结合张量并行也能实现。
- **内存效率（激活值）：** 与纯数据并行相比，流水线并行有时对激活值更具内存效率，因为每个设备只保存模型自身部分的激活值（尽管此优势很大程度上取决于微批次策略）。
- **通信量减少（潜在）：** 通信只发生在相邻阶段之间，通常涉及激活值或梯度。与数据并行中频繁的AllReduce操作或张量并行中的张量拆分/聚合操作相比，这可能对带宽要求较低，尤其当激活值小于模型参数 (parameter)时。

**缺点：**

- **流水线气泡：** 固有的空闲时间会降低硬件利用率，除非通过大量微批次和复杂调度来缓解。
- **负载均衡敏感性：** 性能高度依赖于各阶段间计算负载的平衡。负载不均的阶段会产生瓶颈。
- **复杂性：** 实现高效调度和管理跨微批次状态会增加显著的复杂性。
- **延迟：** 通过阶段的顺序特性增加了每个微批次处理的延迟。

流水线并行很少单独用于大型模型。相反，它通常与数据并行和张量并行结合，采用混合方法。例如，一个常见的配置是在不同多GPU节点之间使用数据并行，同时在每个节点*内部*使用流水线并行和/或张量并行来管理节点内GPU上的模型大小。这允许同时扩展批次大小（通过数据并行）和模型大小（通过流水线并行/张量并行）。

获取即时帮助、个性化解释和交互式代码示例。

---

### 混合并行策略 (DP+TP, DP+PP等)

# 混合并行策略 (DP+TP, DP+PP等)

数据并行（DP）、张量并行（TP）和流水线并行（PP）等方法各自在分布式训练中具备独特的优点。为了提升模型规模上限，通常需要结合使用这些技术。没有单一策略是普遍最佳的；适合的方法取决于具体的模型架构、硬件限制（GPU内存、互联带宽/延迟）和期望的吞吐量 (throughput)。混合方法让工程师能够更好地安排更精细的工作分配，平衡内存节省、计算效率和通信开销。

### 组合并行策略

混合方法的主要思路是同时运用多种并行方式的优点。一种常见做法是使用TP或PP来让模型能够 *容纳* 在一组设备上（处理内存限制），然后使用DP在这些集合的副本之间扩展训练吞吐量 (throughput)。

#### 数据并行 + 张量并行 (DP+TP)

这是一种常用的组合，尤其适合具有非常宽层级的模型，TP在这种情况下能大幅节省内存。

- **工作原理：** 模型在某些层中的参数 (parameter)（例如注意力或MLP块中的大型权重 (weight)矩阵）通过TP分割到一组设备上（例如2向、4向或8向TP）。整个TP组，共同拥有模型的一个分片实例，表现为一个逻辑设备。然后数据并行应用于此TP组的多个副本。每个副本处理全局数据批次中的不同分片。
- **通讯：** 这涉及两类通讯：
  1. **组内（TP）：** 在每个TP组内进行频繁、低延迟通讯（AllReduce、Scatter、Gather），以处理前向和后向传播期间的分片张量操作。这通常需要在节点内部使用NVLink等高带宽、低延迟互联。
  2. **组间（DP）：** 频率较低的AllReduce通讯在DP副本间进行（具体是指不同TP组中对应设备间的通讯），以在反向传播 (backpropagation)后同步梯度。这通常可以接受节点之间（如InfiniBand或以太网）更高延迟的互联。

G

cluster\_dp0

DP 0级 (TP组)

cluster\_dp1

DP 1级 (TP组)

gpu0₀

GPU 0 | TP 0级

gpu0₁

GPU 1 | TP 1级

gpu0₀->gpu0₁

TP通讯

gpu1₀

GPU 2 | TP 0级

gpu0₀->gpu1₀

DP通讯 (AllReduce)

gpu1₁

GPU 3 | TP 1级

gpu0₁->gpu1₁

DP通讯 (AllReduce)

gpu1₀->gpu1₁

TP通讯

> 一个2向TP x 2向DP配置。GPU 0和1构成一个TP组（DP 0级），GPU 2和3构成另一个（DP 1级）。TP通讯发生在组内（蓝色虚线），DP梯度同步发生在对应的TP等级间（红色实线）。

像NVIDIA的Megatron-LM这样的框架尤其适合用于实现TP，并提供将其与用于DP部分的标准PyTorch DistributedDataParallel (DDP) 集成的方法。

#### 数据并行 + 流水线并行 (DP+PP)

这种组合对深度模型很有效，其中PP用于降低激活所需的峰值内存，而DP则提升吞吐量。

- **工作原理：** 模型被划分为多个阶段，每个阶段分配给单个或一组设备（PP）。该完整流水线的多个实例被创建，构成DP副本。每个流水线副本处理全局数据批次中的不同组微批次。
- **通讯：**
  1. **流水线内（PP）：** 相邻流水线阶段之间的点对点通讯，用于前向传递激活，后向传递梯度。这通常涉及激活张量的发送/接收。
  2. **流水线间（DP）：** 跨DP副本的AllReduce通讯（具体是指不同副本中持有相同流水线阶段的设备之间），以同步每个阶段内各层的梯度。
- **气泡消除：** DP有助于减少PP固有的流水线气泡（空闲时间）。当一个流水线副本可能部分空闲等待依赖时，其他副本可以积极计算各自的微批次，从而提升整体硬件利用率。

G

cluster\_dp0

DP 0级 (流水线)

cluster\_dp1

DP 1级 (流水线)

gpu0ₛ0

GPU 0

阶段0

gpu0ₛ1

GPU 1

阶段1

gpu0ₛ0->gpu0ₛ1

PP通讯 (激活)

gpu1ₛ0

GPU 2

阶段0

gpu0ₛ0->gpu1ₛ0

DP通讯 (AllReduce)

gpu1ₛ1

GPU 3

阶段1

gpu0ₛ1->gpu1ₛ1

DP通讯 (AllReduce)

gpu1ₛ0->gpu1ₛ1

PP通讯 (激活)

> 一个2阶段PP x 2向DP配置。GPU 0和1构成一个流水线（DP 0级），GPU 2和3构成另一个（DP 1级）。PP通讯发生在阶段之间（绿线）。DP梯度同步发生在对应的阶段之间（红色虚线）。

DeepSpeed之类的库提供复杂的流水线并行实现，可以轻松地与其基于ZeRO的数据并行结合使用。

#### 张量并行 + 流水线并行 (TP+PP) 和 “3D” 并行 (DP+TP+PP)

对于最大的模型，参数量超过数千亿或万亿时，可能需要将所有三种主要策略结合起来。这通常称之为“3D”并行。

- **工作原理：**
  - **TP+PP：** 流水线中的每个阶段可能对单个设备来说过大，因此使用TP在多个设备上并行化该阶段 *内部* 的层。这构成一个“阶段组”。通讯通过TP在阶段组 *内部* 进行，并通过PP在阶段组 *之间* 进行。
  - **DP+TP+PP：** 在此之上添加数据并行。创建TP+PP配置的多个副本，每个副本处理不同的数据分片。
- **复杂程度：** 这种配置在管理设备映射、通讯调度和潜在的负载均衡问题时带来显著复杂性。通讯模式变得复杂，涉及阶段组内的TP集合操作、阶段组间的PP点对点通讯，以及跨副本的DP AllReduce。
- **应用场景：** 适用于同时满足以下条件下的模型：单个层很大（需要TP）、模型深度很深（需要PP），并且需要高吞吐量（需要DP）。

#### ZeRO (Zero Redundancy Optimizer) 优化

ZeRO，特别是ZeRO Stage 3，并非与DP、TP和PP相同的并行维度，而是一种优化数据并行内存使用的技术。它将优化器状态、梯度以及（可选地）参数本身分布到数据并行等级上。ZeRO几乎总是与其他策略 *配合使用*：

- **ZeRO-DP + TP：** ZeRO降低DP的内存负担，使TP能够专注于在必要时分割模型参数和激活。这种组合使得每个节点能够容纳更大的模型，并扩展更宽的模型。
- **ZeRO-DP + PP：** ZeRO降低由DP组管理的每个流水线阶段内权重和优化器状态的内存占用，补充了PP带来的激活内存节省。这对深度模型有效。

DeepSpeed是用于实现ZeRO的具有代表性的框架，并提供集成功能，能有效结合TP和PP（通常借助Megatron-LM的TP实现）。

### 实现考量

选择并应用合适的混合策略需要仔细分析：

- **模型结构：** 宽模型更适合TP；深度模型更适合PP。
- **硬件：** TP需要低延迟的节点内连接（NVLink）。PP性能很大程度上取决于节点间带宽和流水线调度隐藏通讯延迟的能力。DP的扩展性取决于AllReduce集合操作的性能。
- **内存、计算与通讯：** 每种策略都会改变瓶颈。TP节省内存但增加层内通讯。PP节省激活内存但引入气泡和层间通讯。DP提高计算吞吐量 (throughput)但需要副本的内存（由ZeRO减轻），并增加梯度通讯。
- **框架支持：** 从零开始实现这些复杂策略非常困难。依赖DeepSpeed和Megatron-LM等框架几乎总是必须的。这些框架提供抽象和优化的通讯集合操作。

一个PyTorch代码片段，展示如何组合这些（使用类似于DeepSpeed或Megatron-LM中的高级API），可能如下所示：

```python
import torch
import torch.distributed as dist
from some_framework import (
    initialize_parallelism,
    get_data_parallel_group,
    get_tensor_parallel_group,
    get_pipeline_parallel_group,
    PipelineModule,
    TensorParallelLinear,
    ZeROOptimizer
)

initialize_parallelism(
    data_parallel_size=2,
    tensor_parallel_size=4,
    pipeline_parallel_size=2
)

class Stage0(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = TPInputEmbedding(...)

        self.layer1 = TPLayer(...)
        self.layer2 = TPLayer(...)

    def forward(self, x):

        return self.layer2(self.layer1(self.embedding(x)))

class Stage1(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer3 = TPLayer(...)

        self.output = TPOutputLayer(...)

    def forward(self, x):

        return self.output(self.layer3(x))

model = PipelineModule(
    stages=[Stage0(), Stage1()],
    num_microbatches=8
)

optimizer = ZeROOptimizer(
    model.parameters(),
    lr=1e-4,

)

for data in dataloader:
    optimizer.zero_grad()

    loss = model(data)
    optimizer.step()
```

> PyTorch代码，呈现如何使用张量并行层（`TPLayer`、`TPInputEmbedding`）定义模块并组合成一个`PipelineModule`。`ZeROOptimizer`隐式处理数据并行维度上的梯度同步。

成功训练大型模型通常需要对不同的混合配置（例如，改变TP大小、PP阶段、微批次大小）进行迭代实验，以找到最大化硬件利用率和最小化给定模型和集群架构训练时间的最佳平衡点。因此，理解这些策略之间的关联对于任何从事大规模模型训练的工程师都非常重要。

获取即时帮助、个性化解释和交互式代码示例。

---

### 通信开销分析

# 通信开销分析

将计算负担和模型参数 (parameter)分布到多个设备上对于训练大型模型来说是必要的，但这会带来一个重要的性能考量：通信开销。每当数据需要在设备间交换时，无论是梯度、激活值还是权重 (weight)分片，时间都会花在通信上而不是计算上。将这种开销降到最低对于实现高效扩展和缩短训练时间来说十分重要。

本节分析了我们讨论过的不同并行策略所关联的通信模式和成本。了解这些成本有助于为给定的模型架构和硬件配置选择最合适的策略或策略组合。

### 分布式训练中的通信基本操作

分布式训练依赖于通信集合操作，这些操作涉及多个进程协同交换数据。LLM训练中最常用的基本操作包括：

1. **广播 (`broadcast`)**: 将数据从一个进程发送到所有其他进程。
2. **规约 (`reduce`)**: 使用指定的运算（如求和、求平均）将所有进程的数据合并到一个进程上。
3. **All-Reduce (`all_reduce`)**: 合并所有进程的数据，并将结果分发回所有进程。这实际上是先进行规约再进行广播的操作。它在数据并行中大量用于同步梯度。
4. **分散 (`scatter`)**: 将数据块从一个进程分散到所有其他进程。
5. **收集 (`gather`)**: 将所有进程的数据块收集到一个进程上。
6. **All-Gather (`all_gather`)**: 收集所有进程的数据块，并将完整的、拼接后的数据分发回所有进程。用于某些张量并行实现。
7. **规约-分散 (`reduce_scatter`)**: 使用规约运算合并所有进程的数据，然后分散结果，使得每个进程接收最终规约张量的一个数据块。也用于张量并行。
8. **点对点 (`send`/`recv`)**: 直接通信，一个进程向另一个特定进程发送数据，后者接收数据。这是流水线并行的主要机制。

在PyTorch中，这些操作通常通过`torch.distributed`包来访问。例如，对一个组中所有进程的张量`t`执行All-Reduce操作可能看起来像这样：

```python
import torch
import torch.distributed as dist

dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
```

流水线并行的点对点通信涉及发送和接收进程对：

```python

if rank == i:

    dist.send(tensor=activations, dst=i+1)

elif rank == i+1:

    received_activations = torch.zeros_like(expected_activations_shape)
    dist.recv(tensor=received_activations, src=i)
```

### 影响通信成本的因素

通信所需的时间取决于几个因素：

- **延迟 (α\alphaα)**: 启动任何通信所需的固定启动时间，与数据大小无关。这通常由网络协议开销和设备间的物理距离/跳数决定。
- **带宽 (β\betaβ)**: 数据可以通过网络链路传输的速率，通常以吉比特每秒（Gbps）或吉字节每秒（GB/s）衡量。数据传输时间与带宽成反比。
- **消息大小 (MMM)**: 正在传输的数据量。较大的消息需要更长时间，主要受带宽影响。
- **设备数量 (PPP)**: 许多集合操作的时间复杂度随参与设备的数量而变化。例如，简单的All-Reduce可能呈线性扩展，而像环形All-Reduce这样的优化算法则呈亚线性扩展。
- **网络拓扑和硬件**: 网络的物理布局（例如，胖树、环面）和特定的互连技术（例如，以太网、InfiniBand、NVIDIA NVLink/NVSwitch）对可实现的带宽和延迟有显著影响，特别是对于节点间通信。NVLink为单个节点内的GPU到GPU通信提供高带宽。
- **集合算法**: 用于实现集合操作的特定算法（例如，All-Reduce的环形、树形、蝴蝶算法）可以根据消息大小和网络拓扑表现出不同的性能特点。像NVIDIA集合通信库（NCCL）这样的库为NVIDIA GPU实现了这些算法的高度优化版本。

一个常见且简单的用于表示大小为 MMM 的单个消息的通信时间 TTT 的模型是**alpha-beta模型**：

T≈α+MβeffT \approx \alpha + \frac{M}{\beta\_{eff}}T≈α+βeff​M​

这里，α\alphaα 代表延迟部分，βeff\beta\_{eff}βeff​ 是传输时实现的有效带宽。对于涉及 PPP 个设备的集合操作，模型会变得更复杂，通常会涉及与 PPP 的对数或线性相关的项，具体取决于所用算法。

### 并行策略的通信模式

让我们分析每种主要策略固有的通信成本：

1. **数据并行 (DP)**:

   - **主要通信**: 反向传播 (backpropagation)后执行`all_reduce`操作，以聚合所有 PPP 个副本的梯度。
   - **消息大小**: 整个模型梯度的总大小 (MmodelM\_{model}Mmodel​)。
   - **频率**: 每个训练步骤一次（如果使用梯度累积，则每个累积步骤一次）。
   - **开销**: 主要由All-Reduce的成本决定。时间通常随模型大小 MmodelM\_{model}Mmodel​ 变化，并与设备数量 PPP 呈对数或线性关系，具体取决于All-Reduce算法和网络。对设备间带宽高度敏感。对于大型模型和大量设备，此All-Reduce可能成为一个重要的瓶颈。梯度累积通过降低这种昂贵操作的频率来帮助。
2. **张量并行 (TP)**:

   - **主要通信**: 涉及在特定层（例如MLP或注意力块）的前向和反向传播*内部*的`all_reduce`、`all_gather`或`reduce_scatter`操作，这些层在设备间进行拆分。
   - **消息大小**: 对应层内激活值或梯度张量一部分的较小消息。大小取决于隐藏维度、序列长度和张量并行设备数量。
   - **频率**: 每层多次，在前向和反向传播期间。比数据并行的通信频率高得多。
   - **开销**: 通信频繁发生，通常涉及较小的消息。这使得张量并行可能对延迟(α\alphaα)和带宽(β\betaβ)都敏感。它极大地受益于高带宽、低延迟的节点内互连，如NVLink，因为张量并行通常首先在节点内部应用。由于其频率高，总通信量可能很大。

digraph G {
rankdir=LR;

```
// 设置全局字体大小
fontsize=12;

// 节点设置，带有明确的字体大小
node [shape=circle, style=filled, color="#a5d8ff", fontname="sans-serif", fontsize=12];

// 边设置，带有明确的字体大小
edge [color="#495057", fontname="sans-serif", fontsize=12];

GPU0 -> GPU1 [label="前向/反向"];
GPU1 -> GPU2 [label="前向/反向"];
GPU2 -> GPU3 [label="前向/反向"];
GPU3 -> GPU0 [label="前向/反向"];
```

}
```
> 4-GPU环形All-Reduce中数据流的简化视图，常用于数据并行或张量并行集合操作。每个GPU都从其相邻设备发送和接收数据块。

3. **流水线并行 (PP)**:
   - **主要通信**: 相邻流水线阶段之间的点对点`send`/`recv`操作。阶段 iii 在前向传播期间将其输出激活值发送到阶段 i+1i+1i+1，而阶段 i+1i+1i+1 在反向传播期间将激活值的梯度发送回阶段 iii。
   - **消息大小**: 阶段边界处的激活张量（或梯度张量）的大小。大小取决于批量大小、序列长度和隐藏维度。
   - **频率**: 每个微批量在每个阶段边界处一次（前向和反向）。比张量并行频率低，但比数据并行（不含梯度累积）频率高。
   - **开销**: 主要为点对点通信。托管相邻阶段的设备之间的延迟和带宽十分重要。流水线并行效率低下的一个主要原因不只是通信时间本身，而是“流水线气泡”的可能性——当阶段等待来自前一阶段的数据时，出现的空闲时间。微批量处理有助于减小气泡大小，但会增加通信频率和相关的延迟开销。

### 通信成本比较

| 策略 | 主要操作 | 消息大小 | 频率 | 敏感度 | 瓶颈 |
| --- | --- | --- | --- | --- | --- |
| 数据并行 | `all_reduce` | 模型梯度 (大) | 每 (累积) 步一次 | 带宽 | All-Reduce时间 |
| 张量并行 | `all_reduce`、`all_gather`、`reduce_scatter` | 层激活值/梯度 (小/中) | 每层多次 | 延迟、带宽 | 频繁的集合调用 |
| 流水线并行 | `send`/`recv` | 边界激活值/梯度 (中/大) | 每个微批量/阶段一次 | 延迟、带宽 | 流水线气泡、阶段间通信 |

*表：不同并行策略通信特点的定性比较。*

混合方法将这些策略结合起来，带来更复杂的通信模式。例如，将数据并行与张量并行结合使用意味着每个数据并行组（应用了张量并行的地方）都会执行梯度的All-Reduce操作。同时使用张量并行和流水线并行则涉及阶段内张量并行通信和阶段间流水线并行通信。

### 通信性能分析

虽然理论分析提供了直觉，但特定训练运行中的实际通信开销在很大程度上取决于实现细节、硬件、网络配置和软件堆栈（例如PyTorch版本、NCCL版本）。因此，**性能分析**是必要的。`torch.profiler`、NVIDIA Nsight Systems (`nsys`) 或框架特有的日志记录等工具可以帮助衡量在不同通信操作（`nccl:all_reduce`、`nccl:send`等）与计算内核上花费的时间。分析这些性能文件对于找出瓶颈并优化分布式训练配置十分重要。

```python

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with torch.profiler.record_function("model_training_step"):

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

"通过了解与每种并行策略相关的基本通信模式和成本，并使用性能分析工具来衡量表现，您可以做出明智的决定，了解如何最好地分配您的LLM训练工作负载，以最大程度地提高效率并缩短训练时间。"

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 16 Implementing Distributed Training Frameworks

### 分布式训练库概述

# 分布式训练库概述

分布式训练的基本策略，例如数据并行、张量并行和流水线并行，对于处理大型模型至关重要。然而，在数百甚至数千个处理单元上手动实现和协调这些策略是一个严峻的工程难题。PyTorch 等标准深度学习 (deep learning)框架提供了基本构建块（例如 `torch.distributed`），但要协调复杂的混合并行策略、高效管理通信以及优化数十亿参数 (parameter)模型的内存使用，则需要更专业的工具。

这就是专用分布式训练库派上用场的地方。这些库基于 PyTorch 等框架提供的原始功能构建，提供更高级的抽象和为大规模模型训练优化的实现。它们旨在减少所需的样板代码和专门的工程工作，让您可以更专注于模型架构和训练过程。

可以将这些库看作是为分配模型和数据提供了精密的引擎。您无需手动管理通信集合操作（如 `all_reduce`、`scatter`、`gather`），也无需弄清楚如何跨设备划分模型权重 (weight)和激活，这些库提供了 API 和配置选项来处理这些任务。

### 分布式训练库

为了应对大规模训练的挑战，一些库已经出现。本章我们将主要关注两个有影响力且被广泛采用的框架：DeepSpeed 和 Megatron-LM。不过，了解更广泛的体系也是有益的。

- **PyTorch DistributedDataParallel (DDP)**: 这是 PyTorch 中用于数据并行的标准模块。它在每个 GPU 上复制模型，在每个 GPU 上处理不同的数据分片，并使用 `all_reduce` 等高效通信来同步梯度。尽管对于能放入单个 GPU 的模型有效，但它本身并不能解决单 GPU 无法容纳的超大型模型的内存限制，在这种情况下，即使单个副本的权重 (weight)、梯度和优化器状态也会超出设备内存。
- **PyTorch FullyShardedDataParallel (FSDP)**: FSDP 直接集成到 PyTorch 中，提供与 DeepSpeed ZeRO 阶段 3 类似的功能。它将模型参数 (parameter)、梯度和优化器状态分片到数据并行工作器中，从而大幅降低每个 GPU 的峰值内存需求。它代表了 PyTorch 原生用于大型模型数据并行而非简单复制的解决方案。
- **DeepSpeed**: DeepSpeed 由微软研究院开发，提供一套优化技术，主要用于高效、可扩展地训练大型模型，尤其是在内存使用方面。其最知名的功能是零冗余优化器 (ZeRO)，它将训练状态的不同组件（优化器状态、梯度、参数）划分到数据并行进程中。这使得训练单 GPU 无法容纳的超大型模型成为可能，即使仅使用数据并行。DeepSpeed 还集成了对流水线并行的支持，并集成了优化过的内核。它通常只需对现有 PyTorch 训练脚本进行最少的代码更改。

  ```python

  import deepspeed
  import torch.nn as nn
  from transformers import AutoModelForCausalLM

  model = AutoModelForCausalLM.from_pretrained("gpt2")

  config_params = {
      "train_batch_size": 16,
      "gradient_accumulation_steps": 1,
      "optimizer": {
          "type": "AdamW",
          "params": {
              "lr": 1e-5
          }
      },
      "fp16": {
          "enabled": True
      },
      "zero_optimization": {
          "stage": 2
      }
  }

  model_engine, optimizer, _, _ = deepspeed.initialize(
      model=model,

      config_params=config_params

  )
  ```
- **Megatron-LM**: Megatron-LM 由 NVIDIA 开发，主要侧重于实现高计算吞吐量 (throughput)，主要通过张量并行（将单个层拆分到多个 GPU 上）和流水线并行（将层分阶段部署到多个 GPU 上）的高度优化实现。它通常需要根据其并行模式调整模型代码结构，但为特定类型的并行提供了先进的性能，尤其是在 NVIDIA 硬件上。
- **Hugging Face Accelerate**: 该库充当一个更高级的抽象层，旨在简化 PyTorch 训练脚本在各种硬件设置（单个 GPU、多个 GPU、TPU）和分布式策略上的运行。它可以与 DeepSpeed 或 PyTorch FSDP 等底层库集成，提供一个统一的接口，以更少的代码修改来配置分布式训练。

这些库的选择通常取决于项目的具体需求：模型大小、核心瓶颈（内存还是计算）、期望的并行策略（DP、TP、PP 或混合）、目标硬件以及所需的代码集成工作量。

G

cluster\_focus

主要侧重

clusterₗibs

库

Memory

内存效率

Compute

计算吞吐量

DeepSpeed

DeepSpeed

DeepSpeed->Memory

ZeRO

MegatronLM

Megatron-LM

MegatronLM->Compute

TP / PP 优化

FSDP

PyTorch FSDP

FSDP->Memory

分片

Accelerate

HF Accelerate

Accelerate->DeepSpeed

集成

Accelerate->MegatronLM

集成

Accelerate->FSDP

集成

> 主要分布式训练库及其主要优化侧重的高级概述。HF Accelerate 充当统一层。

DeepSpeed 和 Megatron-LM 虽然都功能强大，但解决问题的方式角度略有不同。DeepSpeed 擅长通过 ZeRO 减轻内存压力，使得训练超大型模型成为可能。Megatron-LM 为张量并行和流水线并行提供高度优化的内核，最大限度地提高计算速度。目前，人们正努力结合这些库的优势，例如将 DeepSpeed 的 ZeRO 与 Megatron-LM 的张量并行结合使用。

以下章节将提供关于配置和使用 DeepSpeed 的 ZeRO 优化以及 Megatron-LM 的张量和流水线并行功能的实用指导，使您能够从理论理解转向功能性分布式训练设置。

获取即时帮助、个性化解释和交互式代码示例。

---

### DeepSpeed 介绍

# DeepSpeed 介绍

要在大规模模型上实现数据并行（DP）、张量并行（TP）和流水线并行（PP）等分布式训练策略，需要用到实用的工具。虽然PyTorch等框架提供了基本的分布式数据并行（`DistributedDataParallel`），但训练拥有数千亿甚至数万亿参数 (parameter)的模型，需要更精密的内存管理和扩展能力。为此，专门的库变得不可或缺。

DeepSpeed由微软研究院开发，是一个开源的深度学习 (deep learning)优化库，旨在显著提高大型模型训练的速度和规模，同时最大限度地减少所需的硬件资源。它与PyTorch结合使用，提供了一系列优化功能，解决了大型训练中遇到的主要瓶颈，特别是GPU内存限制。

标准数据并行中的根本问题是内存冗余。参与DP的每个GPU通常都包含模型权重 (weight)、梯度和优化器状态（如Adam中的动量和方差缓冲区）的完整副本。对于拥有数十亿参数的模型，仅优化器状态所需的内存（通常以32位精度（FP32）存储）就可以超过即使是高端GPU的可用内存。考虑一个拥有 111 亿参数的模型。权重可能需要 444 GB（FP32）或 222 GB（FP16）。然而，Adam优化器状态通常每个参数需要额外 161616 字节（动量 444 字节，方差 444 字节，FP32主权重 444 字节，如果使用FP16，梯度 444 字节），这意味着 *每个GPU* 额外增加 161616 GB。这很快就变得难以维持。

DeepSpeed通过几种创新的技术来应对这一问题及其他扩展难题：

1. **ZeRO（零冗余优化器）：** 这可以说是DeepSpeed最受认可的贡献。ZeRO是一系列优化方法，旨在消除数据并行训练中的内存冗余。ZeRO不是在所有数据并行GPU上复制优化器状态、梯度，甚至可能模型权重，而是将这些状态分区到可用设备上。这意味着每个GPU只保留整体状态的一部分，从而大幅减少了每个设备的内存占用。我们将在下一节中查看ZeRO的不同阶段（阶段1、阶段2和阶段3）。
2. **内存卸载：** DeepSpeed允许将训练状态的部分（优化器状态、激活或参数）从GPU内存卸载到主机CPU的主内存甚至NVMe固态硬盘。虽然访问CPU内存或NVMe比GPU HBM慢，但卸载不常访问的状态可以释放宝贵的GPU内存，从而能够训练比原本能容纳的更大的模型。
3. **高效流水线并行：** 除了用于数据并行的ZeRO之外，DeepSpeed还包含其自身高度优化的流水线并行实现。这使用户能够将模型的层分区到多个GPU上，减少每个GPU激活所需的内存，并通过分散计算图来支持更大的模型。
4. **自定义内核和优化器：** DeepSpeed通常包含高度优化的CUDA内核，用于大型模型中常见的操作，例如自定义Transformer层或像FusedAdam这样高效的优化器，它们将多个步骤组合到一个内核启动中以获得更好的性能。
5. **简化的混合精度训练：** 它提供了易于使用的工具，用于管理FP16或BF16混合精度训练，包括自动损失缩放。

DeepSpeed的设计目标之一是易于集成。对于许多常见的使用场景，特别是使用ZeRO时，对现有PyTorch训练脚本只需进行最小的改动。核心修改通常包括使用DeepSpeed的`initialize`函数封装模型、优化器和数据加载器。

以下是DeepSpeed集成的示意图：

```python
import torch
import deepspeed

config_params = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-5
        }
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2
    }

}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config_params=config_params
)

for batch in dataloader:
    inputs, labels = batch
    inputs = inputs.to(model_engine.local_rank)
    labels = labels.to(model_engine.local_rank)

    outputs = model_engine(inputs)
    loss = calculate_loss(outputs, labels)

    model_engine.backward(loss)
    model_engine.step()
```

在此代码片段中，`deepspeed.initialize`接受标准PyTorch模型和优化器，以及一个配置字典（`config_params`），并返回一个`model_engine`。这个引擎在训练循环中取代了原始模型，并且它的方法（`backward`、`step`）根据提供的配置处理分布式训练、梯度累积、混合精度和ZeRO优化的复杂性。

DeepSpeed提供了一整套工具，旨在使大型语言模型的训练更易于进行且更高效。它通过ZeRO专注于内存优化，结合对卸载和流水线并行的支持，使其成为工程师扩展模型规模的有力选择。我们现在将更仔细地查看ZeRO优化器的不同阶段。

获取即时帮助、个性化解释和交互式代码示例。

---

### 使用 DeepSpeed ZeRO 优化

# 使用 DeepSpeed ZeRO 优化

虽然标准数据并行 (DP) 通过在每个设备上复制模型和处理不同的数据分片来在多个 GPU 上扩展训练，但它很快就会遇到内存限制。每个 GPU 仍需保存模型参数 (parameter)、梯度和优化器状态的完整副本。对于拥有数十亿参数的模型，这种重复的状态会消耗大量 GPU 内存，经常超出高端加速器的容量。DeepSpeed 提供的零冗余优化器 (ZeRO) 在这方面显示出很大优势。ZeRO 旨在消除标准 DP 中固有的内存冗余，让您可以在相同的硬件限制下训练更大的模型或使用更大的批次大小。

ZeRO 通过将模型状态（优化器状态、梯度和参数）分配到数据并行进程中而不是复制它们来实现这一点。它分为三个渐进阶段，每个阶段都能节省更多内存，但可能会增加通信开销。

### 了解数据并行中的内存冗余

在分析 ZeRO 之前，让我们直观地看看标准 DP 中每个 GPU 的内存消耗。对于一个拥有 Ψ\PsiΨ 参数 (parameter)的模型，使用像 Adam 这样的标准优化器（它通常存储参数、梯度、一阶动量和二阶方差），每个 GPU 大致所需的内存是：

- **参数:** Ψ×(参数类型的大小，例如 FP32 为 4 字节)\Psi \times (\text{参数类型的大小，例如 FP32 为 4 字节})Ψ×(参数类型的大小，例如 FP32 为 4 字节)
- **梯度:** Ψ×(梯度类型的大小，例如 FP32 为 4 字节)\Psi \times (\text{梯度类型的大小，例如 FP32 为 4 字节})Ψ×(梯度类型的大小，例如 FP32 为 4 字节)
- **优化器状态:** 对于 Adam（动量和方差），通常是 2×Ψ×(优化器状态类型的大小)2 \times \Psi \times (\text{优化器状态类型的大小})2×Ψ×(优化器状态类型的大小)。如果使用混合精度，优化器可能还会存储参数的 FP32 副本。

激活值也消耗内存，但模型状态通常是大型模型的主要影响因素。在标准 DP 中，*所有*这些状态都会在*每个* GPU 上复制。ZeRO 系统地减少了这些复制。

G

cluster\_dp

标准数据并行（每个 GPU）

gpu

GPU

参数

梯度

优化器状态

> 标准数据并行中，每个 GPU 上复制的内存组件。

### ZeRO 阶段 1: 优化器状态划分

ZeRO 阶段 1 处理第一层冗余：优化器状态。例如，Adam 维护动量和方差缓冲区，它们的大小通常与模型参数 (parameter)本身相同（如果为混合精度训练存储 FP32 副本，甚至更大）。阶段 1 将这些优化器状态分配到可用的数据并行 GPU 上。每个 GPU 只保存对应其数据并行等级的优化器状态总量的部分。

在优化器步进 (`optimizer.step()`) 期间，梯度仍然需要在所有 GPU 之间归约（像标准 DP 一样），但每个 GPU 只更新其持有相应优化器状态分区的参数部分。

**优点:**

- 内存消耗明显减少，特别是使用 Adam/AdamW 等优化器时。
- 通信开销与标准 DP 相似（梯度归约）。

**配置示例 (DeepSpeed JSON):**

```json
{
  "zero_optimization": {
    "stage": 1
  },
  "fp16": {
    "enabled": true
  },
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1
}
```

### ZeRO 阶段 2: 优化器状态和梯度划分

ZeRO 阶段 2 更进一步，将优化器状态*和*梯度都划分到数据并行 GPU 上。在反向传播 (backpropagation)期间，阶段 2 使用 ReduceScatter 操作，而不是使用 AllReduce 操作来汇总所有 GPU 上的梯度。此操作计算总和，但立即分散结果，因此每个 GPU 只接收对应其优化器状态分区的梯度分区。

**优点:**

- 通过消除重复的梯度，内存消耗比阶段 1 进一步减少。
- 梯度归约通信量与标准 DP 或阶段 1 相比减半（ReduceScatter 对比 AllReduce）。

**配置示例 (DeepSpeed JSON):**

```json
{
  "zero_optimization": {
    "stage": 2
  },
  "fp16": {
    "enabled": true
  },
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1
}
```

### ZeRO 阶段 3: 优化器状态、梯度和参数 (parameter)划分

ZeRO 阶段 3 是最具侵略性的优化，它划分所有三个主要模型状态：优化器状态、梯度*以及*模型参数本身。每个 GPU 在任何给定时间只保存参数的一个分片。

这需要更精密的管理。在前向和反向传播 (backpropagation)期间，每个 GPU 都需要访问给定层计算的完整参数。ZeRO 阶段 3 通过在计算需要时动态地从其他 GPU 收集所需参数分片，并在计算后立即丢弃它们以释放内存来处理此问题。

**优点:**

- 内存节省最大化，允许在给定 GPU 内存预算内训练最大的模型。模型状态所需的内存变得与模型大小无关，仅随 GPU 数量 (NNN) 扩展。每个 GPU 的内存大致与 Ψ/N\Psi / NΨ/N 成正比。
- 有效消除了模型状态的内存冗余。

**缺点:**

- 与阶段 1 和 2 相比，通信量增加，这是由于在前向和反向传播期间频繁收集参数分片。性能高度依赖于 GPU 之间的互连速度（例如 NVLink、InfiniBand）。

**配置示例 (DeepSpeed JSON):**

```json
{
  "zero_optimization": {
    "stage": 3
  },
  "fp16": {
    "enabled": true
  },
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1
}
```

标准 DPZeRO-1ZeRO-2ZeRO-3020406080100

> 相对于标准数据并行，ZeRO 各阶段每个 GPU 模型状态内存的近似减少量。实际节省取决于使用的优化器和精度。

### 在 PyTorch 中集成 ZeRO 与 DeepSpeed

使用 ZeRO 需要用 DeepSpeed 的 `initialize` 函数包装您的模型、优化器以及可能的数据加载器。您将配置详细信息（通常通过 JSON 文件）提供给此函数。

```python
import torch
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    model_parameters=model.parameters(),
    config_params=args.deepspeed_config
)

for step, batch in enumerate(dataloader):

    batch = {
        k: v.to(model_engine.local_rank) for k, v in batch.items()
    }

    outputs = model_engine(**batch)
    loss = outputs.loss

    model_engine.backward(loss)

    model_engine.step()
```

DeepSpeed 根据配置文件中指定的 ZeRO 阶段，处理划分、通信（梯度归约、参数 (parameter)收集）和优化器步进。

### 选择合适的阶段

- **阶段 1:** 如果标准 DP 因优化器状态而内存不足，这是一个不错的起点。提供了显著的节省，同时对通信模式的改变最小。
- **阶段 2:** 比阶段 1 提供更好的内存节省，当梯度消耗大量内存时尤其明显。梯度归约的通信量也更低。通常在内存节省和通信开销之间取得了良好平衡。
- **阶段 3:** 提供最大的内存节省，对于即使使用阶段 2 也无法容纳的超大型模型来说是不可或缺的。然而，它引入了参数 (parameter)收集的显著通信开销，使得高带宽互连（如 NVLink 或 NVSwitch）对于性能来说非常重要。还可以考虑阶段 3 中的 `ZeRO-Offload` 变体，它可以将分区卸载到 CPU RAM 或 NVMe 存储，以支持更大的模型，尽管代价是较慢的访问时间。

通常需要进行实验才能找到适合您的特定模型、硬件配置和性能要求的最佳阶段。从阶段 1 或 2 开始，如果内存限制要求，则转向阶段 3，并仔细监控训练吞吐量 (throughput)以评估通信增加的影响。ZeRO 提供了一套强大的工具来克服大规模模型训练中的内存障碍，使其成为构建最新 LLM 的一项基本技术。

获取即时帮助、个性化解释和交互式代码示例。

---

### Megatron-LM 介绍

# Megatron-LM 介绍

有效扩展大型语言模型面临多项挑战。数据并行是一种用于扩展并发处理数据样本数量的技术。内存优化策略，例如 DeepSpeed 的 ZeRO，有助于在数据并行工作器之间管理大型模型的内存占用。尽管有这些方法，但仍然可能遇到一个根本性的限制：单个模型副本（或其优化期间的状态）对于单个加速设备（GPU/TPU）的内存来说可能仍然过大。此外，即使在最快的可用硬件上，单次前向或反向传播 (backpropagation)中的计算也可能太慢。

这正是模型并行变得极其重要的地方，NVIDIA 的 Megatron-LM 库提供了一个高度优化的框架，专门用于为大型 Transformer 模型实现张量并行 (TP) 和流水线并行 (PP)。Megatron-LM 最初是为了训练数十亿参数 (parameter)的语言模型而开发的，它提供了直接应对模型计算和参数在多个设备上分布挑战的构建模块和方法。

### 核心贡献：张量并行与流水线并行

Megatron-LM 的主要优势在于它对张量并行和流水线并行的高效实现。

1. **张量并行（层内模型并行）：** Megatron-LM 能够将单个层，或者更准确地说，这些层*内部*的大型权重 (weight)矩阵，拆分到多个设备上。对于 Transformer 模型来说，这通常针对多头注意力 (multi-head attention) (MHA) 块和多层感知机 (MLP) 块。张量并行不是在一块 GPU 上计算 Y=XAY = XAY=XA，而是可能将矩阵 AAA 按列拆分到 NNN 块 GPU 上（A=[A1,A2,...,AN]A = [A\_1, A\_2, ..., A\_N]A=[A1​,A2​,...,AN​]），在每块 GPU iii 上计算 Yi=XAiY\_i = XA\_iYi​=XAi​，然后聚合结果 Y=[Y1,Y2,...,YN]Y = [Y\_1, Y\_2, ..., Y\_N]Y=[Y1​,Y2​,...,YN​]。类似地，矩阵也可以按行拆分，这需要不同的通信模式（例如，归约部分和）。Megatron-LM 提供了这些拆分计算的优化实现以及必要的通信（例如 `all-gather`、`reduce-scatter`、`all-reduce`），这些通常使用 NVIDIA 的 NCCL 库，适用于 NVLink 等高速互联。

   G

   cluster₀

   GPU 0

   A1

   权重矩阵 A1
   (H x W/2)

   Y1

   Y1

   A1->Y1

   输出 Y1 (B x W/2)

   X

   输入 X (B x H)

   X->A1

   输入 X (B x H)

   A2

   权重矩阵 A2
   (H x W/2)

   X->A2

   输入 X (B x H)

   Y\_combined

   合并输出 Y
   (B x W)

   Y1->Y\_combined

   Y2

   Y2

   A2->Y2

   输出 Y2 (B x W/2)

   Y2->Y\_combined

   Gather

   聚合结果

   Y\_combined->Gather

   通信
   (例如，All-Gather)

   > 一个线性层（Y=XAY=XAY=XA）的列并行张量并行视图。输入 XXX 被广播，权重矩阵 AAA 被按列拆分（A1,A2A\_1, A\_2A1​,A2​）到两块 GPU 上。每块 GPU 计算一个部分结果（Y1,Y2Y\_1, Y\_2Y1​,Y2​），然后通过通信将它们组合起来。

   以 PyTorch 中一个简化的线性层实现为例。Megatron-LM 提供了这些层的版本，它们在内部处理拆分和通信。

   ```python

   import torch
   import torch.nn as nn

   linear_layer = nn.Linear(in_features=1024, out_features=4096)
   input_tensor = torch.randn(32, 1024)
   output = linear_layer(input_tensor)
   ```

   这使得能够训练具有非常大的隐藏维度或注意力头的模型，在这些模型中，即使单个层的权重也可能超过单块 GPU 的内存。
2. **流水线并行（层间模型并行）：** 当模型变得非常深（许多层）时，即使张量并行可能也不够，或者张量并行在太多设备上的通信开销变得过高。流水线并行通过将模型的层按顺序划分成阶段来解决这个问题，将每个阶段分配给不同的 GPU（或 GPU 组）。数据流经这些阶段，就像流水线一样。一个简单的实现会导致显著的空闲时间（“流水线气泡”），因为后面的阶段会等待前面的阶段完成。Megatron-LM 实现了*带有微批处理的流水线*，其中输入小批次被拆分成更小的微批次，然后按顺序送入流水线。这使得不同的阶段能够并发处理不同的微批次，从而大大提高硬件利用率。

   G

   cluster₀

   GPU 0 (阶段 0)

   cluster₁

   GPU 1 (阶段 1)

   cluster₂

   GPU 2 (阶段 2)

   L0

   层 1-8
   (微批次 1)

   L1

   层 1-8
   (微批次 2)

   M0

   层 9-16
   (微批次 1)

   L0->M0

    发送/接收

   L2

   层 1-8
   (微批次 3)

   M1

   层 9-16
   (微批次 2)

   L1->M1

    发送/接收

   M2

   层 9-16
   (微批次 3)

   L2->M2

    发送/接收

   M0->L0

   N0

   层 17-24
   (微批次 1)

   M0->N0

    发送/接收

   M1->L1

   N1

   层 17-24
   (微批次 2)

   M1->N1

    发送/接收

   M2->L2

   N2

   层 17-24
   (微批次 3)

   M2->N2

    发送/接收

   N0->M0

   N1->M1

   N2->M2

   > 简化的流水线并行，包含 3 个阶段和 3 个微批次。每块 GPU 处理一部分层（一个阶段）。微批次流经这些阶段（实线箭头）。虚线箭头表示反向传播 (backpropagation)流程。这种并发减少了空闲时间。

### 使用与集成

使用 Megatron-LM 通常包括：

- **模型定义：** 修改您的 Transformer 模型代码，使用 Megatron-LM 的并行层实现（例如 `ParallelMLP`、`ParallelAttention`），而不是标准的 PyTorch 层。
- **配置：** 通常在启动训练脚本时，通过命令行参数 (parameter)指定张量并行度（`tensor_model_parallel_size`）和流水线并行度（`pipeline_model_parallel_size`）。使用的 GPU 总数将是这些大小与数据并行大小的乘积。
- **初始化：** Megatron-LM 提供了处理这些分布式组件正确初始化的实用程序。
- **训练循环：** 调整训练循环，以处理跨流水线阶段的前向/反向传播 (backpropagation)并管理梯度同步。

Megatron-LM 是一个功能强大的专用工具。它提供了高度优化的内核和通信模式，特别是针对 NVIDIA GPU。虽然它可以独立使用，但也经常与 DeepSpeed 等框架集成。这使得开发者能够将 Megatron-LM 高效的张量并行和流水线并行实现与 DeepSpeed 基于 ZeRO 的数据并行以及其他内存节省功能结合起来，形成一个在极大规模下训练模型的有效组合。接下来的部分会说明如何在 DeepSpeed 和 Megatron-LM 中配置和使用这些特定功能。

获取即时帮助、个性化解释和交互式代码示例。

---

### 配置 Megatron-LM 中的张量和流水线并行

# 配置 Megatron-LM 中的张量和流水线并行

了解张量并行（TP）和流水线并行（PP）的理论很重要，但将它们付诸实践需要专门的工具。Megatron-LM 是 NVIDIA 开发的一个著名库，专门设计用于实现这些复杂的并行策略，尤其适合训练大型 Transformer 模型。它提供并行层的优化实现，并管理所需的复杂通信模式。Megatron-LM 以使用 TP 和 PP 的配置步骤进行展示。

Megatron-LM 中的配置通常通过命令行参数 (parameter)传递给训练脚本（如 `pretrain_gpt.py` 或类似脚本）来处理。我们来看具体如何设置张量并行和流水线并行。

### 张量并行 (TP) 配置

张量并行，有时也称为层内模型并行，涉及将单个大型层（如 MLP 块或注意力机制 (attention mechanism)中的权重 (weight)矩阵）的执行拆分到多个 GPU 上。Megatron-LM 提供专门的层实现（例如 `ColumnParallelLinear`、`RowParallelLinear`），它们自动处理此分区和必要的通信（如 AllReduce 或 AllGather）。

启用和控制 TP 的主要参数 (parameter)是 `--tensor-model-parallel-size`（或根据 Megatron-LM 的特定版本或分支而异的类似变体）。这个值，我们称之为 TPsizeTP\_{size}TPsize​，指定每个张量并行层将被拆分到多少个 GPU 上。

例如，如果设置 `--tensor-model-parallel-size 4`，一个大型线性层的权重矩阵将被划分为 4 个列或行段，每个段驻留在张量并行组内的不同 GPU 上。在前向和反向传播 (backpropagation)过程中，Megatron-LM 协调这 4 个 GPU 之间所需的数据交换。

```python

tp_group = get_tensor_model_parallel_group()

linear_column_parallel = ColumnParallelLinear(
    input_size,
    output_size,
    tp_group=tp_group,
)

linear_row_parallel = RowParallelLinear(
    input_size,
    output_size,
    tp_group=tp_group,
)
```

**TP 的重要考虑因素：**

1. **可除性：** TPsizeTP\_{size}TPsize​ 必须能整除模型的隐藏维度（`--hidden-size`）和注意力头的数量（`--num-attention-heads`）。Megatron-LM 的并行层依赖此进行分区。
2. **通信：** TP 会在张量并行组内引入通信开销，通常是 AllReduce 操作。效率在很大程度上取决于这些 GPU 之间的互连速度（例如 NVLink）。
3. **内存：** TP 减少了组内每个 GPU 上权重、梯度和优化器状态所需的内存，但由于通信需求，增加了激活内存使用量。

### 流水线并行 (PP) 配置

流水线并行涉及将模型的层按顺序划分到不同的 GPU 上，形成一个流水线。每个 GPU（或结合 TP/DP 时的 GPU 组）构成流水线中的一个“阶段”，负责只执行模型层的一个子集。

配置 PP 的主要参数 (parameter)是 `--pipeline-model-parallel-size`（或类似参数）。这个值，PPsizePP\_{size}PPsize​，决定流水线的阶段数量。例如，`--pipeline-model-parallel-size 4` 创建一个 4 阶段的流水线。

为了保持流水线阶段的利用率并最小化空闲时间（“气泡”），Megatron-LM 采用微批处理。整体训练批次被拆分为更小的微批，这些微批并发地流经流水线阶段。微批数量是一个重要的调整参数，通常由 `--num-microbatches` 控制，或根据全局批次大小、微批次大小和数据并行度计算得出。常见的调度方法是 1F1B（每个微批一次前向传播，一次反向传播 (backpropagation)），这有助于平衡计算和内存使用。

G

cluster₀

阶段 0

cluster₁

阶段 1

cluster₂

阶段 2

s0ₘb1

前向 MB1

s0ₘb2

前向 MB2

s1ₘb1

前向 MB1

s0ₘb1->s1ₘb1

 激活值

s0ₘb3

前向 MB3

s1ₘb2

前向 MB2

s0ₘb2->s1ₘb2

 激活值

s1ₘb3

前向 MB3

s0ₘb3->s1ₘb3

 激活值

s2ₘb1

前向 MB1

s1ₘb1->s2ₘb1

 激活值

s2ₘb2

前向 MB2

s1ₘb2->s2ₘb2

 激活值

s2ₘb3

前向 MB3

s1ₘb3->s2ₘb3

 激活值

> 微批（MB1、MB2、MB3）通过 3 阶段流水线（PPsize=3PP\_{size}=3PPsize​=3）的前向传播（Fwd）流程。激活值（Actvs）在阶段之间传递。反向传播遵循类似的反向流程。

**PP 的重要考虑因素：**

1. **层数：** 变压器 (Transformer)层的总数（`--num-layers`）理想情况下应能被 PPsizePP\_{size}PPsize​ 整除，以实现负载均衡，尽管 Megatron-LM 可以处理不均匀的分布。
2. **通信：** PP 涉及相邻阶段之间的点对点通信，以向前传递激活值和向后传递梯度。
3. **流水线气泡：** PP 的效率取决于最小化流水线气泡（处理批次开始和结束时的空闲时间）。增加微批数量有帮助，但也会增加激活内存。
4. **内存平衡：** 内存使用在不同阶段可能无法完美平衡，因为第一和最后阶段通常包含嵌入 (embedding)层和输出层，这些层的大小可能与中间变压器块不同。

### 结合张量并行与流水线并行

Megatron-LM 擅长结合不同的并行策略。通常，TP 和 PP 会一起使用。在这种设置中，模型并行涉及的 GPU 总数是 TPsize×PPsizeTP\_{size} \times PP\_{size}TPsize​×PPsize​。每个流水线阶段本身可能包含多个 GPU 通过张量并行一起工作。数据并行（DP）可以在此之上叠加，复制这种 TP/PP 结构。

训练中使用的 GPU 总数将是 Ngpus=DPsize×TPsize×PPsizeN\_{gpus} = DP\_{size} \times TP\_{size} \times PP\_{size}Ngpus​=DPsize​×TPsize​×PPsize​。

G

cluster\_PP0

流水线阶段 0

cluster\_PP1

流水线阶段 1

cluster\_PP2

流水线阶段 2

cluster\_PP3

流水线阶段 3

PP0

GPU 0

GPU 1

TP 组 0

PP1

GPU 2

GPU 3

TP 组 1

PP0->PP1

 激活值/梯度

PP2

GPU 4

GPU 5

TP 组 2

PP1->PP2

 激活值/梯度

PP3

GPU 6

GPU 7

TP 组 3

PP2->PP3

 激活值/梯度

> 8 个 GPU 的配置示例：PPsize=4PP\_{size}=4PPsize​=4 和 TPsize=2TP\_{size}=2TPsize​=2。每个流水线阶段使用 2 个 GPU 进行张量并行。通信发生在阶段之间（PP）的点对点方式，以及每个 TP 组内部（TP）的集体操作。

### 配置示例

这里有一个简化的示例，说明在使用 TP 和 PP 配置 Megatron-LM 进行 GPT 风格模型训练时，命令行参数 (parameter)可能是什么样子（假设总共有 8 个 GPU，其中 TPsize=2TP\_{size}=2TPsize​=2 和 PPsize=4PP\_{size}=4PPsize​=4，为简单起见，数据并行 DPsize=1DP\_{size}=1DPsize​=1）：

```bash

python pretrain_gpt.py \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 32 \
       --seq-length 2048 \
       \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 4 \
       \
       --micro-batch-size 4 \
       --global-batch-size 128 \

       --num-microbatches 32 \
       \
       --optimizer adam \
       --learning-rate 1.0e-4 \
       --weight-decay 0.01 \
       --clip-grad 1.0 \
       \
       --train-data-path <path_to_train_data> \
       --valid-data-path <path_to_valid_data> \
       --tokenizer-type SentencePieceTokenizer \
       --tokenizer-model <path_to_tokenizer> \
       \
       --distributed-backend nccl \
       --save <path_to_save_checkpoints> \
       --load <path_to_load_checkpoints> \
       --log-interval 10 \
       --save-interval 1000 \
       --eval-interval 100 \
       --num-workers 2
```

在此示例中：

- `--tensor-model-parallel-size 2`：将线性层和注意力层等拆分到 2 个 GPU 上。
- `--pipeline-model-parallel-size 4`：将 24 个层拆分到 4 个顺序阶段（每个阶段大约 6 层）。
- `--num-microbatches 32`：用于高效地为 4 阶段流水线提供数据，根据全局批次大小、微批次大小和数据并行度计算得出。

正确配置这些参数对于平衡计算、内存使用和可用硬件上的通信开销非常重要。Megatron-LM 提供底层的机制，但确定最佳的 TPsizeTP\_{size}TPsize​、PPsizePP\_{size}PPsize​ 和微批数量通常需要根据具体的模型架构、硬件配置（GPU 类型、互连）和所需的批次大小进行实验。

获取即时帮助、个性化解释和交互式代码示例。

---

### 结合框架与策略

# 结合框架与策略

虽然DeepSpeed和Megatron-LM等框架为实现特定并行策略提供了强大的工具，但训练非常大的模型通常需要同时结合多种技术。当模型规模和硬件能力的界限被推向极致时，单独使用数据并行（DP）、张量并行（TP）或流水线并行（PP）可能不足或并非最优。仅依靠DP即使有ZeRO等优化也可能达到内存限制。仅依靠TP可能导致大量设备间的通信开销过高。单独使用PP会引入流水线气泡，降低利用率。因此，复杂的训练设置常会混合这些策略，利用可协同工作的框架或提供集成方案的框架。

### 混合并行化的必要性

设想训练一个拥有万亿参数 (parameter)的模型。

- **仅DP：** 即使ZeRO-3优化了内存，在每个设备上复制完整的正向/反向计算过程可能太慢，或者对于非常大的模型或批量大小，每个设备仍需要过多的激活内存。
- **仅TP：** 分割张量需要张量并行组内大量的通信（AllReduce，点对点）。将TP扩展到非常多的GPU（例如数百个）可能使通信成为主要瓶颈。
- **仅PP：** 虽然PP将层分割到不同设备上，减少了每个设备的内存，但它会受到流水线气泡的影响，特别是当阶段数量很多时，这会导致GPU空闲时间。

将这些策略结合起来可以减轻各自的缺点。一种常见的方法通常被称为“3D并行化”：

1. **流水线并行（PP）：** 将模型的层分割到多个阶段，分布在多个节点或GPU上。这主要减少了每个GPU所需的激活和参数内存。
2. **张量并行（TP）：** 将每个层*内部*的操作（如大型权重 (weight)矩阵）分割到多个GPU上，通常在一个节点内或通过高速互连（如NVLink）连接的一组节点内。这进一步减少了每个GPU的内存需求，并允许适应更大的层。
3. **数据并行（DP）：** 将TP/PP模型配置复制到多组设备上。每个副本处理不同的微批数据。这可以扩大整体批量大小和吞吐量 (throughput)。DeepSpeed的ZeRO优化常在此处应用，以管理数据并行副本间的优化器状态、梯度以及可能的参数。

### 整合DeepSpeed和Megatron-LM

一种流行且有效的方式是使用Megatron-LM实现其高效的TP和PP，并结合DeepSpeed的高级DP优化（ZeRO）以及可能包括激活检查点或高效优化器等其他功能。

它们通常的协同方式如下：

- **Megatron-LM处理TP和PP：** 它提供函数和模块来定义具有张量并行层（例如`ColumnParallelLinear`，`RowParallelLinear`）的模型结构，并管理跨阶段的流水线调度。您首先初始化Megatron-LM以设置TP和PP rank所需的进程组。
- **DeepSpeed封装Megatron-LM模型：** DeepSpeed在Megatron-LM模型设置*之后*初始化。它将Megatron定义的模型（其中已包含TP/PP逻辑）作为输入。然后，DeepSpeed通常使用ZeRO将其DP逻辑应用于数据并行维度。DeepSpeed引擎在DP组*内*管理优化器、梯度累积和通信，同时遵循Megatron-LM建立的底层TP/PP结构。

G

clusterₙode0

节点 0

clusterₙode1

节点 1

gpu0₀

GPU 0
阶段 0
TP Rank 0

gpu0₁

GPU 1
阶段 0
TP Rank 1

gpu0₀->gpu0₁

TP 通信

gpu0₁->gpu0₀

gpu1₀

GPU 0
阶段 1
TP Rank 0

gpu0₁->gpu1₀

PP 通信 (正向)

gpu1₁

GPU 1
阶段 1
TP Rank 1

gpu1₀->gpu1₁

TP 通信

gpu1₁->gpu0₀

PP 通信 (反向)

gpu1₁->gpu1₀

dpₙote

DP Rank 0
(此整个设置是一个DP副本。
ZeRO管理副本间的状态。)

> 一个简化的2阶段流水线并行（PP）、2路张量并行（TP）设置视图。带ZeRO的数据并行（DP）将复制此整个结构，并管理这些副本间的状态。

### 配置与初始化

设置此类混合系统需要细致的配置。您通常需要：

1. **初始化进程组：** 使用`torch.distributed.init_process_group`，然后根据进程的rank定义数据并行、张量并行和流水线并行的特定进程组。Megatron-LM通常提供实用函数来帮助管理这些组。
2. **配置Megatron-LM：** 设置与张量模型并行大小（`--tensor-model-parallel-size`）、流水线模型并行大小（`--pipeline-model-parallel-size`）、虚拟流水线阶段（`--num-layers-per-virtual-pipeline-stage`）等相关的参数 (parameter)。
3. **配置DeepSpeed：** 创建一个DeepSpeed配置JSON文件，指定ZeRO优化阶段（`zero_optimization.stage`）、学习率、批量大小、梯度裁剪、AMP设置以及可能的激活检查点详情。值得注意的是，DeepSpeed需要了解DP组，但其操作应独立于Megatron-LM管理的TP/PP组。

以下是使用PyTorch进行Python初始化的一个示意：

```python
import torch
import deepspeed
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_args

torch.distributed.init_process_group(backend='nccl')

initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

args = get_args()
model = GPTModel(
    num_tokentypes=0,
    parallel_output=True,

)

model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    args=args,
    model=model,

    config_params=args.deepspeed_config
)
```

> PyTorch代码展示了初始化`torch.distributed`、Megatron-LM（用于TP/PP设置）、使用Megatron组件定义模型以及最后初始化DeepSpeed来封装模型并处理DP/ZeRO的顺序。实际实现涉及更多细节，尤其是在参数解析和进程组管理方面。

### 注意事项与挑战

结合框架会增加复杂性：

- **配置：** 管理DeepSpeed和Megatron-LM的配置文件和命令行参数 (parameter)需要细致考量，以确保兼容性和正确性。了解哪个框架控制哪个方面（例如，优化器状态分片与张量分片）很重要。
- **调试：** 在混合设置中调试问题可能具有挑战性，因为问题可能源于DP、TP、PP、ZeRO、激活检查点或底层硬件/通信库（NCCL）之间的相互作用。
- **通信：** 不同通信模式（DP梯度的AllReduce，TP的AllReduce/点对点，PP的点对点）之间的协同需要高效的网络基础设施（例如，节点内TP的NVLink，节点间DP/PP的InfiniBand/RoCE）。
- **兼容性：** 确保DeepSpeed、Megatron-LM、PyTorch以及CUDA/NCCL库的版本兼容。这些框架快速迭代。

尽管存在复杂性，但结合DeepSpeed和Megatron-LM等策略与框架通常是训练先进大型语言模型最实用的方法，它有效地平衡了大型GPU集群中的计算、内存和通信限制。理解如何组织这些组件是当今构建和扩展LLM的一个重要部分。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 17 Optimization Algorithms Llms

### 梯度下降算法变体回顾 (SGD, 动量)

# 梯度下降算法变体回顾 (SGD, 动量)

理解基础的梯度下降 (gradient descent)算法对于掌握现代大型语言模型训练中常用的自适应方法是必不可少的。分析其机制和局限，有助于阐明为何在处理LLM的复杂优化问题时，经常需要AdamW等更高级的优化器。

### 随机梯度下降 (gradient descent) (SGD)

其核心是，深度学习 (deep learning)中的优化涉及调整模型参数 (parameter)（权重 (weight)和偏置 (bias)，统称为θ\thetaθ）以最小化损失函数 (loss function)LLL。梯度下降通过迭代地将参数沿与损失函数对参数的梯度相反的方向移动来实现此目的。

全批量梯度下降使用整个训练数据集计算梯度，这对于LLM预训练 (pre-training)中使用的庞大数据集来说，计算上是不可行的。随机梯度下降（SGD）通过在每一步仅使用一小部分随机数据子集（称为小批量）来近似梯度，从而处理此问题。

SGD的更新规则是：

θ←θ−η∇θL(θ;x(i:i+b),y(i:i+b))\theta \leftarrow \theta - \eta \nabla\_{\theta} L(\theta; x^{(i:i+b)}, y^{(i:i+b)})θ←θ−η∇θ​L(θ;x(i:i+b),y(i:i+b))

这里：

- θ\thetaθ 代表模型参数。
- η\etaη 是学习率，一个控制步长的超参数 (hyperparameter)。
- LLL 是损失函数。
- x(i:i+b),y(i:i+b)x^{(i:i+b)}, y^{(i:i+b)}x(i:i+b),y(i:i+b) 代表一个包含 bbb 个输入样本及其对应目标的小批量数据。
- ∇θL(⋅)\nabla\_{\theta} L(\cdot)∇θ​L(⋅) 是损失函数对参数的梯度，仅使用当前小批量数据计算。

SGD的主要优点是其每一步的计算效率。处理一个小批量数据比处理整个数据集要快得多。更新的随机性（由于随机小批量采样）也会引入噪声，这有时可以帮助优化器摆脱不好的局部最小值。

然而，这种噪声也可能是一个缺点。更新可能显著震荡，导致锯齿状的收敛路径。此外，SGD在具有高曲率或山谷（表面在一个维度上比另一个维度弯曲得多）的优化面中可能表现不佳，可能需要许多步骤才能抵达最小值。SGD也对学习率 η\etaη 的选择相当敏感。

在PyTorch中，使用SGD很简单：

```python
import torch

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for inputs, targets in data_loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
```

### 带有动量的SGD

为了减少SGD固有的震荡并加速收敛，尤其是在山谷中，引入了动量技术。它添加了一个“速度”项 vvv，该项积累了过去梯度的指数衰减移动平均。参数 (parameter)更新随后会包含此速度项。

更新规则通常表述为：

1. 计算当前小批量的梯度：gt=∇θL(θt)g\_t = \nabla\_{\theta} L(\theta\_t)gt​=∇θ​L(θt​)
2. 更新速度：vt←βvt−1+gtv\_t \leftarrow \beta v\_{t-1} + g\_tvt​←βvt−1​+gt​
3. 更新参数：θt+1←θt−ηvt\theta\_{t+1} \leftarrow \theta\_t - \eta v\_tθt+1​←θt​−ηvt​

这里：

- vtv\_tvt​ 是在步长 ttt 时的速度向量 (vector)。
- β\betaβ 是动量系数，通常是一个接近1的值（例如0.9）。它控制过去梯度对当前更新的影响程度。较高的 β\betaβ 表示过去梯度贡献更多。
- η\etaη 是学习率。

速度项 vtv\_tvt​ 有助于平滑更新。如果连续的梯度指向相似的方向，速度会累积，导致更大的步长和更快的收敛。如果梯度震荡，动量项通过平均它们来帮助抑制这些震荡。可以将其想象成一个重球滚下山坡；它在当前方向上保持动量，并且较少受到小颠簸（噪声梯度）的影响。

虽然动量通常比普通的SGD在收敛速度和稳定性上有所改进，但它仍然依赖于所有参数的单一学习率 η\etaη，并且需要仔细调整 η\etaη 和 β\betaβ。

在PyTorch中使用动量，只需向`SGD`优化器添加`momentum`参数：

```python
import torch

learning_rate = 0.01
momentum_beta = 0.9

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=momentum_beta
)
```

虽然SGD和动量构成了许多优化策略的依据，但训练大型语言模型通常涉及应对具有复杂损失表面的极高维参数空间。这些基础方法可能收敛缓慢或停滞不前。这促使人们使用像Adam和AdamW这样的自适应优化算法，它们会根据每个参数调整学习率，并且在实际中通常能使这些大型模型更快地收敛。我们将在以下部分考察这些自适应方法。

获取即时帮助、个性化解释和交互式代码示例。

---

### 自适应优化器：Adam和AdamW

# 自适应优化器：Adam和AdamW

标准的随机梯度下降 (gradient descent)（SGD）及其动量变体是优化方法的基础，但训练像大型语言模型（LLMs）这样庞大的模型，通常能从自适应学习率方法中获得更好的表现。这些算法会为每个参数 (parameter)单独调整学习率，可能加快收敛速度，特别是在梯度稀疏或参数间梯度大小不一的情形下，这在深度神经网络 (neural network)中很普遍。

### Adam：自适应矩估计

最受青睐的自适应优化器之一是Adam（自适应矩估计）。Adam通过梯度的第一和第二矩估计，为不同参数 (parameter)计算各自的自适应学习率。它主要结合了动量（使用第一矩估计，即过去梯度的指数衰减平均）和RMSprop（使用第二矩估计，即过去梯度平方的指数衰减平均）的理念。

设gt\mathbf{g}\_tgt​是目标函数在时间步ttt时对参数θ\thetaθ的梯度。Adam维护两个移动平均值：

1. **第一矩估计（动量）：**

   mt=β1mt−1+(1−β1)gt\mathbf{m}\_t = \beta\_1 \mathbf{m}\_{t-1} + (1 - \beta\_1) \mathbf{g}\_tmt​=β1​mt−1​+(1−β1​)gt​

   这是梯度的均值估计。β1\beta\_1β1​是指数衰减率，通常接近1（例如，0.9）。
2. **第二矩估计（未中心化方差）：**

   vt=β2vt−1+(1−β2)gt2\mathbf{v}\_t = \beta\_2 \mathbf{v}\_{t-1} + (1 - \beta\_2) \mathbf{g}\_t^2vt​=β2​vt−1​+(1−β2​)gt2​

   这是梯度未中心化方差的估计（逐元素平方）。β2\beta\_2β2​是指数衰减率，通常也接近1（例如，0.999）。

由于mt\mathbf{m}\_tmt​和vt\mathbf{v}\_tvt​被初始化为零向量 (vector)，它们会偏向于零，特别是在最初的时间步。Adam对这种偏差进行了校正：

m^t=mt1−β1t\hat{\mathbf{m}}\_t = \frac{\mathbf{m}\_t}{1 - \beta\_1^t}m^t​=1−β1t​mt​​
v^t=vt1−β2t\hat{\mathbf{v}}\_t = \frac{\mathbf{v}\_t}{1 - \beta\_2^t}v^t​=1−β2t​vt​​

其中ttt是当前时间步索引（从1开始）。

最后，参数更新规则是：

θt=θt−1−ηm^tv^t+ϵ\theta\_t = \theta\_{t-1} - \eta \frac{\hat{\mathbf{m}}\_t}{\sqrt{\hat{\mathbf{v}}\_t} + \epsilon}θt​=θt−1​−ηv^t​​+ϵm^t​​

这里，η\etaη是基础学习率，ϵ\epsilonϵ是一个小常数（例如，10−810^{-8}10−8），为了数值稳定性而添加，主要用于避免除以零。项ηv^t+ϵ\frac{\eta}{\sqrt{\hat{\mathbf{v}}\_t} + \epsilon}v^t​​+ϵη​起到了实际的、针对参数的学习率的作用。过去梯度较大的参数（v^t\hat{\mathbf{v}}\_tv^t​较大）获得较小的更新，而过去梯度较小的参数则获得较大的更新。

在PyTorch中，使用Adam很简单：

```python
import torch
import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    eps=epsilon
)
```

### AdamW：解耦权重 (weight)衰减

尽管Adam在多种场景下表现良好，但它对L2正则化 (regularization)（权重衰减）的处理可能不尽理想。标准的L2正则化会向损失函数 (loss function)添加项λ2∥θ∥2\frac{\lambda}{2} \|\theta\|^22λ​∥θ∥2，导致梯度项λθ\lambda \thetaλθ被加到gt\mathbf{g}\_tgt​中。在Adam中，这个权重衰减项λθ\lambda \thetaλθ通过mt\mathbf{m}\_tmt​和vt\mathbf{v}\_tvt​成为自适应学习率计算的一部分。这表明应用于参数 (parameter)的实际权重衰减取决于其梯度历史大小（通过v^t\sqrt{\hat{\mathbf{v}}\_t}v^t​​）。梯度大的参数所受到的实际权重衰减小于预期，而梯度小的参数所受到的实际权重衰减则大于预期。

AdamW提出了一个简单修正：将权重衰减与梯度更新解耦。AdamW不再将λθ\lambda \thetaλθ加到梯度gt\mathbf{g}\_tgt​中，而是仅使用来自主要损失函数的梯度进行标准的Adam更新，然后在Adam步骤*之后*直接对参数应用权重衰减。

AdamW的更新规则如下：

1. 按照Adam的方式计算mt\mathbf{m}\_tmt​、vt\mathbf{v}\_tvt​、m^t\hat{\mathbf{m}}\_tm^t​、v^t\hat{\mathbf{v}}\_tv^t​，只使用来自损失函数的梯度gt\mathbf{g}\_tgt​（不包含λθ\lambda \thetaλθ项）。
2. 执行自适应更新：
   ut=ηm^tv^t+ϵ\mathbf{u}\_t = \eta \frac{\hat{\mathbf{m}}\_t}{\sqrt{\hat{\mathbf{v}}\_t} + \epsilon}ut​=ηv^t​​+ϵm^t​​
3. 应用权重衰减并更新参数：
   θt=θt−1−ut−ηλθt−1\theta\_t = \theta\_{t-1} - \mathbf{u}\_t - \eta \lambda \theta\_{t-1}θt​=θt−1​−ut​−ηλθt−1​

请注意最后一项：−ηλθt−1-\eta \lambda \theta\_{t-1}−ηλθt−1​。权重衰减直接作用于先前的权重值θt−1\theta\_{t-1}θt−1​，并且只按全局学习率η\etaη进行缩放，而非自适应学习率。这使得权重衰减的行为更接近于标准SGD加上动量时的表现，从而在很多情况下，特别是对于Transformer这类深度模型，能够带来更好的泛化性能。

在PyTorch中使用AdamW与Adam类似，只额外需要`weight_decay`参数：

```python
import torch
import torch.optim as optim

optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    eps=epsilon,
    weight_decay=weight_decay_lambda
)
```

由于其对权重衰减的改进处理和出色的经验表现，AdamW已成为训练大型语言模型的一个非常常见的选择。Adam和AdamW之间的选择，以及它们超参数 (hyperparameter)（η,β1,β2,ϵ,λ\eta, \beta\_1, \beta\_2, \epsilon, \lambdaη,β1​,β2​,ϵ,λ）的设置，通常取决于模型架构、数据集和训练配置，这需要仔细调整，本章后面会对此进行讨论。

获取即时帮助、个性化解释和交互式代码示例。

---

### 学习率调度策略

# 学习率调度策略

选择合适的学习率 η\etaη 是模型训练成功的基础。然而，在整个训练过程中使用固定学习率通常不是最优解，特别是对于像 Transformer 这样的大型复杂模型。在训练初期，模型参数 (parameter)离最优值较远，如果学习率过高，大的梯度可能导致不稳定或发散。相反，在训练后期，当模型接近收敛时，通常需要较小的学习率进行微调 (fine-tuning)并找到一个良好的最小值。学习率调度通过在每个训练步 ttt 动态调整学习率 ηt\eta\_tηt​ 来应对此情况。

对于大型语言模型，一种常见且有效的策略是结合*预热*阶段和随后的*衰减*阶段。

### 学习率预热

在训练的初始阶段，特别是当使用 Adam 或 AdamW 等自适应优化器时，由于观察到的样本数量有限，方差估计可能不可靠。此外，随机初始化的模型会产生显著误差，因此初始梯度可能较大且带有噪声。立即应用大学习率可能导致数值不稳定或使模型发散。

为缓解这种情况，通常会采用预热期。在此阶段，学习率会在预设的步数 `warmup_steps` 内，从一个很小的值（通常为 0）逐渐增加到其目标峰值 ηpeak\eta\_{peak}ηpeak​。最常见的方法是线性预热：

ηt=ηpeak×t预热步数对于 0≤t<预热步数\eta\_t = \eta\_{peak} \times \frac{t}{\text{预热步数}} \quad \text{对于 } 0 \le t < \text{预热步数}ηt​=ηpeak​×预热步数t​对于 0≤t<预热步数

这种逐渐增加使 AdamW 等优化器中的自适应动量稳定下来，并防止训练初期出现大的、可能破坏稳定的更新。`warmup_steps` 是一个超参数 (parameter) (hyperparameter)，通常设置为几千步，或是总训练步数的一小部分百分比（例如 1-10%），这取决于数据集大小和批次大小。

### 学习率衰减

一旦预热阶段完成且学习率达到 ηpeak\eta\_{peak}ηpeak​，在剩余的训练步数中逐步降低学习率通常是有益的。这能使模型稳定到一个良好的最小值。有几种常见的衰减策略：

#### 线性衰减

学习率从 `warmup_steps` 到总训练步数 TTT 之间，从 ηpeak\eta\_{peak}ηpeak​ 线性降低到最小值 ηmin\eta\_{min}ηmin​（通常为 0）。

ηt=ηmin+(ηpeak−ηmin)×T−tT−预热步数对于 预热步数≤t≤T\eta\_t = \eta\_{min} + (\eta\_{peak} - \eta\_{min}) \times \frac{T - t}{T - \text{预热步数}} \quad \text{对于 } \text{预热步数} \le t \le Tηt​=ηmin​+(ηpeak​−ηmin​)×T−预热步数T−t​对于 预热步数≤t≤T

#### 余弦衰减

这是一种训练大型模型的流行策略。学习率遵循余弦曲线从 ηpeak\eta\_{peak}ηpeak​ 下降到 ηmin\eta\_{min}ηmin​。它起初下降缓慢，然后加速，接着在接近 ηmin\eta\_{min}ηmin​ 时再次放慢速度。这种平滑的衰减曲线在实践中通常效果良好。

ηt=ηmin+12(ηpeak−ηmin)(1+cos⁡(πt−预热步数T−预热步数))对于 预热步数≤t≤T\eta\_t = \eta\_{min} + \frac{1}{2}(\eta\_{peak} - \eta\_{min}) \left( 1 + \cos\left( \pi \frac{t - \text{预热步数}}{T - \text{预热步数}} \right) \right) \quad \text{对于 } \text{预热步数} \le t \le Tηt​=ηmin​+21​(ηpeak​−ηmin​)(1+cos(πT−预热步数t−预热步数​))对于 预热步数≤t≤T

#### 其他衰减调度

其他调度，如反平方根衰减 (ηt∝1/t\eta\_t \propto 1/\sqrt{t}ηt​∝1/t​) 或多项式衰减也常被使用，但线性衰减和余弦衰减（尤其是余弦衰减）是预训练 (pre-training)大型 Transformer 模型时非常普遍的选择。

### 在 PyTorch 中实现调度

PyTorch 在 `torch.optim.lr_scheduler` 中提供了灵活的学习率调度工具。你可以使用 `LambdaLR` 实现自定义调度，或使用内置调度器。像 Hugging Face 的 `transformers` 等库也提供了便捷的辅助函数。

以下是定义一个适用于 `LambdaLR` 的调度函数的方法，该函数实现线性预热后接余弦衰减：

```python
import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

num_training_steps = 100000
num_warmup_steps = 10000
peak_lr = 3e-4

min_lr = 3e-5

def lr_lambda(current_step: int):

    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = (float(current_step - num_warmup_steps) /
                float(max(1, num_training_steps - num_warmup_steps)))

    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    decay_factor = cosine_decay * (1 - min_lr / peak_lr) + (min_lr / peak_lr)

    return decay_factor

lr_scheduler = LambdaLR(optimizer, lr_lambda)
```

或者，使用 `transformers` 库可以简化这一点：

```python
from transformers import get_scheduler, AdamW

num_training_steps = 100000
num_warmup_steps = 10000
peak_lr = 3e-4

lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps

)
```

以下图表展示了一个典型的学习率调度，包含线性预热和余弦衰减：

020k40k60k80k100k0.0e+05.0e−51.0e−41.5e−42.0e−42.5e−43.0e−4

> 学习率曲线显示，在 10,000 预热步中线性增加到峰值 3e-4，随后在剩余的 90,000 步中余弦衰减至 0。

选择合适的调度、`warmup_steps`、`peak_lr` 和 `min_lr` 通常需要一些试验，但预热与后续衰减（尤其是余弦衰减）的组合是训练大型语言模型的一个良好起点。它在早期提供了稳定性，并使得后期训练能够有效收敛。

获取即时帮助、个性化解释和交互式代码示例。

---

### 梯度裁剪方法

# 梯度裁剪方法

训练大型神经网络 (neural network)，特别是多层Transformer模型，有时会造成数值不稳定。一个常见的问题是“梯度爆炸”现象，即在反向传播 (backpropagation)时，梯度的数值变得过大。这可能导致模型参数 (parameter)进行巨大的更新，可能引发发散（即损失值飙升至无穷大或NaN——非数字）或阻止收敛的震荡。这种不稳定在多层网络中，或在使用某些激活函数 (activation function)或初始化方案时，可能尤其常见，即便使用了层归一化 (normalization)等方法。由于低精度格式的数值范围有限，在混合精度训练（第20章讨论）中，这种情况也可能加剧。

梯度裁剪是一种直接而有效的方法，它通过限制梯度的大小来缓解这个问题，在优化器使用梯度更新模型权重 (weight)之前进行操作。主要思路不是改变梯度更新的方向，而是当梯度大小超过预设阈值时，限制其大小。

### 按梯度范数裁剪

对于大型语言模型来说，最常用的方法是按梯度的L2范数（欧几里得范数）进行裁剪。这种做法将所有模型参数 (parameter)的全部梯度（或有时是每个参数组的梯度）视为一个向量 (vector)，计算其L2范数，如果范数超过给定阈值 ccc，则重新缩放该向量。

从数学上讲，令 g\mathbf{g}g 代表所有梯度串联在一起的向量。L2范数计算如下：

∥g∥=∑igi2\|\mathbf{g}\| = \sqrt{\sum\_{i} g\_i^2}∥g∥=i∑​gi2​​

裁剪操作则应用如下：

g←c∥g∥g如果 ∥g∥>c\mathbf{g} \leftarrow \frac{c}{\|\mathbf{g}\|} \mathbf{g} \quad \text{如果 } \|\mathbf{g}\| > cg←∥g∥c​g如果 ∥g∥>c

如果范数 ∥g∥\|\mathbf{g}\|∥g∥ 小于或等于阈值 ccc，梯度保持不变。如果范数大于 ccc，梯度向量 g\mathbf{g}g 会被按 c/∥g∥c / \|\mathbf{g}\|c/∥g∥ 的比例缩小，从而确保其新范数恰好为 ccc。这在限制梯度大小的同时，保持了梯度更新的方向。

在PyTorch中，这通常使用 `torch.nn.utils.clip_grad_norm_` 函数实现。它在反向传播 (backpropagation)（计算梯度）之后、优化器步骤（根据梯度更新权重 (weight)）之前应用。

```python
import torch
from torch.nn.utils import clip_grad_norm_

optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()

max_grad_norm = 1.0

total_norm = clip_grad_norm_(
    model.parameters(),
    max_norm=max_grad_norm,
    norm_type=2.0
)

optimizer.step()
```

在此代码段中，`clip_grad_norm_` 计算传递给它的参数（`model.parameters()`）的所有梯度的总L2范数。如果此范数超过 `max_norm`（我们的阈值 ccc），它会通过重新缩放 *原地* 修改梯度。该函数返回裁剪前的原始总范数，这有助于观察训练过程。设置 `norm_type=2.0` 明确指定了L2范数。

### 按梯度值裁剪

另一种方法是按值裁剪，尽管在训练大型Transformer模型时不如常用。这种方法独立地裁剪每个梯度分量 gig\_igi​，如果它落在指定范围 [−c,c][-c, c][−c,c] 之外。

gi←max(min(gi,c),−c)g\_i \leftarrow \text{max}(\text{min}(g\_i, c), -c)gi​←max(min(gi​,c),−c)

这意味着任何大于 ccc 的梯度分量都被设为 ccc，任何小于 −c-c−c 的分量都被设为 −c-c−c。与范数裁剪不同，这种方法可以改变总梯度向量 (vector)的 *方向*，因为不同分量可能被不同程度地裁剪或根本不裁剪。

在PyTorch中，这可以使用 `torch.nn.utils.clip_grad_value_` 完成：

```python
import torch
from torch.nn.utils import clip_grad_value_

clip_value = 0.5

clip_grad_value_(model.parameters(), clip_value=clip_value)

optimizer.step()
```

尽管更简单，但与范数裁剪相比，按值裁剪在深度学习 (deep learning)优化中通常被认为理论依据较少，因为它不将梯度视为一个统一的方向向量。对于大多数大型语言模型训练情况，`clip_grad_norm_` 是更受推荐的方法。

### 梯度裁剪的实际考量

- **选择阈值 ccc：** 裁剪阈值（`max_norm` 或 `clip_value`）是一个超参数 (parameter) (hyperparameter)，通常需要凭经验调整。在大型语言模型训练中，`max_norm` 的一个常见起始值是 1.0。
  - 如果设置过高，它可能无法有效防止偶然出现的巨大梯度引起的不稳定。
  - 如果设置过低，它可能通过不必要地缩小有用梯度而阻碍学习，减慢收敛速度。
  - 在训练期间监测梯度范数（`clip_grad_norm_` 在裁剪 *发生前* 返回的值）有助于确定这个选择。如果范数频繁达到阈值，你可能要考虑学习率是否过高，或者阈值是否可以略微提高。如果范数很少接近阈值，它可能没有产生太大影响。
- **与优化器和学习率的相互作用：** 梯度裁剪与优化器和学习率调度配合使用。它起到安全机制的作用，但不能取代对合适的学习率和调度器的需求。如果梯度持续过大并被裁剪，这可能表明学习率对于当前的训练阶段来说过高。
- **必要性：** 尽管梯度裁剪通常有助于稳定性，尤其是在非常大的模型中或在训练的初始阶段（或微调 (fine-tuning)时），但如果其他因素（适当的初始化、像Pre-LN这样的归一化 (normalization)、调整得当的学习率调度、AdamW优化器）有助于稳定训练，它可能并非总是严格必要。然而，考虑到大型语言模型训练的成本，它经常被用作一种预防措施。

下图说明了按范数裁剪的效果。位于圆圈外部（代表范数阈值 ccc）的梯度向量 (vector)会沿径向按比例缩小，趋向于圆周边界，同时保持其原始方向。

G

center

origin

g1

origin->g1

 g₁ (未裁剪)

g2

origin->g2

 g₂ (原始)

g3

origin->g3

 g₃ (原始)

g2\_clipped

origin->g2\_clipped

 g₂ (已裁剪)

g3\_clipped

origin->g3\_clipped

 g₃ (已裁剪)

labelₙode
||g|| = c

> 梯度向量 g2g\_2g2​ 和 g3g\_3g3​ 原始范数大于阈值 ccc。按范数裁剪将它们按比例缩小（蓝色虚线箭头），使其位于由 ccc 定义的边界上，而 g1g\_1g1​ 已经在阈值内，保持不变。

通过防止过大的更新，梯度裁剪显著提升了成功训练大型语言模型所需的稳定性，使AdamW等优化器和精心设计的学习率调度能有效应对复杂的损失曲面。

获取即时帮助、个性化解释和交互式代码示例。

---

### 选择优化器超参数 (lr, betas, eps, weight_decay)

# 选择优化器超参数 (lr, betas, eps, weight\_decay)

为优化器选择合适的超参数 (parameter) (hyperparameter)对于成功训练大型语言模型来说非常重要。尽管像AdamW这样的自适应优化器与传统SGD相比简化了一些方面，但找到学习率、动量项、epsilon以及权重 (weight)衰减的最佳设置仍然是训练过程中的主要组成部分，特别是在考虑到LLM训练的规模和成本时。这些超参数直接影响模型的收敛速度、稳定性和最终性能。本文探讨如何选择这些主要参数的值，包括AdamW、学习率调度和梯度裁剪等概念的考量。

### 学习率 (η\etaη)

学习率也许是唯一最重要的超参数 (parameter) (hyperparameter)。它决定了梯度下降 (gradient descent)过程中采取的步长。设置过高，训练可能会变得不稳定，导致发散或损失波动。设置过低，训练会慢得不切实际，可能陷入次优的局部最小值。

- **LLM的典型范围：** 对于使用AdamW训练的大型Transformer模型，峰值学习率通常在1e−51e-51e−5到6e−46e-46e−4的范围内。具体值很大程度上取决于模型大小、批次大小和架构。较大模型有时能从略小的峰值学习率中获益。例如，对于数十亿参数规模的模型，常见的起始点可能是1e−41e-41e−4到3e−43e-43e−4左右。
- **与批次大小的关联：** LLM训练通常涉及非常大的批次大小（通常每批数百万个token，通过数据和梯度累积实现）。一个常见的经验法则是*线性缩放规则*：如果将批次大小乘以kkk，那么学习率也应乘以kkk。然而，这条规则并非总是完全适用，尤其对于特别大的批次大小。一些研究表明，*平方根缩放规则*（η∝k\eta \propto \sqrt{k}η∝k​）可能更适用于某些批次大小。实际操作中，虽然批次大小会影响选择，但最优学习率通常在所述典型范围内通过经验发现，通常从训练类似大小模型的论文中报告的值开始。
- **与调度的关联：** 请记住，我们通常使用包含预热和衰减的学习率调度（第17章，“学习率调度策略”）。当我们谈论“学习率”时，通常是指在预热阶段后达到的*峰值*学习率。预热时长和衰减策略（例如，线性或余弦）也与峰值学习率的有效性有关。更长的预热可能允许设置略高的峰值学习率。
- **找到最优值：** 考虑到计算成本，穷举网格搜索通常不可行。常见的策略是：
  1. 从文献中报告的、针对相似大小和架构模型（例如，Llama、GPT-3论文）的值开始。
  2. 如果可能，在较短的训练运行或模型的较小规模版本/数据集上，围绕此值进行小范围尝试（例如，1e−4,2e−4,3e−41e-4, 2e-4, 3e-41e−4,2e−4,3e−4）。
  3. 在初始阶段密切关注训练损失曲线。快速下降后出现不稳定表明学习率过高。进展极其缓慢则表明学习率过低。

### Adam/AdamW Betas (β1,β2\beta\_1, \beta\_2β1​,β2​)

Adam和AdamW使用两个类似动量的项，由β1\beta\_1β1​和β2\beta\_2β2​控制。

- β1\beta\_1β1​：控制梯度的指数移动平均（第一阶矩）。典型默认值为0.90.90.9。
- β2\beta\_2β2​：控制梯度平方的指数移动平均（第二阶矩）。典型默认值为0.9990.9990.999。此项针对每个参数 (parameter)调整学习率。

对于大多数LLM训练场景，β1=0.9\beta\_1 = 0.9β1​=0.9和β2=0.999\beta\_2 = 0.999β2​=0.999的默认值效果非常好，通常无需修改即可使用。

- **调整考量：** 尽管调整这些参数不如调整学习率常见，但一些研究已尝试替代方案。
  - 降低β2\beta\_2β2​（例如，到0.950.950.95或0.980.980.98）会使优化器更快地适应梯度方差的变化，这有时有助于跳出尖锐的局部极小值或提高训练初期的稳定性，但也可能导致后期收敛稳定性下降。一些大型模型训练方案使用β2=0.95\beta\_2 = 0.95β2​=0.95或β2=0.98\beta\_2 = 0.98β2​=0.98。
  - 调整β1\beta\_1β1​的情况较少。
  - 除非您观察到通过调整学习率或使用梯度裁剪无法解决的特定不稳定情况，否则通常建议保留默认值（β1=0.9,β2=0.999\beta\_1=0.9, \beta\_2=0.999β1​=0.9,β2​=0.999，或者可以尝试β2=0.98\beta\_2=0.98β2​=0.98）。更改betas会增加调整过程的复杂性。

### Epsilon (ϵ\epsilonϵ)

Adam/AdamW中的epsilon项（ϵ\epsilonϵ）是一个小值，在自适应学习率计算时加到分母中（具体来说，是在计算第二阶矩估计的平方根之前）。它的主要目的是防止除以零并提高数值稳定性，尤其当第二阶矩估计非常接近零时。

- **常见值：** 标准默认值为1e−81e-81e−8。
- **调整考量：** Epsilon很少调整。与学习率或权重 (weight)衰减相比，它对性能的影响通常很小。在涉及混合精度训练（第20章）的场景中，特别是FP16，数值精度较低时，一些实践者可能会略微增加epsilon（例如，到1e−71e-71e−7或1e−61e-61e−6）以进一步提高数值稳定性。然而，如果怀疑存在与优化器分母相关的稳定性问题，这通常是最后的办法。坚持使用默认值1e−81e-81e−8是标准做法。

### 权重 (weight)衰减 (λ\lambdaλ)

权重衰减是一种正则化 (regularization)技术，通过向损失函数 (loss function)添加与模型权重平方大小成比例的惩罚项来防止过拟合 (overfitting)。如前所述，AdamW实现了*解耦*的权重衰减，与自适应梯度一起使用时，通常优于原始Adam优化器固有的L2正则化方法。

- **LLM的典型范围：** LLM训练中权重衰减（λ\lambdaλ）的常见值通常在0.010.010.01到0.10.10.1的范围内。0.10.10.1的值经常用作起始点。
- **调整策略：** 最优的权重衰减值取决于数据和模型。
  - 通常在找到合理的学习率和调度后进行调整。
  - 监测验证损失（或困惑度）。如果验证损失开始增加而训练损失持续下降，模型可能正在过拟合，增加权重衰减可能有所帮助。反之，如果训练和验证损失都过早进入平台期，减少权重衰减可能有利。
  - 小范围尝试（例如，0.01,0.01, 0.01,0.05, 0.10.10.1等值）通常就足够了。
  - 请注意，对于在海量数据集上训练的超大型模型，数据规模带来的隐式正则化可能会减少对强显式权重衰减的需求，但0.10.10.1左右的值仍然常见。

### 实际实现和关联

请记住这些超参数 (parameter) (hyperparameter)彼此关联。更改学习率可能需要调整权重 (weight)衰减。学习率调度的有效性与所选的峰值学习率有关。

以下是如何在PyTorch中定义AdamW优化器并指定这些超参数：

```python
import torch

peak_lr = 2e-4
beta1 = 0.9
beta2 = 0.98
epsilon = 1e-8
weight_decay_lambda = 0.1

decay_params = []
no_decay_params = []
for pn, p in model.named_parameters():
    if p.requires_grad:

        if (pn.endswith("bias") or
                len(p.shape) == 1):
            no_decay_params.append(p)

        else:
            decay_params.append(p)

optimizer_grouped_parameters = [
    {'params': decay_params, 'weight_decay': weight_decay_lambda},
    {
        'params': no_decay_params,
        'weight_decay': 0.0

    }
]

optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters,
    lr=peak_lr,

    betas=(beta1, beta2),
    eps=epsilon
)

print(
    f"优化器已创建，包含 {len(optimizer_grouped_parameters)} "
    f"个参数组。"
)
print(
    f"组 0（衰减）："
    f"{len(optimizer_grouped_parameters[0]['params'])} 个张量，"
    f"weight_decay="
    f"{optimizer_grouped_parameters[0]['weight_decay']}"
)
print(
    f"组 1（无衰减）："
    f"{len(optimizer_grouped_parameters[1]['params'])} 个张量，"
    f"weight_decay="
    f"{optimizer_grouped_parameters[1]['weight_decay']}"
)
```

> 此PyTorch代码片段示例展示了如何使用独立的参数组初始化AdamW优化器，以选择性地应用权重衰减。偏置 (bias)和归一化 (normalization)层参数通常会关闭权重衰减。

**要点：**

- 从研究论文中针对类似LLM训练设置被证明有效的超参数开始。
- 优先调整峰值学习率及其相关调度（预热步数、衰减类型）。
- 根据验证性能调整权重衰减以控制过拟合 (overfitting)。
- 保持AdamW的β1\beta\_1β1​和β2\beta\_2β2​为默认值（0.90.90.9, 0.9990.9990.999），如果观察到稳定性问题，可以尝试β2=0.98\beta\_2=0.98β2​=0.98。除非必要，否则避免大量调整。
- 保持ϵ\epsilonϵ为默认值（1e−81e-81e−8），除非出现特定的数值精度问题，尤其是在混合精度训练中。
- 采用日志记录和监测（第24章）来跟踪损失曲线、梯度范数和其他指标，以诊断与超参数选择相关的问题。

找到最优的超参数集合是一个迭代过程。仔细选择，依据既定实践并通过监测进行经验观察，对于应对大规模模型优化的复杂性非常重要。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 18 Hardware Considerations Llm Training

### GPU 架构 (NVIDIA Ampere, Hopper)

# GPU 架构 (NVIDIA Ampere, Hopper)

训练大型语言模型推动了计算的极限，使得专用硬件不仅有益，而且通常必不可少。尽管存在各种加速器，但 NVIDIA GPU 因其成熟的软件生态系统（CUDA）和专门为深度学习 (deep learning)工作负载设计的硬件，已成为大多数大规模 AI 训练工作的主力。理解这些 GPU 的架构特点，尤其是 Ampere 和 Hopper 等近期代系，对任何计划或执行 LLM 训练的人都很有帮助。

NVIDIA GPU 的核心是数千个称为 CUDA 核心的处理单元，它们实现大规模并行，适用于神经网络 (neural network)中常见的矩阵乘法及其他操作。然而，深度学习的真正加速来自名为张量核心 (Tensor Cores) 的专用单元。张量核心首次出现在 Volta 架构中，并在后续代系中大幅增强，它们以比标准 CUDA 核心在 FP32 精度下操作高得多的吞吐量 (throughput)执行混合精度矩阵乘累加运算。

### NVIDIA Ampere 架构（例如 A100）

NVIDIA Ampere 架构以 A100 GPU 为代表，在推出时代表了 AI 训练的显著进步。它带来了几项直接有利于大型模型训练的改进：

1. \*\*第三代张量核心：\*\*Ampere 张量核心扩展了支持的精度范围。重要的一点是，它们引入了 TensorFloat-32 (TF32)。TF32 在乘法精度上使用与 FP16 相同的 10 位尾数，但保留了 FP32 的 8 位指数，在较低精度的速度/内存优势与 FP32 的数值范围之间取得了良好平衡。这通常可以在许多框架中实现接近 FP32 的精度，并带来明显更高的吞吐量 (throughput)（相比上一代在 FP32 上的矩阵运算，理论速度提升高达 8 倍），同时无需明确的代码修改。Ampere 还提升了 FP16、BF16（BFloat16，提供比 FP16 更宽的范围，有利于训练稳定性）以及用于推理 (inference)加速的 INT8/INT4 的性能。此外，这些张量核心引入了细粒度结构化稀疏性，如果权重 (weight)矩阵的部分可以按特定模式剪枝，则可以实现加速，有可能使吞吐量翻倍。

   ```python
   import torch

   print(f"TF32 enabled on matmul: {torch.backends.cuda.matmul.allow_tf32}")
   print(f"TF32 enabled on cuDNN: {torch.backends.cudnn.allow_tf32}")
   ```
2. \*\*增加的 HBM2e 内存：\*\*大型模型需要大量内存来存储参数 (parameter)、激活值、优化器状态和中间梯度。A100 提供 40GB 或 80GB 高带宽内存 (HBM2e) 配置，相比前几代提供了明显更大的容量和带宽（最高约 2 TB/s）。这种更大的内存占用直接支持训练更大的模型或使用更大的批次大小，而不会遇到内存不足错误。
3. \*\*第三代 NVLink：\*\*训练单个 GPU 内存无法容纳的模型，需要将模型和计算分布到多个 GPU 上。这些 GPU 之间的通信速度成为瓶颈。Ampere 采用第三代 NVLink，相比标准 PCIe 通道，提供高得多的 GPU 到 GPU 直接带宽（例如，每块 A100 总带宽 600 GB/s）。这对于高效实现模型并行（张量并行、流水线并行）和减少数据并行中的通信开销非常重要。

digraph G {
layout=neato;
node [shape=box, style=filled, fillcolor="#a5d8ff", fontsize=12];
edge [color="#495057", fontsize=12];
GPU1 [pos="0,1!"];
GPU2 [pos="2,1!"];
GPU3 [pos="0,-1!"];
GPU4 [pos="2,-1!"];

subgraph cluster\_nvlink {
label = "NVLink 连接";
style=dashed;
color="#adb5bd";
GPU1 -- GPU2 [len=2];
GPU3 -- GPU4 [len=2];
GPU1 -- GPU3 [len=2];
GPU2 -- GPU4 [len=2];
GPU1 -- GPU4 [len=2.8]; // 4-GPU 设置中常见的对角连接
GPU2 -- GPU3 [len=2.8];
}
HostCPU [pos="1,-2.5!", shape=ellipse, fillcolor="#ced4da"];
HostCPU -- GPU1 [style=dashed, color="#868e96", label="PCIe"]; // 代表 PCIe 连接
HostCPU -- GPU2 [style=dashed, color="#868e96"];
HostCPU -- GPU3 [style=dashed, color="#868e96"];
HostCPU -- GPU4 [style=dashed, color="#868e96"];
}

```
```
> NVLink 在服务器节点中提供 GPU 之间的高带宽直接连接的简化视图，相比之下，连接到主 CPU 的 PCIe 连接速度较慢。
```

4. \*\*多实例 GPU (MIG)：\*\*虽然较少直接用于单一、大规模训练运行，但 MIG 允许将单个 A100 分区为最多七个独立的 GPU 实例，每个实例都有自己的内存、缓存和计算核心。这有助于通过并行运行较小的推理工作负载或开发任务来最大限度地提高利用率。

### NVIDIA Hopper 架构（例如 H100）

Hopper 架构以 H100 GPU 为典型代表，在 Ampere 的基础上构建，其功能明确针对 GPT-4 级别 Transformer 等大型模型的需求。

1. \*\*第四代张量核心和 Transformer 引擎：\*\*Hopper 引入了对新 8 位浮点格式 (FP8) 的支持，该格式由两种变体组成：E4M3（4 位指数，3 位尾数）和 E5M2（5 位指数，2 位尾数）。FP8 相比 FP16/BF16 提供两倍的吞吐量 (throughput)和一半的内存占用。重要的是，Hopper 包含 Transformer 引擎。该引擎使用软件和硬件启发式方法，在训练期间动态分析层统计数据，并决定在 Transformer 层内的特定矩阵乘法中使用 FP8 还是 FP16/BF16，同时保持更高精度的累加和高保真结果。这旨在提供 FP8 的速度和内存优势，同时无需大量手动调整或牺牲模型精度，这对于万亿级参数 (parameter)模型尤其有利。

digraph G {
rankdir=LR;
node [shape=record, style=filled, color="#adb5bd", fontsize=24];
Input [label="{输入 (FP16/BF16)}", fillcolor="#e9ecef"];
TransformerEngine [label="{Transformer 引擎|{分析统计|选择精度}}", shape=record, fillcolor="#96f2d7"];
TensorCoreOp [label="{第四代张量核心|{FP8 或 FP16/BF16 矩阵乘法}}", fillcolor="#a5d8ff"];
Output [label="{输出 (FP16/BF16)}", fillcolor="#e9ecef"];

Input -> TransformerEngine:f0;
TransformerEngine:f1 -> TensorCoreOp;
TensorCoreOp -> Output;
}

```
```
```

> 该流程图显示 Transformer 引擎分析层统计数据，以为 Hopper 的张量核心选择最佳精度（FP8 或更高）。

2. \*\*HBM3 内存：\*\*H100 使用 HBM3 内存，相比 A100 的 HBM2e，提升了容量（通常为 80GB）并大幅增加了带宽（最高约 3.35 TB/s）。这进一步缓解了内存瓶颈，允许更大的模型、激活值或训练数据批次直接驻留在 GPU 上。
3. \*\*第四代 NVLink 和 NVLink 交换系统：\*\*Hopper 增加了其 GPU 到 GPU NVLink 直接连接的带宽（每块 H100 总带宽最高达 900 GB/s）。更重要的是，NVIDIA 引入了 NVLink 交换系统。这允许在专用“NVLink 域”内连接多达 256 个 H100 GPU，直接在 GPU 之间提供全到全的高带宽通信，对于域内的某些通信模式，无需通过 InfiniBand 等较慢的网络结构。这旨在大幅提高跨越多个节点的超大型模型训练运行的扩展效率。
4. \*\*DPX 指令：\*\*Hopper 包含旨在加速涉及动态规划算法的新指令。尽管可能适用于序列比对或生物信息学等领域，但它们对标准 Transformer 训练的直接影响可能不如张量核心或内存带宽的改进那么明显。

### 代系比较总结

从 Ampere (A100) 到 Hopper (H100) 为 LLM 训练带来了实在的好处：

| 特性 | Ampere (A100) | Hopper (H100) | 对 LLM 的重要性 |
| --- | --- | --- | --- |
| **张量核心代系** | 第三代 | 第四代 | 更高吞吐量 (throughput)，支持新精度。 |
| **精度** | TF32, FP16, BF16 | FP8 (通过 Transformer 引擎), FP16, BF16 | FP8 大幅提升速度并减少内存。 |
| **内存类型** | HBM2e | HBM3 | 更高带宽和容量支持更大模型。 |
| **最大内存** | 80 GB | 80 GB (SXM 变体) | 容纳更大的状态和激活。 |
| **内存带宽** | 约 2.0 TB/s | 约 3.35 TB/s | 更快的数据访问，更少内存限制的执行。 |
| **NVLink 代系** | 第三代 (600 GB/s) | 第四代 (900 GB/s) + 交换系统 | 更快的 GPU 间通信，更好的大规模扩展。 |

A100 (BF16/FP16)H100 (FP16/BF16)H100 (FP8)0500100015002000

> 张量核心运算的大约理论峰值吞吐量凸显了明显的性能提升，尤其是在 Hopper 上使用 FP8 时。实际性能因工作负载而异。

尽管 Hopper 提供优越性能，但 Ampere GPU 仍然强大且广泛使用。在它们之间或与其他潜在硬件的选择，涉及在性能需求、预算限制和硬件可用性之间取得平衡。理解这些架构差异有助于选择合适的硬件，并配置训练任务以有效发挥其特定优势。

获取即时帮助、个性化解释和交互式代码示例。

---

### TPU 架构（Google TPU）

# TPU 架构（Google TPU）

虽然 GPU 提供了广泛用于深度学习 (deep learning)的通用并行计算能力，但 Google 开发了张量处理单元（TPU），专门用于加速神经网络 (neural network)工作负载，特别是 Transformer 模型中占主导地位的密集矩阵乘法和向量 (vector)运算。TPU 是应用专用集成电路（ASIC），从头开始设计，旨在提高机器学习 (machine learning)的性能和效率，尤其是在大规模应用时。

### 核心设计理念：矩阵乘法加速

多数 TPU 版本的核心是矩阵乘法单元（MXU）。与 GPU 核心中成千上万个简单的算术逻辑单元（ALU）不同，MXU 是一个专用硬件模块，旨在极快地执行矩阵乘法。它通常以*脉动阵列*的形式运行。

想象数据流经一个处理单元网格。在脉动阵列中，输入从边缘进入，在每个处理单元进行运算（执行乘法和加法），部分结果系统地流向相邻单元，最终结果从另一边缘输出。这种设计最大限度地减少了芯片上的数据移动，这是主要瓶颈，从而为矩阵乘法的特定任务实现了高吞吐量 (throughput)和高能效。

G

A1

w11

B1

ALU

A1->B1

B2

ALU

B1->B2

C1

ALU

B1->C1

A2

w21

A2->B2

B3

ALU

B2->B3

C2

ALU

B2->C2

A3

w31

A3->B3

C3

ALU

B3->C3

C1->C2

Y1

y1

C1->Y1

C2->C3

Y2

y2

C2->Y2

Y3

y3

C3->Y3

X1

x1

X1->A1

X2

x2

X2->A2

X3

x3

X3->A3

> 脉动阵列中数据流（权重 (weight) `w` 和激活值 `x`）的示意图，并得到输出 `y`。处理单元（ALU）执行乘加运算。

这种专用硬件意味着 TPU 擅长处理 Transformer 中普遍存在的密集矩阵运算，但对于需要更通用并行计算的工作，其灵活性可能不如 GPU。

### TPU 代次和 Pod 架构

Google 迭代了多个 TPU 代次（v2、v3、v4、v5e、v5p），每个代次都在计算能力（以每秒拍次运算 - 101510^{15}1015 次运算衡量）、内存容量（高带宽内存 - HBM）以及重要的互连速度方面提供了大幅改进。

- **TPU v2/v3:** 确立了核心架构并引入了 TPU Pod 的设计。Pod 通过专用的、高速、低延迟的二维环面芯片间互连（ICI）连接数百或数千个 TPU 芯片。该网络不同于标准以太网或 InfiniBand，并针对大规模分布式训练中常见的集体通信模式（如 `all-reduce`）进行了调整。
- **TPU v4:** 相较于 v3，性能有了显著提升，改进了 ICI 带宽和拓扑结构，并成为训练许多大型模型的主力。每个芯片都拥有专用的 ICI 链接。
- **TPU v5e/v5p:** 进一步提高了性能和效率。例如，TPU v5p 与 v4 相比，显著提升了每秒浮点运算次数（FLOPS）和 HBM 带宽，对训练规模日益增大的 LLM 非常重要。这些新代次通常具有改进的 ICI 拓扑结构，以实现更好的扩展性。

v3v4v5p0100200300400

> TPU 各代次每个芯片近似 bfloat16 峰值性能对比。请注意，实际性能会因工作负载和系统配置而异。

Pod 架构对大型模型来说是一个明显优势。训练最先进的 LLM 通常需要数百或数千个加速器共同工作。TPU Pod 内的高带宽、低延迟 ICI 使这些芯片能够高效通信，使得数据、张量和流水线并行等分布式训练方法比仅仅依靠 GPU 节点之间标准数据中心网络更为有效。

### 软件生态系统和 PyTorch 集成

虽然最初与 TensorFlow 紧密结合，但 TPU 现在拥有更广泛的框架支持。对于 PyTorch 用户来说，一个重要方面是 `torch_xla` 库。PyTorch/XLA 充当桥梁，将 PyTorch 操作编译为 XLA（加速线性代数）表示，然后可以在 TPU 硬件上高效执行。

使用 TPUs 与 PyTorch 通常涉及：

1. **安装：** 设置与目标 TPU 环境（通常在 Google Cloud 上）兼容的特定 `torch` 和 `torch_xla` 版本。
2. **设备指定：** 明确将模型和数据移动到 TPU 设备。

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

if xm.xla_available():

    device = xm.xla_device()
    print(f"Using XLA device: {device}")
else:
    print("XLA device not found. Using CPU/GPU instead.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

虽然核心 PyTorch 模型定义通常保持不变，但使用 `torch_xla` 需要理解 XLA 编译原理，并使用特定函数进行分布式训练编排（`xm.optimizer_step`、`xm.all_reduce` 等），这与原生的 PyTorch 分布式（`torch.distributed`）不同。

### TPU 与 GPU 在 LLM 训练中的权衡

- **性能：** 对于以密集矩阵乘法为主的工作负载（如标准 Transformer），TPU 因其专用 MXU，可以在每美元或每瓦特方面提供更优的性能。其高速 ICI 对于大规模分布式训练具有明显优势。
- **灵活性：** GPU 通常对于各种计算任务更灵活。CUDA 为通用 GPU 计算提供了成熟且广泛使用的编程环境。
- **精度：** TPU 强烈偏好 `bfloat16` 格式，该格式提供与 `fp32` 相似的范围但精度较低，从而平衡了深度学习 (deep learning)的稳定性和性能。现代 GPU 也有效地支持 `bfloat16` 和 `fp16`。
- **内存：** 高端 GPU 和 TPU 都使用 HBM，但容量和带宽因代次和模型而异。内存仍然是两者的一个重要限制。
- **生态系统与可用性：** GPU 生态系统（尤其是 NVIDIA 的）可以说更广泛，在云提供商和本地部署中都有更广的硬件可用性。TPU 主要通过 Google Cloud Platform 提供。
- **软件：** 虽然 `torch_xla` 使 PyTorch 能够在 TPU 上运行，但 JAX 和 TensorFlow 生态系统在 TPU 支持方面有更长的历史，因此其原生开发体验和工具可能感觉更完善。调试 XLA 编译问题有时可能具有挑战性。

选择 TPU 还是 GPU 取决于训练任务的具体规模、模型架构、预算限制、框架偏好和平台可用性。对于需要大规模并行计算的超大型模型，TPU Pod 的一体化设计和高速互连提供了一个值得考虑的选项。

获取即时帮助、个性化解释和交互式代码示例。

---

### 内存需求（HBM、GPU显存）

# 内存需求（HBM、GPU显存）

大型语言模型的庞大规模对原始计算能力以及内存系统都带来了很大压力。训练拥有数百亿参数 (parameter)的模型，需要认真考虑加速器硬件（如GPU和TPU）上可用内存的*容量*（可存储多少数据）和*带宽*（数据访问速度）。内存容量或带宽不足可能成为主要的瓶颈，严重限制模型大小和训练效率。

训练期间GPU显存 (VRAM)（常称为VRAM，即视频随机存取内存）的主要消耗者包括：

1. **模型参数：** 神经网络 (neural network)的权重 (weight)和偏置 (bias)。所需内存与参数数量和所用精度（例如FP32、FP16、BF16）直接相关。
2. **优化器状态：** Adam或AdamW等优化器为每个参数维护运行平均值（动量和方差估计）。这通常会使仅参数所需的内存翻倍或三倍。
3. **梯度：** 在反向传播 (backpropagation)期间，会为每个参数计算梯度。这些通常需要与参数相同数量的内存，并使用相同的精度。
4. **激活值：** 这些是每个层的中间输出，在前向传播期间计算并存储，以供反向传播期间使用。激活值的内存消耗是高度动态的，取决于批次大小、序列长度、模型隐藏层大小和层数。对于大型模型和长序列，激活值通常消耗GPU显存的最大部分。
5. **工作区内存：** GPU内核（例如cuDNN）进行中间计算所需的临时存储。

我们来考虑一个参数和优化器内存的简化计算。对于一个拥有 NNN 个参数、使用32位浮点（FP32）精度（每个参数4字节）的模型，仅参数所需的内存就是 4N4N4N 字节。如果使用AdamW等优化器（它为每个参数存储两个状态，即动量和方差，通常也以FP32存储），优化器状态还需要额外 2×4N=8N2 \times 4N = 8N2×4N=8N 字节。梯度则额外增加 4N4N4N 字节。在FP32精度下，参数、梯度和AdamW状态的总静态内存大约是 4N+8N+4N=16N4N + 8N + 4N = 16N4N+8N+4N=16N 字节。对于一个1750亿参数的模型（如GPT-3），仅此一项就需要 16×175×109≈2.816 \times 175 \times 10^9 \approx 2.816×175×109≈2.8 TB，远远超过任何一块当前GPU的显存容量。这项计算说明了为什么混合精度训练（使用FP16或BF16等16位格式）和分布式技术是十分必要的。

使用混合精度（例如BF16，每个参数2字节）会大幅降低这种静态内存占用。参数可以存储在BF16中（2N2N2N 字节），梯度在BF16中计算（2N2N2N 字节），而优化器状态可能为了稳定性而保留在FP32中（8N8N8N 字节），从而大约总共需要 2N+2N+8N=12N2N + 2N + 8N = 12N2N+2N+8N=12N 字节。即使有了这种减少，内存需求仍然很大。

```python
import torch

num_params_billion = 175
bytes_per_bf16 = 2
bytes_per_fp32 = 4

param_memory_bf16 = num_params_billion * 1e9 * bytes_per_bf16

grad_memory_bf16 = num_params_billion * 1e9 * bytes_per_bf16

optimizer_memory_fp32 = 2 * num_params_billion * 1e9 * bytes_per_fp32

total_static_memory_mixed = (
    param_memory_bf16
    + grad_memory_bf16
    + optimizer_memory_fp32
)
total_static_memory_tb = total_static_memory_mixed / (1024**4)

print(
    f"约 {num_params_billion} 亿参数的静态内存（BF16参数/梯度，FP32 AdamW状态）：{total_static_memory_tb:.2f} TB"
)
```

激活值内存使用通常是最具有挑战性的部分。在Transformer模型中，每层存储的激活值大小大致与 批次大小×序列长度×隐藏层大小×层数批次大小 \times 序列长度 \times 隐藏层大小 \times 层数批次大小×序列长度×隐藏层大小×层数 成比例。由于LLM中的序列长度（LLL）可以很长（数千个token），并且批次大小（BBB）为了效率而增加，因此激活值所需的 O(B⋅L⋅dmodel⋅Nlayers)O(B \cdot L \cdot d\_{model} \cdot N\_{layers})O(B⋅L⋅dmodel​⋅Nlayers​) 内存很快就会占据主导地位，尤其是在使用标准反向传播时，它需要缓存前向传播中的激活值。通常采用激活检查点（或梯度检查点）等方法来缓解此问题，通过在反向传播期间重新计算激活值而不是存储它们，以此来权衡计算时间的增加和内存使用的减少。

### 高带宽内存（HBM）

鉴于庞大的数据需求，仅仅拥有足够的GPU显存 (VRAM)容量是不够的；内存还必须足够快，才能满足GPU计算核心的数据需求。HBM正是在这一点上显得重要。

HBM是一种堆叠式DRAM技术，旨在通过硅中介层与GPU计算核心非常紧密地放置。这种紧密性以及非常宽的通信总线（例如1024位或更多，相比之下，典型GDDR内存为256/384位）能实现比消费级显卡上使用的传统GDDR内存高得多的内存带宽。

用于大规模训练的现代计算GPU（如NVIDIA的A100或H100系列）完全依赖HBM（特别是HBM2e或HBM3）。高带宽（通常每块GPU达到1.5-3 TB/s）对于以下几个方面十分重要：

1. **供给计算单元：** LLM训练涉及大量的矩阵乘法和其他操作。这些操作通常受内存限制，这意味着速度受到从内存中获取数据（权重 (weight)、激活值）的速度影响。高带宽确保计算单元得到高效运用。
2. **梯度和参数 (parameter)更新：** 参数、梯度和优化器状态的频繁读写从高带宽中获得了很大益处。
3. **GPU间通信：** 在分布式设置中，GPU之间传输的数据通常源自或目标于GPU显存。更快的内存访问速度能加快这些通信步骤。

GDDR6 (典型)HBM2e (A100 80GB)HBM3 (H100 80GB)0500100015002000250030003500GPU显存带宽对比 (GB/s)内存类型 / GPU约峰值带宽 (GB/s)

> 不同GPU显存技术的约峰值内存带宽。HBM相比GDDR提供了更高的带宽，这对LLM训练性能非常重要。

虽然HBM提供了卓越的带宽，但其制造复杂性通常导致成本更高，且相对于GDDR，每美元的容量可能更低。因此，硬件选择需要平衡高带宽需求（倾向于配备HBM的GPU）与所需的总容量和预算限制。每块GPU的显存容量（例如40GB、80GB或更高）直接决定了模型（或分布式训练中的模型分片）的大小上限，这会影响并行策略（数据并行、张量并行、流水线并行、ZeRO）的选择，以适应训练期间的模型及其相关状态。了解这些内存限制对于设计大型语言模型的高效且可行的训练设置来说非常重要。

获取即时帮助、个性化解释和交互式代码示例。

---

### 互连技术 (NVLink, InfiniBand)

# 互连技术 (NVLink, InfiniBand)

训练大型语言模型，无论是在单个服务器内的GPU之间还是在集群中多个服务器之间的GPU上，都会引入一个主要依赖：通信。需要交换的大量数据，例如数据并行期间的梯度或模型并行期间的激活和权重 (weight)，很容易成为瓶颈，导致昂贵的计算单元停滞。像典型千兆以太网这样的标准网络接口无法满足这些需求。因此，高性能互连技术是任何LLM训练基础设施的主要组成部分。这些技术侧重于提供两个主要特性：高带宽（数据传输速率）和低延迟（数据传输启动延迟）。

### 节点内通信: NVLink

在配备多个GPU的单个服务器中，主要通信路径通常是PCIe总线。虽然现代PCIe代（如PCIe 4.0或5.0）提供可观的带宽，但它是一个共享资源，GPU之间的通信常常必须通过CPU的内存控制器，这会增加延迟。为克服这一问题，NVIDIA开发了NVLink，这是一种直接连接GPU的专有的高速点对点互连技术。

NVLink允许同一服务器内的GPU直接从各自的高带宽内存（HBM）交换数据，无需通过PCIe总线将数据流经CPU或主系统RAM。与单独使用PCIe相比，这带来了显著更高的带宽和更低的延迟。

G

clusterₚcie

标准PCIe通信

clusterₙvlink

NVLink通信

CPU

CPU与系统内存

PCIe\_Bus

PCIe总线

CPU->PCIe\_Bus

 更高延迟
 更低带宽

GPU1\_PCIe

GPU 1

GPU1\_PCIe->PCIe\_Bus

PCIe x16

GPU1\_PCIe->PCIe\_Bus

 更高延迟
 更低带宽

GPU2\_PCIe

GPU 2

GPU2\_PCIe->PCIe\_Bus

PCIe x16

PCIe\_Bus->CPU

PCIe\_Bus->CPU

 更高延迟
 更低带宽

PCIe\_Bus->GPU2\_PCIe

 更高延迟
 更低带宽

GPU1\_NVLink

GPU 1

GPU2\_NVLink

GPU 2

GPU1\_NVLink->GPU2\_NVLink

 NVLink
(例如，双向900 GB/s)

> GPU通信路径对比。NVLink在GPU之间提供直接、高带宽连接，绕过通常用于GPU-CPU和间接GPU-GPU通信的较慢、共享的PCIe总线。

每代NVLink都增加了可用带宽。例如，NVLink 3.0（用于A100 GPU）提供每个GPU高达600 GB/s的双向带宽，而NVLink 4.0（用于H100 GPU）则将其提升至900 GB/s。将此与典型的PCIe 5.0 x16插槽进行比较，后者提供128 GB/s的双向带宽。这种显著差异对于模型并行技术（张量和流水线并行）尤其有影响，因为在这些技术中，大型中间激活或权重 (weight)段必须在处理同一模型层不同部分或完全不同层的GPU之间频繁交换。当在单个节点内的GPU之间执行时，它还显著加速了集体通信操作，例如数据并行中常用于同步梯度的`AllReduce`。

### 节点间通信: InfiniBand和高速以太网 (RoCE)

当将LLM训练扩展到单个服务器之外时，连接节点的高性能网络架构变得必要。这种架构需要处理分布在多台机器上的数百或数千个GPU之间的通信。这一领域的主要竞争者是InfiniBand和高速以太网，它们通常使用RDMA。

#### InfiniBand (IB)

InfiniBand是一种高性能计算网络标准，从零开始设计，旨在提供低延迟和高带宽。它作为一种交换式架构运行，类似于以太网，但具有针对HPC和AI工作负载优化的一些主要差异：

1. **RDMA（远程直接内存访问）：** 这是其核心功能。RDMA允许一台服务器中的网络适配器直接访问另一台服务器的内存（包括GPU内存），而无需涉及任一端的操作系统或CPU。这显著降低了延迟并释放了CPU周期。
2. **基于信用的流控制：** InfiniBand使用基于硬件的机制来阻止网络拥塞和数据包丢失，这对于RDMA和集体通信操作的性能很重要。
3. **高带宽：** InfiniBand标准持续发展，提供如HDR（每端口200 Gbps）和NDR（每端口400 Gbps）等速度，未来标准承诺提供更高的速率。

InfiniBand历来是大型AI训练集群的首选网络方案，因为它具有持续的低延迟和成熟的RDMA实现。

#### 带RoCE的高速以太网

以太网普遍存在，其进步已将其速度推向数百吉比特每秒（例如，200 GbE，400 GbE）。为了与InfiniBand在HPC/AI方面的低延迟能力竞争，RoCE（基于融合以太网的RDMA）协议被开发出来。RoCE允许RDMA操作运行在标准以太网基础设施上。

- **RoCE v1：** 直接在以太网MAC层上运行。在同一二层域内需要无损以太网网络（通常通过优先级流控制，PFC进行配置）。
- **RoCE v2：** 通过UDP/IP运行。这允许跨三层网络路由，使其更灵活，但仍强烈受益于（且通常需要）精心配置的无损网络，以实现与InfiniBand可比的性能。

RoCE的主要优势在于其利用现有以太网基础设施和专业知识的潜力，可能提供更低的入门成本。然而，与内置无损操作的InfiniBand相比，在大型以太网架构中实现必要的无损行为可能配置和管理起来更复杂。在性能方面，配置良好的现代高速以太网与RoCE可以实现与当今InfiniBand速度非常接近的延迟和带宽。

### 网络拓扑

服务器的连接方式（网络拓扑）也影响通信性能，特别是对于涉及许多节点的集体操作。在大型集群中，常用胖树（Fat-Tree）或蜻蜓（Dragonfly）等拓扑结构。它们旨在提供高二分带宽，这意味着即使许多节点需要在网络的不同部分同时交换数据，也有充足的通信能力。不足的拓扑可能导致瓶颈，即使单个链路速度很高。

### 软件集成

深度学习 (deep learning)框架和通信库抽象化了许多硬件细节。NVIDIA的NCCL（NVIDIA集体通信库）等库针对NVIDIA GPU上的集体操作（例如`AllReduce`、`Broadcast`、`AllGather`）进行了高度优化。

```python
import torch
import torch.distributed as dist
import os

def setup_distributed(backend='nccl'):
  """初始化分布式进程组。"""

  rank = int(os.environ['RANK'])
  world_size = int(os.environ['WORLD_SIZE'])
  master_addr = os.environ['MASTER_ADDR']
  master_port = os.environ['MASTER_PORT']

  dist.init_process_group(
      backend=backend,
      init_method=f'tcp://{master_addr}:{master_port}',
      rank=rank,
      world_size=world_size
  )

  torch.cuda.set_device(rank % torch.cuda.device_count())
  print(
      f"进程 {rank}/{world_size} 已在设备 "
      f"{torch.cuda.current_device()} 上初始化。"
  )
```

> PyTorch 代码，使用 NCCL 后端初始化分布式进程组。NCCL 智能选择最佳可用互连方式（NVLink、InfiniBand、RoCE）进行`dist.all_reduce`等通信操作。

当`torch.distributed`使用`nccl`后端初始化时，NCCL会探测系统硬件，并自动利用NVLink进行同一节点内GPU之间的快速通信，并利用InfiniBand或RoCE（如果可用且正确配置）进行不同节点上GPU之间的通信。因此，选择和配置正确的互连技术是一项硬件和基础设施任务，但其优势通过这些高级软件库在训练期间得以实现。

总之，高速互连是高效大规模LLM训练不可或缺的组成部分。NVLink提供服务器内GPU之间高效的高带宽、低延迟通道，这对模型并行和快速节点内集体操作非常重要。InfiniBand和高速以太网（带RoCE）提供扩展训练到多服务器所需的网络架构，支持大规模数据和流水线并行方案。InfiniBand和以太网/RoCE之间的选择涉及性能一致性、成本和配置复杂性方面的权衡，但两者都旨在最大程度地减少通信开销，否则通信开销可能严重限制训练吞吐量 (throughput)。

获取即时帮助、个性化解释和交互式代码示例。

---

### 硬件选择的权衡 (成本、性能、可用性)

# 硬件选择的权衡 (成本、性能、可用性)

本节考量了在硬件选择时需要衡量的成本、性能和可用性之间的主要取舍。

### 理解各项要素：成本、性能、可用性

硬件选择主要权衡成本、性能和可用性。

1. **成本：** 这不仅包含GPU或TPU的初始购置成本（如果是本地采购），还包含持续的运营开销。对于云端训练，成本通常按每小时使用量计算，并常根据加速器类型和区域分级。总拥有成本（TCO）是对本地部署的更全面考量，包括电力、散热、维护和网络设施。
2. **性能：** 除了原始FLOPS，LLM训练的性能受到以下因素的很大影响：
   - **内存容量 (HBM)：** 大型模型需要大量的GPU内存来存储参数 (parameter)、梯度、优化器状态和激活值。每个设备的内存不足会需要更复杂的并行策略（张量/流水线并行）或内存优化技术（如ZeRO），这可能增加通信开销或训练时间。
   - **内存带宽：** 高带宽内存 (HBM) 非常重要。数据在处理核心和内存之间传输的速度经常成为瓶颈，特别是对于Transformer模型中常见的内存密集型操作（如注意力机制 (attention mechanism)）。
   - **互连速度：** 在多节点甚至单节点内的多GPU配置中，通信链路的速度（例如，节点内的NVLink，节点间的InfiniBand/以太网）决定了并行策略的实施效率。慢速互连可能严重限制扩展效率。
   - **计算能力：** 对BF16、FP16等低精度格式的支持（常由NVIDIA Tensor Cores等专用单元加速）能大幅提高吞吐量 (throughput)并减少内存使用，相较于FP32，尽管可能需要仔细处理数值稳定性。较新的硬件也可能支持更低的精度，如FP8。
3. **可用性：** 需求高的加速器，尤其是拥有最多HBM和最快计算能力的最新一代产品，可能面临供应限制。云服务提供商在特定区域可能有名额限制或可用性问题。购置本地硬件的交货期也可能很长。可用性常常影响成本（高需求可能推高价格）和项目时间表。

### 云端与本地部署的考量

在使用云基础设施（如AWS、GCP、Azure）或构建本地集群之间做出决定，涉及不同的权衡：

- **云端：**
  - *优点：* 更快地获取硬件，可扩展性（按使用量付费），降低前期投入，能获取最新一代硬件，以及托管式基础设施。
  - *缺点：* 长期持续使用成本较高，潜在的数据传输费用，以及依赖提供商的可用性和定价结构。
- **本地部署：**
  - *优点：* 对于非常长或持续的训练任务，TCO可能较低，对硬件和软件栈有更大控制权，以及潜在的更高安全性/隐私性。
  - *缺点：* 大量前期资本支出，需要设置和维护（电力、散热、网络）方面的专业知识，硬件购置周期较长，获取最新一代硬件速度较慢。

### 分析成本-性能曲线

通常，成本与性能之间不存在线性关系。从中端加速器转向高端加速器，每美元性能的边际效益往往会递减，但对于最大的模型而言，绝对性能和内存容量可能是必需的。

8910002345678910k2345010k20k30k40k

> 示意硬件成本与性能之间的非线性关系。高端硬件提供更高的绝对性能，但通常每单位性能的成本相较于中端选项更高。

这条曲线表明，预算翻倍可能无法使有效训练速度也翻倍，尤其当互连瓶颈或低效扩展等因素起作用时。然而，最高级别的硬件可能是容纳超大型模型或实现可接受训练时间的*唯一*选择，即使每单位性能的成本更高。

### 使用PyTorch进行实际硬件评估

您可以使用PyTorch以编程方式检查一些硬件特性，这在云端或共享集群中使用不同机器类型时很有帮助。

```python
import torch
import pynvml

def get_gpu_info():
    """使用PyTorch和pynvml收集可用NVIDIA GPU的基本信息。"""
    info = []
    if not torch.cuda.is_available():
        print("CUDA不可用。无法显示GPU信息。")
        return info

    try:
        pynvml.nvmlInit()
        device_count = torch.cuda.device_count()
        print(f"找到 {device_count} 个CUDA设备。")

        for i in range(device_count):
            gpu_info = {}
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_info['id'] = i
            gpu_info['name'] = torch.cuda.get_device_name(i)

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info['total_memory_gb'] = round(mem_info.total / (1024**3), 2)

            major, minor = torch.cuda.get_device_capability(i)
            gpu_info['compute_capability'] = f"{major}.{minor}"

            gpu_info['supports_bf16'] = major >= 8

            info.append(gpu_info)

        pynvml.nvmlShutdown()

    except pynvml.NVMLError as error:
        print(f"使用NVML获取GPU信息失败：{error}")

        for i in range(torch.cuda.device_count()):
             gpu_info = {
                 'id': i,
                 'name': torch.cuda.get_device_name(i)
             }

             _, total_mem = torch.cuda.mem_get_info(i)
             gpu_info['total_memory_gb'] = round(total_mem / (1024**3), 2)
             major, minor = torch.cuda.get_device_capability(i)
             gpu_info['compute_capability'] = f"{major}.{minor}"
             gpu_info['supports_bf16'] = major >= 8
             info.append(gpu_info)

    return info

if __name__ == '__main__':
    gpu_details = get_gpu_info()
    for gpu in gpu_details:
        print(
            f"GPU {gpu['id']}: {gpu['name']}, "
            f"内存: {gpu['total_memory_gb']} GB, "
            f"计算能力: {gpu['compute_capability']}, "
            f"支持BF16: {gpu['supports_bf16']}"
        )
```

这个脚本提供了一个快速检查内存大小和计算能力的方法，这些都会影响性能特点（如BF16支持）。虽然它不捕捉互连速度或详细的架构特性，但这对于了解给定节点上的可用资源来说是一个有用的初步步骤。

### 做出选择

最终，硬件选择很大程度上取决于具体的LLM项目：

- **模型规模：** 训练一个70亿参数 (parameter)模型的需求与训练一个1750亿或1万亿参数模型的需求有显著差异。更大的模型需要每个加速器配备更多HBM的硬件（如NVIDIA A100 80GB或H100），并且常需要依赖快速互连的复杂分布式训练设置。
- **预算：** 有限预算可能倾向于使用旧一代GPU、利用云端竞价实例，或初步关注较小模型规模。极大的预算可能允许构建配备顶级硬件的专用本地集群。
- **时间表：** 如果完成时间很关键，投资更高性能（且可能更高成本）的硬件通常是必要的。可用性限制可能显著影响时间表。
- **现有基础设施：** 拥有现有本地集群或既定云合作关系的组织会将其纳入决策考量。
- **研究与生产：** 探索性研究可能容忍在成本较低的硬件上进行更长的训练时间，而生产模型训练通常优先考虑速度和可靠性，从而使得更高的成本合理化。

为LLM训练选择硬件涉及平衡这些因素。一种常见做法是，在更容易获取、成本较低的云实例上开始实验，以建立基线并调试训练设置，一旦流程得到验证，再扩展到更强大、更专业的硬件进行全面训练。理解性能特点，特别是内存容量、带宽和互连速度，对于做出符合技术要求和实际限制的明智决策非常重要。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 19 Checkpointing Fault Tolerance

### 长时间训练中检查点的必要性

# 长时间训练中检查点的必要性

训练大型语言模型（LLM）并非一个快速过程。与可能在数小时内完成训练的小型模型不同，预训练 (pre-training)一个先进的LLM通常需要连续运行大型加速器（GPU或TPU）集群数天、数周乃至数月。设想一个训练任务，使用1024块高端GPU持续30天。这表示超过730,000加速器小时。这种长时间、大规模的运行显著增加了遇到潜在中断的可能。

大规模分布式系统的现实是故障时有发生。在长时间运行中，遇到问题的可能性接近必然。这些中断可能源于多种情况：

- **硬件故障：** 单个GPU或TPU可能出现故障。整个计算节点可能因硬件故障（内存错误、电源问题）而崩溃。节点之间的网络连接，对分布式训练通信很要紧，可能变得不可靠或完全失效。
- **软件异常：** 训练脚本、深度学习 (deep learning)框架（如PyTorch）、底层库（CUDA、NCCL）乃至操作系统中都可能出现程序错误。组件间意外的交互可能导致崩溃或死锁。如果资源使用意外激增，可能发生内存不足（OOM）错误。
- **基础设施问题：** 托管计算集群的数据中心可能出现电源波动或停电。集群的计划内维护可能需要停止作业。如果使用基于云的抢占式实例或竞价实例来管理成本，这些实例可能在几乎没有通知的情况下被云服务商收回。

如果没有定期保存进度的机制，任何此类中断都会强制整个训练过程从头开始。这会带来严重后果：

- **计算资源浪费：** 在故障点之前进行的所有计算都将丢失。在我们1024块GPU的例子中，即使训练进行到一半发生故障，也将白白损失超过365,000 GPU小时，这代表了巨大的财务成本和能源消耗。
- **时间损失：** 训练运行通常处于研究项目或产品开发周期的关键路径上。从头开始会引入明显延误，可能以天或周计算，影响项目进度，并可能阻碍竞争优势。
- **挫败感增加：** 调试大型分布式作业中的故障本就复杂。不得不反复重新开始会给相关工程师增加巨大的额外负担和挫败感。

这就是**检查点**变得必不可少的地方。检查点是将训练作业的完整状态定期保存到持久存储（如分布式文件系统或云存储）的做法。这种状态不仅包括模型的参数 (parameter)（权重 (weight)），还包括精确地从上次停止的地方恢复训练所需的一切，例如优化器的状态、学习率调度器的状态、当前的训练迭代或周期数，以及数据加载器的状态。

设想一个简化的训练循环：

```python

import torch
import torch.optim as optim

model = MyLargeModel()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

for step in range(TOTAL_TRAINING_STEPS):

    if some_failure_condition():
        print("发生故障！从第 0 步重新开始。")

        raise SystemExit("训练失败")

    batch = next(iter(data_loader))
    outputs = model(batch['input_ids'])
    loss = calculate_loss(outputs, batch['labels'])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % LOG_INTERVAL == 0:
        print(f"步数: {step}, 损失: {loss.item()}")

print("训练成功完成！")
```

如果在一个百万步的训练运行中，第500,000步发生故障，为这500,000步所做的工作都白费了。检查点机制引入了保存点：

```python

import torch
import torch.optim as optim
import os

CHECKPOINT_DIR = "/path/to/persistent/storage/checkpoints"
CHECKPOINT_FREQ = 1000

def save_checkpoint(model, optimizer, step, filename="checkpoint.pt"):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"step_{step}_{filename}")
    state = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),

    }
    torch.save(state, checkpoint_path)
    print(f"在第 {step} 步保存检查点至 {checkpoint_path}")

def load_checkpoint(model, optimizer):

    latest_checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint_path:
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
        print(f"从第 {start_step} 步的检查点恢复")
        return start_step
    else:
        print("未找到检查点，从头开始。")
        return 0

model = MyLargeModel()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
start_step = load_checkpoint(model, optimizer)

for step in range(start_step, TOTAL_TRAINING_STEPS):

    try:
        batch = next(iter(data_loader))
        outputs = model(batch['input_ids'])
        loss = calculate_loss(outputs, batch['labels'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % LOG_INTERVAL == 0:
            print(f"步数: {step}, 损失: {loss.item()}")

        if step % CHECKPOINT_FREQ == 0 and step > 0:
            save_checkpoint(model, optimizer, step)

    except Exception as e:
        print(f"在第 {step} 步发生故障: {e}")
        print("退出。重新运行脚本以从最新检查点恢复。")
        raise SystemExit("训练中断")

print("训练成功完成！")
```

在这个修改后的循环中，如果发生故障，`load_checkpoint`函数（其实现细节我们稍后会讨论）可以从最近保存的检查点恢复状态，允许训练从例如第500,001步而不是第0步恢复。这大幅减少了计算资源的浪费。

尽管检查点机制的主要原因是应对意外故障的容错能力，但它也提供了操作灵活性。检查点允许计划性停机，例如为了集群的计划维护或重新配置训练作业。它们也能使训练在不同硬件上恢复，或通过从中间状态分叉来寻找不同训练路径。

考虑到LLM训练所需的巨大时间和资源投入，检查点不仅仅是便利；它是成功完成这些要求高的项目的基本要求。接下来的章节将详细说明需要保存的组件、在分布式环境中管理检查点的策略以及实现的最佳实践。

获取即时帮助、个性化解释和交互式代码示例。

---

### 保存模型状态（权重、优化器状态）

# 保存模型状态（权重、优化器状态）

为确保训练在中断后能够恢复，仅保存模型参数 (parameter)是不够的。一个完整的训练状态包含多个组成部分，它们记录了训练中断时的确切位置。未能保存并恢复这些组成部分的任何一项，都可能导致收敛不佳、结果难以复现，或训练过程无法正确继续。让我们看看需要记录的必要状态组成。

### 模型参数 (parameter)（权重 (weight)）

模型参数是检查点的一个主要部分。这些参数，通常称为权重和偏置 (bias)，它们定义了神经网络 (neural network)的学习功能。访问这些参数的标准方式在 PyTorch 中是通过 `state_dict()`。这个字典将每个层或缓冲区名称映射到其对应的张量。保存模型的 `state_dict` 能够保证在恢复时，模型从中断前已达到的确切学习表示开始。

```python

model_state = model.state_dict()
```

对于大型模型，`state_dict` 本身可能非常大，可能达到数百 GB 甚至数 TB。处理这些大文件需要仔细考量存储和 I/O 效率，尤其是在分布式设置中，我们稍后会讨论这些。

### 优化器状态

现代优化器，特别是像 Adam 或 AdamW 这样常用于训练大型语言模型的自适应优化器，它们不仅维护超参数 (parameter) (hyperparameter)（如学习率或权重 (weight)衰减），还维护内部状态。例如，Adam 会为每个参数维护梯度的第一动量（均值）和第二动量（未中心化方差）的估计值。

mt=β1mt−1+(1−β1)gtvt=β2vt−1+(1−β2)gt2\begin{aligned}
m\_t &= \beta\_1 m\_{t-1} + (1 - \beta\_1) g\_t \\
v\_t &= \beta\_2 v\_{t-1} + (1 - \beta\_2) g\_t^2
\end{aligned}mt​vt​​=β1​mt−1​+(1−β1​)gt​=β2​vt−1​+(1−β2​)gt2​​

在这里，mtm\_tmt​ 和 vtv\_tvt​ 代表参数在时间步 ttt 的移动平均值，它们基于梯度 gtg\_tgt​ 和衰减因子 β1\beta\_1β1​ 与 β2\beta\_2β2​。这些动量估计值与训练路径上的特定点对应。如果只恢复模型权重而重新初始化优化器，这些历史梯度统计信息就会丢失。优化器会实际从头开始，这会明显扰乱训练动态，可能减缓收敛速度，或使模型收敛到一个不同且可能更差的局部最小值。因此，保存优化器状态对于顺利恢复训练是必不可少的。

与模型类似，PyTorch 优化器也提供了 `state_dict()` 方法。

```python

optimizer_state = optimizer.state_dict()
```

优化器状态不仅包括动量估计（对于 Adam 等优化器），还包括内部步数计数以及可能由优化器实例管理的其他超参数。

### 学习率调度器状态

大型语言模型训练几乎普遍采用学习率调度策略。常见的策略包括学习率逐渐增加的热身阶段，以及随后的衰减阶段（例如，线性、余弦或多项式衰减）。这些调度策略对于训练稳定性和获得良好性能非常重要。

调度器的行为取决于当前的训练进度，通常以步数或 epoch 数衡量。为正确恢复学习率调度，必须保存其内部状态。这可能包括已进行的步数、上次计算的学习率，或特定调度器逻辑使用的其他内部计数器。

```python

scheduler_state = scheduler.state_dict()
```

恢复调度器状态能够保证学习率从中断点继续其预设的进程，而不是不适当地重新开始热身或衰减阶段。

### 训练进度指标

在记录核心模型和优化组件的同时，您还需要追踪训练运行的整体进度。这通常包括：

- **全局步数计数：** 跨所有 epoch 完成的训练总步数（已处理的批次）。这常被学习率调度器用于日志记录。
- **Epoch 编号：** 当前正在处理的 epoch。
- **Epoch 内迭代次数（可选）：** 有时有用，但通常从与保存的全局步数对应的 epoch 开始恢复就足够了，尤其是在大型数据集下。

保存这些计数器能够让你准确知道在训练进程和数据集迭代的何处恢复。

### 随机数生成器（RNG）状态

为了严格复现性，特别是在研究环境或调试时，保存所使用的随机数生成器（例如 Python 的 `random`、NumPy 的 `numpy.random` 以及 PyTorch 的 `torch.cuda.manual_seed_all` 状态）的状态很重要。这能够确保数据混洗、dropout 模式以及训练流程中的任何其他随机元素在恢复时保持一致。

```python
import torch
import random
import numpy as np

rng_states = {
    'python_rng_state': random.getstate(),
    'numpy_rng_state': np.random.get_state(),
    'torch_rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state_all()
}
```

### 合并检查点

在实践中，所有这些状态组成部分通常会一起保存到单个字典或结构化文件格式中。这使得管理和加载检查点变得更简单。

```python

checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'global_step': global_step,
    'epoch': current_epoch,

    'rng_states': {
        'python_rng_state': random.getstate(),
        'numpy_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all()
    }

    'loss': current_loss
}

checkpoint_path = f"checkpoint_step_{global_step}.pt"
```

通过认真保存这些组成部分，你就为容错奠定了根本。当中断发生时，你可以自信地恢复完整的训练上下文 (context)，并以最小的干扰和计算浪费继续该过程。接下来，我们将讨论如何在分布式环境中有效管理这一过程，以及优化检查点频率和存储的策略。

获取即时帮助、个性化解释和交互式代码示例。

---

### 处理分布式检查点

# 处理分布式检查点

保存单个训练过程的状态很简单，正如前面一节所述。然而，大型语言模型训练几乎总是涉及多个计算节点和设备并行工作。这种分布式特性给检查点保存过程带来了很大的复杂性。仅仅让每个工作节点独立保存其状态是不够的；需要协调以确保所保存的集体状态代表整个训练任务的一个有效且一致的快照。如果缺乏这种协调，恢复训练可能会导致行为偏差或结果不正确。

主要困难源于对一致性的要求。所有参与的进程（通常称为‘秩’，或‘rank’）必须保存其训练状态部分，且这些部分对应于计算中的*同一点*，通常是在特定训练步骤结束时。如果不同的秩在稍微不同的时间保存，例如一个秩完成了其梯度更新，而另一个秩仍在计算梯度，那么生成的检查点将不一致，很可能无法使用。

### 确保所有秩之间的一致性

确保一致性的最基本方法是同步。在启动保存操作之前，所有秩都必须同步，以确保它们在训练循环中到达了相同的逻辑点。在 PyTorch 分布式数据并行（DDP）等框架中，这通常通过使用屏障（barrier）等集体通信操作来实现。

```python
import torch
import torch.distributed as dist
import os

def save_checkpoint_distributed(
    model, optimizer, scheduler, epoch, step, checkpoint_dir
):
    """保存检查点，在所有秩之间协调。"""

    dist.barrier()

    if dist.get_rank() == 0:

        os.makedirs(checkpoint_dir, exist_ok=True)

        state = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),

        }

        checkpoint_filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        torch.save(state, checkpoint_path)
        print(f"Rank 0: 已将检查点保存到 {checkpoint_path}")

    dist.barrier()
```

在上面的示例中，`dist.barrier()` 作为一个同步点。第一个屏障确保所有秩在秩 0 开始保存之前暂停。然后，秩 0 保存必要的状态字典。重要地，对于使用 DDP 包装的模型，我们保存 `model.module.state_dict()` 来存储原始模型的参数 (parameter)，而不是 DDP 包装器本身。第二个屏障确保没有秩会继续下一个训练步骤，直到秩 0 成功完成保存操作。这可以防止在状态被保存时某些秩可能开始修改状态的竞态条件。

虽然这种秩 0 保存方法有效，但它有其局限性，尤其是在大规模情况下。将整个模型状态、优化器状态以及可能很大的梯度收集到单个秩上，会形成网络瓶颈，并对秩 0 提出大量内存要求。此外，保存过程本身也通过秩 0 变为串行。

### 分片检查点以提升可扩展性

一种更具可扩展性的方法，在您使用 ZeRO（零冗余优化器）等内存优化技术或张量/管道并行时特别适合，是保存*分片*检查点。在分片检查点中，每个秩仅保存其在整个训练状态中的一部分。

- **使用 ZeRO 的数据并行（DP）：** DeepSpeed 等带有 ZeRO 的框架会自动将模型参数 (parameter)、梯度和优化器状态分区到数据并行秩中。在保存检查点时，每个秩仅保存其分配的分片。这分散了 I/O 负载，并大幅减少了任何单个秩所需的内存。
- **张量并行（TP）：** 在 TP 中，模型层被拆分到不同设备上。每个秩仅保留某些层的权重 (weight)的一个切片。保存检查点涉及每个秩保存其特定的权重切片。
- **管道并行（PP）：** 在 PP 中，模型的不同层（阶段）位于不同的秩上。每个秩保存其管理的层状态。

DeepSpeed 和 Megatron-LM 等库提供了更高级别的 API，它们简化了管理分片检查点的许多复杂性。它们处理同步，并确保每个秩保存与其在并行配置中角色相符的正确状态。

```python

checkpoint_tag = f"epoch_{epoch}_step_{step}"
checkpoint_dir = "/path/to/sharded/checkpoints"

save_status = model_engine.save_checkpoint(checkpoint_dir, checkpoint_tag)

if save_status:
    print(
        f"秩 {dist.get_rank()}：成功保存了 "
        f"分片检查点 {checkpoint_tag}"
    )
else:
    print(f"秩 {dist.get_rank()}：未能保存分片检查点 {checkpoint_tag}")
```

当使用分片检查点时，`checkpoint_dir` 将包含多个文件，每个文件代表一个来自不同秩的分片，或包含元数据。加载过程也必须了解这种分片结构，以便在所有秩上正确重构全局状态。

### 加载分布式检查点

在分布式设置中加载检查点也需要协调。

1. **秩 0 保存：** 如果秩 0 保存了整个状态，所有秩通常需要加载这个单一的检查点文件。然而，需要注意。通常，秩 0 加载文件，然后需要在恢复训练之前将状态适当地广播或分发给其他秩。DDP 在所有秩上加载 `state_dict` 后会自动处理模型参数 (parameter)的分布，但优化器状态可能需要根据具体设置进行手动处理。

```python

def load_checkpoint_distributed(model, optimizer, scheduler, checkpoint_path):
    """将秩 0 保存的检查点加载到所有秩上。"""

    if not os.path.exists(checkpoint_path):
        print(f"警告：检查点路径 {checkpoint_path} 不存在。 "
              f"将从头开始。")
        return 0, 0

    rank_device = 'cuda:%d' % dist.get_rank()
    map_location = {'cuda:%d' % 0: rank_device}
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    model.module.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    step = checkpoint['step']

    print(f"秩 {dist.get_rank()}：已从 "
          f"{checkpoint_path} 加载检查点（轮次 {epoch}，步数 {step}）")

    dist.barrier()

    return epoch, step
```

2. **分片检查点：** 当使用 DeepSpeed 等库加载分片检查点时，相应的加载函数（例如 `model_engine.load_checkpoint`）会处理为每个秩读取适当的分片并重构分布式状态。从用户的角度来看，这个过程简单很多，因为库管理了复杂性。

```python

load_path, client_state = model_engine.load_checkpoint(
    checkpoint_dir, checkpoint_tag
)

if load_path:
    print(f"秩 {dist.get_rank()}：成功从 {load_path} 加载了分片 "
          f"检查点 {checkpoint_tag}")

    start_epoch = client_state.get('epoch', 0)
    start_step = client_state.get('step', 0)
else:
    print(f"秩 {dist.get_rank()}：未能找到检查点 "
          f"{checkpoint_tag}，将从头开始。")
    start_epoch, start_step = 0, 0
```

正确处理分布式检查点对于可靠的大规模模型训练非常重要。虽然手动实现需要仔细的同步和状态管理，但使用 DeepSpeed 或 Megatron-LM 等分布式训练框架中的功能，通常通过自动化分片和同步，提供了一种更可靠和可扩展的方案。请务必始终彻底测试您的检查点保存和加载过程，以确保它们在您的特定分布式设置中正常工作。

获取即时帮助、个性化解释和交互式代码示例。

---

### 异步检查点与同步检查点

# 异步检查点与同步检查点

在为长时间运行的训练任务实现检查点时，首要的考量是保存过程如何与正在进行的训练计算相互影响。训练在写入检查点时是完全暂停，还是可以同时进行？这引出了两种主要方式：同步检查点和异步检查点。选择它们时需要权衡简单性、一致性和性能开销。

### 同步检查点

同步检查点是最直接的方式。当检查点触发时（例如，在达到一定的训练步数或经过一段时间后），训练过程会明确地暂停所有计算。然后，它收集所需的状态组成部分：模型参数 (parameter)、优化器状态、学习率调度器状态、当前的轮次或步数，以及可能的数据加载器迭代器状态。一旦所有状态收集完毕，它们就会被序列化并写入持久化存储（如分布式文件系统或云存储）。只有在写入操作成功完成后，训练过程才会恢复计算。

在分布式训练环境下，同步检查点需要所有参与工作节点之间的协调。通常，在保存之前会使用屏障同步，以确保所有工作节点都到达同一点。一个工作节点（通常是0号节点）可能会被指定从其他节点收集状态，或者每个工作节点保存自己部分的状态。保存之后可能会使用另一个屏障，以确保所有工作节点都等到检查点完全写入后再继续。

**优点：**

- **简单性：** 更易于实现和理解。保存的状态在训练中某个特定、明确的时间点，保证所有组件和工作节点间的一致性。
- **一致性保证：** 检查点反映了其被触发时训练任务的精确状态。如果恢复，训练会精确地从该点开始。

**缺点：**

- **性能开销：** 训练在保存操作期间完全停止。将大型检查点（对于大型语言模型可能达到数百吉字节或太字节）写入存储可能需要相当长的时间，为GPU等昂贵的计算资源带来大量空闲时间。这种开销随模型大小和分布式工作节点数量的增加而增加。
- **阻塞性：** 整个训练过程被阻塞，影响整体训练吞吐量 (throughput)。

以下是一个在分布式环境中，使用类似PyTorch语法的训练循环中同步检查点的表示：

```python

def save_synchronous_checkpoint(
    rank, world_size, model, optimizer, scheduler, step, path
):

    if world_size > 1:
        torch.distributed.barrier()

    if rank == 0:
        print(
            f"Rank {rank}: Starting synchronous checkpoint save at step {step}..."
        )

        state = {
            'step': step,
            'model_state_dict': model.state_dict(),

            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()

        }
        torch.save(state, path)
        print(f"Rank {rank}: Finished synchronous checkpoint save to {path}.")
    else:

        pass

    if world_size > 1:
        torch.distributed.barrier()

model.train()
for step, batch in enumerate(data_loader):

    outputs = model(batch['input_ids'])
    loss = calculate_loss(outputs, batch['labels'])
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if step % checkpoint_interval == 0 and step > 0:
        checkpoint_path = f"/path/to/checkpoints/step_{step}.pt"

        save_synchronous_checkpoint(
            rank,
            world_size,
            model,
            optimizer,
            scheduler,
            step,
            checkpoint_path
        )
```

下图说明了同步检查点的阻塞特性。

G

clusterₛync

同步检查点

Train1

训练步 N

Barrier1

屏障同步

Train1->Barrier1

Save

保存检查点
(所有节点空闲)

Barrier1->Save

Barrier2

屏障同步

Save->Barrier2

Train2

训练步 N+1

Barrier2->Train2

> 训练在所有节点同步保存检查点时完全停止。

### 异步检查点

异步检查点旨在减轻同步保存的性能开销。其核心思路是将写入检查点这一计算开销大的I/O操作与主训练循环分离。

当检查点触发时，主训练进程会启动保存操作，但不会等待其完成。这通常通过以下方式实现：

1. **复制状态：** 在内存中快速复制所需状态（模型参数 (parameter)、优化器状态等）。这种复制操作相对于磁盘I/O应该较快。
2. **卸载I/O：** 将复制的状态移交给独立的线程、进程，甚至专门的I/O节点，以便在后台执行实际的序列化和写入持久化存储。
3. **继续训练：** 主训练循环在启动复制或移交后几乎立即恢复计算，使检查点I/O与有用计算重叠进行。

**优点：**

- **减少开销：** 最大限度地减少主训练循环停滞的时间，显著提升训练吞吐量 (throughput)和硬件利用率。阻塞时间缩短为初始状态复制所需的时间，而非完整的I/O持续时间。
- **吞吐量提升：** 通过将I/O与计算重叠，可以缩短总体训练时间。

**缺点：**

- **复杂性增加：** 正确实现异步检查点更为复杂。它需要细致地管理后台线程或进程，处理后台保存期间可能出现的错误，并确保数据一致性。
- **潜在状态偏差：** 异步保存的检查点将表示模型在保存操作完成到磁盘*之前*的短暂时间内的状态。如果在异步保存启动后不久发生故障，最新完成的检查点可能比同步情况稍旧。这种“滞后性”通常可以接受，仅代表少量可能丢失的进度（几次迭代）。
- **资源管理：** 需要管理后台保存进程所使用的资源（CPU、内存、网络带宽），使其不会过度干扰主训练计算。

实现异步检查点通常涉及使用线程或多进程库。

```python
import threading
import torch
import time
import os

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    def forward(self, x): return self.linear(x)
    def state_dict(self): return {'param': torch.randn(10, 10)}

model = DummyModel()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.1
)

checkpoint_thread = None

def background_save_task(state, path):
    """由后台线程执行的函数。"""
    print(f"Background Saver: Starting async save to {path}...")
    try:

        time.sleep(5)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
        print(f"Background Saver: Finished async save to {path}.")
    except Exception as e:
        print(f"Background Saver: Error during checkpointing: {e}")

def save_asynchronous_checkpoint(
    rank, world_size, model, optimizer, scheduler, step, path
):
    global checkpoint_thread

    if checkpoint_thread is not None and checkpoint_thread.is_alive():
        print(f"Rank {rank}: Waiting for previous async checkpoint to finish...")
        checkpoint_thread.join()

    if rank == 0:
        print(
            f"Rank {rank}: Initiating asynchronous checkpoint save at step {step}..."
        )

        state = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }

        checkpoint_thread = threading.Thread(
            target=background_save_task, args=(state, path)
        )
        checkpoint_thread.start()
        print(f"Rank {rank}: Background save launched. Training continues.")
    else:

        pass

step = 0
checkpoint_interval = 5
max_steps = 20

print("启动模拟训练循环...")
while step < max_steps:
    step += 1
    print(f"主循环：训练步 {step}")

    time.sleep(0.5)

    if step % checkpoint_interval == 0:

        checkpoint_path = f"/tmp/async_checkpoints/step_{step}.pt"
        save_asynchronous_checkpoint(
            0, 1, model, optimizer, scheduler, step, checkpoint_path
        )

if checkpoint_thread is not None and checkpoint_thread.is_alive():
    print("主循环：等待最终检查点完成...")
    checkpoint_thread.join()
print("模拟训练循环完成。")
```

下图说明了异步检查点如何将I/O与计算重叠。

G

clusterₐsync

异步检查点

clusterₘain

主训练线程

cluster\_bg

后台I/O线程

Train1

训练步 N

CopyState

复制状态
(快速)

Train1->CopyState

Train2

训练步 N+1

CopyState->Train2

SaveBG

保存检查点
(慢速I/O)

CopyState->SaveBG

启动保存

Train3

训练步 N+2

Train2->Train3

> 主训练线程仅短暂暂停以复制状态，然后继续计算，而实际保存则在后台线程中进行。

### 同步与异步检查点的选择

最佳方式取决于具体的训练设置和优先级：

- **简单性与性能：** 同步方式更简单，但开销更高。异步方式更复杂，但显著减少空闲时间。
- **存储速度：** 如果检查点存储极快（例如，高性能并行文件系统），同步保存的开销可能可以接受。如果存储I/O是瓶颈，异步保存则更具吸引力。
- **故障敏感度：** 如果故障时恢复绝对最新状态非常重要，同步检查点所保证的一致性可能更受青睐。如果损失少量迭代进度可以接受以换取更高的吞吐量 (throughput)，异步方式是一个不错的选择。
- **框架支持：** 现代分布式训练框架（如DeepSpeed或PyTorch FSDP）通常提供内置的优化检查点支持，有时包括异步选项或高度优化的同步方法，以最大限度地减少阻塞时间。通常建议利用这些框架功能。

实践中，对于检查点时间可能很长（数分钟或更久）的超大型模型，异步检查点通常更受青睐，以最大限度地利用昂贵的GPU资源，尽管增加了实现复杂性。细致的实现和测试是确保异步保存过程可靠性所必需的。

获取即时帮助、个性化解释和交互式代码示例。

---

### 检查点频率与存储管理

# 检查点频率与存储管理

决定保存检查点的频率以及如何管理生成的文件，需要权衡几个相互制约的因素：因故障而丢失计算进度的风险、保存过程本身带来的开销，以及存储的成本和可用性。恰当的平衡对高效、可靠的大规模模型训练很有必要。

### 确定检查点频率

选择检查点频率时，主要的权衡点在于最大限度减少故障时可能丢失的工作量与最大限度减少保存期间产生的开销。

- **高频率（例如，每几百步）：**
  - **优点：** 减少中断发生时训练进度的损失。您可以在更接近故障点的位置重新开始。
  - **缺点：** 由于频繁保存操作（磁盘I/O、可能的同步）的累积开销，会增加总训练时间。可能很快生成大量检查点文件。
- **低频率（例如，每几千步或每隔几小时）：**
  - **优点：** 最大限度减少检查点开销对整体训练吞吐量 (throughput)的影响。减缓存储消耗的速度。
  - **缺点：** 如果检查点之间发生故障，会增加丢失大量计算进度的风险。重新启动意味着可能需要重新计算数小时的工作量。

有几个因素会影响您特定设置的最佳频率：

1. **系统稳定性：** 您预期故障多久发生一次？在不太可靠的硬件上或经常发生抢占的环境中进行训练，可能需要更频繁地保存检查点。
2. **检查点开销：** 保存一个检查点需要多长时间？这取决于模型大小（包括优化器状态）、选择的保存方法（同步或异步）以及存储后端的速率。同步保存到慢速网络存储的大模型会有高得多的开销。请测量您特定配置下的此时间。
3. **计算成本：** 计算资源的金钱或时间成本是多少？如果计算非常昂贵，丢失哪怕一小时的工作量也可能是不可接受的，这会促使您选择更高的频率。
4. **训练阶段：** 您可以考虑动态频率。例如，在训练初期（可能不太稳定）阶段更频繁地保存检查点，而在训练稳定后减少频率。

触发检查点的常见策略包括：

- **基于迭代：** 每`N`个训练步保存一次。这提供了训练进度方面的可预测间隔。

  ```python

  import torch
  import os

  SAVE_EVERY_N_STEPS = 1000
  checkpoint_dir = "/path/to/checkpoints"
  global_step = 0

  global_step += 1

  if global_step % SAVE_EVERY_N_STEPS == 0:

      state = {
          'step': global_step,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),

      }
      checkpoint_path = os.path.join(checkpoint_dir, f"step_{global_step}.pt")
      print(f"Saving checkpoint to {checkpoint_path} at step {global_step}")

      pass
  ```
- **基于时间：** 每`H`小时保存一次。这实现起来简单，但在检查点之间训练进度的数量上可预测性较低，因为迭代速率可能变化。
- **组合方式：** 每`N`步保存一次，*或* 每`H`小时保存一次，以先发生者为准。这为防止进度过慢和长时间未保存提供了保障。

通常需要进行实验。从一个合理的频率开始（例如，每1000-5000步或每1-2小时），并根据观察到的稳定性及测量的开销进行调整。

### 管理检查点存储

LLM检查点，包含模型权重 (weight)、优化器状态，并可能包含梯度统计信息（特别是使用ZeRO Stage 3时），会非常大，根据模型大小和分布式训练策略，可从千兆字节到太兆字节不等。在长时间训练运行中创建的每一个检查点都进行存储通常不切实际，因为有存储成本和容量限制。

**存储位置的权衡：**

- **本地磁盘：** 提供最快的I/O，最大限度减少同步检查点开销。但是，存储受节点容量限制，且不具备容错性；如果节点发生故障，除非在别处有副本，否则检查点可能会丢失。
- **网络文件系统（NFS、Lustre）：** 提供跨节点的共享访问，对分布式训练来说必不可少，因为任何一个进程都可能保存或加载。I/O性能因系统配置和负载而异。通常比本地SSD慢。
- **云对象存储（S3、GCS、Azure Blob）：** 提供几乎无限的可扩展性、高持久性和可访问性。通常是存储超大检查点和长期存储的最实用选择。但是，I/O延迟通常高于本地或NFS，使得异步检查点（在后台线程/进程中保存）非常受推崇，以避免阻塞训练。成本基于存储容量和数据传输。

**保留策略：**

由于存储所有检查点不可行，您需要一个策略来决定保留哪些、丢弃哪些。

1. **仅保留最新：** 最简单的策略。保存新检查点后，删除上一个。
   - **优点：** 存储使用量最小。
   - **缺点：** 没有历史记录。如果最新检查点损坏或代表短暂的不稳定状态，您无法回滚。
2. **保留最后`K`个：** 保留最近的`K`个检查点。当保存检查点`N`时，删除检查点`N-K`。
   - **优点：** 提供有限的回滚能力。平衡存储和安全性。
   - **缺点：** 需要恰当选择`K`（例如，`K=3`或`K=5`）。仍然主要基于时间，不一定基于性能。
3. **基于验证保留最佳`M`个：** 定期监控一个验证指标（例如，困惑度）。保存与迄今观察到的`M`个最佳验证分数相关的检查点。这通常补充了保留最新检查点(s)的做法。
   - **优点：** 保留了表现良好的模型状态，对后续任务或分析有用。
   - **缺点：** 需要将验证集成到训练循环中，并可能在常规频率计划之外保存检查点。如果验证不频繁，则不能保证保存绝对*最新*的状态。
4. **保留最新 + 最佳：** 一种常见的混合方法。始终保留最新的检查点以立即恢复，并根据验证指标保留表现最好的前`M`个检查点。
5. **保留特定里程碑：** 除了基于时间近期的策略外，在特定的重要步骤（例如，1万、5万、10万步）保留检查点，以供长期参考或实验。

实施保留通常包括列出存储位置中现有的检查点，根据所选标准（步数、时间戳、验证分数）对其进行排序，并删除超出保留窗口的检查点。

```python

import os
import glob
import re

checkpoint_dir = "/path/to/checkpoints"
KEEP_LAST_K = 3

def manage_checkpoints(checkpoint_dir, keep_last_k):
    """删除较旧的检查点，只保留指定数量。"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))

    steps = []
    for ckpt in checkpoints:
        match = re.search(r"step_(\d+)\.pt$", os.path.basename(ckpt))
        if match:
            steps.append((int(match.group(1)), ckpt))

    steps.sort(key=lambda x: x[0], reverse=True)

    if len(steps) > keep_last_k:
        checkpoints_to_delete = [ckpt_path for step, ckpt_path in steps[keep_last_k:]]
        print(f"Found {len(steps)} checkpoints. Deleting {len(checkpoints_to_delete)} older checkpoints.")
        for ckpt_path in checkpoints_to_delete:
            try:
                os.remove(ckpt_path)
                print(f"Deleted {ckpt_path}")
            except OSError as e:
                print(f"Error deleting {ckpt_path}: {e}")
```

G

Train

训练步骤 N

Save

保存检查点 N

Train->Save

保存触发器

List

列出检查点

Save->List

Check

检查策略（保留/丢弃？）

List->Check

Delete

删除旧检查点

Check->Delete

丢弃旧的

Store

存储

Check->Store

保留新的

Delete->Store

> 流程展示了在保存新检查点后进行的检查点保留策略检查。

最终，频率和存储管理的选择取决于对您的训练环境的稳定性、性能特点、计算成本以及对潜在数据丢失的容忍度的仔细评估。使用明确定义的保存机制，结合基于新近度和性能的保留策略，是大型LLM训练的常见做法。

获取即时帮助、个性化解释和交互式代码示例。

---

### 从检查点恢复训练

# 从检查点恢复训练

成功保存检查点只是成功了一半；从该检查点正确恢复训练的能力对于实现容错的好处同等重要。简单的恢复，可能只加载模型权重 (weight)，会导致次优的训练过程或不正确的结果。真正的恢复需要将*整个*训练状态恢复到中断前的精确位置。

恢复完整的训练状态不仅涉及加载模型参数 (parameter)，还包括优化器、学习率调度器以及可能的DataLoader进度和随机数生成器状态。未能恢复这些组件中的任何一个都可能会使后续的训练过程失效。例如，如果自适应优化器（如AdamW）在没有其累积动量和方差估计的情况下重新启动，将有效地重置其学习路径，可能导致之前取得的重大进展失效。同样，从头开始重启带有预热和衰减的学习率调度器会显著改变优化路径。

我们来研究一下实现恢复机制所涉及的步骤。

### 定位和加载检查点

首先，你的训练脚本需要有逻辑来检测是否请求恢复操作，这通常通过命令行参数 (parameter)或指定检查点路径的配置设置来完成。通常自动寻找指定目录中最新的有效检查点是很实用的。

```python
import torch
import os
import glob

def find_latest_checkpoint(checkpoint_dir):
    """根据迭代次数查找最新的检查点文件。"""
    list_of_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pt'))
    if not list_of_files:
        return None

    latest_file = max(
        list_of_files,
        key=lambda f: int(f.split('_')[-1].split('.')[0])
    )
    return latest_file

resume_path = None
if config.resume_from_checkpoint:
    if config.resume_checkpoint_path:
        resume_path = config.resume_checkpoint_path
    else:
        resume_path = find_latest_checkpoint(config.checkpoint_dir)

if resume_path and os.path.isfile(resume_path):
    print(f"从检查点恢复训练：{resume_path}")
    checkpoint = torch.load(resume_path, map_location='cpu')
else:
    print("从头开始训练。")
    checkpoint = None
```

在将组件移动到目标设备之前，先将检查点加载到CPU (`map_location='cpu'`) 可以防止GPU内存突然升高，尤其是在多GPU配置中。

### 恢复模型、优化器和调度器状态

加载检查点字典后，你需要恢复核心训练组件的状态。

```python

model = YourTransformerModel(config)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config.learning_rate, ...
)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, ...)

start_iter = 0
best_val_loss = float('inf')

if checkpoint is not None:

    model.load_state_dict(
        checkpoint['model_state_dict'], strict=True
    )
    print("模型状态已加载。")

    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("优化器状态已加载。")
    else:
        print(
            "警告：检查点中未找到优化器状态。"
            "从头开始初始化优化器。"
        )

    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("调度器状态已加载。")
    else:
        print(
            "警告：检查点中未找到调度器状态。"
            "从头开始初始化调度器。"
        )

    if 'iteration' in checkpoint:
        start_iter = checkpoint['iteration'] + 1
        print(f"从迭代 {start_iter} 恢复。")
    if 'best_val_loss' in checkpoint:
        best_val_loss = checkpoint['best_val_loss']
        print(f"已加载最佳验证损失：{best_val_loss:.4f}")

    if 'rng_states' in checkpoint:
        torch.set_rng_state(
            checkpoint['rng_states']['torch_rng_state']
        )

        print("RNG状态已加载。")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

print(f"训练将从迭代 {start_iter} 开始/恢复")
```

一个经常被忽略的重要事项是确保优化器的状态张量在加载状态字典后被移动到正确的设备上。虽然模型参数 (parameter)通过`model.to(device)`移动，但优化器状态（例如AdamW中的动量缓冲区）驻留在优化器对象中，可能需要明确的设备放置。

### 恢复数据加载器状态

这可能是恢复过程中最复杂的部分。简单地在恢复时从数据集的开头重新启动数据加载器意味着你将重新处理在该周期内中断前已见过的数据样本。这会使该周期的训练数据分布出现偏差并延缓进度。

理想情况下，你希望数据加载器正好从中断的地方继续。策略包括：

1. **保存/加载迭代器状态：** 一些数据加载库可能会提供机制来序列化和反序列化数据迭代器的内部状态。这是最精确的方法，但可能很复杂且依赖于具体框架。
2. **跳过样本：** 更常用且实用的方法涉及保存周期内的当前迭代次数（或样本索引）。恢复时，你为*当前*周期重新初始化数据加载器，然后通过消耗并丢弃批次来快进，直到达到保存的迭代次数。

```python

effective_batch_size = config.batch_size * config.num_gpus

iterations_per_epoch = len(train_dataloader)

start_epoch = start_iter // iterations_per_epoch
resume_iter_within_epoch = start_iter % iterations_per_epoch

print(
    f"恢复到周期 {start_epoch}，从周期内的迭代 "
    f"{resume_iter_within_epoch} 开始。"
)

for epoch in range(start_epoch, config.num_epochs):

    data_iter = iter(train_dataloader)

    if epoch == start_epoch and resume_iter_within_epoch > 0:
        print(
            f"在周期 {epoch} 中跳过 {resume_iter_within_epoch} 个批次，以恢复状态..."
        )
        for _ in range(resume_iter_within_epoch):
            try:
                next(data_iter)
            except StopIteration:

                print("错误：尝试跳过数据加载器末尾。")
                break
        print("跳过完成。")

    for step_in_epoch in range(
        resume_iter_within_epoch, iterations_per_epoch
    ):
        current_global_iter = epoch * iterations_per_epoch + step_in_epoch

        try:
            batch = next(data_iter)
        except StopIteration:
            print(
                f"警告：在周期 {epoch} 的步进 "
                f"{step_in_epoch} 处，数据加载器意外耗尽。"
            )
            break

        if step_in_epoch == resume_iter_within_epoch:
             resume_iter_within_epoch = 0

    resume_iter_within_epoch = 0
```

这种跳过机制确保模型在整个训练运行中大致看到每个数据样本预期的次数，保持训练过程的完整性。像`torch.utils.data.DataLoader`与`DistributedSampler`等采样器结合使用的库，通常需要仔细处理周期种子设定（`sampler.set_epoch(epoch)`），以确保在分布式设置中数据正确混洗和分配，尤其是在恢复时。

### 分布式恢复的注意事项

在分布式训练环境（DDP、FSDP、ZeRO）中，恢复需要仔细协调：

- **一致性：** 所有进程*必须*加载完全相同的检查点文件。加载不同的检查点将导致分歧和错误。使用`torch.distributed.barrier()`来确保所有进程在加载之前都已定位到检查点。
- **分片状态：** 像DeepSpeed（带有ZeRO Stage 3）这样的框架会跨进程分片优化器状态和梯度。检查点将包含这些分片状态。框架的`load_checkpoint`函数通常会自动为每个进程加载相应的分片。直接使用`torch.load`和`optimizer.load_state_dict`可能无法正确处理这些分片状态；请依赖框架的工具。
- **数据加载：** 当使用`DistributedSampler`时，确保在恢复时正确调用`set_epoch()`，并且跳过逻辑考虑了每个进程的数据分片。每个进程在其自己的数据部分中跳过批次。

```python

load_path, client_state = model_engine.load_checkpoint(
    config.checkpoint_dir, tag=config.checkpoint_tag
)

if load_path is not None:
    print(f"从检查点 {load_path} 恢复训练")

    start_iter = client_state.get('iteration', 0) + 1

else:
    print("从头开始训练。")
    start_iter = 0
```

### 验证

恢复后，一个好习惯是验证状态是否正确恢复。一个简单的检查方法是，在恢复训练的第一步之后立即记录损失和学习率，并将其与中断前记录的值（如果可用）进行比较。它们应该非常接近，考虑到微小的浮点差异和下一个数据批次的影响。显著的偏差可能表明恢复逻辑存在问题。在启动大规模任务之前，强烈建议在小规模运行中全面测试保存/恢复功能。

通过仔细恢复模型、优化器、调度器、数据加载器位置和其他元数据，你可以确保训练在中断后继续，从而节省时间和计算资源。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 20 Mixed Precision Training Techniques

### 浮点数格式（FP32、FP16、BF16）简介

# 浮点数格式（FP32、FP16、BF16）简介

训练大型语言模型常会遇到计算瓶颈。标准的32位浮点数（FP32FP32FP32）虽然准确，但会消耗大量内存和计算资源。混合精度训练使用较低精度的格式，如16位浮点数（FP16FP16FP16）和bfloat16（BF16BF16BF16），以减轻这些问题。了解这些格式的特点是有效使用它们的第一步。

浮点数是计算机用来模拟实数的方式。每种格式都使用固定数量的位，通常分为三部分：符号位（表示数字是正数还是负数）、指数（表示数字的量级或范围）和尾数（表示精度或有效数字）。这些格式之间的考量归结为它们如何在指数和尾数之间分配位。

### FP32（单精度）

IEEE 754标准单精度浮点格式，通常称为FP32FP32FP32，使用32位。这通常是深度学习 (deep learning)框架和科学计算中的默认格式。

- **结构：** 1个符号位，8个指数位，23个尾数位。
- **范围：** 8个指数位提供了宽泛的动态范围，大约从10−3810^{-38}10−38到103810^{38}1038。这个范围通常足以在模型训练期间表示激活值和梯度，而不会频繁出现溢出（数字过大）或下溢（数字过小以至于无法准确表示，变为零）。
- **精度：** 23个尾数位提供了高精度，大约能准确表示7位十进制数字。

尽管FP32FP32FP32稳定，但它需要大量资源。每个参数 (parameter)需要4字节存储空间，且涉及FP32FP32FP32数字的计算耗费计算力 (compute)。对于拥有数十亿参数的模型来说，这很快成为一个瓶颈。

```python
import torch

fp32_tensor = torch.tensor([1.0, 2.0, 3.0])
print(f"数据类型: {fp32_tensor.dtype}")
print(f"每个元素的内存占用（字节）: {fp32_tensor.element_size()}")
```

### FP16（半精度）

IEEE 754半精度格式，或称FP16FP16FP16，与FP32FP32FP32相比将位数减半，仅使用16位。

- **结构：** 1个符号位，5个指数位，10个尾数位。
- **优点：**
  - *内存减少：* 每个数字仅使用2字节，与FP32FP32FP32相比，模型权重 (weight)、梯度和激活所需的内存减半。
  - *速度提升：* 在配备有专用单元的硬件上，如NVIDIA的Tensor Cores，操作可以明显加快，这些单元以比FP32FP32FP32操作更高的吞吐量 (throughput)进行FP16FP16FP16矩阵乘法。
- **缺点：**
  - *范围受限：* FP16FP16FP16最主要的问题是其动态范围大幅减少，因为只有5个指数位（大约从10−510^{-5}10−5到655046550465504）。这使其容易出现溢出（梯度变为`Inf`）或下溢（小梯度变为零，停止学习）。
  - *精度降低：* 尾数位减少意味着精度较低（大约3-4位十进制数字）。尽管这通常足够，但有时可能会阻碍精细操作或模型的收敛。

范围受限需要谨慎处理，常需要梯度缩放等技术（我们将在本章稍后介绍）以将值保持在FP16FP16FP16可表示的范围内。

```python
import torch

fp16_tensor = torch.tensor([1.0, 2.0, 3.0]).half()
print(f"数据类型: {fp16_tensor.dtype}")
print(f"每个元素的内存占用（字节）: {fp16_tensor.element_size()}")

small_val_fp32 = torch.tensor(1e-7, dtype=torch.float32)
small_val_fp16 = small_val_fp32.half()
print(f"FP32中的小值: {small_val_fp32}")
print(f"FP16中的小值: {small_val_fp16}")
```

### BF16 (BFloat16)

BFloat16，或称BF16BF16BF16，是谷歌大脑开发的另一种16位格式。与FP16FP16FP16相比，它提供了不同的考量。

- **结构：** 1个符号位，8个指数位，7个尾数位。
- **特点：** BF16BF16BF16保留了FP32FP32FP32的8个指数位。这使其拥有与FP32FP32FP32*相同的动态范围*，有效消除了FP16FP16FP16常遇到的溢出和下溢问题。
- **考量：** 为了在16位中保持FP32FP32FP32的范围，BF16BF16BF16牺牲了尾数位，只剩下7位用于表示精度（FP16FP16FP16有10位，FP32FP32FP32有23位）。这导致了较低的精度（大约2-3位十进制数字）。
- **优点：**
  - *稳定性：* 宽泛的动态范围与FP16FP16FP16相比大大提升了训练稳定性，常常无需梯度缩放等技术。
  - *内存减少：* 与FP16FP16FP16类似，它将内存使用量与FP32FP32FP32相比减半。
  - *速度提升：* 可以在兼容硬件（TPUs、较新的NVIDIA GPU如Ampere和Hopper）上加速。

BF16BF16BF16的较低精度通常被认为是深度学习 (deep learning)训练中可接受的，因为对于范围变化的稳定性通常比极高的精度更为重要。它已成为混合精度训练的热门选择，特别是对于大型模型。

```python
import torch

bf16_available = torch.cuda.is_bf16_supported()
print(f"BF16可用: {bf16_available}")

if bf16_available:

    bf16_tensor = (torch.tensor([1.0, 2.0, 3.0])
                   .bfloat16())
    print(f"数据类型: {bf16_tensor.dtype}")
    print(
        f"每个元素的内存占用（字节）: {bf16_tensor.element_size()}"
    )

    small_val_bf16 = small_val_fp32.bfloat16()
    print(
        f"BF16中的小值: {small_val_bf16}"
    )

else:
    print("当前硬件/PyTorch版本不支持BF16。")
```

### 总结比较

以下是主要特点的简要总结：

| 特性 | FP32（单精度） | FP16（半精度） | BF16（BFloat16） |
| --- | --- | --- | --- |
| 总位数 | 32 | 16 | 16 |
| 符号位 | 1 | 1 | 1 |
| 指数位 | 8 | 5 | 8 |
| 尾数位 | 23 | 10 | 7 |
| 动态范围 | 宽 | 窄 | 宽（与FP32相同） |
| 精度 | 高 | 中 | 低 |
| 每个值的内存占用 | 4 字节 | 2 字节 | 2 字节 |
| 稳定性风险 | 低 | 高（范围） | 低（精度） |
| 硬件支持 | 普遍 | 广泛 | 较新加速器 |

了解这些基本差异对于实施混合精度训练时做出明智决策非常重要。尽管FP16FP16FP16和BF16BF16BF16都提供了内存和潜在的速度优势，但它们不同的范围和精度特点会带来不同的稳定性考量和性能取舍，我们将在后续章节中讨论这些。

获取即时帮助、个性化解释和交互式代码示例。

---

### 低精度的好处（速度、内存）

# 低精度的好处（速度、内存）

从标准32位精度(FP32FP32FP32)过渡到16位浮点数(FP16FP16FP16)或bfloat16(BF16BF16BF16)等低精度格式，带来显著优势，主要体现在减少内存消耗和提高计算吞吐量 (throughput)。这些优势在训练大型语言模型时尤其显著，资源限制常常是主要瓶颈。

### 内存减少

使用16位格式最直接的优势在于内存需求的大幅降低。FP16FP16FP16和BF16BF16BF16都使用16位来表示一个数字，正好是标准单精度浮点数(FP32FP32FP32)所用32位的一半。这种内存减半适用于训练过程中的几个主要组成部分：

1. **模型参数 (parameter)：** 神经网络 (neural network)的权重 (weight)和偏置 (bias)。对于具有数十亿参数的模型，这直接意味着节省了数千兆字节的GPU内存。
2. **梯度：** 在反向传播 (backpropagation)过程中，为每个参数计算的梯度与参数本身具有相同的维度。以16位格式存储这些梯度，可将其内存占用量减半。
3. **优化器状态：** Adam或AdamW等优化器会为每个参数维护状态，例如动量和方差估计。对于Adam/AdamW，这通常涉及为每个参数存储两个额外的值。对这些状态使用16位格式（在适用且数值安全的情况下）会进一步减少内存用量。
4. **激活值：** 正向传播过程中计算的层中间输出通常需要为反向传播存储起来。以FP16FP16FP16或BF16BF16BF16存储这些激活值可以大幅减少所需内存，特别是对于具有大型激活张量或长序列的模型。

考虑一个以FP32FP32FP32与FP16FP16FP16存储的单个参数：

```python
import torch

param_fp32 = torch.tensor(3.14159, dtype=torch.float32)
size_fp32 = param_fp32.element_size()
print(f"FP32 Parameter Size: {size_fp32} bytes")

param_fp16 = param_fp32.to(torch.float16)
size_fp16 = param_fp16.element_size()
print(f"FP16 Parameter Size: {size_fp16} bytes")

param_bf16 = param_fp32.to(torch.bfloat16)
size_bf16 = param_bf16.element_size()
print(f"BF16 Parameter Size: {size_bf16} bytes")
```

对于一个拥有100亿参数的模型，仅将参数从FP32FP32FP32转换为16位格式，就能节省大约 10×109×(4−2)=2010 \times 10^9 \times (4 - 2) = 2010×109×(4−2)=20 吉字节的内存。如果再考虑梯度和优化器状态（它们所需的内存可以是参数的两倍或三倍），总节省量会更加可观。

这种内存减少让开发者能够：

- 在相同的硬件配置上训练显著更大的模型。
- 增加训练批次大小，可能提高训练稳定性和吞吐量 (throughput)。
- 减少所需的模型并行程度，简化分布式训练设置。

模型权重梯度优化器状态 (Adam)激活值050100150200FP32内存使用量（相对）FP16/BF16内存使用量（相对）

> 主要训练组件的相对内存消耗，比较了使用FP32FP32FP32与FP16FP16FP16或BF16BF16BF16的情况。Adam/AdamW的优化器状态通常需要参数本身两倍的内存。

### 计算速度提升

较低精度通常意味着更快的计算，特别是在配备专用处理单元的现代硬件加速器（如GPU和TPU）上。

1. **硬件加速：** NVIDIA GPU具有Tensor Cores，Google TPU则有矩阵乘法单元（MXU），两者都旨在利用混合精度高速执行矩阵乘法。例如，Tensor Cores可以比标准CUDA核心上的等效FP32FP32FP32操作更快地执行FP16×FP16→FP32FP16 \times FP16 \rightarrow FP32FP16×FP16→FP32乘积累加操作。在Transformer层（注意力和前馈网络内部）中，使用FP16FP16FP16或BF16BF16BF16执行大部分耗时的矩阵乘法，直接利用了这种硬件加速，从而显著提高了训练吞吐量 (throughput)（以FLOPS，即每秒浮点运算次数衡量）。
2. **数据移动减少：** 在不同内存级别之间移动数据（例如，从GPU的高带宽内存（HBM）到其计算核心）通常是瓶颈。由于16位值只占用32位值一半的空间，在给定内存带宽下，相同时间内可以传输两倍数量的数字。这种数据移动的减少进一步促进了整体速度提升，因为等待数据到达计算单元的时间减少了。

实际的速度提升很大程度上取决于具体的模型架构、所用硬件以及软件实现的效率（例如，借助支持Tensor Core的cuDNN等库）。然而，对于大型Transformer模型，在兼容硬件上有效使用混合精度训练与纯FP32FP32FP32训练相比，通常可以观察到1.5倍到3倍甚至更高的速度提升。

总之，FP16FP16FP16或BF16BF16BF16的采用提供了一个有吸引力的权衡：通过接受数值精度上潜在的（且通常可控的）降低，我们获得了内存使用量的大幅减少和计算速度的显著提高。这使得混合精度训练成为扩展语言模型规模和加速开发周期的重要技术。

获取即时帮助、个性化解释和交互式代码示例。

---

### FP16训练中的挑战（范围问题）

# FP16训练中的挑战（范围问题）

尽管16位浮点（FP16）提供的计算速度和内存节省很有吸引力，但从标准32位精度（FP32）转换并非没有困难。主要的挑战源于FP16格式明显更窄的*动态范围*。了解这一局限性是成功进行混合精度训练的基础。

FP32数字使用1个符号位、8个指数位和23个有效数（尾数）位。这使得它们能够表示一个广泛的数值范围，大致从1.18×10−381.18 \times 10^{-38}1.18×10−38到3.4×10383.4 \times 10^{38}3.4×1038。相比之下，FP16的IEEE 754标准使用1个符号位、5个指数位和10个有效数位。这种分配方式大大减小了可表示数字的范围。最小的正规化FP16数是2−14≈6.1×10−52^{-14} \approx 6.1 \times 10^{-5}2−14≈6.1×10−5，最大的是(2−2−10)×215=65504(2 - 2^{-10}) \times 2^{15} = 65504(2−2−10)×215=65504。小于2−142^{-14}2−14的数字可能可以表示为*非正规数*（或非规范数），提供逐步下溢，直至大约6.0×10−86.0 \times 10^{-8}6.0×10−8，但这些通常会带来硬件性能损失，并且与FP32相比，其表示范围仍然小得多。

FP32FP16BF1610​−40​10​−30​10​−20​100p110B10​20​10​30​10​40​最大值最小正规正值

> FP32、FP16和BF16（BFloat16）格式的大致最大正规正值和最小正规正值在对数尺度上的比较。请注意FP16明显更小的范围。

这种有限的范围在深度学习 (deep learning)训练中带来两个主要问题：

1. **下溢为零：** 梯度，特别是在深层网络中或对于不常更新的参数 (parameter)，可能变得非常小。如果梯度的幅度低于FP16可表示的最小正值（大约6.1×10−56.1 \times 10^{-5}6.1×10−5），它就会被向下舍入到零。当梯度变为零时，相应的权重 (weight)将不再被更新。这会有效停止模型部分区域的学习，可能阻止收敛或导致次优结果。设想一个场景，FP32中计算出的真实梯度是5×10−55 \times 10^{-5}5×10−5。在FP16中，这个值无法被精确表示，可能变为零，从而完全丢失更新信号。
2. **溢出为无穷大：** 相反，激活值或大梯度，尤其当优化器或损失函数 (loss function)中的中间计算产生大值时，可能超出FP16可表示的最大值（65504）。当这种情况发生时，该值会溢出为无穷大（`Inf`）。涉及`Inf`的操作通常会导致非数字（`NaN`）值（例如，`Inf - Inf`，`0 * Inf`）。一旦`NaN`出现在模型的权重、激活值或损失计算中，它们往往会迅速传播，破坏整个训练过程并导致其崩溃。

在PyTorch中考虑一个简单例子：

```python
import torch

small_value_fp32 = torch.tensor(5e-5, dtype=torch.float32)
small_value_fp16 = small_value_fp32.half()
print(f"FP32 value: {small_value_fp32}")
print(f"FP16 value: {small_value_fp16}")

large_value_fp32 = torch.tensor(70000.0, dtype=torch.float32)
large_value_fp16 = large_value_fp32.half()
print(f"FP32 value: {large_value_fp32}")
print(f"FP16 value: {large_value_fp16}")
```

这些范围问题对于LLM尤其重要，因为它们具有深度和复杂性。网络深层的激活值或通过长反向传播 (backpropagation)路径计算的梯度很容易超出狭窄的FP16范围。此外，梯度累积等技术，常用于LLM训练中，可能增加累积梯度的量级，如果处理不当，会面临溢出风险。

尽管这些挑战可能看起来令人望而生畏，但它们并未否定FP16的优点。重要的是采用稳定技术，最显著的就是损失缩放，我们将在接下来进行讨论。同样值得一提的是，`BFloat16`格式（我们将在本章后续部分介绍）与FP16相比牺牲了精度，但保留了与FP32相同的宽动态范围，这在很大程度上避免了这些特定的下溢和溢出问题，尽管精度较低可能会对收敛产生潜在影响。成功进行FP16训练需要管理好速度、内存和数值稳定性之间的这种精细平衡。

获取即时帮助、个性化解释和交互式代码示例。

---

### 损失缩放技术

# 损失缩放技术

使用 FP16FP16FP16 格式时面临的主要问题是其动态范围相较于 FP32FP32FP32 有限。尽管 FP16FP16FP16 提供了显著的内存和潜在的速度优势，但在反向传播 (backpropagation)期间计算的梯度很容易超出其可表示范围。小梯度可能变为零（下溢），从而丢失重要的更新信息；而大梯度可能变为无穷大或非数值（NaN）（溢出），导致训练过程崩溃。损失缩放是一项旨在缓解这些问题，特别是梯度下溢的重要技术。

其核心思想很直接：如果小梯度在 FP16FP16FP16 中容易下溢，我们可以在反向传播期间将其转换为 FP16FP16FP16 *之前*人为地放大它们。我们通过在启动反向传播*之前*，将计算出的损失值乘以一个缩放因子 SSS 来实现这一点。

考虑标准的反向传播过程，其中梯度 ggg 相对于损失 LLL 计算：g=∂L∂wg = \frac{\partial L}{\partial w}g=∂w∂L​。
通过损失缩放，我们计算相对于缩放后损失 Lscaled=S×LL\_{scaled} = S \times LLscaled​=S×L 的梯度 gscaledg\_{scaled}gscaled​：
gscaled=∂(S×L)∂w=S×∂L∂w=S×gg\_{scaled} = \frac{\partial (S \times L)}{\partial w} = S \times \frac{\partial L}{\partial w} = S \times ggscaled​=∂w∂(S×L)​=S×∂w∂L​=S×g
此缩放操作有效地提升了梯度值，使其在 FP16FP16FP16 中表示时下溢的可能性降低。

当然，这些缩放后的梯度（gscaledg\_{scaled}gscaled​）不能直接被优化器使用，因为它们不代表原始损失的真实梯度。因此，在反向传播计算梯度（可能在 FP16FP16FP16 中）之后，但在优化器更新模型权重 (weight)（通常使用 FP32FP32FP32 梯度）*之前*，我们必须通过将梯度除以相同的因子 SSS 来“反缩放”它们：
g=gscaledSg = \frac{g\_{scaled}}{S}g=Sgscaled​​
这样就恢复了原始梯度大小，现在希望能避免 FP16FP16FP16 下溢导致的信息丢失。整个过程在每次训练步进的训练循环中发生。

选择缩放因子 SSS 的主要方法有两种：静态损失缩放和动态损失缩放。

### 静态损失缩放

这是一种更简单的方法。在训练开始时选择一个固定不变的缩放因子 SSS，并在整个训练过程中使用它。

```python

S = 128.0
scaler = torch.cuda.amp.GradScaler(
    init_scale=S, growth_interval=100000000
)

optimizer.zero_grad()

with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()

scaler.step(optimizer)

scaler.update()
```

静态缩放的主要难点在于选择一个适当的 SSS 值。

- 如果 SSS 过小，可能不足以防止非常小的梯度下溢。
- 如果 SSS 过大，虽然它可以防止最终梯度下溢，但可能导致反向传播 (backpropagation)链中的中间梯度（在混合精度中也使用 FP16FP16FP16 算术计算）在反缩放步骤*之前*发生溢出。

寻找最佳静态 SSS 值通常需要手动调整和实验，这可能非常耗时。如果训练过程中梯度大小发生显著变化，可能还需要进行调整。

### 动态损失缩放

动态损失缩放通过在训练期间自动调整缩放因子 SSS 来解决静态方法的不足。典型算法如下：

1. 将 SSS 初始化为一个相对较大的值。
2. 在每次 `backward()` 反向传播 (backpropagation)后，检查计算出的梯度（在反缩放之前）是否存在溢出（即是否存在 `Inf` 或 `NaN` 值）。
3. 如果检测到溢出：
   - 跳过此批次的优化器步进（以避免使用无效梯度损坏权重 (weight)）。
   - 降低缩放因子 SSS（例如，除以 2）。
4. 如果连续若干步（`growth_interval`）未检测到溢出：
   - 增加缩放因子 SSS（例如，乘以 2）。这会试探是否可以使用更大的缩放因子，从而可能将更小的梯度也推入 FP16FP16FP16 的可表示范围。

这种动态调整有助于保持尽可能大的缩放因子，且不会引起溢出，从而最大限度地保护防止下溢，而无需手动调整。

现代深度学习 (deep learning)框架提供了自动处理此问题的工具。在 PyTorch 中，`torch.cuda.amp.GradScaler` 实现了动态损失缩放。

```python

scaler = torch.cuda.amp.GradScaler(
    init_scale=65536.0, growth_interval=2000
)

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()
```

使用 `GradScaler` 抽象了检查溢出和调整 SSS 的复杂性。您只需如示例所示包装前向传播、损失缩放、反向传播和优化器步进。

### 与梯度裁剪的交互

梯度裁剪是另一种用于稳定训练的技术（在第 17 章中讨论），它经常与混合精度一起使用。重要的是要正确地将梯度裁剪与损失缩放结合起来。标准做法是：

1. 通过 `scaler.scale(loss).backward()` 计算缩放后的梯度（gscaledg\_{scaled}gscaled​）。
2. *首先*反缩放梯度：`scaler.unscale_(optimizer)`。这会就地修改与优化器参数 (parameter)关联的梯度，将其恢复到原始比例，但现在是 FP32FP32FP32 格式。
3. 对反缩放后的 FP32FP32FP32 梯度执行梯度裁剪：`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`。
4. 执行优化器步进：`scaler.step(optimizer)`。请注意，如果已经调用过 `scaler.unscale_`，则 `scaler.step` *不会*再次反缩放。
5. 更新缩放器：`scaler.update()`。

### BF16 与损失缩放

回顾一下，`bfloat16` (BF16BF16BF16) 格式与 FP32FP32FP32 具有相同的动态范围，尽管精度降低。由于其范围比 FP16FP16FP16 宽得多，因此在使用 BF16BF16BF16 时，梯度溢出和下溢发生频率远不那么常见。因此，**使用 BF16 混合精度训练时，损失缩放通常不必要。** 这简化了相对于 FP16FP16FP16 的训练设置，前提是您的硬件高效支持 BF16BF16BF16 操作（如 NVIDIA Ampere 架构 GPU 和 Google TPU）。然而，仍然建议监控梯度，因为极端情况可能仍会从稳定技术中受益或需要它们。

总之，损失缩放，特别是如 `GradScaler` 等框架工具中实现的动态损失缩放，是使用 FP16FP16FP16 格式进行稳定有效混合精度训练的必不可少的技术。它通过动态调整梯度大小来抵消 FP16FP16FP16 有限的数值范围，从而在不牺牲训练稳定性的情况下，实现显著的内存节省和潜在的速度提升。

获取即时帮助、个性化解释和交互式代码示例。

---

### 使用 BF16 (BFloat16) 格式

# 使用 BF16 (BFloat16) 格式

尽管FP16能显著节省内存并可能提升速度，但其有限的动态范围（约 6imes10−56 imes 10^{-5}6imes10−5 到 6.5imes1046.5 imes 10^46.5imes104）常常需要通过细致的处理（比如损失缩放等方法）来应对。另一种16位格式BFloat16（脑浮点，或 BF16BF16BF16）被专门开发出来以解决这个范围限制，它牺牲了部分精度，以换取与FP32相近的更宽的动态范围。

### 了解 BF16 格式

FP16FP16FP16 和 BF16BF16BF16 之间的主要区别在于它们如何在指数位和尾数位（或有效数字位）之间分配16位。

- **FP16 (IEEE 754 半精度):** 1 个符号位, 5 个指数位, 10 个尾数位。
- **BF16:** 1 个符号位, 8 个指数位, 7 个尾数位。

请注意，BF16BF16BF16 与标准32位 FP32FP32FP32 格式使用相同数量的指数位（8位）。这使 BF16BF16BF16 拥有与 FP32FP32FP32 相同的动态范围（大约 1.18×10−381.18 \times 10^{-38}1.18×10−38 到 3.4×10383.4 \times 10^{38}3.4×1038），大幅降低了训练过程中梯度或激活值溢出或下溢的风险。但这牺牲了精度，因为 BF16BF16BF16 只有7个尾数位，而 FP16FP16FP16 有10个，FP32FP32FP32 有23个。

我们可以将位分配差异可视化：

G

fp16

FP16

符号位(1)

指数位(5)

尾数位(10)

bf16

BF16

符号位(1)

指数位(8)

尾数位(7)

> FP16和BF16格式的位分配对比。BF16优先考虑范围（指数位）而非精度（尾数位）。

### 优点与权衡

BF16BF16BF16 的主要优势在于，与 FP16FP16FP16 相比，它提升了大型模型的训练稳定性。由于其动态范围与 FP32FP32FP32 匹配，中间计算（如梯度或激活值）中出现数值溢出或下溢的可能性显著降低。这意味着复杂的动态损失缩放，尽管仍可能有用，但通常不是严格必需的，或者可以比 FP16FP16FP16 稳定训练通常所需的配置更为宽松。

权衡之处在于尾数位较少导致的精度降低。尽管深度神经网络 (neural network)常被认为对一定程度的精度降低具有容忍度，但这种差异*可能*会影响某些特定敏感任务或架构的收敛速度或最终模型精度，相比于 FP16FP16FP16 或 FP32FP32FP32。然而，对于许多大型语言模型训练场景，BF16BF16BF16 更宽范围带来的稳定性优势通常超过了潜在的精度问题。

另一个实际考量是硬件支持。BF16BF16BF16 最初在Google TPUs上引入。NVIDIA从其Ampere架构（A100 GPU）及其后续代次（例如 Hopper H100）开始添加了支持。较旧的GPU可能无法高效支持 BF16BF16BF16 操作，这使得 FP16FP16FP16 成为唯一的16位可行选择。

### 在 PyTorch 中结合 AMP 使用 BF16

与 FP16FP16FP16 类似，现代深度学习 (deep learning)框架提供便捷的封装，以便在自动混合精度 (AMP) 环境中使用 BF16BF16BF16。在PyTorch中，使用 `torch.autocast` 启用 BF16BF16BF16 非常简单。

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

model = nn.Linear(1024, 1024).cuda()
optimizer = torch.optim.AdamW(model.parameters())

data_loader = [
    (torch.randn(64, 1024).cuda(), torch.randn(64, 1024).cuda())
    for _ in range(10)
]

use_bf16 = (
    torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
)

scaler = GradScaler(enabled=False)

print(f"正在使用 BF16: {use_bf16}")

for data, target in data_loader:
    optimizer.zero_grad()

    with autocast(
        device_type='cuda',
        dtype=torch.bfloat16,
        enabled=use_bf16
    ):
        output = model(data)
        loss = nn.functional.mse_loss(output, target)

    scaler.scale(loss).backward()

    scaler.step(optimizer)

    scaler.update()

    print(f"损失: {loss.item():.4f}")
```

在此示例中：

1. 我们使用 `torch.cuda.is_bf16_supported()` 检查硬件是否支持 BF16BF16BF16。
2. 我们初始化 `GradScaler` 但将其 `enabled` 设置为 `False`。虽然损失缩放*可以*与 BF16BF16BF16 一起使用，但由于该格式的范围较宽，通常没有必要。禁用它可简化代码并消除与缩放检查相关的开销。如果您仍然遇到稳定性问题，可以考虑启用它，尽管这种情况比 FP16FP16FP16 少见。
3. 在训练循环内部，`torch.autocast` 与 `dtype=torch.bfloat16` 一起使用。这指示 PyTorch 自动将块内的操作在安全且有利的情况下（通常是矩阵乘法和卷积）转换为 BF16BF16BF16，同时将其他操作（如归约）保留在 FP32FP32FP32 中以确保数值稳定性。
4. `scaler.scale(loss).backward()` 和 `scaler.step(optimizer)` 调用在 scaler 启用或禁用时都能正确执行。如果禁用，它们实质上会成为损失和优化器步进的透传操作。

### 总结：FP16 与 BF16 的选择

FP16FP16FP16 和 BF16BF16BF16 之间的选择取决于几个因素：

1. **硬件支持:** BF16BF16BF16 需要较新的硬件（NVIDIA Ampere+、Google TPU）。如果使用较旧的硬件，FP16FP16FP16 可能是16位精度的唯一选择。
2. **训练稳定性:** BF16BF16BF16 的主要优势在于其类似 FP32FP32FP32 的动态范围，显著降低了溢出/下溢的风险，并且通常无需细致的损失缩放调整。如果 FP16FP16FP16 的稳定性难以管理，BF16BF16BF16（如果可用）是一个有力的替代方案。
3. **精度要求:** FP16FP16FP16 提供略高的精度（更多的尾数位）。如果特定任务对数值精度高度敏感，FP16FP16FP16 *可能*会产生略好的结果，前提是能保持稳定性。然而，对于许多大型模型训练任务而言，与 BF16BF16BF16 的稳定性优势相比，这种差异通常可以忽略不计。

在实践中，如果您的硬件支持 BF16BF16BF16，它通常是训练大型语言模型的首选，因为它易于使用且具有内在的稳定性，简化了混合精度训练的配置。它提供了与 FP16FP16FP16 几乎相同的内存和速度优势，但避免了管理狭窄动态范围的重大麻烦。

获取即时帮助、个性化解释和交互式代码示例。

---

### 框架对混合精度（AMP）的支持

# 框架对混合精度（AMP）的支持

手动实现混合精度训练，小心地在FP32FP32FP32、FP16FP16FP16或BF16BF16BF16之间进行张量类型转换并管理损失缩放，可能既复杂又容易出错。幸运的是，像PyTorch这样的现代深度学习 (deep learning)框架提供了内置工具来自动化此过程的大部分，通常被称为自动混合精度（AMP）。这些工具大大简化了混合精度技术的采用，使工程师能够将精力放在模型架构和训练动态上，而不是底层数值管理。

### PyTorch 自动混合精度 (`torch.cuda.amp`)

PyTorch 通过其 `torch.cuda.amp` 模块提供强有力的 AMP 支持（对于英特尔 GPU 是 `torch.xpu.amp`，对于苹果 M 系列芯片是 `torch.mps.amp` 等，尽管对于大型语言模型（LLM）来说，CUDA 最常用）。你将使用的两个主要组件是 `autocast` 上下文 (context)管理器和 `GradScaler`。

#### `autocast` 上下文管理器

`autocast` 上下文管理器为你代码的选定区域启用自动类型转换，通常是模型的前向传播。启用后，它会识别可以安全高效地以更低精度（FP16FP16FP16 或 BF16BF16BF16）运行的操作，并自动相应地转换它们的输入。被认为对数值敏感或不从更低精度中获益的操作（例如通常配置为在FP32FP32FP32中运行的归约或归一化 (normalization)层）会保持完全精度。

这是你在模型前向传播周围通常如何使用 `autocast` 的方法：

```python
import torch

scaler = torch.cuda.amp.GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for data, target in dataloader:
    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        output = model(data)
        loss = loss_fn(output, target)

    scaler.scale(loss).backward()

    scaler.step(optimizer)

    scaler.update()
```

默认情况下，在CUDA设备上，`autocast` 使用FP16FP16FP16数据类型。如果需要，你可以显式指定数据类型，例如使用BF16BF16BF16：

```python

with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(data)
    loss = loss_fn(output, target)
```

仅使用 `autocast` 可以管理前向传播精度。然而，当使用FP16FP16FP16时，其小的可表示范围可能导致反向传播 (backpropagation)过程中梯度下溢（变为零）。这使得使用梯度缩放变得必要。

#### `GradScaler`

`torch.cuda.amp.GradScaler` 在使用FP16FP16FP16时有助于防止梯度下溢。它通过在调用 `backward()` 前将损失值乘以一个缩放因子来发挥作用。这有效地放大了整个反向传播过程中的梯度，将小值推入FP16FP16FP16的可表示范围。在优化器更新权重 (weight)之前，`GradScaler` 会对梯度进行反缩放（通过除以相同的缩放因子），使其恢复到原始值，从而确保权重更新正确。如果`inf`或`NaN`值在反缩放步骤中在梯度中检测到（如果缩放因子变得过大，可能会发生这种情况），则该批次的优化器步骤将被跳过，并且缩放因子会在后续迭代中被减小。反之，如果梯度在一段时间内保持稳定，`scaler` 会增加缩放因子，以充分使用更多的FP16FP16FP16范围。

典型使用模式包括：

1. 在训练循环开始前初始化一个 `GradScaler` 实例。
2. 在调用 `backward()` 前，使用 `scaler.scale(loss)` 对损失进行缩放。
3. 使用 `scaler.step(optimizer)` 对梯度进行反缩放并调用优化器步骤。此步骤会自动检查 `inf`/`NaN` 梯度，并在必要时跳过更新。
4. 使用 `scaler.update()` 更新下一次迭代的缩放因子。

前面的代码片段已经展示了这种集成方式。请注意，当使用 `autocast` 和BF16BF16BF16时，通常*不*需要 `GradScaler`，因为BF16BF16BF16的动态范围类似于FP32FP32FP32，使得梯度下溢的可能性大大降低。如果使用BF16BF16BF16，你通常可以省略 `GradScaler` 步骤：

```python

optimizer.zero_grad(set_to_none=True)

with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(data)
    loss = loss_fn(output, target)

loss.backward()

optimizer.step()
```

然而，即使使用BF16BF16BF16，检查梯度范数发散可能仍然有用，并且一些实践者为了程序的稳定性仍然选择使用BF16BF16BF16搭配 `GradScaler`，尽管其主要原因（防止FP16FP16FP16下溢）没那么重要。

### 将 AMP 集成到训练循环中

让我们改进典型的训练循环结构，以清楚地展示FP16FP16FP16训练的 AMP 组件：

```python
import torch
import torch.cuda.amp as amp

model = YourLargeModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()
dataloader = YourDataLoader()

scaler = amp.GradScaler(enabled=True)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        with amp.autocast(enabled=True):
            predictions = model(data)
            loss = loss_fn(predictions, target)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                  f"Scale: {scaler.get_scale()}")

    model.eval()
    with torch.no_grad():
        for data_val, target_val in \
                validation_dataloader:
             data_val, target_val = (data_val.cuda(),
                                     target_val.cuda())
             with amp.autocast(enabled=True):
                 val_output = model(data_val)
```

这种结构结合了FP16FP16FP16混合精度训练的必要的 `autocast` 和 `GradScaler` 步骤。`enabled=True` 标志允许你轻松地开启或关闭 AMP，以便进行比较或调试。请注意，可选的梯度裁剪步骤需要在裁剪*之前*对梯度进行反缩放，这由 `scaler.unscale_` 处理。

### 分布式训练的注意事项

AMP 旨在与 PyTorch 的分布式训练工具（`torch.distributed.DistributedDataParallel`）以及像 DeepSpeed 或 Fully Sharded Data Parallel (FSDP) 这样的更高层级库协同工作。通常，`autocast` 可以应用于每个副本的前向传播中，而 `GradScaler` 可以管理跨进程的缩放，这通常只需要对上面所示的标准 AMP 设置进行少量更改。像 DeepSpeed 这样的框架可能提供自己的集成混合精度处理，这可能会取代或封装 PyTorch 的原生 AMP，因此请查阅它们的具体文档。

通过使用框架对 AMP 的支持，你大大降低了实现混合精度训练的复杂性。这让你更容易地使用FP16FP16FP16和BF16BF16BF16等更低精度格式的速度和内存优势，使得在现有硬件上训练大型语言模型变得更可行。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 21 Intrinsic Evaluation Metrics

### 语言模型评估方法

# 语言模型评估方法

投入大量计算资源并精心整理海量数据集来预训练 (pre-training)大语言模型 (LLM)后，随之而来的紧迫问题是：它的效果如何？仅仅完成训练过程并不能保证得到一个有用或有效的模型。我们需要严格的方法来评估其表现，了解其能力，并找出其不足。评估大语言模型是一个多方面的过程，因为“好”本身在不同情境下有不同含义，可以是从原始预测准确性，到对话中的实用性，或在特定下游应用中的表现。

用于大语言模型评估的技术通常分为几大类：

1. **内部评估：** 这种方法侧重于依据模型的主要训练目标进行评估，通常是预测下一个词元 (token)或填充被遮蔽的词元。它衡量模型在训练数据本身中学习统计模式和分布的程度。如章引言所述，困惑度是最突出的内部评估指标。这些评估直接在模型输出（概率）上进行，使用一个保留的测试集，不涉及外部任务。
2. **外部评估：** 这种方法评估模型在预训练阶段并未明确训练过的特定下游任务上的表现。例子包括问答、文本摘要、情感分析或代码生成。这通常涉及对预训练模型在任务专有数据集上进行微调 (fine-tuning)，然后使用任务专有指标（例如，F1分数、BLEU分数、准确率）来衡量其表现。外部评估提供衡量模型实用性和迁移学习 (transfer learning)能力的标准。我们将在第22章详细审视这些方法。
3. **人工评估：** 这涉及人工评估模型输出的质量，依据连贯性、相关性、实用性、无害性、事实准确性或指令遵循等标准。尽管常被认为是判断整体质量和对齐 (alignment)效果的黄金标准，但人工评估通常昂贵、耗时、主观且难以扩展。它常用于开发的后期阶段，或用于特定的对齐目标，例如通过人工反馈强化学习 (reinforcement learning)（RLHF）实现的目标，这在第26章有说明。

EvaluationTypes

LLM\_Evaluation

大语言模型评估

Intrinsic

内部评估
（例如，困惑度）

LLM\_Evaluation->Intrinsic

Extrinsic

外部评估
（下游任务）

LLM\_Evaluation->Extrinsic

Human

人工评估
（质量评分）

LLM\_Evaluation->Human

> 大语言模型评估方法的主要类别。

“本章主要关注内部评估。尽管与评估任务表现相比，它可能显得有限，但内部评估在大语言模型开发周期中扮演着重要角色。其主要优点包括：”

- **直接衡量：** 它直接量化 (quantization)了模型的核心语言模型能力，评估它捕捉其所训练语言的底层概率分布的程度。
- **计算效率高：** 计算困惑度等指标通常比设置和运行多个下游任务评估或协调人工评估快得多且所需资源较少。这有助于在开发过程中快速获得反馈。
- **开发监控：** 内部评估指标对跟踪训练进展极有帮助。在训练期间监控验证集上的困惑度有助于识别收敛情况，诊断不稳定性（如第24章所述），并对学习率或停止标准做出决定。
- **对比分析：** 它提供了一种标准方法，用于在受控条件下比较不同的模型架构、超参数 (parameter) (hyperparameter)或预训练数据变体，前提是比较是在相同的测试集上进行并使用相同的词元化方法。

训练损失与内部评估之间有直接关联。在训练期间，模型通常被优化以最小化交叉熵损失，这在数学上与困惑度相关。在PyTorch中，为评估目的计算此损失涉及在保留数据集上进行前向传播，而不进行反向传播 (backpropagation)。

```python
import torch
import torch.nn.functional as F

model.eval()
total_loss = 0
total_tokens = 0

with torch.no_grad():
    for batch in eval_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)

        loss = outputs.loss

        num_tokens = attention_mask.sum()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

average_loss = total_loss / total_tokens
```

> 简化的 PyTorch 代码片段，说明评估期间的损失计算，与困惑度相关。

然而，承认内部评估的局限性很重要。低困惑度分数并不能自动意味着模型能生成有用、真实、连贯或安全的文本。模型可以通过过拟合 (overfitting)训练数据的统计模式来实现低困惑度，可能学会生成概率高但信息量少的重复或通用序列。此外，正如本章后面会看到的那样，困惑度值对所用的特定词元化方案和评估数据集的性质高度敏感，使得跨不同设置的比较变得困难。

因此，内部评估应被视为全面评估策略的一个必要组成部分。它提供对模型语言模型能力的基本检查，但必须辅以外部评估和可能的人工评估，以全面了解大语言模型的表现及其对特定应用的适用性。我们现在将继续详细定义和分析困惑度。

获取即时帮助、个性化解释和交互式代码示例。

---

### 困惑度：定义与计算

# 困惑度：定义与计算

如前所述，内部评估衡量语言模型的核心能力：预测序列中下一个词元 (token)。我们不需要为此进行特定的下游任务；我们直接衡量模型对所训练语言的统计模式的理解程度，使用一个保留的测试数据集。这方面的标准衡量指标是**困惑度**。

### 了解困惑度

困惑度（PPL）量化 (quantization)了语言模型在预测序列中下一个词元 (token)时的不确定性。可以将其视为模型对于下一个词元所拥有的有效选择数量，在整个序列上取平均值。困惑度较低表明模型对其在给定测试数据上的预测更具信心和准确性。这表示模型对测试集中实际出现的词元赋予了更高的概率。

从数学上讲，对于一个词元序列 W=w1,w2,...,wNW = w\_1, w\_2, ..., w\_NW=w1​,w2​,...,wN​，困惑度定义为该序列的平均负对数似然的指数化值：

PPL(W)=exp⁡(−1N∑i=1Nlog⁡p(wi∣w<i;θ))PPL(W) = \exp\left(-\frac{1}{N}\sum\_{i=1}^N \log p(w\_i | w\_{<i}; \theta)\right)PPL(W)=exp(−N1​i=1∑N​logp(wi​∣w<i​;θ))

这里，p(wi∣w<i;θ)p(w\_i | w\_{<i}; \theta)p(wi​∣w<i​;θ) 是由模型（参数 (parameter)为 θ\thetaθ）赋予词元 wiw\_iwi​ 的概率，已知前面的词元 w<i=w1,...,wi−1w\_{<i} = w\_1, ..., w\_{i-1}w<i​=w1​,...,wi−1​。NNN 是序列中的词元总数。

### 与交叉熵损失的关系

如果你训练过语言模型，你会认出指数内部的这一项。平均负对数似然正是训练期间通常最小化的**交叉熵损失**。

交叉熵损失(W)=−1N∑i=1Nlog⁡p(wi∣w<i;θ)\text{交叉熵损失}(W) = -\frac{1}{N}\sum\_{i=1}^N \log p(w\_i | w\_{<i}; \theta)交叉熵损失(W)=−N1​i=1∑N​logp(wi​∣w<i​;θ)

因此，困惑度就是测试集上计算的交叉熵损失的指数值：

PPL(W)=exp⁡(交叉熵损失(W))PPL(W) = \exp(\text{交叉熵损失}(W))PPL(W)=exp(交叉熵损失(W))

这种直接关系很方便。如果你在训练期间监控验证集上的交叉熵损失，你实际上是在监控一个其指数是困惑度的值。一个训练目标是最小化交叉熵损失的模型，实际上也在训练中隐式地最小化困惑度。

### 实际计算困惑度

要计算训练好的大型语言模型的困惑度，你需要一个具有代表性的保留测试集——模型在训练期间*未曾*见过的数据。过程通常遵循以下步骤：

1. **准备测试集：** 使用模型训练期间使用的*相同*分词 (tokenization)器 (tokenizer)对测试数据进行分词。
2. **处理测试集：** 将分词后的测试序列输入到训练好的语言模型中。对于序列中的每个词元 (token)，获取模型对*下一个*词元在整个词汇表 (vocabulary)上的预测概率分布。通常，模型输出的是 logits，即未归一化 (normalization)的对数概率。
3. **计算对数概率：** 提取与测试序列中观察到的*实际*下一个词元对应的对数概率。这通常涉及对 logits 应用 softmax 函数以获得概率，然后取对数；或者更直接地使用 `log_softmax` 输出并收集相关的对数概率。
4. **取平均：** 计算测试集中所有词元的这些负对数概率的平均值。这给出了测试集的交叉熵损失。
5. **取指数：** 计算平均负对数概率（即交叉熵损失）的指数值，以获得最终的困惑度分数。

我们用一个简化的 PyTorch 示例来说明。假设 `model` 是你训练好的语言模型，`test_loader` 提供来自测试集的词元 ID 批次，而 `loss_fn` 通常是 `torch.nn.CrossEntropyLoss`（它结合了 LogSoftmax 和 NLLLoss）。

```python
import torch
import math

loss_fn = torch.nn.CrossEntropyLoss(
    ignore_index=model.config.pad_token_id
)

model.eval()
total_loss = 0.0
total_tokens = 0

with torch.no_grad():
    for batch in test_loader:

        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)

        labels = input_ids.clone()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        num_tokens_in_batch = (
            shift_labels != loss_fn.ignore_index
        ).sum().item()

        total_loss += loss.item() * num_tokens_in_batch
        total_tokens += num_tokens_in_batch

if total_tokens > 0:
    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)
    print(f"测试集交叉熵损失: {average_loss:.4f}")
    print(f"测试集困惑度: {perplexity:.4f}")
else:
    print("没有评估任何词元。")
```

> 此代码片段演示了如何使用 PyTorch 的 `CrossEntropyLoss` 计算困惑度。它处理批次数据，仅对非填充词元计算损失，聚合总损失，并通过对每个词元的平均损失取指数来计算最终困惑度。

较低的困惑度值表明模型在测试集上的概率分布更“集中”，并赋予观测到的序列更高的概率。这表示模型对测试数据感到更不“困惑”或更不“惊讶”，说明根据这种内部衡量指标，语言建模表现更好。

获取即时帮助、个性化解释和交互式代码示例。

---

### 理解困惑度得分

# 理解困惑度得分

困惑度（PPLPPLPPL）直接来源于对标记 (token)序列 W=w1,w2,...,wNW = w\_1, w\_2, ..., w\_NW=w1​,w2​,...,wN​ 计算得到的平均负对数似然，或者说交叉熵损失：

PPL(W)=exp⁡(1N∑i=1N−log⁡p(wi∣w<i;θ))PPL(W) = \exp\left( \frac{1}{N} \sum\_{i=1}^N -\log p(w\_i | w\_{<i}; \theta) \right)PPL(W)=exp(N1​∑i=1N​−logp(wi​∣w<i​;θ))

本质上，它就是 exp⁡(交叉熵损失)\exp(\text{交叉熵损失})exp(交叉熵损失)。由于较低的交叉熵损失表明训练时模型对数据拟合得更好，因此较低的困惑度得分也类似地显示了一个更优的语言模型，至少在预测评估数据集中下一个标记方面是如此。

困惑度较低的模型，平均而言，对其在测试集中遇到的标记序列“感到意外”的程度较小。它会给实际出现的标记分配更高的概率。可以这样理解：如果模型持续给正确的下一个词分配高概率，那么 −log⁡p(wi∣...)-\log p(w\_i | ...)−logp(wi​∣...) 项会很小，导致平均损失较小，从而困惑度较低。反之，频繁的意外（给正确的下一个词分配低概率）会增加损失，从而提高困惑度。

### 困惑度作为有效分支因子

一种直观地理解困惑度的方式是将其视为语言模型的*有效分支因子*。如果一个模型在给定数据集上的困惑度为，例如，100，这意味着在每个标记 (token)预测步骤中，该模型的平均不确定性等同于它必须在100个可能的下一个标记中进行均匀随机选择。较低的困惑度表明模型更有效地缩小了可能的选择范围。

G

Start

低 PPL
模型

T1\_Opt1

词语 A

Start->T1\_Opt1

 PPL=3

T1\_Opt2

词语 B

Start->T1\_Opt2

 PPL=3

T1\_Opt3

词语 C

Start->T1\_Opt3

 PPL=3

Start\_High

高 PPL
模型

T2\_Opt1

W1

Start\_High->T2\_Opt1

 PPL=5

T2\_Opt2

W2

Start\_High->T2\_Opt2

 PPL=5

T2\_Opt3

W3

Start\_High->T2\_Opt3

 PPL=5

T2\_Opt4

W4

Start\_High->T2\_Opt4

 PPL=5

T2\_Opt5

W5

Start\_High->T2\_Opt5

 PPL=5

> 困惑度较低的模型（例如 PPL=3）在每个步骤中考虑的选择比困惑度较高的模型（例如 PPL=5）更少。

一个能够以100%确定性预测下一个标记的理想模型，其困惑度将为1（因为 log⁡(1)=0\log(1) = 0log(1)=0，损失为0，且 e0=1e^0 = 1e0=1）。当然，这对于自然语言来说是无法实现的。在一个词汇量为 VVV 的情况下进行随机猜测，会产生接近 VVV 的困惑度。实际模型介于这两者之间。

### 相对与绝对理解

理解这一点非常重要：困惑度得分在*相对*意义上最有价值。您可以可靠地使用困惑度来：

1. **比较不同模型**，这些模型在*完全相同*的保留数据集上，并使用*完全相同*的标记 (token)化方案进行评估。如果模型 A 在这些相同条件下达到困惑度 35，模型 B 达到 45，那么您可以肯定地说模型 A 在根据此指标预测该特定数据集方面表现更好。
2. **跟踪单一模型在训练期间的进展。** 在训练周期中监测验证集上的困惑度是标准做法。困惑度下降通常表明模型正在有效学习。

12345050100150训练期间的验证集困惑度周期困惑度

> 验证集困惑度通常随着训练的进行而下降，表明模型拟合度提升。

然而，单独解释一个*绝对*困惑度得分是困难且常具误导性的。对于源代码或密集科学文献等复杂数据集，困惑度为 50 可能是最优水平，但对于简单的儿童故事数据集来说则相当差。基础数据的固有可预测性（或熵）对可达到的困惑度有很大影响。

此外，词汇量和所使用的具体标记化算法等因素会显著影响最终得分。比较使用不同标记器（例如 BPE 与 WordPiece）或不同词汇量获得的困惑度值通常是无效的，因为“标记”的定义发生变化，从而计算基础也随之改变。我们将在本章后面更详细地考察标记化的影响。

### 困惑度的局限性

尽管有用，但困惑度远非衡量语言模型质量的完美标准。请记住以下局限性：

- **它不衡量语义质量：** 模型可能会通过生成语法上无懈可击但无意义、重复或事实不准确的文本来获得较低的困惑度。它只关注根据训练数据预测可能的序列，而不关心统计模式中的真实性或连贯性。
- **对异常值敏感：** 对数概率的平均意味着，模型对正确下一个标记 (token)分配极低概率的单个情况可以显著地提高总体困惑度得分。
- **范围有限：** 困惑度评估模型的核心语言建模能力，但不直接衡量在问答、摘要或翻译等下游任务上的表现。一个困惑度略差的模型有时在特定应用上表现更好。
- **归一化 (normalization)问题：** 如前所述，困惑度对标记化选择和用于评估的特定数据集高度敏感。

### 实际使用

尽管存在局限性，困惑度在大型语言模型开发中仍然是一个标准指标，主要因为它具有以下特点：

- **计算成本低：** 它可以直接从模型在数据集上的前向传播计算得出，无需任务特定设置。
- **与改进相关联：** 通常，困惑度（在相关测试集上）的改进与下游任务表现的改进相关联，尽管这种关联并非完美。
- **有助于监测：** 它在训练期间提供快速的合理性检查，并提供一种在受控条件下比较架构或超参数 (parameter) (hyperparameter)变化的方法。

通常，您会使用深度学习 (deep learning)框架提供的交叉熵损失来计算困惑度。以下是一个 PyTorch 代码片段，说明了这种关系：

```python
import torch
import torch.nn.functional as F

logits = model_outputs.view(-1, model_outputs.size(-1))

targets = target_ids.view(-1)

padding_idx = -100
average_neg_log_likelihood = F.cross_entropy(
    logits,
    targets,
    ignore_index=padding_idx
)

with torch.no_grad():
  perplexity = torch.exp(average_neg_log_likelihood)

loss_val = average_neg_log_likelihood.item()
print(f"平均交叉熵损失: {loss_val:.4f}")
perplexity_val = perplexity.item()
print(f"困惑度: {perplexity_val:.4f}")
```

总之，尽管困惑度提供了一个有价值的定量衡量标准来评估语言模型在文本上的预测表现，但请谨慎解释其得分。主要在受控条件下将其用于相对比较，并将其视为评估拼图的一部分，辅以针对下游任务的评估和定性分析，以获得模型能力的全面情况。

获取即时帮助、个性化解释和交互式代码示例。

---

### 每字符/词比特数

# 每字符/词比特数

困惑度是语言模型最常用的一种内在评估指标，如前所述。与信息论紧密关联的指标则提供不同视角，尤其在比较采用不同分词 (tokenization)方案的模型时。这些指标计算根据模型的概率分布，编码每个文本单位（字符或词/词元 (token)）平均所需的比特数。

训练期间使用的交叉熵损失本质上是真实下一个词元的平均负对数似然，通常使用自然对数（以eee为底）计算：
H(p,q)=−1N∑i=1Nlog⁡eq(wi∣w<i;θ)H(p, q) = -\frac{1}{N}\sum\_{i=1}^N \log\_e q(w\_i | w\_{<i}; \theta)H(p,q)=−N1​∑i=1N​loge​q(wi​∣w<i​;θ)
qqq 是模型的分布，NNN 是词元数量。困惑度是此损失的指数化：PPL=exp⁡(H(p,q))PPL = \exp(H(p, q))PPL=exp(H(p,q))。

然而，信息论通常以*比特*（以2为底的对数）来衡量信息内容。一个为观测序列分配更高概率的模型，其“确定性”更强，根据其概率分配，平均编码该序列所需的比特数更少。这使得我们关注每字符比特数和每词/词元比特数等指标。

### 每字符比特数 (BPC)

每字符比特数 (BPC) 衡量给定模型预测下，编码文本序列中每个*字符*所需的平均比特数。它的计算方法是，取使用以2为底的对数计算的交叉熵损失，然后除以序列中的总字符数 (CCC)，而不是词元 (token)数量：

BPC=−1C∑j=1Clog⁡2p(charj∣context;θ)BPC = -\frac{1}{C}\sum\_{j=1}^C \log\_2 p(char\_j | context; \theta)BPC=−C1​∑j=1C​log2​p(charj​∣context;θ)

然而，大多数大型语言模型是基于词元（词或子词 (subword)）工作的，而非单个字符。计算每个字符的精确概率需要一个字符级别模型，或对词元概率进行复杂的边缘化处理。对于基于词元的模型，一个更实用的方法是计算每个词元的标准交叉熵损失（以eee为底），然后将其转换为比特（以2为底），并除以原始文本中的字符数量。

每词元计算的交叉熵损失（以eee为底，表示为HeH\_eHe​）与 BPC 的关系如下：

BPC=He×NC×ln⁡(2)BPC = \frac{H\_e \times N}{C \times \ln(2)}BPC=C×ln(2)He​×N​

此处，NNN 是词元数量，CCC 是评估文本中的字符数量。ln⁡(2)\ln(2)ln(2) 因子将自然对数转换为以2为底的对数（log⁡2(x)=ln⁡(x)ln⁡(2)\log\_2(x) = \frac{\ln(x)}{\ln(2)}log2​(x)=ln(2)ln(x)​）。

BPC 的主要优点是它对所选分词 (tokenization)方法相对不敏感。由于它通过字符数量进行归一化 (normalization)，因此可以在相同的原始文本数据上，更公正地比较使用不同分词器 (tokenizer)（例如，BPE、WordPiece、字符级别）的模型。一个模型可能仅因为其分词器将词分解成许多小片段而获得不错的困惑度分数，但其BPC可以表明它是否在建模底层字符序列方面确实表现更佳。

我们来看一个 PyTorch 示例。假设您已从评估循环中获得每词元的平均交叉熵损失以及字符/词元计数：

```python
import torch
import math

avg_cross_entropy_loss_per_token = 1.85
total_tokens_in_eval_set = 50000
total_chars_in_eval_set = 200000

avg_bits_per_token = avg_cross_entropy_loss_per_token / math.log(2)

total_bits = avg_bits_per_token * total_tokens_in_eval_set

bpc = total_bits / total_chars_in_eval_set

print(
    f"每词元平均交叉熵损失（以e为底）: "
    f"{avg_cross_entropy_loss_per_token:.4f}"
)
print(f"每词元平均比特数 (BPT): {avg_bits_per_token:.4f}")
print(f"每字符比特数 (BPC): {bpc:.4f}")
```

### 每词/词元 (token)比特数 (BPW/BPT)

每词比特数 (BPW) 或每词元比特数 (BPT) 对于基于词元的模型来说更简单。它表示根据模型的概率分配，编码每个*词元*（如果使用词级别分词 (tokenization)，则为词）所需的平均比特数。

它与在词元上计算的交叉熵损失（以eee为底，HeH\_eHe​）和困惑度（PPL）直接关联：

BPT=−1N∑i=1Nlog⁡2p(词元i∣词元<i;θ)=Heln⁡(2)BPT = -\frac{1}{N}\sum\_{i=1}^N \log\_2 p(词元\_i | 词元\_{<i}; \theta) = \frac{H\_e}{\ln(2)}BPT=−N1​∑i=1N​log2​p(词元i​∣词元<i​;θ)=ln(2)He​​

此外，BPW/BPT 与困惑度有直接关联：

PPL=2BPTPPL = 2^{BPT}PPL=2BPT
BPT=log⁡2(PPL)BPT = \log\_2(PPL)BPT=log2​(PPL)

这意味着 BPW/BPT 只是困惑度的以2为底的对数。如果一个模型在数据集上的困惑度为 16，这意味着平均而言，预测下一个词元的难度，相当于在 16=2416 = 2^416=24 个选项中均匀选择。这对应于每词元 4 比特的 BPT。

```python
import torch
import math

perplexity = 16.0
avg_cross_entropy_loss_per_token = math.log(perplexity)

bpt_from_ppl = math.log2(perplexity)

bpt_from_loss = avg_cross_entropy_loss_per_token / math.log(2)

print(f"困惑度 (PPL): {perplexity:.4f}")
print(
    f"每词元平均交叉熵损失（以e为底）: "
    f"{avg_cross_entropy_loss_per_token:.4f}"
)
print(f"从 PPL 计算的每词元比特数 (BPT): {bpt_from_ppl:.4f}")
print(f"从损失计算的每词元比特数 (BPT): {bpt_from_loss:.4f}")
```

尽管 BPW/BPT 是困惑度的直接转换，但以“比特”来思考，能与信息论及压缩限制建立更强的关联。更低的 BPT 表明模型根据其学习到的概率分布，在压缩文本序列中的信息方面更高效。然而，与 BPC 不同，BPW/BPT 高度依赖于所使用的分词方法。使用子词 (subword)单元的模型在相同文本上通常会比字符级别模型具有更低的 BPT，仅仅因为预测一个更长的词元单元一次能覆盖更多字符。

总之，BPC 和 BPW/BPT 都提供了关于模型内在性能的信息论视角。BPC 提供了一种与分词器 (tokenizer)无关的衡量方法，在比较根本不同的模型时很有用；而 BPW/BPT 是困惑度的直接对数转换，常用于评估基于特定分词方案的词元模型。这两种指标都量化 (quantization)了模型的概率分布与数据的匹配程度，数值越低表示匹配度越好。

获取即时帮助、个性化解释和交互式代码示例。

---

### 分词对困惑度的影响

# 分词对困惑度的影响

正如我们所知，困惑度是评估语言模型的基础固有指标，量化 (quantization)模型预测给定文本序列的优劣。它直接源于模型对该序列中词元 (token)分配的概率。然而，一个常被忽视的重要事项是，计算出的困惑度对所用的特定**分词 (tokenization)**方案高度敏感。回顾困惑度公式：

困惑度是评估语言模型的一种基本内在指标，它量化了模型预测给定文本序列的能力。它直接来源于模型赋予该序列中令牌的概率。一个经常被忽视的事实是，计算出的困惑度对所使用的特定**分词**方案高度敏感。困惑度公式如下：

这里，wiw\_iwi​ 表示序列中的第 iii 个*词元*，NNN 是*词元*总数。这表示单个概率 p(wi∣w<i;θ)p(w\_i | w\_{<i}; \theta)p(wi​∣w<i​;θ) 和序列长度 NNN 都直接取决于原始文本如何被分割成词元。

考虑一个简单句子：“分词影响困惑度。”

来看看不同分词器 (tokenizer)如何处理此句：

1. **词级别分词器：** 可能会生成 `["Tokenization", "impacts", "perplexity", "."]` -> N=4N = 4N=4 个词元。模型会预测在给定“Tokenization”的情况下“impacts”的概率，然后在给定前两个词的情况下“perplexity”的概率，依此类推。
2. **字符级别分词器：** 将生成 `['T', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n', ' ', 'i', 'm', 'p', 'a', 'c', 't', 's', ' ', 'p', 'e', 'r', 'p', 'l', 'e', 'x', 'i', 't', 'y', '.']` -> N=33N = 33N=33 个词元。模型会预测在给定“T”的情况下“o”，在给定“To”的情况下“k”等。预测任务非常不同。
3. **子词 (subword)分词器（例如，BPE/WordPiece）：** 可能会生成 `["Token", "ization", "Ġimpacts", "Ġperplex", "ity", "."]` -> N=6N = 6N=6 个词元（假设是 GPT-2 风格的分词器，其中 `Ġ` 表示空格）。在这里，模型会预测在给定“Token”的情况下“ization”，在给定“Tokenization”的情况下“Ġimpacts”，依此类推。

此示例显示了两个主要影响：

- **序列长度 (N)：** 不同的分词器对于相同的底层文本会产生不同数量的词元 (NNN)。由于困惑度涉及对 NNN 求负对数概率的平均值，因此改变 NNN 会直接改变最终得分，即使分配给整个序列的底层总概率保持不变（通常不会）。通常，对于相同的文本，字符级别分词会导致比子词或词分词更长的序列（更高的 NNN）。
- **预测任务难度：** 预测任务的性质会发生变化。预测下一个字符通常比预测下一个完整词或复杂子词更容易（条件熵更低）。然而，字符模型需要进行更多的预测。词级别模型进行的预测较少，但每个预测可能更难（从更大的词汇表 (vocabulary)中选择，预测更长的单元）。子词模型介于两者之间。

让我们用一个简短的 PyTorch 示例，使用 `transformers` 库来说明词元差异：

```python
import torch
from transformers import AutoTokenizer

tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer_gpt2 = AutoTokenizer.from_pretrained('gpt2')

text = "Tokenization impacts perplexity."

tokens_bert = tokenizer_bert.tokenize(text)
ids_bert = tokenizer_bert.encode(text)
print(f"BERT Tokens ({len(tokens_bert)}): {tokens_bert}")

print(f"BERT IDs ({len(ids_bert)}): {ids_bert}")

tokens_gpt2 = tokenizer_gpt2.tokenize(text)
ids_gpt2 = tokenizer_gpt2.encode(text)
print(f"GPT-2 Tokens ({len(tokens_gpt2)}): {tokens_gpt2}")

print(f"GPT-2 IDs ({len(ids_gpt2)}): {ids_gpt2}")
```

请注意，即使在两个子词分词器（BERT 的 WordPiece 和 GPT-2 的 BPE）之间，分段也不同（`'token', '##ization'` 与 `'Token', 'ization'`），并且 BERT 的 `encode` 方法默认包含 `[CLS]` 和 `[SEP]` 等特殊词元，这会影响标准困惑度计算中使用的序列长度 (NNN)。

直接结果是，**只有当模型在评估数据集上使用完全相同的分词器和词汇表时，它们的困惑度得分才可以直接比较。** 比较一个使用 50,000 次合并的 BPE 模型与一个使用 30,000 词汇表的 WordPiece 模型的困惑度，就像比较苹果和橘子。底层的预测单元不同。

字符BPE-32kBPE-50k词020406080

> 上图说明了对于*相同的底层文本*，困惑度得分仅因所选分词器不同就可能发生显著变化。较低的值表明模型认为*每个词元*的预测任务更容易，但这并不能直接比较不同分词方案下模型质量。

此外，在分词*之前*应用的预处理步骤，例如转换为小写或 Unicode 规范化，也会对此产生影响。如果一项评估将文本转换为小写而另一项没有，则区分大小写的词元分析器会生成不同的词元，导致困惑度得分无法比较。

在报告或解释困惑度时，请务必清楚了解使用了哪种分词器，包括其词汇表大小和任何相关的预处理步骤。没有这些背景信息，一个原始的困惑度数值对于比较在不同分词机制下评估的其他模型的相对能力提供的信息有限。最可靠的比较是在使用相同评估设置（包括分词器）评估不同模型或检查点时进行的。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 22 Extrinsic Evaluation Downstream Tasks

### 下游任务评估的理由

# 下游任务评估的理由

困惑度等内在评估指标能为语言模型在保留文本分布上的流畅性和预测能力提供有价值的理解，但它们通常无法充分说明模型对于特定应用有多么适用。一个模型可能获得很低的困惑度分数，表明它在根据训练数据中的模式预测序列中的下一个token方面表现出色，但当被要求执行准确总结文档或正确回答事实性问题等具体任务时，却可能表现糟糕。统计文本生成能力与实际效用之间的这种差距，使得外部评估成为必需。

外部评估衡量模型在完成特定下游任务时的表现。我们不衡量模型单独预测文本的准确度，而是衡量它在执行以下任务时的表现：

- **文本分类：** 为文本分配类别（例如，情感分析、主题标注）。
- **问答：** 根据给定背景提供问题答案。
- **摘要：** 生成较长文本 (long context)的简洁摘要。
- **机器翻译：** 将文本从一种语言翻译到另一种语言。
- **命名实体识别：** 识别文本中的实体，如名称、日期和地点。

进行下游任务评估的主要原因是为了获得一个**衡量实际关联度的指标**。困惑度是一个抽象的衡量指标；而情感分析任务上的准确率、命名实体识别上的F1分数，或摘要任务的ROUGE分数，则直接量化 (quantization)了模型达到预期结果的能力。这些指标通常更易于理解，并与部署模型的目标直接相关。

考虑两个模型，模型A和模型B，它们都经过大型文本语料库的预训练 (pre-training)。

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
```

在这个场景下，模型A和模型B的困惑度分数非常接近，这表明它们在一般意义上具有相似的语言建模能力。然而，当在特定情感分析任务上进行微调 (fine-tuning)和评估时，模型A的表现显著优于模型B。这种下游表现的差异可能源于预训练期间所获取知识的微小差异、模型在微调期间的适应能力，或其对情感表达相关语言的理解。仅仅依靠困惑度会掩盖这种实际能力上的重要区别。

Model AModel B05101500.20.40.60.8困惑度（越低越好）情感准确率（越高越好）内在与外部指标比较

> 比较显示两个模型困惑度相似但在下游任务准确率上有所不同。

此外，外部评估有助于**发现具体的优点和不足**。一个模型可能擅长需要广泛知识的任务（例如开放域问答），但在需要逻辑推理 (inference)或创意生成的任务上可能表现不佳。对多种下游任务进行评估，能描绘出模型能力更全面的图景，这比任何单一的内在指标所能提供的更为全面。

最后，在GLUE（通用语言理解评估）或SuperGLUE等标准化下游基准上进行评估，使得不同模型和研究成果之间能够进行**客观比较**。这些基准提供精心组织的数据集和既定指标，适用于一系列任务，它们作为衡量该方面进展的共同标准。

总之，尽管内在指标对于监测训练过程和评估一般语言建模能力很有用，但在下游任务上的外部评估对于理解和量化模型的实际效用，有意义地比较不同模型，以及指导发展以构建更具能力和更有用的LLM是必不可少的。本章的后续部分将详细说明用于此目的的具体任务、基准和方法。

获取即时帮助、个性化解释和交互式代码示例。

---

### 常见下游自然语言处理任务

# 常见下游自然语言处理任务

内在衡量标准，例如困惑度，能提供关于语言模型核心能力的信息，但往往无法准确预测模型在特定、实际任务上的表现。为了更全面地了解大型语言模型（LLM）的实用性，外在评估衡量其在一系列成熟的下游自然语言处理（NLP）任务上的表现。这种方法将评估与实际应用相结合，有助于了解模型在不同场景下的优点和不足。

对下游任务进行评估通常涉及调整预训练 (pre-training)的LLM，这常通过微调 (fine-tuning)或提示技术完成，然后使用任务特定的衡量标准来评估其表现。接下来，我们来看一些为此目的最常用的任务。

### 文本分类

文本分类是一项核心NLP任务，其目标是为给定的文本输入分配预设的类别或标签。示例包括：

- **情感分析：** 判断评论、推文或句子的情感基调（例如，积极、消极、中立）。
- **主题分类：** 将文档归类到一个或多个主题（例如，体育、科技、政治）。
- **垃圾邮件检测：** 识别电子邮件或消息是否为垃圾邮件。

**相关性：** 此任务测试模型理解文本段落整体含义、语气和主旨的能力。

**评估：** 性能通常使用准确率、精确率、召回率和F1分数来衡量，尤其是在类别不平衡的情况下（如垃圾邮件检测）。

**LLM应用：** 对于微调 (fine-tuning)，一种常见做法是在特殊标记 (token)（如BERT风格模型中的`[CLS]`）的最终隐藏状态或序列的池化输出之上添加一个线性分类层。然后，模型会在针对分类任务的标注数据集上进行微调。

```python
import torch.nn as nn
from transformers import AutoModel

class SimpleClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)

        self.classifier_head = nn.Linear(
            self.transformer.config.hidden_size, num_labels
        )
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask):

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.last_hidden_state[:, 0]

        logits = self.classifier_head(pooled_output)
        return logits
```

另一种方法是，在少样本或零样本设定下，可以通过指令和示例来提示LLM执行分类，而无需显式微调。

### 问答 (QA)

问答系统旨在为以自然语言提出的问题提供答案。常见变体有：

- **抽取式问答：** 答案是从提供的上下文 (context)文档中直接抽取的一段文本。像SQuAD（斯坦福问答数据集）这样的数据集是此任务的基准。
- **生成式问答：** 模型以自己的话语生成答案，可能综合了上下文信息或其内部知识。
- **开放域问答：** 模型必须从大量文档语料库或其内部知识库中找到答案，而无需在问题旁提供特定上下文。

**相关性：** 问答任务严格测试阅读理解、信息检索，有时还测试综合并生成连贯文本的能力。

**评估：** 抽取式问答通常使用精确匹配（EM）和F1分数来评估预测的答案段与真实答案的比较。生成式和开放域问答常使用像ROUGE（面向召回的摘要评估替补）或BLEU（双语评估替补）这样的指标，尽管人工评估也同样重要。

**LLM应用：** 对于抽取式问答，模型常被微调 (fine-tuning)以预测上下文内答案段落的起始和结束标记 (token)索引。对于生成式问答，序列到序列模型或仅解码器模型会被微调或提示以直接生成答案。

QA

Context

上下文：
阿波罗11号任务于1969年7月
将首批人类送上月球。

Model

LLM (问答专用)

Context->Model

Question

问题：
人类首次登月是何时？

Question->Model

Answer

答案：
1969年7月

Model->Answer

 (抽取式)

> 抽取式问答任务的输入组成部分（上下文、问题）和结果答案。

### 摘要

文本摘要的目标是在保留源文档最主要信息的同时，生成一个更短的版本。

- **抽取式摘要：** 直接从原文中选择显著的句子或短语。
- **生成式摘要：** 生成新的摘要，对源文档进行转述和浓缩，可能使用新颖的词语或措辞。现代LLM在此方面表现出色。

**相关性：** 摘要测试模型识别重要信息、理解上下文 (context)以及生成流畅、简洁文本的能力。

**评估：** ROUGE分数（特别是ROUGE-1、ROUGE-2、ROUGE-L）是标准衡量指标，通过N-gram重叠将生成的摘要与人工编写的参考摘要进行比较。对连贯性、流畅性和信息覆盖度进行人工评估也普遍采用。

**LLM应用：** 序列到序列架构曾是传统做法，但大型仅解码器模型现在非常有效。它们通常在文档-摘要对上进行微调 (fine-tuning)，或通过“总结以下文本：”等指令进行提示。

### 机器翻译

机器翻译涉及将文本从源语言翻译成目标语言。

**相关性：** 测试模型对多种语言的语法、句法、语义和文化背景的理解，以及其生成能力。

**评估：** BLEU分数是广泛使用的衡量指标，衡量生成的译文与参考译文相比的N-gram准确性。METEOR和chrF等其他指标也常被使用。人工评估对于评估翻译质量和流畅性仍然很重要。

**LLM应用：** 序列到序列模型曾是标准做法。大型多语言LLM通常可以在零样本或少样本设定下执行翻译，通过在提示中包含示例或指令（例如，“将以下英文文本翻译成法文：...”）。在并行语料库上进行微调 (fine-tuning)仍是实现特定语言对高性能的常见做法。

### 自然语言推理 (inference) (NLI) / 文本蕴含

NLI任务要求模型判断一对句子：一个“前提”和一个“假设”之间的逻辑关系。关系通常是以下之一：

- **蕴含：** 假设逻辑上来源于前提。
- **矛盾：** 假设与前提相矛盾。
- **中立：** 假设既不蕴含于前提，也不与前提相矛盾。

**相关性：** NLI被认为是衡量一般语言理解和推理能力的良好替代，因为它要求掌握句子的含义和推论。

**评估：** 准确率是主要衡量指标，衡量正确分类的关系的百分比。

**LLM应用：** 类似于文本分类，模型通常通过连接前提和假设（带分隔符标记 (token)），将其输入LLM，并在池化输出上添加分类头部来预测三个标签之一进行微调 (fine-tuning)。

NLI

Premise

前提：
一只猫睡在垫子上。

Model

LLM (NLI专用)

Premise->Model

Hypothesis

假设：
一只动物正在休息。

Hypothesis->Model

Label

标签：
蕴含

Model->Label

> NLI任务的输入组成部分（前提、假设）和结果标签。

这些仅仅是用于LLM外在评估的一些常见下游任务。其他包括命名实体识别（NER）、共指消解、跨不同场景的情感分析，以及更多在GLUE和SuperGLUE等基准测试中发现的专业任务，我们将在后面讨论。通过评估这些多样化的任务，可以对LLM的整体能力及其对多种用途的适应性进行评估。

获取即时帮助、个性化解释和交互式代码示例。

---

### 评估时的微调步骤

# 评估时的微调步骤

为了有效地评估预训练 (pre-training)大语言模型（LLM）在下游任务上的表现，我们通常需要根据目标任务的特定格式和目标对其进行一些调整。这个过程称为微调 (fine-tuning)。虽然零样本或少样本评估（稍后讨论）测试模型固有的泛化能力，但微调使模型能够从标注数据中学习任务特定模式，通常能获得更高表现，并更清晰地呈现预训练模型在该任务上的潜在作用。

微调使用预训练期间学到的强大表征，并使用相对少量任务特定的标注数据进行调整。核心观点是预训练模型已经理解语言结构、语义和语境；微调只是教它如何将这些知识应用于新的问题形式。

### 通用微调 (fine-tuning)流程

对用于评估下游任务的LLM进行微调的标准流程包含以下步骤：

1. **加载预训练 (pre-training)模型：** 从预训练LLM的权重 (weight)开始。这可以是您自己训练的模型，也可以是公开可用的检查点（例如，来自Hugging Face Hub）。
2. **调整模型架构：** 通过添加任务特定的“头部”来修改模型。该头部通常是一个或多个小型神经网络 (neural network)层，放置在预训练模型的核心结构（例如Transformer块）之上。此头部的类型完全取决于下游任务。
3. **准备任务数据集：** 获取下游任务的标注数据集。这涉及根据模型和任务头部的要求格式化输入数据，并使用模型预训练期间使用的*相同分词 (tokenization)器 (tokenizer)*对文本进行分词。
4. **定义目标：** 选择适合任务的损失函数 (loss function)（例如，分类的交叉熵损失，回归的均方误差）。
5. **训练：** 运行训练循环，更新整个模型或仅更新新添加的头部以及预训练模型的顶层权重。使用适合的优化器（如AdamW）和学习率调度。训练通常仅在任务数据集上运行几个周期。
6. **评估：** 使用与下游任务相关的指标（例如，准确率、F1分数、ROUGE、BLEU）在保留测试集上衡量表现。

流程图如下：

G

clusterₚretrained

预训练LLM

cluster\_finetuning

微调设置

cluster\_data

任务数据

PLM

核心Transformer层

Head

任务特定头部

PLM->Head

 输出表征

Loss

任务损失

Head->Loss

 预测

Loss->PLM

 梯度更新

Loss->Head

 梯度更新

Data

标注数据集

Data->Loss

 标签

> 预训练LLM的核心层为新添加的任务特定头部提供输入表征。该头部进行预测，然后使用任务特定的损失函数与任务数据集中的标签进行比较。此损失产生的梯度会更新头部的权重，通常也会更新部分或全部预训练层的权重。

### 任务特定头部和数据格式化

调整步骤主要涉及添加正确的头部和适当的数据格式化。我们来看看常见例子：

#### 文本分类

- **目标：** 为一段文本分配类别标签（例如，情感分析、主题分类）。
- **头部：** 通常是一个线性层，它获取指定token（如BERT风格模型中的`[CLS]` token，或因果模型中的最后一个token）的最终隐藏状态，并将其投射到输出类别的数量。在线性层之前通常会添加一个dropout层用于正则化 (regularization)。
- **输入格式：** 对于BERT等模型，输入通常格式化为`[CLS] text_sequence [SEP]`。对于因果模型（如GPT），输入可能只是`text_sequence`，并使用最后一个token的表征。
- **损失：** 交叉熵损失是多类别分类的标准选择。

```python
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer
)

model_name = "bert-base-uncased"
pretrained_model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

num_labels = 3
hidden_size = pretrained_model.config.hidden_size
classification_head = nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(hidden_size, num_labels)
)

text = "This is an example sentence."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

outputs = pretrained_model(**inputs)

cls_representation = outputs.last_hidden_state[:, 0, :]

logits = classification_head(cls_representation)
```

#### 抽取式问答 (QA)

- **目标：** 在给定的上下文 (context)段落中找到回答特定问题的文本片段（例如，SQuAD基准）。
- **头部：** 通常包含两个线性层。一个预测上下文中每个token作为答案片段*起始*的概率，另一个预测每个token作为答案片段*结束*的概率。
- **输入格式：** 通常将问题和上下文结合起来，并用特殊token分隔：`[CLS] question_text [SEP] context_text [SEP]`。
- **损失：** 交叉熵损失分别计算起始和结束位置。模型被训练来预测上下文中正确的起始和结束token索引。在推理 (inference)时，需要进行后处理以找到最可能且有效的片段（结束索引需大于或等于起始索引）。

```python

qa_head = nn.Linear(hidden_size, 2)

question = "What is the capital of Malaysia?"
context = "Malaysia is a Southeast Asian country. Kuala Lumpur is its capital and largest city."
inputs = tokenizer(
    question,
    context,
    return_tensors="pt",
    padding=True,
    truncation=True
)

outputs = pretrained_model(**inputs)
sequence_output = outputs.last_hidden_state

logits = qa_head(sequence_output)
start_logits, end_logits = logits.split(1, dim=-1)
start_logits = start_logits.squeeze(-1)
end_logits = end_logits.squeeze(-1)
```

#### 序列到序列任务

- **目标：** 根据输入序列生成文本序列（例如，摘要、翻译、对话生成）。
- **头部：** 对于编码器-解码器模型（如T5、BART），预训练 (pre-training)的解码器已作为生成头部。对于仅解码器模型（如GPT），标准语言建模头部（预测下一个token）直接用于生成。微调 (fine-tuning)调整模型，使其根据特定输入格式生成输出（例如，在输入文本前添加“summarize: ”这样的前缀）。
- **输入格式：** 根据任务而异。对于摘要：`input_article`。对于翻译：`translate English to French: input_english_sentence`。目标序列在训练期间用作标签。
- **损失：** 交叉熵损失，计算生成序列token与目标序列的比较（训练期间通常使用教师强制，即模型会看到真实的前一个token）。

### 微调 (fine-tuning)的训练考量

微调涉及训练，但与预训练 (pre-training)相比有一些差异：

- **学习率：** 通常使用显著更小的学习率（例如，1e−51e-51e−5到5e−55e-55e−5）。预训练模型的权重 (weight)已经良好初始化，因此大幅更新可能破坏已学到的表征。
- **优化器：** AdamW因其有效性和对权重衰减的处理能力，仍然是常见选择。
- **批量大小：** 常受限于GPU内存，特别是对于较大模型。有效批量大小有时可通过梯度累积来增加。
- **周期数：** 微调通常只需要少量周期（通常1-5个）。在较小的特定数据集上训练过长时间可能导致过拟合 (overfitting)并降低模型的通用能力。
- **学习率调度：** 带有短暂热身期（例如，在训练步骤的前6-10%进行热身）的线性衰减调度通常有效。
- **权重衰减：** 作为正则化 (regularization)技术应用，类似于预训练。
- **冻结层：** 有时，最初只训练任务特定头部和预训练模型的顶层几层，保持底层冻结。这可以节省计算并防止灾难性遗忘，尽管如果数据充足，全面微调（更新所有参数 (parameter)）通常会带来最佳表现。

### 简化微调 (fine-tuning)循环示例（PyTorch）

```python
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

learning_rate = 3e-5
num_epochs = 3
batch_size = 16
warmup_steps = 100
total_training_steps = len(train_dataset) * num_epochs // batch_size

optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=warmup_steps,
                                           num_training_steps=total_training_steps)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)
        logits = outputs.logits

        loss = loss_fn(logits, labels)

        loss.backward()

        optimizer.step()
        scheduler.step()

    print(f"周期 {epoch + 1} 完成。上一个批次的损失：{loss.item()}")
```

### 评估指标

重要的是，微调 (fine-tuning)后，模型使用下游任务特有的指标进行评估。对于分类，这可能是准确率或F1分数。对于问答，精确匹配（EM）和预测答案token的F1分数是常见的。对于摘要，ROUGE分数（ROUGE-1、ROUGE-2、ROUGE-L）是标准指标，衡量与参考摘要的重叠度。对于翻译，BLEU分数常被使用。这些外部指标直接衡量模型在其所适应任务上的表现，补充了来自困惑度等内部指标的信息。

微调是评估和调整LLM的有效方法。尽管它需要标注数据和计算资源（虽然远少于预训练 (pre-training)），但它使我们能够评估预训练模型所学知识如何有效地迁移以解决特定实际问题。通过微调获得的结果通常代表着该特定任务中潜在的预训练模型具有的强劲表现基准。

获取即时帮助、个性化解释和交互式代码示例。

---

### 标准基准：GLUE 和 SuperGLUE

# 标准基准：GLUE 和 SuperGLUE

评估大型语言模型（LLM）在实际任务中的表现，外部评估至关重要。在预训练 (pre-training)期间常用的一些指标，例如困惑度，虽然能提供有价值的指示，但无法保证LLM能有效地分类情感、回答问题或判断两句话是否为释义。为了衡量这些应用能力，研究界开发了包含多种自然语言处理（NLP）任务的基准测试集。影响力较大的包括GLUE及其后续版本SuperGLUE。

这些基准测试作为比较不同模型的共同标准。研究人员无需在分散、可能特殊化的任务上进行评估，而可以使用这些已有测试集来衡量在一般语言理解方面的进展。GLUE和SuperGLUE都汇总了多个数据集上的表现，旨在提供一个单一、全面的分数，以反映模型的通用性。

### GLUE：通用语言理解评估

GLUE基准测试于2018年推出，是迈向标准化NLP评估的重要一步。它捆绑了九项不同的英语语言理解任务，旨在涵盖多种语言现象。其目的是鼓励开发能学习通用表示、可跨不同任务迁移的模型，而不是只针对单一能力进行特殊化训练的模型。

GLUE中的任务大致可以分为以下几类：

1. **单句任务：** 评估对单个句子内部属性的理解。

   - **CoLA（语言可接受性语料库）：** 判断一个句子在语法上是否可接受。（二元分类）
   - **SST-2（斯坦福情感树库）：** 对电影评论的情感进行分类。（二元分类）
2. **相似性和释义任务：** 评估判断句子对之间语义关系的能力。

   - **MRPC（微软研究释义语料库）：** 判断两个句子是否互为释义。（二元分类）
   - **STS-B（语义文本相似度基准）：** 预测两个句子之间的相似度分数（回归，分数1-5）。
   - **QQP（Quora问题对）：** 判断Quora上发布的两个问题是否语义等效。（二元分类）
3. **自然语言推理 (inference)（NLI）任务：** 评估模型推理前提和假设句之间关系的能力。

   - **MNLI（多类型自然语言推理）：** 给定一个前提，判断一个假设是蕴含、矛盾还是中立。在匹配（同领域）和不匹配（跨领域）测试集上进行评估。（三元分类）
   - **QNLI（问题自然语言推理）：** 判断一个句子是否包含问题的答案。源自SQuAD。（二元分类）
   - **RTE（识别文本蕴含）：** 一个较小的数据集，结合了来自多个来源的文本蕴含数据。（二元分类）
   - **WNLI（Winograd NLI）：** 一个小型但非常难的数据集，专注于代词消解。（二元分类）

GLUE\_Tasks

clusterₛingle

单句

clusterₚair

句子对

clusterₙli

推理

GLUE

GLUE 基准测试

CoLA

CoLA
(语法)

GLUE->CoLA

SST2

SST-2
(情感)

GLUE->SST2

MRPC

MRPC
(释义)

GLUE->MRPC

STSB

STS-B
(相似度)

GLUE->STSB

QQP

QQP
(重复问题)

GLUE->QQP

MNLI

MNLI
(蕴含)

GLUE->MNLI

QNLI

QNLI
(问答蕴含)

GLUE->QNLI

RTE

RTE
(蕴含)

GLUE->RTE

WNLI

WNLI
(代词消解)

GLUE->WNLI

> GLUE基准测试中任务的大致分类。

模型通常会在每个任务的训练数据上分别进行微调 (fine-tuning)。性能使用任务专用指标衡量（例如，分类的准确率、F1分数、CoLA的马修斯相关系数、STS-B的皮尔逊/斯皮尔曼相关系数）。最终的GLUE分数是各个任务分数的未加权平均值。尽管影响力很大，但GLUE分数在许多任务上很快达到了接近人类的表现，这表明该基准测试可能不足以有效区分最先进的模型。

### SuperGLUE：提高标准

为解决GLUE出现的饱和问题并进一步拓展语言理解的边界，SuperGLUE于2019年推出。它遵循相似的结构，但包含更困难的任务，这些任务需要更复杂的推理 (inference)、常识知识和处理歧义的能力。它还包括更多样的任务格式，并更注重有挑战性的示例。

SuperGLUE包含一套新任务，尽管有些与GLUE任务重叠或相关：

- **BoolQ（布尔问题）：** 基于给定段落的是/否问答。需要推理和阅读理解能力。
- **CB（承诺库）：** 专注于识别文本中表达信念和承诺的蕴含关系的NLI任务。（三元分类：蕴含、矛盾、中立）
- **COPA（合理替代选项）：** 因果推理任务，模型必须为给定前提选择更合理的原因或结果。（二元分类）
- **MultiRC（多句阅读理解）：** 问答任务，答案可能来自段落中的多句话，且问题可以有多个正确答案。（每个答案候选的二元分类）
- **ReCoRD（常识推理阅读理解数据集）：** 完形填空式阅读理解，需要常识推理来填充新闻文章中的遮蔽实体。（抽取式问答）
- **RTE（识别文本蕴含）：** 使用来自不同来源的数据，类似于GLUE的RTE但数据经过聚合。（二元分类）
- **WiC（上下文 (context)中的词语）：** 判断一个目标词在两个不同句子中是否具有相同含义。多义词任务。（二元分类）
- **WSC（Winograd图式挑战）：** 指代消解任务，需要常识来解决精心构建句子（Winograd图式）中的歧义代词。（二元分类）

SuperGLUE也使用了适合每个任务的多种指标，最终分数是平均值。它为模型设定了更高的标准，与GLUE相比，在SuperGLUE上的进展通常被认为是模型具有更强高级语言理解能力的指标。

SuperGLUE\_Concept

GLUE

GLUE
(已饱和)

Progress
难度增加
推理更复杂

GLUE->Progress

SuperGLUE

SuperGLUE
(更具挑战性)

Progress->SuperGLUE

> SuperGLUE被设计为GLUE的更具挑战性的后续版本。

### 使用GLUE和SuperGLUE进行评估

在这些基准上评估预训练 (pre-training)的LLM通常涉及每个任务的标准化微调 (fine-tuning)过程：

1. **选择预训练模型：** 从您的基础LLM（例如BERT、RoBERTa或自定义Transformer）开始。
2. **为任务调整：** 对于大多数GLUE/SuperGLUE任务（通常是分类或回归），在基础LLM之上添加一个小的任务专用“头部”层。这个头部获取LLM的最终隐藏状态（通常是对应特殊`[CLS]`标记 (token)的表示），并将其映射到任务的输出空间（例如，类别的logits，单一回归值）。
3. **微调：** 在特定任务的训练数据集上训练这个组合模型（LLM + 头部）。通常，LLM的预训练权重 (weight)在此过程中会更新，尽管通常学习率低于随机初始化的头部。此步骤对基准测试集中的每个任务都 *独立* 进行。
4. **评估：** 使用官方指标衡量微调模型在任务的开发集或测试集上的表现。
5. **汇总：** 计算所有任务的平均分数，以获得最终的GLUE或SuperGLUE分数。

Hugging Face的`transformers`等库极大地简化了这个过程。它们提供了对基准数据集的便捷访问以及用于微调各种预训练模型的标准化脚本。

以下是一个PyTorch代码片段，展示了如何向预训练模型添加一个简单的分类头部，以在类似SST-2（情感分类）的任务上进行微调：

```python
import torch
import torch.nn as nn
from transformers import AutoModel

base_model_name = "bert-base-uncased"
base_model = AutoModel.from_pretrained(base_model_name)

class SentimentClassifierHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, sequence_output):

        pooled_output = sequence_output[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class FullSentimentClassifier(nn.Module):
    def __init__(self, base_model, head):
        super().__init__()
        self.base_model = base_model
        self.head = head

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden_state = outputs.last_hidden_state

        logits = self.head(last_hidden_state)
        return logits

config = base_model.config
classification_head = SentimentClassifierHead(
    config.hidden_size,
    num_labels=2
)
model = FullSentimentClassifier(base_model, classification_head)
```

> PyTorch代码展示了如何向预训练的Transformer模型添加分类头部，以在类似GLUE的任务上进行微调。

### 理解基准测试分数

GLUE和SuperGLUE分数提供了一种有价值的标准化衡量标准，用于评估模型的通用语言能力。更高的分数通常表明模型学习了更具迁移性的表示。然而，解释这些分数需要谨慎：

- **相关性与因果关系：** 在GLUE/SuperGLUE上表现优异并不自动保证在基准测试中未包含的特定、新颖的下游任务上表现出色。基准任务可能无法完全捕捉您目标应用的细微特点。
- **基准伪影：** 模型可能学会利用基准数据集中存在的虚假相关性或伪影，而非实现真正的理解。这对测试集中较小的数据集尤其重要。
- **饱和：** 如前所述，GLUE很快达到饱和，这意味着顶尖模型取得了接近估计人类表现的分数，使得难以区分它们。SuperGLUE更具抵抗力，但随着时间推移可能面临类似问题。
- **数据污染：** 存在一种风险，即基准数据集的部分可能无意中被包含在用于现代LLM的大规模预训练 (pre-training)语料库中，从而可能虚高分数。

尽管有这些局限性，GLUE和SuperGLUE仍然是重要的工具。它们提供了一个共同的衡量标准，用于比较不同团队开发的模型，并追踪此方面的进展。它们迫使模型展现应对多种语言挑战的能力。

### 局限性与考量

尽管有用，但仅依赖GLUE/SuperGLUE存在不足：

- **计算成本：** 在测试集中对每个任务进行大型模型的微调 (fine-tuning)和评估可能计算开销大且耗时。
- **任务侧重：** 它们主要侧重于基于对输入文本理解的分类和回归任务。它们不直接有效评估连贯性、创造性或长文本 (long context)生成等生成能力。
- **静态性质：** 数据集是固定的。它们不一定反映不断变化的语言使用或新兴的NLP挑战。

---

### 少量示例和零示例评估

# 少量示例和零示例评估

虽然微调 (fine-tuning)提供了一种充分定制预训练 (pre-training)模型以适应特定任务的方式，但它需要标注数据和计算资源进行训练。此外，它主要评估模型的*适应性*。通常，我们感兴趣的是评估模型在大规模预训练阶段学到的固有能力，特别是指模型在不进行特定任务梯度更新的情况下理解和遵循指令或执行任务的能力。这就是零示例和少量示例评估方法具有特别的价值之处。这些技术衡量模型如何将预训练知识泛化到新任务，仅通过自然语言提示以及（可能）直接在输入上下文 (context)中提供的少数示例来引导。

### 理解零示例评估

零示例评估评估大语言模型 (LLM)执行从未显式训练过的任务的能力，不使用特定任务的示例。我们不进行微调 (fine-tuning)，而是完全依赖于模型的预训练 (pre-training)知识及其理解提示中提供的任务描述或指令的能力。在评估配置中，模型看不到任何特定下游任务格式的示例（k=0k=0k=0）。

考虑评估预训练大语言模型的情感分析能力。在零示例设置中，您不会在情感数据集上微调模型。相反，您可能会提供如下提示：

```
文本: "这部电影真是太棒了，简直是杰作！"
情感 (正面/负面):
```

或者更明确一些：

```
将以下文本的情感归类为正面或负面。

文本: "航班延误了，服务也很糟糕。"
情感:
```

模型应利用其对语言的理解，包括“棒极了”或“糟糕”等词语的语义，生成正确的分类结果（“正面”或“负面”）。

**实现概述：**

使用预训练模型接口（例如，来自 `transformers` 等库），该过程在 PyTorch 中可能如下所示：

```python
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer
)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def classify_sentiment_zero_shot(text):
    """使用零示例提示对情感进行分类。"""
    prompt = f"""Classify the sentiment of the following text
as 'positive' or 'negative'.

Text: "{text}"
Sentiment:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0, inputs.input_ids.shape[1]:]
    result = tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip().lower()

    if "positive" in result:
        return "positive"
    elif "negative" in result:
        return "negative"
    else:
        return "unknown"

text_to_classify = "The product broke after just one week."
sentiment = classify_sentiment_zero_shot(text_to_classify)
print(f"Text: '{text_to_classify}'")
print(f"Predicted Sentiment (Zero-Shot): {sentiment}")
```

**优点：**

- **无需特定任务数据：** 它直接测试模型的泛化能力，无需目标任务的任何标注示例。
- **快速评估：** 可以快速评估模型在多项任务上的能力。
- **基线表现：** 建立模型“开箱即用”的知识基础。

**缺点：**

- **表现较差：** 与微调模型相比，准确性通常较低，因为它缺少特定任务的适应性。
- **提示敏感性：** 表现高度依赖于提示的确切措辞和格式。微小改变可能导致表现出现较大差异。
- **指令遵循能力：** 非常依赖于模型理解和遵循所提供指令的能力，这在不同模型之间存在差异（特别是未经指令微调的模型）。

### 理解少量示例评估（上下文 (context)学习）

少量示例评估，通常被称为大语言模型 (LLM)的上下文学习，直接在提示中向模型提供少量 (kkk，通常在 1 到 32 之间) 任务示例。重要的是，模型的权重 (weight)*不会*根据这些示例进行*更新*；它们仅作为上下文或演示，用于引导模型对后续实际查询实例的预测。

继续情感分析的示例，一个单示例 (k=1k=1k=1) 提示可能如下所示：

```
对文本情感进行分类。

文本: "我喜欢这场演唱会，乐队太棒了！"
情感: positive

文本: "这本书相当无聊且老套。"
情感:
```

一个双示例 (k=2k=2k=2) 提示：

```
对文本情感进行分类。

文本: "我喜欢这场演唱会，乐队太棒了！"
情感: positive

文本: "客户支持态度不佳，响应缓慢。"
情感: negative

文本: "这家新咖啡店氛围很好，饮品也很美味。"
情感:
```

模型观察提供的示例中的模式（输入文本后接期望的输出标签），并将其应用于最终的未标注实例。

**实现概述：**

代码结构与零示例类似，但提示构建会改变以包含示例。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def classify_sentiment_few_shot(text, examples, k):
    """使用 K 示例提示对情感进行分类。"""
    prompt = (
        "Classify the sentiment of the text as 'positive' or 'negative'.\n\n"
    )

    for i in range(min(k, len(examples))):
        example_text, example_sentiment = examples[i]
        prompt += f"Text: \"{example_text}\"\n"
        prompt += f"Sentiment: {example_sentiment}\n\n"

    prompt += f"Text: \"{text}\"\nSentiment:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0, inputs.input_ids.shape[1]:]
    decoded_text = tokenizer.decode(
        generated_ids, skip_special_tokens=True
    )
    result = decoded_text.strip().lower()

    if "positive" in result:
        return "positive"
    elif "negative" in result:
        return "negative"
    else:
        return "unknown"

few_shot_examples = [
    ("今天天气晴朗美丽。", "positive"),
    ("我的订单收到时已损坏且不完整。", "negative")
]
text_to_classify = "这部电影视觉效果惊艳但情节薄弱。"
k_shots = 2

sentiment = classify_sentiment_few_shot(
    text_to_classify, few_shot_examples, k_shots
)
print(f"Text: '{text_to_classify}'")
print(f"Predicted Sentiment ({k_shots}-Shot): {sentiment}")
```

**优点：**

- **表现提升：** 通常明显优于零示例，因为示例明确了任务格式和预期输出。
- **仍无需微调 (fine-tuning)：** 避免了训练的计算成本和数据要求。
- **上下文适应性：** 允许模型根据推理 (inference)时提供的特定示例调整其行为。

**缺点：**

- **示例敏感性：** 表现高度依赖于 kkk 个示例的选择、质量和顺序。糟糕的示例可能降低表现。
- **上下文长度限制：** 示例数量 (kkk) 受限于模型的最大上下文窗口大小。放入过多长示例可能超出此限制。
- **每次推理的计算成本：** 由于提示更长，推理与零示例相比略慢且更占用内存。
- **提示格式：** 仍然需要仔细的提示设计，包括示例的格式和分隔方式。

### 提示工程 (prompt engineering)的考量

零示例和少量示例评估都突出了提示工程的重要性。任务的描述或演示方式可以显著改变模型的输出。重要考量包括：

- **指令清晰度（零示例）：** 指令应清晰明确，并清楚说明所需的任务和输出格式。
- **示例选择（少量示例）：** 示例应具有任务代表性、多样化且格式正确。示例的选择可能使模型产生偏向。有时平衡正面/负面示例或使用与查询实例相似的示例会有帮助。
- **格式化：** 指令、输入、输出和分隔符（如换行符）的一致格式非常重要。
- **高级提示技巧：** 诸如思维链 (CoT) 提示之类的技巧，即提示模型“一步步思考”（通常在少量示例中演示），可以提升推理 (inference)任务的表现，尽管它们通常单独评估。

### 零示例、少量示例与微调 (fine-tuning)之间的选择

- **零示例：** 最适合快速评估模型广泛的固有能力，比较基础模型，或者在绝对没有任务示例可用时使用。它测试纯粹的泛化能力和指令理解。
- **少量示例（上下文 (context)学习）：** 适用于有少量示例可用时，或者您希望评估模型在不更新权重 (weight)的情况下从上下文学习的水平。通常比零示例能更好地估计潜在表现。
- **完全微调：** 当目标是最大化特定任务的表现，并且有足够的标注数据和计算资源可用时，这是必要的。它评估模型的适应性和特定化潜力。

零示例和少量示例评估是理解大语言模型 (LLM)的重要工具。它们通过提供对模型一般知识及其仅基于上下文和指令将该知识应用于新任务的能力的认识来补充微调，这反映了模型能力的一个不同维度，区别于通过梯度下降 (gradient descent)进行特定任务的适应。分析 GLUE 或 SuperGLUE 等基准测试时，通常会报告所有三种模式（零示例、少量示例、微调）的结果，以提供模型表现的全面情况。

获取即时帮助、个性化解释和交互式代码示例。

---

### 开发定制评估任务

# 开发定制评估任务

标准基准测试（如GLUE和SuperGLUE）提供了有益的比较依据，但它们在描述大语言模型 (LLM)在特定应用中的表现时，常常不够全面。您的具体应用场景可能涉及一个独特的专业范畴、一种新的互动模式，或者需要现有公开数据集未充分测试的能力。这些情况使得开发定制评估任务不仅有益，而且对于确认模型是否真正达到所需的性能标准，这是不可或缺的。设计、实施和分析这些定制评估的过程详细说明。

### 明确评估目标

在编写一行代码或收集任何数据之前，第一步是精确阐明您需要衡量什么。标准基准测试通常评估通用语言能力或在既定自然语言处理任务上的表现。然而，定制评估通常针对与您的应用更相关的特定行为或知识。请自问：

- **对我的应用程序来说，哪些特定能力很重要？** 是准确总结内部法律文件？根据自然语言请求生成语法正确的SQL查询？在客户服务聊天机器人中保持一致、富有同情心的人设？还是根据专有知识库回答问题？
- **这种能力下，“好的表现”是什么样的？** 这要求摆脱对质量的笼统认识。定义具体的成功标准。对于SQL生成，成功可能意味着查询可执行并返回正确数据。对于摘要，可能涉及包含特定实体或遵守长度限制。
- **这与现有基准测试有何不同？** 弄清这种差异有助于明确定制评估所提供的独特价值。

在此阶段保持清晰很重要。像“评估模型是否擅长金融”这样模糊的目标难以操作。而像“评估模型从季度财报中提取‘总收入’数据的能力，准确率超过95%”这样的具体目标，则为任务设计和指标开发提供了明确的目标。

### 设计任务格式

目标明确后，您需要设计任务格式，以引发模型所需的行为。该格式应尽可能模拟模型在生产环境中的使用方式。常见格式包括：

- **分类：** 为输入分配预定义标签（例如，将客户反馈分类为“正面”、“负面”、“中立”；识别用户查询背后的意图）。
- **提取：** 从文本中识别并提取特定信息（例如，从新闻文章中提取姓名、日期和地点；从研究论文中提取重要术语）。
- **生成：** 根据提示生成文本（例如，撰写电子邮件草稿、生成代码文档、会议纪要）。
- **问答（QA）：** 根据提供的上下文 (context)或内部知识回答问题（例如，根据公司政策回答常见问题、查询技术手册）。
- **排序：** 根据相关性或偏好对一组项目进行排序（例如，对搜索结果进行排序，对产品推荐进行排序）。
- **对话：** 模拟多轮对话以评估连贯性、有用性或任务完成能力。

考虑您的模型将接收的输入以及您期望的输出。例如，如果评估模型遵循复杂指令的能力，任务可能涉及提供一个详细的提示，阐明限制和预期输出，然后根据这些限制评估生成的文本。

### 数据收集与整理

定制评估的质量直接取决于评估数据的质量。

- **来源：** 数据可以来自各种地方。生产日志可能提供用户互动的真实示例。特定专业范畴的专家能够创建与其专业范畴高度相关的优质示例。如果真实数据稀缺，合成数据生成（可能使用另一个大语言模型 (LLM)并仔细审查）可以是一种选择，尽管它存在引入偏见或虚假信息的风险。
- **标注：** 如果您的任务需要人工判断（例如，评价回复的有用性，识别摘要是否抓住了主要观点），您将需要清晰的标注指南。这些指南应精确定义标签或分数，提供边缘案例的示例，并旨在尽量减少歧义。投入时间培训标注者，并衡量标注者间一致性（IAA），以确保一致性。科恩卡帕系数（κ\kappaκ）或弗莱斯卡帕系数等工具可以量化 (quantization)标注者间一致性。高IAA表明您的指南清晰且任务定义明确。
- **黄金标准：** 为您的评估集建立基本事实或“黄金标准”答案/标签。对于分类或提取任务，这通常是直截了当的。对于生成任务，则更为复杂。总结文档或回答问题可能存在多种有效方式。在这些情况下，您的黄金标准可能包含多个可接受的参考，或者您的评估指标可能需要考虑到这种可变性。
- **数据集大小：** 所需大小取决于任务和所需的统计显著性。即使是较小的、高质量的数据集（例如，100-500个精心整理的示例）对于识别特定失败模式非常有帮助，尽管更大的数据集更适合定量分析。确保您的数据集涵盖各种输入和潜在挑战。

### 制定评估指标

准确率、F1分数、BLEU或ROUGE等标准指标可以作为起点，但它们往往无法捕捉自定义任务的精细之处。您常常需要制定与您的特定评估目标相符的定制指标。

- **基于规则的指标：** 这些指标涉及基于预定义规则的程序化检查。它们对于评估格式的遵守、所需元素的包含或禁用内容的避免很有用。

  - *示例：* 检查生成的API调用是否包含正确的函数名和所需参数 (parameter)。
  - *示例：* 验证摘要是否在指定的字数范围内。

  ```python
  import re

  def check_report_format(generated_text: str) -> bool:
      """检查生成的文本是否包含“摘要：”部分和“建议：”部分"""
      has_summary = bool(re.search(r"Summary:", generated_text, re.IGNORECASE))
      has_recommendations = bool(re.search(r"Recommendations:", generated_text, re.IGNORECASE))
      return has_summary and has_recommendations

  report = """
  Analysis Complete.
  Summary: Sales increased by 10%.
  Recommendations: Invest in marketing.
  """
  is_valid_format = check_report_format(report)
  print(f"报告格式有效: {is_valid_format}")
  ```
- **基于模型的指标：** 借助其他模型（可能是更小、更专业的模型）来评估输出。

  - *示例：* 使用毒性分类器来评估生成对话的安全性。
  - *示例：* 使用另一个大语言模型 (LLM)或嵌入 (embedding)模型来评估生成答案和黄金标准答案之间的语义相似性，这比单纯的词汇重叠评估（如BLEU/ROUGE）更进一步。
  - *示例：* 使用代码分析工具检查生成的代码是否存在语法错误或漏洞。
- **人工评估：** 在评估主观质量时不可或缺，例如有用性、连贯性、创造力、事实正确性（特别是模型训练数据之外的信息），或遵守特定语调/人设。设计良好的人工评估需要：

  - **清晰的评分标准：** 定义具体标准和评分量表（例如，有用性使用1-5的李克特量表，事实准确性使用二元判断）。
  - **比较评估：** 通常，要求人工比较两个输出（例如，模型A与模型B的输出）并选择更好的一个，比分配绝对分数更容易、更可靠。
  - **盲评：** 确保评估者不知道哪个模型生成了哪个输出，以避免偏见。

### 实施与执行

任务明确、数据收集完毕、指标选定后，您需要构建评估流程。

- **输入/输出处理：** 编写代码将评估数据格式化为适合模型的提示，并解析模型生成的输出。
- **模型推理 (inference)：** 与您的模型服务系统或推理库集成，以在评估数据集上运行模型。
- **指标计算：** 实现您的自定义指标（基于规则、基于模型或处理人工判断）的逻辑。
- **自动化：** 尽可能自动化流程，以便在模型更新时进行高效的重新评估。

这是使用PyTorch运行评估的简化结构：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_custom_evaluation(model_name, custom_data_loader, custom_metric_function):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results = []
    total_score = 0.0
    num_samples = 0

    with torch.no_grad():
        for prompt, gold_reference in custom_data_loader:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            generated_response = generated_text[len(prompt):]

            score = custom_metric_function(generated_response, gold_reference)
            results.append({
                "prompt": prompt,
                "generated": generated_response,
                "gold": gold_reference,
                "score": score
            })
            total_score += score
            num_samples += 1

    average_score = total_score / num_samples if num_samples > 0 else 0
    print(f"平均自定义分数: {average_score:.4f}")
    return results, average_score
```

*注意：这只是一个简化示例。实际评估通常涉及更精密的生成策略、批处理和错误处理。*

### 分析与迭代

仅仅计算一个总分数是不够的。真正的价值在于分析结果，以了解模型成功或失败的*原因*。

- **错误分析：** 人工审查评估实例的样本，特别是模型表现不佳的那些。对错误进行分类。模型是否生成虚假事实？是否未能遵循指令？是否产生了格式不正确的输出？是否表现出偏见？
- **定性分析：** 寻找成功和失败中的模式。模型是否在某些特定类型的提示或主题上存在困难？
- **迭代：** 利用分析所得的洞察来优化模型（例如，通过对与失败案例相似的数据进行进一步微调 (fine-tuning)）、评估任务本身（例如，澄清指令）、评估数据（例如，添加更具挑战性的示例），或优化指标（例如，设计一个更能捕捉特定失败模式的指标）。定制评估的开发通常是一个迭代过程。

CustomEvalDev

DefineGoal

明确目标

DesignTask

设计任务格式

DefineGoal->DesignTask

CollectData

收集/整理数据

DesignTask->CollectData

DevelopMetrics

制定指标

CollectData->DevelopMetrics

Implement

实施流程

DevelopMetrics->Implement

Execute

执行评估

Implement->Execute

Analyze

分析结果
(错误分析)

Execute->Analyze

Refine

优化模型、任务、
数据或指标

Analyze->Refine

识别问题

Refine->DefineGoal

迭代

Refine->DesignTask

Refine->CollectData

Refine->DevelopMetrics

> 定制大语言模型 (LLM)评估任务的迭代开发周期。

### 挑战与考量

开发定制评估需要仔细思考和资源投入：

- **成本与精力：** 创建高质量数据集和标注指南，特别是那些需要专业知识或大量人工标注的数据集和指南，可能耗时且昂贵。
- **指标有效性：** 确保您的自定义指标准确反映真实质量标准具有挑战性。一个指标可能很容易计算，但却无法与实际用户满意度或任务成功率良好关联。
- **偏见：** 评估数据集和指标可能无意中包含源数据或标注过程中存在的偏见。应积极寻找并减少与人口统计、观点或其他敏感属性相关的潜在偏见。
- **可扩展性：** 人工评估虽然有价值，但难以轻易扩展以进行频繁、大规模的测试。应平衡详细的人工分析与更具可扩展性的自动化指标。
- **维护：** 随着应用程序或模型的发展，定制评估可能需要更新以保持相关性。

尽管存在这些挑战，但精心设计的定制评估能够提供关于您大语言模型 (LLM)能力和不足的不可或缺的洞察，比仅仅依赖通用基准测试能更有效地指导开发工作。它们弥合了抽象的语言建模表现与实际取得的成果之间的差距。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 23 Analyzing Model Behavior

### 解读大型语言模型的难题

# 解读大型语言模型的难题

构建和训练大型语言模型需要复杂的机制，但要理解一个训练好的模型为何会如此表现，这本身就带来了一系列重大困难。大型语言模型（LLMs）所具有的庞大规模和复杂的架构使其具备了出色的能力，但同时也让它们难以解释其运行原理。与那些可以追踪特定特征或参数 (parameter)影响的较小、更简单的模型不同，大型语言模型更像是错综复杂、高维度的系统，其内部运作原理并不一目了然。

其中一个主要障碍是其庞大的**规模**。现代大型语言模型包含数十亿，有时甚至数万亿个参数。这些参数在数十甚至数百个层之间以高度复杂、非线性的方式相互影响。以一次单独的预测为例：追踪每个参数通过矩阵乘法、非线性激活函数 (activation function)（如GeLU或SwiGLU）、层归一化 (normalization)和注意力机制 (attention mechanism)序列的确切贡献，这在计算上是不可行的，而且即便能做到，也可能无法得出人类可以理解的解释。最终的输出是这个巨大参数空间中细微而集体的相互影响的结果。

**Transformer架构**本身也增加了理解的难度。自注意力 (self-attention)机制允许模型动态地衡量输入序列中不同词元 (token)的重要性。尽管将这些注意力权重 (weight)（我们将在后面讨论）提供了一些线索，但这并非一个明确的解释。高注意力权重并不总是等同于对最终预测的因果重要性。此外，多头注意力 (multi-head attention)设计明确地促使模型在并行子空间中学习不同的关系模式，这使得任何简单的解释变得碎片化。然后这些模式被结合并通过前馈网络进行处理，以复杂的方式进一步转换表示。

大型语言模型大量依赖于**分布式表示**。与单个神经元可能代表特定想法（例如，“情感”或“对象类型”）的模型不同，大型语言模型中的信息通常编码在大量的神经元群体中。想法以高维嵌入 (embedding)空间（dmodeld\_{model}dmodel​通常 > 1000）中的方向或区域形式存在。特定的语言特征或知识并非存储在一个地方，而是从多个维度的激活模式中浮现出来。

G

clusterₗocal

局部表示（更简单的模型）

cluster\_distributed

分布式表示（大型语言模型）

想法A

想法A

想法B

想法B

单元1

单元1

单元1->想法A

 直接连接

单元2

单元2

单元2->想法B

 直接连接

单元3

单元3

单元4

单元4

想法X

想法X

想法Y

想法Y

N1

N1

N1->想法X

 模式1

N2

N2

N2->想法Y

 模式2

N3

N3

N3->想法X

 模式1

N4

N4

N4->想法X

 模式1

N4->想法Y

 模式2

N5

N5

N5->想法Y

 模式2

N6

N6

N6->想法Y

 模式2

> 这是一个简化的视图，对比了局部表示（其中想法可能与单个单元紧密对应）和大型语言模型中的分布式表示（其中想法从许多单元的模式中浮现出来）。

这种分布式特性意味着，剖析模型的内部状态以找到某个输出的特定“原因”，就如同试图通过观察大脑中单个神经元的放电来理解一个想法一样；有意义的信息存在于集体的活动中。

令情况更复杂的是第1章讨论的**涌现 (emergence)能力**。诸如少样本学习 (few-shot learning)或复杂推理 (inference)等能力并非被明确设计到架构中，而是随着模型规模、数据量和计算能力的增加而产生的。由于这些行为并非直接编程实现，因此通过事后分析来确定产生它们的具体机制极其困难。它们是整个系统的特性。

最后，即便我们现有的分析方法也存在**固有的局限性**。将注意力可视化可能会突出模型关注的区域，但不能说明这种关注是如何或为何转化为输出的。探测任务，即我们在内部激活上训练简单分类器，可以显示关联性（例如，某些层编码语法信息），但在确定因果关系方面存在困难。探测器可能成功地从嵌入中预测词性标签，但这并不能明确证明模型以探测器的方式使用了这些信息，或者它是下游任务的重要因素。

---

### 注意力图可视化

# 注意力图可视化

了解 Transformer 模型内部运作方式最直接的途径之一是检查其注意力机制 (attention mechanism)。自注意力 (self-attention)作为 Transformer 架构的**主要**组成部分，让模型在计算特定标记 (token)的表示时，能够衡量输入序列中不同标记的重要性。这些注意力权重 (weight)，为每一层中的每个头计算，构成图谱，显示信息如何在模型中传递。可视化这些图谱可以提供模型在标记之间学习到的联系的线索。

自注意力的核心思想涉及计算一个标记的查询向量 (vector)(QQQ)与序列中所有标记（包括其自身）的键向量(KKK)之间的分数。这些分数经过缩放，并使用 softmax 归一化 (normalization)，然后用于计算值向量(VVV)的加权和。注意力权重是缩放点积后应用 softmax 的结果：

权重=softmax(QKTdk)\text{权重} = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)权重=softmax(dk​​QKT​)

这里，dkd\_kdk​是键向量的维度。这些权重表示每个查询标记对所有键标记的注意力分布。权重越高表示模型认为对应的键标记在生成查询标记的表示时更重要。

### 提取注意力权重 (weight)

大多数现代深度学习 (deep learning)框架，包括 PyTorch，都提供机制来在前向传播过程中访问这些注意力权重。当使用 PyTorch 的 `nn.MultiheadAttention` 层时，你可以在前向调用中指定`need_weights=True`。这个参数 (parameter)指示层除了输出之外，还返回所有头的平均注意力权重。对于更细致的、特定于头的权重，你可能需要稍微修改层的实现，或使用钩子在权重平均之前捕获它们。

这里是一个简化示例，展示如何在 PyTorch 中从 `nn.MultiheadAttention` 层获取注意力权重：

```python
import torch
import torch.nn as nn

seq_len = 5
embed_dim = 8
num_heads = 2
batch_size = 1

assert embed_dim % num_heads == 0

mha_layer = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

query = torch.randn(batch_size, seq_len, embed_dim)
key = torch.randn(batch_size, seq_len, embed_dim)
value = torch.randn(batch_size, seq_len, embed_dim)

attn_output, attn_output_weights = mha_layer(
    query, value,
    need_weights=True,
    average_attn_weights=True
)

print("平均注意力权重的形状:", attn_output_weights.shape)

first_batch_weights = attn_output_weights[0]
```

请注意，标准的 `nn.MultiheadAttention` 在 `average_attn_weights` 为 `True`（如果 `need_weights` 为 `True` 时的默认值）时，返回在各头之间平均后的权重。访问单个头的权重通常需要修改前向方法，或者更简洁地，在注意力机制 (attention mechanism)的内部 softmax 或矩阵乘法操作上注册一个前向钩子，以便在平均之前捕获权重。

### 可视化方法

一旦提取，注意力权重 (weight)（通常是每个头/层的 `(序列长度, 序列长度)` 大小的矩阵）可以通过几种方式可视化：

1. **热力图：** 这是最常用的方法。热力图显示注意力矩阵，其中行表示查询标记 (token)（输出位置），列表示键标记（输入位置）。单元格 `(i, j)` 的颜色强度表示标记 `i` 对标记 `j` 的注意力权重。颜色越浅通常表示注意力越高。分析这些热力图可以显示出一些模式，例如强对角线（标记关注自身）、对前置标记的注意力，或者特定标记（如标点符号或特殊标记）充当信息接收器或源。

[CLS]thecatsat[SEP][CLS]thecatsat[SEP]0.10.20.30.40.50.60.70.8

> 单个头的注意力权重。请注意表示自注意力 (self-attention)的强对角线，以及“sat”如何强烈关注“cat”。特殊标记 `[CLS]` 主要关注自身，而 `[SEP]` 也显示出高自注意力。

2. **多头可视化：** 由于每个层包含多个注意力头，将它们全部可视化很重要。常用方法包括：

   - **小倍数图：** 显示一个热力图网格，每个头一个。这允许直接比较不同头学习到的模式。
   - **平均热力图：** 显示一个代表某层中所有头平均权重的单个热力图。这提供了一个概览，但可能掩盖特定于头的行为。
3. **基于图的可视化：** 注意力权重可以表示为有向图，其中标记是节点，如果注意力权重 wijw\_{ij}wij​ 超过某个阈值，则存在从标记 `i` 到标记 `j` 的有向边。边的粗细或颜色可以表示权重的量级。这对于可视化较短序列中的连接或突出特定的强联系可能有效。

G

the

the

cat

cat

the->cat

cat->the

sat

sat

sat->the

sat->cat

> 简化图，显示强注意力链接。“sat”强烈关注“cat”，而“cat”对“the”也有明显关注。

### 解释和理解

分析注意力模式有时可以显示出语言学上合理的行为：

- **句法依赖：** 头可能学习到关注句法相关的词语，比如动词关注它们的主语或宾语，或形容词关注它们修饰的名词。
- **指代消解：** 注意力可能将代词与其在文本中较早提及的名词联系起来。
- **位置信息：** 有些头经常集中关注紧邻的前一个或后一个标记 (token)，或相对位置偏移。
- **特殊标记：** 像 `[CLS]` 或 `[SEP]` 这样的标记可能汇总整个序列的信息，这表现为源自或指向它们的广泛注意力模式。
- **层级进展：** 注意力模式在不同层之间常有差异。早期层可能关注局部、句法关系，而更靠后的层可能捕获更复杂、长距离或语义上的连接。

### 局限性和注意事项

虽然注意力可视化是一种有用的工具，但了解其局限性很重要：

- **注意力不等于解释：** 高注意力权重 (weight)不一定意味着某个标记 (token)是特定输出的主要原因。注意力表示在构建下一层表示时，哪些标记的表示被赋予了高权重，但前馈网络内部和跨层间的复杂转换模糊了直接的因果关系。研究表明，注意力权重可能不总是与梯度等其他特征重要性指标强相关。
- **平均化问题：** 在各头之间平均权重（某些框架实现中的默认做法）可能隐藏单个头学习到的多样化甚至矛盾的模式。有些头可能学习到专门的功能，而另一些则显得噪声或冗余。
- **Softmax 饱和/扩散：** Softmax 函数强制权重总和为1。如果没有哪个标记明确重要，注意力可能会分散到许多标记上，使得解释变得困难。反之，如果某个标记高度相关，其权重可能接近1，从而抑制其他潜在相关标记的可见权重。
- **深度模型的复杂性：** 在非常深的 Transformer 中，层之间传递的表示变得越来越抽象。后面层中的注意力模式作用于这些复杂的表示，使得它们直接映射回原始输入标记的解释变得不那么直接。

注意力图可视化提供了一个窗口（尽管有时模糊），用于了解 Transformer 内部的信息流。这是一种有用的诊断技术，用于对模型行为生成假设并找出潜在的关注点，但应谨慎得出结论，最好与本章后面讨论的其他分析方法相互印证，例如探查内部表示或分析神经元激活。

获取即时帮助、个性化解释和交互式代码示例。

---

### 探查内部表示

# 探查内部表示

可视化注意力模式提供了一种了解模型关注点的方式，但这不能直接告诉我们每个层生成的高维隐藏状态向量 (vector)中编码了哪种语言或语义信息。这些通常具有数千维的向量，是Transformer的内部数据流通形式，将信息从一层传递到下一层。为了理解这些表示捕获了什么，我们采用一种称为*探查*的技术。

探查涉及训练简单的辅助模型，称为*探针*，以直接从LLM的内部表示中预测特定的属性。其核心思想是：如果一个简单的探针能够仅使用特定层的隐藏状态向量作为输入，准确预测某个属性（例如词性标签或依存关系），那么该信息可能被明确编码或至少在该表示中是线性可分离的。我们不太关心构建属性本身的最佳预测器；相反，我们将探针的性能用作评估LLM表示质量的诊断工具。

### 探查方法

典型的探查流程包括以下几个步骤：

1. **定义任务和获取数据：** 选择你想研究的属性（例如，识别名词的语法数，预测句法中心词）。你需要一个文本输入已标注这些属性的数据集。对于语言属性，通常使用Universal Dependencies或Penn Treebank等资源。
2. **提取表示：** 使用你想要分析的预训练 (pre-training)LLM处理已标注的数据集。重要的一点是，在此过程中LLM的权重 (weight)保持*冻结*。对于输入序列中的每个token，从你感兴趣的特定层中提取隐藏状态向量 (vector)。
3. **训练探针：** 使用提取的隐藏状态作为输入特征，已标注的属性作为目标标签，训练一个简单的分类器或回归器。探针的常见选择包括线性分类器（逻辑回归、线性支持向量机）或非常浅的多层感知机（MLP）。简单性很重要：一个复杂的探针可能会独立学习任务，而不是仅仅“读取”表示中已经存在的信息。
4. **评估探针：** 使用相关指标（例如，分类准确率，回归的均方误差）在保留的测试集上评估探针的性能。
5. **结果解读：** 高性能表明该层中的表示包含了关于所探查属性的易于访问的信息。比较不同层间的性能可以显示信息在通过模型时是如何转换或优化的。与基线探针（例如，在随机向量或静态词嵌入 (embedding)上训练的探针）进行比较有助于确定结果的重要性。

### 示例：探查词性标签

我们考虑探查预训练 (pre-training)的Transformer模型（如BERT或GPT变体）中的词性（POS）信息。

**1. 数据：** 我们需要一个带有词性标签的语料库（例如，Universal Dependencies English Web Treebank）。
**2. 表示：** 我们将语料库中的句子输入到我们冻结的LLM中，并从例如第6、12和18层收集每个token的隐藏状态向量 (vector)。
**3. 探针：** 我们选择一个简单的线性分类器。
**4. 训练与评估：** 对于每一层，我们训练一个单独的线性探针，仅根据来自该层的隐藏状态向量来预测每个token的词性标签。

这是一个PyTorch代码片段，说明了表示提取和探针定义：

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

for param in model.parameters():
    param.requires_grad = False

sentence = "Probing helps analyze model representations."

inputs = tokenizer(sentence, return_tensors="pt")

target_layer = 8
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

    layer_representations = outputs.hidden_states[target_layer]

hidden_size = layer_representations.shape[-1]
num_pos_tags = 17
probe_classifier = nn.Linear(hidden_size, num_pos_tags)
```

### 探针结果解读

探查实验的结果可能非常具有启发性：

- **性能大小：** 高准确率（例如，使用线性探针达到90%以上的词性标注准确率）强烈表明该特定层以易于访问的格式编码了此信息。
- **分层比较：** 通常，词性标注或依存句法分析等句法任务的性能在Transformer的中间层达到顶点，而需要更多语义整合的任务则可能在更后面的层达到顶点。这表明信息从低级别语言特征向高级别抽象逐步演进。
- **对照任务：** 进行对照任务是好的实践。例如，训练一个探针来预测分配给token的随机标签。如果这个探针的表现明显高于随机水平，则可能表明实验设置存在偏差或问题。另一个对照是*选择性*：确保为词性标注训练的探针在例如预测依存关系时表现不佳，反之亦然。这证实了探针是专门使用目标信息。

Layer 4Layer 8Layer 12Layer 16Layer 20020406080100词性标注准确率依存中心词准确率探针在各层上的准确率Transformer 层探针准确率 (%)

> 比较显示，词性标注准确率在模型层中比依存关系准确率更早达到峰值。

### 探查任务的范围

探查可应用于广泛的语言和语义现象：

- **形态句法：** 时态、数、格、词性标签。
- **句法：** 句法成分边界、依存关系、语法角色识别。
- **语义：** 语义角色标注（SRL）、命名实体识别（NER）、共指消解、关系抽取。
- **知识：** 尽管线性探查较难，但有些研究试图探查表示中嵌入 (embedding)的事实知识。

### 局限与考量

探查是一种有用的分析工具，但注意其局限性很重要：

- **探针的简单性与能力：** 探针必须足够简单，这样我们才能合理地将高表现归因于表示本身，而不是探针的学习能力。总会有权衡；一个稍微复杂一点的探针可能会显现出非线性可分离但仍易于提取的信息。
- **相关性，而非因果性：** 发现某一层编码了词性信息，并不一定意味着模型会以我们预期的方式将这些信息明确用于其主要目标（例如，下一个token预测）。这表明信息是存在且可访问的。
- **数据依赖性：** 用于探查的已标注数据集的质量和性质非常重要。
- **分词 (tokenization)对齐 (alignment)：** 需要仔细对齐LLM的分词和语言标注（可能基于词语）。

尽管有这些方面，探查仍为理解大型语言模型学习到的内部知识结构提供了宝贵的看法。通过系统地检视不同类型的信息在各层中是如何表示的，我们能更好地理解这些模型如何处理语言，这能为调试、模型改进以及构建更可靠、更易解释的系统提供依据。

获取即时帮助、个性化解释和交互式代码示例。

---

### 神经元激活分析

# 神经元激活分析

可视化注意力图能够提供关于令牌关系的信息，探测任务则评估整个隐藏状态中编码的信息。分析单个神经元，特别是前馈网络（FFN）中神经元的激活模式，能够提供对模型内部计算的细致视角。弄清哪些输入会引起特定神经元的强烈“激活”，可以显示出网络中学习到的特征或专门的功能。

可以把FFN层看作是处理注意力机制 (attention mechanism)聚合的信息。这些层中的每个神经元都对其输入计算一个非线性函数。通过观察特定神经元何时具有高激活值，我们可以推断它可能对哪种输入模式敏感。这种分析有助于回答以下问题：这个神经元主要响应特定词语、句法结构、带有情感的短语，还是其他抽象观念？

### 识别高激活输入

一种直接的方法是，从数据集中找出能使给定神经元产生最高激活的特定输入示例。这需要在大规模语料库上运行模型，并记录每个输入序列或令牌位置的目标神经元的激活值。

在实际中，使用PyTorch实现这一点，你可以在包含目标神经元的特定模块（例如，FFN块内的线性层）上注册一个“前向钩子”。前向钩子是一个函数，它在模块的前向传播过程中执行。它接收模块本身、其输入和输出作为参数 (parameter)。

```python
import torch
import torch.nn as nn

target_layer = model.decoder.layers[0].ffn.linear_1
neuron_index = 123

activations = {}

def get_activation_hook(neuron_idx):
    def hook(module, input, output):

        max_activation = torch.max(output[:, :, neuron_idx]).item()

        activations[max_activation] = "placeholder_for_input_example"
    return hook

hook_handle = target_layer.register_forward_hook(get_activation_hook(neuron_index))

model.eval()
with torch.no_grad():
    for batch in dataloader:

        inputs = batch['input_ids'].to(model.device)
        attn_mask = batch['attention_mask'].to(model.device)
        _ = model(input_ids=inputs, attention_mask=attn_mask)

hook_handle.remove()

sorted_activations = sorted(activations.keys(), reverse=True)
print("最高激活示例（按激活值排序）：")
for i in range(min(5, len(sorted_activations))):
    activation_value = sorted_activations[i]

    example = activations[activation_value]
    print(f"激活: {activation_value:.4f}, 示例: {example}")
```

通过检查那些持续引发特定神经元高激活的文本输入，你可能会发现模式。例如，一个神经元可能会对包含否定、特定专有名词、金融术语或疑问的句子强烈激活。这为该神经元在网络处理中可能扮演的专门作用提供了线索。

### 聚合数据集上的激活值

除了只查看最高激活的示例，分析神经元在大规模、多样化数据集上的激活*分布*也很有价值。神经元是很少激活，还是经常激活？它的激活通常很低，偶尔出现高峰，还是经常保持中等激活水平？

你可以使用与上面类似的回调机制收集激活值，但不是存储单个示例，而是累积神经元激活值的统计数据（例如，均值、方差、直方图）。

```python
import torch
import numpy as np

activation_values = []

def collect_activations_hook(neuron_idx):
    def hook(module, input, output):

        neuron_activations = output[:, :, neuron_idx].detach().cpu().numpy().flatten()
        activation_values.extend(neuron_activations)
    return hook

hook_handle = target_layer.register_forward_hook(collect_activations_hook(neuron_index))

model.eval()
with torch.no_grad():
    for batch in dataloader:
        inputs = batch['input_ids'].to(model.device)
        attn_mask = batch['attention_mask'].to(model.device)
        _ = model(input_ids=inputs, attention_mask=attn_mask)

hook_handle.remove()

activations_array = np.array(activation_values)
print(f"神经元 {neuron_index} 的激活统计数据：")
print(f"  平均值: {np.mean(activations_array):.4f}")
print(f"  标准差: {np.std(activations_array):.4f}")
print(f"  中位数: {np.median(activations_array):.4f}")
print(f"  最大值: {np.max(activations_array):.4f}")
print(f"  最小值: {np.min(activations_array):.4f}")

import plotly.graph_objects as go

hist_data = np.random.normal(loc=0.5, scale=0.2, size=1000)
hist_data = hist_data[(hist_data >= 0) & (hist_data <= 1.5)]

fig = go.Figure(data=[go.Histogram(x=hist_data, nbinsx=30, marker_color='#228be6')])
fig.update_layout(
    title_text=f'Activation Distribution for Neuron {neuron_index}',
    xaxis_title_text='Activation Value',
    yaxis_title_text='Frequency',
    bargap=0.1,
    height=300,
    width=500,
    margin=dict(l=20, r=20, t=40, b=20)
)
```

0.30.40.50.60.70.800.511.522.53神经元 123 的激活分布激活值频率

> 直方图显示了神经元不同激活值的频率。稀疏分布且有高峰可能表明其有特定功能。

一个很少激活但激活强度很大的神经元，可能是在检测特定、不常出现的特征。相反，一个激活分布广泛的神经元，可能参与处理更常见的语言现象。

### 神经元激活与语言属性的关联

一种更高级的分析方式尝试将神经元激活与特定语言属性或观念关联起来。这通常包含：

1. **数据集标注：** 创建或获取输入已标注有相关特征（例如，词性标签、句法分析信息、情感标签、特定语义类别存在与否）的数据集。
2. **关联分析：** 在此标注数据集上运行模型，收集神经元激活，并统计分析激活水平与特定标注存在/不存在之间的关联。例如，神经元在处理动词时，其平均激活是否比处理名词时明显提高？

这种分析可能很复杂，需要精心构建的数据集和统计方法。尽管研究表明LLM中的一些神经元似乎专注于可识别的语言任务（例如，检测句子边界、识别引用、跟踪语法），但将一个单一的、人类可理解的观念归因于单个神经元通常是过于简化的。功能常常分布在多个神经元上，并且单个神经元可能参与多项计算。

### 实用价值与局限

分析神经元激活提供了对模型内部机制有价值的细致视角。它可以补充注意力可视化和探测任务，通过提示FFN层可能提取或响应的*具体*特征。这对以下方面有帮助：

- **调试：** 识别行为异常或导致错误的神经元。
- **理解分工：** 了解模型内部如何分配任务。
- **特征发现：** 可能找到与有趣或意外属性（例如，对偏见的敏感性）相关的神经元。

然而，这种方法有其局限性。解释单个神经元激活模式的“含义”可能很困难且带有主观性。神经元的功能依赖于上下文 (context)，并受网络其他部分的影响。此外，大型语言模型中的计算通常是分布式的，这意味着复杂的观念很少由单个神经元表示。尽管存在这些挑战，神经元激活分析仍然是可解释性工具箱中一个有用的工具，用于理解大型语言模型如何处理信息。

获取即时帮助、个性化解释和交互式代码示例。

---

### 找出失效模式

# 找出失效模式

尽管困惑度等指标以及下游任务表现能提供关于大型语言模型能力的有用信息，但它们无法完整描绘模型的可靠性或潜在不足。平均表现良好的模型仍可能在特定情况下出现问题行为。找出这些“失效模式”，即模型产生不正确、有偏见、不安全或其他不良输出的情况，是理解和改进大型语言模型的重要组成部分。这个过程不止于总体分数，旨在找出具体弱点，从而能够进行有针对性的干预，并构建更值得信赖的系统。

失效模式不仅是学术上的好奇点；它们在部署大型语言模型时带来真实风险。生成事实不准确信息的模型可能会误导用户，而放大偏见的模型则可能延续社会危害。了解这些可能发生的失效，对于调试、改进对齐 (alignment)策略（例如稍后会讨论的SFT和RLHF），以及确保应用得到负责任的开发，都具有重大意义。

### 失效模式的常见类别

大型语言模型失效以多种方式表现出来。认识这些模式有助于设计有效的测试：

1. **事实不准确（幻觉 (hallucination)）：** 这也许是讨论最多的失效。模型生成的文本听起来合理且语法正确，但事实错误或毫无意义。这通常发生在模型缺乏特定知识或试图超出其训练数据范围进行推断时。

   - *示例：* 询问一个近期鲜为人知的科学发现，可能会导致模型编造细节或混淆不同背景的事实。
2. **偏见放大：** 在大量互联网文本数据集上训练的模型不可避免地会学习到数据中存在的社会偏见。它们可能会再现甚至放大与性别、种族、职业或其他特征相关的刻板印象。

   - *示例：* 涉及某些职业的提示可能会总是引发假设特定性别的回应，这反映的是历史偏见而非当前现实。
3. **逻辑不一致和矛盾：** 模型可能会在单个回复中或在对话的多轮中自相矛盾。它也可能无法完成对人类来说似乎微不足道的基本逻辑推理 (inference)任务。

   - *示例：* 在同一段解释中先说“所有鸟类都会飞”，然后又提到“企鹅是不会飞的鸟”。
4. **指令遵循错误：** 特别是在复杂或多部分的提示下，模型可能会忽略限制、误解否定词，或未能遵循要求的格式或角色。

   - *示例：* 要求模型“写一个关于猫的故事，*不要*使用字母‘e’”，但结果故事中却大量出现了字母‘e’。
5. **输入扰动的敏感性：** 对输入提示进行细微的、语义无关的改变（例如，添加一个空格、更换一个同义词、轻微改写）有时会导致截然不同的输出，显露模型的不稳定性。

   - *示例：* “告诉我马来西亚的首都是什么。”可能会得到一个好的答案，而“告诉我马来西亚的首都城市是什么？”则可能会让模型困惑或产生质量较低的回复。
6. **对抗性漏洞：** 模型可能容易受到专门制作的输入的影响，这些输入旨在绕过安全过滤器或引出不正确的输出。这些“对抗性攻击”以非预期的方式利用了已学习到的模式。

   - *示例：* 精心构建的提示（对人类来说有时毫无意义）可能会触发模型生成它通常会拒绝的有害内容。
7. **重复或无意义的输出：** 在某些条件下（例如，非常长的生成上下文 (context)、特定的采样设置或模糊的提示），模型可能会陷入重复循环或退化成不连贯的文本。

### 找出失效模式的方法

找出这些弱点需要使用更具针对性的方法：

#### 针对性测试套件

创建或使用专门设计用于检查已知弱点区域的数据集。这包括制作可能引出特定失效模式的提示。

- **偏见探测：** BBQ（QA偏见基准）或Winogender Schemas等数据集包含旨在显现刻板印象关联的提示。评估模型在这些数据集上的响应可以量化 (quantization)偏见。
- **事实核查：** 使用专注于特定知识类别（科学、历史、近期事件）的问答数据集，其中真实情况已知。将模型输出与事实数据库进行比较。
- **指令遵守测试：** 开发带有复杂限制（否定、格式要求、长度限制）的提示，并评估模型是否遵守。

这是一个PyTorch代码片段，说明了如何检查像生成禁用词这样的简单失效模式：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

def check_forbidden_word(prompt, forbidden_word, max_new_tokens=50):
    """
    给定一个提示，检查模型是否生成了特定的禁用词。
    如果找到禁用词则返回True，否则返回False。
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    generated_portion = generated_text[len(prompt):]
    print(
        f"提示: {prompt}\n生成: {generated_portion[:100]}..."
    )
    return forbidden_word.lower() in generated_portion.lower()

prompt_template = "Describe the following animal: {}"
animal = "penguin"
forbidden = "fly"
prompt = prompt_template.format(animal)

failure_detected = check_forbidden_word(prompt, forbidden)

if failure_detected:
    print(
        f"\n失效已识别: 模型在描述'{animal}'时生成了'{forbidden}'。"
    )
else:
    print(
        f"\n测试通过: 模型在描述'{animal}'时未生成'{forbidden}'。"
    )
```

这个简单示例检查一个特定的关键词，但更复杂的测试将涉及语义分析、检查逻辑一致性，或与事实数据库进行比较。

#### 对抗性测试（红队演练）

这涉及人工测试人员积极尝试让模型失效。红队成员利用他们的创造力以及对模型潜在弱点的理解，制作自动化测试可能遗漏的挑战性提示。他们可能会尝试：

- 规避安全准则。
- 在棘手的主题上诱发幻觉 (hallucination)。
- 通过场景显露偏见。
- 测试指令遵循的极限。

红队演练对于发现意想不到的失效模式以及理解模型能力和安全限制的*边界*来说价值很高。

#### 边缘情况压力测试

在统计上稀有或超出典型使用边界的输入上评估模型：

- **非常长或复杂的提示：** 模型是否保持上下文 (context)和连贯性？
- **包含冲突信息的提示：** 模型如何处理矛盾？
- **领域外请求：** 模型如何优雅地处理远超其训练数据的主题？
- **具有模糊要求的代码生成：** 它能处理复杂的编程逻辑或不熟悉的库吗？

#### 分析异常分布（OOD）行为

系统地测试与模型训练分布明显不同的输入。这可能包括：

- **不同语言或方言**（如果模型主要训练于一种）。
- **高度专业化术语**，来自训练数据中未充分体现的方面。
- **不同文本格式**（例如，表格、结构化数据），如果主要训练于散文。

#### 输出模式分析

有时，失效表现为输出中的统计异常。监控以下情况：

- **高重复率：** 使用N-gram重叠等指标来识别过度重复。
- **低多样性：** 回复是否变得过于通用或模板化？
- **异常的Token概率：** 检查模型为token分配异常高或低概率的序列。

一个简单的重复检查：

```python
from collections import Counter

def calculate_repetition_rate(text, n=3):
    """计算重复N-gram的比例。"""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated_ngrams = sum(1 for count in counts.values() if count > 1)
    return repeated_ngrams / len(ngrams)

rep_rate = calculate_repetition_rate(generated_portion, n=4)

print(f"4-gram重复率: {rep_rate:.2f}")

repetition_threshold = 0.1
if rep_rate > repetition_threshold:
     print("潜在失效: 输出中识别到高重复。")
```

#### 运用可解释性工具

尽管注意力可视化和探测等技术（本章其他部分讨论）主要目的是理解模型*如何*工作，但它们有时能帮助诊断失效*为何*发生。例如，异常的注意力模式或表明对某个特定想法存在困惑的探测结果，可能与相关输入上观察到的失效相关。

找出失效模式并非一次性任务，而是一个持续的过程。随着模型的演进并应用于新方面，需要持续的测试和分析，以了解它们的局限性，并确保它们安全有效地使用。从失效分析中获得的认识直接指导模型改进、数据管理策略以及更好的对齐 (alignment)技术的开发。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 24 Identifying Mitigating Training Instabilities

### 不稳定性常见表现

# 不稳定性常见表现

识别训练早期不稳定性表现对于避免计算资源浪费，并确保大型语言模型成功开发是基础。虽然监控将在下一节讨论，但了解这些常见*症状*能帮助你快速判断训练过程何时偏离了正常轨道。这些症状通常体现在训练过程中你追踪的主要指标上，尤其是损失函数 (loss function)和梯度统计数据。

### NaN 损失

最明确的灾难性故障迹象可能就是损失计算中出现 `NaN` (非数字) 值。

损失=NaN\text{损失} = \text{NaN}损失=NaN

损失为 `NaN` 通常会立即停止训练过程，因为后续的梯度计算和权重 (weight)更新在数学上变得未定义。这通常表明存在严重的数值问题，例如：

1. **数值溢出：** 操作结果超出了当前浮点格式（例如 FP16、BF16，甚至在极端条件下的 FP32）所能表示的最大值。这可能发生在矩阵乘法或激活函数 (activation function)内部。
2. **数值下溢：** 操作结果小于（更接近零）最小可表示正值。虽然直接导致 NaN 的可能性较小，但在中间步骤的下溢有时会导致除以零等问题情况。
3. **无效操作：** 诸如计算零或负数的对数 (`log(0)`)、计算负数的平方根或除以零等数学操作。这些可能由于特定数据点或不稳定的中间激活值而发生。

在训练循环早期检测 `NaN` 很重要。你可以在损失计算后直接添加检查。

```python

outputs = model(inputs)
loss = loss_function(outputs, targets)

if not torch.isfinite(loss):
    print(f"检测到不稳定的损失: {loss.item()}。停止训练。")

    break

loss.backward()
```

### 损失尖峰

与 `NaN` 损失相比，它并非立即致命，但仍然是一个严重的警告信号，即损失值突然急剧增加，通常被称为“损失尖峰”。损失可能会在后续步骤中部分或完全恢复，或者该尖峰可能预示着完全的发散。

1002003004005006007001.522.533.5

> 典型的损失曲线，显示在第500次迭代附近出现突然尖峰，之后部分恢复。

损失尖峰可能由以下几个因素引起：

1. **有问题的数据批次：** 单个批次包含损坏数据、异常值或与数据集中其余部分显著不同的样本，可能导致模型产生高度不正确的预测，从而在该步骤产生很大的损失值。
2. **学习率：** 过高的学习率可能导致优化器越过最优值，导致暂时性不稳定。
3. **梯度问题：** 即使没有变成 `NaN`，梯度也可能暂时性地爆炸，导致大的、破坏稳定性的权重 (weight)更新。这在使用 Adam 等自适应优化器时尤其相关。

虽然单个、孤立的尖峰可能不会完全破坏训练，但频繁的尖峰表明存在需要解决的潜在不稳定性。

### 发散损失

与临时尖峰不同，发散损失是指损失值在多次迭代或多个周期中持续呈上升趋势。这表明模型的性能持续下降，优化过程正在远离而不是趋向于一个好的解决方案。

100200300400500600700123456收敛损失发散损失

> 健康的收敛损失曲线与表示训练失败的发散损失曲线的比较。

发散通常指向更根本的问题：

1. **学习率过高：** 最常见的原因。优化器持续越过最小值点。
2. **梯度问题：** 梯度计算或缩放中存在持续性问题。
3. **初始化：** 糟糕的权重 (weight)初始化可能使模型处于梯度持续“上坡”的区域。
4. **架构缺陷：** 模型组件或归一化 (normalization)层实现不正确。
5. **数据问题：** 训练数据集中存在的系统性问题。

### 波动损失或指标

另一个症状是当损失值或验证集上的困惑度或准确率等其他评估指标显著波动，而没有显示出明确的改进趋势时。这些值可能在步骤或周期之间上下跳动，从未稳定地下降（对于损失）或上升（对于准确率）。

这种波动通常表明：

1. **学习率过高：** 学习率可能过高，导致优化器重复跨越一个“谷底”，而不是稳定在底部。
2. **批次间差异：** 数据批次的构成或难度差异大，可能导致性能指标波动。
3. **正则化 (regularization)不足：** 缺乏适当的正则化可能使模型拟合单个批次中的噪声，导致性能不稳定。

### 潜在梯度问题：爆炸与消失

虽然不总是直接表现为 NaN 损失或尖峰等主要症状，但极端的梯度大小通常是根本原因。

- **梯度爆炸：** 发生在梯度的大小 (∣∣ablaL∣∣||
  abla L||∣∣ablaL∣∣) 变得过大时。这会导致大的权重 (weight)更新，可能破坏网络稳定性，通常表现为 `NaN` 或损失尖峰。监控梯度范数对于检测这一点很重要。
- **梯度消失：** 发生在梯度变得极其小，接近零时。这会减慢或停止学习，尤其是在更深层中，因为权重更新变得微不足道。虽然它主要导致停滞而不是突然不稳定，但网络某些部分的严重消失可能间接导致数值问题。

识别这些常见症状是重要的第一步。后续章节将提供关于如何有效监控训练指标的指导，以便及早发现这些迹象并诊断不稳定性或其根本原因。

获取即时帮助、个性化解释和交互式代码示例。

---

### 监控训练指标（损失、梯度范数）

# 监控训练指标（损失、梯度范数）

有效的监控是应对可能破坏大规模语言模型训练稳定性的第一道防线。仅仅启动持续数天或数周的训练任务并寄希望于最好的结果是不现实的。相反，工程师需要通过重要的度量指标持续观察模型的行为。这有助于尽早发现问题，从而在大量计算资源被浪费之前进行及时干预。诊断训练状态最有用的两个度量指标是训练损失和梯度范数。

### 跟踪训练损失

训练损失量化 (quantization)了模型在任何给定时刻对训练数据的表现。对于语言模型而言，这通常是交叉熵损失等度量，它反映了模型在预测下一个词元 (token)时的不确定性。

**预期表现：** 在正常的训练过程中，损失通常会随时间下降，在早期阶段迅速下降，然后随着训练进行而变慢，最终在模型收敛时趋于平稳。轻微的波动是正常的，但总体趋势应是下降的。

**不稳定迹象：**

- **损失骤升：** 损失值突然急剧增加。这通常表明模型遇到了有问题的数据批次，或者学习率过高，导致更新不稳定。
- **NaN 或 Inf 损失：** 损失值变为 `NaN`（非数值）或无穷大。这是一个严重故障，通常由数值溢出（例如除以零、对零或负数取对数）或梯度爆炸引起。一旦出现 NaN，训练就无法继续。
- **发散：** 损失随时间持续增加，表明模型的预测越来越差，而不是变好。
- **震荡：** 损失剧烈波动，没有明确的下降趋势，可能表明学习率、优化算法稳定性或数据质量存在问题。

**实现：** 在标准训练循环中记录损失是简单直接的。通常在前向传播之后计算损失，并定期记录其值。

```python
import torch
import torch.nn as nn

def log_metric(step, metric_name, value):

    print(f"步骤 {step}: {metric_name} = {value:.4f}")

global_step = 0
for batch in data_loader:
    optimizer.zero_grad()

    inputs = batch['input_ids'].to('cuda')
    targets = batch['labels'].to('cuda')

    outputs = model(inputs)

    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(outputs.view(-1, model.config.vocab_size), targets.view(-1))

    if global_step % 10 == 0:
        log_metric(global_step, "train_loss", loss.item())

    loss.backward()
    optimizer.step()
    global_step += 1

    if torch.isnan(loss):
        print(f"在步骤 {global_step} 检测到 NaN 损失！停止训练。")
        break
```

随时间可视化损失曲线很重要。诸如 TensorBoard 或 Weights & Biases 等工具使这变得容易。

0100200300400500468101214稳定骤升发散

> 示例损失曲线显示了稳定的训练、突然的损失骤升以及损失随时间增加的发散情况。请注意对数 Y 轴，这在查看损失时很常见。

### 监控梯度范数

梯度是指示损失函数 (loss function)最陡峭上升方向和大小的向量 (vector)。该向量的*范数*（通常是 L2 范数，∣∣ablaL∣∣2||
abla L||\_2∣∣ablaL∣∣2​）度量其大小。监控梯度范数提供了关于应用于模型权重 (weight)更新尺度的见解。

∣∣∇L∣∣2=∑p∈参数∣∣∇pL∣∣22||\nabla L||\_2 = \sqrt{\sum\_{p \in \text{参数}} ||\nabla\_p L||\_2^2}∣∣∇L∣∣2​=p∈参数∑​∣∣∇p​L∣∣22​​

ablapL
abla\_p Lablap​L 是损失 LLL 相对于特定参数 (parameter)张量 ppp 的梯度。

**为何它重要：**

- **梯度爆炸：** 过大的梯度范数表明权重更新将非常大，可能导致数值溢出、损失骤升和发散。这是训练不稳定的常见原因，尤其是在深度网络中。
- **梯度消失：** 非常小或为零的梯度范数表明更新极小或不存在，导致训练停滞。尽管与旧的 RNN 相比，这在 Transformer 中作为灾难性故障较不常见（归因于残差连接和归一化 (normalization)），但它仍然会阻碍收敛。
- **学习率的合适性：** 梯度范数与学习率有关联。持续较高的范数可能表明学习率需要降低，而持续微小的范数（在某些情况下）可能需要增加，尽管梯度消失更多是架构或初始化问题。

**预期表现：** 梯度范数通常开始时较高，并随着模型收敛和损失在最小值附近趋于平坦而下降。然而，其行为高度依赖于学习率调度、优化器和数据。可能会出现显著波动，但极端大的值是一个危险信号。

**不稳定迹象：**

- **突然的大幅骤升：** 比平时大几个数量级的值通常在 NaN 损失或显著损失骤升之前发生。这是梯度爆炸的显著特征。
- **持续接近零：** 如果梯度范数在训练早期变得非常小并保持如此，可能表明存在梯度消失问题或初始化问题。

**实现：** 计算总梯度范数需要在调用 `loss.backward()` 之后但在调用 `optimizer.step()` *之前*遍历所有模型参数。梯度裁剪，一种稍后讨论的技术，通常无论如何都会涉及计算这个范数。

```python
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

global_step = 0

max_grad_norm = 1.0

for batch in data_loader:
    optimizer.zero_grad()

    inputs = batch['input_ids'].to('cuda')
    targets = batch['labels'].to('cuda')
    outputs = model(inputs)
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(
        outputs.view(-1, model.config.vocab_size),
        targets.view(-1)
    )

    if global_step % 10 == 0:
        log_metric(global_step, "train_loss", loss.item())

    if torch.isnan(loss):
        print(f"在步骤 {global_step} 检测到 NaN 损失！停止训练。")
        break

    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if len(grads) > 0:

      total_norm = torch.norm(
          torch.stack([
              torch.norm(g.detach(), 2.0) for g in grads
          ]),
          2.0
      )

      if global_step % 10 == 0:
          log_metric(global_step, "grad_norm", total_norm.item())

      if total_norm > 100 * max_grad_norm:
            print(
                f"警告：在步骤 {global_step} 检测到高梯度范数（{total_norm:.2f}）"
            )
    else:

        if global_step % 10 == 0:
            log_metric(global_step, "grad_norm", 0.0)

    optimizer.step()
    global_step += 1
```

可视化梯度范数以及损失，提供了关于训练动态的更完整情况。

0100200300400500050100150200稳定爆炸

> 示例梯度 L2 范数曲线显示了稳定的下降和范数急剧增加的梯度爆炸情况。

通过仔细监控训练损失和梯度范数，您能对训练过程获得必要的了解。这些度量指标充当早期预警系统，使您能够在潜在不稳定情况升级为严重故障之前发现它们，从而节省宝贵的时间和计算资源。这些曲线中的异常是诊断具体根本问题的起点，我们将在接下来的内容中查看。

获取即时帮助、个性化解释和交互式代码示例。

---

### 诊断损失飙升

# 诊断损失飙升

损失飙升是大型模型训练中最令人沮丧的情况之一。一个顺利运行了数天或数周的训练，可能会突然出现损失值急剧（通常是垂直）增加的情况，有时甚至崩溃为 `NaN`（非数值）或 `inf`（无穷大）。这通常会中止有效的训练。要找出根本原因，需要进行系统性排查，因为有多种因素可能引发此类事件。

当面对损失飙升时，首要目标是了解它 *何时* 以及 *为何* 发生。监控工具（如TensorBoard、Weights & Biases）在此处不可或缺，它们能帮助你精准定位到发生飙升的具体训练步骤。

0100020003000400050004567891023456

> 上图展示了一个典型的损失飙升，损失值突然增加，随后可能恢复或完全偏离。请注意，损失可视化常使用对数刻度。

以下是常见原因及其排查方式：

### 1. 检查有问题的数据批次

通常，触发因素是单个“异常”批次数据。这可能包括：

- **损坏数据：** 未正确处理的文件，含有无意义字符、过长序列或格式错误。
- **数值异常值：** 当数据点通过模型初始层（如嵌入 (embedding)层）处理时，产生异常大的激活值，可能随后导致溢出。
- **异常输入格式：** 数据不符合预期的分词 (tokenization)或填充方案，可能是由于预处理中的某个边界情况。

**诊断步骤：**

- **隔离批次：** 如果你的训练框架允许，记录紧接在损失飙升之前批次中包含的样本的索引或标识符。
- **手动检查：** 加载并检查识别出的特定数据样本。查找异常长的序列、重复模式、原始数据中的 `NaN` 值（如果适用），或分词器 (tokenizer)词汇表 (vocabulary)外的可能处理不当的字符。
- **单批次复现：** 尝试仅使用怀疑有问题批次进行单次前向和反向传播 (backpropagation)。这可以确认该批次本身是否单独触发了问题。

```python

import torch

def check_tensor_health(tensor: torch.Tensor, name: str):
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    if has_nan or has_inf:
        print(f"Problem detected in tensor '{name}':")
        if has_nan:
            print(f"  - Contains NaN values!")
            print(f"  - NaN count: {torch.isnan(tensor).sum().item()}")
        if has_inf:
            print(f"  - Contains Inf values!")
            print(f"  - Inf count: {torch.isinf(tensor).sum().item()}")

        return False
    return True
```

### 2. 分析梯度和激活值

损失飙升在机制上常由梯度爆炸引起。即使输入数据看起来正常，模型内部的操作也可能导致过大的数值。

**诊断步骤：**

- **检查 `NaN`/`inf`：** 在优化器步骤 (`optimizer.step()`) 之前，检查损失张量本身以及模型参数 (parameter)的梯度中是否含有 `NaN` 或 `inf` 值。 `NaN` 损失是上游数值不稳定的明确信号。 `NaN` 梯度会在优化器步骤时损坏权重 (weight)。
- **监控梯度范数：** 如“监控训练指标”章节所述，追踪总梯度范数（例如，所有梯度拼接后的L2范数）。梯度范数的突然大幅增长常常在损失飙升之前发生或同时发生。这表明参数更新即将变得过大。 `torch.nn.utils.clip_grad_norm_` 等工具不仅进行裁剪，还会返回裁剪 *前* 的范数，这对于记录有帮助。
- **激活值统计：** 高级调试可能涉及在模型中设置钩子，以记录不同层中激活值的统计数据（最小值、最大值、均值、标准差）。激活值中的极端数值可能是梯度问题的先兆。

```python

total_norm = 0.0
nan_or_inf_found = False
for p in model.parameters():
    if p.grad is not None:
        if not check_tensor_health(
            p.grad,
            f"Gradient of {p.name if hasattr(p, 'name') else 'parameter'}"
        ):
             nan_or_inf_found = True

        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2

total_norm = total_norm ** 0.5

print(f"Step {global_step}: Total Gradient Norm: {total_norm}")

if nan_or_inf_found:
    print(
        f"Step {global_step}: NaN or Inf detected in gradients "
        f"BEFORE optimizer step. Skipping update."
    )

elif total_norm > gradient_clipping_threshold * 10:

     print(
         f"Warning: Step {global_step}: Gradient norm ({total_norm}) "
         f"significantly exceeds clipping threshold "
         f"({gradient_clipping_threshold}). Potential instability."
     )
```

### 3. 检查学习率和优化器状态

尽管在稳定训练后不太可能导致 *单次* 突然飙升（更常导致数个步骤内的偏离），但学习率可能会起到作用。

**诊断步骤：**

- **检查当前学习率：** 记录每一步的学习率值。调度器是否表现异常？是否存在导致学习率重置或跳变的错误？
- **优化器状态：** 极其罕见地，自适应优化器（如Adam的动量或方差估计）的内部状态可能会损坏（`NaN`/`inf`）。框架通常会处理这种情况，但在非常罕见的情况下，检查 `optimizer.state_dict()` 可能会发现问题。

### 4. 调查混合精度问题

使用 `FP16` (16位浮点) 训练特别容易出现数值范围问题。尽管 `BF16` (bfloat16) 提供更宽的范围，极端值仍可能导致问题。

**诊断步骤：**

- **损失缩放：** 如果使用 `FP16` 和自动混合精度 (AMP)，请确保损失缩放处于启用状态。如果梯度在被损失缩放器反向缩放 *之前* 变得过大（> 65504，FP16的最大值），则可能发生损失飙升。检查损失缩放值本身是否变为 `NaN` 或零，这可能发生在梯度在溢出前反复下溢的情况下。
- **在 `FP32` 中进行的操作：** 某些操作可能对数值敏感，显式转换为 `FP32` 会有益。检查是否有任何自定义操作或数值不稳定的函数（如对极端值进行某些归约或归一化 (normalization)）以较低精度执行。
- **尝试 `BF16`：** 如果硬件支持，`BF16` 通常比 `FP16` 更稳定，因为它具有更宽的动态范围，通常无需损失缩放。在使用FP16时遇到飙升可能促使切换到BF16。

诊断损失飙升是一个迭代过程。通过系统性地检查数据、监控梯度和激活值、审视优化器配置以及考虑混合精度训练的细节，你通常可以找到不稳定性的来源，并应用本章其他地方讨论的相应缓解技术，例如调整学习率、改进数据清洗或优化梯度裁剪和损失缩放策略。

获取即时帮助、个性化解释和交互式代码示例。

---

### 调试数值精度问题

# 调试数值精度问题

混合精度训练，使用 FP16FP16FP16 (半精度) 或 BF16BF16BF16 (bfloat16) 等较低精度格式，是加快LLM训练和减少内存占用的一种常用方法。然而，这些格式的可表示范围比默认的 FP32FP32FP32 (单精度) 明显更窄。缩小的范围可能导致数值不稳定，表现为激活值、梯度或损失本身中出现 `NaN` (非数字) 或 `Inf` (无穷大) 值，从而使训练过程脱轨。调试这些问题需要仔细检查并弄明白精度限制可能在何处引发问题。

### 理解精度限制

回顾第20章，FP16FP16FP16 的动态范围有限。非常小的数字（比如小梯度）可能变成零（下溢），从而有效阻止这些参数 (parameter)的学习。反之，大数字（激活值或梯度）可能超出最大可表示值，变成 `Inf`（上溢）。 `NaN` 值通常来自数学上未定义的操作，如 0/00/00/0、−1\sqrt{-1}−1​ 或 ∞−∞\infty - \infty∞−∞，当中间计算涉及到 `Inf` 时就可能出现。

BF16BF16BF16 在设计时考虑了深度学习 (deep learning)，提供与 FP32FP32FP32 相同的动态范围，但精度降低（尾数位数更少）。这相比 FP16FP16FP16 大大缓解了上溢问题，通常无需使用损失缩放等方法。然而，其较低的精度有时仍可能在对微小数值差异敏感的操作中导致问题，尽管这比 FP16FP16FP16 的下溢/上溢情况少见。

### 确定不稳定的出现

数值精度问题通常会突然出现。正在顺利进行的训练可能会突然遇到 `NaN` 损失或梯度。重要迹象包括：

1. **NaN/Inf 损失：** 最明显的迹象。如果损失变为 `NaN` 或 `Inf`，反向传播 (backpropagation)就无法正确进行。
2. **NaN/Inf 梯度：** 即使损失保持有限，某些参数 (parameter)的梯度也可能变为 `NaN` 或 `Inf`。这会阻止优化器更新这些参数，如果不处理，可能会很快损坏模型。
3. **梯度范数爆炸：** 全局梯度范数（如前一节所述）突然出现的大幅飙升通常在 `NaN`/`Inf` 值之前出现或同时发生，表明发生了上溢。
4. **训练停滞 / 零梯度：** 如果训练意外停滞，特别是在使用 FP16FP16FP16 且没有有效损失缩放的情况下，这可能表明梯度下溢，即梯度变得太小以至于无法表示并被清零。

### 找出问题来源的方法

一旦你怀疑存在数值精度问题，目标是隔离不稳定性出现的具体操作或模块。一个层中生成的 `NaN` 或 `Inf` 可以迅速传播到后续计算中。

#### 暂时切换到FP32

最简单的第一步通常是禁用混合精度训练，并完全在 FP32FP32FP32 中运行。如果 instabilities 消失，则强烈表明是低精度格式的问题。如果 instability 在 FP32FP32FP32 中仍然存在，根本原因可能是其他问题，例如模型代码中的错误、不良数据或不合适的超参数 (parameter) (hyperparameter)（例如，过高的学习率）。

#### 前向和后向钩子

PyTorch 的钩子机制提供了一种有效方法，可以在前向传播期间检查中间张量（激活值），在后向传播期间检查梯度，而不根本改变模型结构。你可以在特定的模块或张量上注册钩子，以便在它们计算后立即检查 `NaN`/`Inf` 值。

这是注册前向钩子以检查特定线性层后激活值中是否存在 `NaN` 或 `Inf` 值的示例：

```python
import torch
import torch.nn as nn

def check_nan_inf_hook(module, input, output):
    """前向钩子，检查模块输出中是否存在NaNs/Infs。"""
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"在模块 {module} 的输出中检测到NaN/Inf")

    elif isinstance(output, tuple):
        for i, out in enumerate(output):
             if isinstance(out, torch.Tensor):
                if torch.isnan(out).any() or torch.isinf(out).any():
                    print(
                        f"在模块 {module} 的输出元组元素 {i} 中检测到NaN/Inf "
                        f"of module: {module}"
                    )

target_layer = model.transformer.h[5].mlp.c_fc
handle = target_layer.register_forward_hook(check_nan_inf_hook)

handle.remove()
```

同样，你可以注册后向钩子（模块使用 `register_full_backward_hook`，特定张量使用 `register_hook`）来检查梯度（`grad_input`、`grad_output`）。通过策略性地放置这些钩子，你可以缩小不稳定性首次出现的计算步骤范围。

#### 直接检查梯度

在 `loss.backward()` 之后，你可以遍历模型参数并检查它们的 `.grad` 属性：

```python
import torch

def check_gradients(model):
    """检查所有模型参数是否存在NaN/Inf梯度。"""
    nan_inf_found = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"在参数 {name} 的梯度中检测到NaN")
                nan_inf_found = True
            if torch.isinf(param.grad).any():
                print(f"在参数 {name} 的梯度中检测到Inf")
                nan_inf_found = True
    if not nan_inf_found:
        print("未检测到NaN/Inf梯度。")
    return nan_inf_found
```

#### 检查损失缩放器状态 (FP16)

如果使用 FP16FP16FP16 进行动态损失缩放（例如通过 PyTorch 的 `torch.cuda.amp.GradScaler`），缩放器本身可能会提供线索。反向传播 (backpropagation)过程中上溢的梯度将导致缩放器跳过优化器步骤并减小后续迭代的损失缩放。如果这种情况反复发生，损失缩放可能变得极小甚至为零，导致梯度下溢。反之，如果损失缩放顺利增长到很大，则可能会增加后续中间上溢的可能性。

你可以检查 `GradScaler` 当前的缩放因子：

```python
import torch

current_loss_scale = scaler.get_scale()
print(f"当前损失缩放：{current_loss_scale}")
```

长时间监控 `current_loss_scale` 可以发现有问题的情况。反复崩溃的缩放表明存在持续的上溢问题。降到极低值的缩放可能预示着随后的下溢问题。

#### 数值不稳定操作

某些数学运算本身更容易产生 `NaN` 或 `Inf`，尤其是在精度有限的情况下：

- **零或负数的对数：** `torch.log(x)`，其中 x≤0x \le 0x≤0。这通常发生在概率或 softmax 输出可能在数值上评估为零时。使用 `torch.log_softmax` 通常比 `torch.log(torch.softmax(x))` 更稳定。
- **负数的平方根：** `torch.sqrt(x)`，其中 x<0x < 0x<0。
- **除以零：** 确保分母非零，可以通过在适当的地方添加一个小的 epsilon ϵ\epsilonϵ（例如，在 RMSNorm 或 LayerNorm 等归一化 (normalization)层中，如果方差接近零）来实现。
- **大指数：** `torch.exp(x)` 对于大的 xxx 值很容易上溢。

在调试时，要密切关注涉及这些操作的计算，尤其是在自定义层或损失函数 (loss function)中。在平方根或分母内部添加小的 epsilon 值（1e−81e-81e−8 到 1e−61e-61e−6）有时可以防止由接近零的中间值引起的 `NaN`，但请注意这会轻微改变计算。

### 缓解策略回顾

调试通常涉及应用或调整之前讨论的稳定化方法：

- **梯度裁剪：** 对于防止导致 `Inf` 的梯度爆炸非常重要。如果发生不稳定性，请考虑裁剪阈值是否合适。
- **损失缩放 (FP16)：** 对于 FP16FP16FP16 防止下溢是必须的。确保动态缩放器正常运行且缩放没有崩溃。如果使用静态缩放，找到合适的缩放值很重要。
- **切换到 BF16：** 如果 FP16FP16FP16 持续出现问题，特别是上溢，切换到 BF16BF16BF16（如果硬件支持）通常是最有效的解决方案，通常无需损失缩放。
- **数值稳定性：** 用更稳定的等效方法替换可能不稳定的操作序列（例如，`log_softmax`）。在需要的地方谨慎添加 epsilon 值。
- **超参数 (parameter) (hyperparameter)调整：** 过于激进的学习率会加剧数值问题。将学习率调整与预热和衰减策略（第17章）结合使用是标准做法。
- **初始化：** 尽管在稳定训练期间作为 `NaN` 的直接原因不常见，但糟糕的初始化（第12章）可能导致早期激活值过大，从而可能增加上溢风险。

调试大规模训练中的数值精度问题需要耐心和系统的检查。通过监控重要指标，运用钩子等工具，以及了解低精度格式的限制，你可以有效地诊断并解决这些不稳定性，保持你的长时间训练顺利进行。

获取即时帮助、个性化解释和交互式代码示例。

---

### 稳定方法回顾（梯度裁剪、学习率、预热）

# 稳定方法回顾（梯度裁剪、学习率、预热）

在训练大型语言模型时，损失值突然飙升或发散（出现`NaN`值）等不稳定情况遗憾的是很常见。某些技术对于有效训练以及专门预防和减轻这些问题来说都非常重要。三种重要的稳定方法——梯度裁剪、学习率调度和学习率预热——对于解决不稳定问题至关重要。当训练运行开始出现异常时，正确配置和监控这些组成部分通常是第一道防线。

### 梯度裁剪

梯度裁剪直接解决梯度爆炸问题，这是导致`NaN`损失值或突然发散的常见原因。通过对梯度的大小施加一个上限，它能避免模型权重 (weight)更新过大，而过大的更新会使训练过程不稳定，特别是在深度网络或Transformer中固有的循环结构中。

最常用的技术是对所有模型参数 (parameter)的梯度*范数*进行裁剪。如果整个模型（或有时按参数组）的梯度L2范数（欧几里得范数）超过预设阈值`max_norm`，梯度会按比例缩小以匹配此阈值。

g←max\_norm∣∣g∣∣2gif ∣∣g∣∣2>max\_norm\mathbf{g} \leftarrow \frac{\text{max\\_norm}}{||\mathbf{g}||\_2} \mathbf{g} \quad \text{if } ||\mathbf{g}||\_2 > \text{max\\_norm}g←∣∣g∣∣2​max\_norm​gif ∣∣g∣∣2​>max\_norm

这里，g\mathbf{g}g 表示所有梯度的向量 (vector)。

在PyTorch中，这通常在反向传播 (backpropagation)之后，但在优化器步进之前应用：

```python
import torch
from torch.nn.utils import clip_grad_norm_

loss = compute_loss(outputs, targets)

loss.backward()

max_norm = 1.0
clip_grad_norm_(model.parameters(), max_norm)

optimizer.step()
optimizer.zero_grad()
```

**选择`max_norm`**：`max_norm`的值是一个超参数 (hyperparameter)。常用值范围是0.5到10.0，1.0是一个常见的默认值。设置过低会因更新收缩过多而阻碍学习，而设置过高则无法有效对抗大幅飙升。在稳定运行时监控梯度范数（如“监控训练指标”部分所述）可以提供选择合适值的依据。如果在稳定期间观察到范数频繁超过您选择的`max_norm`，您可能裁剪得过于激进。反之，如果发生损失飙升且此时范数远高于您的`max_norm`，裁剪很可能起到帮助作用。

尽管有效，但梯度裁剪不应被视为万能药。它是一种稳定机制。如果裁剪持续活跃或在非常低的阈值下才起作用，那可能表明模型架构、初始化、学习率或数据质量存在潜在问题，也应检查这些问题。

### 学习率调度

固定学习率很少适合训练大型复杂模型。学习率调度能在训练过程中动态调整学习率，平衡早期的探索和后期的精细收敛。这种受控调整对稳定性很重要。整个训练过程中学习率过高很容易导致震荡或发散，而衰减过慢则可能在接近最小值时仍进行大步、可能不稳定的更新。

大型语言模型常用的调度方案，通常在初始预热阶段*之后*应用，包括：

1. **线性衰减**：学习率在设定的步数或整个训练持续时间内，从基础值线性降低到一个最小值（通常为零）。
2. **余弦衰减**：学习率遵循余弦曲线，从基础值降低到一个最小值。与线性衰减相比，这通常导致初始衰减较慢，然后接近结束时衰减加快。

以下是您如何使用PyTorch的调度器结合预热和余弦衰减：

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR
)

optimizer = AdamW(model.parameters(), lr=1e-4)

warmup_steps = 500
total_training_steps = 10000
decay_steps = total_training_steps - warmup_steps

warmup_scheduler = LinearLR(
    optimizer,
    start_factor=1.0/warmup_steps,
    end_factor=1.0,
    total_iters=warmup_steps
)

decay_scheduler = CosineAnnealingLR(
    optimizer, T_max=decay_steps, eta_min=0
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, decay_scheduler],
    milestones=[warmup_steps]
)

optimizer.step()
scheduler.step()
optimizer.zero_grad()
```

配置不当的调度方案可能导致不稳定。例如，如果衰减过慢或最终学习率（`CosineAnnealingLR`中的`eta_min`）过高，优化器可能在训练结束时超调或震荡。反之，衰减过快可能导致收敛停滞。当遇到后期不稳定时，检查学习率调度的行为是一个有益的诊断步骤。

### 学习率预热

训练的最初阶段是一个特别脆弱的时期。模型权重 (weight)通常是随机初始化的，立即应用完整的目标学习率可能导致非常大、混乱的更新，在最初几次迭代中就出现损失值飙升或`NaN`值。像AdamW这样的自适应优化器也需要一些初始步骤来建立梯度矩的可靠估计。学习率预热通过从一个非常小的学习率（通常为零或接近零）开始，并在预设的初始训练步数（`warmup_steps`）内逐步增加到目标基础学习率来解决这个问题。这给模型和优化器时间在应用较大更新前稳定下来。线性预热是最常见的策略。

02k4k6k8k10k0.0e+02.0e−54.0e−56.0e−58.0e−51.0e−4学习率调度：预热 + 余弦衰减

> 一个典型的学习率调度方案，包括500步线性预热，随后在剩余步数中余弦衰减到零。

**选择`warmup_steps`**：预热步数是另一个超参数 (parameter) (hyperparameter)。它通常范围从数百到数千步，取决于总体训练时长、批量大小和数据集。常见做法是将其设置为总预期训练步数的一小部分（例如1-10%）。过少的预热步数可能无法避免初始不稳定，而过多则会减慢有效学习的开始。如果在训练初期就遇到不稳定，增加预热时长或降低初始学习率是首先要考虑的调整。

### 关联与监控

这三种技术彼此联系。例如：

- 有效的预热可能允许稍高的峰值学习率而不会立即发散。
- 激进的梯度裁剪可能使训练过程对次优的学习率调度方案更具容忍度。
- 优化器的选择（例如AdamW与SGD）也会影响对学习率的敏感度和预热的有效性。

诊断不稳定时，监控与这些技术相关的行为很重要：

- **学习率**：记录优化器在每一步实际使用的学习率，以确认调度方案按预期运行。
- **梯度范数**：跟踪梯度范数（裁剪前）以了解更新的幅度以及裁剪是否频繁活跃。梯度范数突然、持续增加通常先于损失值飙升发生。

再次审视并仔细调整梯度裁剪、学习率调度和预热阶段，是解决训练不稳定问题的必要步骤。它们提供了控制更新动态的手段，防止可能阻碍大规模语言模型训练进展的爆炸和发散情况。

获取即时帮助、个性化解释和交互式代码示例。

---

### 架构选择对稳定性的影响

# 架构选择对稳定性的影响

尽管细致的监控和调试技术对于应对大型模型训练中的波动必不可少，但Transformer模型的架构本身对其固有的稳定性起着重要的作用。早期做出的设计选择，可以使模型更易于顺利训练，也可以在模型深度和规模增加时，导致损失值骤增等不稳定现象更容易出现。了解这些架构影响，能让你做出明智的决定，从而促进更可靠的收敛。

### 归一化 (normalization)层的位置：预归一化（Pre-LN）与后归一化（Post-LN）

最具争议且影响较大的架构变体之一，是Transformer块中层归一化（`LayerNorm`）层相对于残差连接的放置位置。

- **后归一化（Post-LN，原版Transformer）：** 在原始论文“Attention Is All You Need”描述的架构中，层归一化在残差连接将子层（自注意力 (self-attention)或前馈网络）的输入和输出求和后应用。
  输出=LayerNorm(x+SubLayer(x))\text{输出} = \text{LayerNorm}(x + \text{SubLayer}(x))输出=LayerNorm(x+SubLayer(x))
- **预归一化（Pre-LN）：** 另一种做法是在子层之前应用层归一化，直接作用于残差分支的输入。
  输出=x+SubLayer(LayerNorm(x))\text{输出} = x + \text{SubLayer}(\text{LayerNorm}(x))输出=x+SubLayer(LayerNorm(x))

G

clusterₚost

后归一化块

clusterₚre

预归一化块

xinₚost
x

addₚost

+

xinₚost:e->addₚost:w

恒等

subₚost

子层
(注意力/FFN)

xinₚost->subₚost

normₚost

层归一化

addₚost->normₚost

subₚost:s->addₚost:n

outₚost
输出

normₚost->outₚost

xinₚre
x

normₚre

层归一化

xinₚre->normₚre

addₚre

+

xinₚre:e->addₚre:w

恒等

subₚre

子层
(注意力/FFN)

normₚre->subₚre

subₚre:s->addₚre:n

outₚre
输出

addₚre->outₚre

> 后归一化与预归一化块结构的比较。

主要区别在于反向传播 (backpropagation)时梯度如何流动。在后归一化架构中，通过残差连接回流的梯度不会经过该块相关的层归一化操作。随着模型变深，这可能导致梯度爆炸，因为残差块的输出量级可能逐层无限制地增长。预归一化通过在输入进入子层变换前进行归一化来解决这个问题。这通常会产生更稳定的梯度，并允许训练更深的网络，对学习率设置不那么敏感，并且可能缩短热身期。尽管后归一化在训练成功时有时能获得略好的表现，但由于其更好的稳定性特点，预归一化通常被认为是大型模型更稳定的选择。

### 激活函数 (activation function)：ReLU之后

前馈网络（FFN）层中激活函数的选择也影响训练动态。尽管ReLU在早期深度学习 (deep learning)模型中是标准选择，但现代Transformer通常使用更平滑的激活函数：

- **GeLU（高斯误差线性单元）：** GeLU根据输入的值来加权，而不是像ReLU那样仅通过符号来门控。它提供了一个更平滑、非单调的激活曲线。
- **SwiGLU（Swish门控线性单元）：** SwiGLU的变体将Swish（一种平滑的自门控激活）与门控机制结合，通常会拆分FFN的中间维度，对一部分应用Swish并将其乘以另一部分。这会引入更多参数 (parameter)，但通常会带来更好的性能和稳定性。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFFN(nn.Module):
    def __init__(self, d_model, d_ff, activation_type='gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'gelu':
            self.activation = nn.GELU()

        elif activation_type == 'swish_like':
            self.activation = nn.SiLU()
        else:
            raise ValueError("Unsupported activation type")

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

d_model = 512
d_ff = 2048
ffn_gelu = SimpleFFN(d_model, d_ff, activation_type='gelu')
input_tensor = torch.randn(32, 128, d_model)
output = ffn_gelu(input_tensor)
print("Output shape:", output.shape)
```

像GeLU和SwiGLU这样更平滑的激活函数通常会带来更平滑的损失曲面和更稳定的梯度流，特别是与ReLU相比，在非常深的神经网络 (neural network)中更是如此。SwiGLU中的门控机制可能进一步帮助调节信息流并防止激活值爆炸。尽管其具体影响可能不那么明显，选择一个现代激活函数通常是提升整体训练稳定性的一个因素。

### 初始化策略的相互作用

如第12章所述，适当的权重 (weight)初始化非常重要。然而，架构选择会改变初始化操作的*环境*。

- **预归一化 (normalization)（Pre-LN）与后归一化（Post-LN）：** 与后归一化相比，预归一化架构对初始化尺度通常不那么敏感。因为在预归一化中，每个子层的输入都被归一化，所以由于初始权重尺度不当导致激活值或梯度在层间累积而爆炸的风险降低了。后归一化通常需要更仔细地调整初始化方差，并且可能需要针对不同层类型采用特定的初始化方案（例如，残差连接层采用较小的方差）。
- **激活函数 (activation function)：** 激活函数的选择会影响推荐的初始化方案（例如，ReLU使用Kaiming初始化，GeLU/SwiGLU可能需要调整方案）。确保初始化与激活函数的特性相符有助于在网络层中保持方差的稳定性。

### 注意力机制 (attention mechanism)的细节

即使在标准的缩放点积注意力机制中，细节也很重要：

- **缩放因子（1/dk1/\sqrt{d\_k}1/dk​​）：** 这种缩放不只是一个优化；它对稳定性非常重要。如果没有它，对于较大的键维度 dkd\_kdk​ 值，点积 QKTQK^TQKT 会变得非常大。softmax函数接收大的输入会导致分布极端尖锐和梯度消失，从而使训练停滞。确保正确实现这种缩放是根本的。
- **替代注意力：** 更先进的注意力机制（如第11章或第13章中讨论的，例如相对位置编码 (positional encoding)或稀疏注意力）可能会以某种方式修改计算，从而不明显地影响稳定性。例如，旋转位置编码（RoPE）在点积之前直接修改查询和键，这与标准的加性偏置 (bias)相比，可能与初始化或精度有不同的相互影响。

### 嵌入 (embedding)层和输出层

- **大词汇量：** 具有非常大的嵌入表（输入嵌入或输出投影层）的模型有时会看到与不常见标记 (token)相关的大梯度，如果梯度裁剪未能管理好，可能会导致损失值骤增。
- **权重 (weight)绑定：** 将输入嵌入权重与最终输出投影权重绑定是一种常见的减少参数 (parameter)的做法。虽然通常有益，但这表示同一个权重矩阵会同时被来自初始嵌入查找和最终预测损失的梯度更新，这有时会使优化动态变得复杂。在使用权重绑定时，可能需要特定的初始化策略。
- **输出层初始化：** 通常建议将最终输出投影层（将最后一个隐藏状态映射到logits）的初始化方差设置得比其他层小。这有助于防止大的初始预测值，这些值可能导致高的初始损失和潜在的不稳定性。

总之，架构决策并非与训练稳定性无关。归一化 (normalization)层的位置、激活函数 (activation function)的选择、与初始化的相互影响，甚至注意力机制 (attention mechanism)内部的细节，都对整体训练动态有所贡献。尽管预归一化（Pre-LN）和像GeLU/SwiGLU这样的激活函数通常被现代大型模型青睐以提高稳定性，但了解这些关联能让你在问题出现时更好地诊断它们，并做出明智的设计选择，从一开始就构建出更易于训练的LLM。持续监控仍然非常重要，以便在训练过程中观察这些选择的实际效果。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 25 Supervised Fine Tuning Alignment

### 大型语言模型对齐的目标

# 大型语言模型对齐的目标

虽然预训练 (pre-training)的大型语言模型 (LLM) 展现出色的能力，能够基于从大量数据集中学习到的模式来理解和生成人类语言，但它们并不会自动地以对特定应用始终有用、真实或安全的方式行事。它们的训练目标通常侧重于预测序列中的下一个标记 (token)，即 P(标记i+1∣标记1,...,标记i)P(\text{标记}\_{i+1} | \text{标记}\_1, ..., \text{标记}\_i)P(标记i+1​∣标记1​,...,标记i​)，这最大化了预训练语料库上的似然度，但并不会直接优化以遵循用户指令或符合人类价值观。

对齐 (alignment)是调整预训练大型语言模型以更好地匹配人类意图和偏好的过程。它旨在引导模型的强大生成能力朝着期望的行为。监督式微调 (fine-tuning) (SFT)，作为本章的侧重点，是此过程中的重要一步。对齐的主要目标，也是SFT开始处理的，常分为三个大类，有时被称为“HHH”标准：有用性、诚实性、无害性。

### 有用性

这或许是SFT处理的最直接的目标。一个有用的模型应该理解并准确遵循提示中呈现的用户指令。它应该有效地执行所请求的任务，无论是回答问题、总结文本、编写代码、翻译语言，还是以特定的对话风格交流。

考虑一个被问及“解释梯度下降 (gradient descent)的原理”的预训练 (pre-training)模型。

- **无用（但预训练模型可能有的行为）：** 模型可能会生成笼统地定义微积分中的梯度，或者列出提及梯度下降的论文，而没有清晰地解释算法本身。
- **有用（对齐 (alignment)后的行为）：** 模型提供清晰、循序渐进的解释，针对可能的语境（例如，机器学习 (machine learning)）进行调整，可能包含类比或简单的例子。

SFT 通过使模型接触大量由高质量、有用响应配对的提示示例来做到这一点。微调 (fine-tuning)过程调整模型的参数 (parameter)，以增加为类似提示生成此类有用响应的概率。这涉及最小化模型生成响应与SFT数据集中目标有用响应之间的损失（例如，交叉熵）。

G

Pretraining

预训练大型语言模型
(预测下一个标记)

SFT

监督式微调 (SFT)
(从示例中学习)

Pretraining->SFT

 通过
指令数据调整

AlignedLLM

对齐的大型语言模型
(有用、诚实、无害)

SFT->AlignedLLM

 初步对齐至
期望行为

> 对齐通过SFT等技术，将通用预训练模型转变为表现出期望行为的模型。

### 诚实性 (真实性)

一个对齐 (alignment)的模型应该力求准确，并避免生成虚假信息，通常被称为“幻觉 (hallucination)”。虽然预训练 (pre-training)使模型接触到事实知识，但其生成性质意味着它很容易构建听起来合理但不正确的陈述。诚实性意味着：

- **事实准确性：** 在可能时提供正确的信息。
- **避免捏造：** 不虚构事实、来源或细节。
- **表达不确定性：** 当模型不知道答案或信息不明确时，应表明出来，而不是自信地猜测。

SFT 有助于提高诚实性，通过包含模型正确回答事实问题或明确说明其局限性的示例。然而，确保深层的事实性和校准的不确定性通常需要更先进的技术，例如整合检索机制或使用强化学习 (reinforcement learning)（如RLHF，在第26章中讨论）以惩罚人类反馈识别出的不真实输出。

### 无害性

此目标侧重于阻止模型生成有害、不道德、带有偏见、有毒或促进非法活动的输出。预训练 (pre-training)数据不可避免地包含互联网和数字化文本中存在的偏见和有害内容。一个未对齐 (alignment)的模型可能会轻易地再现或放大这些问题。无害性要求模型：

- **拒绝不当请求：** 拒绝生成属于有害类别的任何内容。
- **避免偏见：** 减少生成刻板或偏见的内容。
- **保持安全：** 不提供危险活动的指令。

SFT 通过包含模型拒绝有害请求或提供安全、中性响应的示例，在此方面发挥作用。精心策划的SFT数据集过滤掉不理想的示例，并明确呈现安全拒绝。类似于诚实性，在多样化和对抗性输入下实现无害性具有挑战性，并且通常会从后续的RLHF中获得显著益处，在RLHF中，模型根据人类判断被训练以偏好安全输出。

总而言之，对齐旨在使大型语言模型不仅有能力，而且在各种应用中成为有益且安全的伙伴。SFT 作为一个起点，主要增强了有用性和指令遵循能力，同时通过提供期望模型输出的具体示例，也开始了灌输诚实性和无害性的过程。这些目标指导SFT数据集的创建和对齐模型的评估，确保它们从仅仅预测文本转向生成真正有用和负责任的响应。

获取即时帮助、个性化解释和交互式代码示例。

---

### 监督微调（SFT）介绍

# 监督微调（SFT）介绍

预训练 (pre-training)语言模型尽管其表现出色的能力，但通常缺乏可靠地遵循用户指令或遵守期望行为准则所需的特定调整。它们通过海量文本数据训练，以预测序列中的下一个词元 (token)，但这个目标不能直接转化为人类期望中定义的帮助性、诚实性或无害性。监督微调 (fine-tuning)（SFT）是一种旨在弥补这一差距的技术，它通过明确地教导模型如何以更受偏好的方式回应提示。

SFT 借助由精选输入提示及其对应期望输出组成的数据集，来调整预训练的大语言模型 (LLM)。可以将其视为直接为模型提供它*应该*如何表现的示例。模型不再从网络规模文本的隐含模式中学习，而是从优秀回复的明确示范中学习。这个过程包括在这些监督示例上进一步训练预训练模型，通常采用交叉熵等标准序列到序列损失函数 (loss function)。

### SFT 的运行原理

其核心是，SFT 通过使模型生成输出与微调 (fine-tuning)数据集中提供的目标输出之间的差异最小化，从而优化模型的参数 (parameter)。这个过程通常遵循以下步骤：

1. **从预训练 (pre-training)大语言模型 (LLM)开始：** 选择一个已经进行过广泛预训练的基础大语言模型。这个模型提供了基础知识和语言理解能力。
2. **准备指令数据集：** 整理或生成一个包含（提示，期望回复）对的数据集。这个数据集的质量和多样性显著影响SFT的结果。示例可能包含问题与有帮助的答案配对、指令与正确执行的任务配对，或对话轮次与适当的后续内容配对。
3. **格式化数据：** 每对数据通常被格式化为一个单独的序列，常使用特殊词元 (token)来划分提示和回复部分。例如：`<|prompt|> 马来西亚的首都在哪里？<|response|> 马来西亚的首都是吉隆坡。<|endoftext|>`。
4. **微调模型：** 在这个格式化的数据集上训练预训练模型。标准训练目标是预测下一个词元，但重要的是，损失通常*只*针对序列中属于`desired_response`部分的词元计算。提示词 (prompt)元作为上下文 (context)，但不直接参与损失计算或梯度更新。

这种有针对性的损失计算非常重要。我们希望模型学习*如何根据提示生成回复*，而不是简单地预测提示词元本身（这部分它在预训练期间已经学习过）。

考虑目标函数。在预训练期间，模型最大化整个文本语料库的似然，即 P(ext文本)P( ext{文本})P(ext文本)。在SFT中，模型学习一个条件概率：给定特定提示，它最大化期望回复的似然，即 P(ext回复∣ext提示)P( ext{回复} | ext{提示})P(ext回复∣ext提示)。这种转变使模型专注于根据指令输入生成适当的输出。

### SFT 流程图示

我们可以用图示呈现单个SFT训练步骤中信息的基本流向：

SFT\_Flow

Prompt

输入提示

PreTrainedLLM

预训练
大语言模型

Prompt->PreTrainedLLM

Dataset

指令
数据集

Dataset->Prompt

DesiredResponse

期望回复

Dataset->DesiredResponse

GeneratedResponse

生成回复

PreTrainedLLM->GeneratedResponse

LossCalculation

计算损失
(仅针对回复)

DesiredResponse->LossCalculation

GeneratedResponse->LossCalculation

GradientUpdate

更新
模型权重

LossCalculation->GradientUpdate

GradientUpdate->PreTrainedLLM

优化

> SFT 过程的简化示意图，显示了如何使用数据集中的提示和期望回复来计算损失并更新预训练 (pre-training)大语言模型 (LLM)的权重 (weight)。

### 损失掩码

为了在实际操作中，使用 PyTorch 等框架实现有针对性的损失计算，我们通常会创建一个损失掩码。这个掩码确保只有与期望回复对应的词元 (token)才参与损失计算。

这是一个 PyTorch 代码片段，说明了这一点：

```python
import torch
import torch.nn.functional as F

IGNORE_INDEX = -100

def calculate_sft_loss(logits, labels, prompt_lengths):
    """仅针对回复词元计算交叉熵损失。"""
    batch_size, sequence_length, vocab_size = logits.shape

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_mask = torch.ones_like(shift_labels, dtype=torch.bool)

    for i in range(batch_size):

        prompt_end_index = prompt_lengths[i] - 1
        if prompt_end_index > 0:
             loss_mask[i, :prompt_end_index] = 0

    masked_labels = shift_labels.masked_fill(~loss_mask, IGNORE_INDEX)

    shift_logits = shift_logits.view(-1, vocab_size)
    masked_labels = masked_labels.view(-1)

    loss = F.cross_entropy(shift_logits,
                           masked_labels,
                           ignore_index=IGNORE_INDEX)

    return loss
```

> 这个代码片段概述了如何掩盖损失计算，确保在SFT期间只有回复词元影响模型更新。

### SFT 的目的与目标

监督微调 (fine-tuning)有几个重要的对齐 (alignment)目标：

- **指令遵循：** 教导模型理解并执行自然语言提示中的命令或回答问题。
- **格式遵守：** 训练模型生成特定格式的输出（例如 JSON、Markdown、代码块、特定对话风格）。
- **增强可控性：** 使模型行为更可预测，并与用户在特定任务上的意图保持一致。
- **初步安全与帮助性：** 通过提供期望交互的示例（例如，拒绝有害请求、提供礼貌回复），引入基本的安全约束和有帮助的对话模式。

虽然 SFT 在教导模型*何种*回复是基于示例所期望的方面是有效的，但它本身不能完美地捕获人类偏好。它教导模型模仿所提供回复的风格和内容。对于更复杂的对齐目标，例如判断多个合理回复之间的相对质量，或优化“帮助性”等特质，SFT 之后通常会采用来自人类反馈的强化学习 (reinforcement learning)（RLHF）等技术，我们将在下一章讨论。SFT 提供了一个基础，使模型具备遵循指令的基本能力，之后再使用基于偏好的方法进行进一步的优化。

获取即时帮助、个性化解释和交互式代码示例。

---

### 构建高质量指令数据集

# 构建高质量指令数据集

监督微调 (fine-tuning)（SFT）完全取决于所用数据集的质量和构成。与预训练 (pre-training)不同，预训练的目标是在大量通常带有噪声的文本中进行普遍的模式识别，而SFT旨在教导模型特定的、合意的行为。因此，指令数据集充当了模型对齐 (alignment)个性和能力的蓝图。一个精心制作的SFT数据集决定了模型是仅仅能生成文本，还是能够可靠地遵循指令、进行有益对话并遵守安全限制。

主要思想很简单：为模型提供您希望它进行的互动示例。每个示例通常由一个“指令”（或提示、查询、上下文 (context)）和一个预期的“响应”（或输出、完成）组成。通过使用标准语言建模损失（预测下一个词元 (token)）来训练模型，使其根据指令预测预期响应，我们将引导其行为，以便未来生成类似的高质量响应。

### 高质量指令数据集的特点

构建一个有效的SFT数据集需要仔细考量几个因素：

1. **指令多样性：** 数据集应包含各种各样的任务和指令类型。这包括：

   - **开放式生成：** 创意写作提示、头脑风暴、续写任务。
   - **封闭式任务：** 问答（事实型、阅读理解）、摘要、信息提取、分类。
   - **编程：** 代码生成、解释、调试。
   - **推理 (inference)：** 数学问题、逻辑谜题、逐步思考。
   - **对话：** 多轮对话示例。
   - **重写/编辑：** 风格转换、语法修正、简化。
   - **安全/拒绝：** 模型应拒绝有害、不道德或不恰当请求的示例。

   一个多样化的数据集可以防止模型对狭窄的任务集过拟合 (overfitting)，并促进对未见指令的更好泛化能力。
2. **响应质量：** 这也许是最重要的方面。响应应：

   - **有益：** 直接回应指令并提供相关信息。
   - **真实：** 事实准确，避免凭空捏造（幻觉 (hallucination)）。如果不确定，模型理想情况下应表明不确定性。
   - **无害：** 避免生成有害、偏见、非法或危险内容。对有害指令的响应应为拒绝。
   - **清晰且格式良好：** 使用清晰的语言、适当的结构（如列表或段落）和正确的格式（如代码的markdown）。
3. **指令清晰度：** 指令本身应措辞良好且明确。模糊的指令可能导致泛化或无益的响应，使模型难以学习预期的行为。
4. **足够规模：** 尽管质量胜于数量，但一个合理规模的数据集（从数千到数十万个示例不等，取决于模型大小和多样性目标）对于模型有效学习是必要的。

### 数据创建的来源与方法

获取或生成高质量的指令-响应对是一项重大的工程投入。常用方法包括：

1. **使用现有公共数据集：** 一些公开可用的数据集已为指令微调 (fine-tuning)而创建。例如包括FLAN Collection的子集、P3（Prompt公共池）、Alpaca数据集、Dolly数据集和OpenAssistant Conversations。这些可以提供一个良好的起点，但其质量、多样性和许可条款可能有所不同。仔细审查和筛选这些数据集非常重要。
2. **人工标注与整理：** 这通常被认为是质量的黄金标准。人工标注者会获得指导原则，并被要求编写指令和/或高质量响应。

   - **指令编写：** 人工编写多样化和有创意的提示。
   - **响应编写：** 人工编写针对给定指令的详细、准确和安全的响应。
   - **质量评分/排序：** 人工对模型生成的响应进行评分或比较，这可以稍后用于筛选或训练奖励模型（参见第26章）。

   尽管质量高，但这种方法昂贵、耗时，并且需要严格的质量控制流程和明确的标注指导原则。

   G

   Annotator

   人工标注者

   Instruction

   指令（提示）

   Annotator->Instruction

   Response

   预期响应

   Annotator->Response

   Guidelines

   标注指导原则

   Guidelines->Annotator

   Dataset

   SFT数据集

   Instruction->Dataset

   Response->Dataset

   > SFT数据的人工标注流程。
3. **模型生成（自指导/演化方法）：** 这种方法使用一个强大的现有大型语言模型（通常是专有模型或强大的开源模型）来生成新的指令-响应对，有时从少量人工编写的种子示例开始。

   - **指令生成：** 提示模型根据种子示例创建新颖指令。
   - **响应生成：** 提示模型为给定指令生成响应。
   - **筛选：** 使用启发式方法、分类器，甚至生成器模型本身来筛选出低质量或重复的生成示例。

例如，可能会像这样提示一个有能力的LLM（示例）：

```
```python
```

# 使用生成函数示例

```
import hypothetical_llm_client

seed_instructions = [
    {
        "instruction": "Write a Python function to calculate factorial.",
        "response": "def factorial(n): ..."
    },
    {
        "instruction": "Explain the concept of photosynthesis.",
        "response": "Photosynthesis is the process..."
    },
    # ... 更多种子示例
]

prompt_template = """
您的任务是生成新的、多样化的编程相关指令，类似于提供的示例。
确保指令清晰且与示例不同。
生成一条新指令。

示例：
{seed_examples_formatted}

新指令："""

formatted_seeds = "\n".join(
    [f"Instruction: {ex['instruction']}" for ex in seed_instructions]
)
generation_prompt = prompt_template.format(seed_examples_formatted=formatted_seeds)

# 假设 generate_text 生成指令文本
new_instruction_text = hypothetical_llm_client.generate_text(
    prompt=generation_prompt,
    max_length=100
)

print(f"生成的指令: {new_instruction_text}")

# 之后，可能会再次提示以获取此新指令的响应
# response_prompt = f"Instruction: {new_instruction_text}\nResponse:"
# new_response = hypothetical_llm_client.generate_text(
#   prompt=response_prompt,
#   max_length=500
# )
```

尽管可扩展，但这种方法有放大生成器模型中偏见的风险，并且与人工标注相比，有时会生成多样性较差或质量较低的数据。仔细的筛选和潜在的人工审查通常是必要的。
```

4. **混合来源：** 通常，最有效的数据集会结合来自多个来源的示例。例如，从公共数据集开始，使用人工整理的示例进行扩充，涵盖特定方面或安全行为，并可能添加用于特定能力的合成生成数据。

### 数据整理与质量控制

无论来源如何，原始指令-响应对都需要整理：

- **筛选：** 移除包含不正确信息、有害内容、无意义指令或格式不佳响应的示例。可以使用自动化筛选器（例如，毒性分类器、代码规范检查工具、长度启发式规则）和人工审查。
- **去重：** 移除完全或近似重复的指令-响应对，以提高数据效率。哈希或语义相似性检查等技术会有帮助。
- **格式标准化：** 确保所有示例遵循模型在训练期间将看到的一致格式（例如，在指令和响应之间使用特定分隔符）。这将在下一节中更详细地介绍。
- **平衡：** 检查任务和主题的分布。如果数据集严重偏向某一种指令类型（例如，简单问答），模型在其他方面可能表现不佳。考虑对代表性不足的类别进行过采样或为其生成更多数据。

构建高质量指令数据集是一个迭代过程。它涉及仔细规划、生成或收集、严格清洗，以及根据SFT模型在评估期间的表现进行持续改进。在此投入的努力直接转化为一个更有益、真实和无害的语言模型。

获取即时帮助、个性化解释和交互式代码示例。

---

### SFT数据格式（提示词，回应）

# SFT数据格式（提示词，回应）

为了在监督微调 (fine-tuning)（SFT）过程中让模型有效学习，高质量的指令-回应数据集需要进行适当的格式化。SFT的目的是在给定特定输入（提示词 (prompt)）时，教会模型生成期望的输出（回应）。适当的格式化可确保模型理解任务结构，并将其学习重心放在生成正确的回应上。

### 基本提示词 (prompt)-回应结构

最简单地说，每个SFT示例包含两部分：

1. **提示词：** 提供给模型的输入，包含指令、背景信息或问题。
2. **回应：** 模型应生成的期望输出或答案。

模型在给定`提示词`后，会按顺序预测`回应`的标记 (token)。以一个直接的问答示例来说：

- **提示词：** "问题：马来西亚的首都是哪里？\n答案："
- **回应：** " 吉隆坡"

训练期间，模型会处理提示词并学习将其与目标回应关联起来。用于提示词的具体文本会因任务和期望的交互风格而有很大不同。它可能包含明确的指令、示例（在少量样本情况下），或对话历史。

### 针对不同任务类型的格式化

结构需要根据你希望模型学习的具体行为来调整。

#### 指令遵循

对于模型应遵循明确命令的任务，提示词 (prompt)会清楚地说明指令，通常后跟输入数据。

```text
# 示例 1：摘要
提示词: "总结以下文章：\n\n[文章文本在此...]\n\n总结："
回应: "[文章的简明摘要]"

# 示例 2：代码生成
提示词: "编写一个计算数字阶乘的Python函数。\n```python\n"
回应: "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n```"
```

#### 对话

为了训练对话代理，格式必须呈现对话的往复特性。这通常涉及使用特殊标记 (token)或分隔符来区分用户和助手的发言。

```text
# 使用特殊角色标记的对话示例格式
提示词: "<|USER|> 你好，能解释一下光合作用吗？\n<|ASSISTANT|>"
回应: " 光合作用是植物、藻类和蓝细菌将光能转化为化学能的过程..."

# 多轮对话示例格式
提示词: "<|USER|> 伦敦天气怎么样？\n<|ASSISTANT|> 伦敦现在多云，15°C。\n<|USER|> 明天呢？\n<|ASSISTANT|>"
回应: " 伦敦明天的预报是局部多云，最高气温18°C。"
```

使用`<|USER|>`和`<|ASSISTANT|>`等不同标记有助于模型学习对话结构并识别轮到谁发言。

#### 思维链推理 (inference)

对于需要推理的任务，回应本身可能包含得出最终答案的中间步骤。这会教导模型*如何*得到答案。

```text
提示词: "问题：约翰有5个苹果。他又买了3箱，每箱有4个苹果。他总共有多少个苹果？\n答案："
回应: " 约翰最初有5个苹果。他买了3箱 * 4个苹果/箱 = 12个苹果。总共，他有 5 + 12 = 17个苹果。最终答案是17。"
```

### 特殊标记 (token)和分隔符

为了帮助模型在连接的输入序列中清楚地区分提示词 (prompt)和回应，通常会使用特殊标记。这些可以是模型分词 (tokenization)器 (tokenizer)中预定义的标记（如 `[SEP]`、`</s>`、`<|endoftext|>`），也可以是专门为SFT添加的自定义标记（如 `<|PROMPT|>`、`<|COMPLETION|>`、`<|END_OF_TURN|>`）。

考虑一个使用通用标记的简化示例：

```python
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

special_tokens_dict = {'sep_token': '<|SEP|>', 'pad_token': '<|PAD|>'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

prompt = "Translate to French: Hello world"
completion = " Bonjour le monde"

formatted_text = (
    f"{prompt}{tokenizer.sep_token}"
    f"{completion}{tokenizer.eos_token}"
)

tokenized_input = tokenizer(formatted_text, return_tensors="pt")

print("Formatted Text:", formatted_text)
print("Token IDs:", tokenized_input['input_ids'])
```

分隔符的选择会影响模型在训练和推理 (inference)时如何划分输入。在整个数据集中一致地应用这些标记是很重要的。

### 损失计算的掩码处理

SFT训练的一个重要方面是确保损失*只*在回应标记 (token)上计算。模型应学习预测期望的输出，而不是预测它已作为输入收到的提示词 (prompt)本身。这通常通过使用标签掩码或修改注意力掩码来实现。

为模型准备批次数据时，输入ID将包含连接的提示词和回应标记。标签（损失函数 (loss function)的目标）通常是输入ID的右移版本。我们需要告知损失函数（例如CrossEntropyLoss）忽略为提示词标记计算的损失。

```python
import torch
import torch.nn.functional as F

input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])

labels = torch.tensor([[-100, -100, -100, -100, -100, 60, 70, 80]])

vocab_size = 100
sequence_length = input_ids.shape[1]
model_output_logits = torch.randn(1, sequence_length, vocab_size)

loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

loss = loss_fct(
    model_output_logits.view(-1, vocab_size),
    labels.view(-1)
)

print("Calculated Loss (only on completion tokens):", loss.item())
```

在此片段中，将与提示词标记对应的标签值设置为`-100`（PyTorch的CrossEntropyLoss的默认`ignore_index`）可确保这些位置不参与损失计算或梯度更新。只有模型对回应标记（`[60, 70, 80]`）的预测会受到惩罚。

### 连接和一致性

通常，提示词 (prompt)和回应会连接成一个单一序列，并输入给模型，通常像前面所示那样由一个特殊标记 (token)分隔。模型随后会处理整个序列。

G

clusterₗoss

损失计算

Prompt

提示词标记
(例如，'问题：...')

Separator

分隔符标记
(例如，<|SEP|>)风

Prompt->Separator

 输入序列

LossMask

已掩码（损失忽略）

Completion

回应标记
(例如，'答案：...')

Separator->Completion

EOS

EOS 标记
(例如，<|EOS|>)风

Completion->EOS

LossCalc

未掩码（损失计算）

> SFT输入序列的常见结构。提示词和回应被连接起来，通常带有分隔符和序列结束标记。损失通常只在回应部分计算。

在SFT数据集中所有示例中保持格式的绝对一致性是重要的。不一致地使用空格、换行符或特殊标记会使模型感到困惑，并阻碍其学习期望的模式。选择一种格式并统一应用。

### 数据结构示例

SFT数据集通常以JSON Lines (`.jsonl`) 等格式存储，其中每行是一个代表一个示例的JSON对象。

```json
{
    "prompt": "对情感进行分类：'这部电影太棒了！'\n情感：",
    "completion": " 积极"
}
{
    "prompt": "写一首关于月亮的短诗。\n诗歌：",
    "completion": " 银盘悬于夜幕，\n轻柔投下暗影，柔和而明亮"
}
{
    "prompt": "<|USER|> 水的沸点是多少摄氏度？\n<|ASSISTANT|>",
    "completion": " 水的沸点是100摄氏度"
}
```

或者，结构化格式可能会分开指令、输入和输出：

```json
{
    "instruction": "将以下英文文本翻译成西班牙语。",
    "input": "今天天气很好。",
    "output": " Hace buen tiempo hoy."
}
```

所选择的结构应清晰地映射到分词 (tokenization)和训练时使用的提示词 (prompt)-回应格式。

### 处理序列长度

一个实际的考量是模型支持的最大序列长度。如果连接的提示词 (prompt)和回应超过此限制，你需要一个截断策略。常见的方法包括：

- **截断回应：** 优先保留完整的提示词，尤其是当它包含重要背景信息或指令时，并截断回应的末尾。这可能会丢失一些目标信息。
- **截断提示词：** 保留完整的回应，并截断提示词的开头或结尾。这可能会丢失提示词中的重要背景信息。
- **丢弃示例：** 丢弃过长的示例。这是最简单的方法，但会减小数据集大小。

最佳策略取决于具体任务以及提示词与回应内容的相对重要性。

通过仔细格式化你的SFT数据，清晰区分提示词和回应，并确保一致性，你为模型提供了它所需的结构化输入，以便有效学习指令遵循和有益对话等对齐 (alignment)行为。

获取即时帮助、个性化解释和交互式代码示例。

---

### SFT训练过程与超参数

# SFT训练过程与超参数

监督微调 (fine-tuning)（SFT）通过在高质量的提示-响应示例数据集上训练，使预训练 (pre-training)的大型语言模型能够遵循指令或以特定风格生成回复。与预训练不同，预训练侧重于在大量非结构化文本上进行下一个词元 (token)预测，而SFT是一种更有针对性的训练形式，旨在使模型的行为与预期结果保持一致。训练过程本身类似于序列到序列任务的标准监督学习 (supervised learning)，但涉及数据格式、损失计算和超参数 (parameter) (hyperparameter)选择方面的具体考量。

### 训练循环机制

核心SFT训练循环会遍历批量的提示-响应对，执行前向传播，根据模型预测与目标响应之间的差异计算损失，并通过反向传播 (backpropagation)更新模型权重 (weight)。

1. **数据准备：** 每个训练示例通常包含一个提示（例如，一个指令或用户查询）和一个期望的响应（例如，指令的答案或一个有用的回复）。这些通常被连接成一个序列，有时会用特殊词元 (token)分隔提示和响应部分，或指示对话回合的开始/结束。

   ```python

   prompt = "Instruction: Explain the process of photosynthesis."
   response = "Photosynthesis is the process used by plants..."

   tokenizer.add_special_tokens({'pad_token': '[PAD]', 'eos_token': '[EOS]'})

   input_text = prompt + " " + response + tokenizer.eos_token

   tokenized_input = tokenizer(
       input_text,
       return_tensors="pt",
       padding="max_length",
       truncation=True,
       max_length=512
   )
   inputs = tokenized_input["input_ids"]
   attention_mask = tokenized_input["attention_mask"]
   ```
2. **前向传播：** 分词 (tokenization)后的序列被输入模型，以获取序列中每个词元位置的logits。

   ```python

   outputs = model(input_ids=inputs, attention_mask=attention_mask)
   logits = outputs.logits
   ```
3. **损失计算（屏蔽提示）：** 这是SFT的一个显著特点。目标是教导模型根据*提示*生成*响应*。因此，损失通常*仅*在响应词元上计算。对应于提示的词元被屏蔽，因此它们不参与损失计算或梯度更新。未屏蔽（响应）词元上使用标准交叉熵损失。

   ```python
   import torch
   import torch.nn.functional as F

   labels = inputs.clone()

   logits = logits[:, :-1, :]
   labels = labels[:, 1:]

   loss_mask = torch.ones_like(labels, dtype=torch.long)
   for i in range(labels.shape[0]):

       prompt_end_index = prompt_lengths[i] - 1
       loss_mask[i, :prompt_end_index] = -100
       if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            loss_mask[i][labels[i] == tokenizer.pad_token_id] = -100

   active_loss = loss_mask.view(-1) != -100
   active_logits = logits.view(-1, logits.size(-1))[active_loss]
   active_labels = labels.view(-1)[active_loss]

   loss = F.cross_entropy(active_logits, active_labels)
   ```

   实际上，如果数据格式正确（例如，使用特定数据集格式或提供数据收集器），Hugging Face的`transformers.Trainer`或TRL (`trl.SFTTrainer`) 等库会在内部处理这种掩码逻辑。
4. **反向传播与优化：** 标准反向传播根据计算出的损失来计算梯度。优化器（通常是AdamW）更新模型权重。

   ```python

   optimizer.zero_grad()
   loss.backward()

   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   optimizer.step()
   if scheduler is not None:
       scheduler.step()
   ```

### 超参数 (parameter) (hyperparameter)考量

选择合适的超参数对SFT的成功很重要。由于SFT调整的是一个已很强大的预训练 (pre-training)模型，其设置与预训练时使用的不同。

- **学习率：** SFT通常使用比预训练小得多的学习率。1×10−51 \times 10^{-5}1×10−5 到 5×10−55 \times 10^{-5}5×10−5 范围内的值很常见。较小的学习率可以防止预训练期间获得的知识被灾难性遗忘，同时允许模型适应新的指令遵循目标。学习率调度器，如带有短热身阶段（例如，总步数的0-10%）的余弦衰减，通常会带来好处。

  G

  Start

  初始学习率
  (例如, 1e-7)

  Warmup

  热身阶段
  (线性增长)

  Start->Warmup

  步数: 0% -> ~5%

  Peak

  峰值学习率
  (例如, 2e-5)

  Warmup->Peak

  Decay

  余弦衰减阶段

  Peak->Decay

  步数: ~5% -> 100%

  End

  最终学习率
  (接近零)

  Decay->End

  > 一个典型的SFT学习率调度，包括热身和余弦衰减。
- **批量大小：** 批量大小通常受限于GPU内存。更大的模型需要更多内存，从而限制了每个批次的序列数量。梯度累积常用于在不增加每GPU内存使用量的情况下实现更大的*有效*批量大小。典型的有效批量大小可能在64到1024之间，具体取决于模型大小和可用硬件。
- **训练轮次（Epoch）数量：** SFT通常只需要少数几个训练轮次（通常1-3个，有时最多5个）。训练时间过长可能导致在特定SFT数据集上过拟合 (overfitting)，从而可能降低模型对未见指令的泛化能力或降低其通用知识。监控验证集上的表现很重要。
- **优化器：** AdamW仍然是标准选择，类似于预训练。权重 (weight)衰减参数可以保持不变或略作调整（例如，0.01到0.1）。Beta参数（β1,β2\beta\_1, \beta\_2β1​,β2​）通常保持其默认值（例如，0.9，0.999）。
- **序列长度：** 最大序列长度应能容纳SFT数据集中典型提示和响应的组合长度。它可能与预训练期间使用的序列长度不同。将多个短示例打包到一个序列中或使用动态填充可以提高效率。
- **梯度裁剪：** 应用梯度裁剪（例如，将梯度的L2范数裁剪到1.0）有助于稳定训练，尽管与大规模预训练相比，SFT中的不稳定性通常较少发生。

### 实际实施要点

- **计算资源：** 虽然SFT对计算资源的要求低于预训练 (pre-training)数万亿词元 (token)，但大型模型的SFT仍需要大量计算，通常涉及多个高内存GPU（如A100或H100）。
- **分布式训练：** 对于更大的模型或更快的迭代，通常采用数据并行（DP），使用PyTorch的DistributedDataParallel (DDP) 或DeepSpeed（尤其是ZeRO Stage 1或2）等框架。
- **数据集质量：** SFT数据集的质量和多样性可以说比纯粹的数量更重要。几千个高质量、多样化的示例通常能比数百万个嘈杂或重复的示例产生更好的结果。
- **评估：** 使用损失指标评估SFT。通过在保留的指令集上评估表现，使用自动化指标（如用于摘要的ROUGE），或采用人工评估，是了解对齐 (alignment)目标是否达到所必需的。

SFT过程微调 (fine-tuning)模型的能力，将模型的预训练知识引导至指令数据集定义的特定交互模式和任务执行格式。对训练循环和超参数 (parameter) (hyperparameter)的细致管理可确保这种调整有效进行，同时不损害模型内在优势。

获取即时帮助、个性化解释和交互式代码示例。

---

### 评估SFT模型对齐目标

# 评估SFT模型对齐目标

评估监督式微调 (fine-tuning)（SFT）模型的主要目标是验证模型是否已更好地与期望行为*对齐 (alignment)*。这项评估有别于单纯继续预训练 (pre-training)的目标，它侧重于评估模型遵循指令、提供有用回复和遵守指定限制的能力，而非原始的语言建模能力（如在通用语料库上的困惑度）。这一评估步骤非常重要，因为它决定了SFT阶段的成功，并常为后续的对齐阶段（如基于人类反馈的强化学习 (reinforcement learning)（RLHF））提供信息。

### 定义评估的对齐 (alignment)目标

在评估之前，需清晰定义SFT旨在实现的特定对齐目的。这些通常包括：

1. **指令遵循**：模型能否准确理解并执行提示中提供的各类指令？这是许多SFT数据集的核心目标。
2. **实用性**：模型生成的回复在提示语境中是否实用、相关且信息丰富？
3. **格式依从**：如果指令指定了特定的输出格式（例如，JSON、markdown列表、特定语气），模型是否遵从？
4. **安全与无害性**：虽然这常是RLHF的主要侧重，但SFT评估也应检查模型是否避免生成有害、偏见或不适当的内容，以及SFT是否在此方面引入了任何退步。

衡量这些品质常需超越标准自动化指标。

### 人工评估：真实情况

对于主观属性，如实用性、复杂任务上的指令遵循忠实度以及无害性，人工评估仍是最可靠的方法。建立有效的人工评估涉及几个考量：

- **任务设计**：创建专门探测目标对齐 (alignment)目的的评估提示。这些提示可以取自SFT数据的保留集、新创建的提示或既定的对齐基准。
- **评估标准**：制定清晰一致的评分准则（标准）。这可以包括李克特量表（例如，实用性1-5分评级）、成对比较（哪个回复更好，A还是B？）或分类判断（模型是否遵循了格式指令：是/否？）。
- **评估者培训**：确保评估者理解任务、标准和潜在偏见。评估者之间的一致性检查很要紧。
- **抽样**：评估每个可能的输入是不可能的。选择涵盖各类指令类型、复杂度和潜在失败模式的代表性提示样本。

虽然强大，但人工评估是资源密集型的（时间、成本），并可能受到注释者间分歧的影响。它常用于验证自动化指标或进行周期性深入评估。

G

SFT\_Model

SFT模型

Generate

生成回复

SFT\_Model->Generate

Eval\_Prompts

评估提示

Eval\_Prompts->Generate

Responses

模型回复

Generate->Responses

Human\_Eval

人工评估
（标准/比较）

Responses->Human\_Eval

Auto\_Eval

自动化评估
（模型/指标）

Responses->Auto\_Eval

Scores

对齐分数

Human\_Eval->Scores

Auto\_Eval->Scores

> 使用人工和自动化方法评估SFT模型回复的工作流程。

### 自动化评估方法

为补充人工评估并实现更快迭代，会使用几种自动化方法：

1. **基于模型的评估**：借助一个强大、预先存在的LLM（常被称为“评估器模型”，例如GPT-4、Claude）来评估SFT模型回复的质量。

   - **过程**：为评估器模型设计一个提示，向其提供原始指令、SFT模型的回复，并可能包含一个参考答案或清晰的标准。评估器模型随后输出一个分数或判断。
   - **评估器LLM的示例提示**：

     ```
     你是一名公正的评判员，评估AI助手对用户指令回复的质量。
     指令：“将以下文本总结为三点：\n[此处插入长文本片段...]”
     助手回复：“[此处插入模型生成的摘要...]”

     请根据以下标准评估回复：
     1. 准确性：摘要是否准确反映了原文的主要观点？（1-5）
     2. 简洁性：摘要是否简短扼要？（1-5）
     3. 格式符合性：助手是否使用了正好三点？（是/否）

     请以JSON格式提供您的评分：{"准确性": <分数>, "简洁性": <分数>, "格式符合性": "<是/否>"}
     同时提供您评分的简短理由。
     ```
   - **优点**：可扩展，比人工评估快。可以捕捉到简单指标遗漏的细节。
   - **缺点**：取决于评估器模型的质量和潜在偏见。易受提示敏感性的影响。计算成本可能较高。常需与人工判断进行校准。
2. **基准数据集**：在专为指令遵循或实用性设计的既定基准上评估SFT模型。

   - **例子包括**：
     - **AlpacaEval**：使用GPT-4在Alpaca指令集上自动比较模型输出与参考回复（例如，来自`text-davinci-003`）。
     - **MT-Bench**：一个多轮基准，评估不同类别下的对话和指令遵循能力，常由强大的LLM进行评判。
     - **HELM (语言模型的整体评估)**：虽然范围更广，但HELM包含与指令遵循和鲁棒性等对齐 (alignment)目的相关的特定场景和指标。

   使用基准涉及在基准提示上运行SFT模型，然后使用基准规定的评估协议（常基于模型或基于人工）。
3. **基于参考的指标（谨慎使用）**：ROUGE（用于摘要）或BLEU（用于翻译）等指标可以用于*如果*SFT任务涉及生成应与参考文本（例如，针对特定摘要风格进行微调 (fine-tuning)）紧密匹配的文本。然而，它们常是衡量一般指令遵循或实用性的不佳指标，因为：

   - 它们会惩罚有效的转述和风格变化。
   - 它们不衡量事实正确性或逻辑推理 (inference)。
   - 高重叠分数不保正核心指令被遵循。

   考虑这个简单例子：
   指令：“用一句话解释万有引力。”
   参考：“万有引力是行星或其他物体将物体吸引向其中心的力量。”
   模型A：“万有引力是吸引有质量物体相互靠近的基本力。”（好，但BLEU/ROUGE得分低）
   模型B：“万有引力是一种力。物体被行星吸引到它们的中心。”（尚可，BLEU/ROUGE得分高但流畅度较低）

   在此处使用ROUGE可能会误导性地偏向模型B。

### 实现评估

我们来演示获取回复并准备评估。假设您已使用PyTorch和Hugging Face `transformers`加载了您的SFT模型和分词 (tokenization)器 (tokenizer)。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class DummyModel:
    def generate(
        self, input_ids, attention_mask, max_new_tokens, pad_token_id
    ):

        new_tokens = torch.randint(
            100,
            1000,
            (input_ids.shape[0], max_new_tokens),
            device=input_ids.device
        )
        output_ids = torch.cat([input_ids, new_tokens], dim=1)
        return output_ids
class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
    def encode(self, text, return_tensors=None):

        tokens = [101] + [ i+1000 for i in range(len(text.split())) ]
        return torch.tensor([tokens], dtype=torch.long)
    def decode(self, ids, skip_special_tokens=False):

        words = [
            f"word{i-1000}" if i >= 1000 else "[CLS]"
            for i in ids[0].tolist()
        ]
        return " ".join(words)
    def __call__(
        self, text, return_tensors=None, padding=False, truncation=False
    ):

        encoded = self.encode(text, return_tensors)
        return {"input_ids": encoded, "attention_mask": torch.ones_like(encoded)}

model = DummyModel()
tokenizer = DummyTokenizer()
device = torch.device("cpu")

def generate_response(prompt_text, model, tokenizer, max_new_tokens=100):
    """从SFT模型生成回复。"""
    inputs = tokenizer(
        prompt_text, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id
        )

    input_length = inputs.input_ids.shape[1]
    generated_ids = output_ids[:, input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response

eval_prompt = (
    "指令：编写一个计算阶乘的简短 Python 函数。"
    "\n回复："
)

generated_text = generate_response(eval_prompt, model, tokenizer)

print(f"评估提示：\n{eval_prompt}")
print(f"\n生成的回复：\n{generated_text}")
```

### 评估方法比较

不同的评估方法提供不同的信号。将它们结合使用常有益处。例如，使用自动化指标/基准进行广泛覆盖和频繁检查，并定期使用人工评估来验证自动化结果并检查不明显的问题。

指令遵循实用性安全性（基本检查）01234评估方法人工评分（平均1-5）GPT-4评分（平均1-5）自动化检查（通过率）

> 评估SFT模型对齐 (alignment)度的不同方法的分数比较。注意自动化检查与人工或基于模型的评分可能存在的差异。

归根结底，有效评估SFT模型需要清晰理解对齐目的，并周全结合人类洞察力和可扩展的自动化技术。评估结果指引进一步的微调 (fine-tuning)工作，助力创建不仅能干，而且真正实用可靠的LLM。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 26 Reinforcement Learning Human Feedback Rlhf

### RLHF 流程概述

# RLHF 流程概述

尽管监督微调 (fine-tuning)（SFT）为遵循指令提供了良好基础，但要使模型更好地符合复杂的人类偏好，通常需要人类反馈强化学习 (reinforcement learning)（RLHF）。RLHF 过程通过训练模型，使其基于人类判断为更好的输出，从而能够针对有益性和无害性等品质进行优化，这些品质仅通过监督示例难以把握。标准的 RLHF 流程通常包含三个主要阶段：

1. **模型初步准备（预训练 (pre-training)与 SFT）：** 从一个有能力的预训练语言模型开始。然后，使用高质量的提示-响应对数据集（如第 25 章所述）进行监督学习 (supervised learning)微调。SFT 步骤使模型适应遵循指令，并以期望的风格和格式生成响应。这个 SFT 模型将作为后续 RLHF 阶段的起点。
2. **奖励模型 (RM) 训练：** 核心思想是训练一个可以预测人类偏好的模型。

   - **数据收集：** 选择一组不同类型的提示。对于每个提示，使用 SFT 模型（或多个模型变体）生成多个响应。然后向人类标注者展示这些响应的成对（或更多）结果，并要求他们根据有益性、准确性或安全性等标准选择他们更喜欢哪一个。
   - **RM 架构：** 奖励模型（RMRMRM）通常是另一个语言模型（常基于 SFT 模型或较小的预训练模型进行初始化），其最后一层被替换为一个线性层，输出一个表示预测偏好分数的单一标量值。它以提示和响应作为输入并输出此分数。
   - **训练目标：** 奖励模型使用收集到的成对偏好数据进行训练。给定一个提示 ppp 和两个响应 ywy\_wyw​（胜者）和 yly\_lyl​（败者），其中人类更偏好 ywy\_wyw​ 而非 yly\_lyl​，奖励模型被训练为给 ywy\_wyw​ 分配比 yly\_lyl​ 更高的分数。常用的损失函数 (loss function)是成对排序损失：
     LRM=−E(p,yw,yl)∼D[log⁡(σ(rθ(p,yw)−rθ(p,yl)))]\mathcal{L}\_{RM} = - \mathbb{E}\_{(p, y\_w, y\_l) \sim D} [\log(\sigma(r\_{\theta}(p, y\_w) - r\_{\theta}(p, y\_l)))]LRM​=−E(p,yw​,yl​)∼D​[log(σ(rθ​(p,yw​)−rθ​(p,yl​)))]
     这里，rθ(p,y)r\_{\theta}(p, y)rθ​(p,y) 是奖励模型（参数 (parameter)为 θ\thetaθ）对提示 ppp 和响应 yyy 输出的标量分数，σ\sigmaσ 是 sigmoid 函数，DDD 是人类偏好数据集。这个目标最大化了首选响应 ywy\_wyw​ 获得更高分数的概率。
3. **强化学习 (RL) 微调：** 训练好的奖励模型现在作为人类偏好的代表，提供奖励信号，使用强化学习算法（通常是近端策略优化，PPO）进一步微调 SFT 模型。

   - **RL 设置：** SFT 模型在 RL 框架中充当初始*策略*（πSFT\pi\_{SFT}πSFT​）。*动作空间*包含模型可能生成的令牌，而*状态*是迄今为止生成的令牌序列。
   - **优化循环：** 该过程迭代地从数据集中采样提示 ppp。当前的 RL 策略（πRL\pi\_{RL}πRL​）生成一个响应 yyy。奖励模型将奖励 r=rθ(p,y)r = r\_{\theta}(p, y)r=rθ​(p,y) 分配给生成的响应。然后，PPO 算法使用此奖励信号更新策略 πRL\pi\_{RL}πRL​ 的权重 (weight)。
   - **KL 散度惩罚：** RL 微调中的一个重要问题是，策略可能会学习生成最大化奖励模型分数但显著偏离原始 SFT 模型分布的输出，这可能导致无意义或重复的文本（“奖励作弊”）。为了缓解这种情况，RLHF 中的 PPO 在奖励函数或目标中加入一个 Kullback-Leibler (KL) 散度惩罚项。目标是在最大化奖励模型分数的同时，与初始 SFT 策略保持接近：
     目标=E(p,y)∼πRL[rθ(p,y)−β⋅KL(πRL(⋅∣p)∣∣πSFT(⋅∣p))]\text{目标} = \mathbb{E}\_{(p,y) \sim \pi\_{RL}} [r\_{\theta}(p, y) - \beta \cdot \text{KL}(\pi\_{RL}(\cdot|p) || \pi\_{SFT}(\cdot|p))]目标=E(p,y)∼πRL​​[rθ​(p,y)−β⋅KL(πRL​(⋅∣p)∣∣πSFT​(⋅∣p))]
     项 KL(πRL(⋅∣p)∣∣πSFT(⋅∣p))\text{KL}(\pi\_{RL}(\cdot|p) || \pi\_{SFT}(\cdot|p))KL(πRL​(⋅∣p)∣∣πSFT​(⋅∣p)) 衡量了 RL 策略和原始 SFT 策略对提示 ppp 预测的令牌分布之间的散度。超参数 (hyperparameter) β\betaβ 控制此惩罚的强度，防止 RL 策略偏离 SFT 模型学习到的分布过远，从而保持连贯性和语言质量。

以下图表展示了这一多阶段过程：

RLHF\_Pipeline

cluster₀

阶段 1: SFT

cluster₁

阶段 2: 奖励模型训练

cluster₂

阶段 3: RL 微调

SFT

预训练语言模型
+
指令数据
-> 监督微调

GenData

采样提示
生成响应 (来自 SFT 模型)

SFT->GenData

用于
生成

RLPolicy

用 SFT 模型
初始化策略
(π\_SFT)

SFT->RLPolicy

初始化

PPO

使用 PPO 微调策略
利用 RM 奖励 + KL 惩罚

SFT->PPO

KL 参考
(π\_SFT)

HumanPref

收集人类
偏好数据
(y\_w 对比 yₗ)

GenData->HumanPref

标注

TrainRM

训练奖励模型 (RM)
以预测偏好

HumanPref->TrainRM

训练数据

TrainRM->PPO

提供奖励
(r\_θ)

RLPolicy->PPO

更新

FinalModel

对齐的 LLM
(π\_RL)

PPO->FinalModel

输出

> 标准 RLHF 流程的三个阶段：监督微调 (SFT)、基于人类偏好的奖励模型 (RM) 训练，以及由奖励模型和针对原始 SFT 模型的 KL 惩罚引导的强化学习 (PPO) 微调。

这个流程使模型能够从比较性人类反馈中学习，改进其生成与期望属性更一致的响应的能力，比仅通过 SFT 达到的效果更好。接下来的部分将详细说明收集偏好数据、训练奖励模型以及实施 PPO 微调步骤的实际方面。

获取即时帮助、个性化解释和交互式代码示例。

---

### 收集人类偏好数据

# 收集人类偏好数据

监督微调 (fine-tuning) (SFT) 是一个初始阶段，在此阶段语言模型学习遵循指令或特定格式。然而，仅仅SFT往往无法完全体现人类偏好在有用性、诚实性和无害性方面的细节。仅仅模仿演示数据，并不能保证与这些复杂且常带有主观色彩的价值观保持一致。为弥补这一不足，人类反馈强化学习 (reinforcement learning) (RLHF) 引入了一种将人类判断直接纳入模型优化过程的机制。RLHF 根本在于高质量人类偏好数据的收集，这些数据作为后续奖励模型 (RM) 的训练信号。获取这种重要偏好数据的过程有详细说明。

### 目标：获取相对偏好

主要设想并非要求人类写出“好”的回复（这更接近SFT数据生成方式），也不是给出绝对分数（这可能在不同标注员和提示之间存在不一致）。而是采用收集成对比较的标准方法。对于一个特定的输入提示，模型会生成两个或多个候选回复。然后，人类标注员被要求根据预设标准（例如，有用性、连贯性、安全性）选择他们更喜欢的回复。这种相对判断比绝对评分更可靠，也更容易让人类持续保持一致性。

### 生成提示和回复

这一过程始于一组多样化的提示。这些提示应尽可能地与最终对齐 (alignment)模型需要处理的输入类型相似。提示的来源可以包括：

- 从过往交互中记录的真实用户查询（如果可用且符合隐私规定）。
- 旨在覆盖特定主题或行为的合成生成提示。
- 从用于评估或SFT的现有数据集中提取的提示。
- 标注团队专门编写的提示，旨在处理已知模型缺陷或获取所需能力。

需要确保提示在长度、主题、复杂性和交互方式上的多样性（例如，问答、摘要、创意写作、对话轮次）。

选定提示后，SFT 模型（即上一个微调 (fine-tuning)阶段的模型）被用来为该提示生成多个候选回复。通常，对于给定的提示 xxx，会生成至少两个回复 (y1,y2y\_1, y\_2y1​,y2​)。改变解码温度或使用 top-k/top-p 采样等技术可以增加生成回复之间的差异性。

G

Prompt

输入提示 (x)

SFT\_Model

SFT 模型

Prompt->SFT\_Model

Response1

回复 y1

SFT\_Model->Response1

 样本 1

Response2

回复 y2

SFT\_Model->Response2

 样本 2

Human

人类标注员

Response1->Human

Response2->Human

Preference

偏好对
(x, 选定回复 y\_chosen, 拒绝回复 yᵣejected)

Human->Preference

> 生成单个偏好数据点的主要流程。

### 标注界面和任务

人类标注员会看到提示 xxx 以及两个生成的回复 y1y\_1y1​ 和 y2y\_2y2​。界面应清晰地并排显示提示和两个回复。为减少呈现偏见，每次比较任务中回复的顺序（左侧对比右侧，上方对比下方）都应随机化。

标注员会获得具体的指令和选择标准。这些标准通常围绕以下几点：

- **有用性：** 回复是否准确、完整地回应了提示？
- **诚实性/准确性：** 所提供的信息是否真实且不具误导性？
- **无害性：** 回复是否避免了有害、带有偏见或不当的内容？
- **连贯性/可读性：** 回复是否书写良好且易于理解？

标注员根据这些标准选择他们认为更优的回复 (ychoseny\_chosenyc​hosen)。另一个回复则成为被拒绝的回复 (yrejectedy\_{rejected}yrejected​)。界面也可能允许标注员表明两个回复是否同样好或坏，或者两者都不可接受。一些设置可能还会要求对偏好提供自由文本解释，这对于质量控制和理解失败情况很有帮助，尽管这会增加标注成本。

### 偏好数据结构

这一过程的成果是一个偏好元组数据集。每个元组通常包含提示、偏好的（选定的）回复和不偏好的（被拒绝的）回复。这种结构直接用于训练奖励模型。

以下是一个单数据点在 Python 中如何表示的简化示例：

```python
preference_data_point = {
    "prompt": "用简单的话语解释梯度下降。",
    "chosen_response": (
        "想象你蒙着眼睛站在一座山上，想走到山底。 "
        "梯度下降就像是你凭脚感，朝着最陡峭的下坡方向迈小步。 "
        "你会一直走，直到无法再往下。"
    ),
    "rejected_response": (
        "梯度下降利用导数寻找函数的最小值。 "
        "它计算梯度并更新参数。"
    ),
    "labeler_id": "annotator_123",
    "timestamp": "2023-10-27T10:30:00Z",

    "reasoning": (
        "选定回复使用了有帮助的类比，使其更易懂。"
    ),
    "model_version": "sft_model_v1.2"
}
```

### 确保数据质量和规模

RLHF 流程的运行效果很大程度上依赖于偏好数据的质量和数量。

- **标注员培训和一致性：** 标注员需要接受关于指导原则和标准的充分培训。定期校准会议和监控标注员间一致性 (IAA) 有助于保证一致性。意见分歧应进行裁决，必要时修订指导原则。
- **规模：** 训练有效的奖励模型通常需要大量的偏好数据，数量常从数万到数十万个已标注对不等，这取决于目标行为的复杂性和提示的多样性。
- **提示多样性：** 如前所述，提示必须涵盖多方面主题和风格，以确保生成的奖励模型具有良好的泛化能力。将RM过度拟合到一小部分提示可能导致RL微调 (fine-tuning)阶段表现不佳。

用户查询合成生成数据集提取内部团队010203040提示来源分布 (%)

> 收集偏好数据时使用的提示来源示例分布。

### 偏好数据收集中的挑战

收集人类偏好数据是一项重要工作，包含多项固有挑战：

- **成本与时间：** 人工标注费用高昂且耗时，尤其对于大型模型所需的规模更是如此。
- **主观性：** 即使有明确的指导原则，人类偏好也可能带有主观性，导致标注员之间出现分歧。达成共识或理解偏好分布并非易事。
- **标注员偏见：** 标注员可能带入自身的隐性偏见，如果不通过多样化的标注员招募和偏见培训来妥善管理，这些偏见可能会不经意地编码到奖励模型中。
- **任务复杂性：** 评估复杂的输出（例如，长文本 (long context)、代码生成、多轮对话）对标注员来说可能需要较高的认知负荷，从而可能影响质量和效率。
- **指导原则演变：** 随着模型能力的改变或新问题的出现，标注指导原则通常需要修订，这要求对标注员进行再培训，并可能需要重新标注部分数据。

尽管存在这些挑战，收集高质量的偏好数据仍然是RLHF流程的重要支撑。该数据集直接决定了用于引导语言模型行为符合人类价值观的奖励信号。下一步则是利用这些收集到的数据来训练奖励模型本身。

获取即时帮助、个性化解释和交互式代码示例。

---

### 训练奖励模型 (RM)

# 训练奖励模型 (RM)

训练奖励模型 (RM) 是强化学习 (reinforcement learning)与人类反馈 (RLHF) 的一个核心过程，它利用人类偏好数据。这些数据通常以成对比较的形式出现，包含一个提示、一个选择的回答和一个拒绝的回答。奖励模型 (RM) 的作用是学习一个函数，将提示和可能的回答映射为一个标量值。这个值表示该回答与给定提示下人类偏好的一致程度。这个学得的奖励函数随后将作为指导信号，用于使用强化学习对语言模型进行微调 (fine-tuning)。

### 奖励模型架构

奖励模型的架构通常与正在对其进行校准的基础语言模型密切相关。一种常见做法是直接使用经过监督微调 (fine-tuning) (SFT) 的模型，并修改其最终层。RM 的头部不再预测下一个词元 (token)的分布，而是被调整为输出一个单一的标量值。

具体而言，RM 的输入是提示 (xxx) 和候选回答 (yyy) 的拼接。这个组合序列由 Transformer 架构处理。序列中最后一个词元（通常是序列结束词元）对应的隐藏状态随后通过一个线性层（奖励头部）来生成标量奖励分数。

以 SFT 模型作为 RM 的基础具有显著益处：

1. **初始化：** RM 从已通过 SFT 适应目标领域和风格的权重 (weight)开始。
2. **理解能力：** 模型本身就理解与任务相关的语言生成和结构特性。
3. **效率：** 它避免了从头开始训练一个完全独立的模型，可能节省计算资源。

令 rθ(x,y)r\_\theta(x, y)rθ​(x,y) 表示参数 (parameter)为 θ\thetaθ 的 RM 对提示 xxx 和回答 yyy 输出的标量奖励。

### 训练目标与损失函数 (loss function)

RM 是在收集到的偏好数据集 D={(x(i),yc(i),yr(i))}i=1ND = \{(x^{(i)}, y\_c^{(i)}, y\_r^{(i)})\}\_{i=1}^ND={(x(i),yc(i)​,yr(i)​)}i=1N​ 上训练的，其中 ycy\_cyc​ 是人类偏好的回答（被选择的），yry\_ryr​ 是对提示 xxx 而言被认为较不偏好的回答（被拒绝的）。训练目标是使 RM 对同一提示下被选择的回答赋予比被拒绝的回答更高的分数：

rθ(x,yc)>rθ(x,yr)r\_\theta(x, y\_c) > r\_\theta(x, y\_r)rθ​(x,yc​)>rθ​(x,yr​)

这通常被视为一个回答对的二分类问题。一种常见的方法是调整 Bradley-Terry 模型，该模型对 ycy\_cyc​ 比 yry\_ryr​ 更受偏好的概率进行建模。这个概率可以通过它们的奖励分数之差经过逻辑 S 型函数 σ(z)=1/(1+e−z)\sigma(z) = 1 / (1 + e^{-z})σ(z)=1/(1+e−z) 来建模：

P(yc≻yr∣x)=σ(rθ(x,yc)−rθ(x,yr))P(y\_c \succ y\_r | x) = \sigma(r\_\theta(x, y\_c) - r\_\theta(x, y\_r))P(yc​≻yr​∣x)=σ(rθ​(x,yc​)−rθ​(x,yr​))

RM 通过最小化数据集 DDD 中人类偏好的负对数似然来训练。损失函数如下：

L(θ)=−E(x,yc,yr)∼D[log⁡(σ(rθ(x,yc)−rθ(x,yr)))]\mathcal{L}(\theta) = - \mathbb{E}\_{(x, y\_c, y\_r) \sim D} \left[ \log \left( \sigma(r\_\theta(x, y\_c) - r\_\theta(x, y\_r)) \right) \right]L(θ)=−E(x,yc​,yr​)∼D​[log(σ(rθ​(x,yc​)−rθ​(x,yr​)))]

这个损失函数促使差值 rθ(x,yc)−rθ(x,yr)r\_\theta(x, y\_c) - r\_\theta(x, y\_r)rθ​(x,yc​)−rθ​(x,yr​) 变得大且为正，从而有效地最大化根据人类标注正确分类偏好回答的概率。有时可以添加一个边际项，但这种基本形式被广泛使用。

### 训练过程

训练期间，每个数据点 (x,yc,yr)(x, y\_c, y\_r)(x,yc​,yr​) 需要对 RM 进行两次前向传播：一次是提示与选择的回答 (x⊕ycx \oplus y\_cx⊕yc​) 拼接，另一次是提示与拒绝的回答 (x⊕yrx \oplus y\_rx⊕yr​) 拼接。

1. **输入准备：** 将提示 xxx 与选择的回答 ycy\_cyc​ 和拒绝的回答 yry\_ryr​ 拼接。确保适当的词元 (token)化和注意力掩码，将提示和回答在每次传播中作为一个单一序列处理。
2. **前向传播：** 计算标量奖励 rc=rθ(x,yc)r\_c = r\_\theta(x, y\_c)rc​=rθ​(x,yc​) 和 rr=rθ(x,yr)r\_r = r\_\theta(x, y\_r)rr​=rθ​(x,yr​)。
3. **损失计算：** 使用上述公式计算成对排序损失。
4. **优化：** 计算损失相对于 RM 参数 (parameter) θ\thetaθ 的梯度，并使用 AdamW 等优化器更新参数。

以下是一个简化的 PyTorch 代码片段，说明了训练步骤中的损失计算：

```python
import torch
import torch.nn.functional as F

def compute_rm_loss(reward_model, tokenizer, batch):
    """计算一批偏好数据的成对排序损失。"""

    prompts, chosen_responses, rejected_responses = batch

    chosen_inputs = tokenizer(
        [p + c for p, c in zip(prompts, chosen_responses)],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )

    chosen_inputs = {
        k: v.to(reward_model.device) for k, v in chosen_inputs.items()
    }

    rejected_inputs = tokenizer(
        [p + r for p, r in zip(prompts, rejected_responses)],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )
    rejected_inputs = {
        k: v.to(reward_model.device) for k, v in rejected_inputs.items()
    }

    chosen_rewards = reward_model(**chosen_inputs).rewards

    rejected_rewards = reward_model(**rejected_inputs).rewards

    loss = -F.logsigmoid(
        chosen_rewards - rejected_rewards
    ).mean()

    return loss
```

### 评估

在 RL 阶段使用 RM 之前，评估其性能是很重要的。主要衡量标准是在保留偏好对集上的**准确率**。这衡量了 RM 正确预测人类偏好回答 (rc>rrr\_c > r\_rrc​>rr​) 的频率。准确率通常在 65% 到 80% 之间，具体取决于任务难度、数据质量和模型容量。

G

cluster\_data

偏好数据

clusterᵣm

奖励模型训练

D

{ (x, yc, yr), ... }

RM

奖励模型 (r\_θ)

D->RM

提供训练对

Loss

成对排序损失
 -log(σ(rc - rr))

RM->Loss

计算 rc, rr

Opt

优化器
(例如 AdamW)

Loss->Opt

计算梯度

Opt->RM

更新 θ

> 使用偏好数据和成对排序损失训练奖励模型的基本流程。

除了准确率，定性分析也很有用。检查 RM 与人类判断强烈一致或不一致的案例，可以显示模型中的偏差或不足。检查奖励分数是否与回答长度、连贯性或实用性等其他易于理解的质量指标相关联也很有帮助，尽管这些关联可能很弱。

### 挑战与注意事项

- **校准：** 奖励的绝对值通常不直接具有意义或校准良好。训练目标只侧重于 rc−rrr\_c - r\_rrc​−rr​ 的差值。在 RL 训练期间对奖励分数进行归一化 (normalization)（例如白化）是一种常见做法。
- **过拟合 (overfitting)：** RM 可能会过拟合训练数据中的特定偏好，潜在地学习到标注过程中存在的虚假关联或偏差。正则化 (regularization)和使用足够多样化的数据集是很重要的。
- **数据质量依赖：** RM 的质量根本上受到人类偏好数据质量和一致性的限制。有噪声或有偏差的标注将导致次优的 RM。
- **可扩展性：** 训练 RM 需要大量计算，与微调 (fine-tuning)基础 LLM 本身相当。

成功训练奖励模型是 RLHF 流程中的重要一步。一个训练良好的 RM 提供了必要的信号，以在后续强化学习 (reinforcement learning)阶段指导 LLM，使其生成更符合人类期望特性（如有用性、诚实性和无害性）的输出。

获取即时帮助、个性化解释和交互式代码示例。

---

### 近端策略优化 (PPO) 介绍

# 近端策略优化 (PPO) 介绍

监督微调 (fine-tuning) (SFT) 模型通常需要使用人类偏好进行进一步优化。这些偏好通常通过奖励模型 (RM) 获取。然而，简单的监督更新不能直接处理来自此类奖励模型的标量奖励信号。因此，强化学习 (reinforcement learning) (RL) 算法被用于此优化过程。在各种强化学习方法中，近端策略优化 (PPO) 已成为在强化学习人类反馈 (RLHF) 流程中微调大型语言模型的一种流行且有效的方法。

那么，PPO 是什么，为何它在此处适用呢？PPO 属于强化学习中的策略梯度方法。策略梯度方法的核心思想是直接调整策略（在本例中是语言模型 πθ\pi\_\thetaπθ​）的参数 (parameter) θ\thetaθ，以使预期奖励最大化。我们估计预期奖励的梯度并朝该方向前进。然而，简单的策略梯度实现可能不稳定。一次更新步长如果对策略的改变过于剧烈，可能导致性能大幅下降，并可能难以恢复。这对于大型复杂模型（如 LLM）来说风险尤其高。

PPO 通过限制策略在每个更新步骤中的变化幅度来解决这个稳定性问题。它通过一个特定的目标函数来实现这一点，该函数阻止策略大幅偏离之前的策略，同时仍鼓励基于奖励信号的改进。

### PPO 目标函数

PPO 的核心是优化一个替代目标函数。最常见的变体使用裁剪目标。我们先定义一些术语：

- **策略 πθ(at∣st)\pi\_\theta(a\_t|s\_t)πθ​(at​∣st​)：** 我们的语言模型，由 θ\thetaθ 参数 (parameter)化。给定状态 sts\_tst​（输入提示和先前生成的令牌），它输出下一个可能的令牌 ata\_tat​ 的概率分布。
- **旧策略 πθold(at∣st)\pi\_{\theta\_{old}}(a\_t|s\_t)πθold​​(at​∣st​)：** 当前更新迭代之前的策略。我们使用此策略采样轨迹（令牌序列）。
- **优势函数 A^t\hat{A}\_tA^t​：** 估计在状态 sts\_tst​ 中执行动作 ata\_tat​ 相对于当前策略下平均动作的好处程度。它通常使用来自我们 RM 的奖励和一个学习到的价值函数 V(st)V(s\_t)V(st​)（评论员）来计算，常使用广义优势估计 (GAE) 等技术。正优势值表明所采取的动作好于预期，而负优势值则表明它更差。
- **概率比 rt(θ)r\_t(\theta)rt​(θ)：** 这衡量了在新策略 πθ\pi\_\thetaπθ​ 和旧策略 πθold\pi\_{\theta\_{old}}πθold​​ 之间，在状态 sts\_tst​ 中执行动作 ata\_tat​ 的概率变化。
  rt(θ)=πθ(at∣st)πθold(at∣st)r\_t(\theta) = \frac{\pi\_\theta(a\_t|s\_t)}{\pi\_{\theta\_{old}}(a\_t|s\_t)}rt​(θ)=πθold​​(at​∣st​)πθ​(at​∣st​)​

裁剪后的替代目标函数，通常表示为 LCLIP(θ)L^{CLIP}(\theta)LCLIP(θ)，公式如下：

LCLIP(θ)=E^t[min⁡(rt(θ)A^t,clip(rt(θ),1−ϵ,1+ϵ)A^t)]L^{CLIP}(\theta) = \hat{\mathbb{E}}\_t [ \min(r\_t(\theta) \hat{A}\_t, \text{clip}(r\_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}\_t) ]LCLIP(θ)=E^t​[min(rt​(θ)A^t​,clip(rt​(θ),1−ϵ,1+ϵ)A^t​)]

我们来分解一下：

1. **E^t[...]\hat{\mathbb{E}}\_t [...] E^t​[...]：** 这表示我们正在对从与环境的交互中（即生成文本序列并从 RM 获取奖励）收集的一批时间步取平均值。
2. **rt(θ)A^tr\_t(\theta) \hat{A}\_trt​(θ)A^t​：** 这是标准的策略梯度目标。如果优势 A^t\hat{A}\_tA^t​ 为正，我们希望增加执行动作 ata\_tat​ 的概率，因此我们增加 rt(θ)r\_t(\theta)rt​(θ)。如果 A^t\hat{A}\_tA^t​ 为负，我们希望降低概率，因此我们降低 rt(θ)r\_t(\theta)rt​(θ)。
3. **clip(rt(θ),1−ϵ,1+ϵ)\text{clip}(r\_t(\theta), 1-\epsilon, 1+\epsilon)clip(rt​(θ),1−ϵ,1+ϵ)：** 此函数将概率比 rt(θ)r\_t(\theta)rt​(θ) 钳制在 [1−ϵ,1+ϵ][1-\epsilon, 1+\epsilon][1−ϵ,1+ϵ] 范围内。超参数 (hyperparameter) ϵ\epsilonϵ (epsilon) 通常是一个小值，例如 0.1 或 0.2。它定义了旧策略周围的信任区域。
4. **clip(rt(θ),1−ϵ,1+ϵ)A^t\text{clip}(r\_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}\_tclip(rt​(θ),1−ϵ,1+ϵ)A^t​：** 这是目标项的裁剪版本。
5. **min⁡(...)\min(...)min(...)：** 最小值操作符是核心部分。
   - 如果 A^t>0\hat{A}\_t > 0A^t​>0（动作是好的）：目标变为 min⁡(rt(θ)A^t,(1+ϵ)A^t)\min(r\_t(\theta) \hat{A}\_t, (1+\epsilon) \hat{A}\_t)min(rt​(θ)A^t​,(1+ϵ)A^t​)。这意味着如果 rt(θ)r\_t(\theta)rt​(θ) 增加超过 1+ϵ1+\epsilon1+ϵ，目标将受到惩罚。我们限制了一步中可以增加一个好动作概率的幅度。
   - 如果 A^t<0\hat{A}\_t < 0A^t​<0（动作是坏的）：目标变为 min⁡(rt(θ)A^t,(1−ϵ)A^t)\min(r\_t(\theta) \hat{A}\_t, (1-\epsilon) \hat{A}\_t)min(rt​(θ)A^t​,(1−ϵ)A^t​)。由于 A^t\hat{A}\_tA^t​ 是负数，这简化为 max⁡(rt(θ)A^t,(1−ϵ)A^t)\max(r\_t(\theta) \hat{A}\_t, (1-\epsilon) \hat{A}\_t)max(rt​(θ)A^t​,(1−ϵ)A^t​)。如果 rt(θ)r\_t(\theta)rt​(θ) 降低到 1−ϵ1-\epsilon1−ϵ 以下，目标将受到惩罚。我们限制了一步中可以降低一个坏动作概率的幅度。

本质上，裁剪消除了策略发生剧烈变化的动机，从而防止了大的、可能破坏稳定的更新。策略被鼓励改进（rt(θ)r\_t(\theta)rt​(θ) 朝着 A^t\hat{A}\_tA^t​ 所偏好的方向移动），但仅限于由 ϵ\epsilonϵ 设定的界限内。

### Actor-Critic 实现

PPO 通常使用 Actor-Critic 架构实现。

- **Actor（执行者）：** 策略网络 πθ\pi\_\thetaπθ​（我们的 LLM），它决定采取哪个动作（令牌）。
- **Critic（评论员）：** 价值函数网络 Vϕ(st)V\_\phi(s\_t)Vϕ​(st​)（通常与 Actor 共享下层），它估计给定状态 sts\_tst​ 的预期回报（累积未来奖励）。此价值估计用于计算优势 A^t\hat{A}\_tA^t​。

整体优化涉及最大化 LCLIPL^{CLIP}LCLIP 目标（针对 Actor），同时最小化价值函数（Critic）的损失函数 (loss function)，通常是预测值 Vϕ(st)V\_\phi(s\_t)Vϕ​(st​) 与实际观测回报之间的均方误差。目标中还可以添加一个可选的熵奖励项，以鼓励探索。

G

clusterₑnv

环境 (交互循环)

clusterₚpo

PPO 更新

S

状态 (提示 + 历史)

Actor

Actor (策略 πθ - LLM)

S->Actor

获取状态

Critic

Critic (价值 Vφ)

S->Critic

获取状态

A

动作 (下一个令牌)

A->S

更新状态

RM

奖励模型

A->RM

评估动作

R

奖励

RM->R

获取奖励

Adv

优势计算 (Ât)

R->Adv

Actor->A

生成令牌

ClipObj

裁剪目标

Actor->ClipObj

策略比 rt(θ)

Critic->Adv

价值估计

Adv->Critic

更新价值函数

Adv->ClipObj

ClipObj->Actor

更新策略

> PPO Actor-Critic 在 RLHF 中循环的简要概述。Actor（LLM）生成文本，奖励模型提供奖励，Critic 估计状态价值。PPO 使用这些组件计算优势，并通过裁剪目标更新 Actor 和 Critic。

### 为什么 LLM 对齐 (alignment)使用 PPO？

与其他强化学习 (reinforcement learning)算法相比，PPO 取得了良好的平衡：

1. **稳定性：** 裁剪目标函数提供比简单策略梯度方法更稳定的训练更新，这对于训练成本高且易发散的大型模型来说很重要。
2. **样本效率：** 尽管在较简单场景下可能不如一些离策略方法样本效率高，但 PPO 通常被认为比 REINFORCE 等基本策略梯度方法更有效。它在每个数据收集阶段内重复使用在多个训练周期中收集的数据。
3. **实现复杂度：** 与信赖域策略优化 (TRPO) 等涉及二阶优化的一些替代方案相比，PPO 的实现和调整复杂性较低。

在 RLHF 的背景下，PPO 允许我们有效使用来自 RM 的标量奖励信号，指导 LLM 生成更符合人类偏好的输出，同时减轻在微调 (fine-tuning)过程中破坏模型稳定性的风险。下一节将详细介绍使用 PPO 实现此强化学习微调阶段的具体内容。

获取即时帮助、个性化解释和交互式代码示例。

---

### RL PPO 微调

# RL PPO 微调

强化学习 (reinforcement learning) (RL) 结合了监督微调 (fine-tuning) (SFT) 模型和奖励模型 (RM)。SFT 模型为遵循指令提供了坚实基础，RM 经过训练可以根据人类偏好对响应进行评分。这种结合旨在改进 SFT 模型，优化其生成能最大化 RM 评分的输出。目标是有效地引导模型产生人类偏好的行为。近端策略优化 (PPO) 是大多数强化学习人类反馈 (RLHF) 流程中此过程的主要算法，因为它相较于其他 RL 算法具有相对稳定性和样本效率。

### 大语言模型 (LLM)微调 (fine-tuning)中的强化学习 (reinforcement learning)框架

让我们将标准强化学习术语映射到大语言模型语境：

1. **智能体/策略 (πθ\pi\_{\theta}πθ​):** 这是我们正在积极微调的语言模型。它最初是 SFT 模型的副本，其参数 (parameter) θ\thetaθ 在 PPO 过程中更新。策略 πθ(a∣s)\pi\_{\theta}(a|s)πθ​(a∣s) 定义了在给定当前词元 (token)序列 sss 的情况下生成下一个词元 aaa 的概率。
2. **动作 (aaa):** 动作对应于从模型的词表中选择下一个词元并将其附加到序列中。
3. **状态 (sss):** 状态表示到目前为止已生成的词元序列，起始于初始提示。
4. **环境:** 环境是隐式定义的。它接收模型响应提示（初始状态）生成的序列（动作序列），并返回奖励。核心组成部分是提示分布和奖励函数（我们的 RM + KL 惩罚）。
5. **奖励函数 (R(s,a)R(s, a)R(s,a) 或 R(sequence)R(sequence)R(sequence)):** 这是引导优化的重要信号。在 RLHF 中，一个完整生成序列（提示 + 响应）的奖励主要由训练好的奖励模型 (RMRMRM) 的评分决定。为了保持语言连贯性并防止策略与可靠的 SFT 模型偏离过远，RM 评分与基于当前策略 πθ\pi\_{\theta}πθ​ 和初始 SFT 策略 πref\pi\_{ref}πref​ 之间 Kullback-Leibler (KL) 散度的惩罚项相结合。

### PPO 迭代周期

PPO 微调 (fine-tuning)过程是迭代进行的，通常在每次迭代中包含以下步骤：

1. **数据生成 (Rollout):** 采样一批提示（例如，来自 SFT 训练分布或单独的提示数据集）。对于每个提示，使用*当前*策略模型 πθ\pi\_{\theta}πθ​ 生成响应。这涉及自回归 (autoregressive)地采样词元 (token)，直到生成序列结束词元或达到最大长度。在此生成过程中，我们需要为每个词元步长 ttt 存储几项信息：

   - 状态 sts\_tst​（提示 + 截至步长 ttt 生成的词元）。
   - 动作 ata\_tat​（在步长 ttt 生成的词元）。
   - 在当前策略下生成该词元的对数概率：log⁡πθ(at∣st)\log \pi\_{\theta}(a\_t|s\_t)logπθ​(at​∣st​)。
   - 在*参考* SFT 策略下生成该词元的对数概率：log⁡πref(at∣st)\log \pi\_{ref}(a\_t|s\_t)logπref​(at​∣st​)。参考策略 πref\pi\_{ref}πref​ 在整个 RL 阶段保持固定，通常是初始 SFT 模型状态。
2. **奖励计算:** 一旦生成完整序列（提示 + 响应），计算奖励。这通常包含：

   - 使用训练好的奖励模型 (RMRMRM) 对最终序列评分。我们称之为 RRMR\_{RM}RRM​。此评分反映了基于人类偏好的对齐 (alignment)质量。
   - 为每个步长 ttt 计算每词元 KL 散度惩罚：RKLt=−β(log⁡πθ(at∣st)−log⁡πref(at∣st))R\_{KL\_t} = -\beta (\log \pi\_{\theta}(a\_t|s\_t) - \log \pi\_{ref}(a\_t|s\_t))RKLt​​=−β(logπθ​(at​∣st​)−logπref​(at​∣st​))。超参数 (parameter) (hyperparameter) β\betaβ 控制此惩罚的强度。较高的 β\betaβ 值会阻止模型偏离 SFT 模型。
   - 将这些组合成最终的奖励信号。一种常见做法是仅在最后一个词元步长分配 RRMR\_{RM}RRM​ 评分，并在每个步长添加每词元 RKLR\_{KL}RKL​。因此，一个序列的总奖励可能看起来像一个每词元 KL 惩罚序列，其中 RM 评分被添加到最后一个词元的奖励中。

   让我们在 PyTorch 中演示对数概率和 KL 惩罚部分的计算：

   ```python
   import torch
   import torch.nn.functional as F

   input_ids = torch.cat([prompt_tokens, generated_tokens], dim=-1)
   attention_mask = (input_ids != pad_token_id).long()

   with torch.no_grad():
       policy_outputs = policy_model(input_ids=input_ids, attention_mask=attention_mask)
       ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)

   policy_logits = policy_outputs.logits
   ref_logits = ref_outputs.logits

   gen_len = generated_tokens.size(1)
   policy_log_probs = F.log_softmax(policy_logits[:, -gen_len-1:-1, :], dim=-1)
   ref_log_probs = F.log_softmax(ref_logits[:, -gen_len-1:-1, :], dim=-1)

   gathered_policy_log_probs = policy_log_probs.gather(dim=-1, index=generated_tokens.unsqueeze(-1)).squeeze(-1)
   gathered_ref_log_probs = ref_log_probs.gather(dim=-1, index=generated_tokens.unsqueeze(-1)).squeeze(-1)

   log_ratio = gathered_policy_log_probs - gathered_ref_log_probs
   kl_penalty_per_token = -beta * log_ratio

   rewards = kl_penalty_per_token.clone()
   rewards[:, -1] += rm_scores
   ```

   > PyTorch 代码片段，演示从策略模型和参考模型计算对数概率以及由此产生的 KL 惩罚项。`beta` 是一个重要的超参数。
3. **优化 (PPO 更新):** 使用收集到的轨迹（状态、动作、对数概率、奖励）来更新策略模型 πθ\pi\_{\theta}πθ​。PPO 通过优化一个替代目标函数来实现这一点。核心思路是最大化预期优势 At=Q(st,at)−V(st)A\_t = Q(s\_t, a\_t) - V(s\_t)At​=Q(st​,at​)−V(st​)，它表示动作 ata\_tat​ 相较于状态 sts\_tst​ 的平均动作有多大改进。PPO 使用带截断目标的的重要性采样来确保更新的稳定性：

   - **优势估计:** 计算每个步长的优势估计值 A^t\hat{A}\_tA^t​。虽然朴素策略梯度使用回报 GtG\_tGt​（未来奖励的总和），但 PPO 常使用广义优势估计 (GAE) 来降低方差。在许多 RLHF 实现中，RM 评分本身（可能经过归一化 (normalization)或与学习到的价值函数 V(st)V(s\_t)V(st​) 结合）用作 Q(st,at)Q(s\_t, a\_t)Q(st​,at​) 或优势的代理。一个单独的价值模型，通常从 RM 或 SFT 模型初始化并训练以预测预期累积奖励（包括 KL 惩罚），常用于计算 V(st)V(s\_t)V(st​) 和改进优势估计。
   - **策略更新:** 计算概率比率 rt(θ)=πθ(at∣st)πθold(at∣st)=exp⁡(log⁡πθ(at∣st)−log⁡πθold(at∣st))r\_t(\theta) = \frac{\pi\_{\theta}(a\_t|s\_t)}{\pi\_{\theta\_{old}}(a\_t|s\_t)} = \exp(\log \pi\_{\theta}(a\_t|s\_t) - \log \pi\_{\theta\_{old}}(a\_t|s\_t))rt​(θ)=πθold​​(at​∣st​)πθ​(at​∣st​)​=exp(logπθ​(at​∣st​)−logπθold​​(at​∣st​))，其中 πθold\pi\_{\theta\_{old}}πθold​​ 是更新前的策略（来自数据生成阶段）。截断替代目标函数为：
     LCLIP(θ)=E^t[min⁡(rt(θ)A^t,截断(rt(θ),1−ϵ,1+ϵ)A^t)]L^{CLIP}(\theta) = \hat{\mathbb{E}}\_t \left[ \min(r\_t(\theta) \hat{A}\_t, \text{截断}(r\_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}\_t) \right]LCLIP(θ)=E^t​[min(rt​(θ)A^t​,截断(rt​(θ),1−ϵ,1+ϵ)A^t​)]
     截断\text{截断}截断 函数将比率 rt(θ)r\_t(\theta)rt​(θ) 限制在 [1−ϵ,1+ϵ][1-\epsilon, 1+\epsilon][1−ϵ,1+ϵ] 区间内，防止过大的更新导致训练不稳定。ϵ\epsilonϵ 是一个小的超参数（例如 0.2）。
   - 使用收集到的批次数据对该目标进行多轮随机梯度上升。更新策略模型的参数 θ\thetaθ。如果使用单独的价值模型，它通常通过最小化其预测与观察到的回报之间的均方误差进行同步更新。

这种数据生成、奖励计算和 PPO 优化的循环重复进行，逐步调整策略 πθ\pi\_{\theta}πθ​ 以生成从 RM 获得更高评分的响应，同时 KL 惩罚使其保持在 SFT 阶段学到的流畅性和能力上。

### 实际考量

- **价值函数:** 如前所述，在 PPO 中，与策略同时训练一个单独的价值函数是标准做法，以降低优势估计的方差。这个价值函数预测给定状态（词元 (token)序列）的预期折扣未来奖励。它通常通过最小化其预测与数据生成中观察到的回报之间的平方误差进行训练。
- **超参数 (parameter) (hyperparameter)调整:** RLHF，特别是 PPO，包含几个敏感的超参数：KL 系数 β\betaβ、PPO 截断参数 ϵ\epsilonϵ、策略模型和价值模型的学习率、每次数据生成的 PPO 迭代次数、批次大小以及 GAE 参数 (λ\lambdaλ, γ\gammaγ)。找到正确的平衡通常是经验性的且计算密集。
- **计算成本:** RLHF 对计算要求高。它需要维护多个模型（策略、参考、RM，可能还有一个价值模型）、进行数据生成（推理 (inference)）、计算奖励（推理）以及执行 PPO 更新（训练）。像 DeepSpeed 这样的框架和像 `trl` (Transformer Reinforcement Learning) 这样的库提供了优化的实现来管理这种复杂性。

通过精心实现这种基于 PPO 的微调 (fine-tuning)循环，我们能够有效地运用奖励模型中捕获的信号，使语言模型的行为更贴近人类期望的特性，例如有益性和无害性，这建立在监督微调奠定的基础上。

获取即时帮助、个性化解释和交互式代码示例。

---

### KL散度惩罚的作用

# KL散度惩罚的作用

尽管近端策略优化（PPO）为我们基于训练好的奖励模型（RM）的奖励信号更新语言模型提供了一种方法，但简单地应用它可能导致不好的结果。RL策略（πRL\pi\_{RL}πRL​）可能会找到一些序列，这些序列从RM那里获得高分，但却不自然、重复、无意义，或风格与原始监督微调 (fine-tuning)（SFT）模型（πSFT\pi\_{SFT}πSFT​）差异很大。这种现象有时被称为“奖励欺骗”或“模式崩塌”，即模型过度优化代理奖励信号，从而失去在预训练 (pre-training)和SFT期间学到的通用语言能力。

为此，标准的RLHF流程引入了一个惩罚项到PPO目标函数中，该惩罚项基于RL策略的输出分布与SFT模型输出分布之间的Kullback-Leibler（KL）散度。

### 让策略保持稳定

KL散度，表示为 KL(P∣∣Q)KL(P || Q)KL(P∣∣Q)，衡量一个概率分布 PPP 与参考概率分布 QQQ 的差异程度。KL散度为零表示分布完全相同。在RLHF的背景下，我们希望衡量对于给定输入提示 xxx 和可能的输出标记 (token) yyy，当前RL策略 πRL(y∣x)\pi\_{RL}(y|x)πRL​(y∣x) 与参考SFT策略 πSFT(y∣x)\pi\_{SFT}(y|x)πSFT​(y∣x) 偏离了多少。

目标不只是最大化来自RM的奖励，而是要在不 *偏离太远* 于表现良好的SFT模型行为的前提下进行。我们将这一约束直接纳入PPO使用的奖励信号中。给定提示 xxx，生成序列 y=(y1,...,yT)y = (y\_1, ..., y\_T)y=(y1​,...,yT​) 的修改后奖励变为：

总奖励(x,y)=RRM(x,y)−β⋅KL(πRL(y∣x)∣∣πSFT(y∣x))\text{总奖励}(x, y) = R\_{RM}(x, y) - \beta \cdot KL(\pi\_{RL}(y|x) || \pi\_{SFT}(y|x))总奖励(x,y)=RRM​(x,y)−β⋅KL(πRL​(y∣x)∣∣πSFT​(y∣x))

这里：

- RRM(x,y)R\_{RM}(x, y)RRM​(x,y) 是奖励模型分配给生成的序列 (x,y)(x, y)(x,y) 的最终奖励。
- KL(πRL(y∣x)∣∣πSFT(y∣x))KL(\pi\_{RL}(y|x) || \pi\_{SFT}(y|x))KL(πRL​(y∣x)∣∣πSFT​(y∣x)) 是RL策略和SFT策略分配的序列概率分布之间的KL散度。在实际操作中，这通常在生成过程中按每个标记进行近似或计算。
- β\betaβ 是一个超参数 (parameter) (hyperparameter)，用于控制KL惩罚的强度。更高的 β\betaβ 值会对偏离SFT模型的行为施加更强的惩罚，使 πRL\pi\_{RL}πRL​ 保持接近 πSFT\pi\_{SFT}πSFT​。更低的 β\betaβ 值允许RL策略有更大的自由度来生成最大化RM分数的输出，即使它们与SFT模型的输出有明显差异。

### 实践中计算KL惩罚

计算整个序列的精确KL散度通常是不可行的。相反，PPO通常在标记 (token)级别上操作。在PPO的生成（rollout）阶段，对于在给定上下文 (context) x,y<tx, y\_{<t}x,y<t​ 情况下，在步骤 ttt 生成的每个标记 yty\_tyt​，我们计算：

1. 该标记根据当前RL策略的对数概率：log⁡πRL(yt∣x,y<t)\log \pi\_{RL}(y\_t | x, y\_{<t})logπRL​(yt​∣x,y<t​)
2. 同一标记根据参考SFT策略的对数概率：log⁡πSFT(yt∣x,y<t)\log \pi\_{SFT}(y\_t | x, y\_{<t})logπSFT​(yt​∣x,y<t​)

按标记的KL惩罚项通常通过这些对数概率的差值来近似：

按标记的KL惩罚近似≈log⁡πRL(yt∣x,y<t)−log⁡πSFT(yt∣x,y<t)\text{按标记的KL惩罚近似} \approx \log \pi\_{RL}(y\_t | x, y\_{<t}) - \log \pi\_{SFT}(y\_t | x, y\_{<t})按标记的KL惩罚近似≈logπRL​(yt​∣x,y<t​)−logπSFT​(yt​∣x,y<t​)

该项在计算优势和更新策略之前，从RM提供的按标记奖励中减去（乘以 β\betaβ 后）。参考SFT模型的参数 (parameter)在此RL阶段保持冻结；它只执行前向传播来提供参考对数概率。

让我们看一个PyTorch代码片段，说明如何为一批生成的序列计算此惩罚。假设我们有来自两个模型在每个位置上词汇表 (vocabulary)中每个标记的对数概率。

```python
import torch
import torch.nn.functional as F

def calculate_kl_penalty(log_probs_rl, log_probs_sft, actions, beta):
    """
    计算RLHF PPO中使用的KL惩罚项。

    参数:
        log_probs_rl: 来自RL策略模型的对数概率。
        log_probs_sft: 来自参考SFT模型的对数概率
                       （已分离）。
        actions: RL策略生成的标记ID。
        beta: KL惩罚系数。

    返回:
        kl_penalty: 每个标记位置的KL惩罚张量。
                    形状: [批大小, 序列长度]
    """

    log_probs_sft_detached = log_probs_sft.detach()

    log_prob_rl_taken = torch.gather(
        log_probs_rl, 2, actions.unsqueeze(2)
    ).squeeze(2)
    log_prob_sft_taken = torch.gather(
        log_probs_sft_detached, 2, actions.unsqueeze(2)
    ).squeeze(2)

    log_ratio = log_prob_rl_taken - log_prob_sft_taken

    kl_penalty = beta * log_ratio
    return kl_penalty
```

此代码片段说明了将 *所采取动作* 的对数概率比用作KL惩罚项的常见做法。它比在每一步计算整个词汇表上的完整KL散度计算成本更低，但其作用相同，都是为了惩罚偏离行为。

### 权衡考量

KL惩罚作为一个重要的正则化 (regularization)项。它防止RL策略陷入过度依赖奖励模型而牺牲语言质量的狭窄模式。通过将RL策略与SFT策略关联起来，这有助于确保模型保留其流畅性、连贯性和其中包含的广泛知识。

选择合适的 β\betaβ 值很重要。

- 如果 β\betaβ 过高，RL训练可能会受到过度限制，阻止策略基于奖励信号明显改进。模型将非常紧密地遵循原始SFT行为。
- 如果 β\betaβ 过低，策略可能会偏离过多，可能导致奖励欺骗和输出质量下降，即使RM分数很高。

找到一个合适的 β\betaβ 通常需要凭经验调整，并仔细评估最大化RM分数与保持理想语言特性之间的权衡。

RLHF\_Objective

Objective

PPO每标记目标

Maximize

最大化

Objective->Maximize

Reward

来自RM的奖励
(越高越好)

Maximize->Reward

Penalty

KL惩罚项
(越低越好)

Maximize->Penalty

-

LogRatio

log u03c0\_RL(y|x) - log u03c0\_SFT(y|x)

Penalty->LogRatio

Beta

u03b2 (权重)

Penalty->Beta

\*

> PPO目标旨在最大化来自奖励模型的奖励，同时最小化通过 \u03b2 缩放的KL惩罚，这会阻止偏离原始SFT模型的行为。

总之，KL散度惩罚是RLHF中的一种标准方法，用于稳定训练并防止语言模型过度偏离其初始SFT状态。它鼓励模型根据人类反馈（由RM捕获）找到受青睐的输出，同时保持其在之前训练阶段学到的流畅性和通用能力。

获取即时帮助、个性化解释和交互式代码示例。

---

### RLHF中的挑战与考量

# RLHF中的挑战与考量

强化学习 (reinforcement learning)从人类反馈 (RLHF) 流程为使大型语言模型与人类偏好保持一致提供了一个强大的框架。然而，其实现通常很复杂，并带来一些实际困难。要成功应对这些，需要仔细的设计选择、工程实践和持续监督。

### 数据质量与可扩展性

RLHF的根本在于高质量的人类偏好数据。收集这些数据需要大量资源，并带来一些障碍：

1. **主观性与分歧：** 人类偏好本身就具有主观性。不同的标注者可能在哪个响应更好上存在分歧，特别是对于复杂的提示。制定清晰的标注指南并进行标注者校准是重要步骤，但一定程度的噪声和分歧是无法避免的。这种噪声会影响学到的奖励模型的质量和一致性。
2. **标注者偏见：** 标注者带有自己的偏见，可能影响数据集中编码的偏好。这些偏见可能与人口统计学、文化背景，甚至给出的特定指令有关。确保标注人员的多样性并设计任务以减少偏见，是重要但有难度的操作考量。
3. **成本与规模：** 生成成对比较需要人工标注者阅读并评估大量提示的多个模型输出。将此过程扩展到覆盖大型语言模型广泛的潜在输入和输出范围，既昂贵又耗时。训练一个能够良好泛化的可靠奖励模型所需的数据量常常是瓶颈。
4. **数据多样性：** 偏好数据集必须足够多样，以涵盖各种情景、主题和LLM潜在的失败模式。如果数据只涵盖狭窄的交互范围，那么由此产生的奖励模型和对齐 (alignment)策略可能无法很好地泛化到未见过的情况。

### 奖励模型局限

奖励模型（RM）在强化学习 (reinforcement learning)（RL）阶段充当人类偏好的代表。然而，它是一个不完美的代表，可能导致一些问题：

1. **规格博弈（奖励作弊）：** 强化学习策略专门针对奖励模型的分数进行优化。如果奖励模型存在缺陷或可利用的模式，策略可能学会以不对应于在有用性、诚实性或无害性方面真实改进的方式来最大化分数。例如，策略可能学到奖励模型偏爱更长、更冗余的回答，导致不必要的冗长输出，或者它可能发现特定的短语，欺骗奖励模型给出高分。
2. **分布漂移：** 奖励模型是在一个静态的偏好数据集上训练的。在强化学习训练期间，策略模型（πRL\pi\_{\text{RL}}πRL​）会演变，生成可能与奖励模型训练期间所见输出明显不同的输出。当评估这些分布外输出时，奖励模型的准确性会下降，导致不可靠的奖励信号。
3. **难以捕捉细节：** 对齐 (alignment)的复杂方面，如不明显的偏见、响应的长期后果或严格的事实正确性，可能很难被一个简单的基于偏好的奖励模型准确捕捉。成对比较格式简化了判断任务，但也可能过度简化了期望的行为。
4. **校准：** 确保奖励模型分数良好校准（即分数差异准确反映偏好强度）是困难的。差的校准会影响强化学习优化过程的稳定性和有效性。

### 训练不稳定性与超参数 (parameter) (hyperparameter)敏感性

强化学习 (reinforcement learning)，特别是应用于大型Transformer模型的PPO，以其对超参数的敏感性以及潜在的不稳定性而闻名：

1. **PPO的复杂性：** PPO涉及多个组成部分：值函数估计、优势计算（通常使用广义优势估计GAE）、带有裁剪的策略更新以及KL散度惩罚。每个组成部分都引入了需要仔细调整的超参数（GAE的λ\lambdaλ、裁剪的ϵ\epsilonϵ、KL系数β\betaβ）。
2. **奖励与值估计的方差：** 奖励模型的回报可能存在噪声，并且为高维状态空间（由模型激活或输入/输出表示）准确估计值函数具有挑战性。高方差会减缓或破坏学习的稳定性。
3. **KL散度约束：** KL惩罚（β⋅DKL(πRL∣∣πSFT)\beta \cdot D\_{\text{KL}}(\pi\_{\text{RL}} || \pi\_{\text{SFT}})β⋅DKL​(πRL​∣∣πSFT​)）对于防止强化学习策略偏离初始SFT模型太远非常重要，从而减轻奖励作弊并保留通用语言能力。然而，选择正确的系数β\betaβ非常重要。太小，策略可能对奖励模型过度优化；太大，学习就会受阻。这通常需要在训练期间进行动态调整。

考虑一个包含KL惩罚的简化PPO目标函数：

LPPO+KL(θ)=Et[min⁡(rt(θ)At,clip(rt(θ),1−ϵ,1+ϵ)At)]−β⋅Et[DKL(πθ(⋅∣st)∣∣πSFT(⋅∣st))]L\_{\text{PPO+KL}}(\theta) = \mathbb{E}\_t \left[ \min(r\_t(\theta) A\_t, \text{clip}(r\_t(\theta), 1-\epsilon, 1+\epsilon) A\_t) \right] - \beta \cdot \mathbb{E}\_t \left[ D\_{\text{KL}}(\pi\_{\theta}(\cdot|s\_t) || \pi\_{\text{SFT}}(\cdot|s\_t)) \right]LPPO+KL​(θ)=Et​[min(rt​(θ)At​,clip(rt​(θ),1−ϵ,1+ϵ)At​)]−β⋅Et​[DKL​(πθ​(⋅∣st​)∣∣πSFT​(⋅∣st​))]

这里，rt(θ)=πθ(at∣st)πθold(at∣st)r\_t(\theta) = \frac{\pi\_{\theta}(a\_t|s\_t)}{\pi\_{\theta\_{\text{old}}}(a\_t|s\_t)}rt​(θ)=πθold​​(at​∣st​)πθ​(at​∣st​)​是概率比，AtA\_tAt​是优势估计，ϵ\epsilonϵ是裁剪参数，β\betaβ控制对原始SFT策略πSFT\pi\_{\text{SFT}}πSFT​的KL惩罚。调整ϵ\epsilonϵ和β\betaβ会明显影响稳定性和性能。

```python

import torch
import torch.nn.functional as F

def compute_ppo_loss(
    policy_log_probs,
    old_policy_log_probs,
    advantages,
    rewards_from_rm,
    sft_policy_log_probs,
    clip_param,
    kl_beta
):
    """
计算带KL惩罚的PPO损失。
    假定输入是适当形状的张量
    （例如，[batch_size]）。
    """

    ratio = torch.exp(policy_log_probs - old_policy_log_probs)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages

    policy_loss = -torch.min(surr1, surr2).mean()

    kl_div = (policy_log_probs - sft_policy_log_probs).mean()

    total_loss = policy_loss - kl_beta * kl_div

    return total_loss

policy_log_probs = torch.randn(4, requires_grad=True)
old_policy_log_probs = torch.randn(4)
sft_policy_log_probs = torch.randn(4)

advantages = torch.randn(4)

rewards_from_rm = torch.randn(4)

clip_param = 0.2
kl_beta = 0.1

loss = compute_ppo_loss(
    policy_log_probs,
    old_policy_log_probs,
    advantages,
    rewards_from_rm,
    sft_policy_log_probs,
    clip_param,
    kl_beta
)

print(f"计算出的PPO+KL损失: {loss.item()}")
```

> 一个PyTorch代码片段，说明了PPO损失计算的组成部分，特别突出显示了替代目标和简化的KL惩罚项。请注意，一个完整的实现需要仔细处理用于KL计算和值函数训练的分布。

4. **梯度方差与优化：** 使用强化学习优化大型语言模型通常涉及大批量和分布式训练设置。管理梯度同步、通信开销以及确保跨多个工作节点的数值稳定性增加了复杂性。梯度累积和仔细选择优化器（例如AdamW）等技术是标准做法，但强化学习增加了另一层优化挑战。

### 计算成本

RLHF在计算上要求很高：

1. **多模型训练：** 这需要训练至少三个大型模型：初始SFT模型、奖励模型和最终的强化学习 (reinforcement learning)策略。每个阶段都需要大量GPU资源。
2. **推理 (inference)开销：** 在强化学习期间，需要频繁进行推理：强化学习策略生成响应，SFT策略通常用于KL惩罚计算，并且奖励模型评估生成的响应。这个推理循环在许多优化步骤中重复进行，构成了主要的计算负担。
3. **内存需求：** 存储多个模型状态（策略、值函数、奖励模型、可能的SFT参考策略）及其梯度、激活和优化器状态需要大量的GPU内存，通常需要模型并行和ZeRO等内存优化技术。

### 评估困难

衡量RLHF对齐 (alignment)的成功具有挑战性：

1. **标准度量：** 标准的自然语言处理度量（如BLEU或ROUGE），甚至困惑度，都不能充分捕捉有用性或无害性等对齐目标。
2. **人工评估：** 可靠的评估通常需要对最终模型输出进行进一步的人工评估，这既昂贵又耗时。设计评估协议具有挑战性。
3. **对齐税：** 通过RLHF在对齐指标上有所改进，有时可能会以牺牲某些能力或基准测试上的性能下降为代价（即“对齐税”）。量化 (quantization)和平衡这些权衡非常重要。
4. **基准局限：** 尽管存在针对安全性或真实性的特定基准（例如TruthfulQA、ToxiGen），但它们可能无法涵盖期望行为的所有方面，而且模型有时可能会对这些基准过拟合 (overfitting)。

### 伦理考量

定义“人类偏好”的过程引发了伦理问题：

1. **谁的偏好？** 标注者的选择和偏好提示的设计隐含地编码了特定的价值观。确保公平性、代表性，并避免放大数据中或标注者之间存在的社会偏见，是一项重要的伦理责任。
2. **透明度：** RLHF过程的复杂性可能使其不透明。理解模型在RLHF后为何以某种方式行动可能很困难，这使得确保问责制和可信度的努力变得复杂。
3. **潜在滥用：** 像任何强大的技术一样，对齐 (alignment)后的模型也可能被滥用。持续考量安全措施和负责任的部署实践是必要的。

成功实施RLHF需要认识到这些挑战，并在仔细的数据整理、工程实践、全面评估和持续的伦理反思方面投入。像直接偏好优化（DPO）这样的技术，旨在通过绕过显式奖励建模来简化过程，也作为标准RLHF流程的潜在替代方案或补充方案而受到关注。

获取即时帮助、个性化解释和交互式代码示例。

---

### 替代方法：直接偏好优化 (DPO)

# 替代方法：直接偏好优化 (DPO)

尽管使用近端策略优化 (PPO) 的人类反馈强化学习 (reinforcement learning) (RLHF) 在使语言模型与人类偏好保持一致方面已被证明有效，但此过程可能复杂且有时不稳定。它通常包含多个阶段：监督微调 (fine-tuning) (SFT)、基于人类偏好数据训练一个独立的奖励模型 (RM)，然后使用由 RM 指导的强化学习对 SFT 模型进行微调。这种多阶段流程引入了多个超参数 (parameter) (hyperparameter)和潜在的失败点，特别是在强化学习阶段，该阶段对实现细节敏感，且容易出现奖励作弊等问题。

直接偏好优化 (DPO) 提供了一种更简化的偏好对齐 (alignment)方法，绕过了显式奖励模型训练的需求，并彻底避免了强化学习的复杂性。DPO 使用一个简单的分类目标，直接基于偏好数据优化语言模型。

DPO 的核心思路源于 RLHF 寻求的最优策略与潜在（隐式）奖励函数之间的数学关系。回想一下，标准奖励建模通常使用 Bradley-Terry 模型，将提示 xxx 的成对偏好 (yw,yl)(y\_w, y\_l)(yw​,yl​)（其中 ywy\_wyw​ 优于 yly\_lyl​）与潜在奖励函数 rrr 关联起来：

P(yw≻yl∣x)=σ(r(x,yw)−r(x,yl))P(y\_w \succ y\_l | x) = \sigma(r(x, y\_w) - r(x, y\_l))P(yw​≻yl​∣x)=σ(r(x,yw​)−r(x,yl​))

此处，σ\sigmaσ 是 logistic 函数。RLHF 的目标是找到一个策略 πRL\pi\_{RL}πRL​，在使预期奖励 E[r(x,y)]E[r(x, y)]E[r(x,y)] 最大化的同时，通过 KL 散度惩罚，保持与参考策略 πref\pi\_{ref}πref​（通常是 SFT 模型）的接近度：

max⁡πRLE(x,y)∼πRL[r(x,y)]−βDKL(πRL(y∣x)∣∣πref(y∣x))\max\_{\pi\_{RL}} E\_{(x,y) \sim \pi\_{RL}}[r(x, y)] - \beta D\_{KL}(\pi\_{RL}(y|x) || \pi\_{ref}(y|x))maxπRL​​E(x,y)∼πRL​​[r(x,y)]−βDKL​(πRL​(y∣x)∣∣πref​(y∣x))

DPO 使用此约束优化问题的解析解。可以证明，最优策略 πRL\pi\_{RL}πRL​ 具有以下形式：

πRL(y∣x)=1Z(x)πref(y∣x)exp⁡(1βr(x,y))\pi\_{RL}(y|x) = \frac{1}{Z(x)} \pi\_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)πRL​(y∣x)=Z(x)1​πref​(y∣x)exp(β1​r(x,y))

其中 Z(x)Z(x)Z(x) 是一个配分函数，确保概率和为一。此方程连接了最优策略、参考策略和奖励函数。通过将此关系代回 Bradley-Terry 偏好模型，可以消除奖励函数 r(x,y)r(x, y)r(x,y)，直接用策略表示偏好概率：

P(yw≻yl∣x)=σ(βlog⁡πRL(yw∣x)πref(yw∣x)−βlog⁡πRL(yl∣x)πref(yl∣x))P(y\_w \succ y\_l | x) = \sigma\left( \beta \log \frac{\pi\_{RL}(y\_w|x)}{\pi\_{ref}(y\_w|x)} - \beta \log \frac{\pi\_{RL}(y\_l|x)}{\pi\_{ref}(y\_l|x)} \right)P(yw​≻yl​∣x)=σ(βlogπref​(yw​∣x)πRL​(yw​∣x)​−βlogπref​(yl​∣x)πRL​(yl​∣x)​)

DPO 直接训练语言模型 πθ\pi\_\thetaπθ​ 来满足此偏好模型，其中 πθ\pi\_\thetaπθ​ 代替了未知的最优策略 πRL\pi\_{RL}πRL​。其目标是最大化在此模型下观测到的人类偏好的对数似然。这从而得到 DPO 损失函数 (loss function)：

LDPO(πθ;πref)=−E(x,yw,yl)∼D[log⁡σ(βlog⁡πθ(yw∣x)πref(yw∣x)−βlog⁡πθ(yl∣x)πref(yl∣x))]\mathcal{L}\_{DPO}(\pi\_\theta; \pi\_{ref}) = - E\_{(x, y\_w, y\_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi\_\theta(y\_w|x)}{\pi\_{ref}(y\_w|x)} - \beta \log \frac{\pi\_\theta(y\_l|x)}{\pi\_{ref}(y\_l|x)} \right) \right]LDPO​(πθ​;πref​)=−E(x,yw​,yl​)∼D​[logσ(βlogπref​(yw​∣x)πθ​(yw​∣x)​−βlogπref​(yl​∣x)πθ​(yl​∣x)​)]

此处，D\mathcal{D}D 是偏好三元组 (x,yw,yl)(x, y\_w, y\_l)(x,yw​,yl​) 的数据集，πθ\pi\_\thetaπθ​ 是正在优化的语言模型，πref\pi\_{ref}πref​ 是冻结的参考 SFT 模型，β\betaβ 是一个超参数，它隐式控制优化策略 πθ\pi\_\thetaπθ​ 与参考策略 πref\pi\_{ref}πref​ 的偏离程度。更高的 β\betaβ 会赋予偏好数据更大的权重 (weight)，可能导致更大的偏离。

### 实现和使用

实现 DPO 比 RLHF 显著更简单。它需要：

1. 一个偏好数据集 D={(xi,yw,i,yl,i)}\mathcal{D} = \{(x\_i, y\_{w,i}, y\_{l,i})\}D={(xi​,yw,i​,yl,i​)}。
2. 一个基础 SFT 模型作为参考策略 πref\pi\_{ref}πref​。此模型的权重 (weight)在 DPO 训练期间保持冻结。
3. SFT 模型的一个副本 πθ\pi\_\thetaπθ​，其权重将被更新。

训练循环涉及标准监督学习 (supervised learning)：

- 从 D\mathcal{D}D 中采样一批偏好三元组 (x,yw,yl)(x, y\_w, y\_l)(x,yw​,yl​)。
- 计算当前模型 πθ\pi\_\thetaπθ​ 和参考模型 πref\pi\_{ref}πref​ 下，选中 (ywy\_wyw​) 和拒绝 (yly\_lyl​) 响应的对数概率。
- 使用这些对数概率计算 DPO 损失 LDPO\mathcal{L}\_{DPO}LDPO​。
- 执行反向传播 (backpropagation)并使用 AdamW 等优化器更新 πθ\pi\_\thetaπθ​ 的权重。

下面是一个 PyTorch 代码片段，用于在训练步骤中计算 DPO 损失的核心部分，假设您已获得对数概率：

```python
import torch
import torch.nn.functional as F

def dpo_loss(log_probs_policy_w, log_probs_policy_l,
             log_probs_ref_w, log_probs_ref_l, beta):
    """计算一批偏好的 DPO 损失。"""

    log_ratio_w = log_probs_policy_w - log_probs_ref_w
    log_ratio_l = log_probs_policy_l - log_probs_ref_l

    diff_scaled = beta * (log_ratio_w - log_ratio_l)

    loss = -F.logsigmoid(diff_scaled)

    return loss.mean()
```

### 优点和缺点

**优点：**

- **简洁性：** DPO 在 SFT 之后只需要一个训练阶段，无需进行奖励模型拟合和复杂的强化学习 (reinforcement learning)算法。
- **稳定性：** 它避免了 RLHF 中 PPO 经常遇到的优化不稳定问题，例如对 KL 散度系数和奖励缩放的敏感性。
- **效率：** 它通常比基于 PPO 的 RLHF 计算成本更低，因为它在训练期间不需要从策略中采样生成内容，也不需要训练独立的奖励模型。

**缺点：**

- **性能：** 尽管 DPO 通常能达到与 RLHF 可比的结果，但与精心调优的 PPO 流程相比，有时可能表现不佳，特别是在潜在的奖励信号复杂或有噪声的情况下。
- **超参数 (parameter) (hyperparameter)调优：** 超参数 β\betaβ 仍然需要仔细调优，类似于 PPO 中的 KL 系数。
- **数据要求：** 类似于 RLHF，DPO 依赖于高质量的偏好数据集。

总之，DPO 为标准 RLHF 流程提供了一种更简单的替代方案。其稳定性及易于实现使其成为一个有吸引力的选择，尤其是在计算资源或强化学习专业知识有限的情况下，它能有效地使语言模型与人类偏好保持一致。这代表了使大型语言模型更具帮助性、诚实性和无害性的过程中的显著简化。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 27 Model Compression Techniques

### 模型压缩的动因

# 模型压缩的动因

“正如我们之前谈到的，成功训练大型语言模型常常会产生包含数十亿甚至数万亿参数 (parameter)的模型。虽然这些大模型展现出强大的能力，但其庞大的规模在从实验室或训练集群转向实际应用时，会带来不少实际困难。模型压缩的必要性直接源于这些挑战，主要集中在内存需求、推理 (inference)时的计算负担以及相关的运营成本。”

### 内存瓶颈

考虑一个拥有70亿参数 (parameter)的模型。如果每个参数都使用标准的32位浮点精度（FP32FP32FP32）存储，每个参数占用4字节，那么仅仅加载模型权重 (weight)所需的内存就非常可观：

内存占用=参数数量×每参数字节数\text{内存占用} = \text{参数数量} \times \text{每参数字节数}内存占用=参数数量×每参数字节数
内存占用=7×109 参数×4 字节/参数=28×109 字节=28 GB\text{内存占用} = 7 \times 10^9 \text{ 参数} \times 4 \text{ 字节/参数} = 28 \times 10^9 \text{ 字节} = 28 \text{ GB}内存占用=7×109 参数×4 字节/参数=28×109 字节=28 GB

这个28 GB的计算仅考虑了模型权重本身。在推理 (inference)时，还需要额外的内存用于激活、临时计算以及键值（KV）缓存（我们将在第28章谈到它），尤其是在处理长序列或批量处理多个请求时。高端GPU通常配备24 GB、40 GB或80 GB的高带宽内存（HBM），但即使是这些对于最大的模型或有效提供多个模型副本也可能不足。需要这种高内存硬件会大幅增加部署成本，并限制模型可以在哪些类型的设备上运行。

```python
import torch
import torch.nn as nn

hidden_dim = 4096
intermediate_dim = 11008

ffn_layer1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
ffn_layer2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)

params_ffn1 = hidden_dim * intermediate_dim
params_ffn2 = intermediate_dim * hidden_dim

total_ffn_params = params_ffn1 + params_ffn2

print(f"一个FFN块中的参数（约）：{total_ffn_params:,}")

memory_ffn_gb = (total_ffn_params * 4) / (1024**3)
print(f"一个FFN块所需的内存（FP32，约）：{memory_ffn_gb:.2f} GB")
```

这个简单的计算表明，内存需求增长得多么迅速，甚至还没有考虑注意力机制 (attention mechanism)和嵌入 (embedding)表。

1B7B70B175B0100200300400500600700

> 估算不同模型大小在32位精度下仅存储模型权重所需的内存。

### 推理 (inference)延迟和吞吐量 (throughput)

除了内存，通过这些大型网络进行一次前向传播的计算成本也很高。自回归 (autoregressive)生成是大型语言模型生成文本的常见方式，它需要模型为每个生成的token顺序运行。尽管像KV缓存（第28章）这样的技术有所帮助，但每个token所需的矩阵乘法和其他操作数量之多，给延迟设定了下限。

对于聊天机器人、编码助手或实时翻译等交互式应用，高延迟会导致糟糕的用户体验。即使是文档摘要等离线任务，缓慢的推理速度也会增加处理大型数据集所需的时间。此外，每个请求的高延迟会限制部署的总体吞吐量（每秒处理的请求数），需要更多的并行硬件实例来处理给定的负载，这又会推高成本。

### 运营成本和可用性

高内存需求和巨大的计算负担结合在一起，直接转化为更高的运营开支：

1. **硬件成本：** 部署大型语言模型通常需要昂贵的高端GPU或专用加速器。扩展服务需要配置大量此类设备。
2. **能耗：** 这些强大的硬件组件消耗大量电力，增加了运营成本，也引发了环境问题。
3. **基础设施复杂性：** 管理用于推理 (inference)的专业硬件集群，会增加部署和维护流程的复杂性。

这些因素设置了障碍，使得在某些情境下部署先进的大型语言模型变得困难或不可能：

- **设备端部署：** 由于严格的内存、功耗和散热限制，直接在智能手机、笔记本电脑或嵌入 (embedding)式系统上运行模型通常不可行。模型压缩对于实现涉及大型语言模型的边缘AI应用很重要。
- **资源受限环境：** 缺乏大规模云计算基础设施的研究人员、初创公司或组织可能会觉得部署大型模型过于昂贵。
- **可扩展性：** 即使是大型组织，在经济高效地扩展推理基础设施以服务数百万用户时也面临挑战。更高效的模型可以减轻这种负担。

模型压缩技术提供了一种缓解这些挑战的途径。通过减少内存占用和计算需求，我们可以：

- 在成本较低、更容易获得的硬件上部署模型。
- 降低推理延迟，提升交互式应用的用户体验。
- 降低运营成本（硬件、能源）。
- 使模型能够部署在资源受限的环境中，包括边缘设备。
- 促进更广泛地使用强大的语言建模能力。

接下来的章节将探讨实现这些目标的主要方法：量化 (quantization)、剪枝和知识蒸馏 (knowledge distillation)。每种技术都涉及权衡，通常是用一定程度的模型性能换取效率上的大幅提升。理解这些方法及其影响，对于任何负责将大型语言模型投入生产的工程师来说都很重要。

获取即时帮助、个性化解释和交互式代码示例。

---

### 权重量化 (INT8, INT4)

# 权重量化 (INT8, INT4)

权重 (weight)量化 (quantization)是一种主要技术，用于减少大型语言模型的内存占用，并通常加快其推理 (inference)速度。其基本思路是将模型通常用32位浮点数（FP32）存储的权重参数 (parameter)，转换为8位整数（INT8）甚至4位整数（INT4）等低精度整数格式表示。每个参数的比特宽度减少直接导致模型尺寸变小，并可能在支持低精度计算的硬件上实现更快的运算。

### 数值格式的理解

回顾一下，标准FP32格式提供宽泛的动态范围和高精度，这在敏感的训练过程中非常重要。然而，对于推理 (inference)来说，通常可以使用更少的比特位，而不会造成模型准确率的严重下降。

- **FP32 (单精度):** 标准格式。用1位表示符号，8位表示指数，23位表示尾数。提供大范围和高精度。
- **FP16 (半精度):** 用1位表示符号，5位表示指数，10位表示尾数。在兼容硬件（如NVIDIA Tensor Cores）上计算速度更快，与FP32相比内存使用减半，但范围有限，易受溢出/下溢影响（如第20章所述）。
- **BF16 (脑浮点):** 用1位表示符号，8位表示指数（与FP32相同），7位表示尾数。与FP16一样内存减半，但保持FP32的动态范围，以精度为代价减少溢出/下溢问题。在训练和推理中越来越常见。
- **INT8 (8位整数):** 用8位表示数值。显著减少内存（相对于FP32减少4倍），并可在具有专用INT8指令的硬件上实现大幅加速。范围和精度远低于浮点格式。
- **INT4 (4位整数):** 更进一步，每个权重 (weight)仅使用4位。相较于FP32可减少8倍内存，但由于范围和精度受到严格限制，在保持模型准确率方面带来更大挑战。

权重量化 (quantization)的核心难点在于将范围宽泛的FP32权重映射到INT8或INT4值的有限范围，同时最大限度地减少对模型性能非常重要的信息损失。

### 量化 (quantization)原理：浮点数到整数的映射

最常用的方法是仿射量化，它使用一个缩放因子 SSS（一个正浮点数）和一个零点 ZZZ（一个整数，通常与 xintx\_{int}xint​ 类型相同），将浮点数值 xfloatx\_{float}xfloat​ 映射到整数值 xintx\_{int}xint​。关系如下：

xfloat≈S×(xint−Z)x\_{float} \approx S \times (x\_{int} - Z)xfloat​≈S×(xint​−Z)

反之，反量化将整数反向映射回近似浮点数：

xdequant=S×(xint−Z)x\_{dequant} = S \times (x\_{int} - Z)xdequant​=S×(xint​−Z)

- **缩放因子 (SSS):** 决定量化值之间的步长。它根据原始浮点数值的范围（max(xfloat)−min(xfloat)max(x\_{float}) - min(x\_{float})max(xfloat​)−min(xfloat​)）除以可用整数级别数（例如，INT8 为 28−12^8 - 128−1）来计算。
- **零点 (ZZZ):** 确保浮点数值0.0能被整数准确表示。对于*对称量化*，范围 [min(xfloat),max(xfloat)][min(x\_{float}), max(x\_{float})][min(xfloat​),max(xfloat​)] 对称地映射到零附近，零点 ZZZ 可能是隐式为零或固定值。对于*非对称量化*，ZZZ 会被调整以精确映射浮点零点，如果原始权重 (weight)不以零为中心，这可能更有益。ZZZ 通常是目标整数范围内的整数（例如，无符号INT8为0到255，有符号INT8为-128到127）。

缩放因子和零点可以以不同方式确定：

- **逐张量 (Per-Tensor):** 为整个权重张量（例如，线性层的权重矩阵）计算一个单独的 SSS 和 ZZZ。这很简单，但如果张量内部的数值范围差异很大，则效果可能不理想。
- **逐通道/逐轴 (Per-Channel / Per-Axis):** 为张量的切片计算独立的 SSS 和 ZZZ 值，通常沿特定维度（例如，卷积层或线性层权重的输出通道维度）。这提供更细的粒度，并且通常比逐张量量化产生更好的准确率，特别是对于权重分布在不同通道或行/列之间差异显著的层。

G

FP32\_Tensor

FP32 权重张量
（例如，1024x512）

INT8\_Tensor\_PerTensor

INT8 张量（逐张量）
1024x512

FP32\_Tensor->INT8\_Tensor\_PerTensor

量化

INT8\_Tensor\_PerChannel

INT8 张量（逐通道）
1024x512

FP32\_Tensor->INT8\_Tensor\_PerChannel

量化

Scale\_ZP\_PerTensor

单一缩放因子 (S)
单一零点 (Z)

INT8\_Tensor\_PerTensor->Scale\_ZP\_PerTensor

使用

Scale\_ZP\_PerChannel

多个缩放因子 (S1..S1024)
多个零点 (Z1..Z1024)

INT8\_Tensor\_PerChannel->Scale\_ZP\_PerChannel

使用

> 权重张量的逐张量和逐通道量化方法的比较。逐通道量化为每个输出通道（本例中为行）使用不同的缩放因子/零点值。

### 训练后量化 (quantization) (PTQ)

PTQ是一种更简单的方法。您将一个已经用FP32训练好的模型，之后将其权重 (weight)转换为INT8等低精度格式。激活值在推理 (inference)过程中也可能被动态量化。

**流程：**

1. **训练 (Train):** 正常用FP32训练模型直至收敛。
2. **校准 (Calibrate):** 通过FP32模型输入少量有代表性的校准数据集（通常几百个样本就足够）。记录权重的范围（最小值/最大值），如果量化激活值，则记录模型中各个点的激活值范围。
3. **计算量化参数 (parameter) (Calculate Quantization Parameters):** 使用记录的范围来计算每个被量化的张量（或通道）的适当缩放因子 (SSS) 和零点 (ZZZ) 值。
4. **转换权重 (Convert Weights):** 应用计算出的 SSS 和 ZZZ 将FP32权重转换为INT8（或INT4）格式。存储这些量化后的权重及相应的 SSS/ZZZ 值。
5. **推理 (Inference):** 在推理过程中，权重以其整数格式加载。如果激活值也经过量化（动态或静态），它们会实时转换为INT8，或者在预先计算好后加载。如果硬件支持，计算（如矩阵乘法）使用整数运算执行，通常在应用偏置 (bias)或激活函数 (activation function)之前需要反量化回FP32（或中间累加器精度），或者使用直接处理量化操作的专用内核。

**示例 (PyTorch 权重)：**

```python
import torch
import torch.quantization

model.eval()

qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model, inplace=False)
model_prepared.qconfig = qconfig

print("正在运行校准...")
with torch.no_grad():
    for input_batch, _ in calibration_data_loader:
        model_prepared(input_batch)
print("校准完成。")

model_quantized_static = torch.quantization.convert(
    model_prepared, inplace=False
)
print("模型已转换为静态量化版本。")

model_quantized_dynamic = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
print("模型已转换为动态量化版本。")

def print_model_size(mdl, label):
    torch.save(mdl.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")/1e6
    print(f"{label} 的大小: {size:.2f} MB")
    os.remove("temp.p")
```

**PTQ 的优点：**

- 易于实现；不需要更改原始训练流程。
- 转换过程快。

**PTQ 的缺点：**

- 可能导致模型准确率明显下降，尤其是在量化到极低比特宽度（如INT4）或对于敏感模型而言。训练后引入的量化误差未得到补偿。
- 准确率对校准数据集的选择和大小敏感。

### 量化 (quantization)感知训练 (QAT)

QAT通过在微调 (fine-tuning)或训练过程中模拟量化效果来解决PTQ的准确率限制。这使得模型的权重 (weight)能够适应量化引入的精度损失。

**流程：**

1. **从预训练 (pre-training)模型开始 (Start with Pre-trained Model):** 从一个已收敛的FP32模型开始（或者从头开始训练，尽管微调更常见）。
2. **插入伪量化节点 (Insert Fake Quantization Nodes):** 通过在权重层之前和之后，以及可能在激活值之后插入“伪量化”（或量化/反量化）操作符来修改模型图。这些操作符在前向传播过程中模拟量化过程：它们将FP32值量化为目标整数格式（例如INT8），并立即将其反量化回FP32。
   - 前向传播: xout=dequantize(quantize(xin,S,Z),S,Z)x\_{out} = dequantize(quantize(x\_{in}, S, Z), S, Z)xout​=dequantize(quantize(xin​,S,Z),S,Z)
   - 反向传播 (backpropagation): 梯度使用直通估计器（STE）计算，本质上忽略不可微分的量化步骤，并像它是恒等函数一样传递梯度。
3. **微调 (Fine-tune):** 在这些伪量化节点激活的情况下，继续训练（微调）模型少量周期。优化器调整FP32权重，但在调整时“感知”到模拟量化引入的噪声和钳位效应。这有助于模型学习转换后更有效的权重。
4. **转换 (Convert):** 在QAT微调后，使用在QAT期间获得的（通常基于训练期间观察到的范围的移动平均值）学到的量化参数 (parameter)（SSS 和 ZZZ），将模型转换为真正的量化模型。

**示例 (PyTorch)：**

```python
import torch
import torch.quantization

model.train()

qig_config = torch.quantization.get_default_qat_qconfig(
    'fbgemm')
model_prepared_qat = torch.quantization.prepare_qat(model, inplace=False)
model_prepared_qat.qconfig = qig_config

print("开始 QAT 微调...")
optimizer = torch.optim.Adam(
    model_prepared_qat.parameters(),
    lr=1e-5)
num_qat_epochs = 3

for epoch in range(num_qat_epochs):
    for input_batch, target_batch in qat_training_data_loader:
        optimizer.zero_grad()
        output = model_prepared_qat(input_batch)
        loss = loss_function(output, target_batch)
        loss.backward()
        optimizer.step()
    print(f"QAT 第 {epoch+1}/{num_qat_epochs} 周期完成。")
print("QAT 微调完成。")

model_prepared_qat.eval()
model_quantized_qat = torch.quantization.convert(model_prepared_qat, inplace=False)
print("模型已转换为 QAT 量化版本。")
```

**QAT 的优点：**

- 通常比PTQ产生明显更好的准确率，通常接近原始FP32模型的性能，尤其是对于INT8。
- 对量化误差的抵抗力更强，因为模型在微调期间学习补偿。

**QAT 的缺点：**

- 实现更复杂，需要修改训练流程和额外的微调步骤。
- 由于微调阶段，增加了总体训练时间。

### 更低精度：INT4

将量化 (quantization)推向INT4甚至更低比特宽度（例如，三元或二元权重 (weight)）能提供最大的内存节省，但大幅增加了保持准确率的难度。

- **量化误差增大：** 仅有16个不同的值（对于INT4），可表示数字之间的差距大很多，导致更高的量化误差。
- **敏感性：** 在这些低比特级别下，模型对量化噪声敏感得多。
- **专用技术：** 标准的PTQ/QAT可能不足够。通常需要基于梯度的低秩量化（GPTQ）或涉及权重分组（分组量化）等高级技术来保持性能。这些方法可能将小块权重一起量化，使用共享或更复杂的量化参数 (parameter)。
- **硬件支持：** 尽管INT8支持在现代CPU和GPU/TPU中相对常见，但INT4或更低精度的有效硬件加速普及度较低，但正在出现（例如，NVIDIA Hopper架构支持FP8，其比特宽度介于INT4和INT8之间）。如果没有专用硬件内核，使用INT4可能无法带来加速。

`bitsandbytes` 等库在Hugging Face生态系统中被广泛用于对大型模型应用INT4量化（通常是NF4 - NormalFloat 4位等变体），这些库经常使用将量化与专用矩阵乘法内核相结合的技术。

### 权衡与实际考量

- **准确率与效率：** 这是核心权衡。PTQ速度快但可能会牺牲准确率。QAT能更好地保持准确率但需要更多精力。INT4/更低比特提供最大压缩，但如果不配合高级技术谨慎应用，则存在显著准确率损失的风险。
- **硬件依赖：** 量化 (quantization)带来的实际推理 (inference)加速很大程度上取决于底层硬件和软件堆栈。除非硬件具有高效的INT8矩阵乘法单元且框架使用优化过的内核（如NVIDIA GPU的cuDNN，或CPU上的特定指令），否则使用INT8权重 (weight)不会加速计算。
- **框架支持：** 深度学习 (deep learning)框架（PyTorch，TensorFlow）为PTQ和QAT都提供了工具。PyTorch有 `torch.quantization`，而TensorFlow通过TensorFlow Lite或模型优化工具包提供类似工具。Hugging Face的 `optimum` 和 `bitsandbytes` 等库专门为Transformer模型集成了量化功能。
- **层类型：** 量化在具有大型权重矩阵的层上最有效，例如线性（全连接）层和嵌入 (embedding)层，这些层在大型语言模型（LLM）参数 (parameter)数量中占据主导地位。对其他层（例如归一化 (normalization)层）的影响需要仔细考量。

FP32INT8 (PTQ)INT8 (QAT)INT4 (例如, GPTQ)9092949698100020406080100相对准确率 (%)相对尺寸 (%)典型量化权衡（示意性）

> 模型准确率和尺寸之间不同权重量化方法的示意性权衡。实际结果根据模型、任务和使用的具体量化技术而明显不同。

权重量化，特别是通过PTQ或QAT进行的INT8量化，是一种广泛采用的技术，可使大型语言模型更适合部署。尽管INT4提供进一步压缩，但它通常需要更复杂的方法和仔细评估，以确保可接受的性能水平。选择正确的策略取决于模型尺寸、推理延迟和允许的准确率下降的具体要求。

获取即时帮助、个性化解释和交互式代码示例。

---

### 激活量化考量

# 激活量化考量

如前所述，对模型权重 (weight)进行量化 (quantization)可以显著节省内存，但对激活进行量化会带来它自己的一系列困难和考量。激活，即神经元层的输出作为输入传递到下一层，本质上是动态的。与训练后固定的权重不同，激活值会随模型处理的每个输入样本而变动。这种动态特性使其量化更复杂但也同样重要，能实现推理 (inference)速度的最大提升和内存带宽的减少，尤其是在有专用低精度计算单元的硬件上。

### 难题：动态范围和敏感性

量化 (quantization)激活的主要难题源于其可能很宽且不可预测的动态范围。考虑ReLU或GeLU等激活函数 (activation function)的输出。这些非线性特性可以产生跨越多个数量级的值，并且与通常遵循某种可预测分布（例如，围绕零居中）的权重 (weight)不同，激活分布可以根据输入数据和网络中特定层而明显不同。

G

clusterᵣanges

激活范围

Input

输入数据 (x)

LayerNorm

层归一化(x)

Input->LayerNorm

Attention

注意力机制(...)

LayerNorm->Attention

 查询, 键, 值

AddNorm1

相加与归一化

LayerNorm->AddNorm1

 残差

R1
范围 A

Attention->AddNorm1

R2
范围 B

FFN

前馈网络(ReLU/GeLU)

AddNorm1->FFN

AddNorm2

相加与归一化

AddNorm1->AddNorm2

 残差

FFN->AddNorm2

R3
范围 C

Output

输出

AddNorm2->Output

> 激活范围在Transformer块内不同层之间明显不同。前馈网络（FFN）中的ReLU或GeLU等非线性函数尤其可以扩展动态范围。

使用低精度格式（如INT8）量化具有如此宽范围的张量迫使做出权衡。如果按比例量化以适应极端最大值和最小值（异常值），那么表示大多数值（可能聚集在小得多的范围里）的精度会变得非常粗糙。这种精度损失，称为量化误差，会显著降低模型精度。反之，如果您为值的密集聚类优化尺度，异常值将被截断，可能丢失重要信息。激活，尤其是在注意力机制 (attention mechanism)或中间FFN层内，可能对此截断或精度损失特别敏感。

### 校准：确定量化 (quantization)参数 (parameter)

为了有效映射激活的浮点范围到低精度整数范围，我们需要确定适当的量化参数：一个尺度因子 (sss) 和一个零点 (zzz)。一般映射是：

量化值=取整(浮点值s)+z\text{量化值} = \text{取整}(\frac{\text{浮点值}}{s}) + z量化值=取整(s浮点值​)+z

找到最优 sss 和 zzz 值的过程称为**校准**。它通常涉及通过模型输入一个代表性数据集（训练或验证数据的一个子集，通常是几百到几千个样本），并观察网络中不同点激活值的范围。存在几种校准方法：

1. **最小-最大校准：** 这是最简单的方法。记录校准期间观察到的最小 (xminx\_{min}xmin​) 和最大 (xmaxx\_{max}xmax​) 激活值。然后计算尺度 sss 和零点 zzz，将这个观察到的范围 [xmin,xmax][x\_{min}, x\_{max}][xmin​,xmax​] 映射到目标整数范围（例如，有符号INT8的 [−128,127][-128, 127][−128,127]）。

   - *优点:* 实现简单。
   - *缺点:* 对异常值高度敏感。一个极端值会显著恶化典型值的量化精度。
2. **均方误差（MSE）校准：** 此方法迭代观察到的最小-最大范围内的不同潜在截断范围（阈值）。对于每个阈值，它计算量化参数并测量原始浮点激活与其量化-反量化等效值之间的平均平方误差。选择使MSE最小的阈值。

   - *优点:* 与简单最小-最大方法相比，通常能在截断异常值和保持大部分分布的精度之间找到更好的平衡。
   - *缺点:* 由于需要搜索阈值，计算量比最小-最大方法大。
3. **熵（KL散度）校准：** 此方法旨在最小化量化过程中的信息损失。它选择量化参数 (sss 和 zzz)，使得原始浮点激活分布与量化-反量化激活分布之间的Kullback-Leibler（KL）散度最小化。

   - *优点:* 通常被认为是最有效的方法，特别是对于非均匀分布，因为它直接尝试保持原始激活分布的形状。
   - *缺点:* 可能是计算最复杂的校准方法。

以下是PyTorch风格的示例，说明了在校准特定激活张量期间如何使用观察器：

```python
import torch

class MinMaxObserver:
    def __init__(self):
        self.min_val = torch.tensor(float('inf'))
        self.max_val = torch.tensor(float('-inf'))

    def forward(self, x):

        x_detached = x.detach()
        self.min_val = torch.min(x_detached, self.min_val)
        self.max_val = torch.max(x_detached, self.max_val)
        return x

    def calculate_qparams(self, dtype=torch.qint8):

        qmin = torch.iinfo(dtype).min
        qmax = torch.iinfo(dtype).max

        scale = (self.max_val - self.min_val) / float(qmax - qmin)

        if scale == 0.0:
            scale = torch.tensor(1e-8)

        zero_point = qmin - torch.round(self.min_val / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax)
        zero_point = zero_point.to(torch.int)

        return scale, zero_point
```

### 每张量与更细粒度的量化 (quantization)

就像权重 (weight)一样，激活可以使用不同粒度进行量化：

- **每张量：** 对层生成的整个激活张量使用单一的尺度因子 sss 和零点 zzz。这是最简单的方法，开销最低。
- **每通道/组：** 对于可能具有不同激活统计数据的卷积层或线性层，为每个通道或每组通道使用独立的量化参数 (parameter)可以提高精度。
- **每token（Transformer）：** 在Transformer中，激活值通常具有对应于序列长度和隐藏大小的维度。按每个token进行量化（为序列中每个token的隐藏维度向量 (vector)独立计算 sss 和 zzz）有时能更准确地捕获沿序列长度维度的变化，尽管会增加开销。

选择取决于特定的层类型、观察到的激活分布以及可接受的性能开销。每张量量化因其简单性和效率而普遍使用，但更细粒度的量化可能需要在敏感层中恢复精度。

### 校准期间的异常值处理

如前所述，异常值严重影响最小-最大校准，也可能对MSE/熵方法产生负面影响。一种常见的缓解方法是**截断**。在计算量化 (quantization)参数 (parameter)之前，观察到的激活范围根据百分位数进行截断。例如，不是使用绝对最小值和最大值，可以使用第1和第99百分位数，或第0.1和第99.9百分位数。

G

cluster₀

原始范围 (最小-最大)

cluster₁

截断范围 (例如, 1%-99%)

Min
最小值

---- 分布主体 ----

Min->Bulk1

Max
最大值 (异常值)

Bulk1->Max

P1
第1百分位数

---- 分布主体 ----

P1->Bulk2

P99
第99百分位数

Bulk2->P99

> 截断在计算量化参数前移除极端异常值，可能提高大多数值的精度，代价是截断的异常值被饱和。

截断有助于将量化范围集中在数据主体上，提高典型值的精度。然而，它对截断的异常值引入了饱和，如果这些异常值携带重要信息，可能有害。选择正确的截断阈值通常需要经验调整。

### 静态量化 (quantization)与动态量化

激活量化参数 (parameter)可以离线（静态）或即时（动态）确定：

- **静态量化（训练后静态量化 - PTSQ）：** 这是性能敏感应用中最常见的方法。每个激活张量的尺度 sss 和零点 zzz 使用上述校准过程*一次性*确定。这些固定参数随后存储并在推理 (inference)期间使用。这使得量化和与前序操作的潜在融合能够高效进行，而无需范围计算的运行时开销。
- **动态量化（训练后动态量化 - PTDQ）：** 在此方法中，激活张量的最小/最大范围（以及 sss 和 zzz）在推理期间对每个输入*动态*计算。这避免了对校准数据集的需求。虽然应用起来更简单，但计算范围的运行时成本通常会抵消使用低精度算术带来的任何计算加速。其主要好处通常是减少内存占用和带宽，因为激活值仍可以低精度格式存储和传输，即使计算涉及动态反量化。对于延迟很重要的LLM，激活的动态量化使用频率低于静态量化。

### 在量化 (quantization)感知训练 (QAT) 中的作用

量化感知训练直接将量化效果的模拟（包括权重 (weight)*和激活*）集成到训练循环中。它使用“伪量化”节点来模拟在前向传播期间量化和反量化激活的过程，同时允许梯度在反向传播 (backpropagation)期间通过。

```python
import torch
import torch.nn as nn
import torch.ao.quantization as quant

class QuantizableLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 256)
        self.relu = nn.ReLU()

        self.activation_quant_stub = quant.QuantStub()
        self.activation_dequant_stub = quant.DeQuantStub()

    def forward(self, x):

        x = self.activation_quant_stub(x)

        x = self.linear(x)
        x = self.relu(x)

        x = self.activation_dequant_stub(x)
        return x
```

通过在训练期间让模型接触量化噪声，QAT 允许网络调整其权重和激活分布，变得对量化误差更具弹性。这通常会产生明显更高的精度，尤其是在目标位宽非常低（如 INT4）时，或处理高度敏感的激活分布时。

### 总结

重要考量包括：

- **动态范围：** 激活随输入而变动，可以有宽广、不可预测的范围，使其对量化 (quantization)误差敏感。
- **校准：** 使用代表性数据选择适当的量化参数 (parameter) (s,zs, zs,z) 非常重要。最小-最大、MSE和熵等方法在简单性和稳定性之间提供不同的权衡。
- **异常值处理：** 截断等技术通常是必要的，以防止极端值主导量化范围。
- **粒度：** 每张量量化很常见，但更细粒度（每通道、每token）的量化可能为了精度所需。
- **静态与动态：** 静态量化（校准后参数固定）通常在性能方面更受青睐；动态量化避免了校准，但增加了运行时开销。
- **QAT：** 使用模拟量化训练模型通常能带来最佳精度，尤其是在激进的量化方案中。

成功应对这些考量可以显著减少内存带宽并潜在提升计算速度，使大型模型更适合部署。

获取即时帮助、个性化解释和交互式代码示例。

---

### 网络剪枝（结构化与非结构化）

# 网络剪枝（结构化与非结构化）

量化 (quantization)侧重于减小模型中使用数字的精度，而网络剪枝则采用不同方式：它旨在移除被认为不重要的参数 (parameter)（权重 (weight)）甚至整个结构部件，有效地使模型更稀疏。直观想法是，大型、过参数化模型通常包含大量冗余，移除部分内容可能不会显著影响性能，特别是在重新训练或微调 (fine-tuning)之后。剪枝可以显著减小模型大小，并通过减少计算量来加快推理 (inference)速度。

剪枝主要分为两类：非结构化剪枝和结构化剪枝。

### 非结构化剪枝

非结构化剪枝作用于最细粒度：模型层内的单个权重 (weight)。最常用的方式是**基于幅度的剪枝**。核心思想很简单：绝对值较小的权重对网络输出的贡献较小，被认为不那么重要。

进行幅度剪枝，通常需要：

1. **定义稀疏目标：** 确定要移除的权重比例（例如，50%稀疏度意味着移除一半权重）。
2. **权重排序：** 收集模型中（或特定层内）的所有权重，并按其绝对值大小进行排序。
3. **确定阈值：** 找到与所需稀疏度对应的幅度阈值。例如，如果目标是50%稀疏度，阈值将是权重绝对值的中位数。
4. **创建掩码：** 创建一个与权重张量形状相同的二进制掩码。将低于阈值的权重对应的掩码元素设为0，高于阈值的设为1。
5. **应用掩码：** 在前向传播（以及微调 (fine-tuning)时的反向传播 (backpropagation)）期间，将权重与此掩码相乘，有效地将剪枝后的权重置零。

以下是一个PyTorch代码片段，说明了基于幅度为单个线性层创建掩码的核心思想：

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

layer = nn.Linear(100, 50)

amount_to_prune = 0.3

prune.l1_unstructured(layer, name="weight", amount=amount_to_prune)

print(layer.weight)

prune.remove(layer, 'weight')
print(layer.weight)
```

**非结构化剪枝的优点：**

- **高稀疏度潜力：** 通常可以在不显著损失精度的情况下移除大部分权重，尤其是在高度过参数 (parameter)化的模型中。
- **粒度细：** 只针对最不重要的单个连接。

**非结构化剪枝的缺点：**

- **不规则稀疏性：** 导致稀疏权重矩阵中的零不规则地分散。这种模式难以被标准硬件（GPU、TPU）和库（cuBLAS、cuDNN）有效加速，因为它们针对密集矩阵操作进行了优化。
- **专用硬件/软件：** 实现显著的推理 (inference)加速通常需要专门的硬件或软件库，以高效处理稀疏计算（例如，CSR/CSC等稀疏矩阵格式）。
- **元数据开销：** 存储非零元素的索引（稀疏掩码）会增加一些内存开销，部分抵消了移除权重带来的大小减小，尤其是在较低稀疏度时。

### 结构化剪枝

结构化剪枝不是移除单个权重 (weight)，而是移除整个、明确定义的参数 (parameter)块或组。这可能包括移除：

- **神经元/滤波器：** 权重矩阵中对应特定神经元的整个行或列。在卷积神经网络 (neural network)（CNN）中，这类似于移除整个滤波器。在Transformer中，这可能意味着移除前馈网络（FFN）层中的神经元。
- **注意力头：** 移除多头注意力 (multi-head attention)层中完整的注意力头。
- **层：** 在极端情况下，整个层也可能被移除。

移除结构的依据可以不同。它可能基于结构内权重的聚合幅度（例如，与神经元相关的权重的L2范数）、神经元在数据集上的平均激活值，或与结构对模型输出或损失的贡献相关的更复杂的度量。

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

layer = nn.Linear(100, 50)

num_neurons_to_prune = 10

neuron_norms = torch.norm(layer.weight.data, p=2, dim=1)

threshold = torch.kthvalue(neuron_norms, k=num_neurons_to_prune).values
indices_to_prune = torch.where(neuron_norms <= threshold)[0]

prune.ln_structured(
    layer,
    name="weight",
    amount=num_neurons_to_prune,
    n=2,
    dim=0
)

prune.remove(layer, 'weight')
```

**结构化剪枝的优点：**

- **硬件友好：** 产生更小、更密集的矩阵或移除整个组件。标准硬件和库可以高效处理结果模型。
- **直接加速：** 通常能立即带来推理 (inference)加速，无需专门的稀疏计算库。
- **实现更简单：** 通过移除整个块来修改架构有时比管理稀疏矩阵格式更简单。

**结构化剪枝的缺点：**

- **粒度更粗：** 移除整个结构可能不够精确，并且可能比仅移除最不重要的单个权重对精度影响更大，尤其是在较高剪枝率下。
- **稀疏度限制较低：** 与非结构化剪枝相比，通常在精度开始显著下降之前，实现的整体稀疏度更低。

### 比较结构化与非结构化剪枝

结构化剪枝和非结构化剪枝的选择取决于具体的目标和约束：

| 特性 | 非结构化剪枝 | 结构化剪枝 |
| --- | --- | --- |
| **粒度** | 单个权重 (weight) | 神经元、注意力头、层、通道 |
| **稀疏模式** | 不规则 | 规则（更小、更密集的张量/层） |
| **硬件加速** | 困难（需要专门支持） | 更容易（使用标准密集操作） |
| **潜在稀疏度** | 更高 | 通常更低 |
| **实现** | 掩码管理、稀疏核 | 架构修改、密集核 |
| **精度影响** | 可能更低（在高稀疏度下） | 可能更高（在相同稀疏度下） |

### 剪枝与微调 (fine-tuning)

几乎所有剪枝方法的一个重要方面是需要**微调**。简单地移除权重 (weight)或结构通常会降低模型性能。为了恢复精度，剪枝后的模型必须在原始数据集或相关任务特定数据集上重新训练（微调）若干轮次。在此微调阶段，未剪枝的权重会调整以弥补移除的组件。

剪枝也可以迭代进行：剪枝一小部分权重，微调，再次剪枝，再次微调，依此循环。这种渐进过程通常比一次性剪枝模型大部分内容产生更好结果。

网络剪枝提供了一种有效方法来减少LLM的计算开销。尽管非结构化剪枝预示着更高的压缩比，但其实际好处通常取决于专用硬件或软件。结构化剪枝通过创建更小、更密集的模型，为在标准硬件上实现加速提供了更直接的途径，尽管这可能以牺牲较低的最大稀疏度为代价。这两种方法通常都需要仔细微调以恢复模型的预测能力。

获取即时帮助、个性化解释和交互式代码示例。

---

### 知识蒸馏

# 知识蒸馏

知识蒸馏 (knowledge distillation)提供了一种独特的模型压缩方法。它不是通过量化 (quantization)或剪枝直接修改大型模型的参数 (parameter)，而是侧重于将*知识*从一个大型预训练 (pre-training)模型（“教师”模型）转移到一个更小、更高效的模型（“学生”模型）。目的是训练一个学生模型，使其通过从教师模型提供的更丰富的输出信号中学习，从而比从头开始使用相同架构训练的模型获得明显更好的性能。这使得学生模型更适合计算资源有限或对延迟有严格要求的部署环境。

其主要思想在于观察到，一个训练有素的大型模型能从数据中捕获复杂的模式和细微特征，这些不仅体现在其最终预测中，也体现在其内部表示和输出概率分布中。学生模型通过最小化一个损失函数 (loss function)来学习，该函数促使它模仿教师模型的这些行为，同时还使用真实标签来学习原始任务目标。

### 知识迁移机制

有几种策略可用于将知识从教师模型迁移到学生模型：

1. **匹配输出Logits（软目标）：** 这是知识蒸馏 (knowledge distillation)最常见的形式。学生模型不仅单独使用硬真实标签（例如，one-hot编码向量 (vector)）进行训练，还被训练去匹配教师模型在可能输出类别或token上产生的概率分布。为了提供更丰富的学习信号，教师模型和学生模型的输出通常使用softmax函数中的温度参数 (parameter)（TTT）进行“软化”：

   pi=exp⁡(zi/T)∑jexp⁡(zj/T)p\_i = \frac{\exp(z\_i / T)}{\sum\_j \exp(z\_j / T)}pi​=∑j​exp(zj​/T)exp(zi​/T)​

   这里，ziz\_izi​ 表示类别 iii 的logits。较高的温度（T>1T > 1T>1）会生成一个更柔和的类别概率分布，从而显示更多关于教师模型内部“置信度”以及类别之间相似结构的信息。温度 T=1T=1T=1 对应于标准softmax。蒸馏损失通常是教师模型（pTp^TpT）和学生模型（pSp^SpS）软化概率分布之间的Kullback-Leibler（KL）散度或均方误差（MSE）：

   LKD=KL(pT∣∣pS)=∑ipiTlog⁡(piTpiS)L\_{KD} = KL(p^T || p^S) = \sum\_i p^T\_i \log\left(\frac{p^T\_i}{p^S\_i}\right)LKD​=KL(pT∣∣pS)=i∑​piT​log(piS​piT​​)

   或

   LKD=MSE(zT,zS)L\_{KD} = MSE(z^T, z^S)LKD​=MSE(zT,zS)

   其中，zTz^TzT 和 zSz^SzS 分别是教师模型和学生模型的 logits（softmax之前的输出）。直接使用logits与MSE有时会更简单，并且同样有效。
2. **匹配中间特征：** 知识也可以通过鼓励学生模型复制教师模型中间层的激活或隐藏状态来迁移。这迫使学生学习相似的内部表示。损失函数 (loss function)（通常是MSE）计算教师模型和学生模型相应层特征图之间的差异。

   LFeature=MSE(fT(x),fS(x))L\_{Feature} = MSE(f^T(x), f^S(x))LFeature​=MSE(fT(x),fS(x))

   这里，fT(x)f^T(x)fT(x) 和 fS(x)f^S(x)fS(x) 表示输入 xxx 在教师模型和学生模型选定层中的特征激活。这里的挑战是层的对齐 (alignment)，特别是如果架构差异很大。通常，会学习线性变换，以便在计算损失之前将学生模型的特征映射到教师模型特征的维度。
3. **匹配注意力机制 (attention mechanism)：** 对于基于Transformer的模型，教师模型学习到的注意力模式包含token之间有价值的关系信息。蒸馏可以包括训练学生模型生成与教师模型相似的注意力图。损失是根据相应层或头的注意力权重 (weight)矩阵之间的差异计算的。

### 训练过程

知识蒸馏 (knowledge distillation)的标准训练设置包括：

1. **一个预训练 (pre-training)的教师模型：** 该模型在蒸馏过程中通常保持冻结。
2. **一个较小的学生模型架构：** 这是我们希望训练并最终部署的模型。
3. **一个包含真实标签的数据集：** 用于标准任务损失。

学生模型使用一个组合损失函数 (loss function)进行训练，该函数是标准任务损失（LTaskL\_{Task}LTask​，例如，针对真实标签的交叉熵损失）和蒸馏损失（LKDL\_{KD}LKD​）的加权和：

LTotal=αLTask+(1−α)LKDL\_{Total} = \alpha L\_{Task} + (1 - \alpha) L\_{KD}LTotal​=αLTask​+(1−α)LKD​

超参数 (parameter) (hyperparameter) α\alphaα（通常在0到1之间）平衡了匹配真实数据与模仿教师模型的重要性。用于软化logits的温度 TTT 是另一个重要的超参数。

这是一个PyTorch代码片段，展示了使用KL散度进行logits匹配的损失计算逻辑：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

teacher_model.eval()
student_model.train()

optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)

temperature = 4.0
alpha = 0.3

criterion_task = nn.CrossEntropyLoss()

criterion_kd = nn.KLDivLoss(reduction='batchmean')

optimizer.zero_grad()

student_logits = student_model(inputs)

with torch.no_grad():
    teacher_logits = teacher_model(inputs)

loss_task = criterion_task(student_logits, labels)

student_log_probs_soft = F.log_softmax(student_logits / temperature, dim=-1)
teacher_probs_soft = F.softmax(teacher_logits / temperature, dim=-1)

loss_kd = criterion_kd(student_log_probs_soft,
                       teacher_probs_soft) * (temperature ** 2)

total_loss = alpha * loss_task + (1 - alpha) * loss_kd

total_loss.backward()
optimizer.step()

print(
    f"任务损失: {loss_task.item():.4f}, "
    f"KD 损失: {loss_kd.item():.4f}, "
    f"总损失: {total_loss.item():.4f}"
)
```

> 此代码片段展示了将标准交叉熵损失与基于KL散度的蒸馏损失结合（使用软化输出）的主要逻辑。`temperature` 和 `alpha` 参数控制蒸馏过程。

### 学生架构的选择

学生模型的架构通常选择比教师模型小很多、速度快很多。这可能包括：

- 减少层数。
- 减小隐藏嵌入 (embedding)维度（dmodeld\_{model}dmodel​）。
- 使用更少的注意力头。
- 采用更高效的注意力机制 (attention mechanism)或层类型。

学生模型不一定是教师模型的严格子集。可以尝试不同的架构选择，只要学生模型具有足够的容量，能够有效地学习目标任务的蒸馏知识即可。

### 优点与考量

**优点：**

- **明显压缩：** 与教师模型相比，可以在模型大小和推理 (inference)计算方面取得大幅减少。
- **性能提升：** 学生模型通常比从头开始训练的相同架构模型表现更好，因为它们受益于教师模型学习到的泛化能力。
- **“暗知识”的迁移：** 教师模型提供的软化概率比硬标签单独传递的类别关系信息更丰富。
- **灵活性：** 可应用于各种任务和模型架构。

**缺点与考量：**

- **依赖教师模型：** 需要访问一个大型、训练有素的教师模型。学生模型的质量上限取决于教师模型的质量。
- **训练复杂性：** 训练过程涉及调整额外的超参数 (parameter) (hyperparameter)（α\alphaα, TTT）以及管理两个模型（至少在训练期间）。
- **超参数调整：** 找到匹配教师模型和拟合真实数据之间的最佳平衡（α\alphaα），以及最佳温度（TTT），通常需要仔细的实验。
- **架构不匹配：** 在非常不同的架构之间（例如，RNN教师模型到Transformer学生模型）蒸馏知识可能具有挑战性，特别是对于基于特征的蒸馏。
- **任务特异性：** 有效性可能因任务以及教师模型和学生模型之间的容量差距而异。

知识蒸馏 (knowledge distillation)提供了一种有效的方法，用于创建更小、更高效的语言模型，这些模型保留了其大型对应模型的大部分预测能力，使其成为与量化 (quantization)和剪枝并列的LLM压缩工具箱中有价值的方法。

获取即时帮助、个性化解释和交互式代码示例。

---

### 评估性能与压缩的权衡

# 评估性能与压缩的权衡

应用量化 (quantization)、剪枝或知识蒸馏 (knowledge distillation)等压缩方法会带来一个基本取舍：效率的提升（更小的尺寸、更快的推理 (inference)、更低的内存占用）通常会以模型性能的某种程度下降为代价。选择合适的压缩策略和配置需要细致的评估，以便为您的特定应用需求找到一个可接受的平衡点。本节提供关于如何系统地衡量和比较这些权衡的指导。

## 定义评估维度

为理解压缩的影响，我们需要沿着两个主要维度来衡量变化：模型性能和资源效率。

### 性能指标

性能指标的选择很大程度上取决于大语言模型 (LLM)的用途。评估与您部署场景最相关的指标是很重要的。

- **内在指标：** 像**困惑度**（第21章有讨论）这样的指标衡量基础的语言建模能力。较低的困惑度通常表示模型与训练数据分布更契合。虽然在开发过程中有用，但困惑度的变化并不总是直接与特定下游应用的性能相关。压缩后困惑度显著增加通常是一个警告信号。
  "\* **外在指标：** 评估在**下游任务**（第22章有提及）上的性能通常更能体现实用性。这包括微调 (fine-tuning)（如适用）或在零样本/少样本设置中使用压缩模型，用于GLUE、SuperGLUE等基准测试，或与您的应用相关的自定义任务（例如，摘要的ROUGE分数、问答的F1分数、分类准确率）。"
- **对齐 (alignment)与安全指标：** 如果模型经过对齐调整（第25、26章），压缩可能会影响其有用性、诚实性或无害性。评估可能涉及特定的对齐基准或人工评估协议，以检查所需行为的退化或不希望行为的出现。

### 效率指标

压缩带来的效率提升应该在实际中衡量，在目标部署硬件和软件堆栈上进行。

- **模型大小：** 这通常是最直接的指标，以兆字节（MB）或千兆字节（GB）衡量。它直接影响存储需求和下载时间。量化 (quantization)通常提供可预测的大小减小（例如，INT8相比FP32可将权重 (weight)大小减少约4倍）。剪枝的影响取决于达到的稀疏度。蒸馏在设计上会产生一个更小的模型。
- **推理 (inference)延迟：** 衡量处理单个输入或生成单个令牌所需的时间（例如，每令牌毫秒数）。这对于交互式应用来说非常重要。延迟很大程度上取决于硬件（GPU、CPU、专用加速器）、批量大小、序列长度以及压缩技术的具体实现（例如，优化低精度内核的可用性）。
- **吞吐量 (throughput)：** 衡量每单位时间处理的请求或令牌数量（例如，每秒令牌数）。对于同时服务许多用户的应用很重要。批处理策略显著影响吞吐量。
- **内存占用：** 加载和运行模型所需的RAM或VRAM量。较低精度格式显著减少权重所需的内存。KV缓存（第28章）等技术也会影响生成过程中的运行时内存使用。

## 建立基线

在评估压缩模型之前，您必须建立一个可靠的基线。在目标硬件和评估数据集上衡量您原始、未压缩模型的性能和效率指标。此基线作为所有压缩版本进行比较的参考点。

```python
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "您的原始大语言模型检查点"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_dataset = [...]
test_prompt = "从前"
num_tokens_to_generate = 50

original_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
original_model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)

with torch.no_grad():
    baseline_perplexity = evaluate_perplexity(
        original_model, tokenizer, eval_dataset, device
    )
    baseline_downstream_score = evaluate_downstream_task(
        original_model, tokenizer, ...
    )
    baseline_size_mb = get_model_size_mb(original_model)

    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    start_time = time.time()
    _ = original_model.generate(**inputs, max_new_tokens=num_tokens_to_generate)
    end_time = time.time()
    baseline_latency_ms = (
        (end_time - start_time) * 1000 / num_tokens_to_generate
    )

print(f"基线指标:")
print(f"  困惑度: {baseline_perplexity:.2f}")
print(f"  下游任务分数: {baseline_downstream_score:.4f}")
print(f"  大小 (MB): {baseline_size_mb:.1f}")
print(f"  延迟 (毫秒/令牌): {baseline_latency_ms:.1f}")

baseline_metrics = {
    "perplexity": baseline_perplexity,
    "downstream_score": baseline_downstream_score,
    "size_mb": baseline_size_mb,
    "latency_ms_per_token": baseline_latency_ms,
}
```

## 比较压缩策略

一旦有了基线，应用不同的压缩技术和配置，然后使用*相同*的指标和程序重新评估。

- **量化 (quantization)：** 比较不同比特级别（例如，INT8、INT4）的训练后量化（PTQ）和量化感知训练（QAT）。PTQ更简单，但在较低比特宽度下可能会导致更大的性能下降。QAT需要更多精力（重新训练），但通常能更好地保持性能。评估目标硬件上的模型准确度和实际加速效果，因为理论加速如果没有优化内核并不总是能实现。
- **剪枝：** 评估不同的稀疏度级别（例如，20%、40%、60%稀疏度）。比较非结构化剪枝（移除单个权重 (weight)）与结构化剪枝（移除整个神经元或注意力头）。虽然非结构化剪枝在给定准确度下降的情况下可能提供更高的压缩比，但结构化剪枝由于其规则性，通常在标准硬件上带来更实际的加速。衡量随稀疏度增加而产生的性能下降。
- **知识蒸馏 (knowledge distillation)：** 训练不同大小或架构的学生模型。评估学生模型相对于原始基线和更大的教师模型的性能。权衡包括学生模型的训练成本与其最终大小、速度以及相对于基线的性能。

## 可视化权衡

散点图在可视化性能与效率之间的关系方面很有效。在一个轴上绘制性能指标（例如，下游任务准确率），在另一个轴上绘制效率指标（例如，延迟或模型大小）。每个点代表一个特定的压缩模型配置。

FP32 基线INT8 PTQINT8 QATINT4 PTQ60% 剪枝蒸馏 (小)10090807060500.780.80.820.84

> 特定任务上的性能准确率与每生成令牌的平均延迟对比。每个点代表一个不同的模型版本（基线或压缩版本）。通常更偏好较低的延迟（左侧）和较高的准确率（上方）。

这样的图表有助于识别“帕累托前沿”——这是一组模型，您无法在不牺牲另一个指标（例如，降低准确率）的情况下改进一个指标（例如，减少延迟）。位于此前沿的模型代表了所评估配置下最佳可达成的权衡。

## 硬件和软件堆栈依赖性

在用于部署的特定硬件和软件环境中进行效率评估（延迟、吞吐量 (throughput)、内存使用）是绝对必要的。

- **硬件：** INT8量化 (quantization)等技术带来的加速效果很大程度上依赖于GPU或加速器对低精度计算的支持（例如，NVIDIA Tensor Cores）。在一代GPU上观察到的加速效果可能在另一代或在CPU/TPU上差异很大。同样，结构化剪枝的益处取决于硬件/库是否可以利用由此产生的稀疏性。
- **软件：** 推理 (inference)框架（如PyTorch、ONNX Runtime、TensorRT、vLLM）及其配置起着重要作用。优化后的库通常为量化或稀疏操作提供专用内核。没有这些，压缩的理论益处可能不会转化为实际的延迟降低。

"因此，衡量`毫秒/令牌`或`令牌/秒`需要模型在目标部署堆栈中运行。简单的FLOP计数或参数 (parameter)计数不足以作为速度的衡量标准。"

## 做出决策

很少有单一的“最佳”压缩模型。最佳选择由您的特定应用的约束和要求决定：

- **延迟敏感型应用：** （例如，实时聊天机器人）可能会优先考虑低延迟，接受稍大一点的模型或轻微的性能下降。激进的量化 (quantization)（INT4）或适度的剪枝可能适用。
- **资源受限环境：** （例如，移动或边缘设备）可能会优先考虑最小的模型大小和内存占用，可能容忍较低的准确率。量化和蒸馏通常是主要选择。
- **高准确率要求：** （例如，科学分析、复杂推理 (inference)）可能只允许极小的性能下降，倾向于选择不那么激进的压缩方式，如INT8 QAT或非常轻微的剪枝，即使效率提升不明显。

评估过程通常是迭代的。您可以尝试多种压缩技术和设置，使用上述方法衡量它们的影响，可视化权衡，并选择最符合您特定性能目标和资源预算的配置。始终与未压缩的基线进行比较，以了解每种压缩方法的相对成本和益处。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 28 Efficient Inference Strategies

### 自回归解码中的挑战

# 自回归解码中的挑战

大型语言模型，尤其是那些基于 Transformer 架构的模型，通过自回归 (autoregressive)过程生成文本。这表示它们基于目前已生成的所有标记 (token)来预测序列中的下一个标记。尽管功能强大，但这种顺序的、步进式生成方法带来了显著的计算和内存瓶颈，特别是在处理长序列或为实时应用程序提供服务时。

### 生成的顺序特性

基本的挑战源于自回归 (autoregressive)解码的核心定义。为了生成第 iii 个标记 (token)，模型需要将前 i−1i-1i−1 个已生成的标记序列作为输入。这就产生了固有的顺序依赖性：在已知当前标记之前，无法计算下一个标记。

请看一个 PyTorch 中的简化生成循环：

```python
import torch
import torch.nn.functional as F

def generate_next_token(model, input_ids):
    """给定当前序列，生成下一个标记。"""
    with torch.no_grad():

        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]

        probs = F.softmax(next_token_logits, dim=-1)

        next_token_id = torch.argmax(probs, dim=-1).unsqueeze(-1)
    return next_token_id

model = ...
tokenizer = ...

prompt = "The quick brown fox"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

max_new_tokens = 50
for _ in range(max_new_tokens):
    next_token = generate_next_token(model, input_ids)

    if next_token.item() == tokenizer.eos_token_id:
        break

    input_ids = torch.cat([input_ids, next_token], dim=-1)

generated_text = tokenizer.decode(input_ids[0])
print(generated_text)
```

此循环展示了迭代过程。每次调用 `generate_next_token` 都会对模型执行一次完整的正向传播（或其很大一部分）。对于长度为 NNN 的序列，生成 MMM 个新标记需要 MMM 次单独的正向传播。这种顺序执行直接导致了延迟；总时间大致与生成的标记数量成正比。

### 内存带宽限制

Transformer 中的一个主要瓶颈，尤其是在推理 (inference)期间，是内存带宽。在自回归 (autoregressive)生成的每个步骤中，模型都需要计算注意力分数。标准的自注意力 (self-attention)机制 (attention mechanism)需要计算当前标记 (token)的查询向量 (vector)与序列中*所有*先前标记的键向量之间的相互关系。

为了有效进行此操作，为所有先前标记计算的键 (KKK) 和值 (VVV) 通常保存在 GPU 内存中。随着序列变长，这个 K/VK/VK/V 缓存的大小会随序列长度 (nnn) 线性增长。在步骤 iii 的注意力计算需要将这些缓存的 KKK 和 VVV 张量（大小与 iii 成比例）从高带宽内存 (HBM) 加载到计算单元（如 Tensor Cores）中。

Attention(Qi,K1..i,V1..i)=softmax(QiK1..iTdk)V1..i\text{Attention}(Q\_i, K\_{1..i}, V\_{1..i}) = \text{softmax}\left(\frac{Q\_i K\_{1..i}^T}{\sqrt{d\_k}}\right) V\_{1..i}Attention(Qi​,K1..i​,V1..i​)=softmax(dk​​Qi​K1..iT​​)V1..i​

在这里，QiQ\_iQi​ 是在步骤 iii 处生成的标记的查询，而 K1..iK\_{1..i}K1..i​ 和 V1..iV\_{1..i}V1..i​ 代表从 1 到 iii 的所有标记的键和值。在每个步骤中从内存中获取 K1..iK\_{1..i}K1..i​ 和 V1..iV\_{1..i}V1..i​ 会消耗大量的内存带宽。对于大型模型和长序列，数据在 HBM 和计算单元之间传输所花费的时间可能会主导整体延迟，使得该过程受内存限制而非计算限制。

G

T1

标记 1
计算 K1, V1

T2

标记 2
计算 Q2
加载 K1, V1
计算 K2, V2

T1->T2

 依赖于 T1

T3

标记 3
计算 Q3
加载 K1,K2, V1,V2
计算 K3, V3

T2->T3

 依赖于 T2

T\_dots

...

T3->T\_dots

Ti

标记 i
计算 Qi
加载 K1..i-1, V1..i-1
计算 Ki, Vi

T\_dots->Ti

> 自回归生成中的顺序依赖和不断增加的内存访问。每一步都需要加载先前计算的键和值。

### 计算成本的扩展

尽管内存带宽通常是单标记 (token)生成延迟的主要瓶颈，但计算成本本身也不容忽视。自注意力 (self-attention)计算每层的计算复杂度为 O(n2d)O(n^2 d)O(n2d)，nnn 为序列长度，ddd 为模型的隐藏维度。尽管像 KV 缓存（下文讨论）这样的技术避免了对过去标记重新计算键和值，但在步骤 iii 的注意力分数计算仍然涉及到当前查询 QiQ\_iQi​ 与所有先前键 K1..iK\_{1..i}K1..i​ 之间的点积计算，使得该部分计算的成本与 i×di \times di×d 成正比。随后与值向量 (vector) V1..iV\_{1..i}V1..i​ 的乘法也以类似方式扩展。

将所有层和前馈网络的计算相加，每个生成步骤都需要大量的浮点运算 (FLOPs)，并且这个数量会随着当前序列长度的增加而增加。对于非常长的序列，这种计算需求增加了整体延迟。

### 吞吐量 (throughput)方面的挑战

自回归 (autoregressive)解码的顺序特性也给吞吐量带来了挑战，吞吐量是指所有并发请求在单位时间内生成的输出标记 (token)数量。尽管可以在并行批次中处理多个独立的生成请求，但每个序列*内部*的生成仍然是顺序的。这限制了硬件能够被充分利用的程度，特别是当序列长度差异很大或单个请求需要非常长的输出时。优化批处理策略变得重要，但它并没有从根本上改变每个序列的顺序约束。

这些是与标准自回归解码过程相关的固有挑战。顺序依赖性、内存带宽限制、扩展的计算成本和吞吐量约束使得本章后续部分论述的专门优化技术成为必需。KV 缓存、优化的注意力实现以及推测解码等策略直接针对这些瓶颈，以使大型语言模型推理 (inference)更快、更有效。

获取即时帮助、个性化解释和交互式代码示例。

---

### 键值（KV）缓存

# 键值（KV）缓存

自回归 (autoregressive)生成，即基于先前已生成的标记 (token)逐个生成文本的过程，是大型语言模型生成回复的中心方式。然而，简单的实现会遇到一个很大的性能障碍。Transformer架构中的自注意力 (self-attention)机制 (attention mechanism)是此过程的核心。为了生成下一个标记，例如标记 t+1t+1t+1，标准的自注意力计算需要基于*所有*之前的标记 1...t1...t1...t 来计算查询（Q）、键（K）和值（V），然后计算注意力分数。当生成随后的标记 t+2t+2t+2 时，这个完整过程会使用标记 1...t+11...t+11...t+1 再次进行。请注意这种重复：在生成标记 t+1t+1t+1 期间为标记 1...t1...t1...t 计算的键和值向量 (vector)，与生成标记 t+2t+2t+2 时前 ttt 个标记所需的向量是相同的。在每一步重复这些计算是计算上的浪费，特别是当序列长度增加时。

键值（KV）缓存是一种基本优化技术，其目的在于消除自回归推理 (inference)中的这种重复。其中心思想简单而高效：存储自注意力层中为所有先前标记计算出的键（K）和值（（V）张量，并在随后的生成步骤中重复使用它们。

### KV缓存的工作方式

在Transformer的自注意力 (self-attention)层中，输入序列 XXX 被投影到三个矩阵：查询（QQQ）、键（KKK）和值（VVV）。

Q=XWQ,K=XWK,V=XWVQ = X W\_Q, \quad K = X W\_K, \quad V = X W\_VQ=XWQ​,K=XWK​,V=XWV​

WQ,WK,WVW\_Q, W\_K, W\_VWQ​,WK​,WV​ 为可学习的权重 (weight)矩阵。注意力输出随后按以下方式计算：

注意力(Q,K,V)=softmax(QKTdk)V\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V注意力(Q,K,V)=softmax(dk​​QKT​)V

考虑生成标记 (token) t+1t+1t+1。模型将标记嵌入 (embedding)序列 x1,...,xtx\_1, ..., x\_tx1​,...,xt​ 作为输入。在每个注意力层内，它计算 K1,...,KtK\_1, ..., K\_tK1​,...,Kt​ 和 V1,...,VtV\_1, ..., V\_tV1​,...,Vt​。它还基于*最后*一个标记 xtx\_txt​ 的嵌入（或对应于位置 t+1t+1t+1 的位置嵌入）计算查询向量 (vector) Qt+1Q\_{t+1}Qt+1​。然后，注意力计算使用 Qt+1Q\_{t+1}Qt+1​ 和完整的键集合 K=[K1,...,Kt]K = [K\_1, ..., K\_t]K=[K1​,...,Kt​] 以及值集合 V=[V1,...,Vt]V = [V\_1, ..., V\_t]V=[V1​,...,Vt​]。

现在，考虑生成标记 t+2t+2t+2。输入序列为 x1,...,xt+1x\_1, ..., x\_{t+1}x1​,...,xt+1​。模型需要计算 K1,...,Kt+1K\_1, ..., K\_{t+1}K1​,...,Kt+1​ 和 V1,...,Vt+1V\_1, ..., V\_{t+1}V1​,...,Vt+1​。重要的是，对 K1,...,KtK\_1, ..., K\_tK1​,...,Kt​ 和 V1,...,VtV\_1, ..., V\_tV1​,...,Vt​ 的计算与上一步完全相同，因为它们仅依赖于输入标记 x1,...,xtx\_1, ..., x\_tx1​,...,xt​ 以及固定的权重矩阵 WKW\_KWK​ 和 WVW\_VWV​。

KV缓存善用这个特性。它不是在每一步重新计算所有的键和值，而是：

1. **初始化（步骤1）：** 基于输入 x1x\_1x1​ 计算 K1,V1K\_1, V\_1K1​,V1​。将其存入缓存。
2. **步骤 ttt（生成标记 t+1t+1t+1）：**
   - 基于 xtx\_txt​ 计算 QtQ\_tQt​。
   - 基于 xtx\_txt​ 计算 Kt,VtK\_t, V\_tKt​,Vt​。
   - 获取缓存中的 K1..t−1,V1..t−1K\_{1..t-1}, V\_{1..t-1}K1..t−1​,V1..t−1​。
   - 拼接：Kcache=[K1..t−1,Kt]K\_{cache} = [K\_{1..t-1}, K\_t]Kcache​=[K1..t−1​,Kt​]，Vcache=[V1..t−1,Vt]V\_{cache} = [V\_{1..t-1}, V\_t]Vcache​=[V1..t−1​,Vt​]。
   - 使用 QtQ\_tQt​、KcacheK\_{cache}Kcache​、VcacheV\_{cache}Vcache​ 计算注意力。
   - 用 KcacheK\_{cache}Kcache​ 和 VcacheV\_{cache}Vcache​ 更新缓存。
   - 生成标记 t+1t+1t+1。
3. **步骤 t+1t+1t+1（生成标记 t+2t+2t+2）：**
   - 基于新生成的标记 xt+1x\_{t+1}xt+1​ 计算 Qt+1Q\_{t+1}Qt+1​。
   - 仅基于 xt+1x\_{t+1}xt+1​ 计算*新的* Kt+1,Vt+1K\_{t+1}, V\_{t+1}Kt+1​,Vt+1​。
   - 从上一步获取缓存中的 K1..t,V1..tK\_{1..t}, V\_{1..t}K1..t​,V1..t​。
   - 拼接：Kcache=[K1..t,Kt+1]K\_{cache} = [K\_{1..t}, K\_{t+1}]Kcache​=[K1..t​,Kt+1​]，Vcache=[V1..t,Vt+1]V\_{cache} = [V\_{1..t}, V\_{t+1}]Vcache​=[V1..t​,Vt+1​]。
   - 使用 Qt+1Q\_{t+1}Qt+1​、KcacheK\_{cache}Kcache​、VcacheV\_{cache}Vcache​ 计算注意力。
   - 更新缓存。
   - 生成标记 t+2t+2t+2。
   - ...依此类推。

KV\_Cache

clusterₛtepₜ

步骤 t（生成标记 t+1）

clusterₛtepₜₚlus₁

步骤 t+1（生成标记 t+2）

xt

输入: xₜ

compute\_qkvₜ

计算 Qₜ, Kₜ, Vₜ

xt->compute\_qkvₜ

concatₜ

拼接:
K' = [Kcₐcₕₑ, Kₜ]
V' = [Vcₐcₕₑ, Vₜ]

compute\_qkvₜ->concatₜ

 Kₜ, Vₜ

attentionₜ

注意力(Qₜ, K', V')

compute\_qkvₜ->attentionₜ

 Qₜ

cacheᵣeadₜ

读取缓存:
K₁..ₜ₋₁, V₁..ₜ₋₁

cacheᵣeadₜ->concatₜ

concatₜ->attentionₜ

 K', V'

cache\_writeₜ

写入缓存:
K', V'

concatₜ->cache\_writeₜ

 K', V'

outputₜ

输出: 标记 t+1

attentionₜ->outputₜ

cacheᵣeadₜp1

读取缓存:
K₁..ₜ, V₁..ₜ

cache\_writeₜ->cacheᵣeadₜp1

xtp1

输入: xₜ₊₁

outputₜ->xtp1

作为下一个输入

compute\_qkvₜp1

计算 Qₜ₊₁, Kₜ₊₁, Vₜ₊₁

xtp1->compute\_qkvₜp1

concatₜp1

拼接:
K'' = [Kcₐcₕₑ, Kₜ₊₁]
V'' = [Vcₐcₕₑ, Vₜ₊₁]

compute\_qkvₜp1->concatₜp1

 Kₜ₊₁, Vₜ₊₁

attentionₜp1

注意力(Qₜ₊₁, K'', V'')

compute\_qkvₜp1->attentionₜp1

 Qₜ₊₁

cacheᵣeadₜp1->concatₜp1

concatₜp1->attentionₜp1

 K'', V''

cache\_writeₜp1

写入缓存:
K'', V''

concatₜp1->cache\_writeₜp1

 K'', V''

outputₜp1

输出: 标记 t+2

attentionₜp1->outputₜp1

> 简化的流程图，展示了在步骤 `t` 计算出的键（K）和值（V）如何被缓存并在步骤 `t+1` 复用，这样只需要为新标记 `x_{t+1}` 进行计算。

这大大减少了每个生成标记的计算成本。注意力计算复杂度不再像每个步骤中与序列长度 ttt 的平方大致成比例（如果考虑完整矩阵乘法是 O(t2)O(t^2)O(t2)，或者仅将查询应用于现有键是 O(t)O(t)O(t)），而是与过去标记相关的计算实际变成了常数时间（缓存查找和拼接），主要成本变为计算单个新标记的K和V，并将新查询应用于缓存的键（QKTQK^TQKT 部分为 O(t)O(t)O(t)）。

### 内存使用

尽管KV缓存显著加快了推理 (inference)速度，但它也带来了内存成本。缓存需要为批次中的每个序列，存储所有先前标记 (token)在所有层和所有注意力头中的键和值张量。KV缓存的大小可以估算为：

`缓存大小 ≈ batch_size × num_layers × 2（针对 K 和 V） × num_heads × sequence_length × head_dimension × bytes_per_element`

这种内存占用随 `sequence_length` 线性增长。对于具有多层和多头的模型，以及处理长序列或大批次时，KV缓存会占用大量的GPU内存，有时成为可处理最大序列长度的限制因素。管理这种内存使用是一个重要的考量，它促成了分页注意力或缓存本身的量化 (quantization)等技术，尽管这些内容超出了本基本介绍的范围。

### PyTorch中的实作考量

实作KV缓存通常涉及修改Transformer块的 `forward` 方法（或直接修改注意力模块），使其接受一个可选的 `past_key_values` 参数 (parameter)并返回更新后的 `present_key_values`。

下面是一个（高度简化的）草图，对比了标准注意力计算和使用KV缓存的注意力计算：

```python
import torch
import torch.nn as nn

class SimpleMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim 必须能被 num_heads 整除"
        )

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, past_kv=None):
        batch_size, seq_len, _ = query.size()

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        present_kv = None
        if past_kv is not None:

            past_k, past_v = past_kv

            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

            present_kv = (k, v)
        else:

             present_kv = (k, v)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)

        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous()
        output = output.view(
            batch_size, -1, self.embed_dim
        )
        output = self.out_proj(output)

        return output, present_kv
```

在实践中，像Hugging Face Transformers这样的框架对这种缓存机制进行了抽象。当调用 `generate` 方法或使用带有 `use_cache=True` 参数的模型前向传播时，框架会自动处理生成步骤之间KV缓存的创建、传递和更新。然而，理解其基本原理对认识性能提升和内存影响很重要。

---

### 优化的注意力实现 (FlashAttention)

# 优化的注意力实现 (FlashAttention)

即使有KV缓存来减轻跨时间步的重复计算，自注意力 (self-attention)机制 (attention mechanism)本身在推理 (inference)过程中仍是一个重要的性能瓶颈，尤其对于处理长序列或大批量数据的模型。缩放点积注意力的标准实现，虽然数学上精巧，但往往受限于内存带宽，而非原始计算能力。

### 标准注意力中的内存带宽瓶颈

让我们回顾缩放点积注意力的核心计算：

注意力(Q,K,V)=softmax(QKTdk)V\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V注意力(Q,K,V)=softmax(dk​​QKT​)V

在标准实现中，计算这需要几个步骤，这些步骤需要在GPU的高带宽内存（HBM）及其处理单元之间移动大量数据：

1. **读取Q、K、V：** 查询 (QQQ)、键 (KKK) 和值 (VVV) 矩阵（通常很大）从 HBM 中读取。
2. **计算 S=QKTS = QK^TS=QKT：** 执行一次可能非常大的矩阵乘法。
3. **写入/读取 SSS：** 结果注意力得分矩阵 SSS（大小为 N×NN \times NN×N，其中 NNN 是序列长度）常常必须写回 HBM，因为它可能无法完全放入 GPU 更快的片上 SRAM 中，尤其对于长序列。然后将其读回进行 softmax 计算。
4. **计算 Softmax：** Softmax 函数应用于 SSS。这通常涉及读取 SSS，执行计算，并将结果写回 HBM。
5. **计算 O=softmax(S)VO = \text{softmax}(S)VO=softmax(S)V：** 另一次矩阵乘法从 HBM 中读取 softmax 输出和值矩阵 VVV。
6. **写入 OOO：** 最终输出矩阵 OOO 写回 HBM。

此处的核心瓶颈常常是第 3 步：读取和写入中间 N×NN \times NN×N 矩阵 SSS 的需求。对于序列长度 N=4096N=4096N=4096，如果使用 FP32，该矩阵需要 4096×4096×4 字节≈67 MB4096 \times 4096 \times 4 \text{ 字节} \approx 67 \text{ MB}4096×4096×4 字节≈67 MB，如果使用 FP16 则为 33.5 MB。尽管这看起来可管理，但这些读/写操作发生在注意力层*内部*，并且与在快得多的 SRAM 中执行的计算相比，反复访问相对较慢的 HBM 会消耗大量时间和能量。标准注意力的总内存访问量按 O(N2d+Nd2)O(N^2 d + Nd^2)O(N2d+Nd2) 缩放，其中 ddd 是头维度，但与中间矩阵 SSS 相关的 O(N2)O(N^2)O(N2) 项在大 NNN 时主导内存访问成本。

### FlashAttention: 消除瓶颈

FlashAttention 是一种优化的注意力算法，专门设计用于解决此 I/O 瓶颈。该算法由 Dao 等人（2022 年）提出，其主要创新在于计算*精确*的注意力输出，而无需将完整的 N×NN \times NN×N 注意力得分矩阵 SSS 或中间 softmax 输出写入 HBM。这大大减少了 HBM 和 GPU 核心之间的数据传输量，使计算显着更快且更节省内存。

FlashAttention 通过多种技术组合实现这一点：

1. **分块：** 计算被分解成更小的块或“瓦片”。FlashAttention 不再一次性处理整个 Q,K,VQ, K, VQ,K,V 矩阵，而是从 HBM 加载这些矩阵的较小块到 GPU 的快速片上 SRAM 中。
2. **结合核函数：** 多个操作（例如，计算 SSS 的一个块，应用 softmax，乘以 VVV 的一个块）被组合或“结合”成一个 CUDA 核函数。这最大限度地减少了单独核函数启动的数量，并避免在这些结合步骤之间将中间结果写回 HBM。
3. **在线 Softmax：** 在每个分块计算中，softmax 计算通过数值稳定技术（减去运行最大值）小心执行，以确保正确性，而无需一次性获取整个 SSS 矩阵。在最终结果写回 HBM 之前，输出块 OiO\_iOi​ 使用分块输入 Qi,Kj,VjQ\_i, K\_j, V\_jQi​,Kj​,Vj​ 直接在 SRAM 中计算和累加。

设想将 QQQ 矩阵按行划分，将 K,VK, VK,V 矩阵按列划分（或反之，取决于实现细节）。FlashAttention 将 QQQ 的一个块加载到 SRAM 中。然后，它遍历 KKK 和 VVV 的块，逐一将它们加载到 SRAM 中。对于 SRAM 中每一对 QQQ 和 K,VK, VK,V 块，它计算相应的注意力得分块，应用 softmax 操作（同时维护跨块归一化 (normalization)所需的统计量，如运行最大值和总和），并将结果累加到一个输出块中，该输出块也保留在 SRAM 中。只有初始 QQQ 块的最终输出块才会被写回 HBM。

G

cluster₀

GPU 核心 / SRAM

cluster₁

HBM

Qi

Q 块 (Qi)

Sij

计算 Sᵢj = Qi \* Kjᵀ

Qi->Sij

Kj

K 块 (Kj)

Kj->Sij

Vj

V 块 (Vj)

Softmax\_Acc

在线 Softmax 与累加 Oᵢ

Vj->Softmax\_Acc

Sij->Softmax\_Acc

Oi

输出块 (Oi)

Softmax\_Acc->Oi

累加

Oₕbm

完整 O 矩阵

Oi->Oₕbm

写入块

Qₕbm

完整 Q 矩阵

Qₕbm->Qi

加载块

Kₕbm

完整 K 矩阵

Kₕbm->Kj

加载块 (迭代地)

Vₕbm

完整 V 矩阵

Vₕbm->Vj

加载块 (迭代地)

Sᵢntermediate

完整 S 矩阵 (N x N)
\*未实例化\*

> FlashAttention 通过在 GPU 更快的 SRAM 中以分块方式处理计算，避免将大型中间注意力得分矩阵写入 HBM，从而显著减少了内存 I/O。

### 优势与集成

FlashAttention 在推理 (inference)时的主要优势是速度。通过减少 HBM 访问，它能带来显著的性能提升，通常报告与标准注意力实现相比，性能提升 2 到 4 倍甚至更多，尤其对于 N2N^2N2 项占主导地位的长序列。

此外，由于大型中间矩阵 SSS 未存储在 HBM 中，FlashAttention 需要更少的内存（内存复杂度为 O(Nd+d2)O(Nd + d^2)O(Nd+d2)，而标准注意力的峰值使用为 O(N2+Nd)O(N^2 + Nd)O(N2+Nd)）。这使得在相同的 GPU 内存限制下可以处理更长的序列或使用更大的批量大小，这对于处理各种请求负载的推理服务器来说非常有价值。

将 FlashAttention 集成到您的工作流程中通常很简单，尤其是在现代深度学习 (deep learning)框架中。PyTorch 2.0 及更高版本包含 `torch.nn.functional.scaled_dot_product_attention`，在硬件和输入条件允许时，它会自动尝试使用 FlashAttention 等优化核函数（或类似的内存高效实现）。

```python
import torch
import torch.nn.functional as F
from math import sqrt

batch_size, num_heads, seq_len_q, seq_len_kv, head_dim = 2, 8, 1024, 1024, 64
query = torch.randn(
    batch_size, num_heads, seq_len_q, head_dim,
    device='cuda',
    dtype=torch.float16
)
key = torch.randn(
    batch_size, num_heads, seq_len_kv, head_dim,
    device='cuda',
    dtype=torch.float16
)
value = torch.randn(
    batch_size, num_heads, seq_len_kv, head_dim,
    device='cuda',
    dtype=torch.float16
)
is_causal = True

try:

    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    print(
        "使用了 PyTorch 的 scaled_dot_product_attention 后端 "
        "(可能为 FlashAttention)。"
    )

except RuntimeError as e:

    print(
        f"优化注意力后端失败：{e}。 "
        f"使用手动实现。"
    )

    scale_factor = 1 / sqrt(query.size(-1))
    attn_bias = torch.zeros(
        seq_len_q,
        seq_len_kv,
        dtype=query.dtype,
        device=query.device
    )
    if is_causal:
        temp_mask = torch.ones(
            seq_len_q, seq_len_kv, dtype=torch.bool, device=query.device
        ).tril(diagonal=0)
        attn_bias.masked_fill_(
            temp_mask.logical_not(), float("-inf")
        )

    b, h, n, d = query.shape
    q = query.reshape(b*h, n, d)
    k = key.reshape(b*h, n, d)
    v = value.reshape(b*h, n, d)

    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    attn_output_manual = attn_weight @ v
    attn_output = attn_output_manual.reshape(
        b, h, n, d
    )

print(f"输出形状: {attn_output.shape}")
```

您也可以明确使用 `flash_attn` 包等库中的实现，该包提供对高度优化核函数的直接访问，并可能为默认 PyTorch 调度程序未涵盖的特定情况提供更多控制或支持。

### FlashAttention-2

在原有工作基础上，FlashAttention-2 引入了进一步的优化，尤其侧重于提高并行性以及减少与 GPU 线程块和 warp 之间工作分配相关的潜在瓶颈，尤其是在像 H100 (Hopper 架构) 这样的新型 NVIDIA GPU 上。这些改进通常会比第一版带来额外的加速。

### 注意事项

虽然非常有效，但请记住：

- **硬件支持：** FlashAttention 及其变体依赖于特定的 GPU 功能（例如，足够的 SRAM 大小，Tensor Core 能力），这些功能主要存在于 NVIDIA Ampere (A100) 和 Hopper (H100) 等较新一代中。在旧架构上，性能提升可能不明显或不可用。
- **软件兼容性：** 确保您正在使用兼容的深度学习 (deep learning)框架版本（例如 PyTorch >= 2.0）或包含优化核函数的专用库（`flash_attn`）。请查阅文档以获取有关数据类型 (FP16, BF16)、头维度和掩码选项的具体要求。
- **精确性与实现：** FlashAttention 计算*相同*的数学注意力函数。它是一种实现优化，而非近似方法（与某些其他技术不同）。与朴素实现相比的任何差异只应归因于浮点运算的变化。

通过借助 FlashAttention 等优化的注意力实现，您可以在推理 (inference)时显着减少与注意力机制 (attention mechanism)相关的延迟和内存占用，从而使得部署更大模型和更高效地处理长序列成为可能。这是构建高性能 LLM 推理系统工具包中的一个重要组成部分。

获取即时帮助、个性化解释和交互式代码示例。

---

### 吞吐量批处理策略

# 吞吐量批处理策略

优化单个令牌生成（通过KV缓存和专用注意力机制 (attention mechanism)等方法）能够显著降低每一步的延迟和内存开销，但要最大化LLM推理 (inference)服务器的总体*吞吐量 (throughput)*，就需要同时处理多个请求。GPU是深度学习 (deep learning)的运算主力，在进行大型矩阵乘法时能达到最佳性能。逐个处理请求通常会导致GPU利用率不足，因为控制逻辑和步骤间的数据传输可能占据单个序列的大部分计算时间。批处理策略通过将多个推理请求组合在一起，让GPU并行处理它们，从而分摊开销并增加每秒生成的令牌数量。

## 静态批处理

静态批处理是最简单的方法。在这种方法中，推理 (inference)服务器会等待，直到预设数量的请求（`batch_size`）到达或发生超时。然后这些请求被分组，填充到批处理中最长序列的长度，并一起处理。

填充是指在批处理中较短的序列后添加特殊令牌，使所有序列具有相同的长度。这会生成GPU上进行高效矩阵乘法所需的统一张量形状。注意力掩码在这里很实用，它能确保模型在自注意力 (self-attention)计算期间不会关注这些填充令牌。

```python
import torch

requests = [
    "This is sequence one.",
    "A shorter sequence.",
    "This is the third sequence, and it's quite long."
]

seq1_ids = [101, 2023, 2003, 5537, 2028, 1012, 102]
seq2_ids = [101, 1037, 9087, 5537, 1012, 102]
seq3_ids = [
    101, 2023, 2003, 1996, 2353, 5537, 1010, 1998, 2009, 1005, 1055,
    3747, 2146, 1012, 102
]

sequences = [seq1_ids, seq2_ids, seq3_ids]
max_len = max(len(seq) for seq in sequences)
padding_token_id = 0

padded_sequences = []
attention_masks = []

for seq in sequences:
    pad_len = max_len - len(seq)
    padded_seq = seq + [padding_token_id] * pad_len

    attention_mask = ([1] * len(seq) +
                      [0] * pad_len)

    padded_sequences.append(padded_seq)
    attention_masks.append(attention_mask)

input_ids = torch.tensor(padded_sequences)
attention_mask = torch.tensor(attention_masks)

print("Input IDs Shape:", input_ids.shape)
print("Attention Mask Shape:", attention_mask.shape)
```

**静态批处理的局限：**

- **低效：** 大量计算资源浪费在填充令牌上，尤其当批处理中序列长度差异很大时。批处理时间由最长序列决定。
- **队头阻塞：** 较短的请求可能会经历更高的延迟，因为它们需要等待足够多的请求来形成一个完整批处理，或者等待同一批处理中的一个长请求完成。

## 动态批处理

动态批处理旨在通过在小时间窗内对到达的请求进行分组，而不是等待固定的批处理大小，从而提高GPU利用率，优于静态批处理。这通常会导致批处理中的序列长度差异更大。尽管它仍然需要填充，但服务器可以更频繁地开始处理批处理。

其主要思路是灵活性。服务器不使用固定的`batch_size`，而是在短时间内（例如10毫秒）积累请求，然后将所有已到达的请求作为一个批处理进行处理。

```python

import time
import queue

request_queue = queue.Queue()
MAX_WAIT_TIME_MS = 10
MAX_BATCH_SIZE = 16

def process_batch(batch):

    print(f"Processed batch of size {len(batch)}")

while True:
    batch = []
    start_time = time.time()
    while True:
        try:

            request = request_queue.get_nowait()
            batch.append(request)
            if len(batch) >= MAX_BATCH_SIZE:
                break
        except queue.Empty:

            if (time.time() - start_time) * 1000 > MAX_WAIT_TIME_MS:
                break

            time.sleep(0.001)

    if batch:
        process_batch(batch)
    else:

        time.sleep(0.01)
```

尽管动态批处理优于静态批处理，但它仍然存在填充效率低下的问题，并且在自回归 (autoregressive)生成过程中，整个批处理的进度由最慢（最长）的序列决定。如果一个序列需要500个令牌而其他序列只需要50个，那么批处理槽位将一直被占用，直到500个令牌的生成完成，这会导致批处理中已完成序列的GPU空闲（通常称为“气泡”）。

## 连续批处理

连续批处理（也称为迭代级调度或动态拆分融合）是一种更精巧的方法，旨在通过解决简单批处理方法的不足来最大化吞吐量 (throughput)。它将服务器级别的批处理与生成过程的迭代步骤分离。

连续批处理不是处理一个固定批处理直到所有序列都完成，而是按迭代进行操作：

1. **请求池：** 维护一个当前正在生成中的活动请求池。
2. **迭代批处理：** 在每个生成步骤（即为所有活动序列生成下一个令牌时），从池中选择准备好生成下一个令牌的序列子集。然后从这些序列形成一个批处理。
3. **前向传播：** 对本次迭代的批处理执行一次前向传播（一个令牌生成步骤）。
4. **更新池：**
   - 将新生成的令牌添加到已处理批处理中的每个序列。
   - 检查是否有任何序列达到其停止条件（例如，EOS令牌、最大长度）。如果达到，则将它们从活动池中移除并返回完成的结果。
   - 检查是否有新的传入请求，如果容量允许，则将其添加到活动池中。
5. **重复：** 返回步骤2进行下一次生成迭代。

G

Requests

传入请求

Pool

活动请求池
(序列A, 序列B, 序列C...)

Requests->Pool

 添加新请求

IterBatch

迭代批处理
(序列A[t], 序列B[t])

Pool->IterBatch

 选择准备好的

Completed

已完成请求

Pool->Completed

 完成并移除

GPU

GPU前向传播
(令牌生成)

IterBatch->GPU

 处理

GPU->Pool

 更新状态
 添加新令牌

> 连续批处理的简化流程。从活动请求中形成迭代批处理，由GPU处理，然后用新令牌、已完成序列和传入请求更新请求池。

主要的优点是，只要有*任何*活动序列准备好生成，GPU就会保持忙碌。当一个序列完成时，它在下一次迭代批处理中的位置可以立即被池中另一个序列或新到达的请求占用。这避免了静态/动态批处理中GPU等待批处理中最长序列完成的“气泡”问题。Orca、vLLM、TensorRT-LLM和Text Generation Inference (TGI) 等系统都实现了连续批处理的不同形式。

## 实现考量与权衡

- **KV缓存管理：** 批处理，特别是连续批处理，需要细致地管理KV缓存。批处理中的每个序列都需要在生成步骤中维护自己独立的KV缓存状态。这些可能数量众多且动态变化的缓存的内存分配和访问效率至关重要。PagedAttention是vLLM等系统中用于更高效管理KV缓存内存，减少碎片化的一种方法。
- **吞吐量 (throughput)与延迟：** 批处理在根本上是用增加延迟来换取更高的吞吐量。将请求分组意味着与立即处理相比，某些请求在开始处理前会等待更长时间。然而，整个系统的容量（令牌/秒）会显著增加。批处理策略和参数 (parameter)（例如，最大批处理大小，动态批处理的等待时间）的选择取决于特定应用程序对延迟和吞吐量的要求。更大的批处理大小通常会增加吞吐量，但也会增加平均延迟。
- **框架支持：** 深度学习 (deep learning)框架和专用服务工具包通常提供批处理实用程序。例如，当你向模型或分词 (tokenization)器 (tokenizer)提供输入列表时，Hugging Face的`transformers`库会自动处理填充和注意力掩码。专用的推理 (inference)服务器（Triton、TorchServe、TGI、vLLM）提供更先进的批处理功能，包括动态和连续批处理，并针对生产工作负载进行了优化。

总而言之，批处理对于高效的LLM推理不可或缺。尽管静态批处理简单，但动态批处理，尤其是连续批处理，通过并行处理多个请求并动态管理跨生成步骤的工作负载，在GPU利用率和总体吞吐量方面提供了显著的改进。选择和调整正确的批处理策略是在大规模部署LLM时一个重要考量。

获取即时帮助、个性化解释和交互式代码示例。

---

### 推测解码

# 推测解码

生成 NNN 个词元 (token)需要大型语言模型进行 NNN 次顺序前向传递，这是自回归 (autoregressive)生成中的一个根本限制。尽管像 KV 缓存这样的技术优化了单次前向传递内的计算，这种顺序依赖性依然自然限制了可达到的最大速度。推测解码提供了一种巧妙的方法来并行化此过程的部分内容，目标是在主大型模型每次单次前向传递中生成多个词元，从而降低总体的实际运行延迟。

核心思路在于使用两个模型：

1. **目标模型 (Target Model)：** 大型、高质量语言模型，我们希望精确匹配其输出分布。这是我们最终希望用于生成的模型，但它速度慢。
2. **草稿模型 (Draft Model)：** 一个小得多、速度更快的语言模型。该模型不如目标模型精确，但可以非常快地生成词元序列。

目标模型不再一次生成一个词元，过程如下：

1. **草稿生成：** 快速草稿模型“推测”或提议一个短序列，包含 kkk 个候选词元，紧随当前上下文 (context)。令当前序列为 x1..ix\_{1..i}x1..i​。草稿模型生成 x^i+1,x^i+2,…,x^i+k\hat{x}\_{i+1}, \hat{x}\_{i+2}, \dots, \hat{x}\_{i+k}x^i+1​,x^i+2​,…,x^i+k​。
2. **目标验证：** 大型目标模型随后执行*单次*前向传递，将原始上下文 x1..ix\_{1..i}x1..i​ 和*整个*草稿序列 x^i+1..i+k\hat{x}\_{i+1..i+k}x^i+1..i+k​ 作为输入。此单次传递根据目标模型，有效计算每个词元 j=1..kj=1..kj=1..k 的*真实*概率 pT(x^i+j∣x1..i,x^i+1..i+j−1)p\_T(\hat{x}\_{i+j} | x\_{1..i}, \hat{x}\_{i+1..i+j-1})pT​(x^i+j​∣x1..i​,x^i+1..i+j−1​)。重要的是，KV 缓存等技术在此仍然适用，使此验证步骤高效。
3. **接受检查：** 一种统计接受机制（通常基于拒绝采样）用于比较草稿模型的预测与目标模型验证的概率。对于每个草稿词元 x^i+j\hat{x}\_{i+j}x^i+j​（从 j=1j=1j=1 开始）：
   - 比较目标模型分配的概率 pT(x^i+j∣… )p\_T(\hat{x}\_{i+j} | \dots)pT​(x^i+j​∣…) 与草稿模型分配的概率 pD(x^i+j∣… )p\_D(\hat{x}\_{i+j} | \dots)pD​(x^i+j​∣…)。
   - 如果目标模型认为草稿词元 x^i+j\hat{x}\_{i+j}x^i+j​ 足够可能（与草稿模型认为的概率相比），则接受该词元。一种常见方法是当 pT(x^i+j∣… )/pD(x^i+j∣… )≥ujp\_T(\hat{x}\_{i+j} | \dots) / p\_D(\hat{x}\_{i+j} | \dots) \ge u\_jpT​(x^i+j​∣…)/pD​(x^i+j​∣…)≥uj​ 时接受，其中 uj∼U(0,1)u\_j \sim U(0, 1)uj​∼U(0,1) 是一个随机数。
   - 此检查顺序进行，从 j=1j=1j=1 到 kkk。如果任何词元 x^i+j\hat{x}\_{i+j}x^i+j​ 被拒绝，过程在该点停止接受词元。令 nnn 为接受的词元数量（0≤n<k0 \le n < k0≤n<k）。
4. **修正/继续：**
   - 如果接受了 n<kn < kn<k 个词元（意味着 x^i+n+1\hat{x}\_{i+n+1}x^i+n+1​ 被拒绝），序列 x1..i+nx\_{1..i+n}x1..i+n​ 现在已确定。下一个词元 xi+n+1x\_{i+n+1}xi+n+1​ 从一个修改过的分布中采样，该分布来源于目标模型的概率和草稿模型在该位置的概率，确保整体分布与目标模型匹配。
   - 如果所有 kkk 个词元都被接受（n=kn = kn=k），序列 x1..i+kx\_{1..i+k}x1..i+k​ 已确定。目标模型的前向传递已经计算了下一个词元 xi+k+1x\_{i+k+1}xi+k+1​ 的分布，所以我们直接从 pT(x∣x1..i+k)p\_T(x | x\_{1..i+k})pT​(x∣x1..i+k​) 采样。
5. **重复：** 过程从步骤1重复，使用新扩展的序列。

潜在的加速来自于如果一个周期中接受了 n>0n > 0n>0 个词元，我们就有效生成了 n+1n+1n+1 个词元（nnn 个接受的词元加上最终采样的词元），仅使用昂贵的目标模型的*一次*前向传递和草稿模型的 kkk 次快速前向传递。如果草稿模型足够准确，接受率（nnn 接近 kkk）可以很高，从而大幅减少生成时间。重要的是，统计接受机制确保最终生成的序列遵循目标模型的精确概率分布。

G

Start

当前上下文
x₁ ... xᵢ

Draft

草稿模型生成
k 个候选词元:
 x̂₍i+1) ... x̂₍i+k)

Start->Draft

Verify

目标模型验证
(单次前向传递)
计算 p\_T(x̂₍i+j)|...)

Draft->Verify

AcceptLoop

对于 j = 1 到 k:
检查 x̂₍i+j) 的接受情况

Verify->AcceptLoop

Accept

词元 x̂₍i+j) 已接受

AcceptLoop->Accept

p\_T/p\_D >= U(0,1)

Reject

词元 x̂₍i+j) 已拒绝
(已接受 n = j-1)

AcceptLoop->Reject

p\_T/p\_D < U(0,1)

Accept->AcceptLoop

j < k

AllAccepted

所有 k 个词元都已接受
(n = k)

Accept->AllAccepted

j = k

SampleCorrected

从修正分布中
采样 x₍i+n+1)

Reject->SampleCorrected

Update

更新上下文:
x₁ ... x₍i+n+1 或 i+k+1)

SampleCorrected->Update

SampleNext

从目标分布 p\_T 中
采样 x₍i+k+1)

AllAccepted->SampleNext

SampleNext->Update

Update->Draft

继续生成

> 流程图，说明推测解码过程。草稿模型提出词元，目标模型在单次传递中验证它们，接受循环决定在采样下一个词元之前保留多少个提出的词元。

### 实现考量

- **草稿模型选择：** 选择合适的草稿模型很重要。它需要比目标模型快很多（例如，更少的层/参数 (parameter)，蒸馏模型）。但是，如果其预测与目标模型的差异过大，接受率会很低，抵消性能优势。在速度和预测质量之间找到平衡是必要的。
- **步数 (kkk)：** 选择推测步数 kkk 涉及权衡。较大的 kkk 为每次目标模型推理 (inference)提供了更大的加速潜力。但是，草稿模型在较长序列上的预测可能与目标模型偏离更多，降低所有 kkk 个词元 (token)都被接受的概率。最佳 kkk 通常取决于模型和任务。
- **开销：** 尽管在实际运行时间上更快，推测解码需要将目标模型和草稿模型都保留在内存中，增加了内存占用。运行草稿模型和执行接受检查也有计算开销。

以下是一个类似 PyTorch 的代码片段，说明了核心循环结构：

```python
import torch
import torch.nn.functional as F

def speculative_decode_step(target_model, draft_model, input_ids, k):
    """
    执行推测解码的一个步骤。
    假设模型返回对数（logits）并在内部处理 KV 缓存。
    这是一个简化说明。
    """

    draft_output_ids = draft_model.generate(input_ids, max_new_tokens=k, ...)

    draft_ids = draft_output_ids[:, input_ids.shape[-1]:]

    verify_ids = torch.cat([input_ids, draft_ids], dim=-1)

    with torch.no_grad():
        target_logits = target_model(verify_ids).logits

    target_probs = F.softmax(
        target_logits[:, input_ids.shape[-1]-1:-1, :],
        dim=-1
    )

    accepted_count = 0
    for j in range(k):

        p_target = target_probs[:, j, draft_ids[:, j]].unsqueeze(-1)

        p_draft = draft_probs[:, j, draft_ids[:, j]].unsqueeze(-1)

        ratio = p_target / (p_draft + 1e-8)

        random_uniform = torch.rand_like(ratio)

        if (ratio >= random_uniform).all():
            accepted_count += 1
        else:

            break

    if accepted_count == k:

        next_token_probs = F.softmax(target_logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(next_token_probs, num_samples=1)
        final_ids = torch.cat([input_ids, draft_ids, next_token], dim=-1)
    else:

        final_ids = torch.cat(
            [input_ids, draft_ids[:, :accepted_count]], dim=-1
        )

    return final_ids
```

推测解码代表了一个有前景的方向，用于加速大型语言模型推理，在对延迟敏感的应用中尤其有价值。尽管它与标准自回归 (autoregressive)解码相比引入了额外的复杂度，但显著加速的潜力通常值得付出努力，特别是当与 KV 缓存和优化注意力核等其他优化技术结合时。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 29 Serving Llms At Scale

### LLM 交互的 API 设计

# LLM 交互的 API 设计

要使训练和优化好的大型语言模型（LLM）可供应用使用，需要一个定义清晰的应用程序编程接口（API）。此 API 作为服务基础设施与客户端应用之间的约定，规定了请求的格式以及预期的响应。周全地设计此接口对于易用性、性能和可维护性而言十分重要。不良的 API 设计可能导致资源使用效率低下、客户端集成困难以及用户体验不佳。

### 选择 API 协议和风格

多数 LLM 交互通过网络进行，因此标准的网络协议成为常见选择。

- **REST/HTTP 与 JSON**：这是最普遍的方法，因为它简单、普遍且易于与标准网络技术和库配合使用。请求和响应通常以 JSON 格式通过 HTTP(S) 进行编码。它适用于许多应用，特别是面向用户的应用。
- **gRPC**：对于内部微服务或对性能要求高的应用，gRPC 是更优的选择。它使用 Protocol Buffers 进行序列化，并使用 HTTP/2 进行传输，与 REST/JSON 相比，通常能带来更低的延迟和更高的吞吐量 (throughput)。然而，它需要特定的客户端库，并可能增加复杂性。

对于多数通用 LLM 服务，设计良好的 REST/HTTP API 能在性能和可用性之间取得良好平衡。

### 设计请求负载

客户端需要向模型发送足够的信息来生成文本。典型的请求负载可能包含：

- **提示词（Prompt）**：LLM 应进行响应的输入文本或消息序列（针对聊天模型）。
- **生成参数 (parameter)**：用于影响输出文本的控制项。常见参数包含：
  - `max_new_tokens`：要生成的最大令牌数量。防止生成失控并控制响应长度。
  - `temperature`：控制输出的随机性。较低的值（例如 0.2）使输出更具确定性和集中性，而较高的值（例如 0.8）则增加多样性和创造性。值为 0 实际上使贪婪解码成为标准。
  - `top_p` (核采样)：规定一个概率阈值 ppp。模型只考虑累计概率超过 ppp 的最小令牌集合。这通常比单独使用 `temperature` 效果更好。值通常介于 0.8 到 1.0 之间。
  - `top_k`：将采样限制在最有可能的 kkk 个后续令牌。可与 `top_p` 一起使用或替代 `top_p` 使用。
  - `repetition_penalty`：惩罚已在提示词或生成序列中出现过的令牌，阻止重复的输出。值大于 1.0 会增加惩罚。
  - `stop_sequences`：如果生成，则表示响应结束的字符串列表。
- **流式传输标志**：一个布尔值，表明客户端是希望响应逐令牌流式传输，还是作为一个单个完整块接收。
- **模型标识符**（可选）：如果 API 端点提供多个模型或模型版本服务，则需要一个标识符来指定使用哪个模型。

以下是一个可能的文本补全 API 的 JSON 请求体：

```json
{
  "prompt": "解释 LLM 推理中的键值缓存原理：",
  "model": "llm-engine-v1.2",
  "max_new_tokens": 250,
  "temperature": 0.7,
  "top_p": 0.9,
  "stop_sequences": ["\n\n", "---"],
  "stream": false
}
```

### 设计响应负载

响应应提供生成的文本和相关的元数据。

- **非流式响应**：对于 `stream` 为 `false` 的请求，响应通常是一个包含以下内容的 JSON 对象：
  - `generated_text`：模型输出的完整字符串。
  - `finish_reason`：表明生成停止的原因（例如，如果达到 `max_new_tokens` 则为 `length`，如果生成了停止序列则为 `stop_sequence`，如果模型自然完成则为 `eos_token`）。
  - `usage`：令牌计数（例如，`prompt_tokens`、`completion_tokens`、`total_tokens`）。有助于追踪成本或资源消耗。

非流式响应示例：

```json
{
  "id": "cmpl-xyz123",
  "object": "text_completion",
  "created": 1677652288,
  "model": "llm-engine-v1.2",
  "choices": [
    {
      "text": " 键值（KV）缓存是一种重要的优化技术，在基于 Transformer 的大型语言模型（LLM）的自回归解码过程中使用。在生成过程中，每个新令牌都依赖于涉及所有先前令牌的注意力计算。自注意力层中为先前令牌计算的键（K）和值（V）在生成新令牌时保持不变。KV 缓存将这些 K 和 V 张量存储在内存中（通常是 GPU HBM），而不是在每一步重新计算整个序列的这些张量。在生成下一个令牌时，模型只需计算最新令牌的 K 和 V，并重用所有先前令牌的缓存值。这显著减少了每个生成令牌的计算成本，使得推理速度更快，特别是对于长序列。",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop_sequence"
    }
  ],
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 151,
    "total_tokens": 164
  }
}
```

- **流式响应**：当 `stream` 为 `true` 时，服务器会发送回一个事件序列，通常使用服务器发送事件（SSE）。每个事件通常包含生成文本的一个片段。最后一个事件包含 `finish_reason` 和 `usage` 统计信息。这使得客户端能够逐步显示响应，从而提升用户感受到的性能。

SSE 事件序列示例（简化版）：

```
event: token
data: {"text": " "}

event: token
data: {"text": "-值"}

event: token
data: {"text": " (KV"}

event: token
data: {"text": ") 缓存"}

... 更多令牌事件 ...

event: token
data: {"text": "."}

event: done
data: {"finish_reason": "stop_sequence", "usage": {"prompt_tokens": 13, "completion_tokens": 151, "total_tokens": 164}}
```

G

clusterₛtream

流式响应

Client

客户端应用

APIGateway

API 网关 / 负载均衡器

Client->APIGateway

1. POST /generate (stream=true)

APIGateway->Client

3b. 转发片段 1

APIGateway->Client

4b. 转发片段 2

APIGateway->Client

Nb. 转发最终 + 完成

LLMService

LLM 推理服务

APIGateway->LLMService

2. 转发请求

LLMService->APIGateway

3a. 片段 1

LLMService->APIGateway

4a. 片段 2

LLMService->APIGateway

Na. 最终片段 + 完成

> 流式 API 请求和响应的流程。客户端发起请求，服务按生成顺序返回文本片段。

### 处理同步和异步操作

使用 LLM 生成文本可能需要相当长的时间，从几秒到几分钟，特别是对于长输出。

- **同步 API**：客户端发送请求并等待（阻塞）直到收到完整响应。这很简单，但对于长时间生成容易出现 HTTP 超时，并可能导致客户端应用无响应。仅适用于非常短且可预测的生成。
- **异步 API**：客户端发送请求后，服务器立即使用任务 ID 进行确认。然后客户端可以使用任务 ID 轮询一个端点，或等待回调（webhook）来接收结果。这避免了超时，但增加了客户端逻辑的复杂性。
- **流式 API**：如前所述，流式传输提供了一个中间方案。初始连接保持开放，逐步交付结果。这避免了超时并提升了感受到的响应速度，使其成为交互式应用的首选方法。

### 客户端交互示例 (Python)

以下是客户端如何使用 Python 的 `requests` 库与这些 API 端点进行交互的示例：

**标准（非流式）请求：**

```python
import requests
import json

API_URL = "http://your-llm-service.com/generate"
API_KEY = "your_api_key_here"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
  "prompt": "What is the capital of Malaysia?",
  "max_new_tokens": 50,
  "temperature": 0.5,
  "stream": False
}

try:
    response = requests.post(API_URL,
                             headers=headers,
                             json=payload,
                             timeout=30)
    response.raise_for_status()

    result = response.json()
    generated_text = result['choices'][0]['text']
    print(f"生成的文本: {generated_text}")
    print(f"用量: {result['usage']}")

except requests.exceptions.RequestException as e:
    print(f"API 请求失败: {e}")
except KeyError as e:
    print(f"解析响应失败: 缺少 {e}")
```

**流式请求：**

```python
import requests
import json

API_URL = "http://your-llm-service.com/generate"
API_KEY = "your_api_key_here"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "text/event-stream"
}

payload = {
  "prompt": "Write a short story about a robot learning to paint:",
  "max_new_tokens": 300,
  "temperature": 0.8,
  "stream": True
}

try:

    with requests.post(
        API_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=180
    ) as response:
        response.raise_for_status()
        print("流式响应:")
        full_response = ""
        final_data = None

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('event: token'):

                    data_line = next(response.iter_lines()).decode('utf-8')
                    if data_line.startswith('data:'):
                        try:
                            content = json.loads(data_line[len('data: '):])
                            text_chunk = content.get('text', '')
                            print(text_chunk, end='', flush=True)
                            full_response += text_chunk
                        except json.JSONDecodeError:
                            print(f"\n解码 JSON 错误: {data_line}")
                elif decoded_line.startswith('event: done'):
                     data_line = next(response.iter_lines()).decode('utf-8')
                     if data_line.startswith('data:'):
                         try:
                            final_data = json.loads(
                                data_line[len('data: '):]
                            )
                         except json.JSONDecodeError:
                            print(f"\n解码最终 JSON 错误: {data_line}")
                     break

        print("\n--- 流式传输结束 ---")
        if final_data:
            print(f"结束原因: {final_data.get('finish_reason')}")
            print(f"用量: {final_data.get('usage')}")
        else:
            print("未收到最终数据。")

except requests.exceptions.RequestException as e:
    print(f"\nAPI 请求失败: {e}")
```

### 错误处理和状态码

API 定义了清晰的错误响应。请使用标准的 HTTP 状态码：

- `200 OK`：请求成功（非流式或流式传输的最终事件）。
- `400 Bad Request`：无效输入（例如，JSON 格式错误、缺少必需参数 (parameter)、参数值无效）。在响应体中包含描述性的错误消息。
- `401 Unauthorized`：缺少或无效的身份验证凭据。
- `403 Forbidden`：已认证用户没有权限。
- `429 Too Many Requests`：超出速率限制。如果可能，包含 `Retry-After` 标头。
- `500 Internal Server Error`：生成过程中服务器端发生意外错误。
- `503 Service Unavailable`：服务暂时过载或因维护而停机。

### 安全考量

LLM API 与任何网络服务一样，必须加以保护。实施适当的身份验证（例如，API 密钥、OAuth 2.0）和授权机制来控制访问。使用 HTTPS 加密流量。如果提示词 (prompt)包含不受信任的用户输入，请注意潜在的提示注入攻击。输入验证也是一种安全形式，可以防止格式错误的请求在系统内部引发问题。

设计清晰、文档齐全的 API 是使您的 LLM 可用且实用的基础。在定义接口时，请考虑客户端应用的需求、LLM 生成的特性（可能慢速、长度可变）以及标准网络实践。

获取即时帮助、个性化解释和交互式代码示例。

---

### 模型服务框架 (Triton, TorchServe)

# 模型服务框架 (Triton, TorchServe)

虽然使用Flask或FastAPI等框架的简单Web服务器可能足以部署较小的机器学习 (machine learning)模型，但高效地服务大型语言模型（LLM）则会带来一系列不同的工程问题。LLM参数 (parameter)的庞大体积会给内存资源带来压力，自回归 (autoregressive)生成对延迟很敏感，高效处理并发用户请求需要精密的批处理和资源管理。标准Web服务器并未针对这些GPU密集型、有状态的推理 (inference)工作进行优化。这正是专用模型服务框架不可或缺的原因。

这些框架专门设计用于大规模部署机器学习模型，为LLM提供优化的性能、更好的硬件利用率以及运行的稳定性。它们屏蔽了管理推理请求、模型加载、硬件加速和并发处理的许多底层复杂性。让我们考察两个重要的例子：NVIDIA Triton 推理服务器和PyTorch TorchServe。

### NVIDIA Triton 推理 (inference)服务器

NVIDIA Triton 是一个高性能推理服务平台，旨在将来自各种框架（包括PyTorch、TensorFlow、ONNX Runtime、TensorRT和自定义后端）的模型部署到GPU和CPU上。其架构旨在最大限度地提高吞吐量 (throughput)和硬件利用率，使其成为应对严苛LLM工作负载的有力选择。

与LLM服务相关的重要特点包括：

1. **多框架支持：** Triton 可以同时服务在不同框架中训练的模型，这为您的开发和部署流程带来了灵活性。您可以使用PyTorch模型，也可以同时使用经过TensorRT优化的专用版本或用于预处理的自定义C++后端。
2. **并发模型执行：** Triton 可以在单个GPU或跨多个GPU上同时运行多个模型或同一模型的多个实例。这提高了利用率，特别是对于内存容量大的GPU，它允许不同的用户请求或甚至一个集成模型的不同部分并行处理。
3. **动态批处理：** 这是LLM推理的一个重要功能。Triton在服务器端自动将来自不同客户端的传入请求进行批处理。通过批量处理请求，它更有效地利用GPU的并行处理能力，显著提高吞吐量。Triton根据观察到的请求负载和延迟限制动态调整批处理大小，以优化当前的流量模式。
4. **模型组合与流水线：** Triton 允许您定义流水线或组合模型，一个模型的输出作为另一个模型的输入。这对于需要独立分词 (tokenization)/反分词步骤或复杂预处理/后处理逻辑的LLM非常有用，这些可以作为由Triton管理的组合中的独立模型实现。
5. **多种后端：** 它支持多种执行后端。对于PyTorch模型，您可以使用原生的PyTorch后端 (`libtorch`)。为了获得最佳性能，尤其是在NVIDIA GPU上，模型通常可以转换为TensorRT并通过TensorRT后端提供服务，这可能会提供更低的延迟和更高的吞吐量。
6. **HTTP/gRPC 端点：** 为客户端应用程序发送推理请求提供标准化的网络接口。
7. **性能分析工具：** Triton 包含用于分析模型性能、查看延迟/吞吐量瓶颈以及优化部署配置的工具。

Triton采用声明式配置方法。您通常定义一个模型仓库结构，其中每个模型都含有一个`config.pbtxt`文件，指定其平台、后端、输入/输出张量、版本策略和实例组设置（控制有多少实例在哪些设备上运行）。动态批处理也在此处配置。

这里是一个使用动态批处理的PyTorch LLM的`config.pbtxt`简化示例：

```protobuf
# Triton中PyTorch LLM模型的config.pbtxt
name: "my_llm_model"
platform: "pytorch_libtorch" # 指定后端
max_batch_size: 64           # Triton可形成的最大批处理大小

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ] # 变长序列维度
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1 ] # 变长序列维度
  }
]
output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, 50257 ] # 示例：序列长度，词汇表大小
  }
]

dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ] # Triton偏好形成的批次大小
  max_queue_delay_microseconds: 10000    # 请求等待批处理的最大时间 (10ms)
}

instance_group [
  {
    count: 1 # 此模型的实例数量
    kind: KIND_GPU
    gpus: [ 0 ] # 分配给GPU 0
  }
]

# 可选：指定默认模型文件名 (如果不是 model.pt)
# default_model_filename: "my_llm_scripted.pt"
```

G

Client

客户端应用

Triton

Triton 服务器
(HTTP/gRPC 前端)

Client->Triton

推理请求

Triton->Client

推理响应

Scheduler

调度器
(动态批处理)

Triton->Scheduler

Scheduler->Triton

Backend

模型后端
(PyTorch, TensorRT等)

Scheduler->Backend

批处理请求

Backend->Scheduler

GPU

GPU 资源

Backend->GPU

执行推理

GPU->Backend

结果

> NVIDIA Triton 推理服务器内部的基本交互流程。

Triton的优势在于其广泛的兼容性、动态批处理和TensorRT集成等性能优化功能，以及对并发请求的处理，这使其非常适合大规模、多模型的部署。

### PyTorch TorchServe

TorchServe是一个开源的模型服务框架，专门为PyTorch模型开发。它旨在提供一种简单且高效的方式将PyTorch模型部署到生产环境。由于它是PyTorch原生框架，对于已经大量使用PyTorch生态系统的团队来说，它通常提供了一条更直接的路径。

与LLM服务相关的重要特点包括：

1. **原生PyTorch支持：** 从头开始为PyTorch设计，使用`torch-model-archiver`工具使模型打包和部署过程相对简单。
2. **模型打包：** `torch-model-archiver`实用程序将您的模型代码（例如`model.py`）、序列化权重 (weight)（`.pt`或`.pth`文件）和自定义处理程序文件（`handler.py`）打包成一个`.mar`（模型归档）文件，这是TorchServe的部署单位。
3. **自定义处理程序：** 这是TorchServe的一个重要方面。您在Python中定义一个处理程序类（通常继承自`BaseHandler`），以指定精确的预处理（例如，分词 (tokenization)）、推理 (inference)调用和后处理（例如，反分词，从logits生成文本）逻辑。这使您可以直接在Python中对请求生命周期进行细致的控制。
4. **批量推理：** TorchServe支持在服务器端进行请求批处理，类似于Triton，以提高吞吐量 (throughput)。
5. **管理API：** 提供用于注册、注销和动态扩展模型的API，无需重启服务器。
6. **模型版本控制：** 允许同时加载模型的多个版本，便于A/B测试或逐步发布。
7. **日志和指标：** 提供可配置的日志记录，并发出与Prometheus和CloudWatch等监控工具兼容的运行指标（例如，延迟、请求计数、错误率）。

使用TorchServe部署LLM通常涉及创建自定义处理程序来管理分词和生成过程。

以下是一个生成式LLM的自定义处理程序（`handler.py`）的示例片段：

```python

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import os
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class LLMHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.tokenizer = None
        self.model = None
        self.device = None

    def initialize(self, context):
        """
        加载模型和分词器。模型加载时调用一次。
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        use_cuda = (
            torch.cuda.is_available()
            and properties.get("gpu_id") is not None
        )
        if use_cuda:
            self.device = torch.device(
                "cuda:" + str(properties.get("gpu_id"))
            )
        else:
            self.device = torch.device("cpu")

        logger.info(f"将模型加载到设备：{self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
             self.tokenizer.pad_token = self.tokenizer.eos_token
             self.model.config.pad_token_id = self.model.config.eos_token_id

        logger.info(
            "Transformer模型和分词器加载成功。"
        )
        self.initialized = True

    def preprocess(self, requests):
        """
        对输入提示进行分词。requests 是请求列表。
        """
        input_texts = [req.get("data") or req.get("body") for req in requests]

        prompts = []
        for item in input_texts:
            if isinstance(item, (bytes, bytearray)):
                prompt = item.get("prompt", item.decode('utf-8'))
            else:
                prompt = item.get("prompt")
            prompts.append(prompt)

        logger.info(f"收到提示：{prompts}")

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True
        ).to(self.device)

        return inputs

    def inference(self, inputs):
        """
        执行模型推理（生成）。
        """

        max_new_tokens = inputs.pop("max_new_tokens", 50)
        do_sample = inputs.pop("do_sample", True)
        temperature = inputs.pop("temperature", 0.7)
        top_p = inputs.pop("top_p", 0.9)

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        return outputs

    def postprocess(self, outputs):
        """
        对生成的序列进行反分词。
        """

        generated_texts = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        logger.info(f"生成的文本：{generated_texts}")

        return generated_texts
```

TorchServe为部署PyTorch模型提供了一条简化路径，通过基于Python的自定义处理程序提供灵活性，并与PyTorch生态系统的工具和实践良好集成。

### 框架选择

Triton和TorchServe之间的选择通常取决于具体的项目需求和现有基础设施：

- **选择Triton，如果：**
  - 您需要服务来自多个机器学习 (machine learning)框架（PyTorch, TensorFlow, ONNX, TensorRT）的模型。
  - 实现最大可能的吞吐量 (throughput)和最低延迟，可能通过TensorRT优化，是主要目标。
  - 您需要共享GPU上的并发模型执行或声明式定义的复杂模型组合等功能。
  - 您的团队有C++经验，或者习惯通过文本文件（`config.pbtxt`）配置部署。
- **选择TorchServe，如果：**
  - 您的主要关注点是部署PyTorch模型。
  - 您更喜欢直接在Python自定义处理程序中编写预处理/后处理逻辑的灵活性和便捷性。
  - 与PyTorch开发工作流程的良好集成很重要。
  - 内置模型版本控制和简单快照等功能符合您的运维需求。

这两个框架都能够高效地服务大型模型。它们提供了必要的抽象和优化（如批处理和硬件加速集成），这些是难以从头构建且耗时的工作，使工程团队能够专注于模型开发和应用逻辑，而不是低级别的服务基础设施。可靠且可扩展地部署LLM需要超越基本的Web服务器，转向这些专业的推理 (inference)服务解决方案。

获取即时帮助、个性化解释和交互式代码示例。

---

### 处理并发请求

# 处理并发请求

高效处理并发请求是服务大型语言模型（LLMs）时的一项重要挑战。与典型的无状态网络服务不同，LLM推理 (inference)有其独有的特点：

1. **计算成本高昂：** 在自回归 (autoregressive)序列中生成每个token都需要通过大型模型进行一次完整的正向传播。
2. **内存占用庞大：** 模型参数 (parameter)占用大量GPU内存，并且为避免重复计算先前token的注意力而需要的中间键值（KV）缓存会随每个请求的序列长度增长。
3. **请求长度可变：** 用户提交不同长度的提示，且需要不同长度的生成序列，这会导致每个请求的计算负载不均衡。

简单地按顺序处理请求会导致极低的吞吐量 (throughput)和糟糕的硬件利用率，因为昂贵的GPU大部分时间处于空闲状态。为每个并发请求启动独立的模型实例成本高得令人望而却步，由于庞大的内存需求。因此，批处理技术对于优化吞吐量和成本非常重要。

### 静态批处理

最直接的方法是静态批处理。传入的请求会被收集，直到达到预设的批大小，或发生超时。服务器随后会将批次中所有序列进行填充，填充到最长序列的长度，并在一次正向传播中处理整个批次。

**优点：**

- 相比顺序处理，通过利用并行计算能力提升GPU利用率。
- 相对简单易于实现。

**缺点：**

- **队头阻塞：** 较快的请求可能不必要地等待较慢的请求完成或批次填满。
- **计算资源浪费：** 相当多的计算资源浪费在填充token上，特别是如果批次内序列长度差异很大。整个批次的处理时间由最长序列决定。GPU对填充token执行计算，但这些token最终会被丢弃。
- **延迟增加：** 批次形成和处理的等待时间增加了单个请求的延迟。

考虑一个包含两个序列的批次：A（10个token）和 B（100个token）。使用静态批处理，序列 A 将被填充到 100 个token。GPU 会为两个序列的所有 100 个位置计算输出，即使序列 A 的 90 个位置只是填充。

### 动态批处理

动态批处理提供了一种改进，通过更灵活地形成批次。它不是等待固定的批大小，而是收集在短时间窗口内（例如 10 毫秒）到达的请求，并将它们动态地批处理在一起，直到达到最大批大小限制。

G

cluster₀

传入请求

cluster₁

批处理窗口（例如 5 毫秒）

cluster₂

GPU 处理

R1

请求 1 (t=0)

Batch1

批次 1
(R1, R2, R3)

R1->Batch1

R2

请求 2 (t=1)

R2->Batch1

R3

请求 3 (t=3)

R3->Batch1

R4

请求 4 (t=6)

Batch2

批次 2
(R4, R5)

R4->Batch2

R5

请求 5 (t=7)

R5->Batch2

Proc1

处理批次 1

Batch1->Proc1

Proc2

处理批次 2

Batch2->Proc2

> 在定义的时间窗口内到达的请求会被分组进行批处理。

**优点：**

- 通常比静态批处理具有更高的吞吐量 (throughput)，因为它能更好地适应变化的请求到达率。
- 相比等待大型静态批次填满，它减少了平均等待时间。

**缺点：**

- 在批处理窗口内仍然容易发生队头阻塞。
- 填充问题依然存在；计算浪费仍然相当可观，因为动态形成的批次中最长的序列仍然决定了该批次的整体处理时间。

### 连续批处理（在途批处理）

连续批处理（也称为在途批处理或迭代级批处理）是一种更先进的技术，在现代LLM服务框架中实现，例如 vLLM、Text Generation Inference (TGI) 和 NVIDIA Triton 的 TensorRT-LLM 后端。它从根本上改变了自回归 (autoregressive)解码期间批处理的执行方式。

连续批处理不是将整个请求进行批处理，而是在单个生成*步骤*（迭代）的层面进行操作。其核心思路是：

1. 维护一个当前活跃请求（正在生成中的序列）池。
2. 在每一步，服务器会收集所有活跃序列，那些需要预测“下一个”token的序列。
3. 这些序列会形成一个批次，用于模型的一次正向传播。这个批次的大小可以在每一步中变化。
4. 新到达的请求可以立即添加到池中，如果GPU容量允许，并包含在“下一次”迭代的批次中。
5. 一旦序列生成完成（例如，生成了EOS token或达到最大长度），它会从活跃池中移除，释放资源。

G

SeqA

序列 A

长度: 5

下一个Token

GPU

GPU 正向传播
(批次: A, B, C)

SeqA->GPU

SeqB

序列 B

长度: 8

下一个Token

SeqB->GPU

SeqC

序列 C

长度: 3

下一个Token

SeqC->GPU

SeqA1

序列 A

长度: 6

下一个Token

GPU->SeqA1

SeqB1

序列 B

长度: 9

下一个Token

GPU->SeqB1

SeqC1

序列 C

长度: 4

下一个Token

GPU->SeqC1

GPU2

GPU 正向传播
(批次: A, B, C, D)

SeqA1->GPU2

SeqB1->GPU2

SeqC1->GPU2

NewReqD

新请求 D

长度: 1

已添加

NewReqD->GPU2

> 连续批处理在每次迭代中处理下一个token的生成，针对所有活跃序列，从而允许新请求高效加入。

**优点：**

- **最大化GPU利用率：** 通过不断向GPU提供需要下一个token的活跃序列批次，使其持续进行有效计算。
- **最小化填充浪费：** 填充基本被消除，因为每个序列在每一步中都只处理到其当前长度。
- **高吞吐量 (throughput)：** 相比静态或动态批处理，显著增加单位时间内处理的请求数量。
- **更低的平均延迟：** 新请求通常可以几乎立即开始处理，无需等待批处理窗口结束或长序列完成。

**缺点：**

- **实现复杂性：** 需要复杂的调度逻辑和内存管理。
- **内存管理开销：** 有效管理大量并发、动态变化的序列的KV缓存要求很高。

### 并发内存管理：分页注意力（PagedAttention）

连续批处理的有效性在很大程度上取决于高效的KV缓存管理。由于内存碎片化，将数百个并发序列的整个KV缓存连续存储通常不可行。

vLLM 首创的 **分页注意力（PagedAttention）** 等技术解决了这个问题。受到操作系统中虚拟内存和分页机制的启发，PagedAttention 以非连续的固定大小块（页）来分配KV缓存。这使得能够：

- 通过避免碎片化，在GPU内存中存储更多序列。
- 通过将逻辑块映射到相同的物理块，更高效地在不同请求之间共享上下文 (context)（例如，用于束搜索或从相同提示进行并行采样）。

尽管详细的实现很复杂，但理解这个原理可以说明内存管理与LLM服务中实现高并发息息相关。

### 实现考量

从头开始实现这些高级批处理策略是复杂的。幸运的是，专业的服务框架处理了大部分这种复杂性：

- **NVIDIA Triton Inference Server** 与 **TensorRT-LLM** 后端提供了优化的管道，集成了连续批处理和高效的内存管理。
- **vLLM** 是一个专门为快速LLM推理 (inference)和服务设计的开源库，具有分页注意力（PagedAttention）和连续批处理功能。
- **Hugging Face Text Generation Inference (TGI)** 是另一个流行的开源方案，提供连续批处理和其他优化功能。

配置这些系统时，你通常会遇到以下参数 (parameter)：

- `max_num_batched_tokens`：在单次迭代批处理中可以处理的最大token总数（所有序列的总和）。这有助于控制GPU内存使用。
- `max_num_seqs`：服务器可以处理的最大并发序列数。
- `max_seq_len`：支持的最大序列长度。
- KV缓存配置（例如，分配的GPU内存百分比）。

这是一个高度简化的Python代码片段，说明了管理请求的核心循环逻辑，未详细说明KV缓存分页或框架具体细节：

```python

import torch
import queue
import threading
import time
import uuid

request_queue = queue.Queue()

active_requests = {}

result_store = {}

MAX_CONCURRENT_REQUESTS = 16
SCHEDULING_INTERVAL = 0.005

def get_next_batch():
    """ 收集准备好进行下一步推理的请求。 """
    batch_req_ids = []
    batch_input_tokens = []
    batch_kv_caches = []

    for req_id, state in list(active_requests.items()):
        if not state['is_finished']:
            batch_req_ids.append(req_id)

            if not state['output_tokens']:
                input_token = state['prompt_tokens'][:, -1:]
            else:
                input_token = state['output_tokens'][:, -1:]
            batch_input_tokens.append(input_token)
            batch_kv_caches.append(state['kv_cache'])

    available_slots = MAX_CONCURRENT_REQUESTS - len(batch_req_ids)
    for _ in range(available_slots):
        try:
            new_req_id, prompt_str = request_queue.get_nowait()
            prompt_tokens = tokenizer.encode(
                prompt_str, return_tensors="pt"
            ).to(model.device)
            active_requests[new_req_id] = {
                'prompt_tokens': prompt_tokens,
                'output_tokens': None,
                'kv_cache': None,
                'is_finished': False
            }

            batch_req_ids.append(new_req_id)
            batch_input_tokens.append(prompt_tokens[:, -1:])
            batch_kv_caches.append(None)
            print(f"调度器：已添加新请求 {new_req_id}")
        except queue.Empty:
            break

    return (
        batch_req_ids,
        batch_input_tokens,
        batch_kv_caches
    )

def inference_scheduler_loop():
    """ 调度和运行推理批次的主循环。 """
    while True:
        start_time = time.time()

        (
            batch_req_ids,
            batch_input_tokens,
            batch_kv_caches
        ) = get_next_batch()

        if not batch_req_ids:
            time.sleep(SCHEDULING_INTERVAL)
            continue

        logits = torch.randn(
            len(batch_req_ids), 1, tokenizer.vocab_size
        ).to(model.device)
        next_token_ids = torch.argmax(logits, dim=-1)
        updated_kv_caches = [
            f"kv_{req_id}_step_{len(active_requests[req_id].get('output_tokens',[]))}"
            for req_id in batch_req_ids
        ]

        print(f"调度器：已处理批次大小为 {len(batch_req_ids)} 的批次")

        finished_ids = []
        for i, req_id in enumerate(batch_req_ids):
            state = active_requests[req_id]
            current_next_token = next_token_ids[i:i+1]

            if state['output_tokens'] is None:
                state['output_tokens'] = current_next_token
            else:
                state['output_tokens'] = torch.cat(
                    [state['output_tokens'], current_next_token],
                    dim=1
                )

            state['kv_cache'] = updated_kv_caches[i]

            if (current_next_token.item() == 2 or
                    (state['output_tokens'] is not None and
                     state['output_tokens'].shape[1] > 100)):
                 state['is_finished'] = True
                 full_sequence = torch.cat(
                     [state['prompt_tokens'], state['output_tokens']],
                     dim=1
                 )
                 result_store[req_id] = tokenizer.decode(
                     full_sequence[0], skip_special_tokens=True
                 )
                 finished_ids.append(req_id)
                 print(f"调度器：请求 {req_id} 已完成")

        for req_id in finished_ids:
            if req_id in active_requests:
                 del active_requests[req_id]

        elapsed_time = time.time() - start_time
        sleep_time = max(0, SCHEDULING_INTERVAL - elapsed_time)
        time.sleep(sleep_time)
```

综上所述，处理LLM服务的并发请求需要超越简单的静态或动态批处理。连续批处理，结合像分页注意力（PagedAttention）这样精密的内存管理，通过迭代处理所有活跃请求的生成步骤，提供了最高的吞吐量 (throughput)和最佳的资源利用率。使用专业的服务框架通常是在生产环境中实现这些高级技术最实际的方法。

获取即时帮助、个性化解释和交互式代码示例。

---

### 跨模型实例的负载均衡

# 跨模型实例的负载均衡

成功部署大型语言模型不只是需要一个训练好的成品；它需要能够高效可靠地处理大量并发请求的基础设施。如前所述，在大量负载下，即使是强大的硬件，单个模型实例也很快会成为瓶颈。此外，仅依靠单个实例会造成单点故障。为解决这些限制并获得可伸缩性和高可用性，使用负载均衡将推理 (inference)请求分配到多个模型实例上是必要的。

负载均衡充当你的LLM服务的流量管理员。请求不会直接到达单个模型实例，而是首先到达负载均衡器。负载均衡器的任务是根据选定的策略，智能地将每个请求转发给多个可用后端模型实例（副本）中的一个。主要目标是最大化总体吞吐量 (throughput)（每秒处理的请求数），最小化用户感受到的平均响应延迟，确保即使某些实例出现故障服务也能保持可用，并优化GPU等昂贵硬件资源的使用效率。

### LLM为何需要负载均衡

LLM推理 (inference)的独特特点使得负载均衡显得尤为重要：

1. **每次请求资源消耗高：** 使用大型模型生成文本需要大量计算（GPU周期）和内存（用于模型权重 (weight)、激活和KV缓存）。在一个GPU上按顺序处理请求会极大地限制吞吐量 (throughput)。为了同时服务多个用户，并行执行跨多个实例是必需的。
2. **请求负载多变：** LLM应用的客户端流量通常会根据一天中的时间、用户活动或特定事件而波动。负载均衡层使得后端基础设施（模型实例数量）能够根据需求进行水平伸缩（增加或移除实例），从而在不过度预留资源的情况下确保性能一致。
3. **容错与高可用：** 硬件会失效，软件会崩溃，网络问题也会发生。如果你只有一个模型实例，任何此类故障都会导致服务完全中断。通过将请求分配到多个实例，负载均衡器可以检测无响应的实例（通过健康检查），并仅将流量路由到健康的实例，从而显著提高服务的总体可用性。

### 常见负载均衡策略

可以使用多种算法来决定哪个后端实例应接收下一个传入请求。选择取决于你应用程序的具体要求、推理 (inference)请求的性质以及在简单性和最佳负载分配之间所需做的权衡。

- **轮询（Round Robin）：** 这是最简单的策略。负载均衡器维护一个可用后端实例列表，并按顺序循环它们，将每个新请求发送给列表中的下一个实例。

  G

  LB

  负载均衡器
  (轮询)

  S1

  实例 1

  LB->S1

  1, 4

  S2

  实例 2

  LB->S2

  2

  S3

  实例 3

  LB->S3

  3

  R1

  请求 1

  R1->LB

  R2

  请求 2

  R2->LB

  R3

  请求 3

  R3->LB

  R4

  请求 4

  R4->LB

  > 轮询策略按顺序将请求分配给可用实例。

  它易于实现，但假定所有请求的复杂度大致相同，并且所有实例都具有相同的性能。它不适应实例负载或请求处理时间的变化。
- **最少连接（Least Connections）：** 该策略将新请求定向到当前处理活动连接最少的后端实例。这样做的原因是为了让连接较少的实例可能更不繁忙。

  G

  LB

  负载均衡器
  (最少连接)

  S1

  实例 1
  (2个活跃)

  LB->S1

  S2

  实例 2
  (0个活跃)

  LB->S2

  发送到此处

  S3

  实例 3
  (1个活跃)

  LB->S3

  NewReq

  新请求

  NewReq->LB

  > 最少连接策略将新请求定向到当前连接最少的实例（实例 2）。

  当请求持续时间不同时，这通常比轮询提供更好的负载分配，因为它隐式地将流量从被长时间运行请求拖慢的实例中分流。然而，它依赖连接数作为负载的代理，这可能无法完美反映GPU的使用情况。
- **最少负载/基于资源：** 更精密的策略是根据后端实例实际测量的资源使用情况来路由请求。这可能涉及监控GPU使用率、可用GPU内存、CPU负载，甚至像模型服务器内部请求队列长度这样的自定义应用级指标。负载均衡器将新请求定向到当前报告负载最低的实例。这在理论上是平衡*实际工作量*最有效的方式，但它需要一个监控系统来收集这些指标，以及一个能够使用这些指标的负载均衡器，这会增加复杂性。
- **加权策略：** 如果你的后端实例具有不同的能力（例如，某些实例拥有更强大的GPU），你可以为它们分配权重 (weight)。加权轮询或加权最少连接算法将根据这些权重按比例分配请求，向能力更强的实例发送更多流量。
- **哈希/会话粘滞：** 像IP哈希这样的策略会将来自相同客户端IP地址的请求始终定向到相同的后端实例。这被称为会话粘滞或粘性会话。虽然对于用户会话数据必须驻留在特定服务器上的有状态应用程序很重要，但对于典型的无状态LLM推理API来说，它通常*适用性较小*，有时甚至不理想，因为任何实例都可以处理任何请求。如果某些客户端发送的请求比其他客户端多得多，保持粘滞性可能导致负载分配不均。

### 实现考量

为LLM服务实现负载均衡涉及多个组件：

- **负载均衡器选择：** 你可以使用硬件负载均衡器，但在云和容器化环境中，软件解决方案更常见。常用选择包括：

  - 配置为负载均衡的反向代理，如Nginx或HAProxy。
  - 云提供商服务，如AWS弹性负载均衡（ELB）、谷歌云负载均衡或Azure负载均衡器，它们提供托管式、可伸缩的解决方案。
  - Kubernetes服务：当在Kubernetes中将模型实例作为Pod部署时，内置服务（如`LoadBalancer`或`NodePort`）和Ingress控制器会自动处理健康Pod之间的路由和负载均衡。
- **健康检查：** 负载均衡器的一项重要功能是持续监控后端实例的健康状况。它会定期向每个实例的特定端点（例如，`/healthz`）发送一个小请求（健康检查）。如果一个实例未能正确响应或在超时时间内未响应，负载均衡器会将其标记 (token)为不健康，并停止向其发送流量，直到它恢复。为LLM服务器设计有效的健康检查可能不仅涉及检查服务器进程是否正在运行，还可能涉及执行一个最小推理 (inference)任务，以确保模型已加载并可响应。
- **与服务框架集成：** 模型服务框架（如NVIDIA Triton推理服务器或TorchServe）运行在每个后端实例上。负载均衡器位于这些实例的*前面*。框架本身可能处理动态批处理（将负载均衡器接收到的、时间相近的请求分组以进行高效GPU处理）等方面，但实例间的负载均衡通常由外部管理。
- **自动伸缩集成：** 负载均衡与自动伸缩结合使用时最有效。自动伸缩器监控实例池的汇总指标（例如，平均GPU使用率、请求队列长度、延迟）。根据预定义的阈值，它会在高负载时自动添加更多模型实例，并在需求减少时移除它们。负载均衡器会动态调整以适应自动伸缩器提供的可用实例集的变化，确保流量始终在活跃的实例池中分配。

### 挑战与权衡

负载均衡虽然必不可少，但也带来一些考量：

- **延迟开销：** 负载均衡器本身为每个请求增加了一个小的网络跳跃和处理延迟。这通常与LLM推理 (inference)时间相比可以忽略不计，但仍是一个因素。
- **配置复杂性：** 选择和配置正确的策略、健康检查和超时，需要根据观察到的流量模式和性能目标进行仔细调整。
- **冷启动：** 当自动伸缩添加新实例时，会存在一个延迟，因为实例需要启动、容器需要开始，并且大型模型需要加载到GPU内存中（即“冷启动”）。负载均衡器根据健康检查信息，必须等到实例完全就绪后才能向其发送流量。

“总之，负载均衡是构建可伸缩且弹性LLM服务系统的基本技术。通过将请求分配到多个模型副本，它使得更高的吞吐量 (throughput)、更低的平均延迟和容错成为可能，从而确保你强大的语言模型能够有效处理应用需求。选择适当的策略并将其与健康检查和自动伸缩正确结合，是部署生产级LLM的重要组成部分。”

获取即时帮助、个性化解释和交互式代码示例。

---

### 监控服务性能和成本

# 监控服务性能和成本

部署大型语言模型是一项重要的工程成就，但一旦 API 端点上线，工作并未结束。持续监控服务基础设施对于确保可靠性、性能和成本效益非常重要。若无细致的监控，您将面临性能下降、意外中断和成本上涨的风险，所有这些都可能对用户体验和运营预算造成负面影响。有效监控 LLM 服务系统所需的方法和指标将得到阐述。

### LLM 服务的性能指标 (KPIs)

监控 LLM 需要跟踪标准 Web 服务指标以及生成模型特有的指标。以下是需要关注的主要方面：

#### 延迟

延迟衡量处理请求所需的时间。对于 LLM，尤其在自回归 (autoregressive)生成期间，延迟可以是多方面的：

1. **时间到首个令牌 (TTFT)：** 从接收请求到生成首个输出令牌所耗费的时间。这对于用户期望即时反馈的交互式应用非常重要。
2. **每个输出令牌的时间 (TPOT)：** 在生成首个令牌后，每个后续令牌的平均生成时间。这表示模型的生成速度。
3. **端到端延迟：** 从接收请求到发送完整响应的总时间。这反映了用户感知的总延迟。

较低的 TTFT 和 TPOT 通常是相互冲突的目标。批处理等优化措施可能会略微增加 TTFT，但能提高整体吞吐量 (throughput)和 TPOT。衡量这些指标需要对服务代码进行性能监控部署。

```python

import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def handle_request(prompt):
    request_start_time = time.monotonic()

    first_token_time = None
    output_tokens = 0

    for token in model.generate(prompt):
        if first_token_time is None:
            first_token_time = time.monotonic()
        output_tokens += 1

        pass
        time.sleep(0.05)

    request_end_time = time.monotonic()

    if first_token_time:
        ttft = (first_token_time - request_start_time) * 1000
        e2e_latency = (request_end_time - request_start_time) * 1000

        if output_tokens > 1:
             tpot = (request_end_time - first_token_time) / \
                    (output_tokens - 1) * 1000
        else:
             tpot = 0

        logging.info(
            f"请求已处理：TTFT={ttft:.2f}ms, "
            f"TPOT={tpot:.2f}ms/令牌, "
            f"端到端延迟={e2e_latency:.2f}ms, "
            f"令牌数={output_tokens}"
        )
    else:

        e2e_latency = (request_end_time - request_start_time) * 1000
        logging.warning(
            f"请求已处理但未生成令牌："
            f"端到端延迟={e2e_latency:.2f}ms"
        )

    return "Generated response"
```

#### 吞吐量

吞吐量衡量服务系统的容量，通常表示为：

1. **每秒请求数 (RPS)：** 系统每秒可处理的独立请求数量。
2. **每秒令牌数 (TPS)：** 每秒所有并发请求生成的输出令牌总数。这通常是对于 LLM 更具意义的指标，因为请求复杂度（输出长度）差异很大。

吞吐量受批处理大小、模型架构、硬件加速（GPU 类型）以及 KV 缓存或 FlashAttention 等推理 (inference)优化措施的影响。监控吞吐量有助于容量规划和识别瓶颈。

#### 资源利用率

LLM 是资源密集型应用。监控硬件利用率对于效率和稳定性非常重要：

1. **GPU 利用率：** GPU 计算单元处于活动状态的时间百分比。高利用率通常是好的，表示高效使用了昂贵的硬件，但持续达到 100% 可能预示着瓶颈。
2. **GPU 显存 (VRAM)利用率：** 使用的 GPU 高带宽内存 (HBM) 量。这通常是批处理大小或容纳大型模型的限制因素。GPU 显存耗尽会导致内存不足 (OOM) 错误，从而引发请求失败。
3. **CPU 利用率：** 虽然推理受 GPU 限制，但 CPU 处理预处理/后处理、网络 I/O 和编排。高 CPU 使用率仍然可能成为系统的瓶颈。
4. **系统内存 (RAM) 利用率：** 主机上的 RAM 使用量。
5. **网络带宽：** 数据传输速率，对于加载模型、接收请求和发送响应非常重要。

诸如 `nvidia-smi` (针对 NVIDIA GPU) 或平台特定的监控代理 (例如用于 GPU 的 Prometheus Node Exporter 和 DCGM Exporter) 等工具常用于收集这些指标。

#### 错误率

跟踪失败请求的频率（例如，HTTP 5xx 服务器错误、超时错误、OOM 错误）是可靠性的基本要求。错误率上升通常表示模型服务器、基础设施或特定类型的有问题输入提示存在潜在问题。

### 监控成本

服务大型模型，尤其是在 GPU 或 TPU 等高端加速器上，可能成本高昂。有效的成本监控包括：

1. **基础设施成本：** 跟踪计算实例（虚拟机、GPU）、存储（用于模型权重 (weight)）和网络出站流量的每小时/每日/每月成本。云服务提供商提供详细的账单仪表板和 API（例如 AWS Cost Explorer、Google Cloud Billing Reports）。对资源进行适当标记 (token)有助于将成本归因于特定模型或环境。
2. **单位成本：** 计算每生成一千个令牌的成本或每个请求的成本等指标。这需要将基础设施成本与使用指标（吞吐量 (throughput)、令牌计数）关联起来。
   每千令牌成本=总基础设施成本总生成令牌数×1000\text{每千令牌成本} = \frac{\text{总基础设施成本}}{\text{总生成令牌数}} \times 1000每千令牌成本=总生成令牌数总基础设施成本​×1000
   监控此效率指标有助于了解部署的经济可行性以及优化工作的效果。突然增加可能表示扩展效率低下或资源配置问题。

### 系统健康和依赖

监控服务系统的整体健康状况是必要的：

- **健康检查：** 为 Kubernetes 等容器编排器实现活跃度探测（服务是否正在运行？）和就绪度探测（服务是否已准备好接收流量？）。
- **资源限制：** 监控磁盘空间、文件描述符和其他系统级资源以防止耗尽。
- **依赖健康：** 如果 LLM 服务依赖于外部数据库、缓存层或其他微服务，它们的健康和性能也必须作为端到端系统的一部分进行监控。

### 监控工具和技术

监控堆栈通常结合多种工具：

1. **日志记录：** 在您的服务应用中实现结构化日志记录。不要使用纯文本消息，而是将事件记录为包含相关元数据（请求 ID、用户 ID、延迟指标、输入/输出令牌计数、错误）的 JSON 对象。这使得日志易于解析和搜索。

   ```python

   import logging
   import json
   import uuid

   class JsonFormatter(logging.Formatter):
       def format(self, record):
           log_record = {
               "时间戳": self.formatTime(record, self.datefmt),
               "级别": record.levelname,
               "消息": record.getMessage(),
               "请求ID": getattr(record, "request_id", "N/A"),

               **(record.args if isinstance(record.args, dict) else {})
           }
           return json.dumps(log_record)

   logger = logging.getLogger('LLMServer')
   logger.setLevel(logging.INFO)
   handler = logging.StreamHandler()
   handler.setFormatter(JsonFormatter())
   logger.addHandler(handler)

   request_id = str(uuid.uuid4())
   logger.info("正在处理请求", extra={"请求ID": request_id, "提示长度": 15})

   try:

       logger.info("请求成功", extra={
           "请求ID": request_id,
           "TTFT_毫秒": 55.2,
           "TPOT_毫秒": 10.1,
           "令牌数": 120
       })
   except Exception as e:
       logger.error(
           "请求失败",
           extra={"请求ID": request_id, "错误": str(e)},
           exc_info=True
       )
   ```
2. **指标收集：** 使用 Prometheus 等时序数据库存储从导出器抓取的指标。常见的导出器包括：

   - `node-exporter`：系统级指标（CPU、RAM、磁盘、网络）。
   - `dcgm-exporter`：详细的 NVIDIA GPU 指标（利用率、内存、温度、功耗）。
   - 自定义应用指标：使用客户端库（例如 Python 的 `prometheus_client`）直接从您的服务代码中公开应用级指标（延迟分布、吞吐量 (throughput)、令牌计数、缓存命中率）。
3. **分布式追踪：** 对于涉及多个微服务（例如 API 网关、预处理服务、模型推理 (inference)服务器）的复杂服务堆栈，Jaeger 或 Zipkin 等分布式追踪工具，通常通过 OpenTelemetry 集成，有助于可视化请求在服务间的整个生命周期。这对于确定分布式系统中的延迟瓶颈非常宝贵。

   ```python

   ```

# 示例：基本的 OpenTelemetry 追踪 Span

```
# （需要安装 opentelemetry-api、opentelemetry-sdk 和 exporters）
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# 配置 OpenTelemetry（通常在应用启动时完成一次）
provider = TracerProvider()
processor = BatchSpanProcessor(ConsoleSpanExporter()) # 或导出到 Jaeger 等
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# 在处理请求一部分的函数中
def process_sub_task(data):
    with tracer.start_as_current_span("处理子任务") as span:
        span.set_attribute("数据大小", len(data))
        # ... 执行处理 ...
        result = data + "_processed"
        span.set_attribute("结果大小", len(result))
        return result

# 在主请求处理程序中
def handle_llm_request(prompt):
    with tracer.start_as_current_span("处理LLM请求") as span:
        span.set_attribute("提示长度", len(prompt))
        # 调用可能也会创建 span 的其他函数
        processed_data = process_sub_task(prompt)
        # ... 调用模型 ...
        span.set_attribute("响应长度", 100) # 示例值
        return "response"

# handle_llm_request("一些输入")
```
```

4. **可视化和仪表板：** 使用 Grafana、Kibana（用于日志）或云服务提供商仪表板等工具创建重要指标的可视化。仪表板提供系统健康状况和性能趋势的概览。

```
```plotly
{"layout": {"title": "P95 端到端延迟（过去一小时）", "xaxis": {"title": "时间"}, "yaxis": {"title": "延迟 (ms)", "range": [100, 500]}, "margin": {"l": 40, "r": 20, "t": 40, "b": 30}}, "data": [{"x": ["10:00", "10:15", "10:30", "10:45", "11:00"], "y": [210, 235, 220, 250, 240], "type": "scatter", "mode": "lines+markers", "name": "P95 延迟", "marker": {"color": "#228be6"}}]}
```

> P95 延迟跟踪 95% 请求完成时间所处的阈值，突出显示用户遇到的最差性能。
```

5. **告警：** 使用 Prometheus Alertmanager 或云服务提供商服务（例如 AWS CloudWatch Alarms）等工具，根据指标阈值配置告警规则。重要的告警可能包括：
\* 高 P99 延迟（> N 毫秒）。
\* 低吞吐量（< M 令牌/秒）。
\* 高错误率（> X%）。
\* 高 GPU 显存 (VRAM)利用率（> 95%）。
\* 根据预测即将发生的成本超支。

### LLM 特有的监控细节

考虑以下标准指标：

- **输出质量监控：** 这很难完全自动化。然而，您可以跟踪代理指标，例如平均输出长度、每个请求的令牌使用量或特定关键词出现次数。实施用户反馈机制（例如点赞/点踩）并监控这些评分可以提供有关感知质量随时间变化的宝贵信号。
- **KV 缓存性能：** 如果使用键值缓存（第 28 章），监控其命中率和内存使用情况。低命中率可能表明缓存配置不理想或请求模式从缓存中受益不大。

有效的监控并非一次性设置。它需要持续关注。定期审查仪表板，根据观察到的性能调整告警阈值，并随着模型、流量模式和基础设施的变化完善您的监控策略。全面的监控提供了在高规模下可靠、高效且经济地运行 LLM 服务所需的可见性。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 30 Continuous Training Model Updates

### 持续改进的缘由

# 持续改进的缘由

部署大型语言模型到生产环境是一个重要成就，是大量数据处理、架构设计和资源密集型训练的成果。然而，大型语言模型的运行生命周期很少止于首次部署。经过训练的模型其静态特性与它所交互的动态环境形成鲜明对比。信息演变、语言使用变化、用户预期转变。若无持续维护和改进，即使再强大的模型，其性能和关联性也将不可避免地下降。本节概述了使得持续训练和模型更新成为必要的主要驱动因素。

### 知识陈旧与演进

大型语言模型是在大量数据集上训练的，这些数据集代表了截至某个时间点的信息快照，通常被称为“知识截止”日期。然而，现实并非一成不变。新事件发生、科学发现涌现 (emergence)、公众人物出现、文化趋势转变。一个基于2022年数据训练的模型将缺乏对2024年重要事件或进展的认知。

设想一个用户询问近期选举的获胜者或新发布技术的细节。一个知识过时的模型可能提供不正确信息，声明它不知道，甚至“胡编乱造”听似合理但虚假的信息。这种事实准确性下降会损害用户信任并限制模型效用，特别是对于需要最新信息的应用。持续训练提供了一种将更新知识注入模型的机制，使其响应保持相关性和准确性。

### 数据分布漂移

模型在推理 (inference)时遇到的数据分布可能逐渐或有时突然偏离其原始训练数据的分布。这种现象，被称为数据分布漂移或偏移，可通过多种方式表现：

1. **话题漂移：** 用户可能开始询问模型在原始训练语料库中代表性不足或缺失的新话题（例如，一个新的全球事件、一个流行迷因、一个特定技术范畴）。
2. **语言风格漂移：** 用户措辞提示的方式、他们使用的俚语或语言的正式程度可能随时间变化或在不同用户群体间有所差异。
3. **任务漂移：** 用户可能开始将模型用于在其初始微调 (fine-tuning)阶段未明确优化的任务。

当推理分布与训练分布显著偏离时，模型的性能通常会下降。它可能变得流畅性降低、准确性下降或对现在常遇到的输入类型帮助性变差。监测分布漂移是大型语言模型机器学习 (machine learning)操作中的一个重要方面。简单技术包括追踪话题频率或比较训练批次和推理日志之间的嵌入 (embedding)分布。

```python
import torch
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go

def get_embeddings(text_samples):

    return torch.randn(len(text_samples), 768)

train_texts = ["示例训练句 1", "另一个训练示例"]
inference_texts = ["近期用户查询示例", "不同类型的用户输入"]

train_embeddings = get_embeddings(train_texts)
inference_embeddings = get_embeddings(inference_texts)

pca = PCA(n_components=2)
all_embeddings = torch.cat((train_embeddings, inference_embeddings), 0).numpy()
pca.fit(all_embeddings)

train_embeddings_2d = pca.transform(train_embeddings.numpy())
inference_embeddings_2d = pca.transform(inference_embeddings.numpy())

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=train_embeddings_2d[:, 0], y=train_embeddings_2d[:, 1],
    mode='markers', name='训练数据样本',
    marker=dict(color='#1f77b4', size=8, opacity=0.7)
))

fig.add_trace(go.Scatter(
    x=inference_embeddings_2d[:, 0], y=inference_embeddings_2d[:, 1],
    mode='markers', name='推理数据样本',
    marker=dict(color='#ff7f0e', size=8, opacity=0.7)
))

fig.update_layout(
title_text="嵌入分布漂移可视化",
    xaxis_title="PCA 分量 1",
    yaxis_title="PCA 分量 2",
    legend_title_text='数据来源',
    margin=dict(l=20, r=20, t=40, b=20),
    width=600, height=400
)
```

−0.500.511.52−1.2−1−0.8−0.6−0.4−0.200.2数据来源训练数据样本推理数据样本嵌入分布漂移可视化PCA 分量 1PCA 分量 2

> 一张图表，显示了推理过程中（橙色）的数据分布如何在降维嵌入空间中偏离原始训练数据分布（蓝色）。显著的漂移表示模型可能需要再训练或更新。

持续训练模型，无论是通过在更新、更具代表性的数据上进行进一步预训练 (pre-training)，还是通过有针对性的微调，都有助于模型应对这些不断变化的分布。

### 持续演进的对齐 (alignment)与用户预期

对齐，常通过监督式微调 (fine-tuning)（SFT）和基于人类反馈的强化学习 (reinforcement learning)（RLHF）等技术达成，旨在使模型更有帮助、更诚实、更无害。然而，对齐并非固定目标。

- **细化安全需求：** 随着模型部署更广，可能发现新的潜在危害或误用情况，需要更新安全协议和对齐训练数据。
- **偏好变化：** 用户对风格、语气或冗长程度的偏好可能变化。从生产使用中收集的反馈可以指导进一步的对齐调整。
- **新能力：** 用户可能希望获得新能力，例如改进的推理 (inference)能力、更好地遵循复杂指令或对新专业范畴的熟练度，使得更新成为必要。

持续的 SFT 和 RLHF 周期，以持续数据收集和人类反馈为依据，是维持并提升与不断演进的标准和用户需求对齐的必要条件。

### 性能退化与错误修复

尽管部署前经过严格评估，大型语言模型可能出现意想不到的故障模式或性能退化，针对特定输入类型或任务。这些问题可能只在大规模生产使用和监控中显现出来。持续训练为解决这些退化提供了途径，或许通过对代表故障情况的数据进行微调 (fine-tuning)，甚至在根源更深层时，进行架构修复。

### 竞争概况

大型语言模型行业发展迅速。新架构、训练技术和对齐 (alignment)方法不断涌现 (emergence)。不同组织开发的模型持续改进，为性能和能力设定更高标准。为保持竞争力并满足最先进模型设定的用户预期，组织必须投入持续改进周期以提升其自身的大型语言模型。

总而言之，持续训练和模型更新并非可选项，而是大型语言模型生命周期的必要组成部分。它们由应对知识陈旧、应对数据分布漂移、完善与不断演进的用户预期和安全标准的对齐、修复性能退化以及在快速发展的行业保持竞争力的需求所驱动。接下来的章节将阐述实施这些持续改进循环的具体策略和工程实践。

获取即时帮助、个性化解释和交互式代码示例。

---

### 持续预训练策略

# 持续预训练策略

大型语言模型训练部署后，其工作并未画上句号。信息环境不断变化，新数据持续生成，模型在实际运行中遇到的数据分布可能与其初始训练集产生偏差。简单地冻结模型会导致其随着时间推移变得陈旧，性能下降。将旧数据与新数据结合进行从头完全重新训练，通常在计算和时间上成本过高。持续预训练 (pre-training)提供了一个折衷方案：用新数据更新现有模型，同时尽量保持之前学到的知识。

持续预训练中的主要障碍是*灾难性遗忘*。当模型在不同数据分布上顺序训练时，随着其适应新数据分布，模型会迅速丢失对早期数据分布的性能，此现象即为灾难性遗忘。模型的参数 (parameter)会显著调整以最小化新数据上的损失，从而有效地覆盖了对旧数据性能重要的表示。有效地权衡学习新信息（可塑性）与保持旧知识（稳定性）是持续预训练策略要解决的核心难题。

### 朴素持续学习

最直接的方法是直接使用新数据集继续预训练 (pre-training)过程，从先前训练模型的权重 (weight)开始。这有时被称为在新数据上进行微调 (fine-tuning)。

```python

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)

model_path = "path/to/your/pretrained/llm"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

new_dataloader = DataLoader(new_dataset, batch_size=4, shuffle=True)

optimizer = AdamW(model.parameters(), lr=1e-5)
num_training_steps = len(new_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(num_epochs):
    for batch in new_dataloader:
        optimizer.zero_grad()

        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

model.save_pretrained("path/to/updated/llm")
```

尽管简单，但此方法极易受到灾难性遗忘的影响，特别是当新数据分布与旧数据分布差异很大，或者新数据量很大时。模型的参数 (parameter)会显著偏向最小化新数据上的损失，可能抹去早期学到的能力。这种方法可能只适用于新数据与原始训练数据非常相似或仅是小幅更新的情况。

### 重放（Replay）方法

重放策略通过在持续学习阶段将来自先前训练阶段的数据与新数据混合，明确地对抗遗忘。通过让模型重新接触旧样本，这些方法促使其保持在原始数据分布上的性能。

核心思想是创建包含新数据样本（DnewD\_{new}Dnew​）和旧数据集样本（DoldD\_{old}Dold​）的训练批次。

```python

from torch.utils.data import (DataLoader, Dataset, ConcatDataset,
                              WeightedRandomSampler)

new_data_weight = 0.75
old_data_weight = 0.25

weights = [new_data_weight / len(new_dataset)] * len(new_dataset) + \
          [old_data_weight / len(old_dataset)] * len(old_dataset)

combined_dataset_for_sampler = ConcatDataset([new_dataset, old_dataset])
sampler = WeightedRandomSampler(
    weights,
    num_samples=len(combined_dataset_for_sampler),
    replacement=True
)

combined_dataloader = DataLoader(
    combined_dataset_for_sampler,
    batch_size=8,
    sampler=sampler
)
```

重放方法的考量包括：

- **数据存储**：存储全部或部分 DoldD\_{old}Dold​ 需要大量存储空间，特别是对于LLM预训练 (pre-training)中使用的海量数据集。
- **子集选择**：如果存储所有旧数据不可行，选择一个有代表性的子集就变得很重要。可以使用水库采样或选择多样化样本等技术。
- **混合比例**：每个批次中旧数据与新数据的比例是一个重要的超参数 (parameter) (hyperparameter)。旧数据过少可能无法阻止遗忘，而过多则可能减缓模型对新数据的适应速度。
- **隐私和许可**：确保存储和重用旧数据符合隐私法规和数据许可。

### 正则化 (regularization)方法

正则化方法不依赖显式数据重放，而是修改损失函数 (loss function)，惩罚对先前任务或数据分布而言被认为重要的模型参数 (parameter)的改变。

#### 弹性权重 (weight)整合 (EWC)

EWC 使用 Fisher 信息矩阵（FIM），即 FFF，来评估每个参数对于旧数据分布的重要性。FIM 中对角线值高的参数被认为更重要。在新数据（DnewD\_{new}Dnew​）训练期间，EWC 会在标准损失（LnewL\_{new}Lnew​）中添加一个二次惩罚项，该项会抑制这些重要参数（θi\theta\_iθi​）相对于它们在旧数据上训练后的值（θold,i∗\theta\_{old, i}^\*θold,i∗​）的变化。

EWC 损失函数为：

L总=L新(θ)+λ2∑iFi(θi−θold,i∗)2L\_{总} = L\_{新}(\theta) + \frac{\lambda}{2} \sum\_i F\_i (\theta\_i - \theta\_{old, i}^\*)^2L总​=L新​(θ)+2λ​i∑​Fi​(θi​−θold,i∗​)2

这里，λ\lambdaλ 控制正则化强度。更高的 λ\lambdaλ 值会优先保持旧知识。

计算精确的 FIM 对于 LLM 来说计算成本很高。实际实现中，通常使用 FIM 的对角线近似值，它基于旧数据分布的梯度计算。即便如此，为数十亿参数计算和存储这些对角线元素也需要细致的实现。

#### 不遗忘学习 (LwF)

LwF 在新数据（DnewD\_{new}Dnew​）上训练时，使用知识蒸馏 (knowledge distillation)来保持旧模型（MoldM\_{old}Mold​）的行为。其思想是促使新模型（MnewM\_{new}Mnew​）在处理*新*数据时，产生与旧模型相似的输出（例如，logits 或词汇表 (vocabulary)上的概率分布）。

总损失结合了新数据真实标签（ynewy\_{new}ynew​）上的标准交叉熵损失（LCEL\_{CE}LCE​）和蒸馏损失（LdistillL\_{distill}Ldistill​），后者衡量 MnewM\_{new}Mnew​ 和 MoldM\_{old}Mold​ 在 xnewx\_{new}xnew​ 上的输出差异。

L总=L交叉熵(M新(x新),y新)+λL蒸馏(M旧(x新),M新(x新))L\_{总} = L\_{交叉熵}(M\_{新}(x\_{新}), y\_{新}) + \lambda L\_{蒸馏}(M\_{旧}(x\_{新}), M\_{新}(x\_{新}))L总​=L交叉熵​(M新​(x新​),y新​)+λL蒸馏​(M旧​(x新​),M新​(x新​))

蒸馏损失通常使用两个模型软化概率分布（使用温度 T>1T > 1T>1）之间的 KL 散度来实施。

```python

import torch.nn.functional as F

inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
labels = batch['labels'].to(device)

outputs_new = model(**inputs)
logits_new = outputs_new.logits
loss_ce = F.cross_entropy(
    logits_new.view(-1, logits_new.size(-1)),
    labels.view(-1)
)

temperature = 2.0
lambda_distill = 0.5

with torch.no_grad():
    outputs_old = old_model(**inputs)
    logits_old = outputs_old.logits

prob_new_soft = F.softmax(logits_new / temperature, dim=-1)
prob_old_soft = F.softmax(logits_old / temperature, dim=-1)

loss_distill = F.kl_div(
    F.log_softmax(logits_new / temperature, dim=-1)
     .view(-1, logits_new.size(-1)),
    prob_old_soft.view(-1, logits_old.size(-1)),
    reduction='batchmean'
) * (temperature**2)

total_loss = (1.0 - lambda_distill) * loss_ce + \
             lambda_distill * loss_distill
```

LwF 避免了存储旧数据或计算参数重要性矩阵的需要，这使得它与重放或 EWC 相比，在计算上更具吸引力，特别是对于非常大的模型。然而，它的效用取决于一个假设：即保持旧模型在新数据上的预测，是保留与*旧*数据分布相关知识的一个良好替代。

### 架构方法

虽然对于单个整体式LLM的纯持续预训练 (pre-training)来说不那么常见，但架构方法涉及修改模型的结构。渐进式神经网络 (neural network)等技术会冻结旧的网络部分，并为新任务添加新的“列”。另一种方法是使用参数 (parameter)高效模块，例如适配器（在第14章中讨论）。新的适配器可以针对新增数据进行训练，目标是将新知识隔离在这些小型模块中，同时保持主干部分冻结。然而，确保模型能够有效整合跨不同适配器或增量数据的知识，这仍然是一个研究领域。

### 持续预训练 (pre-training)的实际考量

无论选择何种策略，有几个实际方面很重要：

- **数据质量**：新传入数据（DnewD\_{new}Dnew​）的质量十分重要。彻底的清洗、过滤和去重（如第6-8章所述）是避免模型被噪声或不相关信息降低性能的必要步骤。
- **学习率**：与初始阶段相比，持续预训练期间通常使用小得多的学习率。这有助于防止导致遗忘的大幅、突然的参数 (parameter)变化。需要仔细调整学习率计划（例如，衰减、潜在的再热身阶段）。
- **更新频率**：决定持续预训练的频率需要平衡模型的时效性与计算成本，以及频繁更新可能带来的不稳定。这通常取决于相关新数据可用的速度以及应用程序对过时信息的敏感度。
- **评估**：评估非常重要。仅仅检查在 DnewD\_{new}Dnew​ 上的性能是不够的。还必须在代表*原始*数据分布（DoldD\_{old}Dold​）的保留验证集和潜在的中间分布上监控性能，以量化 (quantization)灾难性遗忘。

G

clusterₑval

评估

EvalOld

旧数据评估

EvalNew

新数据评估

OldModel

现有LLM
(在Dₒld上训练)

Strategy

持续训练策略
(重放 / 正则化 / 等)

OldModel->Strategy

UpdatedModel

更新后的LLM

Strategy->UpdatedModel

NewData

新数据 (Dₙew)

NewData->Strategy

UpdatedModel->EvalOld

检查遗忘

UpdatedModel->EvalNew

检查适应性

> 持续预训练的流程图，突出了使用新数据应用策略以及在旧数据和新数据分布上进行评估的重要步骤。

通常，最有效的方法会结合不同策略的元素。例如，将小型重放缓冲区与 EWC 或 LwF 结合使用可以提供互补优势。最佳策略在很大程度上取决于具体的限制（计算预算、存储容量、数据特性）以及保持旧知识与适应新信息之间的所需平衡。持续预训练仍然是一个活跃的研究领域，尤其是在将这些技术有效扩展到具有数万亿参数的基础模型方面。

获取即时帮助、个性化解释和交互式代码示例。

---

### 持续微调（SFT/RLHF）的策略

# 持续微调（SFT/RLHF）的策略

大型语言模型在首次部署后保持其对齐 (alignment)性是一个持续过程，而非一次性任务。用户期望会改变，新的安全问题会出现，并且期望的模型行为可能会随时间推移而漂移，或需要为特定应用进行修正。持续微调 (fine-tuning)，通过监督式方法（SFT）或强化学习 (reinforcement learning)（RLHF），提供逐步调整模型的机制。与初始微调不同，持续微调涉及将新数据或反馈整合到已运行的模型中，带来了效率、稳定性和知识保留方面的特别挑战。

### 持续监督式微调 (fine-tuning)（SFT）

持续SFT旨在根据新的监督式示例（例如，提示-完成对）更新模型遵循指令或执行特定任务的能力。这需要整合新数据的策略，同时不降低现有能力。

#### 数据来源与整合

新的SFT数据可以来自多个地方：

- **用户反馈：** 用户与模型交互时提供的明确更正或示例。
- **定向数据收集：** 识别模型表现不佳的方面（例如，特定类型的问题、安全场景），并积极整理或生成新示例。
- **合成数据：** 使用一个功能强大的模型（有时是模型本身或更大的模型）来生成新的指令-响应对，通常会进行质量筛选。

简单地在新数据上进行微调可能导致*灾难性遗忘*，即模型失去其之前学习过的任务能力。常见的缓解策略包括：

- **数据重放：** 将新的SFT示例与原始SFT数据的子集或之前微调轮次的代表性示例混合。旧数据与新数据的比例是一个重要的超参数 (parameter) (hyperparameter)。
- **权重 (weight)方案：** 在训练期间，对新数据与旧数据分配不同的重要性（损失权重），可能会对重要的安全或能力示例赋予更高的权重。
- **参数高效微调（PEFT）：** 像低秩适应（LoRA）这样的方法在这里特别有用。通过仅更新少量适配器权重，PEFT方法固有地限制了原始模型权重被修改的程度，从而减少了遗忘。仅训练适配器层在计算上也比完全微调便宜得多。

这是一个使用LoRA的简化PyTorch示例（假设`peft`等PEFT库可用），用于持续SFT步骤：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

model_name = "meta-llama/Llama-2-7b-hf"
base_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
```

#### 评估

评估持续SFT涉及检查以下方面的性能：

1. **新任务/指令：** 验证在新数据所针对的特定方面是否有改进。
2. **保留集：** 测量与新数据相关的未见示例上的泛化能力。
3. **回归基准：** 运行标准基准测试（例如，GLUE的子集、特定专业测试）或重要的安全评估，以确保先前能力没有明显下降。

### 持续人类反馈强化学习 (reinforcement learning)（RLHF）

RLHF使模型与复杂的人类偏好对齐 (alignment)，这些偏好通常与有用性、诚实性和无害性相关。持续RLHF涉及根据新的偏好数据更新奖励模型（RM）和/或策略模型。

#### 更新奖励模型（RM）

RM预测人类会偏爱两个响应中的哪一个。它需要定期更新，原因如下：

- 新的偏好数据可用（例如，来自持续的标注工作）。
- 对期望行为的理解有所演变（例如，新的安全指南）。
- 策略模型漂移，可能显示RM不准确的新方面。

**数据来源：** 新的偏好对（y1,y0∣xy\_1, y\_0 | xy1​,y0​∣x，其中给定提示xxx时y1y\_1y1​优于y0y\_0y0​) 的收集方式与初始RM训练类似，通常侧重于*当前*策略模型生成的输出。

**训练：** RM可以通过以下方式更新：

- **完全再训练：** 使用旧数据和新偏好数据从头开始训练新的RM。计算开销大但可能更有效。
- **增量微调 (fine-tuning)：** 在新的偏好数据上继续训练现有RM，有时与旧数据混合（重放）以防止忘记之前的偏好。这更快，但有RM漂移或过度拟合近期数据的风险。

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

rm_optimizer = torch.optim.AdamW(
    rm_model.parameters(), lr=1e-6
)

rm_model.train()
```

#### 更新策略模型

策略模型（即LLM本身）通过RL（通常是PPO）进行微调，以最大化RM预测的奖励，同时保持与原始SFT模型的接近（由KL散度惩罚控制）。持续RLHF更新涉及：

- **使用更新后的RM：** 使用奖励模型的最新版本执行PPO更新，确保策略与最新偏好对齐。
- **频率：** RL更新可能比RM更新发生得更频繁或更不频繁，这取决于新偏好数据的产生速度和观察到的策略漂移。
- **KL惩罚参考：** KL惩罚通常将策略约束在*原始*SFT模型或近期表现良好的快照，而不一定是紧随其前的策略版本。这作为一个锚点，防止过度偏离。
- **奖励作弊：** 持续监控是否存在策略利用RM漏洞而非根据人类偏好真正改进的迹象。这可能需要更新RM或调整RL过程。

使用`trl`等库的框架可能如下所示：

```python

```

#### 评估

评估持续RLHF是复杂的。指标包括：

- **RM准确性：** 更新后的RM与保留偏好集上的人类判断相关程度如何。
- **策略性能：** 自动化评估（例如，检查有害提示的拒绝情况）和人工评估（例如，与先前版本进行A/B测试）是必要的。
- **KL散度：** 监控与参考模型的偏离程度。
- **奖励分数：** 追踪策略获得的平均奖励，但需谨慎，因为奖励可能被作弊。

G

cluster\_SFT

持续SFT循环

cluster\_RLHF

持续RLHF循环

SFT\_Data

新SFT数据
(指令, 反馈)

SFT\_Mix

混合数据
(新 + 重放)

SFT\_Data->SFT\_Mix

SFT\_Train

微调
(例如，LoRA)

SFT\_Mix->SFT\_Train

SFT\_Model

更新的SFT模型

SFT\_Train->SFT\_Model

SFT\_Eval

评估SFT
(回归, 新任务)

SFT\_Eval->SFT\_Data

 识别需求

SFT\_Model->SFT\_Eval

Policy\_Train

RL微调策略
(PPO + KL惩罚)

SFT\_Model->Policy\_Train

 基础策略 / KL参考

Current\_Model

当前部署
模型状态

Current\_Model->SFT\_Train

 基础模型

Current\_Model->SFT\_Model

 更新

RLHF\_Model

更新的策略模型

Current\_Model->RLHF\_Model

 更新

Pref\_Data

新偏好数据

RM\_Train

更新奖励模型

Pref\_Data->RM\_Train

Updated\_RM

更新的RM

RM\_Train->Updated\_RM

Updated\_RM->Policy\_Train

 指导策略

Policy\_Train->RLHF\_Model

RLHF\_Eval

评估对齐性
(人工评估, 安全性)

RLHF\_Eval->Pref\_Data

 识别需求

RLHF\_Model->RLHF\_Eval

> 简化工作流程，展示持续SFT和RLHF的并行循环，更新已部署的模型状态。

### 挑战与注意事项

- **灾难性遗忘：** 在SFT和RLHF中仍然是一个重要的挑战。PEFT方法、重放缓冲区和正则化 (regularization)技术是必要的，但并非总是足够。需要细致的调整和评估。
- **数据质量：** 持续微调 (fine-tuning)的有效性取决于传入数据（指令或偏好）的质量和相关性。有偏差或噪声的数据可能降低性能或引入意外行为。数据验证管道很重要。
- **计算开销：** 即使是持续微调大型模型，也需要大量的计算资源。高效的策略（PEFT、优化的训练循环、分布式设置）对于实际实施是必要的。
- **评估复杂性：** 衡量对齐 (alignment)漂移和改进需要持续的、可能昂贵的人工评估以及自动化基准测试。定义“正确”行为也可能演变，使评估复杂化。
- **同步：** 决定SFT更新、RM更新和策略RL更新的节奏和相互依赖性是复杂的。过时的RM可能误导策略训练，而策略漂移可能需要更频繁的RM或SFT更新。

实施持续微调需要成熟的MLOps基础设施，能够处理数据管道、频繁的再训练任务、可靠的版本控制、分阶段推出以及全面的监控，以确保更新能提高对齐性，同时不导致模型能力或安全性的有害退步。

获取即时帮助、个性化解释和交互式代码示例。

---

### 安全地整合新数据源

# 安全地整合新数据源

将新数据源整合到现有的大语言模型 (LLM)训练流程中，对于保持模型最新、扩展其知识范围以及纠正部署后发现的偏差或知识空白，是必不可少的。然而，这个过程并非没有风险。简单地添加新数据可能会引入噪声、有害内容，或导致模型遗忘之前学到的信息（灾难性遗忘）。需要一种系统而审慎的方法来安全有效地整合新数据。

### 审查新数据源

在引入任何新数据集之前，它必须经过严格审查，类似于第7章中描述的初始数据预处理流程。在持续训练期间，风险可能更高，因为模型质量的退步可能会影响正在运行的应用程序。

1. **质量筛选：** 应用与初始预训练 (pre-training)时相同（或更严格）的质量启发式方法。这包括移除样板文本，根据长度或复杂度指标进行筛选，以及可能使用基于分类器的方法来移除低质量文本。目的是确保新数据达到既定的质量标准。
2. **去重：** 进行近似重复和精确重复检测，不仅在新数据集内部，还要与现有训练语料库进行比对。添加冗余信息会浪费计算资源，且学习收益甚微。MinHash（第7章介绍过）等方法仍然适用。
3. **内容安全与偏见分析：** 仔细检查新数据是否存在潜在的毒性、偏见和个人身份信息（PII）。自动化工具可以帮助标记 (token)有问题的内容，但有针对性的人工审查或抽样可能是必要的，特别是当数据源新颖或已知存在噪声时。考虑新数据对模型对齐 (alignment)（第25章和第26章）可能带来的影响。
4. **主题与语言验证：** 确保数据与模型更新预期的语言和主题保持一致。使用语言识别工具（第7章）并分析新数据的主题分布。不匹配的数据可能会意外地扭曲模型的能力。
5. **许可与来源：** 重新验证与新数据源相关的法律权限和许可条款。确保符合版权和使用限制，就像最初数据收集时一样（第6章）。

### 数据整合策略

一旦新数据源通过审查，可以采用几种策略将其整合到持续训练流程中。

1. **简单混合与重采样：** 最直接的方法是将清理过的新数据添加到现有训练池中，然后继续训练，从组合数据集中进行重采样。这需要仔细考虑*混合比例*或*源权重 (weight)*（第9章）。对新数据赋予过多权重会加速遗忘，而过少权重可能导致从新数据中学习效率低下。最佳比例通常取决于新数据相对于现有语料库的大小和相关性。
2. **数据复习（重放）：** 为了明确对抗灾难性遗忘，一种常用技术是*复习*或*重放*。训练时不是单独使用新数据或简单混合，而是每个训练批次都由新数据和*旧*数据的抽样子集组合构建。这迫使模型在学习新信息的同时回顾旧知识。每个批次中旧数据与新数据的比例成为一个重要的超参数 (parameter) (hyperparameter)。从旧数据中抽样可以是均匀的，也可以基于更复杂的方法，尽管均匀抽样通常在规模化实现时既有效又更简单。
3. **课程学习：** 逐步引入新数据。这可能涉及从新数据源的低抽样权重开始，并随时间逐步增加（退火策略，参见第9章）。或者，如果新数据代表一个显著不同的主题范围，可以安排课程，首先用旧数据巩固通用知识，然后重点关注新主题范围。

G

clusterₒld

现有训练语料库

clusterₙew

新数据源

OldData

审查过的旧数据

Sampler

数据采样器 / 混合器
（权重分配与复习逻辑）

OldData->Sampler

复习

RawNewData

原始新数据

Vetting

质量筛选
去重
安全检查

RawNewData->Vetting

NewData

审查过的新数据

Vetting->NewData

NewData->Sampler

新信息

Training

持续训练
流程

Sampler->Training

> 一个简化的流程图，展示了经过审查的旧数据源和新数据源如何通过采样器组合，然后输入到持续训练流程中。复习涉及从现有语料库中进行采样。

### 整合过程中的监控

当使用新数据源进行训练时，仔细监控是必不可少的。

1. **训练动态：** 密切跟踪训练损失、梯度范数和激活统计数据（第24章）。突然的峰值或不稳定可能表明新数据质量存在问题，或学习过程适应不良。如果可能，请关注来自不同数据源的损失贡献。
2. **验证性能：** 定期在多个验证集上评估模型：
   - 反映*原始*数据分布的验证集，用于检测灾难性遗忘。
   - 反映*新*数据分布的验证集，用于衡量适应情况。
   - 可能还有针对与新数据相关的特定能力或安全问题的验证集。
3. **下游任务评估：** 定期对下游任务（第22章）进行评估。在重要基准上的性能退步是一个明确的信号，表明整合可能正在造成损害。零样本或少样本评估可以提供快速反馈。

### 在PyTorch中实现数据混合

管理多个数据源通常涉及在PyTorch中创建自定义的 `Dataset` 或 `Sampler`。下面是一个使用 `torch.utils.data.ConcatDataset` 和 `WeightedRandomSampler` 进行简单混合并进行源加权的示例。

```python
import torch
from torch.utils.data import (Dataset, ConcatDataset, DataLoader,
                              WeightedRandomSampler)

class PlaceholderDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):

        return {
            "input_ids": torch.randint(0, 50000, (1024,)),
            "labels": torch.randint(0, 50000, (1024,))
        }

old_dataset = PlaceholderDataset(num_samples=1_000_000)
new_dataset = PlaceholderDataset(num_samples=200_000)

combined_dataset = ConcatDataset([old_dataset, new_dataset])

old_data_weight = 0.7 / len(old_dataset)
new_data_weight = 0.3 / len(new_dataset)

sample_weights = torch.cat([
    torch.full((len(old_dataset),), old_data_weight),
    torch.full((len(new_dataset),), new_data_weight)
])

effective_epoch_size = 500_000
sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=effective_epoch_size,
    replacement=True
)

batch_size = 8
data_loader = DataLoader(
    combined_dataset,
    sampler=sampler,
    batch_size=batch_size,
    num_workers=4
)

print(f"Combined dataset size: {len(combined_dataset)}")
print(f"Sampler using {len(sample_weights)} weights, sampling "
      f"{effective_epoch_size} indices per epoch.")

print(f"Sample weights (first 5): {sample_weights[:5]}")

print(f"Sample weights (last 5): {sample_weights[-5:]}")
```

> PyTorch 设置示例，组合了两个数据集并使用 `WeightedRandomSampler` 在训练期间控制混合比例。这种方法实现了简单的源加权混合。实现复习需要更复杂的采样器或数据集逻辑，以便在每个批次中明确地从旧数据源和新数据源中获取样本。

### 处理整合问题

如果监控发现引入新数据后出现显著的性能退步或训练不稳定等问题：

1. **暂停并分析：** 停止持续训练过程。仔细分析问题数据源和模型的失效模式。审查过程是否不足？是否是新数据的某个特定部分导致了问题？
2. **调整策略：** 修改整合方法。这可能意味着：
   - 降低新数据的采样权重 (weight)。
   - 增加复习数据的比例。
   - 更严格地筛选新数据源。
   - 调整学习率或其他优化器超参数 (parameter) (hyperparameter)（第17章）。
3. **回滚：** 在严重情况下，可能需要放弃最近的训练进度，并回滚到引入问题数据之前的检查点。这强调了检查点（第19章）和版本化策略（本章稍后介绍）的重要性。

引入新数据源是模型演进的有力工具，但它需要一种仔细、有条理的方法。严格的审查、策略性整合、持续监控以及调整或回滚的准备，都是在动态环境中安全更新大语言模型 (LLM)的必要组成部分。

获取即时帮助、个性化解释和交互式代码示例。

---

### 模型架构变动后的更新

# 模型架构变动后的更新

持续用新数据重新训练模型有助于保持其适用性，但有时性能会达到瓶颈，或新的研究显示出更好的架构组成部分。部署后更新模型的架构是一个复杂但可能带来丰厚回报的持续模型改进方面。这涉及修改神经网络 (neural network)的底层结构，从细微调 (fine-tuning)整到重大改造不等。

### 为何要变动架构？

初次部署后，有几个因素可能会促使架构修改：

1. **效率提升：** FlashAttention 或稀疏注意力模式等新技术可能会大大提升推理 (inference)速度或内存使用效率，使模型运行成本更低，或能在性能较低的硬件上部署。
2. **性能提升：** 研究可能表明，替代的激活函数 (activation function)（如 SwiGLU 替代 GeLU）、不同的归一化 (normalization)方案（如 Pre-LN 对比 Post-LN）或新型位置编码 (positional encoding)（如 RoPE）能在目标任务上带来更好的收敛或更高的准确度。
3. **处理限制：** 当前架构可能存在固有限制（例如，绝对位置编码带来的上下文 (context)长度限制），而较新的设计能够解决这些问题。
4. **新增功能：** 引入专家混合（MoE）层等结构可以大幅增加模型容量，而推理浮点运算次数不会按比例增加，这可能促成新的涌现 (emergence)能力。添加适配器层有助于未来更高效的领域适应。

### 架构修改的类别

架构变动在复杂性和影响方面有所不同：

- **小幅调整：** 这些涉及更改通常具有相似大小参数 (parameter)张量的组件，使得权重 (weight)重用可能可行。例如：
  - 交换前馈网络（FFN）层中的激活函数 (activation function)。
  - 调整层归一化 (normalization)（Pre-LN 对比 Post-LN）的位置。
  - 对隐藏维度或注意力头数量进行少量修改，前提是能保持兼容性。
- **重大改造：** 这些引入了根本上不同的结构或大幅改变张量形状，通常需要更复杂的权重迁移策略或重新训练。例如：
  - 整合优化过的注意力机制 (attention mechanism)（如 FlashAttention）。这需要更改注意力实现，但可能不会大幅改变核心权重矩阵，而是侧重于计算图。
  - 添加参数高效模块，如适配器或 LoRA 权重。这些会增加*新*参数。
  - 将标准 FFN 层转换为 MoE 层。这涉及添加门控网络和多个专家网络，从而大幅改变层结构。
  - 切换位置编码 (positional encoding)方法（例如，从学习的绝对编码到 RoPE）。这修改了位置信息注入的方式，影响输入嵌入 (embedding)或注意力计算。

### 更新架构的难题

修改已部署模型的架构并非易事，会带来一些工程方面的挑战：

1. **权重 (weight)兼容性和初始化：** 这通常是最主要的困难。如果新架构的层与旧架构相比形状或类型不同，简单加载旧检查点将会失败。
   - **分批加载：** 对于小幅更改，你可能可以使用 `strict=False` 加载现有检查点（PyTorch 中的 `state_dict`），然后手动映射或初始化不兼容的部分。
   - **权重修改：** 对于更复杂的更改，可能需要“权重修改”等技术，即对旧模型的权重进行重塑、平均或以其他方式转换，以智能地初始化新架构。这很复杂，且高度依赖具体的架构变动。
   - **从头重新训练：** 有时，变动是如此根本，以至于用旧权重初始化新架构几乎没有益处，或根本不可能。在这种情况下，更新后的模型可能需要从较早的初始检查点，甚至从头开始重新训练（或至少进行大量微调 (fine-tuning)），使用改进后的架构。
2. **训练动态：** 架构变动会改变损失和训练稳定性。为先前架构优化过的超参数 (parameter) (hyperparameter)（学习率、优化器设置、权重衰减、预热步数）可能不再是最优的。仔细监控梯度、损失曲线和激活统计数据十分必要，同时可能需要重新调整这些超参数。第 24 章中的技术在此变得重要。
3. **计算资源：** 新架构可能具有不同的计算特性。添加 MoE 层会大幅增加总参数数量，影响存储和可能的训练设置，即使每令牌的推理 (inference)浮点运算次数保持相似。优化过的注意力可能减少内存带宽需求，但可能依赖特定的硬件功能。有必要对训练时间、推理延迟/吞吐量 (throughput)和硬件要求进行投入产出权衡。
4. **评估与对比：** 确保新旧架构之间进行公平对比很重要。使用相同的评估数据集和步骤。如果架构变动旨在提升特定功能（例如，更长上下文 (context)处理），请确保评估指标能有效反映这一点。必须仔细监控既定基准测试上的性能退步。
5. **基础设施影响：** 影响推理的变动（例如，采用 FlashAttention、量化 (quantization)兼容性）需要与部署基础设施协调。服务框架、硬件加速器（GPU/TPU）和推理库可能需要更新或特定配置，以有效支持新架构。

### 实施架构更新的策略

考虑到其复杂性，建议采用结构化方法：

- **增量式更改：** 如果可能，逐步引入架构变动，而非一次性全部实施。这有助于更轻松地调试和归因性能变化。
- **消融研究：** 在进行大规模重新训练之前，在较小的模型变体或数据集上进行消融研究，以验证架构变动带来的预期益处。
- **知识蒸馏 (knowledge distillation)：** 如果从头重新训练成本过高，可以考虑使用原始模型作为“教师”来将知识蒸馏到新的“学生”架构中。这可以加速训练，并帮助新模型更快地达到可比性能。
- **全面测试：** 不仅要通过标准基准测试，还要通过定性分析和针对性探查来严格测试更新后的模型，以确保其保留所需行为，且不会引入新的故障类型（第 23 章）。
- **金丝雀发布和 A/B 测试：** 最初将架构更新后的模型部署给一小部分用户（金丝雀发布），或运行 A/B 测试，在生产环境中将其与先前版本直接对比。在全面推广之前，密切监控性能、稳定性和用户反馈（第 29 章）。

Here's a simplified workflow diagram:

G

IdentifyNeed

识别需求
(效率, 性能, 限制)

ProposeChange

提出架构
变动

IdentifyNeed->ProposeChange

CompatCheck

权重兼容性
检查

ProposeChange->CompatCheck

Retrain

重新训练 / 微调
(可能蒸馏)

CompatCheck->Retrain

不兼容 /
需要训练

Eval

评估
(基准, 定性)

CompatCheck->Eval

兼容 /
加载权重

Retrain->Eval

Deploy

部署
(金丝雀, A/B 测试)

Eval->Deploy

通过

NoChange

不变动 /
重新评估

Eval->NoChange

失败

Deploy->IdentifyNeed

监控与迭代

> 持续训练模型中实施架构变动的决策流程。

我们来看一个 PyTorch 示例。设想在前馈网络（FFN）模块中，将标准的 `nn.GELU` 激活函数 (activation function)替换为 `nn.SiLU`（与 Swish/SwiGLU 相关）。

```python
import torch
import torch.nn as nn

class OriginalFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

class UpdatedFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)

        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

d_model = 512
d_ff = 2048

old_ffn = OriginalFFN(d_model, d_ff)
new_ffn = UpdatedFFN(d_model, d_ff)

old_state_dict = old_ffn.state_dict()

try:
    new_ffn.load_state_dict(old_state_dict)
    print("成功将权重加载到更新的架构中。")
except Exception as e:
    print(f"直接加载权重失败：{e}")
    print("可能需要手动权重映射或重新训练。")
```

在这个简单的例子中，因为 `nn.GELU` 和 `nn.SiLU` 是 `forward` 传递中的功能性改变，并且不会在状态字典中引入具有这些特定名称的新可学习参数 (parameter)，所以加载线性层的权重 (weight)可能会成功。然而，由于激活函数的不同，模块的*行为*将发生变化。仅此一项改变就可能影响收敛动态，需要在持续训练期间调整学习率或训练计划。更复杂的架构变动，例如修改维度或添加全新层（如用于 MoE 的门控网络），将需要明确处理状态字典不匹配的情况，并且可能需要更广泛的重新训练或微调 (fine-tuning)。

更新模型架构是 LLM MLOps 生命周期中的一种高级技术。它要求周密的规划、权重管理和训练的工程实践、全面的评估以及协调一致的部署策略。尽管具有挑战性，但这对于模型长期保持性能、效率和功能上的竞争力极为重要。

获取即时帮助、个性化解释和交互式代码示例。

---

### 版本管理、部署与回滚策略

# 版本管理、部署与回滚策略

管理持续更新的大语言模型 (LLM)生命周期需要严谨的工程实践，特别是在新版本如何追踪、部署以及可能回退方面。如果没有健全的策略，将更新后的模型引入生产环境可能导致性能下降、异常行为或服务中断。本节阐述了适用于持续训练中的大语言模型的版本管理、部署和回滚的实用方法。

### 模型版本管理

有效的版本管理对于追踪模型演进、确保可复现性并实现安全回滚非常重要。仅仅保存模型权重 (weight)是不够的；版本管理必须包含所有相关成果物和元数据。

**版本管理方案：** 采用语义化版本管理（SemVer - `MAJOR.MINOR.PATCH`）提供了一种实用的结构：

- **MAJOR（主版本，例如2.0.0）：** 在发生重大架构变化、不兼容的API变更，或在显著不同数据集上重新训练导致模型能力或行为从根本上改变时增加。
- **MINOR（次版本，例如1.1.0）：** 在进行实质性更新时增加，例如在新的数据切片上持续预训练 (pre-training)，使用大量新指令数据进行新的SFT阶段，或进行重大超参数 (parameter) (hyperparameter)调整以提升性能但保持核心能力的向后兼容性。
- **PATCH（修订版本，例如1.0.1）：** 在进行小错误修复、微调 (fine-tuning)数据的小幅调整或不会显著改变模型核心行为或性能特征的小优化时增加。

**成果物和元数据追踪：** 每个版本标签应与以下内容相关联：

1. **模型权重：** 模型实际的参数。
2. **分词 (tokenization)器 (tokenizer)配置：** 包括词汇文件和任何特定配置（`tokenizer.json`、`vocab.txt`等）。使用不兼容的分词器可能导致静默故障或性能下降。
3. **模型配置：** 详细说明架构、隐藏层大小、层数、激活函数 (activation function)等的文件（`config.json`）。
4. **训练元数据：** 使用的数据集来源和版本、重要的超参数、训练脚本的提交哈希、在标准基准上的评估指标，以及可能的特定硬件/软件环境。

诸如Git大文件存储（LFS）之类的工具可以在Git仓库中管理大型权重文件，而机器学习 (machine learning)实验追踪平台（例如MLflow、Weights & Biases）旨在系统地记录成果物和元数据。

```python

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_version = "1.1.0"
model_save_path = f"./llm_model_v{model_version}"
tokenizer_save_path = f"./llm_tokenizer_v{model_version}"

model.save_pretrained(model_save_path)

tokenizer.save_pretrained(tokenizer_save_path)

print(f"模型保存至：{model_save_path}")
print(f"分词器保存至：{tokenizer_save_path}")
```

### 部署策略

部署数千兆字节或数太字节的模型需要仔细规划，以最大程度地减少停机时间和风险。常见策略包括：

**蓝绿部署：** 维护两个相同的生产环境：“蓝色”（当前线上版本）和“绿色”（新版本）。一旦“绿色”环境测试完毕并准备就绪，负载均衡器将所有流量从“蓝色”环境重定向到“绿色”环境。

G

cluster\_blue

蓝色环境 (v1.0)

cluster\_green

绿色环境 (v1.1)

Blue\_Instance1

Blue\_Instance1

Blue\_Instance2

Blue\_Instance2

Green\_Instance1

Green\_Instance1

Green\_Instance2

Green\_Instance2

LoadBalancer

LoadBalancer

LoadBalancer->Blue\_Instance1

 100% 流量

LoadBalancer->Blue\_Instance2

UserTraffic

用户
流量

UserTraffic->LoadBalancer

> 蓝绿部署：活动流量导向蓝色环境。绿色环境存放新版本，准备切换。

- **优点：** 瞬间流量切换，切换时零停机，回滚简单（只需将流量切换回蓝色环境）。
- **缺点：** 需要两倍的基础设施资源，可能成本较高。在切换前彻底测试绿色环境很重要。

**金丝雀发布：** 逐渐将小部分流量路由到新的模型版本（“金丝雀”版本）。密切监控性能和错误指标。如果金丝雀版本表现良好，逐渐增加流量百分比，直到100%的流量路由到新版本。

G

clusterᵥ1

主部署 (v1.0)

clusterᵥ2\_canary

金丝雀部署 (v1.1)

V1\_Instance1

V1\_Instance1

V1\_Instance2

V1\_Instance2

V1\_Instance3

V1\_Instance3

V2\_Canary\_Instance

V2\_Canary\_Instance

LoadBalancer

LoadBalancer

LoadBalancer->V1\_Instance1

 90% 流量

LoadBalancer->V1\_Instance2

LoadBalancer->V1\_Instance3

LoadBalancer->V2\_Canary\_Instance

 10% 流量

UserTraffic

用户
流量

UserTraffic->LoadBalancer

> 金丝雀发布：一小部分流量路由到新版本（金丝雀），而大多数用户仍停留在稳定版本。

"\* **优点：** 限制潜在问题的影响范围，支持测试和性能比较，渐进式发布降低风险。"

- **缺点：** 更复杂的基础设施和监控设置，发布阶段可能出现用户体验不一致，发布过程较慢。

**影子部署：** 将新模型版本与当前版本并行部署。将线上流量路由到当前版本，但也将请求镜像或“影子”到新版本。比较影子模型的输出和性能，而不影响用户。

- **优点：** 在实际生产负载下测试新模型的最安全方式，可以比较其行为与当前版本而不影响用户。
- **缺点：** 需要大量额外的计算资源来运行两个模型，设置镜像和比较逻辑的复杂性较高。

策略的选择取决于风险承受能力、资源可用性以及模型更新的性质。对于大语言模型 (LLM)而言，由于难以察觉的细微退步可能难以发现，金丝雀或影子部署尽管复杂，但常被选择。

### 回滚策略

即使经过仔细测试和部署，新模型版本在生产中仍可能出现未预料的问题（例如，延迟增加、错误率升高、有害生成模式、在特定用户群体上表现不佳）。明确定义的回滚策略是必需的。

**回滚规划：**

1. **准备就绪：** 确保以前的稳定版本（模型成果物、分词 (tokenization)器 (tokenizer)、配置）在成果物存储中随时可用。
2. **机制：** 回滚机制与部署策略紧密关联：
   - **蓝绿部署：** 通过将流量重定向回之前活动的（蓝色）环境，实现简单的回滚。
   - **金丝雀发布：** 将金丝雀流量百分比降至0%，并可能移除金丝雀部署。
   - **影子部署：** 只需移除影子部署；未影响用户流量。
3. **触发条件：** 定义清晰的回滚触发条件。这可以根据监控性能指标（KPI）或健康度指标自动化：
   - 推理 (inference)延迟显著增加。
   - 与模型输出相关的应用层面错误率升高。
   - 用户参与度指标下降（如适用）。
   - 关键评估分数（例如，毒性、对齐 (alignment)指标）超过预设阈值。
   - 基于用户投诉或定性审查的手动触发。
4. **回滚后分析：** 回滚后，进行彻底的根本原因分析，以理解新版本为何在生产中失败，然后才能尝试重新部署。

**自动化回滚示例：**

```python

import time
import random

CURRENT_MODEL_VERSION = "1.0.1"
CANARY_MODEL_VERSION = "1.1.0"
CANARY_TRAFFIC_PERCENT = 10

MAX_ERROR_RATE_THRESHOLD = 0.05
MAX_LATENCY_MS_THRESHOLD = 500

def get_canary_metrics():

  simulated_error_rate = random.uniform(0.01, 0.07)
  simulated_latency = random.uniform(300, 600)
  print(
      f"金丝雀指标 - 错误率: {simulated_error_rate:.3f}, "
      f"延迟: {simulated_latency:.0f}毫秒"
  )
  return {"error_rate": simulated_error_rate, "latency_p95": simulated_latency}

def set_traffic_split(canary_percent):

  global CANARY_TRAFFIC_PERCENT
  CANARY_TRAFFIC_PERCENT = canary_percent
  print(f"--- 将金丝雀流量设置为 {canary_percent}% ---")

def rollback_deployment():
  print("!!! 正在回滚金丝雀部署 !!!")
  set_traffic_split(0)

  print(
      f"--- 回滚完成。流量已恢复到 {CURRENT_MODEL_VERSION} ---"
  )

while CANARY_TRAFFIC_PERCENT > 0:
  time.sleep(60)
  metrics = get_canary_metrics()

  if metrics["error_rate"] > MAX_ERROR_RATE_THRESHOLD or \
     metrics["latency_p95"] > MAX_LATENCY_MS_THRESHOLD:
    rollback_deployment()

    break
  else:

    print("金丝雀指标稳定。")

if CANARY_TRAFFIC_PERCENT > 0 :
    print(
        f"金丝雀部署 {CANARY_MODEL_VERSION} 看起来稳定，"
        f"流量在 {CANARY_TRAFFIC_PERCENT}%。考虑全面发布。"
    )
```

实施版本管理、部署和回滚策略，将持续的模型更新从高风险的工作转变为可管理的工程过程。这些实践对于维护在动态生产环境中运行的大语言模型 (LLM)的可靠性和性能非常重要。

获取即时帮助、个性化解释和交互式代码示例。

---
