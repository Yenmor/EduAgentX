# 高级Transformer架构

## Chapter 1 Revisiting Sequence Modeling Limitations

### 循环网络中的顺序计算

# 循环网络中的顺序计算

循环神经网络 (neural network)（RNN），包括LSTMs和GRUs等更复杂的变体，都遵循逐元素处理序列的原理进行设计。在每个时间步 ttt，RNN 接收输入 xtx\_txt​ 和来自上一个时间步的隐藏状态 ht−1h\_{t-1}ht−1​，以计算当前隐藏状态 hth\_tht​。这个过程可以抽象地表示为：

ht=f(ht−1,xt;θ)h\_t = f(h\_{t-1}, x\_t; \theta)ht​=f(ht−1​,xt​;θ)

其中 fff 代表循环函数（例如，涉及矩阵乘法和tanh或sigmoid等激活函数 (activation function)，或者LSTMs/GRUs中更复杂的门控逻辑），并由权重 (weight) θ\thetaθ 参数 (parameter)化。在每个时间步也可以生成一个可选输出 yty\_tyt​，通常由 hth\_tht​ 得到。

这种表述显示了循环模型的一个基本特性：一种固有的**顺序依赖**。时间步 ttt 隐藏状态的计算*必须*等待时间步 t−1t-1t−1 的计算完成后才能进行。这种依赖形成了一个贯穿整个序列长度的链条。

G

ht₁

h(t-1)

ht

h(t)

ht₁->ht

状态传递

xt₁

x(t-1)

xt₁->ht₁

htₚlus₁

h(t+1)

ht->htₚlus₁

状态传递

xt

x(t)

xt->ht

xtₚlus₁

x(t+1)

xtₚlus₁->htₚlus₁

> 一个展开的RNN，说明了信息流的顺序性。隐藏状态 hth\_tht​ 直接依赖于前一个状态 ht−1h\_{t-1}ht−1​ 和当前输入 xtx\_txt​。

尽管单个时间步*内部*的计算（例如函数 fff 内的矩阵乘法）通常可以利用GPU等并行硬件，但*跨*时间步的计算无法并行化。你不能同时计算 hth\_tht​ 和 ht+1h\_{t+1}ht+1​，因为 ht+1h\_{t+1}ht+1​ 需要 hth\_tht​ 作为输入。

这种顺序约束对性能有重要影响：

1. **训练时间：** 对于长度为 LLL 的序列，前向和后向传播的时间复杂度与 LLL 呈线性关系。处理更长的序列会直接导致更长的训练时间，不论针对步内计算可用的并行处理能力有多强。
2. **推理 (inference)延迟：** 同样，生成输出序列或基于长输入序列进行预测，需要顺序迭代每个时间步，从而增加了延迟。

这与应用于序列的前馈网络或卷积神经网络（CNN）（例如1D卷积）等架构形成鲜明对比。在这些模型中，输入序列不同部分的计算通常可以独立并行地执行，从而带来更高的效率，特别是在专用硬件上。

尽管LSTMs和GRUs引入了更先进的门控机制，以更好地控制信息流并解决梯度问题（接下来会讨论），但它们本质上仍然遵循相同的顺序处理方法。时间 ttt 的门（输入、遗忘、输出）和单元状态的计算仍然依赖于时间 t−1t-1t−1 的隐藏状态和单元状态。

因此，无法在序列长度维度上并行化计算，是循环网络主要设计中固有的一个基本瓶颈。这个局限是促使我们发展出其他架构（如Transformer）的主要驱动力，这些架构能够更并发地处理序列元素。

获取即时帮助、个性化解释和交互式代码示例。

---

### 梯度消失与梯度爆炸问题

# 梯度消失与梯度爆炸问题

递归神经网络 (neural network) (RNN) 的序列特性使其能够逐步处理序列，但在训练时会造成明显的障碍，尤其是对于深度网络或长序列。训练 RNN 的主要机制是时间反向传播 (backpropagation) (BPTT)，它本质上将递归网络沿序列长度展开，并应用标准的反向传播算法。这个过程涉及到通过对每个时间步重复应用链式法则反向传播，来计算损失函数 (loss function)相对于网络参数 (parameter)的梯度。

考虑计算最终时间步 TTT 处损失 LLL 相对于某个较早时间步 kkk 处隐藏状态 hkh\_khk​ 的梯度。使用链式法则，这涉及将中间时间步的雅可比矩阵相乘：

∂L∂hk=∂L∂hT∂hT∂hT−1∂hT−1∂hT−2…∂hk+1∂hk\frac{\partial L}{\partial h\_k} = \frac{\partial L}{\partial h\_T} \frac{\partial h\_T}{\partial h\_{T-1}} \frac{\partial h\_{T-1}}{\partial h\_{T-2}} \dots \frac{\partial h\_{k+1}}{\partial h\_k}∂hk​∂L​=∂hT​∂L​∂hT−1​∂hT​​∂hT−2​∂hT−1​​…∂hk​∂hk+1​​

这个公式表明了核心问题。项 ∂ht∂ht−1\frac{\partial h\_t}{\partial h\_{t-1}}∂ht−1​∂ht​​ 代表了时间 ttt 的隐藏状态相对于时间 t−1t-1t−1 的隐藏状态的变化方式。这个雅可比矩阵取决于循环权重 (weight)矩阵 WhhW\_{hh}Whh​ 和循环转换中使用的激活函数 (activation function)的导数。来自最终损失的梯度信号必须通过这些雅可比矩阵的乘积反向传播。

### 梯度消失

如果这些雅可比矩阵 ∂ht∂ht−1\frac{\partial h\_t}{\partial h\_{t-1}}∂ht−1​∂ht​​ 的范数（或更正式地说，奇异值）持续小于 1，它们的乘积在反向时间传播 (T−kT-kT−k 步) 时会呈指数级缩小。

∂L∂hk≈∂L∂hT∏t=k+1TJt这里 Jt=∂ht∂ht−1\frac{\partial L}{\partial h\_k} \approx \frac{\partial L}{\partial h\_T} \prod\_{t=k+1}^{T} J\_t \quad \text{这里 } J\_t = \frac{\partial h\_t}{\partial h\_{t-1}}∂hk​∂L​≈∂hT​∂L​t=k+1∏T​Jt​这里 Jt​=∂ht−1​∂ht​​

如果 ∣∣Jt∣∣<1||J\_t|| < 1∣∣Jt​∣∣<1 平均而言，那么当 T−kT-kT−k 增加时，∣∣∏t=k+1TJt∣∣|| \prod\_{t=k+1}^{T} J\_t ||∣∣∏t=k+1T​Jt​∣∣ 会非常迅速地趋近于零。这意味着梯度信号在到达较早的时间步之前就有效地消失了。

**影响：**

- **难以学习长程依赖：** 网络难以学习序列中相距较远的输入和输出之间的关联。负责捕获这些长期效应（主要受来自早期时间步如 kkk 的梯度影响）的权重 (weight)获得的更新微乎其微。
- **训练缓慢：** 与序列开头相关的参数 (parameter)学习速度远慢于接近末尾的参数。

双曲正切 (tanh) 或 Sigmoid 等激活函数 (activation function)的选择（这些在旧式 RNN 中常用）使这个问题更为严重，因为它们的导数严格小于 1（tanh 在单一点除外）。

### 梯度爆炸

相反，如果雅可比矩阵 ∂ht∂ht−1\frac{\partial h\_t}{\partial h\_{t-1}}∂ht−1​∂ht​​ 的范数持续大于 1，它们的乘积在反向传播 (backpropagation)时会呈指数级增大。

**影响：**

- **训练不稳定：** 梯度变得过大，导致权重 (weight)的更新量巨大。这可能导致优化过程发散，常在计算中导致数值溢出（NaN，即“非数字”值）。
- **破坏性更新：** 即使训练没有完全发散，大的梯度更新也可能抹去先前学到的知识。

虽然梯度爆炸通常更容易检测和缓解（例如，使用梯度裁剪，即当梯度超过某个阈值时将其缩小），但它们仍然对稳定训练构成明显挑战。

024681012140246810消失 (例如，||J|| = 0.8)爆炸 (例如，||J|| = 1.2)梯度随时间传播示意图反向时间步 (T-k)相对梯度幅度

> 该图显示了当梯度通过时间步反向传播时，重复乘以略小于 1 的值（蓝线，范数 0.8）会导致指数衰减（消失），而重复乘以略大于 1 的值（红线，范数 1.2）会导致指数增长（爆炸）。Y 轴被截断以呈现这两种趋势。

这些梯度问题从根本上限制了简单 RNN 架构有效建模序列的能力，尤其是在存在长期模式时。这一限制是推动 LSTM 和 GRU 等更复杂循环单元发展的主要动力，我们将在接下来审视它们。

获取即时帮助、个性化解释和交互式代码示例。

---

### 长短期记忆（LSTM）门控机制

# 长短期记忆（LSTM）门控机制

简单循环神经网络 (neural network)（RNN）内部的核心数学运算，即跨时间步重复进行的矩阵乘法，直接导致了梯度消失和梯度爆炸问题。训练深度循环网络变得不稳定，使得捕获序列中相距较远元素之间的依赖关系变得困难。长短期记忆（LSTM）网络由Hochreiter和Schmidhuber于1997年提出，专门设计用于通过一个更精巧的内部结构来解决这些问题，该结构内含*门控机制*。

LSTM的核心创新是除了隐藏状态（hth\_tht​）之外，还引入了**单元状态**（CtC\_tCt​）。可以将单元状态想象成一条信息高速公路或记忆传送带。它贯穿整个序列，只有轻微的线性交互。信息可以被添加到单元状态中或从其中移除，这些操作由称为**门**的结构精确地调控。

这些门由一个sigmoid神经网络层和一个逐点乘法操作组成。sigmoid层输出0到1之间的数字，用于描述每个分量应该通过多少。值为1表示“让所有信息通过”，而值为0表示“不让任何信息通过”。一个LSTM单元通常包含三个这样的门，以保护和控制单元状态。

让我们在特定时间步 ttt 查看每个门，考虑当前输入 xtx\_txt​、前一个隐藏状态 ht−1h\_{t-1}ht−1​ 和前一个单元状态 Ct−1C\_{t-1}Ct−1​。

### 遗忘门 (ftf\_tft​)

第一步是决定我们要从单元状态中丢弃哪些信息。这个决定由遗忘门做出。它查看 ht−1h\_{t-1}ht−1​ 和 xtx\_txt​，并为单元状态 Ct−1C\_{t-1}Ct−1​ 中的每个数字输出一个介于0和1之间的值。

ft=σ(Wf[ht−1,xt]+bf)f\_t = \sigma(W\_f [h\_{t-1}, x\_t] + b\_f)ft​=σ(Wf​[ht−1​,xt​]+bf​)

这里，[ht−1,xt][h\_{t-1}, x\_t][ht−1​,xt​] 表示前一个隐藏状态和当前输入向量 (vector)的拼接。WfW\_fWf​ 代表遗忘门的权重 (weight)矩阵，bfb\_fbf​ 代表其偏置 (bias)向量。sigmoid函数 σ\sigmaσ 将输出压缩到 [0, 1] 的范围。接近0的值表示忘记 Ct−1C\_{t-1}Ct−1​ 中相应的信息，而接近1的值表示保留它。

### 输入门 (iti\_tit​) 和候选值 (C~t\tilde{C}\_tC~t​)

接下来，我们需要决定将哪些新信息存储到单元状态中。这包括两个部分：

1. **输入门** (iti\_tit​) 是另一个sigmoid层，它决定我们将更新哪些值。
2. 一个 `tanh` 层创建新的候选值向量 (vector) C~t\tilde{C}\_tC~t​，这些值可以被添加到状态中。

it=σ(Wi[ht−1,xt]+bi)i\_t = \sigma(W\_i [h\_{t-1}, x\_t] + b\_i)it​=σ(Wi​[ht−1​,xt​]+bi​)
C~t=tanh⁡(WC[ht−1,xt]+bC)\tilde{C}\_t = \tanh(W\_C [h\_{t-1}, x\_t] + b\_C)C~t​=tanh(WC​[ht−1​,xt​]+bC​)

类似于遗忘门，Wi,bi,WC,bCW\_i, b\_i, W\_C, b\_CWi​,bi​,WC​,bC​ 是在训练过程中学到的权重 (weight)矩阵和偏置 (bias)向量。`tanh` 函数输出介于-1和1之间的值，代表对单元状态的潜在更新（正向或负向）。

现在，我们将旧的单元状态 Ct−1C\_{t-1}Ct−1​ 更新为新的单元状态 CtC\_tCt​。我们将旧状态乘以 ftf\_tft​，忘记了我们之前决定忘记的内容。然后我们加上 it∗C~ti\_t \* \tilde{C}\_tit​∗C~t​。这是新的候选值，根据我们决定更新每个状态值的程度进行缩放。

Ct=ft∗Ct−1+it∗C~tC\_t = f\_t \* C\_{t-1} + i\_t \* \tilde{C}\_tCt​=ft​∗Ct−1​+it​∗C~t​

符号 ∗\*∗ 表示逐元素乘法。

### 输出门 (oto\_tot​)

最后，我们需要决定输出什么。这个输出将基于我们的单元状态，但会是一个经过筛选的版本。首先，我们运行一个sigmoid层，它决定我们将输出单元状态的哪些部分。

ot=σ(Wo[ht−1,xt]+bo)o\_t = \sigma(W\_o [h\_{t-1}, x\_t] + b\_o)ot​=σ(Wo​[ht−1​,xt​]+bo​)

然后，我们将单元状态 CtC\_tCt​ 通过 `tanh` （将值推至-1和1之间）并将其乘以sigmoid门 oto\_tot​ 的输出，这样我们只输出了我们决定要输出的部分。这个结果就是新的隐藏状态 hth\_tht​。

ht=ot∗tanh⁡(Ct)h\_t = o\_t \* \tanh(C\_t)ht​=ot​∗tanh(Ct​)

这个 hth\_tht​ 被传递到下一个时间步，也可以作为当前时间步LSTM单元的输出进行预测。

G

clusterₗstm

LSTM单元 (时间 t)

xt

xₜ

forget\_gate

σ
f

xt->forget\_gate

input\_gate

σ
i

xt->input\_gate

output\_gate

σ
o

xt->output\_gate

candidateᵥalues

tanh
C

xt->candidateᵥalues

ht₁

hₜ₋₁

ht

hₜ

ht₁->forget\_gate

[h,x]

ht₁->input\_gate

[h,x]

ht₁->output\_gate

[h,x]

ht₁->candidateᵥalues

[h,x]

Ct₁

Cₜ₋₁

Ct

Cₜ

Ct₁->Ct

mul\_f\_Ct1

\*

Ct₁->mul\_f\_Ct1

tanh\_Ct

tanh

Ct->tanh\_Ct

forget\_gate->mul\_f\_Ct1

mulᵢ\_Ctilda

\*

input\_gate->mulᵢ\_Ctilda

mulₒₜanhCt

\*

output\_gate->mulₒₜanhCt

candidateᵥalues->mulᵢ\_Ctilda

add\_Ct

+

mul\_f\_Ct1->add\_Ct

mulᵢ\_Ctilda->add\_Ct

mulₒₜanhCt->ht

add\_Ct->Ct

tanh\_Ct->mulₒₜanhCt

> LSTM单元的内部结构。门（sigmoid σ\sigmaσ）控制信息进出单元状态 (CtC\_tCt​) 的流动，由绿色路径表示。隐藏状态 (hth\_tht​) 是单元状态的筛选版本。

### 门控如何缓解梯度消失问题

主要的发现是单元状态的加性交互。新信息被*添加*到单元状态中（通过 it∗C~ti\_t \* \tilde{C}\_tit​∗C~t​），旧信息被*移除*（通过乘以 ftf\_tft​），而不像简单RNN那样通过矩阵乘法和非线性操作重复转换。遗忘门允许单元状态在需要时长时间保留信息（通过将 ftf\_tft​ 设置接近1）。

这种结构创建了梯度可以反向传播 (backpropagation)而不迅速消失的路径。门学习控制这种流动，根据上下文 (context)打开或关闭对单元状态的访问。如果遗忘门大部分是打开的 (ft≈1f\_t \approx 1ft​≈1) 并且输入门大部分是关闭的 (it≈0i\_t \approx 0it​≈0)，单元状态可以将其信息在许多时间步内大致不变地传递，从而保留梯度。

尽管LSTM代表了显著的进展，并使得许多以前对简单RNN来说难以处理的序列建模任务取得了进展，但它们并不是一个完美的解决方案。它们仍然按顺序处理信息，限制了训练和推理 (inference)过程中的并行化。此外，尽管它们在捕获更长依赖方面远优于简单RNN，但它们在处理数千时间步中存在细微依赖的极长序列时仍然可能遇到困难。门控机制的复杂性也增加了相比简单模型的计算开销。这些未解决的问题为Transformer等架构的出现创造了条件，该架构完全放弃了循环。

获取即时帮助、个性化解释和交互式代码示例。

---

### 门控循环单元 (GRU) 架构

# 门控循环单元 (GRU) 架构

门控循环单元 (GRU)，由 Cho 等人于 2014 年提出，提供了一种门控循环架构中的替代方案。这些单元与长短期记忆 (LSTM) 网络类似，旨在缓解简单循环神经网络 (neural network) (RNN) 中固有的梯度问题。GRU 的目标是实现类似的效果，即控制信息随时间流动，但与 LSTM 相比，它通过一种略微简化的结构来实现。这种简化通常会减少参数 (parameter)，并可能加快计算速度，同时其性能常与 LSTM 相当。

### GRU 门和状态更新

GRU 单元不使用独立的细胞状态，而是直接通过两个主要门控机制来修改其隐藏状态 hth\_tht​：重置门和更新门。我们来分析它们的作用和计算方式。

#### 重置门 (rtr\_trt​)

重置门决定了在提出新的候选隐藏状态时，前一个隐藏状态 ht−1h\_{t-1}ht−1​ 有多少应该被有效地“遗忘”或忽略。如果重置门输出接近 0 的值，它允许单元丢弃过去被认为与当前计算无关的信息。反之，接近 1 的值则保留了前一状态的大部分信息。

计算涉及当前输入 xtx\_txt​ 和前一个隐藏状态 ht−1h\_{t-1}ht−1​。一个 Sigmoid 函数 σ\sigmaσ 将输出压缩到 [0, 1] 范围：

rt=σ(Wrxt+Urht−1+br)r\_t = \sigma(W\_r x\_t + U\_r h\_{t-1} + b\_r)rt​=σ(Wr​xt​+Ur​ht−1​+br​)

这里，WrW\_rWr​、UrU\_rUr​ 和 brb\_rbr​ 是针对重置门学习到的权重 (weight)矩阵和偏置 (bias)向量 (vector)。

#### 更新门 (ztz\_tzt​)

更新门的作用类似于 LSTM 中遗忘门和输入门的组合。它决定了前一个隐藏状态 ht−1h\_{t-1}ht−1​ 有多少信息应该传递到新的隐藏状态 hth\_tht​。同时，它也控制着新计算出的*候选*隐藏状态 h~t\tilde{h}\_th~t​ 有多少应该被纳入。

它的计算结构与重置门相似：

zt=σ(Wzxt+Uzht−1+bz)z\_t = \sigma(W\_z x\_t + U\_z h\_{t-1} + b\_z)zt​=σ(Wz​xt​+Uz​ht−1​+bz​)

WzW\_zWz​、UzU\_zUz​ 和 bzb\_zbz​ 是用于更新门学习到的参数 (parameter)。

#### 候选隐藏状态 (h~t\tilde{h}\_th~t​)

候选隐藏状态表示在时间步 ttt 新隐藏状态的一个提议。它的计算受重置门 rtr\_trt​ 的影响。具体来说，前一个隐藏状态 ht−1h\_{t-1}ht−1​ 的贡献通过重置门的输出进行调节（按元素相乘，表示为 ⊙\odot⊙），然后与处理后的输入 xtx\_txt​ 结合。通常使用双曲正切函数 (tanh⁡\tanhtanh) 作为激活函数 (activation function)：

h~t=tanh⁡(Whxt+Uh(rt⊙ht−1)+bh)\tilde{h}\_t = \tanh(W\_h x\_t + U\_h (r\_t \odot h\_{t-1}) + b\_h)h~t​=tanh(Wh​xt​+Uh​(rt​⊙ht−1​)+bh​)

WhW\_hWh​、UhU\_hUh​ 和 bhb\_hbh​ 是用于计算候选状态的学习参数。按元素相乘 rt⊙ht−1r\_t \odot h\_{t-1}rt​⊙ht−1​ 是允许 GRU 根据 rtr\_trt​ 选择性地丢弃前一状态部分内容的机制。

#### 最终隐藏状态 (hth\_tht​)

当前时间步的最终隐藏状态 hth\_tht​ 通过对前一个隐藏状态 ht−1h\_{t-1}ht−1​ 和候选隐藏状态 h~t\tilde{h}\_th~t​ 进行线性插值计算得出。更新门 ztz\_tzt​ 决定了这种插值的平衡。

ht=(1−zt)⊙ht−1+zt⊙h~th\_t = (1 - z\_t) \odot h\_{t-1} + z\_t \odot \tilde{h}\_tht​=(1−zt​)⊙ht−1​+zt​⊙h~t​

当 ztz\_tzt​ 接近 1 时，候选状态 h~t\tilde{h}\_th~t​ 贡献更多，有效地用新信息更新隐藏状态。当 ztz\_tzt​ 接近 0 时，前一状态 ht−1h\_{t-1}ht−1​ 大部分被保留，允许信息远距离传递。

GRU

clusterₜ

时间步 t

xt

xₜ

rt\_gate

重置门 (rₜ)
σ(...)

xt->rt\_gate

zt\_gate

更新门 (zₜ)
σ(...)

xt->zt\_gate

htₜilde\_calc

候选状态 (h̃ₜ)
tanh(...)

xt->htₜilde\_calc

htₚrev

hₜ₋₁

htₚrev->rt\_gate

htₚrev->zt\_gate

resetₘult

⊙

htₚrev->resetₘult

ztₘult1

⊙

htₚrev->ztₘult1

rt\_gate->resetₘult

ztₘult2

⊙

zt\_gate->ztₘult2

oneₘinus\_zt
1 - zₜ

zt\_gate->oneₘinus\_zt

htₜilde\_calc->ztₘult2

resetₘult->htₜilde\_calc

ht\_calc

最终状态 (hₜ)

htₒut

hₜ

sumₕt

+

ztₘult1->sumₕt

ztₘult2->sumₕt

oneₘinus\_zt->ztₘult1

sumₕt->htₒut

htₒut->htₙextₒut

xtᵢn->xt

htₚrevᵢn->htₚrev

> 门控循环单元 (GRU) 单元在时间步 ttt 内的数据流。xtx\_txt​ 是输入，ht−1h\_{t-1}ht−1​ 是前一个隐藏状态。重置门 (rtr\_trt​) 和更新门 (ztz\_tzt​) 控制着候选状态 (ildeht ilde{h}\_tildeht​) 和最终隐藏状态 (hth\_tht​) 的计算。虚线表示某个值在计算中的使用。

### GRU 与 LSTM 对比

GRU 架构可被视为 LSTM 的一种简化：

1. **门：** GRU 使用两个门（重置、更新），而 LSTM 使用三个（输入、遗忘、输出）。
2. **细胞状态：** GRU 没有 LSTM 中独立的细胞状态 (ctc\_tct​)。隐藏状态 hth\_tht​ 直接包含了控制长期依赖的机制。
3. **参数 (parameter)数量：** 由于门的数量减少且没有细胞状态路径，GRU 通常比具有相同隐藏状态维度的 LSTM 具有更少的训练参数。这使得它们在计算上可能略便宜，并且在较小数据集上可能更不容易过拟合 (overfitting)。

从经验来看，两种架构在所有任务上都没有哪一个持续表现更优。LSTM 和 GRU 之间的选择通常取决于特定问题的实验结果，尽管当计算资源或参数效率是主要考量时，GRU 可能更受青睐。

### 依然存在的局限性

尽管有精密的门控，GRU 仍保留了循环模型的基本特点：顺序计算。信息必须按序列长度一步步传播。这种固有的顺序性限制了训练时的并行化，使得它们在处理非常长的序列时，与 Transformer 等架构相比训练速度明显更慢。此外，尽管它们在捕捉更长距离的依赖关系方面远优于简单的 RNN，但依赖于将过去信息总结到一个固定大小的隐藏状态中，对于复杂依赖关系跨越很长的距离的极长序列，这仍然可能成为瓶颈。这些尚存的挑战促使了基于注意力的机制出现，我们接下来将讨论这些机制。

获取即时帮助、个性化解释和交互式代码示例。

---

### 远距离依赖的挑战

# 远距离依赖的挑战

尽管 LSTM 和 GRU 引入了门控机制，以应对简单 RNN 中存在的严重梯度消失问题，但它们在捕获序列中相距很远元素之间的关系时，仍面临固有的困难。这一局限性直接源于信息处理的序列性质。

可以将 RNN 中的隐藏状态 hth\_tht​ 视为截至时间步 ttt 所见序列的运行概括或记忆。为了影响在晚得多的时间步 NNN 的输出或状态，来自早期时间步 ttt 的信息必须成功地传播经过所有中间步骤 t+1,t+2,…,N−1t+1, t+2, \dots, N-1t+1,t+2,…,N−1。这条序列路径充当了瓶颈。

从数学角度来看，这与时间反向传播 (backpropagation)过程中梯度的传播有关。损失 LLL 对早期隐藏状态 hth\_tht​ 的梯度，取决于代表每一步状态转换的雅可比矩阵的乘积：

∂L∂ht=∂L∂hN(∏k=t+1N∂hk∂hk−1)\frac{\partial L}{\partial h\_t} = \frac{\partial L}{\partial h\_N} \left( \prod\_{k=t+1}^{N} \frac{\partial h\_k}{\partial h\_{k-1}} \right)∂ht​∂L​=∂hN​∂L​(k=t+1∏N​∂hk−1​∂hk​​)

即使 LSTM 和 GRU 被设计成使这些雅可比范数平均更接近 1，但在大量步骤（N−tN-tN−t 很大）上传播信息仍然是困难的。门控提供了对信息流的*控制*，使得网络能够比简单 RNN 更长时间地保留重要信息。然而，这种控制并非完美。

- **信息衰减：** 经过许多步骤后，每一步即使是少量的信息丢失或转换也会累积，使得网络难以精确回忆或使用来自久远过去的信息。状态向量 (vector) hth\_tht​ 具有固定大小，这迫使网络将任意长的历史压缩成固定维度的表示。这种压缩不可避免地导致信息丢失，特别是对于序列中更早的细节。
- **路径长度：** 信息在序列中两点之间传播所需的计算步骤数与其距离成正比。这意味着学习远距离元素之间的依赖关系，需要梯度在相应长的路径上成功传播，即使有门控也仍然困难。LSTM 可能会忘记信息，即使它后来被证明是相关的，或者难以在数千个步骤中传输特定信息而没有退化。

考虑一下这种局限性变得明显的任务：

- **文档分析：** 在处理一篇长文章时，将一个结论性陈述与第一段中引入的前提联系起来，需要在数千字间保持最初的语境。
- **机器翻译：** 翻译句子时，语法一致性或词汇选择取决于源句中相距很远的词语（例如，跨多个从句的主谓一致）。
- **长篇问答：** 回答一个问题，其答案需要综合一份长文档中分散的信息。
- **代码生成：** 在大型代码文件中，保持对早前进行的变量声明或函数定义的语境。

下图说明了循环模型中固有的序列依赖路径。来自 `Input 1` 的信息必须经过每一个中间隐藏状态才能影响 `Output N`。

G

cluster₀

cluster₁

cluster₂

I1

输入 1

I2

输入 2

h1

h₁

I1->h1

Idots
...

h2

h₂

I2->h2

IN

输入 N

hN

h\_N

IN->hN

h0

h₀

h0->h1

 序列路径

h1->h2

 序列路径

O1

输出 1

h1->O1

hdots
...

h2->hdots

 序列路径

O2

输出 2

h2->O2

hdots->hN

 序列路径

ON

输出 N

hN->ON

Odots
...

> RNN 中时间步 `t` 和 `N` 之间信息流的序列路径长度与 `N-t` 成正比。

尽管 LSTM 和 GRU 是显著的改进，但在处理极长距离依赖方面的持续困难直接促使了能够创建相距遥远的序列元素之间更短路径的架构的发展。Transformer 模型，主要是通过其自注意力 (self-attention)机制 (attention mechanism)，提供了一种直接建模序列中任意两个位置之间关系的方法，无论它们距离多远，从而克服了循环处理的这一根本局限。我们将在下一章详细考察这种机制。

获取即时帮助、个性化解释和交互式代码示例。

---

### 循环模型中的并行化限制

# 循环模型中的并行化限制

循环模型（如RNN、LSTM和GRU）逐步处理序列。这种按序处理的特性，尽管对时间序列或语言建模来说很直观，但却对计算并行化带来了基本限制，尤其是在训练阶段。

### 序列依赖性瓶颈

任何RNN中的主要操作是计算时间步 ttt 的隐藏状态 hth\_tht​，其依据是该时间步的输入 xtx\_txt​ 和*前一个*时间步的隐藏状态 ht−1h\_{t-1}ht−1​。这种关系通常表示为：

ht=f(ht−1,xt)h\_t = f(h\_{t-1}, x\_t)ht​=f(ht−1​,xt​)

其中 fff 代表RNN单元执行的变换（可以是简单的RNN更新、LSTM单元或GRU单元）。这个公式清楚地显示了固有的序列依赖性：计算 hth\_tht​ 需要 ht−1h\_{t-1}ht−1​ 的结果，而 ht−1h\_{t-1}ht−1​ 又需要 ht−2h\_{t-2}ht−2​，依此类推，直到最初的状态 h0h\_0h0​。

G

x₁

x₁

f₁

f

x₁->f₁

h₀

h₀

h₀->f₁

h₁

h₁

f₁->h₁

f₂

f

h₁->f₂

依赖关系

x₂

x₂

x₂->f₂

h₂

h₂

f₂->h₂

...
...

h₂->...

xₜ

xₜ

fₜ

f

xₜ->fₜ

hₜ

hₜ

fₜ->hₜ

...->xₜ

...->fₜ

hₜ₋₁

> 隐藏状态 hth\_tht​ 的计算按序依赖于前一个状态 ht−1h\_{t-1}ht−1​。这种依赖关系阻止了在单个序列中跨时间步的并行计算。

这种时间上的依赖性造成了瓶颈。现代硬件加速器，例如GPU和TPU，擅长并行执行大规模矩阵运算。然而，在RNN处理的单个序列中，时间步 ttt 的计算必须等到时间步 t−1t-1t−1 的计算完成后才能开始。单个时间步*内部*的操作（例如LSTM门内的矩阵乘法）可以并行化，但*跨*时间步的计算仍然是按序进行的。

### 对训练效率的影响

这种限制明显制约了训练期间可获得的加速效果。尽管可以在小批量数据（数据并行）中对*不同序列*进行RNN训练的并行化，但每个*单独序列*的处理仍受其长度限制，这是因为前向传播是按序进行的。

此外，网络中的反向传播 (backpropagation)（称为时间反向传播，BPTT）也受到同样的按序处理限制。为了计算损失函数 (loss function)对时间步 ttt 所用参数 (parameter)的梯度，BPTT需要逐步地将梯度反向传播通过序列。时间步 ttt 的梯度计算依赖于时间步 t+1t+1t+1 的梯度计算。这反映了前向传播的依赖关系，阻止了梯度计算在时间上的并行化。

实际结果是，在非常长的序列上训练RNN变得计算成本高且缓慢。增加序列长度会导致该序列的计算时间大致呈线性增长，并且无法充分发挥并行硬件的性能意味着GPU在部分计算过程中可能未被充分利用。

这种限制是促使Transformer等架构出现的原因之一，其目的是捕获序列依赖性而不依赖于按序的循环，从而在训练期间实现更大的并行化。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 2 Attention Mechanism Core Concepts

### 动机：克服固定长度上下文向量的不足

# 动机：克服固定长度上下文向量的不足

循环神经网络 (neural network)（RNN）及其变体，如LSTM和GRU，通过维护一个隐藏状态来处理序列。尽管这些模型对许多序列任务有效，但一个主要限制源于其核心机制：无论输入序列长度如何，都将其所有信息压缩成一个固定大小的隐藏状态向量 (vector)（或在编码-解码器设置中的上下文 (context)向量）。

以标准序列到序列（seq2seq）模型为例，它常用于机器翻译等任务。编码器逐步处理输入序列，并在每一步更新其隐藏状态。编码器的最终隐藏状态，通常称为上下文向量，旨在总结整个输入序列。这个单一向量随后被传递给解码器，解码器将其用作初始状态来生成输出序列。

G

clusterₑncoder

编码器

cluster\_decoder

解码器

x1

输入 1

x2

输入 2

x1->x2

处理输入

x3

输入...

x2->x3

处理输入

C

固定上下文向量
(最终隐藏状态)

x3->C

处理输入

y1

输出 1

C->y1

生成输出

y2

输出 2

y1->y2

生成输出

y3

输出...

y2->y3

生成输出

> 传统编码器-解码器架构的简化视图。编码器输出一个固定大小的上下文向量`C`，它表示整个输入，并成为解码器获取输入信息的唯一来源。

这个固定大小的上下文向量代表着一个固有的信息瓶颈。想象一下，尝试将冗长的段落或文档总结成一个简短的句子；细节必然会丢失。同样，随着序列长度的增加，迫使模型将长输入序列（例如，翻译中的复杂句子）的所有细节编码到一个向量中，变得越来越困难。解码器的性能根本上受到这个单一总结向量的质量和完整性的制约。在编码器的序列处理过程中，来自输入序列早期部分的信息可能会被后续输入“覆盖”或稀释。

这个瓶颈使得解码器在生成输出的不同部分时，难以访问输入中特定且相关的信息片段。例如，在翻译句子时，特定输出词的选择可能严重依赖于*输入*句子开头附近的一个特定词语或短语。仅仅依赖最终压缩的上下文向量，使得访问这种特定且远距离的信息变得不可靠。

为了克服这一限制，我们需要一种机制，它允许模型（特别是解码器）在输出生成过程的每一步“回看”编码器隐藏状态的整个序列（或输入的表示）。模型不应仅仅依赖一个单一的静态总结，而应能够根据它当时试图预测的内容，动态地为输入序列的不同部分分配不同程度的重要性。

这就是**注意力机制 (attention mechanism)**背后的主要原因。它提供了一种方法，可以创建针对每个输出步骤专门定制的动态的、依赖于上下文的输入序列总结。与其将所有内容压缩到一个固定向量中，注意力机制允许模型选择性地关注最相关的输入元素，从而有效地绕过固定上下文瓶颈。

接下来各部分将详细说明这种注意力机制的实现方式，从查询（Queries）、键（Keys）和值（Values）的基本抽象开始，最终形成在Transformer架构中被普遍使用的缩放点积注意力公式。

获取即时帮助、个性化解释和交互式代码示例。

---

### 通用框架：查询、键、值表示

# 通用框架：查询、键、值表示

如前所述，传统的序列到序列模型常将整个输入序列压缩成一个固定大小的上下文 (context)向量 (vector)。这种方法在处理较长序列时可能难以保留信息，从而造成信息瓶颈。注意力机制 (attention mechanism)提供了一种更具适应性的替代方案，使模型能够回顾整个输入序列，并有选择地侧重于与生成当前输出最相关部分。

为使这种有选择的关注形式化，注意力机制采用了一种基于三个要素的表示：查询、键和值。设想你在数字图书馆中研究一个主题。

- 你的**查询** (QQQ) 是你目前感兴趣的特定问题或主题。
- 图书馆中的每份文档都附有**键** (KKK)，它们类似于简洁的摘要或关键词列表，表示文档内容与潜在查询的关联性。
- 每份文档还包含其实际内容，即**值** (VVV)，这是如果文档相关你希望检索的详细信息。

注意力机制的工作方式与此类似：对于给定的查询，它会将查询与所有可用键进行比较，以确定它们的匹配程度。此匹配过程会生成一组分数，常被称为注意力权重 (weight)。这些权重表明每个键（及其对应的值）与查询的关联性。最后，该机制通过汇总值来得出输出，根据其计算出的注意力分数对每个值进行加权。与查询高度匹配的键所对应的值对最终输出的贡献更大。

在神经网络 (neural network)中，查询、键和值表示为从模型内部表示（嵌入 (embedding)或隐藏状态）派生的向量：

- **查询 (QQQ):** 一个向量，代表当前需要从输入序列中获取信息的上下文或元素。例如，在翻译句子时，查询可能代表解码器在决定下一个要生成的词时的当前状态。它会问：“输入句子的哪些部分现在最相关？”
- **键 (KKK):** 一组向量，输入序列中的每个元素对应一个。每个键向量都设计用于与查询向量进行比较。它本质上表示其对应的输入元素提供了何种信息。
- **值 (VVV):** 一组向量，输入序列中的每个元素也对应一个，常与键配对。值向量包含其对应的输入元素的实际内容或表示，如果该元素被查询-键比较判定为相关，则应将其汇总。

G

clusterᵢnput

输入序列元素

K1

键 1

V1

值 1

Compare

比较
(例如，点积)

K1->Compare

键

Combine

组合
(加权和)

V1->Combine

值

K2

键 2

V2

值 2

Kn

键 n

Vn

值 n

Q

查询

Q->Compare

Weights

计算
权重
(例如，Softmax)

Compare->Weights

Weights->Combine

注意力权重

Output

输出
(上下文向量)

Combine->Output

> 使用查询、键和值表示的注意力流。查询与所有键进行比较以得出权重，然后这些权重用于形成相应值的加权和。

核心观点是，查询与键互动以了解*何处*进行侧重，而由此产生的注意力分数则决定了每个值对最终输出表示的*贡献程度*。此输出是一个具备上下文感知能力的向量，它概括了与特定查询相关的输入序列元素。

在注意力机制中，查询、键和值向量是核心组成部分，它们的维度通常分别表示为 dqd\_qdq​、dkd\_kdk​ 和 dvd\_vdv​。查询和键之间的兼容性常使用它们的向量表示来计算，通常通过点积，这要求 dq=dkd\_q = d\_kdq​=dk​。维度 dkd\_kdk​ 在注意力分数如何计算和缩放方面发挥着重要作用，我们很快会看到。值向量的维度 dvd\_vdv​ 决定了在任何最终转换之前的输出上下文向量的维度。

值得注意的是，这些 Q、K 和 V 向量的来源界定了注意力的类型。在变压器 (Transformer)架构中居于核心地位的*自注意力 (self-attention)*中，Q、K 和 V 都源自*同一*序列。这使得单个序列内的不同位置可以相互关注。在编码器-解码器架构中常见的*交叉注意力*中，查询可能来自解码器，而键和值则来自编码器的输出，从而使解码器能够关注输入序列的相关部分。目前，我们侧重于 QKV 的通用表示本身。

这种查询-键-值框架提供了一种适应性强且有效的方法来模拟依赖关系，无论它们在输入序列中的距离如何，直接解决了固定上下文向量的局限性。下一步是审视用于实现这种比较和加权过程的特定数学运算，从缩放点积注意力机制开始。

获取即时帮助、个性化解释和交互式代码示例。

---

### 点积注意力机制的数学形式

# 点积注意力机制的数学形式

点积注意力是一种常用的注意力机制 (attention mechanism)，它以查询 (QQQ)、键 (KKK) 和值 (VVV) 为核心。其主要计算步骤是：注意力会计算一系列分数，这些分数反映了每个查询与每个键之间的关联度或匹配度。这些分数随后用于生成值的加权和，使得模型能够根据查询-键关系，侧重于值所携带的最重要数据。

“点积”这个名称直接来源于这些匹配分数的计算方式。对于给定的查询向量 (vector) qqq 和一组向量 K={k1,k2,...,km}K = \{k\_1, k\_2, ..., k\_m\}K={k1​,k2​,...,km​}， qqq 与特定向量 kjk\_jkj​ 之间的分数就是它们的点积：

分数(q,kj)=q⋅kj=qTkj分数(q, k\_j) = q \cdot k\_j = q^T k\_j分数(q,kj​)=q⋅kj​=qTkj​

通俗地说，点积体现了两个向量之间的对齐 (alignment)程度或相似性。如果查询向量 qqq 与键向量 kjk\_jkj​ 高度相似（指向相似的方向），它们的点积会很大。反之，如果它们不相似或正交，点积会很小或为零。

在实际应用中，我们不会逐一计算这些分数。深度学习 (deep learning)框架善于并行处理，尤其是矩阵乘法。我们通常使用查询、键和值的矩阵进行操作：

- QQQ: 一个包含 nnn 个查询向量的矩阵，每个向量的维度为 dkd\_kdk​。形状：[n×dk][n \times d\_k][n×dk​]。
- KKK: 一个包含 mmm 个向量的矩阵，每个向量的维度也为 dkd\_kdk​。形状：[m×dk][m \times d\_k][m×dk​]。
- VVV: 一个包含 mmm 个值向量的矩阵，每个向量的维度为 dvd\_vdv​。形状：[m×dv][m \times d\_v][m×dv​]。

请注意，查询和键向量必须具有相同的维度 (dkd\_kdk​) 才能进行点积计算。值向量可以具有不同的维度 (dvd\_vdv​)。

为了同时计算所有查询分数，我们对查询矩阵 QQQ 和矩阵 KTK^TKT 的转置进行矩阵乘法：

分数=QKT\text{分数} = Q K^T分数=QKT

我们来看看这些矩阵的尺寸：将 QQQ ([n×dk][n \times d\_k][n×dk​]) 乘以 KTK^TKT ([dk×m][d\_k \times m][dk​×m]) 会得到一个形状为 [n×m][n \times m][n×m] 的分数矩阵。这个分数矩阵中的每个元素 (i,j)(i, j)(i,j) 代表了第 iii 个查询 (QiQ\_iQi​) 和第 jjj 个键 (KjK\_jKj​) 之间的点积，这体现了它们的匹配程度。

G

cluster\_Q

查询 (Q)

cluster\_KT

键的转置 (Kᵀ)

cluster\_Scores

分数

Qₘatrix

q₁

q₂

...

qₙ

Times

X

Qₘatrix->Times

Q\_dim
[n x dk]

KTₘatrix

k₁ᵀ

k₂ᵀ

...

kₘᵀ

KTₘatrix->Times

KT\_dim
[dk x m]

Scoresₘatrix

q₁·k₁

q₁·k₂

...

q₁·kₘ

q₂·k₁

q₂·k₂

...

q₂·kₘ

...

...

...

...

qₙ·k₁

qₙ·k₂

...

qₙ·kₘ

Scores\_dim
[n x m]

Times->Scoresₘatrix

> 矩阵乘法 QKTQK^TQKT 计算了查询向量（Q 的行）与键向量（KTK^TKT 的列）之间所有的成对点积。

这些原始分数 (QKTQK^TQKT) 代表了点积注意力机制中主要的匹配度指标。然而，它们尚未准备好用作值的权重 (weight)。点积的取值区间可能非常大，这可能在训练期间引出问题，尤其是在梯度方面。另外，我们需要将这些分数转换为一个概率分布，使得每个查询的分数之和为1，这反映了该查询应如何分配对每个值的关注度。

接下来的步骤，即对这些分数进行缩放并应用softmax函数，将解决这些情况，并将原始点积分数转化为实际可用的注意力权重。我们将在后续章节中考察这些步骤。

获取即时帮助、个性化解释和交互式代码示例。

---

### 缩放点积注意力

# 缩放点积注意力

缩放点积注意力是对稳定且高效训练的基本改进。它使用查询 (QQQ) 和键 (KKK) 向量 (vector)之间的点积来衡量对齐 (alignment)。

核心的点积注意力机制 (attention mechanism)计算 QKTQK^TQKT。思考所涉及的维度。如果查询和键向量的维度是 dkd\_kdk​，那么单个查询 qqq 与单个键 kkk 之间的点积是 ∑i=1dkqiki\sum\_{i=1}^{d\_k} q\_i k\_i∑i=1dk​​qi​ki​。如果我们假设 qqq 和 kkk 的分量是均值为零、方差为一的独立随机变量，那么点积的期望值为0，但其方差是 dkd\_kdk​。

实际中，特别是在 dkd\_kdk​ 值很大时（例如，64、128 或更高，这在 Transformer 中很常见），这些点积的数值会变得非常大。为什么这有问题？这些点积是 softmax 函数的输入，该函数计算注意力权重 (weight)。

softmax(z)i=ezi∑jezj\text{softmax}(z)\_i = \frac{e^{z\_i}}{\sum\_j e^{z\_j}}softmax(z)i​=∑j​ezj​ezi​​

softmax 函数表现出饱和行为。如果其输入（点积分数）的数值很大（无论是大的正值还是大的负值），函数会进入梯度极其接近零的区域。想象一下 softmax 函数的导数；随着输入值远离零，它会趋近于零。

−10−5051000.20.40.60.81Softmax 输出 (单个元素)梯度 (近似)

> softmax 函数的梯度在输入值接近零时最大，并且对于大的正输入或负输入，梯度迅速减小。大的点积会将计算推到这些低梯度区域。

小梯度会急剧减慢甚至停止反向传播 (backpropagation)期间的学习过程，因为权重更新变得极其微小。这使得训练深度模型非常困难。

为了抵消这种影响，论文《Attention Is All You Need》引入了一个 1/dk1/\sqrt{d\_k}1/dk​​ 的缩放因子。缩放点积注意力的完整公式是：

Attention(Q,K,V)=softmax(QKTdk)V\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)VAttention(Q,K,V)=softmax(dk​​QKT​)V

这里，QQQ、KKK 和 VVV 是矩阵，分别代表打包的查询、键和值集合。dkd\_kdk​ 是键（和查询）向量的维度。

通过将点积 QKTQK^TQKT 除以 dk\sqrt{d\_k}dk​​，我们有效地调整了它们的数值大小，使 softmax 函数的输入更接近梯度更为显著的区域。这种缩放确保 softmax 输入的方差保持大致不变，无论 dkd\_kdk​ 如何选择，这对更稳定的训练过程有很大帮助。它防止注意力权重仅仅因为表示的维度，在训练初期就变得过于集中（集中在单个输入上）或过于分散。

这种缩放不仅仅是一个小的实现细节；它是一个重要组成部分，通过缓解直接源于核心注意力计算的潜在梯度问题，使得深度 Transformer 模型能够成功训练。它使得模型能够有效地学习，即使在使用高维向量表示查询和键时。

获取即时帮助、个性化解释和交互式代码示例。

---

### 注意力权重的Softmax函数

# 注意力权重的Softmax函数

查询向量 (vector) (QQQ) 和键向量 (KKK) 之间的缩放点积分数计算会产生一个原始对齐 (alignment)分数矩阵。此计算结果为 QKTdk\frac{QK^T}{\sqrt{d\_k}}dk​​QKT​。尽管这些分数反映了查询和键向量之间的匹配程度，但它们未经归一化 (normalization)处理，值域可以是任意范围，这使得它们难以直接被视为贡献权重 (weight)。

为了将这些原始分数转换为一组可用的、表示注意力分布的权重，我们对分数矩阵的每一行独立应用softmax函数。对于特定的查询 qiq\_iqi​ (对应 QQQ 的第 iii 行)，它与键 kjk\_jkj​ (对应 KTK^TKT 的第 jjj 列) 对齐的原始分数记作 sij=qikjTdks\_{ij} = \frac{q\_i k\_j^T}{\sqrt{d\_k}}sij​=dk​​qi​kjT​​。softmax函数将查询 qiq\_iqi​ 与所有 NNN 个键的分数向量 si=[si1,si2,...,siN]s\_i = [s\_{i1}, s\_{i2}, ..., s\_{iN}]si​=[si1​,si2​,...,siN​] 转换为注意力权重向量 αi=[αi1,αi2,...,αiN]\alpha\_i = [\alpha\_{i1}, \alpha\_{i2}, ..., \alpha\_{iN}]αi​=[αi1​,αi2​,...,αiN​]，其中每个权重 αij\alpha\_{ij}αij​ 的计算方式如下：

αij=softmax(sij)=exp⁡(sij)∑l=1Nexp⁡(sil)\alpha\_{ij} = \text{softmax}(s\_{ij}) = \frac{\exp(s\_{ij})}{\sum\_{l=1}^{N} \exp(s\_{il})}αij​=softmax(sij​)=∑l=1N​exp(sil​)exp(sij​)​

这里，NNN 代表键/值对的序列长度。

### 特性和解释

应用softmax函数会为得到的注意力权重 (weight) αij\alpha\_{ij}αij​ 带来几个重要特性：

1. **归一化 (normalization)：** 分母 ∑l=1Nexp⁡(sil)\sum\_{l=1}^{N} \exp(s\_{il})∑l=1N​exp(sil​) 保证了给定查询 qiq\_iqi​ 在所有键上的注意力权重总和为1。即 ∑j=1Nαij=1\sum\_{j=1}^{N} \alpha\_{ij} = 1∑j=1N​αij​=1。这使得我们可以将这些权重理解为概率分布。
2. **非负性：** 由于指数函数 exp⁡(x)\exp(x)exp(x) 对于任何实数输入 xxx 总是正数，每个单独的注意力权重 αij\alpha\_{ij}αij​ 都保证是正数。
3. **概率解释：** 对于查询 qiq\_iqi​ 的权重集合 {αi1,αi2,...,αiN}\{\alpha\_{i1}, \alpha\_{i2}, ..., \alpha\_{iN}\}{αi1​,αi2​,...,αiN​} 代表了在输入序列上的注意力概率分布。权重 αij\alpha\_{ij}αij​ 表示模型在计算第 iii 个位置的输出表示时，分配给第 jjj 个输入元素 (由其值向量 (vector) vjv\_jvj​ 代表) 的注意力比例。
4. **突出重要性：** 指数函数本质上会放大更大的分数，使其比小分数更为突出。如果分数 siks\_{ik}sik​ 在行 sis\_isi​ 中明显大于其他分数，其对应的权重 αik\alpha\_{ik}αik​ 将接近1，而其他权重将接近0。这种机制使得模型能够有效地关注通过点积评分确定的最相关输入元素。

### 在计算最终输出中的作用

这些计算出的注意力权重 (weight) αij\alpha\_{ij}αij​ 是用于计算值向量 (vector) (VVV) 加权和的系数。注意力机制 (attention mechanism)对第 iii 个查询的输出，是通过将注意力权重分布 αi\alpha\_iαi​ 与值矩阵 VVV 相乘而获得的：

输出i=∑j=1Nαijvj\text{输出}\_i = \sum\_{j=1}^{N} \alpha\_{ij} v\_j输出i​=j=1∑N​αij​vj​

在矩阵表示中，这直接对应于缩放点积注意力公式的最后一步：

注意力(Q,K,V)=AV=softmax(QKTdk)V\text{注意力}(Q, K, V) = A V = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V注意力(Q,K,V)=AV=softmax(dk​​QKT​)V

其中 AAA 是注意力权重矩阵，其元素为 Aij=αijA\_{ij} = \alpha\_{ij}Aij​=αij​。因此，softmax函数在将原始相似性分数转换为归一化 (normalization)分布方面起着重要作用，该分布决定了如何聚合来自输入序列不同部分 (由值向量表示) 的信息以形成输出。

### 可视化示例

考虑一个简化的例子，单个查询有4个原始对齐 (alignment)分数：s=[1.0,0.5,2.5,−0.1]s = [1.0, 0.5, 2.5, -0.1]s=[1.0,0.5,2.5,−0.1]。应用softmax函数会转换这些分数：

- exp⁡(1.0)≈2.718\exp(1.0) \approx 2.718exp(1.0)≈2.718
- exp⁡(0.5)≈1.649\exp(0.5) \approx 1.649exp(0.5)≈1.649
- exp⁡(2.5)≈12.182\exp(2.5) \approx 12.182exp(2.5)≈12.182
- exp⁡(−0.1)≈0.905\exp(-0.1) \approx 0.905exp(−0.1)≈0.905

指数之和为 2.718+1.649+12.182+0.905=17.4542.718 + 1.649 + 12.182 + 0.905 = 17.4542.718+1.649+12.182+0.905=17.454。

得到的softmax权重 (weight)是：

- α1=2.718/17.454≈0.156\alpha\_1 = 2.718 / 17.454 \approx 0.156α1​=2.718/17.454≈0.156
- α2=1.649/17.454≈0.094\alpha\_2 = 1.649 / 17.454 \approx 0.094α2​=1.649/17.454≈0.094
- α3=12.182/17.454≈0.698\alpha\_3 = 12.182 / 17.454 \approx 0.698α3​=12.182/17.454≈0.698
- α4=0.905/17.454≈0.052\alpha\_4 = 0.905 / 17.454 \approx 0.052α4​=0.905/17.454≈0.052

请注意，最高的原始分数 (2.5) 对应着主导的注意力权重 (0.698)，在此示例中，有效地将注意力机制 (attention mechanism)集中在第三个输入元素上。这些权重之和约为1 (0.156+0.094+0.698+0.052=1.0000.156 + 0.094 + 0.698 + 0.052 = 1.0000.156+0.094+0.698+0.052=1.000)。

位置 1位置 2位置 3位置 4012原始分数Softmax权重

> 单个查询的原始对齐分数与应用softmax函数后得到的注意力权重的对比。Softmax将分数转换为概率分布，突出显示最相关的位置。

总之，softmax函数在注意力机制中是一个重要的归一化 (normalization)步骤。它将原始对齐分数转换为概率分布，使得模型能够基于查询-键相似性，选择性地加权并组合来自值向量 (vector)的信息，为Transformer如何处理序列信息奠定基础。

获取即时帮助、个性化解释和交互式代码示例。

---

### 计算考量与矩阵运算

# 计算考量与矩阵运算

通过矩阵运算实现的高效性赋予了注意力机制 (attention mechanism)实际效果，尽管其定义利用查询、键和值来进行概念理解。这种方法使模型能够同时计算序列中所有位置的注意力分数和上下文 (context)向量 (vector)，使其非常契合现代并行硬件如GPU和TPU。

我们来回顾一下缩放点积注意力公式：
注意力(Q,K,V)=softmax(QKTdk)V\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V注意力(Q,K,V)=softmax(dk​​QKT​)V

与一次只计算一个查询的注意力不同，我们同时处理整个序列。假设我们有一个长度为 nnn 的输入序列。每个位置的查询、键和值向量堆叠起来形成矩阵：

- QQQ: 一个维度为 (n×dk)(n \times d\_k)(n×dk​) 的矩阵，其中每行 iii 是位置 iii 的查询向量 qiq\_iqi​。
- KKK: 一个维度为 (n×dk)(n \times d\_k)(n×dk​) 的矩阵，其中每行 jjj 是位置 jjj 的键向量 kjk\_jkj​。
- VVV: 一个维度为 (n×dv)(n \times d\_v)(n×dv​) 的矩阵，其中每行 jjj 是位置 jjj 的值向量 vjv\_jvj​。值得一提的是，值的维度 dvd\_vdv​ 有时可能与 dkd\_kdk​ 不同。

现在，我们使用这些矩阵逐步分析计算过程：

### 1. 计算点积分数：QKTQK^TQKT

注意力计算的主要部分是确定每个查询应关注每个元素的程度。这通过计算每个查询 qiq\_iqi​ 与每个元素 kjk\_jkj​ 之间的点积来完成。矩阵乘法 QKTQK^TQKT 并行执行所有这些点积：

分数=QKT\text{分数} = Q K^T分数=QKT

结果得到的 `分数` 矩阵的维度为 (n×dk)×(dk×n)=(n×n)(n \times d\_k) \times (d\_k \times n) = (n \times n)(n×dk​)×(dk​×n)=(n×n)。此矩阵中的每个元素 (i,j)(i, j)(i,j) 代表位置 iii 的查询与位置 jjj 的键之间的原始对齐 (alignment)分数。值越高意味着相关性越强。

### 2. 应用缩放：/dk/\sqrt{d\_k}/dk​​

如前所述，缩放可以避免点积变得过大，从而可能将softmax函数推向梯度很小的区域。这种缩放按元素应用于 `分数` 矩阵：

缩放分数=分数dk\text{缩放分数} = \frac{\text{分数}}{\sqrt{d\_k}}缩放分数=dk​​分数​

维度仍为 (n×n)(n \times n)(n×n)。

### 3. 使用Softmax函数归一化 (normalization)分数：softmax(⋅)\text{softmax}(\cdot)softmax(⋅)

为了将缩放分数转换为概率（注意力权重 (weight)），softmax函数独立地应用于 `缩放分数` 矩阵的每一行。对于给定行 iii，softmax确保查询 iii 对所有键 j=1...nj=1...nj=1...n 分配的注意力权重总和为1。

W=softmax(缩放分数)按行W = \text{softmax}(\text{缩放分数})\_{\text{按行}}W=softmax(缩放分数)按行​

结果得到的注意力权重矩阵 WWW 的维度也为 (n×n)(n \times n)(n×n)。WijW\_{ij}Wij​ 代表位置 iii 的查询对位置 jjj 的键（及关联值）关注的比例。

### 4. 计算值的加权和：WVW VWV

最后，注意力权重 (weight) WWW 用于计算值向量 (vector) VVV 的加权和。这通过矩阵乘法完成：

输出=WV\text{输出} = W V输出=WV

输出矩阵的维度为 (n×n)×(n×dv)=(n×dv)(n \times n) \times (n \times d\_v) = (n \times d\_v)(n×n)×(n×dv​)=(n×dv​)。`输出` 矩阵的每行 iii 是位置 iii 对应的上下文 (context)向量。它是序列中所有值向量的组合，根据为查询 iii 计算的注意力分布进行加权。

### 并行化带来的效率提升

这种基于矩阵的表述的精妙之处在于其可并行性。每个步骤，主要涉及矩阵乘法，都可以在为这类操作设计的硬件上高效执行。与按顺序处理标记 (token)（t=1,2,...,nt=1, 2, ..., nt=1,2,...,n）的循环模型不同，注意力机制 (attention mechanism)能够基本并行地计算所有位置对 (i,j)(i, j)(i,j) 之间的交互。这解决了限制RNN和LSTM的顺序瓶颈，从而实现更快的训练和对更长序列的处理（在适用情况下）。

G

clusterᵢnputs

输入矩阵

cluster\_computation

注意力计算

clusterᵢntermediates

中间结果

clusterₒutput

最终输出

Q

Q
(n x dk)

MatMul1

矩阵乘法
(QKᵀ)

Q->MatMul1

查询

K

K
(n x dk)

K->MatMul1

键 (转置)

V

V
(n x dv)

MatMul2

矩阵乘法
(WV)

V->MatMul2

值

Scores

分数
(n x n)

MatMul1->Scores

Scale

缩放
(/ sqrt(dk))

ScaledScores

缩放分数
(n x n)

Scale->ScaledScores

Softmax

Softmax
(按行)

Weights

权重 (W)
(n x n)

Softmax->Weights

Output

输出
(n x dv)

MatMul2->Output

Scores->Scale

ScaledScores->Softmax

Weights->MatMul2

注意力权重

> 流程图描绘了使用矩阵运算计算缩放点积注意力的过程。维度表示为（行 x 列），其中 n 是序列长度，dk 是键的维度，dv 是值的维度。

了解这种矩阵表述是基础。它不仅说明了注意力如何高效计算，而且作为在深度学习 (deep learning)框架中实现注意力层的基础，这些框架高度依赖优化的矩阵运算。下一节将通过动手实践来实现缩放点积注意力机制，从而加深您的理解。

获取即时帮助、个性化解释和交互式代码示例。

---

### 实践：实现缩放点积注意力

# 实践：实现缩放点积注意力

提供了一个使用 PyTorch 进行的缩放点积注意力实际操作实现。此实践应用侧重于注意力机制 (attention mechanism)的主要计算。在构建像多头注意力 (multi-head attention)这样更复杂的结构之前，理解此实现是必需的。

回顾一下公式：
注意力(Q,K,V)=softmax(QKTdk)V\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V注意力(Q,K,V)=softmax(dk​​QKT​)V

我们将逐步实现它。

### 设置

首先，确保您已安装 PyTorch。我们将需要基本的 `torch` 库和 `math` 模块来进行平方根运算。

```python
import torch
import torch.nn.functional as F
import math
import plotly.graph_objects as go
import numpy as np
```

### 实现注意力函数

让我们定义一个函数 `scaled_dot_product_attention`，它接受查询 (QQQ)、键 (KKK)、值 (VVV) 以及一个可选的掩码作为输入。

为简化本初始示例，我们假设 QQQ、KKK 和 VVV 的形状类似于 `[batch_size, sequence_length, dimension]`。在完整的 Transformer 中，这些张量通常会有一个额外的“头”维度，我们将在之后处理。维度 dkd\_kdk​ 对应于键张量的最后一个维度。

```python
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    计算缩放点积注意力。

    参数：
        query: 查询张量；形状 (batch_size, seq_len_q, d_k)
        key: 键张量；形状 (batch_size, seq_len_k, d_k)
        value: 值张量；形状 (batch_size, seq_len_k, d_v)
               注意：key 和 value 的 seq_len_k 必须匹配。
               d_v（值维度）可以与 d_k（查询维度）不同。
        mask: 可选掩码张量；形状可广播到 (batch_size, seq_len_q, seq_len_k)。
              值为 True 或 1 的位置表示保留值，False 或 0 表示屏蔽掉。

    返回：
        output: 注意力输出张量；形状 (batch_size, seq_len_q, d_v)
        attention_weights: 注意力权重；形状 (batch_size, seq_len_q, seq_len_k)
    """

    d_k = key.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1))

    scores = scores / math.sqrt(d_k)

    if mask is not None:

        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scores, dim=-1)

    attention_weights = torch.nan_to_num(attention_weights)

    output = torch.matmul(attention_weights, value)

    return output, attention_weights
```

### 示例用法

让我们创建一些示例张量并测试我们的函数。为了演示，我们将使用较小的批量大小、序列长度和维度。

```python

batch_size = 1
seq_len_q = 3
seq_len_k = 4
d_k = 8
d_v = 16

query = torch.randn(batch_size, seq_len_q, d_k)
key = torch.randn(batch_size, seq_len_k, d_k)
value = torch.randn(batch_size, seq_len_k, d_v)

output, attention_weights = scaled_dot_product_attention(query, key, value)

print("--- 无掩码输出 ---")
print("输出形状:", output.shape)
print("注意力权重形状:", attention_weights.shape)

print("注意力权重总和（第一个查询）:", attention_weights[0, 0, :].sum())

mask = torch.ones(batch_size, 1, seq_len_k, dtype=torch.bool)
mask[:, :, -1] = 0

print("\n掩码形状:", mask.shape)
print("掩码内容:\n", mask)

output_masked, attention_weights_masked = scaled_dot_product_attention(query, key, value, mask=mask)

print("\n--- 有掩码输出 ---")
print("输出形状:", output_masked.shape)
print("注意力权重形状:", attention_weights_masked.shape)
print("屏蔽后的注意力权重（第一个查询）:\n", attention_weights_masked[0, 0, :])

print("注意力权重总和（第一个查询，已屏蔽）:", attention_weights_masked[0, 0, :].sum())
```

您应该观察到输出形状符合我们的预期。当应用掩码时，对应于被屏蔽位置（本例中为最后一个）的注意力权重 (weight)变为零，并且剩余的权重通过 softmax 重新归一化 (normalization)，使其总和为 1。

### 可视化注意力权重 (weight)

可视化注意力权重矩阵 (softmax(QKTdk)\text{softmax}(\frac{QK^T}{\sqrt{d\_k}})softmax(dk​​QKT​)) 可以帮助我们理解模型如何关联序列的不同部分。热图常用于此目的。让我们可视化来自我们无掩码示例的权重。

```python

weights_np = attention_weights[0].detach().numpy()

fig_data = go.Heatmap(
    z=weights_np,
    x=[f'位置 {i}' for i in range(seq_len_k)],
    y=[f'查询位置 {i}' for i in range(seq_len_q)],
    colorscale='Blues',
    colorbar=dict(title='注意力权重')
)

fig_layout = go.Layout(
    title='注意力权重（查询 vs 键）',
    xaxis_title="序列位置",
    yaxis_title="查询序列位置",
    yaxis_autorange='reversed',
    width=500, height=400, margin=dict(l=50, r=50, b=100, t=100, pad=4)
)

fig = go.Figure(data=[fig_data], layout=fig_layout)
plotly_json = fig.to_json()
```

位置 0位置 1位置 2位置 3查询位置 2查询位置 1查询位置 00.20.250.3注意力权重注意力权重（查询 vs 键）序列位置查询序列位置

> 热图显示了每个查询位置（行）对每个位置（列）的注意力权重。值越高（颜色越深），表示注意力越强。每行总和为 1。

这种可视化显示了对于每个查询位置（行），在计算其输出时，它给予每个位置（列）多少“注意力”或权重。在具有有意义数据的实际应用中，此矩阵中的模式可以显示模型学到的句法或语义关系。例如，一个动词可能会强烈关注其主语和宾语。

这种缩放点积注意力的实现构成了 Transformer 中每个注意力头内的核心计算单元。在下一章中，我们将在此之上构建，以实现多头注意力 (multi-head attention)，从而使模型能够共同关注来自不同表示子空间的信息。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 3 Multi Head Self Attention

### 自注意力：查询、键、值源于同一来源

# 自注意力：查询、键、值源于同一来源

上一章我们介绍了通用注意力机制 (attention mechanism)，它是一种将查询和一组键值对映射到输出的强大技术。回顾其核心思想：输出是值的加权和，其中分配给每个值的权重 (weight)是通过查询与对应键之间的兼容性函数（如缩放点积）计算得出的。

现在，我们着重介绍这种机制的一种专门应用，称为**自注意力 (self-attention)**。自注意力定义上的特点是查询、键和值都源于*同一*源序列。自注意力并非关联两个不同序列（如传统编解码器注意力中那样），而是允许一个序列内部的不同位置之间产生关联。此过程使得模型在编码特定词时，能够衡量序列中其他词的重要性，直接捕获序列内部的依赖关系。

考虑用矩阵 XXX 表示的输入序列，它的每行对应一个词元 (token)嵌入 (embedding)（可能与位置信息结合，我们将在第四章讨论）。为计算自注意力，我们首先将输入 XXX 投影到三种不同的表示形式：查询 (QQQ)、键 (KKK) 和值 (VVV)。这通常通过学习到的线性变换（权重矩阵 WQW\_QWQ​、WKW\_KWK​ 和 WVW\_VWV​）来实现：

Q=XWQQ = X W\_QQ=XWQ​
K=XWKK = X W\_KK=XWK​
V=XWVV = X W\_VV=XWV​

这里，WQW\_QWQ​、WKW\_KWK​ 和 WVW\_VWV​ 是训练期间学习到的参数 (parameter)矩阵。这些矩阵的维度允许模型将输入嵌入投影到适合计算注意力得分和加权值的空间中。

一旦我们有了 QQQ、KKK 和 VVV，注意力得分将使用之前介绍的缩放点积注意力机制计算：

注意力(Q,K,V)=softmax(QKTdk)V\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V注意力(Q,K,V)=softmax(dk​​QKT​)V

其中 dkd\_kdk​ 是向量 (vector)的维度。核心点依然是：QQQ、KKK 和 VVV 都源自相同的输入 XXX。此操作的输出是一个新的序列表示，其中每个位置的向量是原始序列中所有位置的值向量的加权组合，基于查询-键的相似性。

SelfAttention

clusterᵢnput

输入序列 (X)

clusterₐttention

缩放点积注意力

X

输入嵌入 + 位置编码

WQ

权重 W\_Q

X->WQ

WK

权重 W\_K

X->WK

WV

权重 W\_V

X->WV

Q

查询 (Q)

WQ->Q

K

键 (K)

WK->K

V

值 (V)

WV->V

Attn

注意力 = softmax(QK
T
 / √d
k
)V

Q->Attn

计算权重

K->Attn

计算权重

V->Attn

应用权重

Attn->Attn

应用权重

Output

输出表示

Attn->Output

> 查询、键和值从同一输入序列 XXX 经过不同线性投影得到，随后进行缩放点积注意力计算。

例如，在处理句子“The animal didn't cross the street because it was too tired”（动物没有过马路因为它太累了）时，自注意力机制使模型能够了解“it”（它）指的是“animal”（动物）而非“street”（街道）。与“it”关联的查询会强烈关注与“animal”和“street”关联的键，但兼容性函数（通过 WQW\_QWQ​ 和 WKW\_KWK​ 学习）与 softmax 结合后，会给与“animal”关联的值分配更高的权重。

尽管自注意力在捕获序列内部关系方面很有效，但依赖单一的投影集合（WQ,WK,WVW\_Q, W\_K, W\_VWQ​,WK​,WV​）可能会迫使注意力机制对可能冲突或不同类型的关系进行平均处理。例如，它可能难以仅通过一次注意力计算就同时关注句法依赖性和语义相似性。这一局限性促使了多头注意力（Multi-Head Attention）的出现，在多头注意力中，我们使用不同的学习投影并行地执行多次自注意力计算。

获取即时帮助、个性化解释和交互式代码示例。

---

### 单一注意力头的局限性

# 单一注意力头的局限性

自注意力 (self-attention)机制 (attention mechanism)通过让查询、键和值（Q,K,VQ, K, VQ,K,V）源自同一输入序列，使模型能够衡量不同token的相对重要性。然而，每个位置依赖于*单一*注意力计算会带来显著的制约。

设想缩放点积注意力函数对于特定查询（代表一个token）的输出。它是值向量 (vector)的加权和，其中权重 (weight)由查询与所有键的匹配度决定。此过程为序列中每个位置生成一个单一的上下文 (context)向量。

主要制约出现是因为这种单一注意力机制必须学习仅使用一套注意力权重来编码多种类型的关系和特征。考虑哪些信息可能是相关的：

1. **句法关联：** 当前token在结构上与其它token如何关联（例如，主谓一致）？
2. **语义相似性：** 哪些其它token具有相关含义？
3. **位置信息：** 哪些token在附近，或处于特定的相对位置？
4. **指代关系：** 哪些其它token指代同一实体？

单一注意力头被迫将这些可能不同的关联信号平均成一个表示。例如，强烈关注一个句法上关联的动词可能需要降低对一个语义相似但句法上相距较远的名词的关注。这种平均化效应会造成信息瓶颈，可能阻止模型同时获取不同的、细致的模式。如果模型学习了一个尝试满足所有需求的“平均”注意力模式，它在获取任何特定模式时都可能表现不佳。

此外，将输入嵌入 (embedding)转换为单一Q,K,VQ, K, VQ,K,V空间的初始线性投影也可能具有限制性。模型学习一套单一的投影矩阵（WQ,WK,WVW^Q, W^K, W^VWQ,WK,WV）。这种单一转换可能将输入投影到一个子空间，该子空间会突出某些特征，但会掩盖其他特征。这限制了模型查看输入嵌入中不同表示子空间的能力，而这些子空间中不同类型的关系可能更易于辨识。

设想处理这个句子：“该模型架构，其依赖于注意力机制，表现良好。” 对于token“architecture”，一个单一注意力头可能需要同时确定它与“model”（修饰语）以及“performs”（执行动作的主语）之间的关系。学习到的注意力权重将代表了这些不同关联需求之间的折衷。

单一注意力计算的这种固有局限促使人们发展出一种更精巧的方法：多头注意力 (multi-head attention)。通过并行进行多次注意力计算，并使用不同的学习到的线性投影，模型能够共同关注来自不同表示子空间的信息，获取更丰富的特征和关系集合，而无需将它们强行通过一个单一瓶颈。我们将在下一节审视这一机制。

获取即时帮助、个性化解释和交互式代码示例。

---

### 引入多头注意力

# 引入多头注意力

如前所述，自注意力 (self-attention)提供了一种将序列中不同位置关联起来的有效机制。然而，应用单一的缩放点积注意力函数可能会像一个瓶颈。它迫使模型将可能多种不同类型的关系或依赖平均成一个单一的加权表示。设想一个句子，对其的理解需要同时追踪句法结构（如主谓一致）和语义（如词语相似性）。单一的注意力函数可能难以同时有效地捕捉这两个方面，可能会将它们模糊地混合在一起。

这一局限促使我们使用**多头注意力 (multi-head attention)**。其核心思想简单而巧妙：我们不是只计算一次注意力，而是并行执行多次注意力计算。每次并行计算被称为一个**注意力头**。

可以将其想象为对相同输入拥有多个视角。如果你正在分析一个复杂系统，你可能会咨询来自不同领域的专家。每位专家（注意力头）带来一个独特观点，侧重于系统（表示子空间）的不同方面。多头注意力机制 (attention mechanism)的运行方式与之类似。

每个注意力头执行相同的核心缩放点积注意力计算。然而，它们并非都基于直接来源于输入序列嵌入 (embedding)的相同查询 (QQQ)、键 (KKK) 和值 (VVV) 矩阵进行运算。相反，在每个头的注意力计算之前，QQQ、KKK 和 VVV 矩阵会经历独立的、每个头特有的线性变换（投影）。这意味着每个注意力头学会将输入投影到一个子空间中，该子空间可能更适合捕捉特定类型的关系。

为什么这样做有益？

- **关注不同子空间：** 它允许模型一同处理来自不同位置、不同表示子空间的信息。例如，一个头可能学会侧重于短期依赖，另一个侧重于语义关联 (semantic relationship)，还有一个侧重于位置信息。
- **捕捉多类模式：** 通过并行操作不同的学习投影，这些注意力头可以共同捕捉比单一注意力头单独处理更丰富的特征和关系。
- **稳定注意力：** 单一注意力机制中固有的平均效应有时可能是有害的。多头注意力允许不同的头进行专业化，它们的输出稍后会组合，从而提供一个更灵活且通常更稳定的学习过程。

本质上，多头注意力让模型对输入序列进行多次“审视”，每次都通过一个不同的学习视角（线性投影）进行聚焦。这种并行处理使得模型能够对数据中复杂的关联有更全面的理解。接下来的部分将详细解释这些线性投影在每个头中如何运作，以及它们的并行输出最终如何整合。

获取即时帮助、个性化解释和交互式代码示例。

---

### 每个注意力头的Q、K、V线性投影

# 每个注意力头的Q、K、V线性投影

已然明确，仅依赖单一自注意力 (self-attention)机制 (attention mechanism)，会限制模型获取序列内多重关系的能力。单次注意力计算可能平均化各类关联信号，或会因此遗失特定有益信息。多头注意力 (multi-head attention)通过并行执行多次注意力计算来应对此况，使得模型能同时考量来自不同视角或“表示子空间”的信息。

多头注意力（Multi-Head Attention）机制通过为每个“头”生成不同的查询（QQQ）、键（KKK）和值（VVV）向量 (vector)来执行并行处理。这意味着我们并非直接将原始输入嵌入 (embedding)送入每个头的注意力计算，而是对每个头单独应用学得的线性变换到输入序列上。

设输入序列的表示为一个矩阵 X∈Rn×dmodelX \in \mathbb{R}^{n \times d\_{model}}X∈Rn×dmodel​，其中 nnn 是序列长度，dmodeld\_{model}dmodel​ 是输入嵌入的维度（以及模型的隐藏状态大小）。对于一个有 hhh 个头的多头注意力机制，我们引入 hhh 组可学习的权重 (weight)矩阵：

- 查询权重矩阵：WiQ∈Rdmodel×dkW^Q\_i \in \mathbb{R}^{d\_{model} \times d\_k}WiQ​∈Rdmodel​×dk​，适用于 i=1,...,hi=1, ..., hi=1,...,h
- 键权重矩阵：WiK∈Rdmodel×dkW^K\_i \in \mathbb{R}^{d\_{model} \times d\_k}WiK​∈Rdmodel​×dk​，适用于 i=1,...,hi=1, ..., hi=1,...,h
- 值权重矩阵：WiV∈Rdmodel×dvW^V\_i \in \mathbb{R}^{d\_{model} \times d\_v}WiV​∈Rdmodel​×dv​，适用于 i=1,...,hi=1, ..., hi=1,...,h

这里，dkd\_kdk​ 表示每个头的键和查询的维度，dvd\_vdv​ 表示每个头的值的维度。一种常见且高效的设定，在原版Transformer论文《Attention Is All You Need》中有所采用，是将 dk=dv=dmodel/hd\_k = d\_v = d\_{model} / hdk​=dv​=dmodel​/h。这可确保所有头的总计算开销，与作用于完整 dmodeld\_{model}dmodel​ 维度的单一注意力机制大致相同，同时分散了表示能力。

对于每个头 iii，其相应的查询、键和值矩阵（QiQ\_iQi​, KiK\_iKi​, ViV\_iVi​）通过将输入 XXX 投影，并使用这些权重矩阵来计算：

Qi=XWiQQ\_i = X W^Q\_iQi​=XWiQ​
Ki=XWiKK\_i = X W^K\_iKi​=XWiK​
Vi=XWiVV\_i = X W^V\_iVi​=XWiV​

所得矩阵 Qi,Ki∈Rn×dkQ\_i, K\_i \in \mathbb{R}^{n \times d\_k}Qi​,Ki​∈Rn×dk​ 和 Vi∈Rn×dvV\_i \in \mathbb{R}^{n \times d\_v}Vi​∈Rn×dv​ 现在表示输入序列已变换至与头 iii 相关的特定子空间。

G

clusterₕead1

头1

clusterₕeadₕ

头h

X

输入嵌入
X (n x dₘodel)

proj1

线性投影
W₁^Q, W₁^K, W₁^V

X->proj1

dₘodel ➞ dₖ/dᵥ

projₕ

线性投影
Wₕ^Q, Wₕ^K, Wₕ^V

X->projₕ

dₘodel ➞ dₖ/dᵥ

Q1

Q₁ (n x dₖ)

proj1->Q1

K1

K₁ (n x dₖ)

proj1->K1

V1

V₁ (n x dᵥ)

proj1->V1

dots

...

Qh

Qₕ (n x dₖ)

projₕ->Qh

Kh

Kₕ (n x dₖ)

projₕ->Kh

Vh

Vₕ (n x dᵥ)

projₕ->Vh

> 流程图描绘了输入嵌入 XXX 如何为每个注意力头 iii 独立进行投影，使用不同的权重矩阵（WiQ,WiK,WiVW^Q\_i, W^K\_i, W^V\_iWiQ​,WiK​,WiV​）以生成头专属的查询 (QiQ\_iQi​)、键 (KiK\_iKi​) 和值 (ViV\_iVi​) 矩阵。投影通常将维度从 dmodeld\_{model}dmodel​ 降低到 dkd\_kdk​ 或 dvd\_vdv​。

### 使用线性投影的缘由

这些独立的线性变换之所以具有重要性，缘由如下：

1. **学习不同子空间：** 每组投影矩阵 (WiQ,WiK,WiV)(W^Q\_i, W^K\_i, W^V\_i)(WiQ​,WiK​,WiV​) 实际定义了一个不同的线性子空间。将输入嵌入 (embedding)投影到这些不同子空间中，使得每个注意力头能专注于序列中不同类型或方面的关系。例如，一个头可以学会关注短程句法依赖，而另一个则可能侧重于长程语义关联 (semantic relationship)。
2. **增强表示能力：** 采用 hhh 组投影矩阵，使得模型拥有更多参数 (parameter)，因而比单一组投影更能学习复杂模式。即使每个头在降维空间 (dk,dvd\_k, d\_vdk​,dv​) 上运作，在汇集所有头的信息后，其综合效果也可能比在完整 dmodeld\_{model}dmodel​ 维度上运行的单个头更为强大。
3. **促成并行专长化：** 由于每个头 iii 接收到其自身的投影版本 Qi,Ki,ViQ\_i, K\_i, V\_iQi​,Ki​,Vi​，因此它在训练过程中可以相对独立于其他头来学习专门的注意力模式。这鼓励了学得表示的多样性。

在实践中，这些线性投影通过标准前馈层（在PyTorch或TensorFlow等框架中常称为线性层或全连接层）实现，不含偏置 (bias)项。权重 (weight)矩阵 WiQ,WiK,WiVW^Q\_i, W^K\_i, W^V\_iWiQ​,WiK​,WiV​ 随机初始化，并在训练过程中通过反向传播 (backpropagation)与网络其余参数一同习得。

一旦这些头专属的 Qi,Ki,ViQ\_i, K\_i, V\_iQi​,Ki​,Vi​ 矩阵计算完成，它们将作为并行缩放点积注意力计算的输入，此部分我们接下来将查看。这一投影步骤对于多头注意力 (multi-head attention)机制 (attention mechanism)如何同时从多个视角分析输入序列来说，扮演着基础角色。

获取即时帮助、个性化解释和交互式代码示例。

---

### 并行注意力计算

# 并行注意力计算

并行注意力计算涉及将原始的查询 (QQQ)、键 (KKK) 和值 (VVV) 投影到 hhh 个不同的子空间中，对每个头 iii 使用不同的学习到的线性变换 WiQ,WiK,WiVW^Q\_i, W^K\_i, W^V\_iWiQ​,WiK​,WiV​。对于每个头，注意力计算独立且同时地进行。这种并行处理是多头注意力 (multi-head attention)的一种明确特征，并且对它的效率和计算特性有很大的贡献。

对于每个头 iii（iii 的范围从 1 到 hhh），我们准确地按照之前定义的方式计算带缩放的点积注意力，但使用的是该头特有的*投影*矩阵 QiQ\_iQi​、KiK\_iKi​ 和 ViV\_iVi​:

headi=Attention(Qi,Ki,Vi)=softmax(QiKiTdki)Vi\text{head}\_i = \text{Attention}(Q\_i, K\_i, V\_i) = \text{softmax}\left(\frac{Q\_i K\_i^T}{\sqrt{d\_{k\_i}}}\right) V\_iheadi​=Attention(Qi​,Ki​,Vi​)=softmax(dki​​​Qi​KiT​​)Vi​

这里:

- Qi=QWiQQ\_i = Q W^Q\_iQi​=QWiQ​ 表示为头 iii 投影的查询。
- Ki=KWiKK\_i = K W^K\_iKi​=KWiK​ 表示为头 iii 投影的键。
- Vi=VWiVV\_i = V W^V\_iVi​=VWiV​ 表示为头 iii 投影的值。
- dkid\_{k\_i}dki​​ 是头 iii *内部*键（和查询）的维度。这个缩放因子，类似于单头带缩放点积注意力中使用的，能在训练期间稳定梯度。

正确管理维度是很重要的。如果输入嵌入 (embedding)维度是 dmodeld\_{\text{model}}dmodel​ 并且我们使用 hhh 个头，投影通常被设计成使每个头的键、查询 (dkid\_{k\_i}dki​​) 和值 (dvid\_{v\_i}dvi​​) 的维度相等：dki=dvi=dmodel/hd\_{k\_i} = d\_{v\_i} = d\_{\text{model}} / hdki​​=dvi​​=dmodel​/h。这种划分确保了总计算成本与使用完整 dmodeld\_{\text{model}}dmodel​ 维度的单头注意力机制 (attention mechanism)相似，同时将表示能力分配到多个头。此外，它保证了当所有头的输出稍后被拼接时，结果维度与后续层所需的输入维度 dmodeld\_{\text{model}}dmodel​ 相匹配，从而保持了模型架构的统一性。

假设输入序列的长度为 NNN（词元 (token)数量），为了简化，忽略批处理维度，头 iii 的矩阵形状通常是：

- QiQ\_iQi​: N×dkiN \times d\_{k\_i}N×dki​​
- KiK\_iKi​: N×dkiN \times d\_{k\_i}N×dki​​ （因为在自注意力 (self-attention)中，键和查询来自相同的输入序列）
- ViV\_iVi​: N×dviN \times d\_{v\_i}N×dvi​​

因此，头 iii 的注意力计算输出，记为 headi\text{head}\_iheadi​，将具有形状 N×dviN \times d\_{v\_i}N×dvi​​。由于我们通常设置 dvi=dki=dmodel/hd\_{v\_i} = d\_{k\_i} = d\_{\text{model}} / hdvi​​=dki​​=dmodel​/h，输出形状为 N×(dmodel/h)N \times (d\_{\text{model}} / h)N×(dmodel​/h)。

从计算角度看，这种结构非常适合并行处理。现代深度学习 (deep learning)框架和像 GPU 这样的硬件擅长执行大型矩阵乘法。所有 hhh 个头的计算通常可以并行执行，而不是顺序地迭代每个头。这通常通过在注意力计算之前重塑投影的 Q、K 和 V 张量来实现，使其包含一个独立的“头”维度。例如，表示批处理查询的张量可以从 `(batch_size, seq_len, d_model)` 重塑为 `(batch_size, num_heads, seq_len, d_k_i)`。用于 QiKiTdki\frac{Q\_i K\_i^T}{\sqrt{d\_{k\_i}}}dki​​​Qi​KiT​​ 项的批量矩阵乘法（`matmul`），然后是 `softmax` 和与 ViV\_iVi​ 的最终 `matmul`，可以同时高效地在批处理和头维度上运行。

G

clusterᵢnput

输入投影

clusterₕeads

并行注意力头 (1 到 h)

clusterₕead1

头 1

clusterₕeadᵢ

头 i

clusterₕeadₕ

头 h

Q

Q (N x dₘodel)

Proj1

Q₁, K₁, V₁
(N x dk₁, N x dk₁, N x dv₁)

Q->Proj1

WQ₁

Proji

Qᵢ, Kᵢ, Vᵢ
(N x dki, N x dki, N x dvi)

Q->Proji

WQᵢ

Projh

Q<SUB>h</SUB>, K<SUB>h</SUB>, V<SUB>h</SUB>
(N x dkh, N x dkh, N x dvh)

Q->Projh

WQ<SUB>h</SUB>

K

K (N x dₘodel)

K->Proj1

WK₁

K->Proji

WKᵢ

K->Projh

WK<SUB>h</SUB>

V

V (N x dₘodel)

V->Proj1

WV₁

V->Proji

WVᵢ

V->Projh

WV<SUB>h</SUB>

Attn1

Attention(Q₁, K₁, V₁)

Proj1->Attn1

Head1ₒut

head₁
(N x dv₁)

Attn1->Head1ₒut

Dots
...

Concat

拼接与最终投影
(下一节)

Head1ₒut->Concat

Attni

Attention(Qᵢ, Kᵢ, Vᵢ)

Proji->Attni

Headiₒut

headᵢ
(N x dvi)

Attni->Headiₒut

Dots2
...

Headiₒut->Concat

Attnh

Attention(Q<SUB>h</SUB>, K<SUB>h</SUB>, V<SUB>h</SUB>)

Projh->Attnh

Headhₒut

head<SUB>h</SUB>
(N x dvh)

Attnh->Headhₒut

Headhₒut->Concat

> 对每个头，使用其特有的投影 Q、K、V 矩阵 (Qi,Ki,ViQ\_i, K\_i, V\_iQi​,Ki​,Vi​) 执行独立的带缩放点积注意力计算。输出（head1,...,headh\text{head}\_1, ..., \text{head}\_hhead1​,...,headh​），每个的维度为 N×dviN \times d\_{v\_i}N×dvi​​，在传递到下一阶段之前并行生成。注意，dkid\_{k\_i}dki​​ 和 dvid\_{v\_i}dvi​​ 代表 dmodel/hd\_{\text{model}}/hdmodel​/h。

这种并行结构的主要优点不限于计算效率。它允许每个注意力头可能专门学习不同类型的关系，或者同时关注来自不同表示子空间的信息。例如，一个头可以学习侧重于局部的句法依赖（如形容词与名词的一致性），而另一个头则捕捉更远距离的语义联系（如跨句子的指代消解），还有一个头可能侧重于位置关系。单个注意力机制将被迫平均这些可能不同的信号，这可能会稀释信息。多头注意力为信息流提供了多个独立的“通道”，让模型能够汇集多样化的关系信息，并最终构建更丰富、更具上下文 (context)意识的表示。

这些并行计算的输出，head1,head2,...,headh\text{head}\_1, \text{head}\_2, ..., \text{head}\_hhead1​,head2​,...,headh​，捕获了输入序列内部关系的不同方面。它们现在已准备好在下一步中组合：先拼接，再进行最终的线性投影。

获取即时帮助、个性化解释和交互式代码示例。

---

### 拼接与最终线性投影

# 拼接与最终线性投影

在为每个 hhh 个头并行计算缩放点积注意力后，我们会得到 hhh 个不同的输出向量 (vector)，通常表示为 head1,head2,...,headhhead\_1, head\_2, ..., head\_hhead1​,head2​,...,headh​。每个 headihead\_iheadi​ 都是通过对原始查询、键和值使用投影版本来应用注意力机制 (attention mechanism)的结果。

headi=Attention(XWiQ,XWiK,XWiV)head\_i = \text{Attention}(XW\_i^Q, XW\_i^K, XW\_i^V)headi​=Attention(XWiQ​,XWiK​,XWiV​)

其中 XXX 表示输入序列嵌入 (embedding)（或来自上一层的输出），并且 WiQW\_i^QWiQ​、WiKW\_i^KWiK​、WiVW\_i^VWiV​ 是第 iii 个头的参数 (parameter)矩阵。每个 headihead\_iheadi​ 的维度为 dvd\_vdv​，其中 dv=dmodel/hd\_v = d\_{model} / hdv​=dmodel​/h。这种划分使得所有头的计算成本与具有完整 dmodeld\_{model}dmodel​ 维度的单头注意力机制的计算成本相当。

然而，Transformer 中的后续层，例如逐位置前馈网络和残差连接，需要一个具有模型隐藏维度 dmodeld\_{model}dmodel​ 的单个输入张量。因此，我们需要一种机制将这些并行头的输出整合回一个单一的表示形式。

### 组合头输出：拼接

整合不同头所捕获信息的首要步骤是直接的：拼接。所有 hhh 个头的输出向量 (vector)沿其特征维度进行拼接。

如果每个 headihead\_iheadi​ 是一个形状为（序列长度，dvd\_vdv​）的矩阵，拼接操作会将这些矩阵并排堆叠，产生一个形状为（序列长度，h×dvh \times d\_vh×dv​）的新矩阵。由于我们将 dvd\_vdv​ 定义为 dmodel/hd\_{model} / hdmodel​/h，得到的拼接矩阵具有（序列长度，dmodeld\_{model}dmodel​）的维度。

拼接(head1,head2,...,headh)∈R序列长度×(h⋅dv)=R序列长度×dmodel\text{拼接}(head\_1, head\_2, ..., head\_h) \in \mathbb{R}^{\text{序列长度} \times (h \cdot d\_v)} = \mathbb{R}^{\text{序列长度} \times d\_{model}}拼接(head1​,head2​,...,headh​)∈R序列长度×(h⋅dv​)=R序列长度×dmodel​

此操作有效汇聚了每个头所关注的不同表征子空间中获取的见解。

G

clusterₕeads

并行注意力头

cluster\_combine

组合阶段

head1

头 1 输出
(seqₗen, dv)

concat

拼接
(seqₗen, h\*dv)

head1->concat

head2

头 2 输出
(seqₗen, dv)

vdots

...

head2->concat

headₕ

头 h 输出
(seqₗen, dv)

headₕ->concat

linear

线性投影 (W^O)
(dₘodel, dₘodel)

concat->linear

output

多头输出
(seqₗen, dₘodel)

linear->output

> 流程图显示了单个注意力头的输出如何被拼接，然后通过最终线性层进行投影。

### 最终线性投影

拼接汇聚了头部的输出，但结果张量只是简单地将这些专用表示并置。为了让这些表示能够有效交互和组合，并确保输出维度与后续层（特别是残差连接）所需的 dmodeld\_{model}dmodel​ 相匹配，会应用一个最终线性投影。

这种投影通过另一个学习到的权重 (weight)矩阵 WOW^OWO 实现，其维度为 (h×dv,dmodel)(h \times d\_v, d\_{model})(h×dv​,dmodel​)，简化后为 (dmodel,dmodel)(d\_{model}, d\_{model})(dmodel​,dmodel​)。拼接后的输出与此权重矩阵相乘：

多头输出=拼接(head1,...,headh)WO\text{多头输出} = \text{拼接}(head\_1, ..., head\_h) W^O多头输出=拼接(head1​,...,headh​)WO

这种最终线性变换作用显著：

1. **信息混合：** 它使得从不同头（代表不同子空间）中学习到的信息得以组合和整合。线性层充当一个学习到的组合函数。
2. **维度匹配：** 它确保输出张量具有Transformer块架构其余部分所需的精确 dmodeld\_{model}dmodel​ 维度，从而能够进行残差连接（加和归一化 (normalization)）等操作。

根本上，多头机制首先将模型的表示能力分解为多个子空间（hhh 个头，每个维度为 dvd\_vdv​），允许每个头在其子空间内专门关注输入的不同方面，然后使用拼接，再进行线性投影（WOW^OWO），将这些专业化的视图合并回一个维度为 dmodeld\_{model}dmodel​ 的单一、更丰富的表示。

这种结构使模型能够同时捕获多种关系模式（例如，短距离依赖、长距离依赖、句法关系），这对于单一注意力机制 (attention mechanism)而言将更具挑战。投影矩阵（每个头的 WiQ,WiK,WiVW\_i^Q, W\_i^K, W\_i^VWiQ​,WiK​,WiV​）和最终输出投影矩阵（WOW^OWO）的所有参数 (parameter)都在模型训练期间进行端到端学习。

获取即时帮助、个性化解释和交互式代码示例。

---

### 不同注意力头学习内容的分析

# 不同注意力头学习内容的分析

引入多个注意力头提出了一个重要问题：这些并行的注意力机制 (attention mechanism)是否真的学习了不同的内容？如果每个头只是学习相同的模式，那么增加的计算复杂度相比单个、更大的注意力头将提供很少的优势。幸运的是，经验证据和分析表明，不同的头通常会专门化，学习关注输入序列中不同类型的关系。

为了使多头自注意力 (self-attention)机制中的每个注意力头能够学习并专注于不同的关系类型，其实现方式是为每个头对查询（QQQ）、键（KKK）和值（VVV）应用独立的线性投影。头 iii 的投影具体如下：
Qi=QWiQQ\_i = Q W^Q\_iQi​=QWiQ​
Ki=KWiKK\_i = K W^K\_iKi​=KWiK​
Vi=VWiVV\_i = V W^V\_iVi​=VWiV​
其中 WiQ∈Rdmodel×dkW^Q\_i \in \mathbb{R}^{d\_{model} \times d\_k}WiQ​∈Rdmodel​×dk​、WiK∈Rdmodel×dkW^K\_i \in \mathbb{R}^{d\_{model} \times d\_k}WiK​∈Rdmodel​×dk​ 和 WiV∈Rdmodel×dvW^V\_i \in \mathbb{R}^{d\_{model} \times d\_v}WiV​∈Rdmodel​×dv​ 是头 iii 的可学习权重 (weight)矩阵。由于这些矩阵是独立初始化并通过反向传播 (backpropagation)更新的，因此每个头都有能力将输入嵌入 (embedding)投影到一个子空间，使某种特定关系在其中更明显或对模型目标更有用。

### 观察头部专门化

准确理解每个头学习到什么是一个活跃的研究区域，通常被称为“可解释性”。一种常用技术涉及可视化不同头针对给定输入序列生成的注意力权重 (weight)。通过检查特定头内哪些标记 (token)强烈关注哪些其他标记，我们可以推断其优先处理的模式。

例如，考虑句子：“The quick brown fox jumps over the lazy dog。”我们可能会观察到如下模式：

- **头1 (局部关联)：** 注意力权重在相邻词之间可能最高，作用类似于二元语法模型。“quick”可能强烈关注“The”和“brown”。
- **头2 (句法关联)：** 注意力可能集中在句法相关的词语上，即使它们相距较远。“jumps”可能强烈关注其主语“fox”。
- **头3 (特定标记)：** 如果标点符号或特殊标记对任务携带重要信息，一个头可能会持续强烈关注它们。
- **头4 (内容相似性)：** 注意力可以关联具有相关含义或作用的词，例如“fox”和“dog”。

G

cluster₀

输入序列

The₀

The

quick₁

quick

brown₂

brown

quick₁->brown₂

Head 1

brown₂->quick₁

fox₃

fox

jumps₄

jumps

dog₈

dog

fox₃->dog₈

Head 4

jumps₄->fox₃

Head 2

over₅

over

the₆

the

lazy₇

lazy

lazy₇->dog₈

dog₈->fox₃

dog₈->lazy₇

> 说明不同的头在句子“The quick brown fox jumps over the lazy dog”中可能关注不同关系。头1（蓝色，虚线）局部关注，头2（粉色，实线）连接动词和主语，头4（绿色，虚点）连接相关名词。

可视化通常使用热力图，其中行和列代表标记位置，颜色强度表示注意力分数 softmax(QiKiTdk)softmax(\frac{Q\_i K\_i^T}{\sqrt{d\_k}})softmax(dk​​Qi​KiT​​)。不同的头产生不同的热力图，突显其不同的关注点。

更严谨的分析涉及“探测”。这意味着在单个头的输出表示（Zi=Attention(Qi,Ki,Vi)Z\_i = Attention(Q\_i, K\_i, V\_i)Zi​=Attention(Qi​,Ki​,Vi​)）上训练简单的线性分类器或其他小型模型，以查看它们能多好地预测特定语言属性（例如，词性标注、句法依存）。成功预测某一属性表明该头编码了与之相关的信息。

### 学习模式的例子

分析训练好的Transformer的研究已发现头部之间有几种常见的专门化类型：

1. **位置/局部关注：** 有些头主要关注当前标记 (token)周围小窗口内的标记，有效捕捉局部上下文 (context)。这可能类似于卷积网络或N元语法模型的行为。
2. **句法关联：** 某些头善于追踪语法关系。它们可能学习连接动词与主语或宾语、形容词与修饰的名词，或介词与宾语，有时甚至跨越长距离。
3. **指代/相关实体：** 头可以关联文本中同一实体的提及，或连接语义相关的词语。
4. **关注分隔符/特殊标记：** 在使用特殊标记（如`[CLS]`、`[SEP]`）的模型中，一些头通常会强烈关注这些标记，可能将其用作序列级别信息的聚合点。
5. **稀有词处理：** 一些头可能专门关注从稀有词到更常见、上下文信息丰富的词语的连接，帮助模型理解不常用词。
6. **恒等/复制（较不常见）：** 偶尔，一个头可能学习强烈关注当前标记自身的位置，有效复制其表示。

### 头部多样性的益处

不同头专门化的能力提供了几个优势：

- **捕捉多样信息：** 序列包含多层信息（句法、语义、位置）。多头注意力 (multi-head attention)允许模型同时捕捉这些不同方面，而无需强制单个机制平均可能冲突的信号。
- **更丰富的表示：** 通过拼接这些专门化头的输出（Z=Concat(Z1,...,Zh)WOZ = Concat(Z\_1, ..., Z\_h)W^OZ=Concat(Z1​,...,Zh​)WO），模型为每个标记 (token)构建更丰富、多方面的表示，融入了来自各种关系视角的见解。
- **改进的模型能力：** 并行子空间允许模型表达比具有相同总维度的单个注意力机制 (attention mechanism)更复杂的函数和依赖关系。

### 考量和注意事项

虽然头部专门化的思想很有吸引力并有证据支持，但有些方面需要考量：

- **可解释性具有挑战性：** 为每个头分配单一、明确的功能通常很困难。头可能扮演多种角色，或者其行为可能复杂且依赖于上下文 (context)。可视化提供线索，但不是确切答案。
- **冗余：** 并非所有头都必然学习独特的模式。可能存在一定程度的冗余，或者某些头对最终输出的贡献较小。修剪不那么重要的头是模型压缩的一个研究区域。
- **层级依赖性：** 头学习到的模式可能因其在Transformer堆栈中的深度而异。较低层的头可能更关注局部句法，而较高层的头可能捕捉更复杂的语义或长距离依赖关系。

总而言之，多头结构不仅是关于并行计算；它是一种鼓励功能专门化的设计。通过允许不同的头关注不同表示子空间中的信息，模型可以整合多样的关系模式，从而形成更有效的序列表示。最终的线性投影（WOW^OWO）学习如何最好地结合这些专门化的视角，供前馈网络和后续层进行下游处理。

获取即时帮助、个性化解释和交互式代码示例。

---

### 动手实践：构建多头注意力层

# 动手实践：构建多头注意力层

使用 PyTorch 构建多头自注意力 (self-attention)的实际实现方法。从零开始构建此层有助于巩固对数据流动和张量操作的理解。本实践活动要求对 PyTorch 基础模块和张量操作有一定了解。

我们的目标是创建一个 `MultiHeadAttention` 模块，该模块接受输入序列，并将其投射为多个注意力头的查询（Q）、键（K）和值（V），并行为每个头计算缩放点积注意力，将结果拼接，并进行最终的线性投射。

### 定义模块结构

我们将定义一个继承自 `torch.nn.Module` 的 Python 类。构造函数（`__init__`）将初始化用于初始 Q、K、V 投射以及最终输出投射所需的线性层。我们还需要保存嵌入 (embedding)维度（`d_model`）和注意力头数量（`num_heads`）。一个重要条件是 `d_model` 必须能被 `num_heads` 整除，以便投射维度 (dkd\_kdk​, dvd\_vdv​) 为整数。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """实现了多头注意力机制。"""

    def __init__(self, d_model: int, num_heads: int):
        """
        参数:
            d_model (int): 输入和输出嵌入的维度。
            num_heads (int): 注意力头的数量。
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """计算缩放点积注意力。"""

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, V)
        return output, attn_probs

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        执行多头注意力的前向计算。

        参数:
            query (torch.Tensor): 查询张量，形状 (batch_size, seq_len_q, d_model)
            key (torch.Tensor): 键张量，形状 (batch_size, seq_len_k, d_model)
            value (torch.Tensor): 值张量，形状 (batch_size, seq_len_v, d_model)
                                   (seq_len_k 和 seq_len_v 必须相同)
            mask (torch.Tensor, optional): 用于阻止对特定位置注意力的掩码张量。
                                            形状取决于应用（例如，填充掩码，先行掩码）。
                                            默认为 None。

        返回:
            torch.Tensor: 输出张量，形状 (batch_size, seq_len_q, d_model)
        """
        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(attn_output)

        return output
```

### 代码解析

1. **初始化（`__init__`）**：我们设置了四个 `nn.Linear` 层。`W_q`、`W_k`、`W_v` 将输入嵌入 (embedding) `d_model` 投射为 Q、K、V 向量 (vector)，其大小也为 `d_model`。此处的实现细节是，我们首先投射到 `d_model`，然后重塑，而不是直接为每个头投射到 `d_k`。两种方法都可行。`W_o` 是最终的输出变换层。我们将 `d_k`（每个头的维度）计算为 `d_model / num_heads`。
2. **缩放点积注意力**：为清晰起见，我们包含了一个单独的 `scaled_dot_product_attention` 方法。这封装了注意力算法：Attention(Q,K,V)=softmax(QKTdk)VAttention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d\_k}})VAttention(Q,K,V)=softmax(dk​​QKT​)V。它计算注意力分数，应用缩放因子 1/dk1/\sqrt{d\_k}1/dk​​，可选地应用掩码（在 softmax 之前将掩蔽位置设置为一个很大的负数），使用 softmax 计算注意力概率，最后计算值的加权和。请注意，PyTorch 提供了一个高度优化的版本（`torch.nn.functional.scaled_dot_product_attention`），在生产代码中应优先使用以获得性能，但此处所示的手动实现有助于理解。
3. **前向计算（`forward`）**：
   - **输入**：`forward` 方法接受 `query`、`key` 和 `value` 张量。对于自注意力 (self-attention)（本章的主题），这三个张量将是相同的（来自*同一*输入序列）。我们将它们分开是为了保持通用性，因为此模块也可用于编码器-解码器交叉注意力（稍后讨论）。也可以提供一个可选的 `mask`。
   - **投射**：输入 `query`、`key`、`value` 张量通过各自的线性层（`W_q`、`W_k`、`W_v`）。
   - **重塑**：这一步很重要。投射的输出，形状 `(batch_size, seq_len, d_model)`，需要重塑为 `(batch_size, num_heads, seq_len, d_k)`。这将每个头的计算独立开来。`.view()` 方法重塑张量，而 `.transpose(1, 2)` 则交换 `num_heads` 和 `seq_len` 维度，为注意力函数中的批次矩阵乘法做准备。
   - **注意力计算**：`scaled_dot_product_attention` 方法使用重塑后的 Q、K、V 张量进行调用。批次矩阵乘法同时处理批次维度和头维度上的计算。
   - **拼接/重塑回原形**：注意力头的输出，形状 `(batch_size, num_heads, seq_len_q, d_k)`，需要组合。我们首先 `.transpose(1, 2)` 回到 `(batch_size, seq_len_q, num_heads, d_k)`。`.contiguous()` 确保张量存储在连续的内存块中，这有时在调用 `.view()` 之前是必需的。最后，`.view(batch_size, -1, self.d_model)` 将其重塑回所需的 `(batch_size, seq_len_q, d_model)` 格式，有效地沿嵌入维度拼接了各头的输出。
   - **最终投射**：这个拼接后的张量通过最终的线性层 `W_o` 产生模块的输出。

### 可视化流程

我们可以可视化多头注意力 (multi-head attention)层的数据流：

G

clusterᵢnput

输入张量

clusterₐttention

3. 缩放点积注意力（并行注意力头）

clusterᵣeshape\_concat

4. 拼接注意力头与重塑

cluster\_finalₚroj

5. 最终线性投射

clusterₒutput

输出张量

Query

查询
(B, Lq, dm)

Wq

线性 Wq

Query->Wq

Key

键
(B, Lk, dm)

Wk

线性 Wk

Key->Wk

Value

值
(B, Lv, dm)

Wv

线性 Wv

Value->Wv

ReshapeQ

重塑
(B, h, Lq, dk)

Wq->ReshapeQ

ReshapeK

重塑
(B, h, Lk, dk)

Wk->ReshapeK

ReshapeV

重塑
(B, h, Lv, dk)

Wv->ReshapeV

SDPA

SDPA
每个头

ReshapeQ->SDPA

 Q

ReshapeK->SDPA

 K

ReshapeV->SDPA

 V

ConcatReshape

转置与重塑
(B, Lq, dm)

SDPA->ConcatReshape

(B, h, Lq, dk)

Wo

线性 Wo

ConcatReshape->Wo

Output

输出
(B, Lq, dm)

Wo->Output

> 多头注意力模块内的数据流。B=批次大小，Lq/Lk/Lv=Q/K/V 的序列长度，dm=模型维度，h=注意力头数量，dk=每个头的维度。对于自注意力 (self-attention)，Lq=Lk=Lv。

此实现提供了对多个注意力头如何并行工作的具体认识。每个注意力头可能通过使用不同的投射来关注输入关联的不同方面，它们的综合信息通过最终的线性层进行整合。该层是 Transformer 编码器和解码器堆栈中反复使用的基本组成部分。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 4 Positional Encoding Embedding Layer

### 位置信息的必要性

# 位置信息的必要性

自注意力 (self-attention)机制 (attention mechanism)提供了一种有效方法，让模型在计算输入序列中每个元素的表示时，能够动态地衡量不同元素的权重 (weight)。它通过使用从输入中获得的查询（Query）、键（Key）和值（Value）向量 (vector)，计算元素对之间的对齐 (alignment)分数来实现这一点。

然而，核心自注意力计算，特别是缩放点积注意力，有一个重要特点：

Attention(Q,K,V)=softmax(QKTdk)VAttention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)VAttention(Q,K,V)=softmax(dk​​QKT​)V

它本质上是*排列不变*的。这意味着如果我们重新排列输入词元 (token)（及其对应的Q,K,VQ, K, VQ,K,V向量），在考虑其与所有*其他*词元的关系时，任何给定词元的注意力输出实际上会保持不变，只是根据初始排列重新排序。注意力机制在其基本形式中，将输入视为一个向量*集合*，而非一个有序序列。

设想一个简单句子：“robot detects anomaly”。如果模型仅使用词元嵌入 (embedding)和自注意力来处理此句，那么“robot”和“detects”之间，或“robot”和“anomaly”之间的注意力分数，将完全基于这些词的向量表示来计算，而不论它们的位置（第一、第二或第三）。如果我们把输入打乱成“anomaly detects robot”，那么嵌入之间的成对注意力分数会是相同的。由此产生的语境化表示将有所不同，*仅仅*是因为输入向量本身在每个位置不同，但机制本身并未处理*顺序*。

这与循环神经网络 (neural network)（RNN）或长短期记忆网络（LSTM）等循环架构形成鲜明对比。在RNN中，时间步ttt的计算直接依赖于时间步t−1t-1t−1的隐藏状态：ht=f(ht−1,xt)h\_t = f(h\_{t-1}, x\_t)ht​=f(ht−1​,xt​)。这种序列处理自然地包含了元素的顺序。而Transformer通过自注意力并行处理所有词元，失去了这种内置的序列顺序感。

对于几乎任何涉及序列的任务，特别是在自然语言处理中，顺序对意义非常重要。“狗追猫”与“猫追狗”的含义完全不同。时间序列预测严重依赖观测的时间顺序。因此，由于自注意力机制本身未捕获位置信息，我们必须找到另一种方法向模型提供此信息。馈入Transformer层的表示需要包含信号，这些信号不仅指示词元*是什么*，还指示它在序列中*位于何处*。这是引入位置编码 (positional encoding)的主要原因，我们接下来会进行了解。

获取即时帮助、个性化解释和交互式代码示例。

---

### 输入嵌入层转换

# 输入嵌入层转换

为了准备引入序列顺序信息，Transformer需要一种方式来表示输入词元 (token)本身，使其适合神经网络 (neural network)处理。原始文本首先通过词元化过程（使用字节对编码或WordPiece等方法）转换为整数序列，称为词元ID。然而，这些离散的词元ID不直接适合作为注意力机制 (attention mechanism)或后续神经网络层的输入。它们无法体现不同词元之间的相似性或关联。

输入嵌入 (embedding)层作为最初的转换步骤，将这些整数词元ID转换为密集的、连续的向量 (vector)表示。这与独热编码等旧技术形成鲜明对比，后者会生成维度非常高且稀疏的向量（大部分为零，只有一个一）。而密集嵌入则能在维度低得多的空间中捕获词元间的语义相似性，这个空间通常称为 dmodeld\_{model}dmodel​（模型的隐藏维度）。

### 机制：嵌入 (embedding)查找

其核心是，输入嵌入层作用为一个查找表。该表由一个权重 (weight)矩阵表示，我们称之为 EEE，其维度为 V×dmodelV \times d\_{model}V×dmodel​，这里的 VVV 是词汇表 (vocabulary)的大小（模型识别的唯一词元 (token)总数），dmodeld\_{model}dmodel​ 是选定的嵌入维度。

当给定输入词元ID序列 [t1,t2,...,tn][t\_1, t\_2, ..., t\_n][t1​,t2​,...,tn​] 时，嵌入层会从矩阵 EEE 中获取每个词元ID对应的向量 (vector)。如果 tkt\_ktk​ 是位置 kkk 的词元ID，它的嵌入向量 eke\_kek​ 就是矩阵 EEE 中由 tkt\_ktk​ 索引的行：

ek=Etke\_k = E\_{t\_k}ek​=Etk​​

此操作有效地将输入序列中的每个整数ID映射到大小为 dmodeld\_{model}dmodel​ 的密集向量 (dense vector)。

G

cluster₀

输入词元ID

cluster₁

嵌入矩阵 E (V x dₘodel)

cluster₂

输出嵌入

Input
[ 71, 8, 1234, 5 ]
(序列长度 = 4)

EmbeddingTable

ID

嵌入向量 (dₘodel)

0

[0.1, -0.2, ...]

...

...

8

[0.5, 0.1, ...]

...

...

71

[-0.3, 0.9, ...]

...

...

1234

[0.8, -0.1, ...]

...

...

5

[0.0, 0.4, ...]

...

...

V-1

[...]

Input->EmbeddingTable

查找

Output

[-0.3, 0.9, ...]

[0.5, 0.1, ...]

[0.8, -0.1, ...]

[0.0, 0.4, ...]

EmbeddingTable->Output

Dim
(序列长度 x dₘodel)

> 通过嵌入查找将离散词元ID映射到连续向量表示。

### 可学习表示

值得注意的是，嵌入 (embedding)矩阵 EEE 不是固定的；它的值是可学习的参数 (parameter)。在训练过程中，梯度通过网络反向传播 (backpropagation)，嵌入向量 (vector)被调整以最小化总损失函数 (loss function)。这意味着模型学会在 dmodeld\_{model}dmodel​ 维嵌入空间中，将具有相似语义或功能的词元 (token)放置得更近。例如，训练后“国王”和“王后”等词语可能最终具有相似的嵌入向量，反映出它们的语义关联 (semantic relationship)。

### 输出与维度

输入嵌入 (embedding)层的输出是一个向量 (vector)序列，[e1,e2,...,en][e\_1, e\_2, ..., e\_n][e1​,e2​,...,en​]，每个 eke\_kek​ 都是大小为 dmodeld\_{model}dmodel​ 的向量。此序列现在在连续空间中表示输入词元 (token)，但保留了原始序列长度。维度 dmodeld\_{model}dmodel​（例如512、768、1024）是Transformer架构的一个基本超参数 (parameter) (hyperparameter)，决定了模型大部分层中向量表示的宽度。

这个词元嵌入序列是后续加入位置信息的基本组成，正如接下来将讨论的。尽管这些嵌入现在带有从数据中学到的语义，但它们仍然缺乏对其在原始序列中位置的固有表示，这是一个通过位置编码 (positional encoding)解决的局限。

获取即时帮助、个性化解释和交互式代码示例。

---

### 正弦型位置编码：公式表述

# 正弦型位置编码：公式表述

自注意力 (self-attention)机制 (attention mechanism)在处理输入元素时，无法自然地理解它们在序列中的顺序。这种置换不变性在捕获无关距离的关系方面表现出色，但未能对语言及其他有序数据中必需的序列特性进行建模。为此，Transformer架构引入了位置编码 (positional encoding)，这些向量 (vector)被添加到输入嵌入 (embedding)中，用以向模型提供每个token的位置信息。

原始Transformer论文（“Attention Is All You Need”）提出了一种固定的、无需学习的方法，该方法利用不同频率的正弦和余弦函数。这种被称作正弦型位置编码的方法，能为序列中直至预设最大长度的每个位置生成独特的编码向量。

### 数学表述

对于序列中的给定位置 pospospos（其中 pos=0,1,2,...pos = 0, 1, 2, ...pos=0,1,2,...），以及 dmodeld\_{model}dmodel​ 维嵌入 (embedding)向量 (vector)中的给定维度 iii（其中 i=0,1,...,dmodel−1i = 0, 1, ..., d\_{model}-1i=0,1,...,dmodel​−1），位置编码 (positional encoding) PEPEPE 的定义是：对偶数维度索引 (2i2i2i) 应用正弦函数，对奇数维度索引 (2i+12i+12i+1) 应用余弦函数：

PE(pos,2i)=sin⁡(pos100002i/dmodel)PE\_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d\_{model}}}\right)PE(pos,2i)​=sin(100002i/dmodel​pos​)
PE(pos,2i+1)=cos⁡(pos100002i/dmodel)PE\_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d\_{model}}}\right)PE(pos,2i+1)​=cos(100002i/dmodel​pos​)

其中：

- pospospos 表示token在输入序列中的位置索引。
- iii 表示编码向量中的维度索引。请注意，这里的 iii 实际范围是从 000 到 dmodel/2−1d\_{model}/2 - 1dmodel​/2−1，因为每个 iii 定义了一对维度（2i2i2i 和 2i+12i+12i+1）。
- dmodeld\_{model}dmodel​ 是整个Transformer模型中使用的嵌入向量的维度（例如，512，768）。

项 1/100002i/dmodel1 / 10000^{2i/d\_{model}}1/100002i/dmodel​ 决定了正弦和余弦波的角频率。我们来分析其特性：

- 当 iii 较小（代表编码向量的起始维度）时，指数 2i/dmodel2i/d\_{model}2i/dmodel​ 接近于 0。底数 100002i/dmodel10000^{2i/d\_{model}}100002i/dmodel​ 接近于 100000=110000^0 = 1100000=1。这导致参数 (parameter) pos/1pos/1pos/1 相对较大，意味着这些维度上的正弦/余弦函数在位置上以高频率（短波长）振荡。
- 随着 iii 增加并接近 dmodel/2d\_{model}/2dmodel​/2，指数 2i/dmodel2i/d\_{model}2i/dmodel​ 接近于 1。底数 100002i/dmodel10000^{2i/d\_{model}}100002i/dmodel​ 接近于 100001=1000010000^1 = 10000100001=10000。这导致参数 pos/10000pos/10000pos/10000 较小，意味着这些后续维度上的函数在位置上以极低频率（长波长）振荡。

这种设计创建了位置编码向量，其中每个维度都对应特定频率的正弦波。这些正弦波在所有维度上的组合为每个位置 pospospos 生成了独特的标识。

010203040806040200−1−0.500.51正弦型位置编码可视化

> 热图显示了前50个位置（每隔2个位置采样）以及前128个维度（每隔4个维度采样）的正弦型位置编码值。每行代表一个维度索引，每列代表序列中的一个位置。颜色深浅表示编码值（范围从-1到1）。请注意，沿维度轴（y轴）的振荡频率有所不同。

从这些三角恒等式得出的一个重要性质是，位置 pos+kpos + kpos+k 的编码可以表示为位置 pospospos 编码的线性变换。这种线性性质可能有助于模型轻松地根据相对位置进行注意力计算，因为相距固定距离 kkk 的token编码之间的关系，无论其绝对位置如何，都保持一致。

总之，正弦型位置编码提供了一种确定性且计算高效的方法，用以向Transformer传入序列顺序信息。它能生成独特的位置标识，并具备有利于建模相对位置的性质，同时无需额外的可学习参数。

获取即时帮助、个性化解释和交互式代码示例。

---

### 正弦编码的特性

# 正弦编码的特性

在上一节中，我们已经定义了正弦位置编码 (positional encoding)的数学结构，具体为：

正弦位置编码的数学结构定义如下：

现在，我们来分析*为何*这种特殊的公式有效且常用。这些固定的、非学习的编码具有几个理想的特性，与Transformer架构良好匹配。

### 确定性和唯一性编码

与需要训练的学习型位置嵌入 (embedding)不同，正弦编码由固定函数生成。这意味着它们是确定性的：对于给定的位置 `pos` 和维度索引 `i`，其值始终相同。这也不需要专门针对位置信息的额外可训练参数 (parameter)。

此外，正弦和余弦函数在不同频率（由 100002i/dmodel10000^{2i/d\_{model}}100002i/dmodel​ 项确定）上的组合，确保在合理的序列长度内，每个位置 `pos` 获得唯一的编码向量 (vector) PEpos∈RdmodelPE\_{pos} \in \mathbb{R}^{d\_{model}}PEpos​∈Rdmodel​。虽然在极长序列中理论上可能发生冲突，但在典型的模型限制下，这种情况实际上不存在。

### 有界值

正弦和余弦函数自然生成值在固定范围 [−1,1][-1, 1][−1,1] 内。当这些位置编码 (positional encoding)添加到词元 (token)嵌入 (embedding)中（词元嵌入通常也在可控范围内，通常通过归一化 (normalization)或初始化）时，这种有界性可以避免位置信息大幅改变组合嵌入的幅度。与可能无界的位置信号相比，这有助于更稳定的训练过程。

### 相对位置表示

正弦编码最重要的优点，或许是其固有的通过线性变换表示相对位置的能力。考虑位置 pos+kpos+kpos+k 的编码。使用三角函数和角公式：

sin⁡(a+b)=sin⁡(a)cos⁡(b)+cos⁡(a)sin⁡(b)\sin(a+b) = \sin(a)\cos(b) + \cos(a)\sin(b)sin(a+b)=sin(a)cos(b)+cos(a)sin(b)
cos⁡(a+b)=cos⁡(a)cos⁡(b)−sin⁡(a)sin⁡(b)\cos(a+b) = \cos(a)\cos(b) - \sin(a)\sin(b)cos(a+b)=cos(a)cos(b)−sin(a)sin(b)

令 ωi=1/100002i/dmodel\omega\_i = 1 / 10000^{2i/d\_{model}}ωi​=1/100002i/dmodel​。PEpos+kPE\_{pos+k}PEpos+k​ 的分量可以表示为 PEposPE\_{pos}PEpos​ 的形式：

PE(pos+k,2i)=sin⁡((pos+k)ωi)=sin⁡(pos⋅ωi)cos⁡(k⋅ωi)+cos⁡(pos⋅ωi)sin⁡(k⋅ωi)PE\_{(pos+k, 2i)} = \sin((pos+k)\omega\_i) = \sin(pos \cdot \omega\_i)\cos(k \cdot \omega\_i) + \cos(pos \cdot \omega\_i)\sin(k \cdot \omega\_i)PE(pos+k,2i)​=sin((pos+k)ωi​)=sin(pos⋅ωi​)cos(k⋅ωi​)+cos(pos⋅ωi​)sin(k⋅ωi​)
=PE(pos,2i)cos⁡(k⋅ωi)+PE(pos,2i+1)sin⁡(k⋅ωi)= PE\_{(pos, 2i)}\cos(k \cdot \omega\_i) + PE\_{(pos, 2i+1)}\sin(k \cdot \omega\_i)=PE(pos,2i)​cos(k⋅ωi​)+PE(pos,2i+1)​sin(k⋅ωi​)

PE(pos+k,2i+1)=cos⁡((pos+k)ωi)=cos⁡(pos⋅ωi)cos⁡(k⋅ωi)−sin⁡(pos⋅ωi)sin⁡(k⋅ωi)PE\_{(pos+k, 2i+1)} = \cos((pos+k)\omega\_i) = \cos(pos \cdot \omega\_i)\cos(k \cdot \omega\_i) - \sin(pos \cdot \omega\_i)\sin(k \cdot \omega\_i)PE(pos+k,2i+1)​=cos((pos+k)ωi​)=cos(pos⋅ωi​)cos(k⋅ωi​)−sin(pos⋅ωi​)sin(k⋅ωi​)
=PE(pos,2i+1)cos⁡(k⋅ωi)−PE(pos,2i)sin⁡(k⋅ωi)= PE\_{(pos, 2i+1)}\cos(k \cdot \omega\_i) - PE\_{(pos, 2i)}\sin(k \cdot \omega\_i)=PE(pos,2i+1)​cos(k⋅ωi​)−PE(pos,2i)​sin(k⋅ωi​)

这可以表示为每对维度 (2i,2i+1)(2i, 2i+1)(2i,2i+1) 的矩阵乘法：

(PE(pos+k,2i)PE(pos+k,2i+1))=(cos⁡(kωi)sin⁡(kωi)−sin⁡(kωi)cos⁡(kωi))(PE(pos,2i)PE(pos,2i+1))\begin{pmatrix} PE\_{(pos+k, 2i)} \\ PE\_{(pos+k, 2i+1)} \end{pmatrix}
=
\begin{pmatrix} \cos(k \omega\_i) & \sin(k \omega\_i) \\ -\sin(k \omega\_i) & \cos(k \omega\_i) \end{pmatrix}
\begin{pmatrix} PE\_{(pos, 2i)} \\ PE\_{(pos, 2i+1)} \end{pmatrix}(PE(pos+k,2i)​PE(pos+k,2i+1)​​)=(cos(kωi​)−sin(kωi​)​sin(kωi​)cos(kωi​)​)(PE(pos,2i)​PE(pos,2i+1)​​)

这表明 PEpos+kPE\_{pos+k}PEpos+k​ 的位置编码 (positional encoding)是 PEposPE\_{pos}PEpos​ 的一个线性函数（具体来说，是一个旋转）。变换矩阵仅取决于偏移量 kkk，而不是绝对位置 pospospos。这个特性使得自注意力 (self-attention)机制 (attention mechanism)（它涉及线性投影（查询、键、值）和点积）更容易学习根据词元 (token)之间的相对距离进行注意力计算。模型不需要为位置 5 上的 +2 偏移量和位置 50 上的 +2 偏移量学习单独的规则；这种关系被一致地编码。

### 平滑插值

正弦函数随位置平滑变化。这意味着相邻位置 pospospos 和 pos+1pos+1pos+1 的位置编码 (positional encoding)是相似的，这体现了相邻词语通常具有紧密相关的上下文 (context)作用的直觉。这种平滑变化与结构性较差的编码方案可能发生的突变形成对比。

### 外推能力

由于正弦编码由固定函数生成，而非从固定序列长度范围内的数据中学习，它们在处理比训练期间遇到的更长序列时提供了优势。该函数可以为任何位置 pospospos 生成编码。虽然模型在处理更长序列时的整体性能可能仍会因其他因素（如注意力模式无法泛化）而下降，但位置编码 (positional encoding)机制本身不会出现固有故障，也不会为未见过的位置产生未定义的值，这与学习型嵌入 (embedding)不同，后者会缺乏超出训练最大值的位置的表示。

### 编码模式可视化

频率选择（ωi=1/100002i/dmodel\omega\_i = 1 / 10000^{2i/d\_{model}}ωi​=1/100002i/dmodel​）产生的信号从高频（对于小 iii，随位置快速变化）到极低频（对于大 iii，在整个序列中缓慢变化）不等。这使得模型能够以不同粒度捕捉位置信息。

0510152025012345−1−0.500.51

> 热力图显示了128维嵌入 (embedding)（`d_model=128`）中，前30个位置（`pos`）和前6个维度（`d=0`到`d=5`）的正弦位置编码 (positional encoding)值。请注意，较低维度（顶部行）的振荡速度更快，而较高维度（底部行）的振荡速度更慢。

总而言之，正弦位置编码提供了一种简单而有效的无参数 (parameter)方法，可将序列顺序信息注入Transformer。它们的数学特性与注意力机制 (attention mechanism)在词元 (token)间建模关系的能力良好匹配，尤其是在相对位置方面，同时保持稳定性并为泛化到更长序列提供了可能。

获取即时帮助、个性化解释和交互式代码示例。

---

### 结合嵌入与位置编码

# 结合嵌入与位置编码

将输入令牌表示为密集向量 (vector) (dense vector)（令牌嵌入 (embedding)）以及生成表示其位置的独特信号（位置编码 (positional encoding)）是Transformer模型处理输入的一个基本步骤。这两种信息源需要被整合。自注意力 (self-attention)机制 (attention mechanism)是Transformer的核心组件，在处理输入元素时本身不了解它们的顺序。因此，我们必须将这种位置信息直接提供给输入到Transformer堆栈的表示中。

原始Transformer论文（《Attention Is All You Need》）中提出的一种标准且非常有效的方法十分直接：逐元素相加。如果 E∈RL×dmodelE \in \mathbb{R}^{L \times d\_{model}}E∈RL×dmodel​ 表示长度为 LLL 的序列的令牌嵌入矩阵，且 P∈RL×dmodelP \in \mathbb{R}^{L \times d\_{model}}P∈RL×dmodel​ 表示对应的位置编码，那么组合后的输入表示 X∈RL×dmodelX \in \mathbb{R}^{L \times d\_{model}}X∈RL×dmodel​ 的计算方式如下：

X=E+PX = E + PX=E+P

这表示序列中每个令牌 iii（iii 的范围从 000 到 L−1L-1L−1），其最终输入向量 xix\_ixi​ 是其令牌嵌入 eie\_iei​ 和其位置编码 pip\_ipi​ 的和：

xi=ei+pix\_i = e\_i + p\_ixi​=ei​+pi​

在此，令牌嵌入 eie\_iei​ 和位置编码 pip\_ipi​ 都必须具有相同的维度，dmodeld\_{model}dmodel​。这种维度上的一致性是加法运算的基本要求。

### 加法的理由

为什么是简单的加法？尽管可以设想其他组合函数，但加法具有多项优势：

1. **简洁和效率：** 它计算成本低，只增加很小的额外开销。
2. **信息保留：** 加法使得网络能够获取语义信息（来自 EEE）和位置信息（来自 PPP）。由于后续层涉及线性变换（如多头注意力 (multi-head attention)中的Q、K、V投影），它们理论上能够学习以所需方式投影组合后的嵌入 (embedding) XXX，从而分离或使用与 EEE 或 PPP 相关的部分。
3. **与正弦编码的兼容性：** 对于正弦编码，加法保留了它们设计用于捕获的相对位置信息。因为正弦函数具有与相对位置的线性变换相关的属性（例如，PEpos+kPE\_{pos+k}PEpos+k​ 可以表示为 PEposPE\_{pos}PEpos​ 的线性函数），将它们添加到令牌嵌入中，为模型提供了学习相对距离的一致方式。

### 实现流程

实际操作中，组合嵌入 (embedding)和位置编码 (positional encoding)通常涉及以下步骤：

1. **令牌嵌入查找：** 输入令牌ID通过嵌入层（通常是一个可训练的查找表）映射到其对应的嵌入向量 (vector)。这会得到一个形状为 `[batch_size, sequence_length, d_model]` 的张量。
2. **位置编码生成/查找：**
   - 对于正弦编码，它们通常会预先计算到最大预期序列长度并存储起来。然后选择当前序列长度的相应编码。形状通常是 `[1, sequence_length, d_model]` 或 `[sequence_length, d_model]`，可以进行广播或切片。
   - 对于学习到的位置嵌入，则使用一个单独的嵌入层（查找表），通过位置ID（0, 1, 2,...）进行索引。这也会得到一个形状为 `[1, sequence_length, d_model]` 或类似的张量，其中包含可训练向量。
3. **逐元素相加：** 位置编码被添加到令牌嵌入中。如果位置编码张量不明确包含批次维度，这里通常会应用广播规则。

G

clusterᵢnput

输入处理

cluster\_combination

组合

clusterₒutput

到Transformer层

tokens

令牌ID

[T1, T2, ..., TL]

embeddings

令牌嵌入 (E)

查找

[batch, L, dₘodel]

tokens->embeddings

嵌入层

addₒp

{加法 (+) | 逐元素}

embeddings->addₒp

positions

位置编码 (P)

正弦或学习

[1, L, dₘodel]

positions->addₒp

combined

组合输入 (X)

X = E + P

[batch, L, dₘodel]

addₒp->combined

transformerᵢnput

第一层的输入

combined->transformerᵢnput

> 流程图，说明了通过逐元素相加组合令牌嵌入和位置编码，从而创建Transformer层的输入表示。

### 学习到的位置嵌入 (embedding)的考虑事项

如果使用学习到的位置嵌入而不是正弦嵌入，组合机制保持不变：逐元素相加。主要区别在于，表示位置 0,1,2,…0, 1, 2, \dots0,1,2,… 的向量 (vector)现在是在训练过程中优化的参数 (parameter)，而不是由固定函数确定。模型会根据数据学习对任务最有效的表示。加法步骤仍然将这些学习到的位置信号与语义令牌嵌入合并。

通过将位置编码 (positional encoding)直接添加到令牌嵌入中，我们创建了一个输入表示 XXX，它为后续的Transformer层提供了精密的序列理解所需的“什么”（来自 EEE 的语义）和“哪里”（来自 PPP 的序列顺序）信息。这个组合表示构成了编码器或解码器堆栈第一层的输入。

获取即时帮助、个性化解释和交互式代码示例。

---

### 替代方案：学习型位置嵌入

# 替代方案：学习型位置嵌入

正弦位置编码 (positional encoding)提供了一种巧妙且固定的方法，用于注入序列顺序信息。然而，这并非唯一的方法。另一种策略，被 BERT 和最初的 GPT 模型等多个有影响力的架构所采纳，是直接*学习*位置嵌入 (embedding)，这与词嵌入的习得方式类似。

### 学习型位置嵌入 (embedding)的原理

不再依赖预定义的数学函数（正弦波和余弦波），我们可以将序列中的每个位置视为一个独立单元，它需要自己的向量 (vector)表示。核心思想简单明了：

1. 定义一个模型预期能处理的最大可能序列长度，我们称之为 LmaxL\_{max}Lmax​。这通常在模型设计时确定（例如，512、1024、2048）。
2. 创建一个标准嵌入矩阵 PPP，其形状为 (Lmax,dmodel)(L\_{max}, d\_{model})(Lmax​,dmodel​)，其中 dmodeld\_{model}dmodel​ 是嵌入维度（与词元 (token)嵌入维度相同）。
3. 此矩阵中的每一行 PiP\_iPi​ 代表序列中位置 iii 的学习型嵌入向量，其中 0≤i<Lmax0 \le i < L\_{max}0≤i<Lmax​。
4. 这些位置嵌入会被初始化（通常随机初始化或采用某种结构化初始化），然后通过反向传播 (backpropagation)与模型的其他参数 (parameter)一起训练。

### 与词元 (token)嵌入 (embedding)的整合

整合过程与正弦编码相似。给定一个长度为 LLL 的输入序列（其中 L≤LmaxL \le L\_{max}L≤Lmax​），我们首先获取词元嵌入 Etoken∈RL×dmodelE\_{token} \in \mathbb{R}^{L \times d\_{model}}Etoken​∈RL×dmodel​。然后我们从矩阵 PPP 中查找前 LLL 个位置对应的学习型位置嵌入，得到 Eposition=[P0,P1,...,PL−1]∈RL×dmodelE\_{position} = [P\_0, P\_1, ..., P\_{L-1}] \in \mathbb{R}^{L \times d\_{model}}Eposition​=[P0​,P1​,...,PL−1​]∈RL×dmodel​。

输入到第一个 Transformer 层的最终表示通常是词元嵌入与学习型位置嵌入的和：

Einput=Etoken+EpositionE\_{input} = E\_{token} + E\_{position}Einput​=Etoken​+Eposition​

这种逐元素相加的方式使得模型能够结合词元的语义含义与其在序列中的绝对位置。

### 优点

- **灵活性：** 主要优点是灵活性。模型不受正弦和余弦等固定函数形式的约束。它理论上可以学习最适合其训练的特定任务和数据集的位置表示。如果某些位置或位置关系对任务来说特别重要，模型有能力在训练时编码这些信息。
- **简洁性：** 实现可以说比计算正弦值更简单。它使用了深度学习 (deep learning)框架中已有的标准嵌入 (embedding)查找机制。

### 缺点

- **外推限制：** 学习型嵌入 (embedding)在处理训练期间遇到的长度超过 LmaxL\_{max}Lmax​ 的序列时存在根本性问题。由于没有针对 LmaxL\_{max}Lmax​ 及更长位置的学习型嵌入向量 (vector)，模型无法在不进行重新训练或采用临时策略的情况下直接处理更长的序列。正弦编码，由于基于函数，可以为任意长度的序列生成编码。
- **参数 (parameter)量增加：** 这种方法会为模型引入 Lmax×dmodelL\_{max} \times d\_{model}Lmax​×dmodel​ 个额外的可训练参数。对于较大的 LmaxL\_{max}Lmax​ 和 dmodeld\_{model}dmodel​ 值，这会显著增加模型大小、内存需求，并可能增加过拟合 (overfitting)的风险，尤其是在训练数据有限的情况下。
- **数据依赖性：** 从头开始学习有效的位置表示需要足够的训练数据。在数据有限的场景中，正弦编码的固定结构可能提供更强的归纳偏置 (bias)，从而带来更好的泛化能力。

### 学习型编码与正弦编码的选择

学习型和固定正弦位置编码 (positional encoding)之间的选择通常取决于具体的应用、模型架构理念以及可用数据。

- **学习型嵌入 (embedding)：** 通常在最大序列长度明确且不过分大，并且有足够数据来学习有意义的表示时更受青睐。像 BERT 这样通常在海量数据集上进行预训练 (pre-training)的架构，普遍使用学习型嵌入。
- **正弦编码：** 当处理可变或可能非常长的序列的能力很重要时，或者当关注参数 (parameter)效率时，正弦编码是一个不错的选择。最初的 Transformer 论文采用了正弦编码，它们仍然是可行且有效的选择。

归根结底，两种方法目的相同：为排列不变的自注意力 (self-attention)机制 (attention mechanism)提供关于输入序列中元素顺序的重要信息。下一节将比较它们的特性。

获取即时帮助、个性化解释和交互式代码示例。

---

### 比较：正弦式与学习式嵌入

# 比较：正弦式与学习式嵌入

在了解了使用固定三角函数注入序列顺序信息的正弦式位置编码 (positional encoding)方案后，我们现在转而关注另一种方法：学习式位置嵌入 (embedding)。这两种方法都旨在解决同一个问题——使置换不变的自注意力 (self-attention)机制 (attention mechanism)感知序列顺序——但它们具有不同的特点和权衡。理解这些差异对于为特定任务或模型架构选择合适的方法很重要。

让我们从几个方面系统地比较这两种技术。

### 参数 (parameter)效率与模型大小

**正弦式位置编码 (positional encoding)：** 这种方法无参数。编码向量 (vector)通过预定义公式生成：
PE(pos,2i)=sin⁡(pos/100002i/dmodel)PE\_{(pos, 2i)} = \sin(pos / 10000^{2i/d\_{model}})PE(pos,2i)​=sin(pos/100002i/dmodel​)
PE(pos,2i+1)=cos⁡(pos/100002i/dmodel)PE\_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d\_{model}})PE(pos,2i+1)​=cos(pos/100002i/dmodel​)
这些值可以预先计算到最大预期序列长度并存储，也可以在正向传播时动态计算。无论哪种情况，它们都不会为模型增加任何可训练参数。这使得它们对于参数数量受限的模型很有吸引力。

**学习式位置嵌入 (embedding)：** 这种方法将位置视为离散标记 (token)。会创建一个嵌入矩阵，通常大小为 Lmax×dmodelL\_{max} \times d\_{model}Lmax​×dmodel​，其中 LmaxL\_{max}Lmax​ 是模型设计用于处理的最大序列长度，dmodeld\_{model}dmodel​ 是嵌入维度（与标记嵌入维度匹配）。该矩阵中的每一行 jjj 都是一个表示位置 jjj 的可学习向量。在正向传播期间，根据位置索引查找相应的学习式位置嵌入向量，并将其添加到对应的标记嵌入中。这会为模型引入 Lmax×dmodelL\_{max} \times d\_{model}Lmax​×dmodel​ 个额外的可训练参数。对于具有非常大上下文 (context)窗口（大 LmaxL\_{max}Lmax​）或高维度（dmodeld\_{model}dmodel​）的模型，这可能表示模型大小和内存需求的显著增加。

### 灵活性与归纳偏置 (bias)

**正弦式位置编码 (positional encoding)：** 正弦编码的固定性质为模型引入了强的*归纳偏置*。三角函数固有地编码位置，有助于建模相对位置。由于 sin⁡(α+β)\sin(\alpha+\beta)sin(α+β) 和 cos⁡(α+β)\cos(\alpha+\beta)cos(α+β) 等三角恒等式，位置偏移 kkk 的编码 PEpos+kPE\_{pos+k}PEpos+k​ 可以表示为 PEposPE\_{pos}PEpos​ 的线性变换。这种结构可能使自注意力 (self-attention)机制 (attention mechanism)更容易学习基于相对偏移的关联，独立于绝对位置。然而，这种固定结构可能并非适用于所有任务或数据分布。

**学习式位置嵌入 (embedding)：** 其可学习的特性提供了最大的灵活性。原则上，模型可以学习任何能最小化目标任务训练损失的位置表示。这使得模型能够根据数据的独特之处调整位置信息。然而，这种灵活性代价是较弱的归纳偏置。由于没有正弦函数的固有结构，模型必须完全从数据中学习位置关联，这可能需要更多数据或更长的训练时间，尤其是在捕捉复杂相对位置模式时。此外，还存在学习到的嵌入无法获得像正弦函数那样平滑或更具泛化性的表示的风险。

### 外推到未见序列长度

**正弦式位置编码 (positional encoding)：** 数学公式允许正弦编码生成任何位置索引的编码，即使是超出训练期间遇到的最大长度的位置。这表明在*外推*到更长序列方面具有潜在优势。尽管模型的注意力机制 (attention mechanism)本身并未在这些更长的交互上训练，但位置信号仍然定义明确。这种外推的实际效果可能有所不同，因为模型性能仍可能因分布偏移而在显著更长的序列上下降，但编码机制本身不会失效。

**学习式位置嵌入 (embedding)：** 这种方法通常难以进行外推。由于嵌入只学习了截至 LmaxL\_{max}Lmax​ 的位置，模型对于 Lmax,Lmax+1,…L\_{max}, L\_{max}+1, \dotsLmax​,Lmax​+1,… 等位置没有定义好的表示。在推理 (inference)时简单地延长序列将需要分配零向量 (vector)、重用现有向量，或采用嵌入插值或使用更长序列重新训练/微调 (fine-tuning)模型等更复杂的技术。当遇到长度超过 LmaxL\_{max}Lmax​ 的序列时，标准实现通常会失败或产生不可预测的结果。

### 经验表现

**正弦式位置编码 (positional encoding)：** 在最初的“Attention Is All You Need”Transformer中采用，它们在机器翻译任务上表现出色。它们通常提供一个可靠的基线，并在各种序列建模问题上表现良好。

**学习式位置嵌入 (embedding)：** 被BERT和GPT系列等许多有影响力的模型采用。经验结果表明，学习式嵌入可以达到先进水平的性能，表明它们提供的灵活性允许模型在有足够数据和模型容量的情况下有效地捕获位置信息。

在实践中，对于训练期间常见的序列长度，两种方法之间的性能差异通常很小。选择可能更多取决于参数 (parameter)预算、所需的外推能力或特定的架构选择（例如，后面会讨论的相对位置编码会改变这种比较）。

### 实现复杂度

两种方法在现代深度学习 (deep learning)框架中实现起来都相对简单。

**正弦式：** 需要实现数学公式，通常涉及矩阵运算以提高效率。需要注意数值稳定性，并正确广播编码以匹配输入批次维度。

**学习式：** 通常更简单，通常只需实例化一个嵌入 (embedding)层（例如，`torch.nn.Embedding` 或 `tf.keras.layers.Embedding`）并将其输出添加到标记 (token)嵌入中。

### 总结表

| 特性 | 正弦式位置编码 (positional encoding) | 学习式位置嵌入 (embedding) |
| --- | --- | --- |
| **参数 (parameter)** | 无 (固定函数) | 增加 Lmax×dmodelL\_{max} \times d\_{model}Lmax​×dmodel​ 个参数 |
| **灵活性** | 较低 (固定结构) | 较高 (从数据中学习) |
| **归纳偏置 (bias)** | 强 (鼓励相对定位) | 较弱 (必须从头学习位置关联) |
| **外推能力** | 理论上可行，生成有效编码 | 较差，对未见位置需要特殊处理 |
| **最大长度限制** | 由计算定义，而非参数 | 训练时基于 LmaxL\_{max}Lmax​ 的硬性约束 |
| **数据需求** | 由于内置结构，潜在地对数据需求较少 | 可能需要更多数据来学习有效表示 |
| **典型应用** | 原始Transformer | BERT, GPT系列 |
| **实现** | 需要实现公式 | 通常涉及标准嵌入层 |

### 结论

正弦式和学习式位置嵌入 (embedding)之间的选择涉及归纳偏置 (bias)、参数 (parameter)效率、灵活性和外推能力之间的权衡。正弦式编码提供了一种无参数、结构化的方式来表示位置，在相对定位和外推方面具有良好的理论特性。学习式嵌入提供更大的灵活性，允许模型根据特定任务调整位置表示，但它们会增加参数，并且难以处理比训练期间遇到的序列更长的序列。最佳选择通常取决于模型的具体要求、数据规模和所需的序列长度处理能力。正如我们将在后续章节中看到的，相对位置编码 (positional encoding)等发展旨在结合结构化位置感知与依赖上下文 (context)表示的优势。

获取即时帮助、个性化解释和交互式代码示例。

---

### 实践：生成与可视化位置编码

# 实践：生成与可视化位置编码

正弦位置编码 (positional encoding)对于向Transformer告知序列顺序很重要。直观了解这些编码的特性可以带来有益的直观认识。在此，将实现位置编码函数并可视化生成的向量 (vector)。

我们将使用Python配合NumPy进行数值计算，并使用Plotly进行交互式可视化，这非常适合基于网络的课程材料。

### 实现位置编码 (positional encoding)函数

首先，让我们将正弦位置编码的数学公式转换为代码。回顾这些公式：

PE(位置,2i)=sin⁡(位置/100002i/dmodel)PE\_{(位置, 2i)} = \sin(位置 / 10000^{2i/d\_{model}})PE(位置,2i)​=sin(位置/100002i/dmodel​)
PE(位置,2i+1)=cos⁡(位置/100002i/dmodel)PE\_{(位置, 2i+1)} = \cos(位置 / 10000^{2i/d\_{model}})PE(位置,2i+1)​=cos(位置/100002i/dmodel​)

此处 `pos` 是序列中的位置，iii 是嵌入 (embedding)向量 (vector)中维度的索引，dmodeld\_{model}dmodel​ 是嵌入的维度。

下面是一个使用NumPy生成这些编码的Python函数：

```python
import numpy as np

def get_positional_encoding(max_seq_len, d_model):
    """
    生成正弦位置编码。

    Args:
        max_seq_len: 最大序列长度。
        d_model: 模型嵌入的维度。

    Returns:
        一个形状为 (max_seq_len, d_model) 的NumPy数组，包含
        位置编码。
    """
    if d_model % 2 != 0:
        raise ValueError("d_model 必须是偶数才能容纳正弦/余弦对。")

    pos_encoding = np.zeros((max_seq_len, d_model))

    position = np.arange(max_seq_len)[:, np.newaxis]

    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pos_encoding[:, 0::2] = np.sin(position * div_term)

    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return pos_encoding

max_len = 50
d_model = 128

positional_encodings = get_positional_encoding(max_len, d_model)

print(f"生成的位置编码形状：{positional_encodings.shape}")
```

此函数接收最大序列长度和模型的嵌入维度作为输入。它计算偶数索引的正弦值和奇数索引的余弦值，基于位置和表示频率成分的 `div_term`。结果是一个矩阵，其中每行对应序列中的一个位置，每列对应位置编码向量中的一个维度。

### 可视化位置编码 (positional encoding)

可视化这个矩阵有助于理解这些编码的结构。热力图是查看编码值如何跨位置和维度变化的有效方式。我们将生成序列长度为50、嵌入 (embedding)维度为128的编码。

020406080100120−0.500.511.522.5−0.8−0.6−0.4−0.200.20.40.60.81

> 热力图可视化了长度为50、嵌入维度为128的序列的正弦位置编码。每行代表一个位置，每列代表一个维度索引。颜色强度指示编码值。

### 分析可视化结果

从热力图中，之前讨论的几个特性变得直观显现：

1. **每个位置的独特编码：** 每行（位置）都有独特的颜色模式，代表其独特的编码向量 (vector)。这种独特性使得模型能够区分不同位置。
2. **变化的频率：** 观察维度轴（X轴）上的波长。
   - 最左侧的列（低维度索引，小 iii）呈现高频变化（沿位置轴快速的颜色变化）。这些维度编码细粒度的位置信息。
   - 最右侧的列（高维度索引，大 iii）显示低频变化（缓慢的颜色变化）。这些维度编码较粗略的、长距离的位置信息。
3. **平滑过渡：** 正弦特性确保了相邻位置编码 (positional encoding)之间的平滑过渡。
4. **有界值：** 由于正弦和余弦函数的作用，所有值本身都在 [-1, 1] 的范围内。

让我们通过绘制几个特定位置（例如位置0、10和25）在所有维度上的编码向量来进一步检查这种独特性。

0102030405060−0.500.51位置 0位置 10位置 25

> 线图比较了位置0、10和25的128维位置编码向量。每条线独特的形状突出了分配给每个序列位置的独特编码。

这些可视化结果证实了正弦位置编码为每个位置提供了独特的信号，并在不同频率的维度上平滑变化。随后，这个位置信号被添加到输入词元 (token)嵌入 (embedding)中，使后续的自注意力 (self-attention)层能够考虑序列中元素的顺序。

在下一章中，我们将组装这些组件以及多头注意力 (multi-head attention)机制 (attention mechanism)，构成完整的Transformer编码器和解码器堆栈。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 5 Encoder Decoder Stacks

### Transformer 整体架构概览

# Transformer 整体架构概览

自注意力 (self-attention)机制 (attention mechanism)和位置编码 (positional encoding)等基本组件被组装成完整的 Transformer 架构。最初的 Transformer 模型，在论文《Attention Is All You Need》中提出，采用了编码器-解码器结构，这在机器翻译或文本摘要等序列到序列任务中是一种常见模式。与逐步处理序列的循环模型不同，Transformer 使用注意力机制同时处理整个输入序列，捕获不同距离的关联。

该架构包含两个主要部分：一个编码器堆栈和一个解码器堆栈。

G

clusterᵢnput

输入处理

clusterₒutput

输出处理

Input\_Tokens

输入词元

Input\_Embed

输入嵌入

Input\_Tokens->Input\_Embed

Add\_Norm\_In

+

Input\_Embed->Add\_Norm\_In

Pos\_Encode\_In

位置编码

Pos\_Encode\_In->Add\_Norm\_In

Encoder\_Nx

编码器层 (Nx)

Add\_Norm\_In->Encoder\_Nx

Decoder\_Nx

解码器层 (Nx)

Encoder\_Nx->Decoder\_Nx

 编码器
输出 (K, V)

Linear

线性层

Decoder\_Nx->Linear

Output\_Tokens

输出词元
(右移)

Output\_Embed

输出嵌入

Output\_Tokens->Output\_Embed

Add\_Norm\_Out

+

Output\_Embed->Add\_Norm\_Out

Pos\_Encode\_Out

位置编码

Pos\_Encode\_Out->Add\_Norm\_Out

Add\_Norm\_Out->Decoder\_Nx

 解码器
输入 (Q)

Softmax

Softmax

Linear->Softmax

Output\_Probs

输出概率

Softmax->Output\_Probs

> Transformer 架构的整体示意图，展示了从输入词元 (token)通过编码器和解码器堆栈到输出概率的数据流向。注意承载着编码器输出到解码器的连接。

### 编码器堆栈

编码器的作用是处理整个输入序列，并生成一系列编码了输入信息的连续表示（带有上下文 (context)信息的嵌入 (embedding)）。它由堆叠的 NNN 个相同层组成（在原始论文中通常 N=6N=6N=6）。每个层有两个主要子层：

1. 一个多头自注意力 (self-attention)机制 (attention mechanism)。
2. 一个逐位置全连接前馈网络。

在每个子层周围都使用残差连接，之后是层归一化 (normalization)。这意味着每个子层的输出是 LayerNorm(x+Sublayer(x))LayerNorm(x + Sublayer(x))LayerNorm(x+Sublayer(x))，其中 Sublayer(x)Sublayer(x)Sublayer(x) 是子层本身实现的功能（例如，多头注意力 (multi-head attention)或前馈网络）。自注意力机制允许编码器中的每个位置关注前一层输出中的所有位置，有效捕获输入序列内的关联。前馈网络独立应用于每个位置。

### 解码器堆栈

解码器的作用是生成输出序列，通常以自回归 (autoregressive)方式逐个元素生成。与编码器类似，它也由堆叠的 NNN 个相同层组成。然而，每个解码器层有三个主要子层：

1. 一个*掩码*多头自注意力 (self-attention)机制 (attention mechanism)。
2. 一个关注编码器堆栈输出的多头*交叉注意力*机制。
3. 一个逐位置全连接前馈网络。

与编码器中一样，在每个子层周围都应用了残差连接和层归一化 (normalization)。*掩码*自注意力确保对位置 iii 的预测只能依赖于位置小于 iii 的已知输出，保留了生成所需的自回归属性。*交叉注意力*机制是序列到序列功能的核心：它允许解码器中的每个位置关注输入序列中的所有位置（通过编码器的输出表示）。

### 连接编码器和解码器

编码器和解码器之间的连接主要通过每个解码器层中的交叉注意力机制 (attention mechanism)实现。整个编码器堆栈首先处理输入序列，生成一系列输出向量 (vector) z=(z1,...,zn)z = (z\_1, ..., z\_n)z=(z1​,...,zn​)。然后，这些向量 zzz 被用作*每个*解码器层中交叉注意力子层中\*\*键（K）**和**值（V）**的来源。该交叉注意力层的**查询（Q）\*\*来自解码器内部的前一个子层（掩码自注意力 (self-attention)层）的输出。这使得解码器在生成输出序列的每一步都能关注编码在 zzz 中的输入序列的相关部分。

### 输入和输出处理

在输入序列进入编码器堆栈之前，输入词元 (token)使用嵌入 (embedding)层转换为向量 (vector)，并将位置编码 (positional encoding)添加到这些嵌入中，以注入序列顺序信息。同样，对于解码器，目标输出词元（训练时右移，推理 (inference)时为先前生成的词元）在进入解码器堆栈之前会进行嵌入并与位置编码结合。

在最终解码器层生成其输出向量后，通常会使用最后的线性变换，接着是 Softmax 函数，将这些向量转换为目标词汇表 (vocabulary)上的概率分布，从而预测输出序列中的下一个词元。

这种整体结构建立在包含注意力机制 (attention mechanism)、残差连接和层归一化 (normalization)的堆叠层之上，构成了 Transformer 模型的基础。后续章节将更详细地分析每个组件，例如编码器和解码器层的具体结构、掩码注意力、交叉注意力、前馈网络和归一化。

获取即时帮助、个性化解释和交互式代码示例。

---

### 编码器层结构

# 编码器层结构

编码器堆栈由多个相同层组成，常表示为 NNN 层（例如，在原始 Transformer 论文中 N=6N=6N=6）。每个编码器层的主要作用是将输入嵌入 (embedding)序列转换为语境化表示序列。这些表示纳入了整个输入序列的信息，使模型能够理解每个元素在其语境中的作用。

单个编码器层由两个主要子层构成：

1. 一个多头自注意力 (self-attention)机制 (attention mechanism)。
2. 一个简单的、按位置全连接的前馈网络（FFN）。

尤其重要的是，每个子层都包裹在残差连接和层归一化 (normalization)之中。这种“添加与归一化”模式是 Transformer 架构的一个典型特点，对于有效训练深度模型来说非常必要。我们来查看单个编码器层内的结构和数据流。

### 数据流经编码器层

假设编码器层的输入是向量 (vector)序列 X=(x1,x2,...,xn)X = (x\_1, x\_2, ..., x\_n)X=(x1​,x2​,...,xn​)，且 nnn 是序列长度，每个 xix\_ixi​ 是一个向量（例如，对于第一层，是词元 (token)嵌入 (embedding)和位置编码 (positional encoding)的和，或前一个编码器层的输出）。

1. **多头自注意力 (self-attention)：** 输入序列 XXX 首先通过多头自注意力子层。如第三章所述，这种机制使序列中的每个位置 iii 能够关注序列 XXX 中的所有位置（包括自身）。它根据从 XXX 本身得到的查询、键和值计算注意力分数，产生一个输出序列，其中每个向量是值向量的加权和，反映语境信息。将此子层的输出表示为 MultiHeadAttention(X)\text{MultiHeadAttention}(X)MultiHeadAttention(X)。
2. **添加与归一化 (normalization)（第一块）：** 自注意力子层的输出随后通过残差连接（相加）与原始输入 XXX 结合。这有助于避免深度网络中的梯度消失问题，通过允许梯度直接流经网络。为了正则化 (regularization)，通常在加法步骤*之前*将 Dropout 应用于自注意力子层的输出。加法之后，应用层归一化。层归一化稳定激活并改善训练动态。此块的操作可以表示为：

   子层输出1=层归一化(X+Dropout(多头注意力(X)))\text{子层输出}\_1 = \text{层归一化}(X + \text{Dropout}(\text{多头注意力}(X)))子层输出1​=层归一化(X+Dropout(多头注意力(X)))

   结果 子层输出1\text{子层输出}\_1子层输出1​ 是一个与 XXX 维度相同的中间表示序列。
3. **按位置前馈网络（FFN）：** 该中间序列 子层输出1\text{子层输出}\_1子层输出1​ 随后输入到按位置前馈网络。该网络由两个线性变换构成，其间带有一个激活函数 (activation function)（通常是 ReLU 或 GELU）。重要的是，相同的 FFN（具有相同的权重 (weight)）独立地应用于序列 子层输出1\text{子层输出}\_1子层输出1​ 中的每个位置 iii。它提供非线性变换，进一步处理表示。将输出表示为 FFN(子层输出1)\text{FFN}(\text{子层输出}\_1)FFN(子层输出1​)。其结构通常是：

   FFN(z)=max⁡(0,zW1+b1)W2+b2\text{FFN}(z) = \max(0, zW\_1 + b\_1)W\_2 + b\_2FFN(z)=max(0,zW1​+b1​)W2​+b2​

   其中 zzz 是特定位置的输入向量，而 W1,b1,W2,b2W\_1, b\_1, W\_2, b\_2W1​,b1​,W2​,b2​ 是两个线性层的可学习参数 (parameter)。内部维度通常大于模型的嵌入维度 dmodeld\_{model}dmodel​。
4. **添加与归一化（第二块）：** 与第一个子层相似，FFN 的输出通过残差连接与它的输入（子层输出1\text{子层输出}\_1子层输出1​）结合，随后是 dropout 和层归一化。

   层输出=层归一化(子层输出1+Dropout(FFN(子层输出1)))\text{层输出} = \text{层归一化}(\text{子层输出}\_1 + \text{Dropout}(\text{FFN}(\text{子层输出}\_1)))层输出=层归一化(子层输出1​+Dropout(FFN(子层输出1​)))

   层输出\text{层输出}层输出 是此编码器层的最终向量输出序列。此输出与输入 XXX 具有相同的维度，并作为堆栈中下一个相同编码器层的输入。

以下图表说明了一个遵循此描述的单个编码器层结构（通常称为 Post-LN，即归一化发生在加法*之后*）：

G

clusterₑncoderₗayer

编码器层

clusterₛublayer1

子层 1

clusterₛublayer2

子层 2

Input

输入 (X 或 第 i-1 层输出)

MHA

多头
自注意力

Input->MHA

Add1

相加

Input->Add1

残差

Output

输出 (第 i 层输出)

MHA->Add1

 Dropout(·)

Norm1

层归一化

Add1->Norm1

FFN

按位置
前馈网络

Norm1->FFN

Add2

相加

Norm1->Add2

FFN->Add2

 Dropout(·)

Norm2

层归一化

Add2->Norm2

Norm2->Output

> 标准 Transformer 编码器层（Post-LN 变体）的结构。输入流经多头注意力 (multi-head attention)，与残差输入相加，并归一化。这个结果随后流经前馈网络，再次与它的残差输入（第一个归一化的输出）相加，并再次归一化以产生该层的输出。在每次加法之前应用 Dropout。

### Pre-LN 变体

值得注意的是，有一种常见的架构修改称为 Pre-LN Transformer。在此变体中，层归一化 (normalization)步骤在输入进入每个子层（自注意力 (self-attention)和 FFN）*之前*应用，而残差连接将子层的输出直接添加到其输入。

Pre-LN 的流程如下所示：

1. X归一化=层归一化(X)X\_{\text{归一化}} = \text{层归一化}(X)X归一化​=层归一化(X)
2. Z=多头注意力(X归一化)Z = \text{多头注意力}(X\_{\text{归一化}})Z=多头注意力(X归一化​)
3. X′=X+Dropout(Z)X' = X + \text{Dropout}(Z)X′=X+Dropout(Z)
4. X归一化′=层归一化(X′)X'\_{\text{归一化}} = \text{层归一化}(X')X归一化′​=层归一化(X′)
5. Y=FFN(X归一化′)Y = \text{FFN}(X'\_{\text{归一化}})Y=FFN(X归一化′​)
6. 层输出=X′+Dropout(Y)\text{层输出} = X' + \text{Dropout}(Y)层输出=X′+Dropout(Y)

Pre-LN 通常带来更稳定的训练，特别是对于非常深的 Transformer，并且与原始 Post-LN 结构相比，可能需要较少仔细的学习率预热。理解这两种配置很有帮助，因为实现和研究论文可能使用任一变体。

堆栈中最后一个编码器层（第 NNN 层）的输出作为解码器堆栈中每一层交叉注意力机制 (attention mechanism)的键（KKK）和值（VVV）输入，我们接下来将讨论这个。

获取即时帮助、个性化解释和交互式代码示例。

---

### 解码器层结构

# 解码器层结构

编码器将输入序列处理成丰富的表示，而解码器的作用通常是生成输出序列，每次一个元素（自回归 (autoregressive)地）。为达此目的，每个解码器层会整合先前生成的输出元素和编码器堆栈产生的最终表示的信息。

单个解码器层由三个不同的子层组成，每个子层之后都跟着一个残差连接和层归一化 (normalization)步骤，这与编码器中的结构相似，但在注意力机制 (attention mechanism)上存在显著差异。

标准Transformer解码器层内的组成部分有：

1. **遮蔽多头自注意力 (self-attention)：** 这个子层允许解码器输入（部分生成的输出序列）中的每个位置关注解码器输入中所有及自身之前的位置。这里的重要补充是“遮蔽”。在序列生成的训练和推理 (inference)过程中，解码器必须仅依赖于先前生成的词元 (token)。遮蔽确保自注意力机制不能“提前看”输出序列中的未来词元，从而保持了逐步生成序列所需的自回归属性。这通常通过在softmax操作之前，向与未来位置对应的注意力分数添加一个大的负值（接近负无穷大）来实现，从而有效地将它们的概率归零。底层多头注意力 (multi-head attention)计算方式与第三章中描述的相同。
2. **多头交叉注意力（编码器-解码器注意力）：** 这是解码器整合来自输入序列信息的地方。与自注意力中查询 (QQQ)、键 (KKK) 和值 (VVV) 都源自同一序列不同，交叉注意力运作方式有别。查询 (QQQ) 源自*前一个*解码器子层（遮蔽自注意力层）的输出。然而，键 (KKK) 和值 (VVV) 直接来自*编码器堆栈*最后一层的输出。这个机制使解码器输出序列中的每个位置都能关注*输入*序列中的所有位置，从而使其在生成下一个输出词元时，能够集中于输入语境中最相关部分。这个子层对于机器翻译等需要输入和输出词对齐 (alignment)的任务而言非常基本。
3. **逐位置前馈网络（FFN）：** 这个子层在结构和功能上与编码器层中的FFN相同，它由两个线性变换和一个非线性激活函数 (activation function)（通常是ReLU或GeLU）组成。它独立应用于从交叉注意力子层产生的每个位置向量 (vector)。这提供了额外的建模能力，并使网络能够更有效地处理从注意力机制中收集的信息。

这三个子层中的每一个都有一个残差连接，之后是层归一化。子层输出的公式可以表示为：

层归一化(x+子层(x))\text{层归一化}(x + \text{子层}(x))层归一化(x+子层(x))

其中 xxx 是子层的输入，ext子层(x) ext{子层}(x)ext子层(x) 是子层本身实现的函数（例如，遮蔽自注意力、交叉注意力或FFN）。这些“添加与归一化”步骤对于训练深度Transformer模型非常重要，它们通过改善梯度流和稳定层输入来帮助训练。

以下图表说明了单个解码器层内的数据流：

DecoderLayer

cluster\_decoderₗayer

单个解码器层

input

来自前一层 / 嵌入的输入

maskedₐttn

遮蔽多头
自注意力

input->maskedₐttn

 Q, K, V

addₙorm₁

添加与归一化

input->addₙorm₁

残差

encoderₒutput

编码器输出 (K, V)

crossₐttn

多头
交叉注意力

encoderₒutput->crossₐttn

 K, V

maskedₐttn->addₙorm₁

addₙorm₁->crossₐttn

 Q

addₙorm₂

添加与归一化

addₙorm₁->addₙorm₂

残差

crossₐttn->addₙorm₂

ffn

逐位置
前馈网络

addₙorm₂->ffn

addₙorm₃

添加与归一化

addₙorm₂->addₙorm₃

残差

ffn->addₙorm₃

output

到下一层 / 最终输出的输出

addₙorm₃->output

> 标准Transformer解码器层内的数据流。输入代表目标序列嵌入 (embedding)（加位置编码 (positional encoding)）或前一个解码器层的输出。编码器输出为交叉注意力提供键和值。每个子层（遮蔽自注意力、交叉注意力、FFN）之后是添加与归一化。

理解这种分层结构很基本。遮蔽自注意力处理目前已生成的序列，交叉注意力通过编码器整合来自输入序列的语境，前馈网络提供进一步的变换。多个相同的解码器层随后堆叠起来，形成Transformer的完整解码器部分。后续章节将更详细地查看遮蔽注意力、交叉注意力、FFN和“添加与归一化”操作的具体细节。

获取即时帮助、个性化解释和交互式代码示例。

---

### 解码器中的掩码自注意力

# 解码器中的掩码自注意力

在解码器堆栈中，自注意力 (self-attention)机制 (attention mechanism)的运作方式与编码器不同。回顾一下，编码器同时处理整个输入序列，使每个位置能够关注所有其他位置（包括相对于自身的未来位置）。这种双向上下文 (context)对于理解输入序列结构很有帮助。

解码器的主要作用，尤其是在机器翻译或文本生成等任务中，通常是自回归 (autoregressive)的。这表示它一次生成一个输出序列中的词元 (token)，从左到右。当预测位置 iii 的词元时，解码器应仅能访问先前生成的词元（位置 111 到 i−1i-1i−1）以及完整的编码输入序列。它绝不能“预先查看”当前正在生成的目标序列中位置 i,i+1,…i, i+1, \dotsi,i+1,… 的词元。允许这种访问会使训练期间的生成任务变得过于简单，因为模型可以简单地复制下一个词元，而不是学习去预测它。

为实现这种单向信息流动，解码器采用**掩码自注意力**。其主要思路是通过遮蔽（设置为负无穷）任何对应于未来位置连接的注意力分数，来修改标准的缩放点积注意力计算。

### 掩码机制

掩码操作发生在softmax函数应用于缩放注意力分数之前。标准的缩放点积注意力分数计算如下：

分数=QKTdk\text{分数} = \frac{QK^T}{\sqrt{d\_k}}分数=dk​​QKT​

其中 QQQ、KKK 和 VVV 是从解码器输入（或前一个解码器层的输出）获得的查询、键和值矩阵，dkd\_kdk​ 是键的维度。

创建一个掩码矩阵 MMM。该矩阵的维度通常与注意力分数兼容（序列长度 \u00d7 序列长度）。对于位置 iii 关注位置 jjj 的情况：

- 如果 j≤ij \le ij≤i （当前或之前的位置），掩码值 MijM\_{ij}Mij​ 为 000。
- 如果 j>ij > ij>i （未来位置），掩码值 MijM\_{ij}Mij​ 为 −∞-\infty−∞ （或在实践中为非常大的负数，如 -1e9）。

然后将此掩码矩阵 MMM 添加到注意力分数中：

掩码分数=分数+M\text{掩码分数} = \text{分数} + M掩码分数=分数+M

最后，对这些掩码分数应用 softmax 函数：

注意力权重=softmax(掩码分数)\text{注意力权重} = \text{softmax}(\text{掩码分数})注意力权重=softmax(掩码分数)

添加 −∞-\infty−∞ 的作用是，在 softmax 函数内部进行指数运算后，这些分数变为 e−∞=0e^{-\infty} = 0e−∞=0。因此，未来位置的注意力权重 (weight)变为零，从而有效地阻止了来自这些位置的任何信息流动。解码器位置 iii 只能关注位置 111 到 iii。

位置 1位置 2位置 3位置 4位置 5位置 5位置 4位置 3位置 2位置 1

> 序列长度为 5 的注意力掩码的可视化。蓝色单元格（值 1）表示查询（行）可以关注的位置（列）。灰色单元格（值 0）表示被遮蔽的位置（未来词元 (token)）。请注意，每个位置都可以关注自身以及所有先前的位置。

这种因果注意力机制 (attention mechanism)，使Transformer解码器得以高效地学习序列生成任务。它确保每一步的预测仅依赖于之前步骤的已知输出，这与实际推理 (inference)或生成时的条件相符。这与编码器的自注意力 (self-attention)形成鲜明对比，后者可以自由地引入整个输入序列的信息。掩码自注意力（用于处理已生成的序列）和交叉注意力（用于整合来自编码器的信息）的结合，使得解码器能够生成连贯且上下文 (context)相关的输出序列。

获取即时帮助、个性化解释和交互式代码示例。

---

### 编码器-解码器交叉注意力

# 编码器-解码器交叉注意力

尽管掩码自注意力 (self-attention)机制 (attention mechanism)使解码器能够考虑其正在生成的序列中的先前令牌，但它无法直接获取从*输入*序列编码的信息。为弥补这一不足，并使解码器能够有选择地关注源信息的对应部分，Transformer架构在每个解码器层中引入了第二个注意力机制：**编码器-解码器交叉注意力**。

与自注意力不同，自注意力中的查询（Q）、键（K）和值（V）都源自同一序列（无论是编码器中的输入序列还是解码器掩码自注意力中部分生成的输出序列），而交叉注意力则从不同来源获取其组成部分：

1. **查询（Q）：** 这些向量 (vector)来自解码器中*前一个子层*的输出。具体来说，它们派生自残差连接和层归一化 (normalization)步骤后掩码自注意力层的输出。这些查询代表了解码器的当前状态以及生成下一个令牌所需的信息。
2. **键（K）和值（V）：** 这些向量直接来自*编码器堆栈最后一层的输出*。编码器已处理整个输入序列，其输出表示包含关于源的上下文 (context)信息。从该表示派生的键和值使解码器能够“回顾”输入序列。

这种结构使解码器在每一步都能够根据其当前上下文（到目前为止已生成的序列）提出查询，并将其与代表整个输入序列的键进行匹配。所得的注意力分数决定了在构建该步骤的解码器输出时，应给予编码器输出中对应值多少权重 (weight)。

### 数学表述

计算本身使用之前讨论的相同缩放点积注意力函数。然而，输入明确区分了来源：

注意力(Qdec,Kenc,Venc)=softmax(QdecKencTdk)Venc\text{注意力}(Q\_{dec}, K\_{enc}, V\_{enc}) = \text{softmax}\left(\frac{Q\_{dec} K\_{enc}^T}{\sqrt{d\_k}}\right) V\_{enc}注意力(Qdec​,Kenc​,Venc​)=softmax(dk​​Qdec​KencT​​)Venc​

其中：

- QdecQ\_{dec}Qdec​ 表示从解码器前一个子层输出派生出的查询矩阵。
- KencK\_{enc}Kenc​ 和 VencV\_{enc}Venc​ 表示从编码器堆栈输出派生出的矩阵。
- dkd\_kdk​ 是向量 (vector)的维度，用于缩放。

softmax函数确保分配给编码器值向量的权重 (weight)之和为一，基于查询-键交互所决定的相关性，生成加权平均值。

### 多头交叉注意力

正如自注意力 (self-attention)一样，交叉注意力也受益于多个并行运行的注意力头。每个头对输入的QdecQ\_{dec}Qdec​、KencK\_{enc}Kenc​和VencV\_{enc}Venc​应用单独的线性投影，将它们映射到不同的表示子空间。

头i=注意力(QdecWiQ,KencWiK,VencWiV)\text{头}\_i = \text{注意力}(Q\_{dec}W\_i^Q, K\_{enc}W\_i^K, V\_{enc}W\_i^V)头i​=注意力(Qdec​WiQ​,Kenc​WiK​,Venc​WiV​)

其中 WiQW\_i^QWiQ​、WiKW\_i^KWiK​ 和 WiVW\_i^VWiV​ 是第 iii 个头的学习投影矩阵。

这些独立头的输出随后被拼接起来，并通过一个最终的线性投影层，这与多头自注意力中的过程相同：

多头(Qdec,Kenc,Venc)=拼接(头1,...,头h)WO\text{多头}(Q\_{dec}, K\_{enc}, V\_{enc}) = \text{拼接}(\text{头}\_1, ..., \text{头}\_h) W^O多头(Qdec​,Kenc​,Venc​)=拼接(头1​,...,头h​)WO

使用多个头使解码器能够同时关注编码器输出中的信息，基于不同的标准或从不同的表示角度。例如，在翻译中，一个头可能专注于句法对齐 (alignment)，而另一个头则关注语义对应。

### 作用与集成

编码器-解码器交叉注意力主要作用是使解码器的生成过程依赖于输入序列的相关部分。没有它，解码器将只能通过初始解码器状态获取输入信息，缺少在生成过程中动态关注特定源令牌的能力。

考虑将“The black cat”（输入）翻译为“Le chat noir”（输出）。

1. 当生成“Le”时，解码器的掩码自注意力 (self-attention)查看序列起始令牌。它的交叉注意力可能会微弱地关注整个输入或专注于“The”。
2. 当生成“chat”时，解码器的掩码自注意力查看“Le”。它的交叉注意力机制 (attention mechanism)，利用受“Le”影响的查询，应该强烈地关注编码器中“cat”的表示。
3. 当生成“noir”时，掩码自注意力查看“Le chat”。交叉注意力现在应该强烈地关注编码器中“black”的表示。

这种动态聚焦是Transformer在序列到序列任务中取得成功的核心。

### 解码器层内的数据流

交叉注意力机制 (attention mechanism)位于每个解码器层内的掩码自注意力 (self-attention)子层和逐位置前馈网络子层之间。与其它子层一样，残差连接和层归一化 (normalization)也应用在其周围。

G

clusterₑncoder

编码器堆栈输出

cluster\_decoderₗayer

解码器层

EncoderOutput

编码器输出
(Kₑnc, Vₑnc)

CrossAttn

多头
交叉注意力

EncoderOutput->CrossAttn

键 (K), 值 (V)

DecoderInput

解码器输入
(来自掩码自注意力 + 相加与归一化)
(Q\_dec)

DecoderInput->CrossAttn

查询 (Q)

AddNorm2

相加与归一化

DecoderInput->AddNorm2

 残差

CrossAttn->AddNorm2

FFN\_Input

到前馈网络的输出

AddNorm2->FFN\_Input

> 这是一个在Transformer解码器层内的编码器-解码器交叉注意力子层的数据流简化图。查询源自解码器的状态，而键和值来自编码器的最终输出。

理解这种交叉注意力机制，对于掌握Transformer解码器如何有效使用输入序列的编码表示来引导输出序列的生成非常重要。它作为主要连接，将编码器对源序列的处理与解码器对目标序列的逐步生成联系起来。

获取即时帮助、个性化解释和交互式代码示例。

---

### 逐位置前馈网络 (FFN)

# 逐位置前馈网络 (FFN)

逐位置前馈网络（FFN）是每个编码器和解码器层中的一个重要组成部分。虽然注意力机制 (attention mechanism)旨在处理序列间的关联并整合来自不同位置的信息，但FFN提供额外的计算深度和非线性，对序列中的每个位置独立进行操作。

可以将注意力子层视为通信中心，允许令牌交换信息。然后，FFN则作为每个令牌的独立处理单元，转换通过注意力接收到的信息。

### 结构和运作

FFN通常是一个简单的全连接前馈网络，包含两个线性变换，中间夹有一个非线性激活函数 (activation function)。原始Transformer论文中，激活函数的标准选择是修正线性单元（ReLU）。

FFN在特定位置 ttt 的输入是来自前一个子层的输出向量 (vector) ztz\_tzt​（无论是自注意力 (self-attention)还是交叉注意力，且在Add & Norm步骤之后，具体取决于Post-LN或Pre-LN等层配置）。计算过程如下：

1. **第一次线性变换：** 输入向量 ztz\_tzt​（维度为 dmodeld\_{model}dmodel​）被投影到更高维空间 (high-dimensional space)，通常是 dffd\_{ff}dff​。这种扩展通过权重 (weight)矩阵 W1W\_1W1​（形状为 dmodel×dffd\_{model} \times d\_{ff}dmodel​×dff​）和偏置 (bias)向量 b1b\_1b1​（形状为 dffd\_{ff}dff​）实现。
   线性1(zt)=ztW1+b1\text{线性}\_1(z\_t) = z\_t W\_1 + b\_1线性1​(zt​)=zt​W1​+b1​
2. **非线性激活：** 第一次变换的结果通过ReLU激活函数（或在较新模型中有时使用GeLU等其他激活函数）处理。ReLU引入非线性，使得网络能够学习更复杂的函数。
   ReLU(x)=max⁡(0,x)\text{ReLU}(x) = \max(0, x)ReLU(x)=max(0,x)
3. **第二次线性变换：** 激活后的输出使用第二个权重矩阵 W2W\_2W2​（形状为 dff×dmodeld\_{ff} \times d\_{model}dff​×dmodel​）和偏置向量 b2b\_2b2​（形状为 dmodeld\_{model}dmodel​）重新投影回原始模型维度 dmodeld\_{model}dmodel​。
   线性2(x)=xW2+b2\text{线性}\_2(x) = x W\_2 + b\_2线性2​(x)=xW2​+b2​

综合这些步骤，对于单个位置 ttt 的完整FFN操作可以表示为：

FFN(zt)=ReLU(ztW1+b1)W2+b2FFN(z\_t) = \text{ReLU}(z\_t W\_1 + b\_1) W\_2 + b\_2FFN(zt​)=ReLU(zt​W1​+b1​)W2​+b2​

或者，更一般地使用 fff 代表激活函数：

FFN(zt)=f(ztW1+b1)W2+b2FFN(z\_t) = f(z\_t W\_1 + b\_1) W\_2 + b\_2FFN(zt​)=f(zt​W1​+b1​)W2​+b2​

### 逐位置应用

该网络的一个重要方面是其“逐位置”应用。虽然在序列的所有位置都使用相同的FFN（指完全相同的权重 (weight)矩阵 W1,W2W\_1, W\_2W1​,W2​ 和偏置 (bias) b1,b2b\_1, b\_2b1​,b2​），但它被独立应用于每个位置的向量 (vector)表示。

如果注意力子层后的输入序列表示是 Z=[z1,z2,...,zn]Z = [z\_1, z\_2, ..., z\_n]Z=[z1​,z2​,...,zn​]，其中 nnn 是序列长度，则FFN计算结果为：

FFNoutput=[FFN(z1),FFN(z2),...,FFN(zn)]FFN\_{output} = [FFN(z\_1), FFN(z\_2), ..., FFN(z\_n)]FFNoutput​=[FFN(z1​),FFN(z2​),...,FFN(zn​)]

这与注意力机制 (attention mechanism)形成鲜明对比，后者明确地建模不同位置*之间*的关联。FFN独立处理每个位置的表示，使得模型能够学习单个令牌表示的复杂变换，这些变换由通过注意力收集的上下文 (context)信息来辅助。

FFN

逐位置FFN应用

clusterₚos1

位置 1

clusterₚos2

位置 2

clusterₚosN

位置 n

z1

z₁ (dₘodel)

l1₁

线性 1

z1->l1₁

act1

ReLU

l1₁->act1

W1b1

共享权重
W₁, b₁

l1₁->W1b1

l2₁

线性 2

act1->l2₁

out1

FFN(z₁) (dₘodel)

l2₁->out1

W2b2

共享权重
W₂, b₂

l2₁->W2b2

z2

z₂ (dₘodel)

l1₂

线性 1

z2->l1₂

act2

ReLU

l1₂->act2

l1₂->W1b1

l2₂

线性 2

act2->l2₂

out2

FFN(z₂) (dₘodel)

l2₂->out2

l2₂->W2b2

zn

zₙ (dₘodel)

l1ₙ

线性 1

zn->l1ₙ

actn

ReLU

l1ₙ->actn

l1ₙ->W1b1

l2ₙ

线性 2

actn->l2ₙ

outn

FFN(zₙ) (dₘodel)

l2ₙ->outn

l2ₙ->W2b2

> 该图显示了相同的两个线性层（具有共享权重 W1,b1W\_1, b\_1W1​,b1​ 和 W2,b2W\_2, b\_2W2​,b2​）和ReLU激活函数 (activation function)如何独立应用于每个序列位置的输入向量 (ztz\_tzt​)。

### 维度和实现

在原始Transformer论文《Attention Is All You Need》中，模型维度 dmodeld\_{model}dmodel​ 为512，FFN的内层维度 dffd\_{ff}dff​ 设置为2048。这种四倍扩展（dff=4×dmodeld\_{ff} = 4 \times d\_{model}dff​=4×dmodel​）是一种常用的启发式方法，尽管也存在变体。这种扩展使得FFN能够将表示投影到更高维空间 (high-dimensional space)，在那里可以更容易地学习复杂模式，然后再将其投影回标准模型维度。

从实现角度看，在每个位置独立应用相同的线性变换是计算高效的。它可以使用1x1卷积实现。如果将向量 (vector)序列 ZZZ（形状：序列长度 ×\times× dmodeld\_{model}dmodel​）视为一个高度为1、宽度等于序列长度的“图像”，那么FFN的线性变换对应于核大小为1x1、输入通道为 dmodeld\_{model}dmodel​、输出通道为 dffd\_{ff}dff​（对于第一层）或 dmodeld\_{model}dmodel​（对于第二层）的卷积。这种形式使得深度学习 (deep learning)框架能够利用高度优化的卷积实现，以进行跨序列长度的并行处理。

FFN子层与注意力子层一起，构成每个编码器和解码器层中的核心计算块。理解其结构和逐位置操作对于掌握Transformer如何在序列交互层面（注意力）和单个令牌层面（FFN）处理信息是必要的。此子层之后是残差连接和层归一化 (normalization)，我们将在接下来讨论。

获取即时帮助、个性化解释和交互式代码示例。

---

### 残差连接 (相加)

# 残差连接 (相加)

深度神经网络 (neural network)，包括包含多头注意力 (multi-head attention)机制 (attention mechanism)和前馈网络的 Transformer 编码器和解码器层，通常面临一个与训练效果相关的基本挑战。堆叠多层可以增加模型能力，但也可能使优化变得困难。梯度，即学习过程中使用的信号，在通过多层反向传播 (backpropagation)时可能会减弱或消失，阻碍了前面层的更新。同样地，梯度有时也可能变得过大，导致不稳定。

为了解决这些问题，并使得训练 Transformer 典型的深层架构成为可能，采用了残差连接，也叫跳跃连接。这种技术因计算机视觉中的残差网络 (ResNets) 而广为人知，它为梯度在网络中传输提供了一条直接通路。

### 残差连接的原理

在 Transformer 层中，每个主要子层（多头自注意力 (self-attention)机制 (attention mechanism)和逐位置前馈网络）都包裹在残差连接中。核心思路非常直观：将子层的输入与其输出相加。

令 xxx 表示子层的输入（例如，来自前一层的输出或初始嵌入 (embedding)）。令 SubLayer(x)SubLayer(x)SubLayer(x) 表示子层本身实现的函数（例如，多头注意力 (multi-head attention)或 FFN）。该操作的输出，*在* 归一化 (normalization)之前，计算方法如下：

残差输出=x+SubLayer(x)\text{残差输出} = x + SubLayer(x)残差输出=x+SubLayer(x)

这种相加操作通常紧接着层归一化，构成了 Transformer 图示中常见的“加和归一化”模块。

G

input

输入 (x)

sublayer

子层
(例如，多头注意力
或前馈网络)

input->sublayer

add

+

input->add

 x

sublayer->add

 SubLayer(x)

output

输出
(至层归一化)

add->output

> 残差连接跳过一个子层，并将原始输入 xxx 添加到子层的输出 SubLayer(x)SubLayer(x)SubLayer(x)。

### 残差连接为何有效

1. **改善梯度传输：** 在反向传播 (backpropagation)时，残差块的梯度计算包含一条通过加法操作的直接回传路径。输出对输入 xxx 的梯度包含来自该直接连接的 +1+1+1 项。这确保了即使通过 SubLayer(x)SubLayer(x)SubLayer(x) 路径的梯度变得非常小，仍然有梯度信号通过恒等连接 (xxx) 相对不受阻碍地回传。这大大缓解了梯度消失问题，使得信息能够更有效地在多层之间传递。
2. **促进恒等映射学习：** 残差连接使得层更容易学习恒等函数。如果给定层的理想变换仅仅是保持输入不变，网络可以通过将 SubLayerSubLayerSubLayer 内的权重 (weight)趋近于零来实现这一点。没有残差连接，使用复杂的非线性变换来学习精确的恒等映射会很困难。这一特性使得可以在不一定损害性能的情况下增加层；如果需要，网络可以有效地“忽略”一个层。
3. **实现更深网络：** 通过缓解梯度问题和简化恒等映射的学习，残差连接在成功训练那些代表强大 Transformer 模型特性的非常深的网络（编码器和解码器堆栈中通常有 6、12、24 或更多层）方面发挥着重要作用。

### 在 Transformer 层中的实现

在标准 Transformer 的编码器和解码器层中，你会发现两个主要子层，每个子层后面都跟着这个“加和归一化 (normalization)”步骤：

1. **多头注意力 (multi-head attention)：** 输入 xxx 传入多头注意力机制 (attention mechanism)。其输出 Attention(x)Attention(x)Attention(x) 随后与原始输入 xxx 相加。层归一化应用于这个和。
   OutputAttn=LayerNorm(x+MultiHeadAttention(x))Output\_{Attn} = LayerNorm(x + MultiHeadAttention(x))OutputAttn​=LayerNorm(x+MultiHeadAttention(x))
2. **前馈网络：** 第一个归一化步骤的输出，我们称之为 y=OutputAttny = Output\_{Attn}y=OutputAttn​，作为逐位置前馈网络的输入。FFN 的输出 FFN(y)FFN(y)FFN(y) 与其输入 yyy 相加。应用最终的层归一化。
   OutputLayer=LayerNorm(y+FFN(y))Output\_{Layer} = LayerNorm(y + FFN(y))OutputLayer​=LayerNorm(y+FFN(y))

这些残差连接是一种看似简单却极度有效的技术，它们构成了 Transformer 设计中不可或缺的一部分，并对其在建模复杂序列数据方面的成功起到了很大作用。没有它们，训练现代大型语言模型中常见的深层堆栈会困难得多。

获取即时帮助、个性化解释和交互式代码示例。

---

### 层归一化

# 层归一化

在构建包含大量编码器或解码器层的Transformer模型时，确保训练的稳定性和效率成为一个重要难题。网络中流动的激活值在不同层和训练步中可能呈现出差异很大的分布，这种现象有时与内部协变量偏移有关。此外，深层网络容易出现梯度消失或梯度爆炸问题，阻碍有效的学习。层归一化 (normalization)（LayerNorm）是集成到Transformer架构中的一项技术，专门用于缓解这些问题。

与在批次维度上进行归一化的批归一化不同，层归一化独立地作用于每个数据样本（即批次中的每个序列），并在特征维度（即嵌入 (embedding)维度 dmodeld\_{model}dmodel​）上进行归一化。这使得它特别适合序列数据，因为在序列数据中批次统计信息可能代表性不足，或可变序列长度使得批归一化变得复杂。

### 层归一化 (normalization)如何工作

对于表示层内序列中特定位置激活值的给定输入向量 (vector) xxx（通常维度为 dmodeld\_{model}dmodel​），层归一化首先计算其元素的均值 (μ\muμ) 和方差 (σ2\sigma^2σ2):

μ=1dmodel∑i=1dmodelxi\mu = \frac{1}{d\_{model}} \sum\_{i=1}^{d\_{model}} x\_iμ=dmodel​1​i=1∑dmodel​​xi​
σ2=1dmodel∑i=1dmodel(xi−μ)2\sigma^2 = \frac{1}{d\_{model}} \sum\_{i=1}^{d\_{model}} (x\_i - \mu)^2σ2=dmodel​1​i=1∑dmodel​​(xi​−μ)2

接下来，它使用这些均值和方差对输入向量 xxx 进行归一化，并加入一个小的 epsilon (ϵ\epsilonϵ，例如 10−510^{-5}10−5) 以保证数值稳定性:

x^i=xi−μσ2+ϵ\hat{x}\_i = \frac{x\_i - \mu}{\sqrt{\sigma^2 + \epsilon}}x^i​=σ2+ϵ​xi​−μ​

最后，归一化后的输出 x^\hat{x}x^ 通过两个可学习的参数 (parameter)向量进行缩放和偏移：一个增益（或缩放）参数 γ\gammaγ 和一个偏差（或偏移）参数 β\betaβ，两者维度均为 dmodeld\_{model}dmodel​。这些参数在训练过程中与模型的其他权重 (weight)一同学习。它们使得网络能够自适应地确定归一化激活值的最佳缩放比例和位置，如果需要，甚至可能恢复原始激活值（当 γ=σ2+ϵ\gamma = \sqrt{\sigma^2 + \epsilon}γ=σ2+ϵ​， β=μ\beta = \muβ=μ 时）。

yi=γix^i+βiy\_i = \gamma\_i \hat{x}\_i + \beta\_iyi​=γi​x^i​+βi​

整个操作 (LayerNorm(x)=yLayerNorm(x) = yLayerNorm(x)=y) 标准化了后续子层（如多头注意力 (multi-head attention)或前馈网络）的输入，有助于稳定隐藏状态的动态并改善梯度流动。

### 位置：前置层归一化 (normalization)与后置层归一化

原始的Transformer论文（《Attention Is All You Need》）将层归一化放在残差连接*之后*，这种配置现在称为后置层归一化 (Post-LN):

output=LayerNorm(x+Sublayer(x))output = LayerNorm(x + Sublayer(x))output=LayerNorm(x+Sublayer(x))

这里，xxx 是子层的输入（例如多头注意力 (multi-head attention)或前馈网络），而 Sublayer(x)Sublayer(x)Sublayer(x) 是该子层的输出。

G

clusterₚostₗn

后置层归一化子层

xᵢn

输入 (x)

sublayer

子层
(如多头注意力/FFN)

xᵢn->sublayer

add

+

xᵢn->add

 残差

sublayer->add

ln

LayerNorm

add->ln

out

输出

ln->out

> 后置层归一化Transformer子层中的数据流。归一化在残差相加之后进行。

虽然有效，但后置层归一化有时会导致训练困难，尤其是在非常深的模型中，因为每个块的输出在传递给下一个块之前未进行归一化。梯度可能难以有效地通过未归一化的残差流反向传播 (backpropagation)。

近期更多的实现通常倾向于前置层归一化 (Pre-LN)，其中层归一化应用于子层*之前*，在残差连接的主分支内：

output=x+Sublayer(LayerNorm(x))output = x + Sublayer(LayerNorm(x))output=x+Sublayer(LayerNorm(x))

G

clusterₚreₗn

前置层归一化子层

xᵢn

输入 (x)

ln

LayerNorm

xᵢn->ln

add

+

xᵢn->add

 残差

sublayer

子层
(如多头注意力/FFN)

ln->sublayer

sublayer->add

out

输出

add->out

> 前置层归一化Transformer子层中的数据流。归一化在子层转换之前对输入进行。

前置层归一化倾向于稳定训练，通常允许使用更高的学习率，并减少对复杂学习率预热策略的需求（尽管预热仍普遍使用）。梯度通过归一化后的激活值流动得更直接，残差路径保持“干净”。大多数现代大型语言模型采用前置层归一化结构。

无论位置如何，层归一化在每个编码器和解码器层内应用两次：一次在自注意力 (self-attention)机制 (attention mechanism)之前（前置层归一化）或之后（后置层归一化），另一次在位置前馈网络之前或之后。在解码器中，与编码器-解码器交叉注意力机制相关的还有一个层归一化。

总之，层归一化是一个重要组成部分，它与残差连接配合使用，使得深层Transformer堆栈的训练成为可能。通过独立地稳定每个序列位置的激活分布，它使优化过程更平稳，并为这些强大模型的成功做出重要贡献。

获取即时帮助、个性化解释和交互式代码示例。

---

### 多层堆叠

# 多层堆叠

单个编码器层和解码器层尽管具有内部结构，但它们对输入表示执行的转换通常是有限的。Transformer架构的真正威力源于顺序组合这些层，为编码器和解码器构建深层堆叠。例如，原始Transformer模型为编码器使用了N=6N=6N=6个相同层的堆叠，为解码器也使用了N=6N=6N=6个相同层的堆叠。

### 为何堆叠层？

堆叠层使得模型能够分层学习输入数据渐趋复杂的表示。正如卷积神经网络 (neural network) (CNN)在不同层中从简单边缘到复杂对象构建表示一样，堆叠的Transformer层逐步优化序列表示。

1. **分层处理：** 初始层可能专注于序列中的局部上下文 (context)和依赖关系。后续层可以整合更长距离的信息，借助较低层的优化表示来捕捉更多全局关系和抽象特征。每层内的多头注意力 (multi-head attention)机制 (attention mechanism)允许不同的头关注不同方面，而堆叠使模型能够逐层在这些多元视角上进行构建。
2. **模型能力增强：** 每层都增加了计算深度和参数 (parameter)复杂性。层数越多，模型就具备更大的能力来近似机器翻译或文本生成等任务所需的复杂函数。通过多层顺序应用自注意力 (self-attention)、交叉注意力（在解码器中）和前馈转换，可实现从输入序列到输出序列的高度非线性和强大映射。

### 堆叠机制

在标准Transformer架构中，编码器和解码器均由指定数量NNN个相同层连续堆叠组成。尽管这些层共享相同的结构（相同的子层和维度），但每层都有其独有的可训练权重 (weight)集。

- **编码器堆叠：** 输入序列（词元 (token)嵌入 (embedding) + 位置编码 (positional encoding)）首先由编码器层1处理。层1的输出与输入具有相同维度，并作为层2的输入，以此类推，直到层NNN。最终编码器层（层NNN）的输出张量封装了整个输入序列的丰富表示。这个最终编码器输出随后被重要地用作NNN个解码器层中*每个*交叉注意力子层的键（KKK）和值（VVV）输入。
- **解码器堆叠：** 类似地，解码器堆叠处理目标序列嵌入（加位置编码）。解码器层iii的输出成为解码器层i+1i+1i+1的输入。每个解码器层对目标序列执行带掩码的自注意力 (self-attention)，接着与最终编码器输出进行交叉注意力，最后通过一个位置前馈网络处理结果。最终解码器层（层NNN）的输出随后被输入到最终线性变换和softmax层，以生成词汇表 (vocabulary)上的输出概率。

G

clusterᵢnput

输入处理

clusterₑncoder

编码器堆叠 (N层)

cluster\_decoder

解码器堆叠 (N层)

clusterₒutput

输出处理

inp

输入序列

emb

嵌入 +
位置编码

inp->emb

enc1

编码器层 1

emb->enc1

输入表示

enc\_dots
...

enc1->enc\_dots

encN

编码器层 N

encₒut

编码器输出

encN->encₒut

enc\_dots->encN

dec1

解码器层 1

encₒut->dec1

键, 值
(交叉注意力)

decN

解码器层 N

encₒut->decN

键, 值
(交叉注意力)

decᵢnp

目标序列
(右移)

decₑmb

嵌入 +
位置编码

decᵢnp->decₑmb

decₑmb->dec1

目标表示

dec\_dots
...

dec1->dec\_dots

finalₗinear

线性层

decN->finalₗinear

dec\_dots->decN

softmax

Softmax

finalₗinear->softmax

output

输出概率

softmax->output

> 数据通过堆叠的编码器和解码器层流动。最终编码器输出通过交叉注意力机制 (attention mechanism)为所有解码器层提供上下文 (context)（键和值）。

### 启用多层堆叠：残差连接与归一化 (normalization)

简单地堆叠层可能导致训练困难，特别是深度网络中常见的梯度消失问题，即梯度变得过小，无法有效更新早期层的权重 (weight)。Transformer架构在每层中引入了两个重要机制来减轻此问题：

1. **残差连接（加）：** 每个子层（自注意力 (self-attention)、前馈）周围都有一个残差连接。子层的输入xxx在经过dropout后，与子层的输出子层(x)\text{子层}(x)子层(x)相加。这为梯度在网络中反向流动创建了一条直接路径，或“捷径”。这通过确保梯度在通过多层反向传播 (backpropagation)时不会过度减小，从而大幅简化了深度模型的优化。在原始的Post-LN公式中，该操作形式上定义为层归一化(x+Dropout(子层(x)))\text{层归一化}(x + \text{Dropout}(\text{子层}(x)))层归一化(x+Dropout(子层(x)))。
2. **层归一化（归一）：** 应用于残差连接路径内（在Pre-LN变体中是在子层之前，在原始Post-LN公式中是在相加之后）。层归一化通过对每个位置在嵌入 (embedding)维度上独立地归一化特征来稳定每层内的激活。这有助于防止激活值爆炸或消失，减少对初始化的敏感度，并且通常允许更快、更稳定的训练，特别是对于更多层堆叠。

如果没有这些组件，训练具有大量层（例如N>2N > 2N>2）的Transformer将极其困难，甚至不可能。它们确保信息和梯度即使通过数十个堆叠层也能有效传播。

### 模型深度带来的影响

增加层数NNN直接影响模型：

- **性能：** 通常，更深的模型（更大的NNN）在复杂序列任务上能取得更佳的性能，直到出现边际效益递减或优化困难。最佳深度通常取决于任务复杂性、可用训练数据量以及计算预算。
- **计算成本：** 训练时间和推理 (inference)时间都大致与NNN呈线性关系。将层数加倍，通过编码器和解码器堆叠进行一次前向传播所需的计算量也大致加倍。
- **参数 (parameter)：** 假设每层结构相同（这是标准做法），总参数量也与NNN呈线性关系。这会增加训练期间存储模型检查点和激活值的内存需求。

选择NNN是Transformer设计中的一个基本超参数 (hyperparameter)。虽然原始论文使用了N=6N=6N=6，但现代大型语言模型常采用更多层堆叠（例如，N=24,48,96N=24, 48, 96N=24,48,96甚至更多）。这种深度的增加得益于对庞大数据集的访问、大量计算资源以及架构细节（如广泛采用Pre-LN归一化 (normalization)以提高稳定性）和训练技术的持续改进。

总之，通过堆叠多个结构相同的编码器和解码器层，Transformer模型实现了高性能所需的深度。这种深度允许对序列信息进行分层处理，并为复杂序列建模任务提供了所需的模型能力。这些多层堆叠的成功训练很大程度上依赖于每个组成块中残差连接和层归一化的精心整合。

获取即时帮助、个性化解释和交互式代码示例。

---

### 最终线性层和Softmax输出

# 最终线性层和Softmax输出

Transformer架构中的最终解码器层对序列进行处理，它会整合来自输入序列的信息（通过交叉注意力）和之前生成的输出标记 (token)（通过遮蔽自注意力 (self-attention)）。这个过程会产生一系列高维表示向量 (vector)。对于输出序列中的每个位置，解码器堆栈会生成一个维度为 dmodeld\_{model}dmodel​ 的向量。尽管这些向量编码了丰富的上下文 (context)信息，但它们不能直接解读为目标词汇表 (vocabulary)上的概率分布。这种分布对于机器翻译或文本生成等任务是必不可少的。

Transformer解码器架构的最后一步涉及将这些输出向量转换为可用的概率。这通常通过两个连续操作来实现：一个最终线性变换，然后是一个softmax激活函数 (activation function)。

### 最终线性变换

顶部解码器层的输出是一个形状为 (batch\_size, target\_sequence\_length, dmodeld\_{model}dmodel​) 的张量。最终线性层的作用是将每个位置上的这个 dmodeld\_{model}dmodel​ 维表示向量 (vector)投影到一个维度与目标词汇表 (vocabulary)大小相等的向量上，我们将这个大小记作 VVV。

这是一个标准的、不带激活函数 (activation function)（或有时视为具有恒等激活函数）的全连接线性层。如果我们将最终解码器层的输出表示为 Hdec∈Rbatch\_size×target\_sequence\_length×dmodelH\_{dec} \in \mathbb{R}^{\text{batch\\_size} \times \text{target\\_sequence\\_length} \times d\_{model}}Hdec​∈Rbatch\_size×target\_sequence\_length×dmodel​，将线性层的权重 (weight)矩阵表示为 Wout∈Rdmodel×VW\_{out} \in \mathbb{R}^{d\_{model} \times V}Wout​∈Rdmodel​×V，将其偏置 (bias)表示为 bout∈RVb\_{out} \in \mathbb{R}^{V}bout​∈RV，则该操作可以描述为：

Logits=HdecWout+bout\text{Logits} = H\_{dec} W\_{out} + b\_{out}Logits=Hdec​Wout​+bout​

在这里，矩阵乘法独立地应用于 `target_sequence_length` 维度上每个位置的 dmodeld\_{model}dmodel​ 维向量。得到的 `Logits` 张量形状为 (batch\_size, target\_sequence\_length, VVV)。序列中特定位置 `t` 处大小为 VVV 的每个向量都包含目标词汇表中每个可能标记 (token)的原始、未归一化 (normalization)分数（logits）。分数越高，表示该位置上该标记的可能性越大。

*实现备注：* 一种常见的方法，尤其是在源和目标词汇表共享（或紧密关联）的模型中，是在输入嵌入 (embedding)层和这个最终线性层之间共享权重。输入嵌入层将词汇表索引（实际上是独热向量）投影到 dmodeld\_{model}dmodel​ 维度，而最终线性层将 dmodeld\_{model}dmodel​ 维度投影回词汇表分数。与输入嵌入矩阵（可能转置）共享权重矩阵 WoutW\_{out}Wout​ 可以大幅减少模型参数 (parameter)的数量，特别是对于大型词汇表，并且已被经验证实效果良好。

### Softmax 函数

线性层生成的 logits 是实值分数，不构成概率分布。为了将这些分数转换为概率，softmax 函数独立地应用于目标序列中每个位置的 logit 向量 (vector)。

对于序列中的特定位置 ttt，令 zt∈RVz\_t \in \mathbb{R}^Vzt​∈RV 为 logit 向量。softmax 函数计算词汇表 (vocabulary)中第 iii 个标记 (token)的概率 pip\_ipi​（其中 iii 的范围从 1 到 VVV），如下所示：

pi=Softmax(zt)i=ezt,i∑j=1Vezt,jp\_i = \text{Softmax}(z\_t)\_i = \frac{e^{z\_{t,i}}}{\sum\_{j=1}^{V} e^{z\_{t,j}}}pi​=Softmax(zt​)i​=∑j=1V​ezt,j​ezt,i​​

此操作为每个位置 ttt 产生一个概率向量 pt∈RVp\_t \in \mathbb{R}^Vpt​∈RV，其中：

1. 每个元素 pip\_ipi​ 都是非负的 (pi≥0p\_i \ge 0pi​≥0)。
2. 向量中所有元素的和为 1 (∑i=1Vpi=1\sum\_{i=1}^{V} p\_i = 1∑i=1V​pi​=1)。

因此，整个Transformer解码器堆栈的输出是一个形状为 (batch\_size, target\_sequence\_length, VVV) 的张量，其中沿最后一个维度的每个向量都表示目标词汇表上的概率分布。

### 训练和推理 (inference)中的使用

- **训练：** 在训练期间，此输出概率分布用于计算损失，通常是交叉熵损失，与每个位置的实际目标标记 (token)进行对比。模型的参数 (parameter)（包括最终线性层、注意力机制 (attention mechanism)、前馈网络等中的参数）通过反向传播 (backpropagation)更新以最小化此损失，从而有效学习在给定上下文 (context)的情况下预测正确的下一个标记。标签平滑等技术通常在计算损失前应用于目标分布，这可以提高泛化能力。
- **推理：** 在推理期间（例如，生成文本），模型以自回归 (autoregressive)方式运行。在每个步骤 ttt，softmax层生成的概率分布 ptp\_tpt​ 用于选择下一个标记。常见的策略包括：
  - **贪婪解码：** 简单选择概率最高的标记 (argmaxipiargmax\_i p\_iargmaxi​pi​)。
  - **束搜索：** 保持多个候选序列（束），并在每个步骤中寻找最有可能的扩展，可能获得比贪婪解码更好的结果。
  - **采样方法：** 通过从概率分布中采样（例如，温度采样、top-k采样、核采样）引入随机性，以生成更多样化的输出。

这个最终线性层和softmax函数连接了Transformer学习到的复杂内部表示与语言生成任务的离散、概率性质。它们提供了所需的机制，将模型的理解转换为词汇表 (vocabulary)上的明确预测。

获取即时帮助、个性化解释和交互式代码示例。

---

### 动手实践：构建编码器块

# 动手实践：构建编码器块

Transformer编码器包含多个组件，其中包括用于捕捉语境关系的多头自注意力 (self-attention)、用于非线性转换的位置维度前馈网络、用于梯度流动的残差连接以及用于稳定激活的层归一化 (normalization)。这些组件被组合成一个功能性的`EncoderBlock`。构建`EncoderBlock`呈现了这些子层在一个编码器层中如何协同工作，从而形成堆叠以创建完整编码器的可重复单元。

我们的目标是实现一个标准的Transformer编码器块模块，通常使用像PyTorch或TensorFlow这样的框架。本例中，我们将使用PyTorch语法。

### 编码器块结构回顾

回顾单个编码器块内部的数据流：

1. 输入序列嵌入 (embedding)（结合了位置编码 (positional encoding)）首先通过多头自注意力 (self-attention)机制 (attention mechanism)。
2. 一个残差连接将原始输入添加到注意力子层的输出。
3. 结果随后由层归一化 (normalization)处理。
4. 这个归一化后的输出进入一个位置维度前馈网络。
5. 另一个残差连接将输入*到前馈子层*的输入添加到其输出。
6. 最后，应用第二次层归一化。

Dropout通常应用于自注意力和前馈子层之后，*在*残差连接和归一化*之前*。

G

inp

输入 (x)

mha

多头
自注意力

inp->mha

add1

加法

inp->add1

drop1

Dropout

mha->drop1

drop1->add1

norm1

层归一化

add1->norm1

ffn

位置维度
前馈网络

norm1->ffn

add2

加法

norm1->add2

drop2

Dropout

ffn->drop2

drop2->add2

norm2

层归一化

add2->norm2

out

输出

norm2->out

> 标准Transformer编码器块内部的数据流（后归一化变体）。虚线表示残差连接。

### 在PyTorch中的实现

让我们为`EncoderBlock`定义一个PyTorch `nn.Module`。我们假设你已经有了`MultiHeadAttention`（如第3章所述）和`PositionwiseFeedForward`（本章前面已讨论）的实现。

```python
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    """
    实现一个Transformer编码器块。

    此块遵循“Attention Is All You Need”中描述的结构：
    输入 -> 自注意力 -> 加法 & 归一化 -> 前馈网络 -> 加法 & 归一化 -> 输出
    使用后置层归一化（先加法，后归一化）。
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_prob: float = 0.1):
        """
        参数：
            d_model: 输入和输出嵌入的维度（模型维度）。
            num_heads: 注意力头的数量。
            d_ff: 前馈网络的内部维度。
            dropout_prob: 在注意力和FFN后应用的dropout概率。
        """
        super().__init__()
        if d_model % num_heads != 0:
             raise ValueError(f"'d_model' ({d_model}) 必须能被 'num_heads' ({num_heads}) 整除")

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        将输入通过编码器块。

        参数：
            x: 输入张量，形状为 (batch_size, seq_len, d_model)。
            mask: 自注意力层的可选掩码。通常用于填充。
                  形状应能广播到 (batch_size, num_heads, seq_len, seq_len)。

        返回：
            输出张量，形状为 (batch_size, seq_len, d_model)。
        """

        attn_output = self.self_attn(x, x, x, mask)

        x = self.norm1(x + self.dropout(attn_output))

        sublayer1_output = x

        ff_output = self.feed_forward(sublayer1_output)

        x = self.norm2(sublayer1_output + self.dropout(ff_output))

        return x
```

### 理解代码

- **初始化 (`__init__`)**：我们实例化了必要的子模块：`MultiHeadAttention`、`PositionwiseFeedForward`、两个`LayerNorm`层和一个`Dropout`层。`LayerNorm`层对特征维度（`d_model`）进行归一化 (normalization)。为了多头注意力 (multi-head attention)机制 (attention mechanism)，代码进行了一个检查，确保`d_model`能够被`num_heads`整除。
- **前向传播 (`forward`)**：
  - 第一部分处理多头自注意力 (self-attention)子层。输入`x`作为查询、键和值。输出`attn_output`使用dropout进行正则化 (regularization)。
  - 第一个残差连接将原始输入`x`添加到（经过dropout修改的）注意力输出。这个和随后通过第一个层归一化（`self.norm1`）。结果更新变量`x`。
  - 第二部分处理位置维度前馈子层。第一个归一化（`x`）的输出通过前馈网络（`self.feed_forward`）。
  - 它的输出（`ff_output`）使用dropout进行正则化。
  - 第二个残差连接将*到前馈层*的输入（即`self.norm1`的输出，在优化后的代码中临时存储为`sublayer1_output`）添加到（经过dropout修改的）前馈输出。
  - 这个和通过第二个层归一化（`self.norm2`）。
  - 最终的张量，经过两个子层以及残差和归一化处理后，被返回。

### 实例化与使用示例

以下是如何创建和使用`EncoderBlock`，包括子模块的占位符定义，以使示例可运行：

```python

batch_size = 4
seq_len = 50
d_model = 512
num_heads = 8
d_ff = 2048
dropout_prob = 0.1

dummy_input = torch.rand(batch_size, seq_len, d_model)

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
             raise ValueError("d_model 必须能被 num_heads 整除")
        self.head_dim = d_model // num_heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        projected_q = self.fc_q(query)
        return self.fc_out(projected_q)

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):

        return self.linear2(self.dropout(self.relu(self.linear1(x))))

encoder_block = EncoderBlock(d_model, num_heads, d_ff, dropout_prob)

output = encoder_block(dummy_input)

print(f"输入形状: {dummy_input.shape}")
print(f"输出形状: {output.shape}")
```

输出张量保持形状`(batch_size, seq_len, d_model)`，这非常重要。这使得一个`EncoderBlock`的输出可以直接作为输入传入堆叠中的下一个`EncoderBlock`，从而能够构建深度编码器模型。

### 配置与变体

- **超参数 (parameter) (hyperparameter)**：`d_model`、`num_heads`、`d_ff`和`dropout_prob`的选择显著影响模型的容量、计算成本和泛化能力。原始的Transformer使用了`d_model=512`、`num_heads=8`、`d_ff=2048`和`dropout_prob=0.1`。更大的模型通常使用更大的值。
- **归一化 (normalization)位置（前置归一化）**：此实现使用后置归一化（层归一化在残差加法*之后*）。另一种方法是前置归一化，它在自注意力 (self-attention)和前馈子层*之前*应用层归一化，残差连接在之后添加。前置归一化通常带来更稳定的训练，尤其对于更深的模型，并且需要修改`forward`方法的结构。我们将在第6章讨论前置归一化与后置归一化的权衡。

这个实践示例提供了一个编码器块的具体实现，结合了前面讨论的理论组件。通过理解如何构建这个基础单元，你将能够构建整个编码器堆栈，并理解Transformer架构内发生的数据转换。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 6 Advanced Architectural Variants Analysis

### 自注意力机制的计算复杂度

# 自注意力机制的计算复杂度

如前所述，标准的自注意力 (self-attention)机制 (attention mechanism)虽然强大，但会带来显著的计算负担，尤其当输入序列变长时。了解这种计算成本，有助于把握本章后续会介绍的更高效架构的设计初衷。

缩放点积注意力公式中的主要运算构成了Transformer中自注意力机制的基本形式。

注意力(Q,K,V)=softmax(QKTdk)V\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V注意力(Q,K,V)=softmax(dk​​QKT​)V

这里，QQQ (查询)、KKK (键) 和 VVV (值) 是从输入序列嵌入 (embedding)中得到的矩阵。设 NNN 为序列长度，dkd\_kdk​ 为键和查询的维度，dvd\_vdv​ 为值的维度。

主要计算步骤如下：

1. **查询-键相似度计算：** 计算矩阵乘积 QKTQK^TQKT。

   - QQQ 的维度为 (N×dk)(N \times d\_k)(N×dk​)。
   - KTK^TKT 的维度为 (dk×N)(d\_k \times N)(dk​×N)。
   - 得到的注意力得分矩阵 S=QKTS = QK^TS=QKT 的维度为 (N×N)(N \times N)(N×N)。
   - 此矩阵乘法的计算成本大约为 O(N⋅dk⋅N)=O(N2dk)O(N \cdot d\_k \cdot N) = O(N^2 d\_k)O(N⋅dk​⋅N)=O(N2dk​) 浮点运算 (FLOPs)。
2. **缩放与Softmax：** 将得分按 1/dk1/\sqrt{d\_k}1/dk​​ 进行缩放，并对每行应用softmax函数。

   - 缩放涉及 N2N^2N2 次元素级乘法。
   - Softmax涉及对每个 NNN 行进行指数运算和归一化 (normalization)，每行所需的运算量与行长 NNN 成正比。总成本大约为 O(N2)O(N^2)O(N2)。
   - 与矩阵乘法相比，对于较大的 NNN 和 dkd\_kdk​，此步骤的计算量通常较小。
3. **值聚合：** 将softmax输出 (即注意力权重 (weight)矩阵 AAA，维度为 (N×N)(N \times N)(N×N)) 与值矩阵 VVV 相乘。

   - AAA 的维度为 (N×N)(N \times N)(N×N)。
   - VVV 的维度为 (N×dv)(N \times d\_v)(N×dv​)。
   - 得到的输出矩阵 O=AVO = AVO=AV 的维度为 (N×dv)(N \times d\_v)(N×dv​)。
   - 此矩阵乘法的计算成本大约为 O(N⋅N⋅dv)=O(N2dv)O(N \cdot N \cdot d\_v) = O(N^2 d\_v)O(N⋅N⋅dv​)=O(N2dv​) 浮点运算 (FLOPs)。

### 主要影响因素和总体复杂度

综合这些步骤，总计算复杂度主要取决于两次大型矩阵乘法：O(N2dk+N2dv)O(N^2 d\_k + N^2 d\_v)O(N2dk​+N2dv​)。

在许多标准Transformer配置中，dkd\_kdk​ 和 dvd\_vdv​ 的维度与整体模型嵌入 (embedding)维度 dmodeld\_{model}dmodel​ 成比例 (通常 dk=dv=dmodel/hd\_k = d\_v = d\_{model} / hdk​=dv​=dmodel​/h，其中 hhh 是注意力头的数量)。因此，复杂度通常概括为：

O(N2⋅dmodel)O(N^2 \cdot d\_{model})O(N2⋅dmodel​)

这种对序列长度 NNN 的二次方依赖是重要瓶颈。尽管这些运算在序列维度上高度并行化 (与循环模型不同)，但总运算次数随 NNN 的增长而迅速增加。

### 内存复杂度

除了计算之外，还有显著的内存需求。中间注意力得分矩阵 QKTQK^TQKT (softmax之前或之后) 的维度为 (N×N)(N \times N)(N×N)。存储该矩阵需要：

O(N2)O(N^2)O(N2)

的内存。对于长序列 (例如 N>4096N > 4096N>4096)，存储一个 (N×N)(N \times N)(N×N) 的浮点数矩阵可能会超出典型加速器 (如GPU) 的内存容量，甚至在考虑激活值、梯度和模型参数 (parameter)所需的内存之前。

01000200030004000100100010k100k1M10MO(N) (例如，线性)O(N^2) (自注意力)

> 该图表显示了计算成本的迅速分化。随着序列长度 NNN 的增加，标准自注意力 (self-attention)机制 (attention mechanism)的 O(N2)O(N^2)O(N2) 成本迅速超过线性 O(N)O(N)O(N) 增长，使其对于非常长的序列变得不切实际。Y轴使用对数刻度以适应较大的数值范围。

### 对长序列的影响

这种二次方计算和内存复杂度严重限制了标准Transformer应用于涉及极长序列的任务，例如：

- 处理被视为图像块序列的整个高分辨率图像。
- 分析完整的长篇文档或书籍。
- 对长篇时间序列数据建模 (例如，音频、传感器读数)。
- 处理基因组序列。

即使配备强大的硬件，序列长度超过几千个token后，在训练和推理 (inference)期间，从运行时和内存角度来看都会变得困难。此限制直接促使了本章后续部分讨论的其他注意力机制 (attention mechanism)和架构变体的发展，这些旨在减少这种二次方依赖。

获取即时帮助、个性化解释和交互式代码示例。

---

### 稀疏注意力机制

# 稀疏注意力机制

标准Transformer的计算瓶颈在于自注意力 (self-attention)机制 (attention mechanism)。计算注意力权重 (weight)需要将序列中的每个词元 (token)（查询）与所有其他词元（键）进行比较。这会导致计算和内存复杂度达到O(N2d)O(N^2 d)O(N2d)，其中NNN是序列长度，ddd是模型维度。虽然当ddd被视为常数时，复杂度常被简化为O(N2)O(N^2)O(N2)，但这种平方级增长使得处理非常长的序列（数千或数万个词元）在计算上难以承受。

稀疏注意力机制旨在缓解这一瓶颈，通过减少需要计算的查询-键对的数量。稀疏注意力不是让每个词元关注*所有*其他词元，而是将注意力限制在精心选择的词元*子集*上，从而形成一个稀疏注意力矩阵。核心设想是，完全注意力通常是多余的，对于给定词元，大部分相关信息可以通过关注更小、策略性选择的其他词元集来获取。

### 为何采用稀疏性？

经验分析常表明，训练有素的Transformer中学习到的注意力矩阵确实是稀疏的。许多注意力权重 (weight)接近零，这表明一个词元 (token)只强烈关注有限数量的其他词元。稀疏注意力方法试图借助这一观察结果，通过预定义或学习模式来将计算集中在潜在的重要关系上，从而大幅减少N2N^2N2次比较。

### 常见的稀疏模式

已提出几种结构化的稀疏模式，这些模式通常结合不同类型的注意力，以平衡局部上下文 (context)与更宽广的远程交互。

#### 1. 滑动窗口（局部）注意力

最直观的模式是局部或滑动窗口注意力。在此，每个词元 (token)只关注一个固定大小为kkk的邻近词元窗口（例如，左侧k/2k/2k/2个词元和右侧k/2k/2k/2个词元）。

G

clusterₛeq

序列（词元位置）

clusterₐttn

词元4的注意力（窗口k=4）

1

1

2

2

3

3

4

4

4->1

5

5

7

7

4->7

attn2

2

4->attn2

关注

attn3

3

4->attn3

关注

attn4

4

4->attn4

关注

attn5

5

4->attn5

关注

attn6

6

4->attn6

关注

6

6

> 局部注意力模式，其中词元4关注自身及其窗口大小内的邻近词元（此处k=4，包括左侧2个，右侧2个，以及自身）。计算被限制在这些邻近词元上。

这会将每个词元的复杂度从O(N)O(N)O(N)降至O(k)O(k)O(k)，从而使总复杂度变为O(N⋅k)O(N \cdot k)O(N⋅k)。由于kkk是一个通常远小于NNN的固定超参数 (parameter) (hyperparameter)，因此这在NNN上是有效的线性复杂度。然而，明显的局限性是信息无法在单个注意力层内传播超出窗口大小kkk。获取远程依赖关系需要堆叠许多这样的层。

#### 2. 扩张（步进）注意力

为了减轻局部注意力感受野受限的问题而不牺牲效率，可以使用扩张或步进注意力。类似于扩张卷积，一个词元关注具有递增间隔或步长的邻近词元。例如，一个词元可能关注位置i±1i \pm 1i±1、i±2i \pm 2i±2、i±4i \pm 4i±4、i±8i \pm 8i±8等。

G

clusterₛeq

序列（词元位置）

clusterₐttn

词元5的注意力（扩张）

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

5->2

6

6

8

8

5->8

attn1

1 (间隔4)

5->attn1

关注

attn3

3 (间隔2)

5->attn3

关注

attn4

4 (间隔1)

5->attn4

关注

attn5

5 (自身)

5->attn5

关注

attn6

6 (间隔1)

5->attn6

关注

attn7

7 (间隔2)

5->attn7

关注

attn9

9 (间隔4)

5->attn9

关注

7

7

9

9

> 扩张（步进）注意力模式，其中词元5关注自身、紧邻词元（间隔1）、间隔2的词元以及间隔4的词元。这允许以每词元固定的计算量覆盖更宽广的感受野。

这使得感受野可以随层数呈指数级增长，使得捕获更长距离的依赖关系比纯局部注意力更高效，同时保持O(N⋅kdilated)O(N \cdot k\_{dilated})O(N⋅kdilated​)的复杂度，其中kdilatedk\_{dilated}kdilated​是关注位置的数量（通常与NNN呈对数关系）。

#### 3. 全局 + 局部注意力

许多成功的稀疏注意力模型，如Longformer和BigBird，将局部注意力与少量预先选择的`全局`词元结合起来。这些全局词元可以关注序列中的所有其他词元，并且所有其他词元也可以关注它们。

- **局部注意力：** 大多数词元使用滑动窗口注意力来获取局部上下文。
- **全局注意力：** 某些词元（例如，像`[CLS]`这样的特殊词元，或被识别为特别重要的词元）具有完全注意力能力。这确保了信息可以全局汇聚并分发回局部上下文。

这种混合方法旨在兼顾两者的优点：通过局部注意力为大多数词元提供线性扩展，以及通过全局词元获取重要的远程依赖的能力。其复杂度通常由局部注意力部分主导，保持接近O(N⋅k)O(N \cdot k)O(N⋅k)，前提是全局词元的数量相对于NNN较小。

### 实现考虑与权衡

实现稀疏注意力通常需要比标准注意力的密集矩阵乘法更复杂的索引操作。不是计算完整的N×NN \times NN×N注意力矩阵，而是必须收集或计算与允许的稀疏连接相对应的特定索引。库和框架越来越多地为常见的稀疏模式（例如，块稀疏操作）提供优化实现。

**优点：**

- **复杂度降低：** 计算成本从O(N2)O(N^2)O(N2)显著降至通常接近O(N)O(N)O(N)，从而能够处理更长的序列。
- **内存减少：** 内存占用更低，因为不再实例化完整的N×NN \times NN×N矩阵。

**缺点：**

- **潜在信息损失：** 如果预定义的稀疏模式遗漏了与任务相关的某些重要远程依赖，性能可能会比完全注意力下降（假设有足够的计算资源用于完全注意力）。
- **模式设计：** 稀疏模式（窗口大小、扩张率、全局词元 (token)）的选择成为一组需要调整的重要超参数 (parameter) (hyperparameter)。
- **实现复杂性：** 比标准密集注意力更难以正确高效地实现。

稀疏注意力代表着重要一步，使Transformer能够应用于涉及超长文档、高分辨率图像（被视为补丁序列）或扩展时间序列的任务。这是修改核心架构以克服固有扩展限制的一个典型例子，我们将在接下来的章节中通过近似技术继续讨论这一主题。

获取即时帮助、个性化解释和交互式代码示例。

---

### 近似注意力机制：线性Transformer

# 近似注意力机制：线性Transformer

标准自注意力 (self-attention)机制 (attention mechanism)的主要计算瓶颈是其相对于输入序列长度 NNN 的二次复杂度。通过计算注意力得分矩阵 A=softmax(QKTdk)A = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)A=softmax(dk​​QKT​) 涉及计算 N×NN \times NN×N 矩阵 QKTQK^TQKT。这需要查询（query）向量 (vector)和键（key）向量之间进行 N2N^2N2 次点积（每个向量维度为 dkd\_kdk​），接着进行缩放、softmax归一化 (normalization)，最后乘以值（value）矩阵 VVV。整体复杂度通常由 QKTQK^TQKT 的计算主导，导致注意力矩阵本身的计算时间复杂度为 O(N2dk)O(N^2 d\_k)O(N2dk​)，空间复杂度为 O(N2)O(N^2)O(N2)。这种随规模增长的特性使得将标准Transformer应用于超长序列在计算上变得难以承受（例如，被视为补丁序列的高分辨率图像、长文档或基因组数据）。

为克服此局限，一项重要的研究方向是开发 *线性Transformer*，这些变体以线性 O(N)O(N)O(N) 的时间复杂度与内存复杂度来近似自注意力机制。核心思想是避免显式计算和存储完整的 N×NN \times NN×N 注意力矩阵。

### Softmax的难点

如果我们考虑不带softmax归一化 (normalization)的注意力计算，Ounnorm=(QKT)VO\_{unnorm} = (QK^T)VOunnorm​=(QKT)V，矩阵乘法的结合律允许我们将计算重排为 Ounnorm=Q(KTV)O\_{unnorm} = Q(K^TV)Ounnorm​=Q(KTV)。在此，KTK^TKT 是 dk×Nd\_k \times Ndk​×N 维矩阵，VVV 是 N×dvN \times d\_vN×dv​ 维矩阵。计算 KTVK^TVKTV 需要 O(Ndkdv)O(N d\_k d\_v)O(Ndk​dv​) 次运算。接着，将结果乘以 QQQ (N×dkN \times d\_kN×dk​) 又需要 O(Ndkdv)O(N d\_k d\_v)O(Ndk​dv​) 次运算。如果嵌入 (embedding)维度 d=dk=dvd = d\_k = d\_vd=dk​=dv​ 被视为固定值或远小于 NNN，则整体复杂度在 NNN 上呈线性关系。

然而，按行进行的softmax函数，定义为：
softmax(Z)i=exp⁡(Zi)∑jexp⁡(Zj)\text{softmax}(Z)\_i = \frac{\exp(Z\_i)}{\sum\_j \exp(Z\_j)}softmax(Z)i​=∑j​exp(Zj​)exp(Zi​)​
其中 ZiZ\_iZi​ 代表输入矩阵的第 iii 行（在此处，是 QKTdk\frac{QK^T}{\sqrt{d\_k}}dk​​QKT​ 的第 iii 行），阻碍了这种简单的重排。归一化项 ∑jexp⁡(… )\sum\_j \exp(\dots)∑j​exp(…) 将一行中的所有元素耦合在一起，似乎需要为给定查询计算所有 NNN 个点积，然后才能进行归一化和与值相乘的操作。

### 线性化策略

线性Transformer的变体采用不同的策略，以近似注意力机制 (attention mechanism)并绕过softmax瓶颈，有效地实现了类似于 Q(KTV)Q(K^TV)Q(KTV) 的重排：

1. **核函数近似：** 该方法近似点积相似度，然后进行softmax中固有的指数函数计算，通常将 exp⁡(qiTkj/dk)\exp(q\_i^T k\_j / \sqrt{d\_k})exp(qiT​kj​/dk​​) 视为核函数。如果这个核函数可以通过特征映射 ϕ(⋅)\phi(\cdot)ϕ(⋅) 的内积来近似，即 K(qi,kj)≈ϕ(qi)Tϕ(kj)K(q\_i, k\_j) \approx \phi(q\_i)^T \phi(k\_j)K(qi​,kj​)≈ϕ(qi​)Tϕ(kj​)，那么注意力求和 ∑jK(qi,kj)vj\sum\_j K(q\_i, k\_j) v\_j∑j​K(qi​,kj​)vj​ 可以重写为：
   ∑j(ϕ(qi)Tϕ(kj))vj=ϕ(qi)T(∑jϕ(kj)vjT)\sum\_j \left( \phi(q\_i)^T \phi(k\_j) \right) v\_j = \phi(q\_i)^T \left( \sum\_j \phi(k\_j) v\_j^T \right)∑j​(ϕ(qi​)Tϕ(kj​))vj​=ϕ(qi​)T(∑j​ϕ(kj​)vjT​)
   （注意：此处需要仔细处理向量 (vector)/矩阵的维度，这仅演示了重排方式）。
   项 ∑jϕ(kj)vjT\sum\_j \phi(k\_j) v\_j^T∑j​ϕ(kj​)vjT​ 计算基于键和值的聚合表示。这个聚合可以计算一次（在 NNN 上呈线性），然后乘以每个查询的特征映射 ϕ(qi)\phi(q\_i)ϕ(qi​)（同样，在 NNN 上呈线性）。难点在于找到合适的特征映射 ϕ\phiϕ，使其能够准确近似指数核函数，同时允许这种分解。**Performer** 等模型使用基于随机特征的技术（特别是通过正交随机特征实现快速注意力，即FAVOR+）来构建这些映射，并对近似质量提供理论保证。这些方法的目标是达到 O(N)O(N)O(N) 复杂度。我们将在“基于核函数的注意力近似（Performer）”一节中更详细地考察基于核函数的方法。
2. **低秩投影：** 该策略基于注意力矩阵 A=softmax(QKT/dk)A = \text{softmax}(QK^T / \sqrt{d\_k})A=softmax(QKT/dk​​) 通常是低秩的，或可以被低秩矩阵很好地近似的观察或假设。**Linformer** 模型通过引入可学习的投影矩阵 EEE (k×Nk \times Nk×N) 和 FFF (k×Nk \times Nk×N) 来体现这一点，其中 kkk 是一个固定维度，远小于 NNN (k≪Nk \ll Nk≪N)。它计算投影后的键矩阵 K′=EKK' = EKK′=EK 和值矩阵 V′=FVV' = FVV′=FV。这些投影矩阵的维度分别为 k×dkk \times d\_kk×dk​ 和 k×dvk \times d\_vk×dv​。注意力机制随后使用这些投影矩阵进行操作，例如，通过基于 Q(K′)TQ(K')^TQ(K′)T (N×kN \times kN×k) 计算注意力权重 (weight)，并将其应用于 V′V'V′ (k×dvk \times d\_vk×dv​)。复杂度从 O(N2)O(N^2)O(N2) 降低到 O(Nk)O(Nk)O(Nk)，如果 kkk 是常数或随 NNN 亚线性增长，则实现了线性扩展。这种方法将在“低秩投影方法（Linformer）”一节中详细说明。

### 影响与权衡

线性注意力机制 (attention mechanism)在计算效率方面提供了显著优势，尤其对于长序列。

| 特性 | 标准注意力 | 线性注意力（近似） |
| --- | --- | --- |
| 时间复杂度 | O(N2d)O(N^2 d)O(N2d) | O(Ndr)O(N d r)O(Ndr) （其中 rrr 取决于方法，例如特征维度或投影大小 kkk） |
| 内存（注意力矩阵） | O(N2)O(N^2)O(N2) | O(Nr)O(Nr)O(Nr) 或更少 |
| 计算 | 显式 N×NN \times NN×N 矩阵 | 避免显式 N×NN \times NN×N 矩阵 |
| 精确性 | 精确 | 近似 |
| 适用性 | 短到中等长度序列 | 长序列 |

主要的权衡在于，这些方法计算的是完整自注意力 (self-attention)机制的*近似*。虽然实证结果常表明这些近似表现出色，有时甚至在特定任务上胜过标准注意力（可能归因于隐式正则化 (regularization)效应），但它们可能无法捕捉与原始公式完全相同的关系信息。近似质量及其对下游任务性能的影响必须通过实证评估。

通过将计算负担从二次降低到线性，这些方法显著扩展了Transformer架构的适用范围，到以前受序列长度限制阻碍的领域，使模型能够更有效地处理整个文档、高分辨率图像或长时间序列数据。

获取即时帮助、个性化解释和交互式代码示例。

---

### 基于核的注意力近似 (Performers模型)

# 基于核的注意力近似 (Performers模型)

尽管标准自注意力 (self-attention)提供了出色的序列建模能力，但其二次方的计算和内存复杂度（序列长度为 NNN 时为 O(N2)O(N^2)O(N2)）仍然是一个主要的限制，尤其是在处理高分辨率图像生成、文档摘要或基因组数据分析等场景中遇到的长序列时。Choromanski 等人 (2020) 提出的 Performer 架构，通过以线性复杂度近似注意力机制 (attention mechanism)，提供了一种有效的解决办法。它通过一种巧妙的技术，称为“通过正交随机特征实现快速注意力”（FAVOR+），实现了这一点。

### 限制：显式注意力矩阵计算

回顾标准缩放点积注意力：

注意力(Q,K,V)=softmax(QKTdk)V\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V注意力(Q,K,V)=softmax(dk​​QKT​)V

这里，Q,K,VQ, K, VQ,K,V 分别是查询、键和值向量 (vector)的矩阵，每个矩阵有 NNN 行（序列长度）和 dkd\_kdk​（或 dvd\_vdv​）列（嵌入 (embedding)维度）。主要计算成本在于计算 N×NN \times NN×N 的注意力矩阵 A=softmax(QKT/dk)A = \text{softmax}(QK^T / \sqrt{d\_k})A=softmax(QKT/dk​​)。这需要 O(N2dk)O(N^2 d\_k)O(N2dk​) 次操作和 O(N2)O(N^2)O(N2) 内存，随着 NNN 的增加，这很快变得难以承受。

### 使用核与随机特征近似注意力

Performer 模型的核心思路是避免显式计算 N×NN \times NN×N 矩阵 AAA。相反，它寻求使用核方法来近似注意力机制 (attention mechanism)。让我们将注意力机制的第 iii 个输出向量 (vector)（归一化 (normalization)之前）重写为：

注意力′(Q,K,V)i=∑j=1Nexp⁡(qiTkjdk)vj\text{注意力}'(Q, K, V)\_i = \sum\_{j=1}^N \exp\left(\frac{q\_i^T k\_j}{\sqrt{d\_k}}\right) v\_j注意力′(Q,K,V)i​=j=1∑N​exp(dk​​qiT​kj​​)vj​

标准注意力输出通过使用分母 Di=∑j=1Nexp⁡(qiTkj/dk)D\_i = \sum\_{j=1}^N \exp(q\_i^T k\_j / \sqrt{d\_k})Di​=∑j=1N​exp(qiT​kj​/dk​​) 对此和进行归一化而得到。

Performer 模型利用了函数 相似度(qi,kj)=exp⁡(qiTkj/dk)\text{相似度}(q\_i, k\_j) = \exp(q\_i^T k\_j / \sqrt{d\_k})相似度(qi​,kj​)=exp(qiT​kj​/dk​​) 类似于核函数的思路，特别是在经过适当缩放和变换后的高斯核。核方法通常允许使用特征映射隐式计算相似度。Performer 模型提出寻找一个特征映射 ϕ:Rdk→Rr\phi: \mathbb{R}^{d\_k} \rightarrow \mathbb{R}^rϕ:Rdk​→Rr，其中 rrr 是随机特征的维度（通常 r≪Nr \ll Nr≪N），使得核相似度可以通过特征空间中的内积来近似：

exp⁡(qiTkjdk)≈ϕ(qi)Tϕ(kj)\exp\left(\frac{q\_i^T k\_j}{\sqrt{d\_k}}\right) \approx \phi(q\_i)^T \phi(k\_j)exp(dk​​qiT​kj​​)≈ϕ(qi​)Tϕ(kj​)

如果存在这样的特征映射 ϕ\phiϕ，注意力计算可以巧妙地重新排序，以避免 N×NN \times NN×N 的计算。未归一化的注意力求和变为：

注意力′(Q,K,V)i≈∑j=1N(ϕ(qi)Tϕ(kj))vj=ϕ(qi)T∑j=1N(ϕ(kj)vjT)\text{注意力}'(Q, K, V)\_i \approx \sum\_{j=1}^N (\phi(q\_i)^T \phi(k\_j)) v\_j = \phi(q\_i)^T \sum\_{j=1}^N (\phi(k\_j) v\_j^T)注意力′(Q,K,V)i​≈j=1∑N​(ϕ(qi​)Tϕ(kj​))vj​=ϕ(qi​)Tj=1∑N​(ϕ(kj​)vjT​)

类似地，分母可以近似为：

Di≈∑j=1Nϕ(qi)Tϕ(kj)=ϕ(qi)T∑j=1Nϕ(kj)D\_i \approx \sum\_{j=1}^N \phi(q\_i)^T \phi(k\_j) = \phi(q\_i)^T \sum\_{j=1}^N \phi(k\_j)Di​≈j=1∑N​ϕ(qi​)Tϕ(kj​)=ϕ(qi​)Tj=1∑N​ϕ(kj​)

请注意操作顺序的重要变化。我们可以首先计算求和 ∑j=1N(ϕ(kj)vjT)\sum\_{j=1}^N (\phi(k\_j) v\_j^T)∑j=1N​(ϕ(kj​)vjT​)（一个 r×dvr \times d\_vr×dv​ 矩阵）和 ∑j=1Nϕ(kj)\sum\_{j=1}^N \phi(k\_j)∑j=1N​ϕ(kj​)（一个 r×1r \times 1r×1 向量）。这些计算只需处理键和值一次，大约需要 O(Nrdk+Nrdv)O(N r d\_k + N r d\_v)O(Nrdk​+Nrdv​) 的时间。然后，对于每个查询 qiq\_iqi​，我们计算 ϕ(qi)\phi(q\_i)ϕ(qi​) 并执行两次矩阵-向量乘法，每个查询需要 O(rdk+rdv)O(r d\_k + r d\_v)O(rdk​+rdv​) 的时间。总时间复杂度近似为 O(Nr(dk+dv))O(N r (d\_k + d\_v))O(Nr(dk​+dv​))，如果 rrr 被视为常数或增长远慢于 NNN，则在 NNN 上呈线性关系。内存复杂度也降低到 O(Nr+Ndk+Ndv)O(N r + N d\_k + N d\_v)O(Nr+Ndk​+Ndv​)，主要用于存储映射后的查询、键、值和中间求和结果。

G

cluster₀

标准注意力计算

cluster₁

Performer (FAVOR+) 计算

Q0

查询 (N x dk)

QK0

QKᵀ (N x N)

Q0->QK0

 O(N²dk)

K0

键 (N x dk)

K0->QK0

V0

值 (N x dv)

Out0

输出 (N x dv)

V0->Out0

SoftQK0

Softmax(QKᵀ/√dk)
(N x N)

QK0->SoftQK0

SoftQK0->Out0

 O(N²dv)

Q1

查询 (N x dk)

phi\_Q

φ(Q) (N x r)

Q1->phi\_Q

 O(Nrdk)

K1

键 (N x dk)

phi\_K

φ(K) (N x r)

K1->phi\_K

 O(Nrdk)

V1

值 (N x dv)

phiK\_V

Σ φ(K) Vᵀ
(r x dv)

V1->phiK\_V

Out1

输出 (N x dv)

phi\_Q->Out1

 O(Nrdv)

phi\_K->phiK\_V

 O(Nrdv)

phiKₛum

Σ φ(K)
(r x 1)

phi\_K->phiKₛum

 O(Nr)

phiK\_V->Out1

phiKₛum->Out1

 O(Nr)

> 计算流程对比。标准注意力需要形成代价高昂的 N×NN \times NN×N 矩阵（红色节点）。Performer 模型使用特征映射 ϕ\phiϕ 来计算中间求和（黄色节点），这些操作具有线性复杂度 O(N)O(N)O(N)，从而避免了二次方的限制。

### 构建特征映射 ϕ\phiϕ

Performer 模型的有效性取决于能否找到一个合适的特征映射 ϕ\phiϕ。FAVOR+ 机制通过使用随机特征构建 ϕ\phiϕ 来实现这一点，其灵感来源于用于近似高斯核的技术（如随机傅里叶特征）。Performer 模型的一个重要贡献是开发了能保证非负性（ϕ(x)Tϕ(y)≥0\phi(x)^T \phi(y) \ge 0ϕ(x)Tϕ(y)≥0）的特征映射，这有助于保持稳定性并更好地近似 softmax 函数的特性（其输出分量是非负的）。

具体来说，Performer 模型基于随机投影和三角函数定义特征映射，例如：

ϕ(x)=h(x)m[f1(w1Tx),…,fm(wmTx),f1′(w1Tx),…,fm′(wmTx)]T\phi(x) = \frac{h(x)}{\sqrt{m}} \left[ f\_1(w\_1^T x), \dots, f\_m(w\_m^T x), f'\_1(w\_1^T x), \dots, f'\_m(w\_m^T x) \right]^Tϕ(x)=m​h(x)​[f1​(w1T​x),…,fm​(wmT​x),f1′​(w1T​x),…,fm′​(wmT​x)]T

其中 wiw\_iwi​ 是随机采样的向量 (vector)，h(x)h(x)h(x) 是像 exp⁡(∥x∥2/2)\exp(\|x\|^2 / 2)exp(∥x∥2/2) 这样的函数，而 fl,fl′f\_l, f'\_lfl​,fl′​ 是像 (sin⁡,cos⁡)(\sin, \cos)(sin,cos) 或 (exp⁡,exp⁡)(\exp, \exp)(exp,exp) 这样的对。特征映射 rrr 的维度是 2m2m2m。精确的构造确保了 softmax 核的无偏或近无偏估计，并具有非负输出。

### 优点与注意事项

**优点：**

- **线性复杂度：** 将注意力的时间和空间复杂度从 O(N2)O(N^2)O(N2) 降低到 O(N)O(N)O(N)，使得能够应用于更长的序列。
- **强大的理论保证：** 为 softmax 核提供了可证明的近似界限。
- **良好的实证表现：** 通常能达到与标准 Transformer 模型相当的性能，同时在长序列任务上效率显著提高。
- **兼容性：** 可以作为标准注意力模块的替代品进行集成，对整个 Transformer 架构的修改很小。

**注意事项：**

- **近似质量与成本：** 近似质量取决于随机特征维度 rrr。较大的 rrr 会产生更好的近似，但会增加线性复杂度中的常数因子（O(Nrd)O(N r d)O(Nrd)）。选择 rrr 需要在精度和计算成本之间进行权衡。 rrr 的典型值可能在 64 到 256 甚至更高，具体取决于任务和资源。
- **随机性：** 使用随机特征会引入随机性。然而，方差通常较低，特别是在 rrr 的选择合理时，并且 Performer 模型采用正交随机特征等技术来进一步降低这种方差。在实际应用中，不同随机种子下的结果通常是稳定的。

通过使用非负随机特征近似 softmax 核，Performer 模型提供了一种计算高效且有理论依据的方法，能够将 Transformer 模型扩展到以前认为不可行的长度，显著拓宽了其应用范围。

获取即时帮助、个性化解释和交互式代码示例。

---

### 低秩投影方法（Linformer）

# 低秩投影方法（Linformer）

标准自注意力 (self-attention)机制 (attention mechanism)的二次复杂度，即 O(N2d)O(N^2 d)O(N2d)（其中 NNN 是序列长度，ddd 是模型维度），在处理长序列时是一个主要瓶颈。尽管功能强大，但随着 NNN 的增加，计算每对标记 (token)之间的注意力分数会变得计算成本过高。

有几种方法旨在减轻这种计算负担。一个重要方向是利用低秩投影来近似注意力机制，Linformer 模型就是其中的一个例子。

### 低秩假设

Linformer 的核心思路是基于这样一个假设：尽管自注意力 (self-attention)机制 (attention mechanism)有能力模拟整个序列中复杂的关联，但它通常可以通过一个低秩矩阵进行近似。本质上，N×NN \times NN×N 的注意力矩阵 P=Softmax(QKTdk)P = Softmax(\frac{QK^T}{\sqrt{d\_k}})P=Softmax(dk​​QKT​) 可能在结构上存在冗余。这意味着从输入上下文 (context)（由值 VVV 表示）到输出上下文（加权和 PVPVPV）的映射不一定需要任意 N×NN \times NN×N 矩阵的完整表达能力。如果这个假设成立，我们就可以通过使用压缩表示来大幅提高效率。

### Linformer：键和值的投影

Linformer（线性 Transformer）提出了一种巧妙的方法来实现线性复杂度，即引入学习到的投影矩阵 EiE\_iEi​ 和 FiF\_iFi​。Linformer 不计算完整的 N×NN \times NN×N 注意力矩阵，而是在注意力计算*之前*沿序列长度维度投影键 (KKK) 和值 (VVV) 矩阵。

设输入序列长度为 NNN，头维度为 dkd\_kdk​（用于键/查询）或 dvd\_vdv​（用于值）。原始矩阵为：

- 查询 Q∈RN×dkQ \in \mathbb{R}^{N \times d\_k}Q∈RN×dk​
- 键 K∈RN×dkK \in \mathbb{R}^{N \times d\_k}K∈RN×dk​
- 值 V∈RN×dvV \in \mathbb{R}^{N \times d\_v}V∈RN×dv​

Linformer 引入了两个投影矩阵 E∈Rk×NE \in \mathbb{R}^{k \times N}E∈Rk×N 和 F∈Rk×NF \in \mathbb{R}^{k \times N}F∈Rk×N，其中 kkk 是一个投影维度，远小于 NNN (k≪Nk \ll Nk≪N)。这些矩阵用于创建投影键矩阵和投影值矩阵：

Kproj=EK(维度 k×dk)K\_{proj} = E K \quad (\text{维度 } k \times d\_k)Kproj​=EK(维度 k×dk​)
Vproj=FV(维度 k×dv)V\_{proj} = F V \quad (\text{维度 } k \times d\_v)Vproj​=FV(维度 k×dv​)

请注意，投影如何将序列维度从 NNN 降低到 kkk。主要步骤是注意力分数现在在原始查询矩阵 QQQ 和*投影*键矩阵 KprojK\_{proj}Kproj​ 之间计算：

Pproj=Softmax(QKprojTdk)(维度 N×k)P\_{proj} = Softmax\left(\frac{Q K\_{proj}^T}{\sqrt{d\_k}}\right) \quad (\text{维度 } N \times k)Pproj​=Softmax(dk​​QKprojT​​)(维度 N×k)

最终输出通过将这个投影注意力分数矩阵 PprojP\_{proj}Pproj​ 乘以*投影*值矩阵 VprojV\_{proj}Vproj​ 获得：

AttentionLinformer(Q,K,V)=PprojVproj(维度 N×dv)Attention\_{Linformer}(Q, K, V) = P\_{proj} V\_{proj} \quad (\text{维度 } N \times d\_v)AttentionLinformer​(Q,K,V)=Pproj​Vproj​(维度 N×dv​)

### 复杂度分析

让我们分析计算复杂度。原始注意力计算主要由 QKTQ K^TQKT 矩阵乘法决定，这需要 O(N2dk)O(N^2 d\_k)O(N2dk​) 时间，以及随后与 VVV 的乘法，这需要 O(N2dv)O(N^2 d\_v)O(N2dv​) 时间。

在 Linformer 中：

1. 投影键 K：EKE KEK 需要 O(Nkdk)O(Nk d\_k)O(Nkdk​) 时间。
2. 投影值 V：FVF VFV 需要 O(Nkdv)O(Nk d\_v)O(Nkdv​) 时间。
3. 计算 QKprojTQ K\_{proj}^TQKprojT​：这涉及将一个 N×dkN \times d\_kN×dk​ 矩阵乘以一个 dk×kd\_k \times kdk​×k 矩阵，结果需要 O(Nkdk)O(Nk d\_k)O(Nkdk​) 时间。
4. 计算 PprojVprojP\_{proj} V\_{proj}Pproj​Vproj​：这涉及将一个 N×kN \times kN×k 矩阵乘以一个 k×dvk \times d\_vk×dv​ 矩阵，结果需要 O(Nkdv)O(Nk d\_v)O(Nkdv​) 时间。

由于 kkk 被选择为使得 k≪Nk \ll Nk≪N，Linformer 注意力机制 (attention mechanism)的总体复杂度变为 O(Nk)O(Nk)O(Nk)，这相对于序列长度 NNN 是线性的。这是对标准 O(N2)O(N^2)O(N2) 复杂度的大幅改进。

LinformerVsStandard

clusterₛtandard

标准注意力 (O(N²))

clusterₗinformer

Linformer 注意力 (O(Nk))

Q

查询 (N x dk)

QK\_T

Q @ Kᵀ (N x N)

Q->QK\_T

K

键 (N x dk)

KT

Kᵀ (dk x N)

K->KT

V

值 (N x dv)

Attn\_Std

输出 (P @ V)
(N x dv)

V->Attn\_Std

KT->QK\_T

Scale

缩放与Softmax

QK\_T->Scale

P

注意力矩阵 P (N x N)

Scale->P

P->Attn\_Std

Q\_L

查询 (N x dk)

QKₚrojT

Q @ Kₚrojᵀ
(N x k)

Q\_L->QKₚrojT

K\_L

键 (N x dk)

Kₚroj

投影键 (E @ K)
(k x dk)

K\_L->Kₚroj

V\_L

值 (N x dv)

Vₚroj

投影值 (F @ V)
(k x dv)

V\_L->Vₚroj

E

投影矩阵 E (k x N)

E->Kₚroj

F

投影矩阵 F (k x N)

F->Vₚroj

KₚrojT

Kₚrojᵀ (dk x k)

Kₚroj->KₚrojT

Attn\_L

输出 (P' @ Vₚroj)
(N x dv)

Vₚroj->Attn\_L

KₚrojT->QKₚrojT

Scale\_L

缩放与Softmax

QKₚrojT->Scale\_L

Pₚroj

投影注意力矩阵 P'
(N x k)

Scale\_L->Pₚroj

Pₚroj->Attn\_L

> 标准自注意力 (self-attention)与 Linformer 投影注意力的计算流程比较。Linformer 引入了投影矩阵 (E, F)，以在计算注意力分数之前降低键和值沿着序列长度轴的维度。

### 实现考量

- **投影矩阵：** 投影矩阵 EEE 和 FFF 通常在训练期间学习。它们可以在不同的注意力头甚至层之间共享，以进一步减少参数 (parameter)数量。一个常见实现是使用简单的线性层作用于转置的键和值矩阵 (KTK^TKT, VTV^TVT)，以高效地执行投影。
- **k 的选择：** 投影维度 kkk 是一个超参数 (hyperparameter)。通常使用 128、256 或 512 等值，这些值远小于典型序列长度（数千或数万），Linformer 在这些情况下变得有优势。这个选择影响计算效率和模型准确性之间的权衡。较小的 kkk 运算更快，但可能导致更大的近似误差。
- **理论保证：** Linformer 论文提供了理论分析，表明在某些假设下，自注意力 (self-attention)矩阵确实是低秩的，并且可以通过这种投影方法很好地近似。

### 优点和缺点

**优点：**

- **线性时间复杂度：** 将注意力时间复杂度从 O(N2)O(N^2)O(N2) 降低到 O(Nk)O(Nk)O(Nk)。
- **线性空间复杂度：** 减少了存储注意力矩阵所需的内存占用。
- **可扩展性：** 能够处理比标准 Transformer 长得多的序列。
- **修改简单：** 通过增加投影层，实现起来相对简单。

**缺点：**

- **近似误差：** 作为一种近似方法，它可能无法捕获完整注意力矩阵的所有细节，可能导致在某些任务上性能下降，特别是对于 N2N^2N2 计算量可控的较短序列。
- **超参数 (parameter) (hyperparameter)调优：** 需要调整投影维度 kkk。
- **低秩假设：** 其有效性依赖于底层假设，即注意力矩阵可以很好地由低秩结构近似，但这对于所有数据或任务可能并非都同样适用。

Linformer 代表着构建更高效 Transformer 模型的一个重要步骤。通过质疑完整二次注意力计算的必要性，并借助低秩近似的思路，它提供了一种实用的方法来扩展 Transformer 到更长的序列，为涉及大量文档、高分辨率图像或长篇音频的应用带来了可能性。

获取即时帮助、个性化解释和交互式代码示例。

---

### Transformer-XL：分段循环

# Transformer-XL：分段循环

标准的Transformer架构以固定长度、独立的块或窗口处理序列，但这种方式会遇到一个明显局限性，即上下文 (context)碎片化。在处理长度超出固定窗口限制的长文档或序列数据流时，模型必须将输入分成独立的分段进行处理。在处理新分段时，来自前序分段的信息通常会丢失，这阻碍了模型捕获跨越分段边界的长距离依赖关系的能力。

Transformer-XL（意为具有超长上下文的Transformer）通过引入分段级别的循环机制直接解决了这一限制。Transformer-XL不再孤立地处理每个分段，而是重复使用从先前分段计算得到的隐藏状态。

### 分段级别状态复用

主要思想简单但有效。当模型处理一个分段（例如分段 τ\tauτ）时，它会在每个层计算一系列隐藏状态，这与标准Transformer类似。这些隐藏状态随后会被缓存或存储在内存中。当模型移动到下一个分段 τ+1\tau+1τ+1 时，各层不仅可以关注当前分段 τ+1\tau+1τ+1 内的隐藏状态，还可以关注来自前一个分段 τ\tauτ 的缓存隐藏状态。

设 hτn∈RL×dh\_{\tau}^n \in \mathbb{R}^{L \times d}hτn​∈RL×d 表示第 τ\tauτ 个分段中第 nnn 个Transformer层产生的隐藏状态序列，其中 LLL 是分段长度，ddd 是隐藏维度。在计算下一个分段的隐藏状态 hτ+1nh\_{\tau+1}^nhτ+1n​ 时，第 (n)(n)(n) 层接收源自 hτ+1n−1h\_{\tau+1}^{n-1}hτ+1n−1​（当前分段下方层的输出）和 hτn−1h\_{\tau}^{n-1}hτn−1​（前一个分段下方层的输出）的输入。

具体而言，分段 τ+1\tau+1τ+1 中层 nnn 的扩展上下文 (context)是通过将来自前一个分段的缓存状态与来自当前分段的状态沿着序列长度维度连接而形成的：

h~τ+1n−1=[SG(hτn−1)∘hτ+1n−1]\tilde{h}\_{\tau+1}^{n-1} = [SG(h\_{\tau}^{n-1}) \circ h\_{\tau+1}^{n-1}]h~τ+1n−1​=[SG(hτn−1​)∘hτ+1n−1​]

这里，SG(⋅)SG(\cdot)SG(⋅) 表示一个停止梯度操作，这意味着梯度不会通过缓存状态 hτn−1h\_{\tau}^{n-1}hτn−1​ 反向传播 (backpropagation)。这很要紧：它防止了计算图变得过长，并避免了相关的优化困难。∘\circ∘ 运算符表示沿着序列长度维度的连接。

层 nnn 内的注意力机制 (attention mechanism)随后仅基于当前分段的表示 hτ+1n−1h\_{\tau+1}^{n-1}hτ+1n−1​ 计算其查询 (QQQ)，而键 (KKK) 和值 (VVV) 则源自扩展上下文 h~τ+1n−1\tilde{h}\_{\tau+1}^{n-1}h~τ+1n−1​：

Qτ+1n=hτ+1n−1WqnKτ+1n=h~τ+1n−1WknVτ+1n=h~τ+1n−1WvnQ\_{\tau+1}^n = h\_{\tau+1}^{n-1} W\_q^n \\
K\_{\tau+1}^n = \tilde{h}\_{\tau+1}^{n-1} W\_k^n \\
V\_{\tau+1}^n = \tilde{h}\_{\tau+1}^{n-1} W\_v^nQτ+1n​=hτ+1n−1​Wqn​Kτ+1n​=h~τ+1n−1​Wkn​Vτ+1n​=h~τ+1n−1​Wvn​
AttentionOutputτ+1n=Attention(Qτ+1n,Kτ+1n,Vτ+1n)AttentionOutput\_{\tau+1}^n = Attention(Q\_{\tau+1}^n, K\_{\tau+1}^n, V\_{\tau+1}^n)AttentionOutputτ+1n​=Attention(Qτ+1n​,Kτ+1n​,Vτ+1n​)

这使得当前分段 τ+1\tau+1τ+1 中的每个位置都能够关注其自身内部以及前一个分段 τ\tauτ 中的所有位置，有效地将每一步可用的上下文长度加倍，同时不跨分段边界传播梯度。

G

Sₜau\_Input

输入词元 τ

Sₜau\_Layers

Transformer 层 (n-1, n)

Sₜau\_Input->Sₜau\_Layers

处理

Sₜau\_Hidden

隐藏状态 h\_τ

Sₜau\_Layers->Sₜau\_Hidden

Sₜau1\_Layers

Transformer 层 (n-1, n)

Sₜau\_Hidden->Sₜau1\_Layers

缓存状态（内存）
[停止梯度]

Sₜau1\_Input

输入词元 τ+1

Sₜau1\_Input->Sₜau1\_Layers

处理

Sₜau1\_Hidden

隐藏状态 h\_τ+1

Sₜau1\_Layers->Sₜau1\_Hidden

> Transformer-XL 中的信息流。来自分段 τ\tauτ 的隐藏状态被缓存，并用作处理分段 τ+1\tau+1τ+1 的扩展上下文，同时不通过缓存反向传播梯度。

### 相对位置编码 (positional encoding)

状态复用机制给标准位置编码（如第4章中描述的正弦或学习到的绝对嵌入 (embedding)）带来了挑战。如果我们只是简单地将相同的绝对位置编码添加到每个分段，那么一个位置索引（例如，第10个词元 (token)）将具有相同的编码，无论它是第一个分段的第10个词元还是第二个分段的第10个词元。这种位置歧义性使得模型难以区分跨分段的时间顺序。

Transformer-XL 通过采用一种**相对位置编码**方案来解决这个问题。它不编码词元的绝对位置 iii，而是编码位于位置 iii 的查询词元与位于位置 jjj 的词元之间的相对距离（或偏移）i−ji-ji−j。这种相对信息被直接注入到注意力分数计算中。

在查询 qiq\_iqi​ 和键 kjk\_jkj​ 的标准自注意力 (self-attention)分数计算中，我们计算 qiTkjq\_i^T k\_jqiT​kj​。在带有相对位置编码的Transformer-XL中，这个计算被修改为包含仅依赖于相对距离 i−ji-ji−j 的项。具体公式涉及将键向量 (vector)中的绝对位置信息替换为相对位置嵌入。这确保了注意力机制 (attention mechanism)能够知晓词元之间的距离，而不受它们在可能非常长的、分段处理的序列中绝对位置的影响。

### Transformer-XL的优势

引入分段级别循环和相对位置编码 (positional encoding)带来了多项益处：

1. **建模更长距离的依赖关系：** 通过复用状态，Transformer-XL 可以学习跨越比单个分段长度更远距离的依赖关系，缓解了上下文 (context)碎片化问题。其有效上下文长度可以明显大于标准Transformer在训练时因内存限制而能达到的长度。
2. **更快的评估：** 在推理 (inference)或评估期间，前一个分段的隐藏状态可以计算一次并重复用于后续分段。与标准Transformer中为实现类似效果而重新处理重叠分段的朴素方法相比，这避免了冗余计算，从而带来明显的加速，特别是对于自回归 (autoregressive)生成任务。
3. **提高连贯性：** 能够访问过去上下文使得生成更加连贯，并能更好地理解长文本 (long context)或时间序列。

Transformer-XL 代表了使Transformer能够有效地处理更长序列的重要一步，为涉及长篇文档、文章或连续数据流的应用创造了途径，在这些应用中，保持长距离连贯性很要紧。尽管它引入了缓存状态的开销，但在建模能力和评估速度方面的益处对于特定任务通常会超过此成本。

获取即时帮助、个性化解释和交互式代码示例。

---

### 相对位置编码

# 相对位置编码

尽管绝对位置编码 (positional encoding)为Transformer提供了理解序列顺序的方式，但它们独立处理每个位置。正弦编码为隐式建模相对距离提供了良好的特性，但它是在注意力机制 (attention mechanism)运行*之前*添加的。学习到的绝对嵌入 (embedding)可能难以泛化到训练时未见的更长序列。另一种方法是将token之间的*相对距离*直接纳入注意力计算本身。这是相对位置编码（RPE）背后的主要思想。

直观来看，两个词之间的关系通常更多地取决于它们相距多远，而非它们在序列中的绝对位置。例如，知道一个动词紧跟其主语一个位置，可能比知道主语在位置5、动词在位置6更具泛化性。RPE旨在让模型直接感知这些相对距离。

### 修改注意力得分

RPE没有将位置信息添加到输入嵌入 (embedding)中，而是修改了自注意力 (self-attention)得分机制。标准缩放点积注意力计算位置 iii 的查询 qiq\_iqi​ 与位置 jjj 的键 kjk\_jkj​ 之间的得分，如下所示：

得分(qi,kj)=qiTkjdk\text{得分}(q\_i, k\_j) = \frac{q\_i^T k\_j}{\sqrt{d\_k}}得分(qi​,kj​)=dk​​qiT​kj​​

其中 qi=xiWQq\_i = x\_i W^Qqi​=xi​WQ 和 kj=xjWKk\_j = x\_j W^Kkj​=xj​WK， xi,xjx\_i, x\_jxi​,xj​ 是输入嵌入，WQ,WKW^Q, W^KWQ,WK 是投影矩阵。

相对定位方案将有关 iii 和 jjj 之间关系的信息直接注入此计算。存在几种变体，但它们通常涉及添加取决于相对距离 i−ji-ji−j 的项。

### Shaw 等人 (2018) 的公式

一种较早且有影响的方法提出在点积之前，将学习到的相对位置嵌入 (embedding)直接添加到键（有时是值）中。令 aijKa\_{ij}^KaijK​ 和 aijVa\_{ij}^VaijV​ 表示与查询 iii 和键/值 jjj 之间的相对位置相对应的可学习嵌入向量 (vector)。注意力得分计算修改为：

eij=(xiWQ)T(xjWK+aijK)dke\_{ij} = \frac{(x\_i W^Q)^T (x\_j W^K + a\_{ij}^K)}{\sqrt{d\_k}}eij​=dk​​(xi​WQ)T(xj​WK+aijK​)​

接着，使用类似的修改对值向量计算输出值 ziz\_izi​：

zi=∑jsoftmax(eij)(xjWV+aijV)z\_i = \sum\_j \text{softmax}(e\_{ij}) (x\_j W^V + a\_{ij}^V)zi​=j∑​softmax(eij​)(xj​WV+aijV​)

这里，aijKa\_{ij}^KaijK​ 和 aijVa\_{ij}^VaijV​ 通常通过相对距离 j−ij-ij−i 从嵌入查找表中检索。为了使嵌入数量可控，相对距离通常被裁剪到最大值 kkk。也就是说，所有 j−i>kj-i > kj−i>k 的距离都映射到相同的嵌入 ai,i+kKa\_{i, i+k}^Kai,i+kK​，而 j−i<−kj-i < -kj−i<−k 的距离则映射到 ai,i−kKa\_{i, i-k}^Kai,i−kK​。

这种方法直接将相对空间偏差注入注意力得分。然而，它需要在注意力矩阵计算中为每个查询-键对计算和存储这些相对嵌入，这可能导致计算量大。

### Transformer-XL / Dai 等人 (2019) 的公式

随 Transformer-XL 一同引入的一种更高效且被广泛采用的方法，重新构建了注意力计算，以巧妙地纳入相对位置。回顾涉及绝对位置嵌入 (embedding) Pi,PjP\_i, P\_jPi​,Pj​ 的标准注意力得分：

Ai,j绝对=(Exi+Pi)TWQTWK(Exj+Pj)A\_{i,j}^{\text{绝对}} = (E\_{x\_i} + P\_i)^T W^{Q T} W^K (E\_{x\_j} + P\_j)Ai,j绝对​=(Exi​​+Pi​)TWQTWK(Exj​​+Pj​)

展开后得到四个项：内容-内容 (ExiT…ExjE\_{x\_i}^T \dots E\_{x\_j}Exi​T​…Exj​​)、内容-位置 (ExiT…PjE\_{x\_i}^T \dots P\_jExi​T​…Pj​)、位置-内容 (PiT…ExjP\_i^T \dots E\_{x\_j}PiT​…Exj​​)，以及位置-位置 (PiT…PjP\_i^T \dots P\_jPiT​…Pj​)。

相对公式修改了这种展开方式：

1. \*\*替换键投影中的绝对位置 PjP\_jPj​：\*\*在涉及键位置的项中，将绝对位置 PjP\_jPj​ 替换为表示 iii 和 jjj 之间偏移量的相对位置编码 (positional encoding) Ri−jR\_{i-j}Ri−j​。 RRR 可以是固定的正弦编码矩阵（类似于原始Transformer的位置编码，但使用方式不同），也可以是学习到的嵌入。
2. \*\*引入可训练的位置偏差：\*\*将查询的绝对位置项 PiTWQTP\_i^T W^{Q T}PiT​WQT 替换为两个可训练向量 (vector) uuu 和 vvv。这些向量分别代表内容和相对位置的全局“位置偏差”。

由此得到以下分解后的注意力得分计算：

Ai,j相对=ExiTWQTWKExj⏟(a) 基于内容的+ExiTWQTWKRi−j⏟(b) 内容-相对位置+uTWKExj⏟(c) 全局内容偏差+vTWKRi−j⏟(d) 全局位置偏差A\_{i,j}^{\text{相对}} = \underbrace{E\_{x\_i}^T W^{Q T} W^K E\_{x\_j}}\_{\text{(a) 基于内容的}} + \underbrace{E\_{x\_i}^T W^{Q T} W^K R\_{i-j}}\_{\text{(b) 内容-相对位置}} + \underbrace{u^T W^K E\_{x\_j}}\_{\text{(c) 全局内容偏差}} + \underbrace{v^T W^K R\_{i-j}}\_{\text{(d) 全局位置偏差}}Ai,j相对​=(a) 基于内容的Exi​T​WQTWKExj​​​​+(b) 内容-相对位置Exi​T​WQTWKRi−j​​​+(c) 全局内容偏差uTWKExj​​​​+(d) 全局位置偏差vTWKRi−j​​​

- 项 (a) 与标准注意力中的内容交互相同。
- 项 (b) 捕获了位置 iii 的查询内容如何关联到相对位置 i−ji-ji−j。
- 项 (c) 提供了纯粹基于位置 jjj 的键内容的偏差。
- 项 (d) 提供了纯粹基于相对位置 i−ji-ji−j 的偏差。

重要的一点是，此公式可以高效实现。涉及 Ri−jR\_{i-j}Ri−j​ 的项无需显式构建所有 (i,j)(i, j)(i,j) 对的成对相对嵌入即可计算。相反，巧妙的张量操作允许同时高效地计算所有位置上的项 (b) 和 (d)。

### 实现考量

- \*\*裁剪距离：\*\*与 Shaw 等人的方法一样，在索引相对位置嵌入 (embedding) (Ri−jR\_{i-j}Ri−j​) 时，通常使用最大相对距离 kkk。这假设超长距离的相互作用可能不需要精确的距离信息。
- \*\*嵌入类型：\*\*相对位置表示 RRR 可以基于正弦函数（提供泛化能力），也可以是学习到的嵌入（可能更具表达力但需要更多数据和参数 (parameter)）。
- \*\*共享：\*\*相对位置嵌入（无论是正弦的还是学习到的）通常在不同注意力头之间共享，有时也在层之间共享，以减少参数数量。

### 相对位置编码 (positional encoding)的优势

- \*\*提高泛化能力：\*\*RPE，特别是正弦式的或带裁剪的RPE，相比于学习到的绝对位置编码，能更好地泛化到训练时未见的序列长度。模型学习基于*距离*而非*具体位置*的模式。
- \*\*直接距离建模：\*\*注意力机制 (attention mechanism)能直接感知token间的相对定位，这有利于局部语法或相对顺序很重要的任务。
- \*\*经验性成功：\*\*RPE是包括Transformer-XL、T5和DeBERTa在内的几种高性能模型的组成部分，显示了它们的实际效用。

### 相对位置编码 (positional encoding)的不足

- \*\*复杂性增加：\*\*与基线Transformer相比，注意力计算变得更复杂，尽管高效实现（如 Transformer-XL 公式）相比于朴素方法显著减轻了计算开销。
- \*\*超参数 (parameter) (hyperparameter)：\*\*引入了诸如裁剪距离 kkk 和相对编码类型（正弦 vs. 学习）的选择，这可能需要调整。

总而言之，相对位置编码通过将序列顺序信息直接嵌入 (embedding)到注意力机制 (attention mechanism)的得分计算中，为绝对位置编码提供了一个有吸引力的替代方案。通过侧重于成对距离而非绝对位置，它们可以提供更好的泛化能力，并更有效地捕获对距离敏感的关系，在各种现代 Transformer 架构中都很有价值。

获取即时帮助、个性化解释和交互式代码示例。

---

### 预归一化与后归一化 (预LN与后LN)

# 预归一化与后归一化 (预LN与后LN)

层归一化 (normalization)（LN）是每个Transformer块中的基本组成部分，与自注意力 (self-attention)子层和前馈子层周围的残差连接一同应用。它的主要作用是通过对每个位置的激活值在特征维度上独立进行归一化，来稳定训练过程中隐藏状态的动态变化。这有助于保持激活尺度的一致性，平滑损失曲面，并通常改善梯度流动，使得训练具有更多层级的网络成为可能。

然而，层归一化步骤相对于残差连接的 *放置位置* 显著影响训练动态和稳定性。两种主要策略是后归一化（Post-LN），用于原始论文“Attention Is All You Need”中，以及预归一化（Pre-LN），因其增强的稳定性而受到欢迎。接下来分别介绍这两种方法。

### 后归一化 (normalization) (Post-LN)

在原始Transformer架构中，层归一化应用于子层（如多头注意力 (multi-head attention)或前馈网络）的输出通过残差连接加回到输入 *之后*。

后归一化块中子层的计算流程如下所示：

1. **子层计算：** SubLayerOutput=SubLayer(x)SubLayerOutput = SubLayer(x)SubLayerOutput=SubLayer(x)
2. **残差相加：** Added=x+SubLayerOutputAdded = x + SubLayerOutputAdded=x+SubLayerOutput
3. **层归一化：** Output=LayerNorm(Added)Output = LayerNorm(Added)Output=LayerNorm(Added)

G

Xᵢn

输入 (x)

SubLayer

子层
(注意力或前馈网络)

Xᵢn->SubLayer

Add

+

Xᵢn:e->Add:w

x

SubLayer->Add

子层(x)

LN

层归一化

Add->LN

Output

输出

LN->Output

> 后归一化Transformer块中的数据流向。归一化在残差相加之后进行。

**特点：**

- **原始形式：** 这是Vaswani等人（2017）论文中描述的设置。
- **潜在的不稳定性：** 后归一化面临的主要问题是，残差分支的输出（和 x+SubLayer(x)x + SubLayer(x)x+SubLayer(x)）在传递给下一层 *之前* 未经归一化。在深度网络中，激活值的幅度可能在层与层之间显著变化，这可能导致训练初期出现梯度爆炸或梯度消失。
- **预热要求：** 后归一化配置通常需要仔细的学习率预热阶段（从较小的学习率开始并逐渐增加）。如果没有预热，由于未经归一化的加法，初始梯度可能过大，导致训练发散。

### 预归一化 (normalization) (Pre-LN)

为解决后归一化的稳定性问题，预归一化方法被提出。在这里，层归一化应用于输入，在它进入子层模块 *之前*，但 *在* 残差分支内部。残差连接随后将原始的、未经修改的输入 xxx 添加到子层的输出中。

预归一化块中子层的计算流程是：

1. **层归一化：** Normalized\_x=LayerNorm(x)Normalized\\_x = LayerNorm(x)Normalized\_x=LayerNorm(x)
2. **子层计算：** SubLayerOutput=SubLayer(Normalized\_x)SubLayerOutput = SubLayer(Normalized\\_x)SubLayerOutput=SubLayer(Normalized\_x)
3. **残差相加：** Output=x+SubLayerOutputOutput = x + SubLayerOutputOutput=x+SubLayerOutput

G

Xᵢn

输入 (x)

LN

层归一化

Xᵢn->LN

Add

+

Xᵢn:e->Add:w

x

SubLayer

子层
(注意力或前馈网络)

LN->SubLayer

层归一化(x)

SubLayer->Add

子层(层归一化(x))

Output

输出

Add->Output

> 预归一化Transformer块中的数据流向。归一化在子层计算之前进行。

**特点：**

- **提高稳定性：** 通过对每个子层的 *输入* 进行归一化，预归一化防止了传递到这些可能复杂的函数中的激活值发生爆炸。通过残差连接 (xxx) 的输出梯度路径保持清晰，使得梯度能够更顺畅地流经深度网络，而不会被归一化层过度缩放。
- **对预热的敏感性降低：** 预归一化配置通常对学习率调度不那么敏感，即使没有特定的预热阶段，或者预热阶段很短，也能稳定训练。这简化了超参数 (parameter) (hyperparameter)的调整。
- **常见实践：** 由于其稳定性优势，预归一化已成为许多现代大规模Transformer实现（例如GPT-2、GPT-3、ViT）中的事实标准。与后归一化相比，它使得训练更深的模型成为可能。

### 比较与权衡

| 特点 | 后归一化 (normalization) (Post-LN) | 预归一化 (Pre-LN) |
| --- | --- | --- |
| **放置位置** | `LayerNorm(x + SubLayer(x))` | `x + SubLayer(LayerNorm(x))` |
| **稳定性** | 稳定性较差，尤其在深层模型中 | 稳定性更好，有助于训练更深的模型 |
| **预热** | 通常需要仔细的学习率预热 | 对学习率预热不那么敏感，常无需预热也能训练 |
| **梯度流动** | 梯度在相加后通过归一化层 | 梯度通过残差路径绕过归一化层 |
| **原始论文** | 是 | 否 (后续改进) |
| **现代应用** | 在非常大的模型中较少见 | 被广泛采用，尤其对于大型模型 |
| **最佳表现** | 经过大量调整有时能达到略好的最佳结果 | 通常更容易调整以获得良好、稳定的结果 |

010020030040050046810121416预归一化 (稳定)后归一化 (无预热 - 发散)后归一化 (有预热)训练损失比较训练步数训练损失

> 训练损失曲线。预归一化通常显示稳定收敛。无预热的后归一化可能会发散，而经过适当预热的后归一化可以良好收敛，有时能达到比预归一化略低的最终损失，但需要仔细调整。

### 总结

尽管原始Transformer使用了后归一化 (normalization)，但预归一化变体在实际应用中具有显著优势，包括训练稳定性提高以及对学习率调度等超参数 (parameter) (hyperparameter)选择的敏感性降低。通过在输入通过复杂的自注意力 (self-attention)和前馈层 *之前* 进行归一化，预归一化确保了更平滑的优化过程，这在将Transformer扩展到数十甚至数百层时尤为重要。基于这些原因，预归一化常是当代Transformer架构中的首选。但对两种配置的了解，能够帮助我们更好地理解这些强大模型的设计选择和训练过程。

获取即时帮助、个性化解释和交互式代码示例。

---

### 神经网络语言模型的缩放法则

# 神经网络语言模型的缩放法则

理解性能如何随着资源的增加——特别是计算能力、数据和模型大小——而变化，对提高Transformer架构的效率具有重要作用。这种关系对于设计实验、分配预算以及预测未来大型模型的能力非常重要。经验研究发现了一些令人惊讶的可预测规律，这些规律常被称为缩放法则。

### 核心思想：可预测的性能缩放

开创性的工作，特别是Kaplan等人（2020年）的研究表明，语言模型的性能（通常通过在未见过数据上的交叉熵损失LLL衡量）随着三个主要因素的变化而可预测地提高：

1. **模型大小 (NNN)**: 模型中非嵌入 (embedding)参数 (parameter)的数量。
2. **数据集大小 (DDD)**: 训练期间处理的token数量。
3. **计算量 (CCC)**: 训练的总计算成本，以浮点运算（FLOPs）衡量，对于Transformer模型通常近似为C≈6NDC \approx 6NDC≈6ND。

主要发现是，损失与这些因素之间的关系通常遵循幂律，至少在几个数量级范围内。这意味着随着规模的增加，损失会平滑且可预测地降低：

L(N,D,C)≈L∞+((NcN)αN+(DcD)αD+(CcC)αC)L(N, D, C) \approx L\_\infty + \left( \left( \frac{N\_c}{N} \right)^{\alpha\_N} + \left( \frac{D\_c}{D} \right)^{\alpha\_D} + \left( \frac{C\_c}{C} \right)^{\alpha\_C} \right)L(N,D,C)≈L∞​+((NNc​​)αN​+(DDc​​)αD​+(CCc​​)αC​)

或者，更简单地说，当一个因素是瓶颈时：

- 损失与模型大小：L(N)∝N−αNL(N) \propto N^{-\alpha\_N}L(N)∝N−αN​
- 损失与数据集大小：L(D)∝D−αDL(D) \propto D^{-\alpha\_D}L(D)∝D−αD​
- 损失与计算量：L(C)∝C−αCL(C) \propto C^{-\alpha\_C}L(C)∝C−αC​

这里，Nc,Dc,CcN\_c, D\_c, C\_cNc​,Dc​,Cc​ 是表示特征尺度的常数，而 αN,αD,αC\alpha\_N, \alpha\_D, \alpha\_CαN​,αD​,αC​ 是正指数，表示性能随每个因素的缩放程度。L∞L\_\inftyL∞​ 表示一个不可减少的损失分量。这种平滑的幂律行为表明，性能提升并非源于特定规模下的突然突破，而是来自对资源的持续投入。

1e+72e+75e+71e+82e+85e+81e+92e+95e+91e+102.42.62.833.23.43.63.8经验数据幂律拟合 (示意)

> 一张双对数图表，说明了测试损失通常如何随着模型大小（参数数量）的增加而降低，遵循可预测的幂律趋势。

### 优化资源分配：计算预算限制

Kaplan等人的一项核心发现与在固定计算预算 (CCC) 下优化资源分配有关。由于 C≈6NDC \approx 6NDC≈6ND，如果计算预算固定，增加模型大小 (NNN) 可能意味着你需要减少数据集大小 (DDD)。他们的分析表明，为了获得最佳性能（在给定 CCC 下实现最低损失），通常最好优先更快速地增加模型大小 (NNN) 而非数据集大小 (DDD)。他们发现性能与模型大小的缩放关系强于与数据集大小的缩放关系（αN\alpha\_NαN​ 略大于 αD\alpha\_DαD​）。

这意味着，在相对较小的数据集上训练较少步骤的大型模型，可能比在更大型数据集上训练更长时间的小型模型更具计算效率。

### 修正的缩放法则：Chinchilla的发现

然而，Hoffmann等人（2022年）的后续研究，即“Chinchilla”论文，通过更广泛的实验重新审视了这些缩放法则。他们的发现对优化分配提出了不同的看法。

Chinchilla研究得出结论，对于计算优化训练，模型大小 (NNN) 和数据集大小 (DDD) 应大致*等比例*缩放。具体来说，每当模型大小翻倍时，训练token的数量也应翻倍，以期为投入的计算量获得最佳性能。

这表明，在此发现之前开发的许多大型语言模型（如GPT-3、Gopher）相对于其尺寸而言，都严重*训练不足*。它们相对于所训练的数据量而言，模型过大，未达优化。根据Chinchilla的缩放法则，一个在大量更多数据上训练的较小模型可以用相同的计算预算达到相同的性能，或者，以相同的预算获得更好的性能。

例如，Chinchilla模型（700亿参数 (parameter)）在1.4万亿token上进行训练，在众多基准测试中，其性能优于大得多的Gopher模型（2800亿参数，在3000亿token上训练），尽管两者使用了相似的训练计算量。

### 实际意义

这些缩放法则为实践者提供了有益的指导：

- **性能预测**: 它们允许合理准确地预测通过扩展计算、模型大小或数据可实现的性能提升。
- **资源分配**: 它们为如何分配固定预算（例如GPU小时）提供了参考。Chinchilla的发现表明，模型参数 (parameter)和训练token的平衡增长通常是最佳选择。
- **识别瓶颈**: 如果性能显著偏离预测的缩放曲线，则可能表明训练设置、数据质量或架构选择存在问题。
- **训练策略**: 相较于Kaplan最初研究的含义，修正后的法则更倾向于在更大的数据集上训练模型更长时间。

### 局限性与注意事项

认识到这些法则的局限性很重要：

- **经验性**: 它们是源自特定模型家族（主要是密集型Transformer）和数据集（大多是网络文本）的经验观察。它们可能无法完美推广到所有架构或数据模态。
- **关注预训练 (pre-training)损失**: 缩放法则主要预测预训练损失（如交叉熵）。虽然这与下游任务性能强相关，但这种关系并非总是完美或线性的。特定任务的性能提升可能以不同的方式达到饱和。
- **不可约损失**: 性能不能无限期地提升。存在一个与语言本身固有不可预测性相关的理论最小损失 (L∞L\_\inftyL∞​)。随着模型变得非常大，当它们接近此极限时，收益会减少。
- **常数因素**: 精确指数 (αN,αD,αC\alpha\_N, \alpha\_D, \alpha\_CαN​,αD​,αC​) 和常数 (Nc,Dc,CcN\_c, D\_c, C\_cNc​,Dc​,Cc​) 可能因模型架构细节、数据质量和分词 (tokenization)方法等因素而异。

尽管有这些考量，缩放法则代表了在理解如何有效构建和训练大型语言模型方面的重大进步。它们提供了一个量化 (quantization)框架，用于推理 (inference)扩展AI系统所涉及的权衡。当你设计或使用高级Transformer架构时，理解这些经验关系对于对模型大小、数据需求和计算资源做出明智的决策是基本的。

获取即时帮助、个性化解释和交互式代码示例。

---

### 参数效率与共享技术

# 参数效率与共享技术

标准Transformer架构，尤其是在规模扩大后，需要大量计算资源和内存。自注意力 (self-attention)机制 (attention mechanism)的二次复杂度是一个瓶颈，可通过稀疏注意力或线性注意力等方法解决。然而，导致资源消耗巨大的另一个重要因素是参数 (parameter)数量庞大，尤其是在多层深层模型中。旨在减少参数数量的方法，使模型更小，可能（每轮训练）更快（因为数据移动减少），部署更方便，通常对性能影响甚微。

核心思想围绕着**参数共享**，即在网络的不同部分使用同一组权重 (weight)。这与标准深层网络形成对比，后者通常每层都有自己独特的参数集。

### 因子化嵌入 (embedding)参数 (parameter)化

在许多自然语言处理（NLP）任务中，词汇量 (VVV) 会非常大（数万或数十万）。输入嵌入层将这些离散标记 (token)映射到隐藏维度为 HHH 的稠密向量 (vector)。这会生成一个大小为 V×HV \times HV×H 的嵌入矩阵。在BERT等大型模型中，隐藏维度 HHH 也很大（例如768或1024）。因此，嵌入矩阵在总参数中占据了相当大的比例。例如，当 V=50,000V=50,000V=50,000、H=1024H=1024H=1024 时，仅嵌入矩阵就有超过5000万个参数。

因子化嵌入参数化背后的核心观点，在ALBERT（一种用于语言表征自监督学习 (supervised learning) (self-supervised learning)的轻量BERT）等模型中得到突出体现，即嵌入服务于两个目的：捕捉与上下文 (context)无关的标记表征，并将其投影到Transformer层的隐藏空间。ALBERT认为，与上下文无关的表征可能不需要完整的 HHH 维度。

该技术不是直接学习一个 V×HV \times HV×H 矩阵，而是将其分解为两个较小的矩阵：

1. 一个大小为 V×EV \times EV×E 的低维嵌入矩阵，其中 EEE 是嵌入维度，E≪HE \ll HE≪H。
2. 一个大小为 E×HE \times HE×H 的投影矩阵，将低维嵌入 (EEE) 投影到Transformer层所需的预期隐藏维度 (HHH)。

嵌入的总参数数量变为 V×E+E×HV \times E + E \times HV×E+E×H。与原始的 V×HV \times HV×H 相比，如果 EEE 显著小于 HHH，这可以带来大量的节省。例如，如果 V=50,000V=50,000V=50,000， H=1024H=1024H=1024，我们选择 E=128E=128E=128，参数将从 50,000×1024≈51.2M50,000 \times 1024 \approx 51.2M50,000×1024≈51.2M 变为 (50,000×128)+(128×1024)=6.4M+0.13M≈6.53M(50,000 \times 128) + (128 \times 1024) = 6.4M + 0.13M \approx 6.53M(50,000×128)+(128×1024)=6.4M+0.13M≈6.53M。这使得嵌入参数减少了近8倍。

这种因子化将词汇量与Transformer的隐藏维度解耦，使得在拥有大词汇量的情况下，不会因嵌入而过度增加模型体积。它隐含地假定单个词语的语义丰富性可以在低维空间 (EEE) 中被捕捉，随后投影到Transformer层内部用于上下文依赖处理的更高维度空间 (HHH)。

### 跨层参数 (parameter)共享

另一个有效方法，也是ALBERT的核心，是在Transformer层之间共享参数。在一个包含 LLL 层的标准Transformer编码器堆栈中，每层通常有自己独特的权重 (weight)矩阵，用于多头自注意力 (self-attention)（Q、K、V投影、输出投影）和逐位前馈网络（FFN）。这表示参数数量随层数线性增长。

跨层参数共享打破了这一假设。它不是学习 LLL 个不同的层参数集，而是学习一个参数集并在所有 LLL 层中重复使用。常见变体包括：

- **共享所有参数：** 注意力和FFN子层的精确相同权重用于每一层。这是最激进的形式，可带来最大的参数减少。
- **仅共享FFN参数：** 注意力参数每层独立，但FFN参数共享。
- **仅共享注意力参数：** FFN参数每层独立，但注意力参数共享。

共享所有参数，如ALBERT中所示，显著减少了与Transformer层相关的参数数量。如果单层有 PlayerP\_{layer}Player​ 个参数，一个标准 LLL 层模型在这些层中有 L×PlayerL \times P\_{layer}L×Player​ 个参数，而一个完全共享的模型所有层加起来只有 PlayerP\_{layer}Player​ 个参数（不包括嵌入 (embedding)层和归一化 (normalization)层，这些层可能共享也可能不共享）。

嵌入层 (V=30k)编码器层 (总计)020406080标准BERT-Base (L=12, H=768)ALBERT-Base (L=12, H=768, E=128, 共享层)

> 标准BERT-Base模型与ALBERT-Base模型（采用因子化嵌入 E=128 和完全跨层共享）的嵌入层和编码器层参数数量的大致比较。请注意，ALBERT的这两个组成部分都显著减少了。

经验结果，特别是ALBERT论文中的结果，表明跨层参数共享能够生成显著更小但性能与非共享模型相当，甚至有时更好的模型，在各种NLP基准测试中表现出色，尤其是在与因子化嵌入结合使用时。它似乎作为一种正则化 (regularization)形式，可能通过阻止后续层的功能与早期层过度偏离来提高泛化能力。然而，与非共享模型相比，它有时会减慢训练收敛速度，可能是因为共享参数必须适应网络深度内不同处理阶段的功能需求。

### 其他共享技术

- **权重 (weight)绑定：** 一种常见做法，尤其在序列到序列模型或语言模型中，是将输入嵌入 (embedding)层和将解码器输出投影回词汇空间的最终输出线性层的权重绑定。由于这两个层都在标记 (token)空间和模型的隐藏表征空间之间操作，共享这些权重可减少参数 (parameter)并通过强制对称性来提高性能。这与因子化嵌入兼容；如果使用因子化（V×EV \times EV×E 和 E×HE \times HE×H），V×EV \times EV×E 矩阵可以与最终的投影矩阵绑定（后者实际上将是一个 H×EH \times EH×E 矩阵，后跟一个 E×VE \times VE×V 矩阵，共享 V×EV \times EV×E 部分）。

### 权衡与考量

参数 (parameter)共享技术具有显著优点：

- **模型体积减小：** 参数减少意味着模型文件更小，存储和部署所需的内存更少。
- **训练可能更快（每轮）：** 虽然收敛可能需要更多步骤，但由于分布式训练中通信开销减少或内存压力减轻，每轮训练可能更快。
- **正则化 (regularization)效应：** 共享可以稳定训练，有时能提升泛化能力。

然而，也存在潜在缺点：

- **表征能力：** 过于激进的共享可能限制模型在不同层学习高度特定功能的能力。
- **收敛速度：** 与非共享模型相比，训练可能需要更多步骤或不同的优化超参数 (hyperparameter)（如学习率）才能有效收敛。

选择正确的参数效率技术需要权衡这些利弊。因子化嵌入 (embedding)通常是一种相对安全的方法，可在对性能影响最小的情况下减少参数，尤其适用于大词汇量。跨层参数共享更为激进，但已被证明出乎意料地有效，如ALBERT所示。在特定任务和数据集上对这些技术进行经验评估仍然是模型开发的重要步骤。这些方法对构建强大而实用的Transformer模型的持续努力做出了重要贡献。

获取即时帮助、个性化解释和交互式代码示例。

---

## Chapter 7 Implementation Details Optimization

### 选择框架 (PyTorch, TensorFlow, JAX)

# 选择框架 (PyTorch, TensorFlow, JAX)

实现和优化强大的 Transformer 模型需要考虑实际操作事项。此过程中的一个重要首步是选择合适的深度学习 (deep learning)框架。这项选择显著影响开发速度、调试便捷性、性能提升途径以及部署方案。当前用于认真构建 Transformer 的主要选择是 PyTorch、TensorFlow 和 JAX，它们各自提供不同的优势和理念。

### PyTorch

PyTorch 主要由 Meta AI 开发，获得了相当大的关注，尤其是在研究群体中。它的吸引力常源于其“Python风格”的设计。调试通常更直接，因为它默认采用即时执行模式，能立即执行操作，这与标准 Python 程序流程相仿。这使得检查中间张量和使用 `pdb` 等标准 Python 调试工具相对简单。

Transformer 开发的主要优势包括：

- **灵活性与控制力：** PyTorch 提供一个相对低级的接口（与 Keras 相比），对模型架构和训练循环提供细致的控制，这对于实现自定义 Transformer 变体或高级优化方法很有益处。
- **完善的生态：** 它与流行的 Hugging Face `transformers` 库集成，方便使用大量的预训练 (pre-training)模型和分词 (tokenization)器 (tokenizer)。像 `Accelerate` 这样的库简化了分布式训练和混合精度使用。
- **研究群体：** 它在研究领域的普遍使用意味着新方法和模型架构常首先在 PyTorch 中实现。
- **性能：** 尽管传统上以易用性著称，PyTorch 凭借 `torch.compile` 在性能优化方面取得了显著进展，它能融合操作并利用 Triton 等后端加速模型执行，性能常接近编译图。`Torch Distributed` 提供数据和模型并行化的工具。

### TensorFlow

TensorFlow 最初由 Google Brain 开发，是一个成熟的框架，非常注重生产部署和可扩展性。其高级 API Keras 现在是与 TensorFlow 交互的标准方式，提供用户友好的界面来定义模型和训练流程。

与 Transformer 有关的重要方面：

- **生产部署：** TensorFlow 提供一套完整的部署工具，包括用于高性能推理 (inference)服务器的 TensorFlow Serving、用于移动和边缘设备的 TensorFlow Lite，以及用于端到端 MLOps 流水线的 TensorFlow Extended (TFX)。
- **图编译：** TensorFlow 主要通过首先构建计算图（`tf.function` 装饰器）然后执行它来运行。这允许通过其 XLA (加速线性代数) 编译器进行广泛的图级优化，可能带来高性能，尤其是在 Google TPU 等硬件加速器上。
- **可扩展性：** TensorFlow 对分布式训练策略有成熟的支持，能够训练跨多个 GPU 或 TPU pod 的大型模型。
- **生态系统：** 它受益于 TensorBoard 等可视化工具和一个提供支持和扩展的庞大社区，尽管 PyTorch 在最新研究思想的快速采纳方面可以说已经超越它。

### JAX

JAX 同样由 Google Brain 开发，是一个较新的库，旨在进行高性能数值计算，特别适合涉及大型模型和硬件加速器的机器学习 (machine learning)研究。它不像 PyTorch 或 TensorFlow 那样是一个完整的深度学习 (deep learning)框架，但为 NumPy 代码提供了可组合的函数变换。

Transformer 工作的重要特征：

- **函数变换：** JAX 的核心优势在于其变换：
  - `grad`：自动微分。
  - `jit`：使用 XLA 进行即时编译，显著提升速度。
  - `vmap`：自动向量 (vector)化（批处理）。
  - `pmap`：跨多个设备（GPU/TPU）的自动并行化，简化数据和模型并行化实现。
- **性能侧重：** JAX 常被选择用于提升性能和规模的极限，尤其是在 TPU 上，因为 `pmap` 与硬件架构天然契合。
- **函数式方法：** JAX 鼓励函数式编程风格（纯函数），这可以带来更清晰、更可预测的代码，但对于习惯面向对象框架的人来说可能是一个学习曲线。状态管理（如模型参数 (parameter)和优化器状态）是明确处理的。
- **成长中的生态：** Flax 和 Haiku 等库在 JAX 之上提供了更高级的抽象，使 Transformer 实现更有条理。

### Transformer 的框架考量

在这些框架中选择取决于项目需求、团队专长和目标基础设施。以下是比较汇总：

| 特性 | PyTorch | TensorFlow (带 Keras) | JAX |
| --- | --- | --- | --- |
| **主要 API** | 命令式 (Eager), Python 风格 | 声明式 (Keras), 基于图 (`tf.function`) | 函数式, 类似 NumPy, 变换 |
| **调试** | 通常较容易 (Eager 模式) | 可能较难 (图模式), Keras 简化 | 需理解 JIT/变换 |
| **性能** | 卓越 (尤其使用 `torch.compile`) | 卓越 (尤其使用 XLA) | 潜力最高 (尤其在 TPU 上, `pmap`) |
| **灵活性** | 高, 控制良好 | 中等 (Keras), 高 (低级 TF) | 很高 (低级, 函数式) |
| **生态系统** | 强大的研究社群, Hugging Face 集成 | 成熟的生产环境, TFX, TensorBoard | 快速发展, 注重研究 |
| **部署** | 良好 (TorchServe, ONNX) | 卓越 (TF Serving, TFLite) | 需要更多专业/定制化 |
| **分布式** | (`DistributedDataParallel`, FSDP) | (`MirroredStrategy`, DTensor) | 集成 (`pmap`) |
| **学习曲线** | 中等 | 中等 (Keras), 更陡峭 (TF Core) | 更陡峭 (函数式, 变换) |

**建议：**

- **对于快速原型开发、研究以及获取最新模型：** PyTorch 常是一个强大的选择，因为其灵活性以及与 Hugging Face 生态系统的紧密结合。
- **对于具有不同部署目标（服务器、移动、边缘）的生产流水线：** TensorFlow 成熟的部署工具提供显著优势。
- **对于尖端性能、大规模训练（尤其在 TPU 上）以及偏好函数式编程：** JAX 提供强大的工具，尽管它可能需要更多的初始投入来掌握其思想。

最终，所有这三个框架都能高效实现复杂的 Transformer 架构。熟悉团队现有的技能和基础设施常起决定作用。如果可行，在不同框架中尝试小型 Transformer 实现，可获得关于各自工作流程和取舍的宝贵见解。对于任何认真从事现代大型语言模型工作的工程师而言，至少精通其中一个框架是必不可少的。

获取即时帮助、个性化解释和交互式代码示例。

---

### 权重初始化策略

# 权重初始化策略

正确初始化神经网络 (neural network)的权重 (weight)是实现稳定高效训练的**主要步骤**。对于Transformer这类层数较多的架构，梯度通过多层传播，不当的初始化很容易导致梯度消失或梯度爆炸，在学习过程开始前就使其停滞。虽然现代的优化器和归一化 (normalization)层可以缓解这些问题中的一部分，但周全的权重初始化仍然是实际操作中一个**主要组成部分**。

大多数标准初始化方案的目标是，在激活值和梯度在网络中向前和向后传播时，保持其方差不变。如果方差随每层呈指数增长，梯度就会爆炸；如果呈指数减小，梯度就会消失。

### 线性层初始化

Transformer模型在多头注意力 (multi-head attention)模块（用于Q、K、V投影和最终输出投影）和逐位置前馈网络（FFNs）中大量使用线性（或全连接）层。这些层最常用且有效的初始化策略是Glorot (Xavier) 和 He 初始化。

- **Glorot (Xavier) 初始化：** 由Glorot和Bengio（2010年）提出，此方法旨在保持激活值和梯度在各层间的方差恒定。它对于使用对称激活函数 (activation function)（如tanhtanhtanh）的层尤其有效。对于将输入大小为faninfan\_{in}fanin​转换为输出大小为fanoutfan\_{out}fanout​的线性层，权重 (weight)WWW通常从均匀分布中抽取：

  W∼U[−6fanin+fanout,6fanin+fanout]W \sim U\left[-\sqrt{\frac{6}{fan\_{in} + fan\_{out}}}, \sqrt{\frac{6}{fan\_{in} + fan\_{out}}}\right]W∼U[−fanin​+fanout​6​​,fanin​+fanout​6​​]

  或者，也可以使用正态分布：

  W∼N(0,2fanin+fanout)W \sim N\left(0, \frac{2}{fan\_{in} + fan\_{out}}\right)W∼N(0,fanin​+fanout​2​)
- **He 初始化：** 由He等人（2015年）提出，此方法专为后接修正线性单元（ReLU）或其变体的层设计，这在Transformer FFN中很常见。它考虑了ReLU将一半激活值归零从而减小方差的事实。权重通常从正态分布中抽取：

  W∼N(0,2fanin)W \sim N\left(0, \frac{2}{fan\_{in}}\right)W∼N(0,fanin​2​)

  也存在均匀分布版本。

在实践中，许多深度学习 (deep learning)框架在预期使用ReLU激活时，默认对线性层采用He初始化，否则采用Glorot初始化。对于Transformer模型，为FFN层使用He初始化，而为注意力投影层（这些层在进一步组合之前通常不会对其直接输出立即应用ReLU）使用Glorot/Xavier初始化，是一个合理的起点。像最初的“Attention Is All You Need”论文中的实现就使用了Glorot均匀初始化。

### 嵌入 (embedding)层初始化

输入和输出嵌入层将离散的令牌ID映射到稠密向量 (vector)。这些层通常与标准线性层的初始化方式不同。一个常见的做法是从均值为0、标准差相对较小的正态分布（例如N(0,0.022)N(0, 0.02^2)N(0,0.022)）初始化嵌入权重 (weight)。这可以避免初始嵌入值过大，从而可能导致后续计算不稳定，尤其是位置编码 (positional encoding)的加入。

一些实现可能会绑定输入和输出嵌入权重，为两者共享相同的矩阵（可能经过转置），这也会影响初始化方法的选择。

### 层归一化 (normalization)初始化

层归一化层也有可训练参数 (parameter)：增益（γ\gammaγ）和偏差（β\betaβ）。这些参数通常初始化为γ=1\gamma = 1γ=1和β=0\beta = 0β=0。这确保了层归一化在初始时仅将激活值归一化为零均值和单位方差，而不应用任何缩放或平移，从而让网络在训练过程中学习合适的变换。

### 实际考量

大多数现代深度学习 (deep learning)库（如PyTorch、TensorFlow、JAX）为其标准层提供了合理的默认初始化，通常与Glorot或He方法一致。例如，Hugging Face Transformers库通常使用标准差由`initializer_range`配置参数 (parameter)（通常默认为0.02）指定的正态分布初始化权重 (weight)，并将偏差初始化为零，而LayerNorm的权重设置为1，偏差设置为0。

−0.04−0.0200.020.04020406080100初始化方法N(0, 0.02^2)Xavier 均匀分布 (512 单元)权重初始化分布权重值频率

> 比较了层具有512个输入/输出单元时，使用小标准差正态分布初始化和Xavier均匀初始化所产生的权重分布。请注意，与Xavier方法的更宽泛分布相比，N(0, 0.02^2)方法在零点附近聚集更紧密。

虽然这些默认设置通常表现良好，但理解其基本原理有助于在训练变得不稳定时进行更明智的调试。偶尔，可以尝试对初始化标准差进行微小调整，特别是对于嵌入 (embedding)层或最终输出层，但对于核心线性层而言，大幅偏离Glorot或He等既定方法通常是不必要的，并可能适得其反。

获取即时帮助、个性化解释和交互式代码示例。

---

### 适用于Transformer的优化器 (Adam, AdamW)

# 适用于Transformer的优化器 (Adam, AdamW)

优化器在训练深度神经网络 (neural network)时处理复杂的损失情况方面发挥着重要作用。虽然标准随机梯度下降 (gradient descent)（SGD）提供了初始方法，但其固定的学习率对于Transformer模型的规模和复杂性来说往往不够用。自适应学习率算法能够动态调整每个参数 (parameter)的步长，已成为不可或缺的工具。在这些算法中，Adam及其改进版AdamW是训练Transformer时普遍采用的优化器。

## Adam: 自适应矩估计

Adam是自适应矩估计的简称，它结合了动量方法和RMSprop的思路。它根据梯度的一阶和二阶矩的估算值，为不同的参数 (parameter)计算各自的自适应学习率。

1. **一阶矩（动量）：** Adam维持着过去梯度的一个指数衰减平均值。这起到了动量的作用，帮助优化器在损失曲面浅平的山谷中更快地前进，并减缓高曲率方向的振荡。时间步 ttt 时的一阶矩估算值 mtm\_tmt​ 使用梯度 gtg\_tgt​ 计算：
   mt=β1mt−1+(1−β1)gtm\_t = \beta\_1 m\_{t-1} + (1 - \beta\_1) g\_tmt​=β1​mt−1​+(1−β1​)gt​
   超参数 (hyperparameter) β1\beta\_1β1​ 决定了指数衰减率，通常设为0.9这样的值。
2. **二阶矩（方差自适应）：** Adam还维持着过去梯度平方的一个指数衰减平均值。这作为每个参数梯度变化程度的估算（更精确地说，是未中心化的二阶矩）。该估算值被用来反向调整学习率；梯度历史值波动较大的参数更新步长较小，而梯度较小或稀疏的参数更新步长较大。二阶矩估算值 vtv\_tvt​ 计算方式如下：
   vt=β2vt−1+(1−β2)gt2v\_t = \beta\_2 v\_{t-1} + (1 - \beta\_2) g\_t^2vt​=β2​vt−1​+(1−β2​)gt2​
   超参数 β2\beta\_2β2​ 决定此衰减率，通常设为0.999。
3. **偏差修正：** 由于矩估算值 mtm\_tmt​ 和 vtv\_tvt​ 初始化为零，它们在训练初期存在偏向零的偏差。Adam引入了偏差修正项来抵消这种影响：
   m^t=mt1−β1t\hat{m}\_t = \frac{m\_t}{1 - \beta\_1^t}m^t​=1−β1t​mt​​
   v^t=vt1−β2t\hat{v}\_t = \frac{v\_t}{1 - \beta\_2^t}v^t​=1−β2t​vt​​
   这些修正后的估算值 m^t\hat{m}\_tm^t​ 和 v^t\hat{v}\_tv^t​ 提供了对真实矩更准确的估算，尤其是在训练初期。
4. **参数更新：** 最后，使用偏差修正后的矩估算值来更新参数 www。更新规则通过修正后的一阶矩与修正后的二阶矩的平方根的比值，来调整基础学习率 α\alphaα：
   wt+1=wt−αm^tv^t+ϵw\_{t+1} = w\_t - \alpha \frac{\hat{m}\_t}{\sqrt{\hat{v}\_t} + \epsilon}wt+1​=wt​−αv^t​​+ϵm^t​​
   小常数 ϵ\epsilonϵ (例如 10−810^{-8}10−8) 被添加到分母中，以防止数值异常，从而避免除以零。

Adam在多种深度学习 (deep learning)任务中常表现出快速的初期收敛和良好性能。然而，当Adam与标准L2正则化 (regularization)（权重 (weight)衰减）结合使用时，会浮现一个不明显的问题。L2正则化旨在通过在损失函数 (loss function)中添加惩罚项 λ2∣∣w∣∣2\frac{\lambda}{2} ||w||^22λ​∣∣w∣∣2 来减少过拟合 (overfitting)。这导致梯度 gtg\_tgt​ 中增加一个额外的项 −λwt-\lambda w\_t−λwt​。在传统的Adam实现方式中，这个梯度组成部分 gt=∇L(wt)+λwtg\_t = \nabla \mathcal{L}(w\_t) + \lambda w\_tgt​=∇L(wt​)+λwt​ 被用于计算 mtm\_tmt​ 和 vtv\_tvt​。这使得权重衰减作用与自适应学习率机制耦合在一起。具体来说，更新过程中施加的实际权重衰减 (αλwtv^t+ϵ\alpha \frac{\lambda w\_t}{\sqrt{\hat{v}\_t} + \epsilon}αv^t​​+ϵλwt​​) 会受到自适应分母 v^t\sqrt{\hat{v}\_t}v^t​​ 的调整。这意味着与历史梯度大小（较大的 v^t\hat{v}\_tv^t​）相关的权重，其实际衰减作用会减弱，可能阻碍正则化作用。

## AdamW: 带有解耦权重 (weight)衰减的Adam

AdamW被提出是为了解决原始Adam算法中权重衰减与自适应学习率之间不理想的配合问题。Loshchilov & Hutter (2019) 提出的主要思路简单而有效：将权重衰减更新与Adam执行的基于梯度的更新分离开来。

AdamW没有将L2惩罚纳入用于矩估算的梯度 gtg\_tgt​ 中，而是仅使用来自主要损失函数 (loss function) ∇L(wt)\nabla \mathcal{L}(w\_t)∇L(wt​) 的梯度来计算矩估算值和主要的自适应更新步骤。权重衰减随后在自适应步骤 *之后* 单独且直接施加到权重上。

该过程可概括为：

1. 仅根据任务损失计算梯度 gt=∇L(wt)g\_t = \nabla \mathcal{L}(w\_t)gt​=∇L(wt​)。
2. 使用 gtg\_tgt​ 更新一阶和二阶矩估算值（mtm\_tmt​, vtv\_tvt​），并计算它们的偏差修正版本（m^t\hat{m}\_tm^t​, v^t\hat{v}\_tv^t​），与标准Adam中相同。
3. 计算主要的Adam更新步长（尚未考虑学习率）：
   Δwt′=m^tv^t+ϵ\Delta w'\_t = \frac{\hat{m}\_t}{\sqrt{\hat{v}\_t} + \epsilon}Δwt′​=v^t​​+ϵm^t​​
4. 通过施加缩放后的自适应步长和解耦的权重衰减来更新权重。使用 ηt\eta\_tηt​ 作为时间步 ttt 的调度学习率：
   wt+1=wt−ηt(Δwt′+λwt)w\_{t+1} = w\_t - \eta\_t (\Delta w'\_t + \lambda w\_t)wt+1​=wt​−ηt​(Δwt′​+λwt​)
   或者，更接近原始论文的表达，衰减可以在更新前以乘法形式施加：wt+1=(1−ηtλ)wt−ηtΔwt′w\_{t+1} = (1 - \eta\_t \lambda) w\_t - \eta\_t \Delta w'\_twt+1​=(1−ηt​λ)wt​−ηt​Δwt′​。主要观察点是，衰减项 λwt\lambda w\_tλwt​ 不再受 1/(v^t+ϵ)1 / (\sqrt{\hat{v}\_t} + \epsilon)1/(v^t​​+ϵ) 的影响。

通过解耦，AdamW保证了权重衰减在所有参数 (parameter)上施加得更一致，仅与权重大小 wtw\_twt​ 和学习率 ηt\eta\_tηt​ 成比例，不受 v^t\hat{v}\_tv^t​ 中捕获的梯度历史影响。这一修改常能带来更好的模型泛化能力和最终表现，相比使用L2正则化 (regularization)的标准Adam而言，特别是对于Transformer这类有效正则化非常有益的复杂模型。因此，AdamW已成为训练现代Transformer的事实标准优化器。

## 超参数 (parameter) (hyperparameter)与考量

有效运用Adam或AdamW需要设定多个超参数：

- **学习率 (α\alphaα 或 ηt\eta\_tηt​):** 这依然是一个非常敏感的超参数。对于Transformer而言，它几乎从不固定；相反，包含一个初始“预热”阶段，随后是衰减阶段的学习率调度十分重要（将在后续章节中讨论）。通常的峰值学习率处于 10−510^{-5}10−5 到 10−310^{-3}10−3 的范围，受模型大小、批量大小以及所用具体调度等因素影响。
- **β1\beta\_1β1​:** 一阶矩估算值的指数衰减率。默认值0.9在多数情况下效果良好。
- **β2\beta\_2β2​:** 二阶矩估算值的指数衰减率。虽然默认值0.999普遍使用，但一些研究表明略低的值（例如0.98、0.99）偶尔可能带来一些好处。
- **ϵ\epsilonϵ:** 一个小值，用于防止数值异常。像 10−810^{-8}10−8 或 10−610^{-6}10−6 这样的默认值通常足够。
- **权重 (weight)衰减 (λ\lambdaλ):** 可直接用于AdamW，或者在标准Adam中作为L2惩罚系数。这是一个重要的正则化 (regularization)超参数，需要通过实验调整。常见的起始值在0.01或0.1左右，但最佳值取决于数据集和模型。将 λ=0\lambda=0λ=0 设为0实际等于关闭权重衰减。

总而言之，尽管Adam在优化方面取得了显著进展，但AdamW提供了一种更优化的方法来纳入权重衰减，在训练大型语言模型时带来了明显的改进。对于Transformer架构，它通常是推荐的选择，总是与仔细调整的学习率调度和正则化参数配合使用。

获取即时帮助、个性化解释和交互式代码示例。

---

### 学习率调度 (热身, 衰减)

# 学习率调度 (热身, 衰减)

选择合适的优化器，如Adam或AdamW，只是成功训练大型Transformer模型的一部分。学习率作为控制优化过程中步长的基本超参数 (parameter) (hyperparameter)，通常需要在整个训练过程中仔细调整。简单使用固定学习率通常是次优的，并可能导致不稳定或收敛缓慢，尤其考虑到涉及的复杂损失曲面和深度结构。因此，学习率调度变得必要。

学习率调度定义了一种在训练期间动态改变学习率的策略。对于Transformer模型，一种常见且高效的方法是结合“热身”阶段和随后的“衰减”阶段。

### 热身阶段: 稳定早期训练

在训练的初始阶段，模型的参数 (parameter)是随机初始化的，梯度可能很大且不稳定。从一开始就使用高学习率可能会导致显著的更新，从而破坏训练过程的稳定性，可能使优化器偏离参数空间的良好区域。这对于Transformer模型来说尤其如此，因为层归一化 (normalization)、残差连接和注意力机制 (attention mechanism)之间的互相影响对早期大参数变化很敏感。

热身阶段通过从非常小的学习率（通常为零）开始，并在预设的初始训练步数（即 `warmup_steps`）内逐渐增加学习率来解决这个问题。线性增加是常见的：

lrstep=lrpeak×stepwarmup\_steps当 step<warmup\_stepslr\_{step} = lr\_{peak} \times \frac{step}{warmup\\_steps} \quad \text{当 } step < warmup\\_stepslrstep​=lrpeak​×warmup\_stepsstep​当 step<warmup\_steps

在此，lrpeaklr\_{peak}lrpeak​ 是调度在热身阶段后将达到的最大学习率。这种渐进式增加允许模型在应用更大参数更新之前稳定下来，防止早期发散并促进更平滑的收敛。

### 衰减阶段: 优化收敛

一旦热身阶段完成且学习率达到峰值 (lrpeaklr\_{peak}lrpeak​)，逐步降低它是有益的。这个衰减阶段有以下几个作用：

1. **精细调整:** 随着训练的进行，当模型接近一个良好的最小值时，较小的学习率允许更精细的调整，防止优化器越过最小值。
2. **避免震荡:** 训练后期的高学习率可能导致在最小值附近震荡。降低学习率有助于抑制这些震荡。
3. **提高泛化能力:** 一些研究表明，降低学习率有助于找到泛化能力更好的最小值，使其在未见过的数据上表现更佳。

热身阶段后，几种衰减策略常被使用：

- **逆平方根衰减:** 这种策略在原始论文“Attention Is All You Need”中被采用。热身之后，学习率与步数的逆平方根成比例地减小：

  lrstep=lrpeak×warmup\_stepsstep当 step≥warmup\_stepslr\_{step} = lr\_{peak} \times \sqrt{\frac{warmup\\_steps}{step}} \quad \text{当 } step \ge warmup\\_stepslrstep​=lrpeak​×stepwarmup\_steps​​当 step≥warmup\_steps

  或者，更常见的是直接参照论文中的公式实现（使用模型维度 dmodeld\_{model}dmodel​ 和一个缩放因子）：

  lrstep=dmodel−0.5×min⁡(step−0.5,step×warmup\_steps−1.5)lr\_{step} = d\_{model}^{-0.5} \times \min(step^{-0.5}, step \times warmup\\_steps^{-1.5})lrstep​=dmodel−0.5​×min(step−0.5,step×warmup\_steps−1.5)

  此公式将线性热身和逆平方根衰减结合到一个表达式中。请注意，lrpeaklr\_{peak}lrpeak​ 由 dmodel−0.5d\_{model}^{-0.5}dmodel−0.5​ 和 warmup\_stepswarmup\\_stepswarmup\_steps 隐式定义。
- **余弦衰减 (余弦退火):** 学习率遵循余弦曲线，在剩余的训练步数中从 lrpeaklr\_{peak}lrpeak​ 降至最小值（通常为零）。这提供平滑、渐进的下降。

  lrstep=lrmin+0.5×(lrpeak−lrmin)×(1+cos⁡((step−warmup\_steps)×πtotal\_steps−warmup\_steps))lr\_{step} = lr\_{min} + 0.5 \times (lr\_{peak} - lr\_{min}) \times \left(1 + \cos\left(\frac{(step - warmup\\_steps) \times \pi}{total\\_steps - warmup\\_steps}\right)\right)lrstep​=lrmin​+0.5×(lrpeak​−lrmin​)×(1+cos(total\_steps−warmup\_steps(step−warmup\_steps)×π​))

  其中 lrminlr\_{min}lrmin​ 是目标最小学习率（例如0），total\_stepstotal\\_stepstotal\_steps 是计划的总训练步数。
- **线性衰减:** 学习率在剩余步数中从 lrpeaklr\_{peak}lrpeak​ 线性下降到最小值（通常为零）。

  lrstep=lrmin+(lrpeak−lrmin)×total\_steps−steptotal\_steps−warmup\_stepslr\_{step} = lr\_{min} + (lr\_{peak} - lr\_{min}) \times \frac{total\\_steps - step}{total\\_steps - warmup\\_steps}lrstep​=lrmin​+(lrpeak​−lrmin​)×total\_steps−warmup\_stepstotal\_steps−step​
- **指数衰减:** 学习率以固定间隔或每一步乘以一个小于1的衰减因子。

衰减策略的选择会影响最终模型性能，经验评估通常是必要的。余弦衰减和逆平方根衰减是训练Transformer模型时尤其受欢迎的选择。

### 组合调度可视化

热身和衰减的组合随时间形成一个典型的学习率曲线。以下图表示例一个典型的调度，包含线性热身和随后的逆平方根衰减。

0204060800100μ200μ300μ400μ500μ

> 学习率调度，包含10步热身阶段，达到0.0005的峰值，随后是逆平方根衰减。

### 实现方面的考量

大多数深度学习 (deep learning)框架都提供内置支持，使学习率调度器可以轻松与优化器集成。

- **PyTorch:** `torch.optim.lr_scheduler` 模块提供多种调度器，例如 `LambdaLR`（用于逆平方根等自定义函数）、`CosineAnnealingLR`、`LinearLR` 和 `SequentialLR`（用于连接热身和衰减）。通常在每次优化器 `step()` 调用后调用调度器的 `step()` 方法。
- **TensorFlow/Keras:** `tf.keras.optimizers.schedules` 模块提供诸如 `PolynomialDecay`（可以实现线性衰减）、`CosineDecay` 的类，并允许通过继承 `LearningRateSchedule` 创建自定义调度。这些调度在初始化时直接传递给优化器。

调度的具体参数 (parameter)（lrpeaklr\_{peak}lrpeak​、warmup\_stepswarmup\\_stepswarmup\_steps、衰减类型、total\_stepstotal\\_stepstotal\_steps）是重要的超参数 (hyperparameter)，通常需要根据所使用的特定模型大小、数据集和批次大小进行调整。选择合适的调度并调整其参数是获得最佳性能和Transformer模型稳定训练的必要步骤。

获取即时帮助、个性化解释和交互式代码示例。

---

### 正则化方法 (Dropout, 标签平滑)

# 正则化方法 (Dropout, 标签平滑)

训练大型Transformer模型时，管理过拟合 (overfitting)以及确保模型对未见过的数据有良好的泛化能力，是令人关注的问题。仅仅构建一个大型网络并在大量数据集上训练是不够的；需要技巧来防止模型仅仅记住训练样本。正则化 (regularization)方法是实现工具箱中获得稳定表现的必不可少的工具。两种广泛用于Transformer的技巧是Dropout和标签平滑。

### Dropout

Dropout是一种简单而有效的正则化 (regularization)技术，最初是为了对抗前馈神经网络 (neural network)中的过拟合 (overfitting)而提出的。其核心思想是在每次训练更新时，随机地将一部分神经元输出“置零”。这可以防止单元过度依赖于特定的其他单元，促使网络学习到更分散、更具弹性的表示。

在标准的Transformer架构中，Dropout应用于以下几个位置：

1. **在嵌入 (embedding)和位置编码 (positional encoding)相加之后：** 应用于合并的输入表示进入编码器或解码器堆栈之前。
2. **在每个子层内部：** 应用于多头注意力 (multi-head attention)操作之后（在残差连接和层归一化 (normalization)之前）以及位置前馈网络之后（同样，在残差连接和层归一化之前）。
3. **可选地，在注意力权重 (weight)上：** 有时，Dropout直接应用于注意力权重 softmax(QKT/dk)softmax(QK^T/\sqrt{d\_k})softmax(QKT/dk​​)，然后才与 VVV 相乘。这有时被称为注意力Dropout。

将单元输出置零的概率 pdropp\_{drop}pdrop​ 是一个超参数 (parameter) (hyperparameter)。典型值范围为0.1到0.3，尽管最佳值取决于具体的模型大小、数据集和任务。在训练期间，Dropout**之前**那一层的输出会以 pdropp\_{drop}pdrop​ 的概率随机置零。其余的输出通常会按 1/(1−pdrop)1 / (1 - p\_{drop})1/(1−pdrop​) 的因子进行缩放，以保持输出的期望总和。这种缩放确保了下一层的期望输入在训练和推理 (inference)之间保持一致。

在推理或评估期间，Dropout被禁用。所有单元都被使用，并且不应用任何缩放（假设缩放已在训练期间完成）。这确保了给定输入的确定性输出。

考虑一个子层内的简化示例：

```python

import torch
import torch.nn as nn

dropout_prob = 0.1

dropout_layer = nn.Dropout(p=dropout_prob)

output_dropped = dropout_layer(sublayer_output)
normalized_output = layer_norm(x + output_dropped)

output_no_drop = dropout_layer(sublayer_output)
normalized_output = layer_norm(x + output_no_drop)
```

通过这种方式注入噪声，Dropout促使模型发展出冗余性，并防止神经元之间复杂的共同适应，从而提高了泛化表现。

### 标签平滑

标签平滑处理了过拟合 (overfitting)的另一个方面，它与模型对其预测的置信度相关。在分类任务中（例如在语言建模中预测下一个词元 (token)），模型通常使用交叉熵损失进行训练，并采用硬性、独热编码的目标标签。例如，如果正确的下一个词对应词汇表 (vocabulary)中大小为 KKK 的索引5，则目标向量 (vector)为 `[0, 0, 0, 0, 1, 0, ..., 0]`。

使用这种硬性目标进行训练会促使模型将正确类别对应的logit推向正无穷，而将所有其他类别对应的logit推向负无穷，导致预测类别具有极高的置信度（概率接近1.0）。这种过度的置信度可能是有害的：

- 它使模型适应性变差。
- 如果模型即使给正确类别分配了很小的概率，但对错误类别却非常自信，这可能会在训练期间对其进行严厉惩罚。
- 它可能无法准确体现语言或数据中固有的真实不确定性。

标签平滑正则化 (regularization)（LSR）修改目标标签，以纳入少量的不确定性。我们不再要求模型将1.0的概率分配给正确类别，而是将一个小的概率质量 ϵ\epsilonϵ（epsilon）均匀地分配给所有类别，包括不正确的类别。

原始的独热目标分布 yky\_kyk​（即当真实类别为 k=tk=tk=t 时 yk=1y\_k=1yk​=1，否则 yk=0y\_k=0yk​=0）被平滑分布 yk′y'\_kyk′​ 所取代：

yk′=yk(1−ϵ)+ϵKy'\_{k} = y\_{k} (1 - \epsilon) + \frac{\epsilon}{K}yk′​=yk​(1−ϵ)+Kϵ​

这里，KKK 是类别的总数（例如，词汇表大小）。真实类别现在具有 1−ϵ+ϵ/K1 - \epsilon + \epsilon/K1−ϵ+ϵ/K 的目标概率，而所有其他类别都具有 ϵ/K\epsilon/Kϵ/K 的目标概率。

我们通过一个小的例子来说明这一点。假设我们有 K=5K=5K=5 个类别，真实类别是索引2（基于0）。当 ϵ=0.1\epsilon=0.1ϵ=0.1 时：

- **独热目标：** `[0.0, 0.0, 1.0, 0.0, 0.0]`
- **平滑目标：**
  - 索引2： 1×(1−0.1)+0.1/5=0.9+0.02=0.921 \times (1 - 0.1) + 0.1 / 5 = 0.9 + 0.02 = 0.921×(1−0.1)+0.1/5=0.9+0.02=0.92
  - 其他索引（0, 1, 3, 4）： 0×(1−0.1)+0.1/5=0.0+0.02=0.020 \times (1 - 0.1) + 0.1 / 5 = 0.0 + 0.02 = 0.020×(1−0.1)+0.1/5=0.0+0.02=0.02
  - 结果向量： `[0.02, 0.02, 0.92, 0.02, 0.02]` (注意：总和为1.0)

0123400.20.40.60.81独热平滑 (ε=0.1)独热编码 vs. 标签平滑 (K=5, ε=0.1)

> 比较了5类别问题中独热目标向量和标签平滑目标向量（当 ϵ=0.1\epsilon=0.1ϵ=0.1 时）。真实类别的概率质量被降低，并均匀地分配给所有类别。

在计算交叉熵损失时，模型现在会因对正确预测过于自信而受到惩罚，并被鼓励为其他合理输出分配小的非零概率。损失函数 (loss function)变为：

LLS=−∑k=1Kyk′log⁡(pk)L\_{LS} = - \sum\_{k=1}^{K} y'\_{k} \log(p\_k)LLS​=−k=1∑K​yk′​log(pk​)

这里 pkp\_kpk​ 是模型对类别 kkk 预测的概率。这促使正确类别和不正确类别之间logit的差异是有限的，起到正则化作用。

ϵ\epsilonϵ 的常用值是0.1。研究表明，标签平滑通常能改善序列到序列任务中的困惑度和BLEU分数，并能带来更好的模型校准。

Dropout和标签平滑都是大型Transformer模型训练方案中的标准组成部分。它们与其他元素（如适当的权重 (weight)初始化、优化算法（AdamW）和学习率调度）相互配合，以稳定训练并提高最终模型在未见过数据上的表现。为 pdropp\_{drop}pdrop​ 和 ϵ\epsilonϵ 选择合适的值通常需要根据验证集表现进行实验和调整。

获取即时帮助、个性化解释和交互式代码示例。

---

### 梯度裁剪

# 梯度裁剪

虽然像 AdamW 这样的优化器和学习率调度等技术对 Transformer 模型训练很有帮助，但这些模型自身的层数多以及其计算的特点有时会带来另一个数值问题：梯度爆炸。与循环网络中常提及的梯度消失不同，梯度爆炸指的是在反向传播 (backpropagation)过程中，梯度幅度变得过大的情况。这些过大的梯度可能导致在优化器步进期间模型权重 (weight)产生相应的大幅更新，从而可能引起数值溢出、训练不稳定（例如，损失震荡或发散），甚至丢失先前学到的信息。

梯度爆炸可能在深层网络中出现，因为在反向传播过程中，梯度是逐层相乘的。如果梯度范数持续大于1，它们的乘积会随网络深度呈指数级增长。虽然层归一化 (normalization)等组件有助于缓解此问题，但仍可能突然出现大梯度，这可能是由于特定输入序列或注意力机制 (attention mechanism)或前馈网络内部的交互所致，尤其是在训练初期模型参数 (parameter)尚未稳定时。

梯度裁剪是一种直接用于解决此问题的技术。它的作用是在优化器更新模型权重之前，对梯度的幅度（范数）设定上限。如果所有模型参数梯度的总范数超过预设阈值，梯度将按比例缩小到与该阈值相等。这能防止单个大梯度事件破坏训练的稳定性。

### 按全局范数裁剪

最常用的方法是**按全局范数裁剪**。这包括计算L2范数（欧几里得范数），该范数针对将模型中*所有*可训练参数 (parameter)的*所有*梯度连接起来形成的梯度向量 (vector)。令 GGG 表示这个包含所有参数 θi\theta\_iθi​ 的梯度 ∂L∂θi\frac{\partial L}{\partial \theta\_i}∂θi​∂L​ 的全局梯度向量。L2范数计算方式如下：

全局范数=∑i(∂L∂θi)2\text{全局范数} = \sqrt{\sum\_i \left( \frac{\partial L}{\partial \theta\_i} \right)^2}全局范数=i∑​(∂θi​∂L​)2​

然后我们将这个 `全局范数` 与预设的 `max_norm` 阈值进行比较。如果 `全局范数` 超过 `max_norm`，则整个梯度向量 GGG 将按比例缩放：

G裁剪后=G×max\_norm全局范数G\_{\text{裁剪后}} = G \times \frac{\text{max\\_norm}}{\text{全局范数}}G裁剪后​=G×全局范数max\_norm​

如果 `全局范数` 小于或等于 `max_norm`，则梯度保持不变：G裁剪后=GG\_{\text{裁剪后}} = GG裁剪后​=G。

这种方法的一个重要特点是它统一缩放所有梯度。这意味着参数空间中总梯度向量的*方向*得以保留；仅其幅度受到限制。这有助于防止步长过大，而不会大幅改变梯度所指示的优化路径。

G

start

计算梯度
(loss.backward())

computeₙorm

计算全局L2范数
(全局范数 = ||G||u2082)

start->computeₙorm

checkₜhreshold

全局范数 > maxₙorm 吗？

computeₙorm->checkₜhreshold

scale\_gradient

重新缩放梯度
(G = G \* maxₙorm / 全局范数)

checkₜhreshold->scale\_gradient

 是

no\_clipping
保持原始梯度

checkₜhreshold->no\_clipping

 否

update\_weights

应用优化器步进
(optimizer.step())

checkₜhreshold->update\_weights

scale\_gradient->update\_weights

> 流程图，说明了按全局范数裁剪梯度的过程。

### 实现与阈值选择

深度学习 (deep learning)框架提供了内置函数用于梯度裁剪，使其易于集成到训练循环中。它通常在 `loss.backward()` 调用（计算梯度）*之后*、`optimizer.step()` 调用（根据梯度更新权重 (weight)）*之前*应用。

```python

outputs = model(inputs)
loss = compute_loss(outputs, targets)

loss.backward()

gradient_threshold = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_threshold)

optimizer.step()

optimizer.zero_grad()
```

在 TensorFlow 中，等效操作可能涉及使用 `tf.clip_by_global_norm`，通常在自定义训练循环中，或者在指定时由更高级别的API隐式处理。

`max_norm` 值是一个超参数 (parameter) (hyperparameter)，通常需要进行一些调整。常见值通常在 [0.5, 5.0] 范围内，1.0 是一个常用起始点。

- **过低：** 可能过度减慢训练速度，因为它过度限制梯度幅度，尤其是在训练初期大梯度确实包含有用信息时。
- **过高：** 可能无法防止由真正有问题的梯度爆炸引起的不稳定。

在训练期间监测实际梯度范数（如果你的框架或监测工具允许）有助于确定阈值的选择。如果你观察到频繁的裁剪事件或裁剪前范数非常大，这证实了裁剪的必要性，并有助于调整阈值。否则，观察训练损失曲线以判断稳定性是主要的经验指导。

梯度裁剪是训练大型 Transformer 模型中的标准做法。它起到保护作用，补充了仔细的初始化、归一化 (normalization)层和学习率调度等其他技术，以促进稳定有效的收敛，特别是对于梯度动态可能很复杂的深层架构。

获取即时帮助、个性化解释和交互式代码示例。

---

### 混合精度训练

# 混合精度训练

使用标准32位浮点精度（FP32）训练大型Transformer模型会消耗大量计算资源并占用大量内存。混合精度训练提供了一个有效的办法，它在较低精度格式（如16位浮点数FP16或BF16）下执行某些操作，同时将主权重 (weight)等主要部分保留在FP32中。这种方式能显著加快计算速度并减少内存占用，通常对最终模型的精度影响很小或没有影响。

### 动机：速度与内存

现代硬件加速器，特别是配备NVIDIA Tensor Cores等专用单元的GPU，在较低精度（FP16或BF16）下执行矩阵乘法操作时，相比FP32能提供很大的性能提升。以16位精度执行前向和后向传播的部分步骤，直接意味着更快的训练迭代。

此外，与FP32相比，使用16位格式可将存储激活值、梯度和可能的模型权重 (weight)所需的内存减半。这种内存节省使得以下成为可能：

1. 训练使用FP32时无法放入内存的更大模型。
2. 使用更大的批处理大小，这可以提高梯度准确性并可能加快收敛速度。
3. 减少分布式训练设置中的通信开销。

### 混合精度训练的工作原理

主要思想是发挥较低精度在速度和内存上的优势，用于大部分计算，同时通过策略性地使用FP32来保持数值稳定性。虽然实现方式略有不同，并且通常由深度学习 (deep learning)框架自动处理，但典型过程包含多个组成部分：

1. **FP32主权重 (weight)：** 模型权重的主要副本通常保存在FP32中。这确保了权重更新（涉及随时间累积的小梯度值）不会受到16位格式有限精度的影响。
2. **FP16/BF16计算：** 在前向和后向传播过程中，权重被转换为FP16或BF16，用于自注意力 (self-attention)层和前馈层中的矩阵乘法等计算密集型操作。在这些传播过程中生成的激活值和梯度也以较低精度格式存储。
3. **损失缩放：** FP16的动态范围比FP32小很多。在FP16中计算的梯度，特别是对于深层网络或小损失值，可能变为零（下溢）。为避免此情况，计算出的损失值在反向传播 (backpropagation)开始*之前*会乘以一个缩放因子。这会放大梯度，将它们推入FP16的可表示范围。
4. **权重更新：** 在更新FP32主权重之前，计算出的梯度（FP16/BF16格式且已放大）会除以相同的缩放因子，以使它们恢复到正确的数量级。这些未缩放的梯度随后被转换为FP32，并使用选定的优化器（例如AdamW）来更新FP32主权重。

现代框架通常使用*动态损失缩放*，其中缩放因子在训练期间自动调整。如果检测到溢出（梯度变为`Inf`或`NaN`），缩放因子会减小。如果梯度在一定步数内保持稳定，缩放因子可能会增加，以更好地使用FP16的动态范围。

### FP16与BF16的选择

使用两种常见的16位格式：

- **FP16 (IEEE 半精度)：** 使用1位符号位、5位指数位和10位尾数位。它比BF16提供更高的精度，但与FP32相比动态范围小得多。这个有限的范围使其更容易受到下溢和溢出的影响，因此需要仔细的损失缩放。大多数现代GPU都对FP16有出色的硬件支持。
- **BF16 (BFloat16)：** 使用1位符号位、8位指数位（与FP32相同）和7位尾数位。它的主要优点是动态范围与FP32相似，因此远不那么容易出现下溢/溢出问题。损失缩放通常不需要或只需要更简单的静态缩放。然而，它比FP16提供较低的精度（尾数位更少）。在TPU和新一代GPU（例如NVIDIA Ampere及更高版本）中很常见对它的硬件支持。

选择通常取决于硬件是否可用。如果两者都支持，BF16可能会提供稍微简单的训练设置，因为它在数值范围上具有鲁棒性，而FP16在其更高精度有利的场景下可能会略微更好，前提是使用了有效的损失缩放。

### 实际实施

深度学习 (deep learning)框架提供方便的API，通过最少的代码改动来实现混合精度训练。

- **PyTorch：** 使用`torch.cuda.amp`（自动混合精度）模块。它提供上下文 (context)管理器（`autocast`）和梯度缩放工具（`GradScaler`）。

  ```python

  import torch
  from torch.cuda.amp import GradScaler, autocast

  scaler = GradScaler()
  model = YourTransformerModel().cuda()
  optimizer = torch.optim.AdamW(model.parameters(), lr=...)

  for inputs, targets in dataloader:
      inputs, targets = inputs.cuda(), targets.cuda()

      optimizer.zero_grad()

      with autocast(dtype=torch.float16):
          outputs = model(inputs)
          loss = compute_loss(outputs, targets)

      scaler.scale(loss).backward()

      scaler.step(optimizer)

      scaler.update()
  ```
- **TensorFlow：** 使用`tf.keras.mixed_precision` API。您可以设置全局策略或按层应用它。当使用`model.fit`时，TensorFlow会自动处理损失缩放。

  ```python

  import tensorflow as tf

  policy = tf.keras.mixed_precision.Policy('mixed_float16')
  tf.keras.mixed_precision.set_global_policy(policy)

  inputs = tf.keras.Input(...)

  outputs = tf.keras.layers.Dense(vocab_size, activation='softmax', dtype='float32')(x)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  optimizer = tf.keras.optimizers.AdamW(...)

  model.compile(optimizer=optimizer, loss='...', metrics=[...])
  model.fit(dataset, epochs=...)
  ```

虽然混合精度训练非常有效，但建议监控训练稳定性，并偶尔将最终模型性能与基线FP32运行进行比较，尤其是在首次将其应用于新架构或任务时。某些数值操作，例如大范围约简或需要高精度的计算，有时可能会因框架启发式算法而从自动转换中排除，或可能需要手动配置以保留在FP32中。

FP32 BaselineMixed Precision (FP16/BF16)020406080100120140160180指标相对速度（百分比）相对内存占用（百分比）混合精度训练的优点（示意）相对百分比（%）

> 示意性比较，显示混合精度训练可能带来的速度提升（例如，快1.8倍）和内存节省（例如，减少45%）。实际收益取决于模型、硬件和具体的实现方式。

混合精度训练已成为深度学习从业者工具箱中的一项标准技术，特别是对于Transformer等资源密集型模型。通过智能地结合较低精度计算和保持数值稳定性的机制，它使得训练迭代更快，并且在现有硬件限制下使用更大、能力更强的模型成为可能。

获取即时帮助、个性化解释和交互式代码示例。

---

### 高效注意力算法实现 (FlashAttention)

# 高效注意力算法实现 (FlashAttention)

Transformer的计算需求，尤其是自注意力 (self-attention)在序列长度NNN上的二次复杂度(O(N2)O(N^2)O(N2))，构成了一个主要瓶颈。优化方法可以直接解决这一挑战。尽管稀疏或线性注意力等架构变体旨在以较低的理论复杂度**逼近**注意力机制 (attention mechanism)，但重点在于优化**精确**的缩放点积注意力计算。这使其在实践中，特别是在现代GPU等硬件上，明显更快且内存效率更高。

标准注意力实现中的主要挑战不仅仅是浮点运算（FLOPs）的数量，而是内存带宽的限制。计算涉及几个大型中间矩阵，尤其值得注意的是N×NN \times NN×N的注意力得分矩阵S=QKTS = QK^TS=QKT和概率矩阵P=softmax(S)P = \text{softmax}(S)P=softmax(S)。这些矩阵必须从GPU的高带宽内存（HBM）中读写，而HBM比片上SRAM慢得多。对于长序列，在不同内存层级之间传输这些矩阵所花费的时间通常会主导实际的计算时间。

## I/O感知型注意力算法

认识到这一内存瓶颈，研究人员发展出了“I/O感知型”注意力算法。这些方法旨在最小化慢速HBM和快速SRAM之间的数据移动。最突出且被广泛采用的例子是**FlashAttention**。

### FlashAttention：融合算子与分块处理

FlashAttention不改变注意力的数学定义；它计算出的输出与标准算法完全相同。其创新之处在于它**如何**执行计算。核心思想包括：

1. **算子融合（Kernel Fusion）**：FlashAttention没有为矩阵乘法（QKTQK^TQKT）、缩放、遮掩（如适用）、softmax以及与VVV的最终乘法执行独立的GPU操作（算子），而是将这些操作融合到一个更大的单一算子中。这大幅减少了数据需要从HBM读取和写回的次数。
2. **分块处理（Tiling）**：融合后的算子以更小的块或“瓦片”处理输入矩阵（QQQ, KKK, VVV），这些块可以完全放入GPU的快速SRAM中。它通过遍历键和值的块来计算查询块的注意力输出。中间结果，例如注意力得分矩阵的块和softmax归一化 (normalization)统计数据，会尽可能地保留在SRAM中。
3. **在线Softmax（Online Softmax）**：softmax计算以数值稳定的方式逐块执行。当算法为给定查询块遍历键和值的块时，它会维护softmax所需的运行统计数据（用于减法的最大值，用于归一化的指数和）。这避免了在应用softmax之前计算和存储完整的N×NN \times NN×N得分矩阵SSS。

G

clusterₛtandard

标准注意力

cluster\_flash

FlashAttention

HBM1

HBM (慢速内存)

Compute1

计算单元 (GPU核心)

HBM1->Compute1

读取 Q, K

HBM1->Compute1

读取 S

HBM1->Compute1

读取 P, V

SRAM1

SRAM (快速缓存)

Compute1->HBM1

写入 S=QKᵀ

Compute1->HBM1

写入 P=softmax(S)

Compute1->HBM1

写入 O=PV

HBM2

HBM (慢速内存)

SRAM2

SRAM (快速缓存)

HBM2->SRAM2

加载 Q, K, V 块

SRAM2->HBM2

写入输出块 O

Compute2

计算单元 (GPU核心)

SRAM2->Compute2

处理块

Compute2->SRAM2

累积输出块

Compute2->Compute2

融合操作 (QKᵀ, Softmax, PV)
无中间HBM写入

> 内存访问模式对比。标准注意力涉及对较慢的HBM进行多次读/写操作，以处理中间矩阵（SSS, PPP）。FlashAttention对加载到较快SRAM中的数据块执行融合操作，从而最小化HBM访问。

### 优势与影响

这种I/O感知型方法带来了显著的成果：

- **加速：** 与标准的PyTorch或TensorFlow实现相比，FlashAttention可以提供明显的加速（通常是2-4倍或更多），特别是对于内存带宽是主要限制的长序列。
- **内存效率：** 由于大型N×NN \times NN×N中间矩阵（SSS和PPP）不会在HBM中完全实例化，注意力的内存使用量相对于序列长度变为线性O(N)O(N)O(N)，而不是二次方O(N2)O(N^2)O(N2)（不包括Q,K,VQ, K, VQ,K,V本身的内存）。这使得在典型GPU内存限制内训练比以前长得多的序列模型成为可能。
- **精确性：** 与近似方法不同，FlashAttention计算出精确的注意力输出，确保不会因优化而导致模型质量下降。
- **易于集成：** 像官方FlashAttention库这样的实现通常被设计为PyTorch等框架中标准注意力模块的直接替代品，只需要很少的代码改动。

02000400060008000512510251002标准注意力 (时间/内存)FlashAttention (时间)FlashAttention (内存)

> 图表说明性能扩展。标准注意力在时间和内存方面呈现二次方扩展(O(N2)O(N^2)O(N2))，主要由中间矩阵决定。FlashAttention的目标是接近线性的时间扩展（更接近计算限制）和线性的内存扩展(O(N)O(N)O(N))。实际加速效果因硬件和维度而异。

尽管FlashAttention是一个具体的实现，但算子融合和最小化HBM流量的原理对于优化现代硬件上的计算密集型操作非常重要。在训练或部署大型Transformer模型时，特别是处理长上下文 (context)的模型，运用此类优化的注意力实现通常是达到可接受性能和效率的必不可少。许多流行的库和框架正越来越多地直接或通过集成来整合这些技术。

获取即时帮助、个性化解释和交互式代码示例。

---

### 模型并行与数据并行策略

# 模型并行与数据并行策略

如今训练大型Transformer模型，经常会超出单个加速器（如GPU或TPU）的计算和内存限制。当模型拥有数十亿参数 (parameter)，或数据集需要非常大的批处理尺寸以实现稳定收敛时，将工作负载分散到多个设备上就变得必须。实现此目的的两种主要策略是：数据并行和模型并行。

## 数据并行

数据并行可能是分布式训练中最直接、最常用的方法。其核心思想简单：在每个可用设备上复制整个模型，并让每个设备处理输入数据批次的不同部分。

### 工作原理

1. **模型复制**：将完整的模型（包含所有参数 (parameter)）复制到每个参与设备上（例如，单台机器上的多个GPU或跨多台机器）。
2. **数据分片**：全局训练批次被分割成更小的迷你批次。每个设备接收一个独特的迷你批次。
3. **前向与反向传播 (backpropagation)**：每个设备独立地使用其局部迷你批次执行前向传播以计算损失，然后执行反向传播，根据该局部数据计算模型参数的梯度。
4. **梯度同步**：这是重要的通信步骤。在每个设备上计算的梯度会在所有设备之间收集并汇总。一种常见的汇总方法是平均。这通常通过高效的AllReduce通信原语实现。
5. **参数更新**：优化器使用汇总的（例如，平均的）梯度来更新模型参数。由于梯度已汇总，所有模型副本都以相同方式更新，从而确保它们保持同步。

G

cluster\_data

输入数据批次

cluster\_device1

设备 1

cluster\_device2

设备 2

cluster\_deviceN

设备 N

clusterₛync

同步

DataBatch

全局批次

MiniBatch1

迷你批次 1

DataBatch->MiniBatch1

分割

MiniBatch2

迷你批次 2

DataBatch->MiniBatch2

分割

MiniBatchN

迷你批次 N

DataBatch->MiniBatchN

分割

Model1

模型副本

MiniBatch1->Model1

前向/反向传播

Model2

模型副本

MiniBatch2->Model2

前向/反向传播

ModelN

模型副本

MiniBatchN->ModelN

前向/反向传播

Grad1

局部梯度 1

Model1->Grad1

Sync

汇总梯度
(例如，AllReduce平均)

Grad1->Sync

Grad2

局部梯度 2

Model2->Grad2

Grad2->Sync

GradN

局部梯度 N

ModelN->GradN

GradN->Sync

Update

优化器步骤

Sync->Update

Update->Model1

更新参数

Update->Model2

更新参数

Update->ModelN

更新参数

> 数据并行工作流程。模型被复制，数据被分割，梯度在本地计算，跨设备汇总，并用于同步更新所有模型副本。

### 优点和局限性

数据并行主要优点是其相对简单性，以及随着设备数量增加，训练时间可能实现近乎线性的加速，尤其是在计算量远超通信量时。大多数深度学习 (deep learning)框架都提供其实现（例如PyTorch中的`torch.nn.parallel.DistributedDataParallel`或TensorFlow中的`tf.distribute.MirroredStrategy`）。

然而，数据并行有一个重要限制：*整个*模型必须能够放入单个设备的内存中。如果您的Transformer模型对于一个GPU来说太大，那么仅数据并行将无法满足需求。此外，随着设备数量的增加，同步梯度的通信开销可能成为瓶颈，从而降低增加更多设备带来的效益。

## 模型并行

当模型过大无法放入单个设备时，模型并行就变得必须。我们不是复制模型，而是将模型本身分割到多个设备上。每个设备仅负责存储和计算模型的一部分。

分割模型主要有两种方式：

### 1. 层间（流水线）并行

这种策略将模型垂直划分。不同的层（或层序列）被分配到不同的设备。数据顺序流经这些设备，形成一个处理流水线。

- **机制**：输入数据（或来自前一阶段的激活）进入负责第一组层的设备。这些层的输出激活随后传递给处理下一组层的设备，以此类推，直到产生最终输出。反向传播 (backpropagation)则沿相反方向流动。
- **挑战 - 流水线气泡**：简单的实现会导致设备资源的大量闲置。当一个设备（阶段）正在处理一个批次时，其他设备可能处于空闲状态，等待输入或已完成处理上一个批次。这种空闲时间被称为“流水线气泡”。
- **缓解 - 微批处理**：为了减少气泡开销，输入迷你批次被进一步分割成更小的微批次。这些微批次被顺序送入流水线，使得不同阶段能够并发处理不同的微批次，从而重叠计算并减少空闲时间。GPipe或PipeDream等框架为此实现了复杂的调度。

G

clusterₚipeline

流水线并行（3 阶段）

cluster\_bubble

流水线气泡（简化）

Device1

设备 1

层1..k

Device2

设备 2

层k+1..m

Device1->Device2

激活

Device3

设备 3

层m+1..N

Device2->Device3

激活

Output

输出

Device3->Output

Input

输入
迷你批次

Input->Device1

t0

时间 ->

d1

设备 1

mb1₁

MB1

d2

设备 2

idle1

...

d3

设备 3

idle2

...

mb1₂

MB1

mb1₃

MB1

idle3

...

idle4

...

> 流水线并行将模型层分割到不同设备上。数据顺序流动。在没有微批处理的情况下（简化显示），设备会经历空闲时间（“气泡”）。

### 2. 层内（张量）并行

这种策略将模型水平划分。它涉及将单个大型层（如自注意力 (self-attention)或FFN中的权重 (weight)矩阵）*内部*的计算分割到多个设备上。

- **机制**：考虑一个大型矩阵乘法 Y=XAY = XAY=XA。并非在单个设备上计算，我们可以将矩阵 AAA 按列分割（A=[A1∣A2]A = [A\_1 | A\_2]A=[A1​∣A2​]），并在设备1上计算 XA1XA\_1XA1​，在设备2上计算 XA2XA\_2XA2​。然后将结果 Y1=XA1Y\_1 = XA\_1Y1​=XA1​ 和 Y2=XA2Y\_2 = XA\_2Y2​=XA2​ 连接起来形成 Y=[Y1∣Y2]Y = [Y\_1 | Y\_2]Y=[Y1​∣Y2​]。类似的分割可以按行应用，或应用于Transformer块内的其他操作。
- **要求**：这需要参与计算单个层的设备之间有很大的通信带宽，因为中间激活通常需要交换（例如，使用AllGather或ReduceScatter操作）。
- **应用场景**：张量并行对于具有特大型单个层的模型很有效，在这种情况下，仅流水线并行可能不足或导致阶段工作负载不平衡。它通常在流水线的特定阶段使用。NVIDIA的Megatron-LM等库提供了高效的实现。

### 优点和局限性

模型并行使得训练那些超出单个设备内存容量的模型成为可能。流水线并行通常更容易理解，但受气泡开销影响，需要微批处理以提高效率。张量并行可以处理非常大的层，但要求高设备间带宽并增加了实现复杂性。这两种形式通常比数据并行需要更细致的实现和调试。

## 混合策略和高级技术

实际中，训练目前先进的大型语言模型通常涉及组合这些策略。一种常见配置是使用流水线并行将层块分布到节点上，并使用张量并行分割每个流水线阶段内的大层。数据并行随后常应用于这种模型并行配置之*上*，在多组设备上复制整个流水线/张量分割模型以并发处理更多数据。这有时被称为3D并行（数据、流水线、张量）。

此外，诸如ZeRO（零冗余优化器）及其框架实现（例如DeepSpeed、PyTorch FSDP - 完全分片数据并行）提供了精巧的组合。它们作用类似数据并行，但不仅分片数据，还分片优化器状态、梯度，以及可选地将模型参数 (parameter)本身分片到数据并行工作器中。这大幅减少了每个设备的内存占用，使得数据并行能够扩展到比以前大得多的模型，有时甚至能消除中等大小模型对复杂流水线或张量并行的需求。

## 选择合适的策略

- **从数据并行开始**：如果您的模型适合单个设备但训练太慢，数据并行（可能结合ZeRO/FSDP）通常是首先尝试的最简单策略。
- **内存受限时使用流水线并行**：如果模型激活内存或参数 (parameter)数量超过单个设备容量，则需要流水线并行。仔细考虑阶段数量和微批次大小，以平衡负载并最大程度减少气泡开销。
- **针对大型层使用张量并行**：如果特定层（如大型嵌入 (embedding)表或FFN）即使在流水线阶段内也是瓶颈，请为这些层引入张量并行，确保足够的设备间带宽。
- **组合以实现扩展**：对于真正大型的模型（数千亿或数万亿参数），结合数据、流水线和张量并行的混合方法（通常由专门的库管理）通常是必须的。

理解这些分布式训练模式对于有效地使用大型Transformer架构非常重要。尽管深度学习 (deep learning)框架提供了实现这些策略的工具，但掌握数据流、通信模式和潜在瓶颈的底层机制，能让您选择正确的方法并优化针对特定模型和硬件配置的训练过程。

获取即时帮助、个性化解释和交互式代码示例。

---

### 实践：分析注意力权重分布

# 实践：分析注意力权重分布

理解训练好的Transformer模型*如何*进行预测，与优化其训练速度或内存使用同等重要。分析注意力权重 (weight)分布能够帮助我们了解模型的内部推理 (inference)过程。通过它，我们能观察模型在处理特定标记 (token)时，将注意力集中在输入序列的哪些部分。这种做法对于调试、验证模型行为以及弄清模型是学到了有意义的关联还是依赖于虚假相关性非常有帮助。

### 获取注意力权重 (weight)

大多数现代深度学习 (deep learning)框架都提供了机制来访问模型中的中间激活，包括注意力权重。具体方法取决于所使用的框架以及Transformer模型的实现方式。常见的方法包括：

1. **修改`forward`传递：** 修改模型的`forward`方法（或等效方法），使其在返回主要输出的同时也返回注意力权重。如果您能控制模型的源代码，这通常很简单。
2. **使用钩子（Hooks）：** PyTorch等框架提供“钩子”，可以注册在特定模块（如`Attention`层）上。这些钩子可以在正向或反向传播 (backpropagation)过程中捕获模块的输出（或输入），而无需永久修改模型代码。
3. **模型配置：** 某些预构建的Transformer实现（例如，来自Hugging Face Transformers等库）带有一个配置标志（如`output_attentions=True`），可以指示模型返回注意力权重。

无论采用哪种方法，目标都是获取注意力概率矩阵，这些矩阵通常是在每个注意力头内部的softmax操作之后计算得到的。对于标准的多头注意力 (multi-head attention)层，输出的注意力权重通常具有`[batch_size, num_heads, sequence_length_query, sequence_length_key]`的形状。对于自注意力 (self-attention)，`sequence_length_query`和`sequence_length_key`是相同的（即输入序列长度）。对于解码器中的交叉注意力，`sequence_length_key`对应于编码器输出序列长度。

假设您已经获取了批处理中单个样本的特定层和头的注意力权重。结果张量，我们称之为`attention_probs`，可能具有`[num_heads, seq_len, seq_len]`的形状。我们可以选择一个特定的头进行可视化，从而得到一个形状为`[seq_len, seq_len]`的二维矩阵。

### 使用热力图可视化注意力

可视化这些二维注意力矩阵最常用的方法是使用热力图。热力图中的每个单元格 (i,j)(i, j)(i,j) 表示从第iii个查询标记 (token)到第jjj个键标记的注意力权重 (weight)。值越高（颜色越亮）表示注意力越强。

考虑一个简单的输入句子：“The quick brown fox jumps”。我们来可视化第一层、第一个头的自注意力 (self-attention)权重。

```python

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

attention_matrix = np.random.rand(5, 5)

np.fill_diagonal(attention_matrix, attention_matrix.diagonal() + 0.3)

attention_matrix /= attention_matrix.sum(axis=1, keepdims=True)

tokens = ["The", "quick", "brown", "fox", "jumps"]
seq_len = len(tokens)

plt.figure(figsize=(7, 6))
sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis", annot=True, fmt=".2f")
plt.xlabel("标记（指向）")
plt.ylabel("查询标记（源自）")
plt.title("自注意力权重（第一层，第一个头）")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
```

这是使用Plotly进行网页可视化的一个示例，显示了相同句子的注意力权重。

ThequickbrownfoxjumpsjumpsfoxbrownquickThe0.10.20.30.40.50.6

> 热力图显示了标记之间的注意力分数。行代表生成查询的标记，列代表生成键/值的标记。更亮的单元格表示更高的注意力分数。

### 解读注意力模式

在分析这些可视化时，寻找重复出现的模式：

- **自身注意力：** 通常，标记 (token)会强烈关注自身（自注意力 (self-attention)中对角线很强）。这有助于保持标记本身的表示。
- **局部上下文 (context)：** 相邻标记之间存在高注意力分数。这表明注意力头侧重于局部词语依赖性。
- **句法依赖：** 寻找反映语法的模式。例如，动词可能会强烈关注其主语或宾语，即使它们在序列中相距较远。形容词可能会关注它们所修饰的名词。
- **特殊标记：** 观察对`[CLS]`、`[SEP]`、`[BOS]`、`[EOS]`等特殊标记的注意力情况。有时，`[CLS]`标记（在类似BERT的模型中）会汇总整个序列的信息，显示出广泛的注意力。在解码器中，标记通常会关注*前一个*句子或片段的`[EOS]`（序列结束）标记。
- **头部专长：** 为同一层内不同的注意力头生成热力图。您会经常发现截然不同的模式。一个头可能侧重于局部上下文，另一个侧重于特定的句法关系（如名词-动词配对），而另一个可能表现出更广泛、几乎一致的注意力。这种专长是多头注意力 (multi-head attention)的动因。
- **层级深度：** 较低层中的模式通常更具局部性，并侧重于句法结构。在较高层中，注意力模式可以变得更抽象，可能反映语义关系或聚合更长距离的信息。
- **交叉注意力（解码器）：** 当可视化编码器-解码器交叉注意力时（例如，在翻译中），寻找源语言标记（键）和目标语言标记（查询）之间的强对齐 (alignment)。理想情况下，目标词应该强烈关注其对应的源词。这里的错位可能表明翻译错误。

### 考量与局限性

尽管富有洞察力，但注意力可视化并非对模型行为的明确解释。

- **相关性与因果性：** 高注意力并不*证明*某个标记 (token)是特定输出的唯一或主要原因。它表明了信息流的强度。
- **平均效应：** 对注意力头之间权重 (weight)进行平均可能会模糊单个注意力头学习到的专门模式。
- **基于梯度的方法：** 注意力权重反映的是正向传递。其他方法（如基于梯度的显著图）提供了补充性的视角，侧重于哪些输入对输出预测影响最大。
- **Softmax之后：** 这些权重是softmax函数产生的概率。Softmax之前的分数可能显示出不同的相对重要性，尽管概率更容易解释。

分析注意力权重是一种实用技术，可帮助您定性地了解Transformer模型。它通过帮助您理解模型*如何*处理信息来补充定量评估指标，这对于构建更有效和可靠的系统至关重要。这种做法有助于确认复杂机制（如学习到的位置嵌入 (embedding)、层归一化 (normalization)策略或特定优化器）是否带来了合理的内部表示和信息流。

获取即时帮助、个性化解释和交互式代码示例。

---
