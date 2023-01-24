# Model Parameters Count

## 问题描述

在联邦学习过程中，对于DNN网络中的某层权重参数 $\Theta=\{\theta_1,\theta_2,...,\theta_m\}$，所有client都会计算得到一个本地的结果，server不希望每轮通信所有client均传输完整的 $\Theta$ ，而意图指派不同的client传输 $\Theta$ 的部分分量以减少加密和通信开销。在此基础上，server更加希望各个client达成较高的参数共识，即少量的client获得其他client的认可后，可以完成所有参数的传输任务。

参数共识的标准基于参数的数值相近程度。事实上，DNN网络中的各个权重参数值差别较小，如果将某层的权重参数统一数量级后表示为 $\theta_i=p_i+r_i$ 的形式（$p_i$ 表示整数部分，$r_i$ 表示浮点数部分），对 $r_i$ 的哈希操作将可以将数值较为接近的参数 $\theta_i$ 映射到同一哈希值。将多个client对参数 $\theta_i$ 的计算结果的哈希值进行统计后，server可以得到各个不同哈希值的出现频数和其来源client的集合，这些client对于参数 $\theta_i$ 的计算值可视为高度相似。通常情况下，server会将频数最高的哈希值所对应的client集合直接作为本轮传输参数 $\theta_i$ 的候选集，候选集中任一client均可被指派进行参数传输。

直接取频数最高的做法不利于参数共识的实现，各个参数的候选集合通常各不相交。因此这里将对频数较高的哈希值的选取从1个扩展到k个，并将其对应的client集合取并集以扩展参数 $\theta_i$ 的候选集。经此处理后，server即有较高的概率找到在多个参数的候选集中均存在的client，从而指派其完成多个参数的传输工作。但同时，k的增大也会导致选取过程的开销增加。如何寻求选取开销和传输开销的trade-off，是此处需要解决的问题。

## 说明

输入参数以`.npy` 文件存储在`./paras` 文件夹下，运行`run.py`进行计算：

```shell
python run.py dnn_weights.npy
```

- 参数说明

  **k**: 较高频数的哈希值选取个数

  **sparse**: 稀疏度

  **batch_enc**: 每次加密的参数批大小

- 函数说明

  **get_paras()**: 读入参数

  **generate_hashlist()**: 生成哈希值

  **count_process()**: 参数共识过程

## 数学描述

### Topk选择

输入：$k$ , 哈希列表 $\mathbf{h} = \{h_1,h_2,...,h_m\}$

输出：每个参数的可贡献Client集合 $\mathbf{S} = \{S_p^{l_1},S_p^{l_2},...,S_p^{l_n}\}$ ，其中 $S_p^{l_i}\subseteq \{0,1\}^n$

- Top k 选择

  得到每个参数的可贡献Client集合 $\mathbf{S} = \{S_p^{1},S_p^{2},...,S_p^{m}\}$

  以及每个参数的可贡献Client个数 $l = \{l_1,l_2,...,l_m\}$  其中 $l_i = S_p^{i} \cdot \textbf{1}$ 

  按照 $l$ 的降序对 $\mathbf{S}$ 进行排序，得到新的 $\mathbf{S} = \{S_p^{l_1},S_p^{l_2},...,S_p^{l_n}\}$

### 参数共识

输入：每个参数的可贡献Client集合 $\mathbf{S} = \{S_p^{l_1},S_p^{l_2},...,S_p^{l_n}\}$ ，其中 $S_p^{l_i}\subseteq \{0,1\}^n$

输出：共识服务器集合（同样向量表示）：$S_{inter} \subseteq \{0,1\}^n$

$S_{inter}=S_m$ ，对 $i=2,3,..., m$
$$
\begin{align*}
& S_1 = S_p^{l_1} \\
& S_{i+1}=S_i \cdot (\textbf{diag}(S_p^{l_i})+D_i \cdot \textbf{diag}(\textbf{1}-S_p^{l_i}) \\
& D_i = \varepsilon(S_i \cdot S_p^{l_i})
\end{align*}
$$

其中 $\varepsilon$ 为阶跃函数：
$$
\varepsilon(x)=
\begin{cases}
0& ,x \le 0\\
1& ,x > 0
\end{cases} \\
$$

### 计算传输成本

输入：共识服务器集合：$S_{inter} \subseteq \{0,1\}^n$

输出：传输成本 $C_{trans}$

固定参数：加密批次：$b_{enc}$ 、单次加密开销：$c_{enc}$ 

首先计算共识参数个数： $B_{inter}=S_{inter} \cdot \textbf{1} $

不在 $S_{inter}$ 内的参数随机选择传输的Client，假设各参数选择的各不相同
$$
C_{trans} = (B_{inter}/b_{enc} + (m-B_{inter})) \cdot c_{enc}
$$
同时设定稀疏度指标 $\eta$ 和其约束值 $\eta_{sparse}$：
$$
\eta = 1 - B_{inter}/m ， \eta \le \eta_{sparse}
$$

### 优化问题

$$
\begin{equation}
\begin{split}
&\min_{k,\eta} C_{trans} = (B_{inter}/b_{enc} + (m-B_{inter})) \cdot c_{enc}\\
&s.t.\quad  \left\{\begin{array}{lc}
 &\mathbf{S} = F_{topk}(k,\mathbf{h}), \mathbf{S} = \{S_p^{l_1},S_p^{l_2},...,S_p^{l_n}\}\\
& S_{inter} = S_p^{l_1} \cdot \prod_{i=2}^m \textbf{diag}(S_p^{l_i})+D_i \cdot \textbf{diag}(\textbf{1}-S_p^{l_i}) \\
& D_i = \varepsilon(S_i \cdot S_p^{l_i}) \\
& B_{inter}=S_{inter} \cdot \textbf{1} \\
& \eta \le \eta_{sparse}
\end{array}\right.
\end{split}
\end{equation}
$$

