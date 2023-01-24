# Residue-based Partially Homomorphic Encryption

## 依赖

Python >= 3.7

基本加解密需要：gmpy2, numpy

加密网络训练需要：pytorch

## 示例

### 加密网络训练

运行main.py，可以直接调用client, leader, server协同加密训练网络。

其中：

1. CryptNet_client.py包含clients训练本地模型、加密并上传梯度的例子。

2. CryptNet_leader.py包含leader分发公钥，解密全局梯度的例子。注意，在此例中将leader用一个单独的进程表示，实际也可能是某个被选举出的用户。
3. CryptNet_server.py包含server聚合加密梯度的过程。

训练网络时的参数在**network/train_params.py**中设置：

- **BACKEND**: 加密方案，可选"plain", "obrbphe", "batchcrypt"，其中plain是不加密训练，batchcrypt是基于BatchCrypt方案实现的加密算法（在comparison/batchcrypt中）

### 加解密功能

在unit_test.py中，有加解密功能正确性验证的示例；在benchmark.py中，有用于测试开销的示例。

## 加解密函数

实现了两套基于Paillier的packing加密方案。

1. 基本的Residue-based Partially Homomorphic Encryption (RBPEH) 方案在**rbphe/residue_cryptosystem.py**中的RBPHE类，该方案将质数$p, q$编码的数字都放到同一个密文中，效率较低。

2. 论文中使用的为Obfuscated RBPHE方案，实现为obfuscated_residue_cryptosystem.py中的ObfuscatedRBPHE类。基本函数包括：

   - ```python
     def _init__(self, sec_param=80, batch_size="auto", precision=10, lg_max_add=10, key_size=1024, encryptor=None):
     ```

     **sec_param**: type int, 随机值的位数

     **batch_size**: type int or "auto"批加密大小，"auto"表示自动计算最大可用批大小

     **precision**: type int，加密精度

     **lg_max_add**: type int，预先设置的最大相加次数的$log_2$对数，通常为$log_2(\lceil client \ number\rceil)$，该值的最大有效值等于precision

     **key_size**: Paillier加密的密钥长度

     **encryptor**: 可以输入已有的Paillier加密器作为该系统的Paillier加密初始化方法

   - ```python
     def encrypt(self, messages):
     ```

     **messages**: type list or numpy.array，待加密的数据，长度若小于设置的batch_size，则会填充额外的0；长度若大于batch_size，会截断前batch_size个。

     **return**: ObfuscatedRBPHECiphertext类型的密文

   - ```python
     def decrypt(self, ciphertext: ObfuscatedRBPHECiphertext)
     ```

     **ciphertext**: type ObfuscatedRBPHECiphertext，密文

     **return**: list类型的解密数据

   - ```python
     def convert_weight(self, weight):
     ```

     **weight**: type list or numpy.array，明文权重值

     **return**: ObfuscatedRBPHEPlaintext类型的明文编码权重值。（明文数据需要先转换为ObfuscatedRBPHEPlaintext类型才能与类型为ObfuscatedRBPHECiphertext的密文数据进行同态计算）

## 协议2

### 新增点

- model_paras_count模块

  `sep_anch_resd()`：分离anchors和residues

  `generate_hashlist()`：生成anchors的hash值

  `count_process()`：选取topk个client作为参数候选集

  `pick_represents()`: 迭代选取所有参数的代表

- switch_1_2开关

  用于切换协议1到协议2，目前采用CON_ACC大于某值时切换

  每轮通信开始时clients需要上传switch_1_2的状态使server及时同步

- retore mask
  **2022.4.26**
  因每个selected client不控制的参数需补0，故直接静态存储长度为batchsize的0向量的密文，在构建anchor的密文时直接加上即可，从而减少加密开销。但因为每个selected client的参数本来分散，故进行一次映射顺序排列，最后收到全局参数时通过restore mask映射回去.

### 修改点

1. 弃用leader，server直接和clients相连

   server工作：分发密钥、接收哈希值、topk、聚合anchors和residues、广播

   clients工作：分离anchors和residues、哈希、上传并接收anchors和residues、训练网络

   具体过程的修改较复杂见代码

2. 在trans_prams.py中新增控制项

   **PARASCOUNT**：控制是否切换协议2（bool）

   **SAVE_MODEL**： 控制是否保存模型（bool）

   ~~**WASTE=False**：控制是否统计加密浪费（bool）~~

   **K**：topk（int:{1,3,5,...}）

   **SPARSE**： 选举出的clients未控制参数的比例（float: {0.05,0.01}）

   **SELECT_MSG = 'selected'**: 某client本轮被选中

   **JUMP_MSG = 'jump'**：某client本轮被跳过

   **POWER_CHOICE**：控制anchor的指数阶p：原参数=anchor*10^(-p)+residues

   第一个分量决定选取模式，第二个为传入参数：

   - [0,_]：p选取最常见的指数阶

   - [1,a]：p选取最小的指数阶，p=最小阶*a

   - [2,a]：直接指定p，p=a

   **CON_ACC**：控制准确率达到何时时切换到协议2（float:{0.75,0.85,0.9})

   **CHOICE_PATH**：存储选举出的clients分布

   **MODEL_PATH**：存储模型

3. 在rbphe_network.py中修改函数

   - 新增`get_raw_grads()`：用于直接返回flatten的模型参数

   - 修改`pack_grad()`：新增参数raw_grad，用于直接对传入的raw_grad加密

     return_waste参数和waste相关代码已弃用

   - 修改`unpack_grad()`：将gradients除以CLIENT_NUM做了平均，原代码中未作平均
   - 新增`unpack_grad_with_residues()`：用于解密anchors后再和residues组合

