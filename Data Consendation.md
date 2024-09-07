# 单模态数据蒸馏

 将大型数据集压缩成一组小的信息合成样本，使由合成样本集与大数据集训练的模型有相近的效果，降低训练开销。数据集蒸馏任务的**主要难点在于如何设计一个生成算法来高效可行地生成需要的样本**

## Data Distillation

#### 之前方法的缺点

数据选择：需要更多的IPC,有代表性的样本可能无法找出

#### 方法

优化目标：![image-20240616002009820](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240616002009820.png)

采用单步（多步）梯度下降，先在distilled images上更新网络参数，然后用这个网络参数在真实的数据集上去做loss评价，多步和多个epoch的效果更好，优化参数是合成样本矩阵，拟合真实训练集训练网络的参数![image-20240615194759611](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240615194759611.png)



#### 效果

![image-20240616003920562](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240616003920562.png)

#### 用于不同目标

- 用于微调
- 用于训练

- 用于攻击

![img](https://upload-images.jianshu.io/upload_images/9933353-885acacfea05ccfb.png?imageMogr2/auto-orient/strip|imageView2/2/w/957/format/webp)

#### 改进和研究方向

初始化策略，大规模数据集

#### 问题

![image-20240616004711290](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240616004711290.png)

算法里面对于图像的更新是按每个像素点进行更新吗

## Gradient Matching（DC)

#### 之前方法的缺点

数据选择：基于启发式的，不能保证任何下游任务都有最佳解决方案，并且不能保证能选出代表性样本

数据蒸馏：==没有在跨模型架构上尝试方法==

#### 方法

优化目标：minimize在small synthetic set上训练出来的模型在large training set上的training loss(之前的方法，会涉及嵌套循环)，提出近似的优化目标，最小化small synthetic set上训练出来的模型参数与在large training set上训练出来的模型参数的距离

![image-20240616014738827](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240616014738827.png)

经过很多step优化的网络最终的参数相似不容易实现。但如果改成希望每一个step优化之后的模型参数相似就会容易很多，进一步简化优化目标变为每个step两个模型的参数梯度相似：

![image-20240616015339983](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240616015339983.png)

先随机初始化一组small synthetic set，用large training set 和 small synthetic set 分别训练一个模型（同种架构和初始化），在训练的每一个step，计算出两个模型的梯度，用两个模型在每一个step的梯度的距离来作为损失函数(matching loss)，来更新(这里并不是求梯度然后反向传播，因为这样就会嵌套地求两次梯度，论文里用了一种opt-alg的优化方法来替代这里的反向传播)small synthetic set。

![image-20240612131611100](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240612131611100.png)

1. 随机选取一个net，在某一类里面挑选256张真实图片放入net里面输出output_real,得到一个loss_real,计算loss_real关于net参数的梯度，gw_real是所有网络参数梯度的列表

![image-20240712205249356](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240712205249356.png)

2. 随机初始化ipc张合成集（合成集是待优化的参数），将合成集放入同一个net里面输出output_syn，得到一个loss_syn，计算loss_syn关于net参数的梯度，gw_syn是网络参数梯度的列表，将这两个梯度匹配做一个loss_match，再将loss_match反向传播，更新合成集

![image-20240712221859994](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240712221859994.png)

3. 再利用更新的合成集在net上训练，并更新网络参数

![image-20240616015643613](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240616015643613.png)

#### 效果

![image-20240616020138748](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240616020138748.png)

跨架构性能：相同的大数据集选出来的合成数据集，在不同模型上的泛用性

![image-20240616022816973](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240616022816973.png)

## **Dataset Condensation with Distribution Matching(DM)**



## Dataset Distillation by Matching Training Trajectories（MTT)

#### **之前方法的缺点**

使用端到端训练，但这通常需要大量计算和内存，并且会受到不精确的松弛或执行多次迭代导致训练不稳定的影响，Data Distillation

使用单步短程匹配，在计算中可能会累积错误，DC

#### 方法

模仿在真实数据集上训练模型的长程训练动态，将在合成数据上训练的参数轨迹片段与在真实数据上训练的模型中预先记录的轨迹片段相匹配，包括以下3个步骤

1. 生成专家轨迹
2. 诱导合成数据集跟随真实数据集相似的参数轨迹
3. 将合成数据集划分为多个batch减少内存消耗，每个batch仍然包含来自不同类的图像，但每个类的图像要少得多

![image-20240627153545182](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240627153545182.png)

![image-20240627155121686](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240627155121686.png)

#### 效果

![image-20240628102012868](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240628102012868.png)

![image-20240628102954845](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240628102954845.png) 

#### 改进和研究方向

训练和存储专家轨迹的开销高，无法扩展到像Imagenet-1k这样的大规模数据集上->TESLA

#### 问题

为什么N不变M增大准确率下降

和梯度匹配的区别

如何更新合成数据集像素

L为什么要归一化

![image-20240629023035568](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240629023035568.png)

## TESLA

#### 之前方法的缺点

MTT需要对这个式子求导，这涉及到计算和存储T个高阶梯度项的计算图，GPU内存需求变得非常大

![image-20240721142813656](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240721142813656.png)



#### 方法

1.将上述展开

![image-20240721144625235](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240721144625235.png)

2.利用预训练的教师模型来生成软标签

![image-20240721152710740](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240721152710740.png)

#### 效果

## SR2L

#### 之前方法的缺点

以前的方法主要分为四类：双重优化，梯度匹配，分布匹配，轨迹匹配

之前的方法：围绕样本生成和模型训练的双层优化 (bi-level optimization) 来展开

之前方法的局限：由于计算量和 GPU 显存的限制，现有方法主要擅长压缩低分辨率小数据集，没法 scale-up 到大规模高分辨率数据集上，如ImageNet-1K 

#### 方法

提出三阶段数据集蒸馏算法：解耦数据生成和模型训练

目前唯一实现了大规模高分辨率数据集蒸馏的框架，可以将 Imagenet-1K 原始的 1.2M 数据样本压缩到 0.05M (压缩比 1:20)，使用常用的 224x224 分辨率进行蒸馏，在 ImageNet-1K 标准验证集上取得了目前最高的 60.8% Top-1 精度，远超之前所有 SOTA 方法![img](https://imagepphcloud.thepaper.cn/pph/image/262/532/605.jpg)

![img](https://imagepphcloud.thepaper.cn/pph/image/262/532/607.jpg)

第一步是将整个数据集的核心信息压缩进一个模型之中，通过模型参数来存储原始数据集中的信息，类似于我们通常进行的模型训练；

第二步是将这些高度抽象化的信息从训练好的模型参数中恢复出来，类似于DeepInversion,从随机高斯噪声开始合成图像，本文讨论了多种不同损失和正则函数对于恢复后图像的质量以及对数据集蒸馏任务的影响；==将多张图像的信息压缩在边缘层中，再优化高斯噪声训练的边缘层结果和真实数据集训练的边缘层结果之间的差距==

第三步也是提升最大的一步：对生成的数据进行类别标签重新校准。采用了 FKD（Fast Knowledge Distillation） 的方式，生成每个 crop 对应的 soft label，并作为数据集新的标签存储起来。

#### 效果

在相同 IPC 情况下，本文实验结果远超之前方法 TESLA。同时当模型结构越大，训练得到的精度越高

![image-20240616031108206](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240616031108206.png)

生成的图像也更接近于真实图像

![image-20240616031643649](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240616031643649.png)

#### 改进和研究方向

更大规模数据集，除图像外的其他数据类型

数据集多样性

- Progressive trajectory matching for medical dataset distillation

## DiM: Distilling Dataset into Generative Model

#### 之前方法的缺点



#### 方法

![image-20240710164159426](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240710164159426.png)

![image-20240710164211749](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240710164211749.png)

#### 效果

#### 改进和研究方向

## DATASET DISTILLATION IN LARGE DATA ERA

在SR2L的基础上，引入了课程数据增强Curriculum Data Augmentation (CDA)，将方法拓展到完整的ImageNet-1K/21Kd

#### 效果

![image-20240616033823204](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240616033823204.png)

![image-20240616105506719](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240616105506719.png)

![image-20240616105517597](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240616105517597.png)

![image-20240616105552331](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240616105552331.png)

## Generalized Large-Scale Data Condensation via Various Backbone and Statistical Matching



## 噪声标签数据蒸馏实验

### MTT

#### NO ZCA

##### cifar10

cifar10-N ipc1 mtt nozca

![image-20240713225813562](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240713225813562.png)

cifar10-N ipc10 mtt nozca

![image-20240713094957248](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240713094957248.png)

cifar10-N ipc50 mtt nozca

![image-20240713190346345](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240713190346345.png)

##### cifar100

whole dataset

![image-20240717113442621](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240717113442621.png)

ipc1 mtt nozca

![image-20240718104238670](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240718104238670.png)

ipc10 mtt nozca

![image-20240717212312949](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240717212312949.png)

ipc50 mtt nozca

![image-20240719173804477](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240719173804477.png)

#### ZCA

##### cifar10

cifar10-N whole dataset zca 0.58?

cifar10-N ipc1 mtt zca 

![image-20240714190418583](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240714190418583.png)

cifar10-N ipc10 mtt zca

cifar10-N ipc50 mtt zca

##### cifar100

cifar100-N whole dataset zca

![image-20240714124650295](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240714124650295.png)

cifar100-N ipc1 mtt zca 

![image-20240714174542232](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240714174542232.png)

cifar100-N ipc10 mtt zca 

cifar100-N ipc50 mtt zca 

![image-20240716013230616](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240716013230616.png)

### DC

#### cifar10

cifar10-N ipc1 dc 

![image-20240713095823132](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240713095823132.png)

cifar10-N ipc10 dc

![image-20240714122519652](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240714122519652.png)

cifar10-N ipc50 dc

#### cifar100

cifar100-N ipc1 dc

![image-20240713100412714](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240713100412714.png)



### DSA

#### cifar10

ipc1

![image-20240716161347730](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240716161347730.png)

ipc10 

![image-20240716110218880](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240716110218880.png)

ipc50

![image-20240719174110529](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240719174110529.png)

#### cifar100

ipc1

![image-20240716230443044](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240716230443044.png)

ipc10

![image-20240814114314345](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240814114314345.png)

ipc50

![image-20240814114126212](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240814114126212.png)

## BN迁移实验

### 用预训练中最好的模型model_best

#### ConvNet

cifar10 ipc10 bn+param eval=Conve

![image-20240719194713248](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240719194713248.png)

cifar10 ipc10 bn r-bn=0.1 eval=Conve

![image-20240719221254059](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240719221254059.png)cifar10 ipc10 bn r-bn=1 eval=Conve

![image-20240720120920851](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240720120920851.png)

cifar10 ipc10 pa lr=1e3 eval=Conve

![image-20240720153357950](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240720153357950.png)

![image-20240721231218618](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240721231218618.png)

cifar10 ipc10 bn lr=1e3 eval=ConveBN

![image-20240723115856487](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240723115856487.png)

cifar10 ipc10 pa lr=1e3 eval=ConveBN ==最高的0.54==

![image-20240723172232089](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240723172232089.png)

#### ResNet

pre-training acc>0.86

![image-20240722115538636](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240722115538636.png)

cifar10 ipc10 pa lr=1e3 eval=resnet18BN 未跑完，准确率逐渐降低，最高0.44

![image-20240723180017868](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240723180017868.png)

cifar10 ipc10 bn lr=1e3 eval=resnet18BN

### 用target_param重新构建模型，跑一个epoch得到BN参数

cifar10 ipc10 lr=1e3 eval=ConveBN

# 基于生成的数据集蒸馏

## 生成模型

### GAN

### VAE

auto encoder没有对特征的隐空间分布做约束，导致没有办法采集新的隐空间向量生成新的图像

给定原始图像Xi，寻找p(z|xi)

### DDPM

### latent space

潜空间就是为了降维，去除冗余的信息

https://zhuanlan.zhihu.com/p/666649803

### stable diffusion

总体框架

![img](https://pic1.zhimg.com/80/v2-f7cb90f184781343a24cf5f377e8575e_720w.webp)

![img](https://i-blog.csdnimg.cn/blog_migrate/ced5d6f779aad44056161d49510d897e.png#pic_center)

Stable Diffusion 本身并不是一个模型，而是一个由多个模块和模型组成的系统架构，它由三大核心部件组成，每个组件都是一个神经网络系统，也称为三大基础模型：

**1. CLIPText 用于文本编码，使文本数字化：**

如何使CLIPText 文本编码器 与 U-Net 噪声预测器的校正

![img](https://pic4.zhimg.com/80/v2-99abcfcbc978418a4466346de90a530f_720w.webp)噪点强度级别noise amount采用时间步长timestamp来表示

![img](https://pic2.zhimg.com/80/v2-0c319d215d118f843d8d144775b1f2cd_720w.webp)

**2. U-Net + Scheduler 用于逐步处理/扩散被转化到潜空间中的信息：**

![img](https://pica.zhimg.com/80/v2-255544696ee2f2301f09d22266bddc20_720w.webp)

**3. AutoEncoder Decoder （主要是一个VAE：Variational AutoEncoder ）使用处理后的信息矩阵解码绘制出最终图像，把潜空间的运算结果解码成实际图片维度**

![img](https://picx.zhimg.com/80/v2-b9c25e476ba116256e226464bad7422b_720w.webp)

### Stable Diffusion源码分析

## D4M

代码分析：

- 训练阶段

每个类有一个kmeans model

每个kmeans model里面的图像latent数量达到args.km_expand * args.ipc时更新model,然后重置latents[prompt] = []

- 匹配阶段

采用FKD技术给每个合成图像打上软标签->通过教师模型

在学生模型上进行训练并验证



# 多模态数据蒸馏

## Vison-Language任务定义

- 图像说明VC：图像说明的目标是为给定的图像生成“标题”，即用一句话总结图像内容。标题通常包含感兴趣的对象、对象的行为以及对象之间的位置关系。
- 视觉问答VQA：图像说明的目标是为给定的图像生成“标题”，即用一句话总结图像内容。标题通常包含感兴趣的对象、对象的行为以及对象之间的位置关系。
- 图文检索ITM：给定一个特定模态 (视觉或语言) 的query ，它的目标是从另一个模态中找到语义上最接近的目标。根据query和目标模式，它包含两个子任务: 图像-文本检索和文本-图像检索。
- 文本生成图像
- 视觉对话VD

评估指标

## VISION-LANGUAGE DATASET DISTILLATION

#### 之前方法的缺点

只适用于图像

视觉语言数据集蒸馏难点：1.视觉语言数据集不包含离散的类，多模态数据集的方差较大	2.高分辨率图像的计算复杂性3.文本数据不可微，没法基于梯度优化

![image-20240628105832170](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240628105832170.png)

#### 方法

![image-20240627150041565](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240627150041565.png)

- 问题设定：输入数据集为图文对，一张图可以对应多个文本， 本实验中每张图片配有5个说明文字，根据来自另一个模态的查询，使用来自一个模态的余弦距离来检索最接近的匹配。
- 目标：Let’s consider what we hope that this model learns: **we want it to learn “similar representations (vectors)” for a given image and the caption describing it. Meaning that either we give it an image or the text describing it, we want it to produce same 256 sized vectors for both.**让我们考虑一下我们希望这个模型学习什么：**我们希望它学习给定图像和描述它的标题的“类似表示（向量）”。这意味着我们给它一个图像或描述它的文本，我们希望它为两者生成相同的 256 大小的向量**因此，在最佳情况下，text_embeddings 和 image_embedding 矩阵应该相同，相似度矩阵应该是单位矩阵![image-20240628113618807](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240628113618807.png)
- 双向对比损失函数训练多模态模型（bidirectional contrastive loss）

![image-20240628120636373](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240628120636373.png)

![image-20240628120721465](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240628120721465.png)

- 基于双轨迹匹配损失更新合成集（ bi-trajectory matching loss）

![image-20240628121211613](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240628121211613.png)

- 低秩自适应匹配(LORA)

  将VIT模型中每一层的权重矩阵W拆分成两个低秩矩阵的乘积W‘=W+AB

![image-20240627150718620](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240627150718620.png)

#### 代码

`buffer.py#训练并存储专家轨迹`

`distill.py#合成数据集`

`networks.py#网络架构，包含对比损失`

#### 效果

评估指标：采用图文检索的评估指标（召回率Recall@k)，即检索系统返回的前 K 个结果中包含正确图像的概率。举例来说,如果一个系统在100次查询中,有90次的前5个结果中包含了正确的图像,那么它的Recall@5指标就是90%。



![img](https://picx.zhimg.com/v2-36f1d92a9561bfc2d6ed79aaeb6ccca3_720w.jpg?source=d16d100b)

![img](https://picx.zhimg.com/v2-80256f4362de18b1535da167f6b68c86_720w.jpg?source=d16d100b)

![image-20240628164020857](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240628164020857.png)

左：蒸馏前的图像和文本对。右图：经过 2000 次蒸馏步骤后的图像和文本对。请注意，此处可视化的文本是训练集中与精炼文本嵌入相对应的最近句子解码。

#### 问题

为什么图像encoder是trainable的，文本encoder是frozened

如何更新合成集

## Low-Rank Similarity Mining for Multimodal Dataset Distillation

#### 之前方法的缺点

传统ITC模型(image-text-contrasive)只有配对的图像和文本是正样本，其余都是负样本，目标是优化相似矩阵为单位矩阵


我们强调学习模态的对应关系，而不是总结每个类别的数据模式。假设数据xi, yj的每个组合是一对，但具有不同的GT相似度sij，将N对图像文本对扩展到N的平方对图像文本对

![image-20240628165500741](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240628165500741.png)

#### 方法

1. 在上一篇基础上同时学习图像-文本相似矩阵作为辅助信息，只需修改对比损失函数以拟合相似矩阵

- eNCE:同时考虑正负样本

  ![image-20240629011605335](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240629011605335.png)

- BCE:多标签二元交叉熵

![image-20240629011713288](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240629011713288.png)

![img](https://pic1.zhimg.com/80/v2-c62cc6aeca6faf5af01f144dba2e8760_720w.webp)

- 加权二元交叉熵：BCE loss对负样本有很大的惩罚，提出加权版本平衡正负样本的loss

==使合成数据集和真实数据集的相似矩阵尽可能相同==         

![image-20240629012154457](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240629012154457.png)

2. 对相似矩阵分解为可学习的低秩矩阵乘积

![image-20240629014602863](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240629014602863.png)

3. LoRS算法，使用了TESLA技术来减少内存损耗

![image-20240629014803597](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240629014803597.png)

![image-20240629014857935](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240629014857935.png)

4. 使用LoRS合成数据集来训练网络

![image-20240629015239494](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240629015239494.png)

5. 证明相似度挖掘的合理性

#### 效果

减少合成对的数量以减少合成参数，和其他方法公平比较

![image-20240629020214489](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240629020214489.png)

![image-20240629020650712](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240629020650712.png)

![image-20240629020229897](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240629020229897.png)

对假阴性的识别效果好

![image-20240629021234564](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240629021234564.png)



#### 改进和研究方向

#### 问题

train真实数据集需要wBCEloss吗

之前方法有用到相似矩阵吗，不是也在优化相似矩阵吗

S初始化成I不已经是优化的目标了吗

## 代码问题汇总

1. text-encoder冻结

2. 为什么要排开自身

![image-20240628120636373](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240628120636373.png)

3. 最后合成的文本从何而来

4. 分母是什么

![image-20240629012154457](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240629012154457.png)

# 大模型训练流程

## 专有名词

- token:词元，词表的元素，可以理解为单词或者词组，数字，标点符号，尽可能用少量的token覆盖整体的单词表。模型一次最多能处理的文本长度叫做token长度
- prompt:用于引导模型生成文本的输入文本，可以是一个问题，一个主题，一个描述等，可以帮助模型理解用户的意图

### 归一化技术比较

BatchNorm、LayerNorm 和 GroupNorm ,InstanceNorm都是深度学习中常用的归一化方式。它们通过将输入归一化到均值为 0 和方差为 1 的分布中，来防止梯度消失和爆炸，并提高模型的泛化能力

#### BatchNorm

像sigmod和tanh在输入值比较大的时候，会很容易进入梯度接近0的区域(即饱和区域)，随着网络层数的增多，容易产生梯度消失的问题。而前面提到的内部协变量偏移会导致网络中每一层输入的分布产生频繁的变化，这也就导致激活层的输入进入饱和区的概率变高。

![img](https://img-blog.csdnimg.cn/d1d0fd6b73304258899a4d113c03dda1.png)

#### LayerNorm

#### InstanceNorm

![image-20240628215754598](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240628215754598.png)

大体都包括三个训练步骤：预训练，有监督微调，对齐

- 预训练阶段：模型会学习海量无标注数据集的知识
- 然后用有监督微调的方式来细化模型，以便在后期推理过程中更好地遵守特定指令
- 最后使用对齐技术使大模型更好地相应用户的提示Prompt

## 预训练

#### 为什么需要预训练

很多深度学习项目没有大数据支持，只有小数据集，直接在小数据集上训练模型会导致模型的精度不高

而很多大型数据集训练模型的浅层部分对于下游任务是可以通用的，可以采用冻结或微调的方式

#### 预训练是什么

通过一个已经训练好的模型A，去完成一个小数据量的任务B(使用了A的浅层参数)

==任务A和任务B极其相似,但不一定完全相同==

## 微调

#### 为什么需要微调

随着模型的增长训练过程必须调整更多的参数，对算力的要求更大，微调允许少量地重新调整预训练大模型的权重参数，降低重新进行训练的成本

#### 微调的分类

- 按参数规模划分
  - 全参数微调FPFT：用预训练模型作为初始化权重，在特定数据集上继续训练，全部参数都更新
  - 参数高效微调PEFT：只更新一部分参数，或者通过对参数进行某种结构化约束，例如稀疏化或者低秩近似来降低微调的参数量
- 按训练流程划分
  - 提示微调Prompt Tuning
  - 有监督微调STF
  - 人类反馈强化学习RLHF

## Transform之前语言模型

#### 神经网络语言模型

- 词（填空）2.比较

##### 词向量（word_embedding)

用一个向量来表示一个单词

给任何一个词，可以根据词的独热编码W1,训练好的降维Q矩阵，W1*Q=C1，C1就是这个词的低维词向量

##### word2vec模型

最初的神经网络语言模型是NNLM,重点是预测下一个词，架构是双层感知机

CBOW和Skip-gram两种架构的重点是得到一个Q矩阵

- CBOW：给出一个词的上下文，得到这个词
- Skip-gram：给出一个词，得到这个词的上下文

![img](https://pic3.zhimg.com/v2-ded34fa32e7e10eb5c5f24b6c4048d3e_r.jpg)

word2vec模型的缺点：无法解决多义词的问题，比如苹果可能代表水果或者手机，但对应的词向量是同一个

##### 预训练语言模型的下游任务改造

以NLP的问答任务为例，给一个问题X,给出一个回答Y

给出一句话，先试用独热编码，再使用word2vec预训练好的Q矩阵直接得到词向量，然后进行接下来的任务：

1. 冻结：可以不改变Q矩阵
2. 微调：随着任务的改变，改变Q矩阵

##### ELMo模型（专门做词向量，通过预训练）

#### 注意力机制

注意力相当于是一个权重

对于模型而言，很难决定什么数据重要，什么数据不重要

“Query，Key，Value的概念取自于信息检索系统，举个简单的搜索的例子来说。当你在某电商平台搜索某件商品（年轻女士冬季穿的红色薄款羽绒服）时，你在搜索引擎上输入的内容便是Query。
然后搜索引擎根据Query为你匹配Key（例如商品的种类，颜色，描述等）。
然后根据Query和Key的相似度得到匹配的内容（Value)。”

![image-20240705190407685](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240705190407685.png)

eg:判断婴儿在干嘛这句话与图中哪些位置更有关系

图像中：Q:查询对象（比如婴儿）K:被查询对象的某个部分 V:被查询对象

![image-20240705192555208](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240705192555208.png)

![image-20240705193430766](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240705193430766.png)

![image-20240705200731953](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240705200731953.png)

#### 自注意力self-attention

Q,K,V同源，来自于同一个X乘以不同的参数矩阵WQ,WK,WV，这三个参数矩阵是可训练的

##### 为什么需要self-attention

首先，看下面这两个短句：

- **句子I：**The bank of the river.
- **句子II：**Money in the bank.

在翻译成中文的过程中，机器算法是如何知道“句子I”中的“bank”指的是自然环境中的“岸边”，而“句子II”中的“bank”指的是金融体系中的“银行”呢？

![img](https://pic4.zhimg.com/80/v2-e0579734be22188e1c8e6ba4e2419ca7_720w.webp)

由于 “bank” 是一个多义词，所以它在 Embedding 空间中的定位本来是有多个“分身”，我们取其中的两个分身，即“bank1”和“bank2”。那么，我们需要做的就是定位清晰“bank1”和“bank2”这两个单词在空间中到底各自离“river”和“money”的哪个单词更近一些。在图中很明显，“bank1”离“river”更近，而“bank2”离“money”更近，于是这两句话就变成了：

- **变形后的句子I：**The bank1 of the river.
- **变形后的句子II：**Money in the bank2.

如之前所说，虽然此时机器算法压根也不知道“river”和“money”到底是何物，但它知道在Embedding 空间中，“river”周边有很多和大自然有关的词汇，比如“water”、“tree”、“fish”等等。而“money”周边有许多与金融有关的词汇，比如“currency”, “cash”, “withdraw”等等。于是，机器算法知道了“bank1”代表的是与“river”有关的一个单词，与他们比较近的单词还有“water”、“tree”、“fish”等等。而“bank2”代表的是与“money”有关的一个单词，与他们比较接近的单词还有“currency”, “cash”, “withdraw”等等。这就是***“Attention 注意力机制”的工作原理，也就是 Attention 让一个单词在句子中找到与它产生强语义联系的其他单词，并组成一个新的“变体单词”\***：“bank1”、“bank2”。

##### 在语言模型的运用

- 三种任务的区别：
  1. 词性标注：让机器自动决定每一个词是什么词性，每一个输入的vector都有一个对应的输出lablel
  2. 情感分析：判断某段评价是正面的还是负面的,输入是一个sequence，输出是一个label
  3. seq2seq如机器翻译：模型自己决定输出多少个label,输入是N个vector，输出是M个label

- 任务一的解法

目标：输入a1,a2,a3,a4；输出b1,b12,b3,b4；b1的生成要考虑a1,a2,a3,a4，其他同理

![image-20240711122024691](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711122024691.png)

1. step1:根据a1vector找出整个sequence里面跟a1相关的vector

如何计算关联性α

![image-20240711120437864](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711120437864.png)

2. step2:计算相互两个vector的attention score,==不能忽略自己跟自己计算关联性==

![image-20240711121259662](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711121259662.png)

3. step3:计算vector b

![image-20240711121707822](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711121707822.png)

4. 矩阵形式

![image-20240711134642471](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711134642471.png)

![image-20240711134949882](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711134949882.png)

![image-20240711135220869](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711135220869.png)

![image-20240711135347132](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711135347132.png)

##### 在图像模型的运用

- 图像也可以看作是一个vector set

- Self-attention可以看作是复杂的CNN

##### 位置编码

self-attention对于每个词都是无位置关系的，把每个词的位置关系打乱，得到的注意力值不变

通过position encoding知道每个词的位置关系,给每个向量添加一个位置向量positional vector **ei**

![image-20240705204137773](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240705204137773.png)

![image-20240705203748810](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240705203748810.png)

最初的position encoding的合理性解释

![](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240705203926938.png)

#### self-attention和RNN,LSTM的区别

![image-20240711135506404](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711135506404.png)

X2通过X1间接使用到X0的信息，但是X0的信息在逐个传播的过程中会越来越少，因此RNN无法解决长序列问题，也无法做并行

![image-20240711135650583](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711135650583.png)

LSTM通过各种门如遗忘门选择性记忆之前的信息

#### 多头自注意力muti-head-self-attention

- 为什么要做改进:相关性可能不止一个，需要多个q,多个不同的相关性。例如==（我想要个苹果，这个“苹果”指的是“fruit”还是“iphone”，这就需要两层QKV识别，也就是多头mutil-head）==

将qi分成qi,1;qi,2....,计算的bi,1;bi,2组合起来，得到的bi‘相比于之前单头的bi

![image-20240711141307063](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711141307063.png)

#### 掩码自注意力Masked self-attention

- 为什么要做改进：生成任务生成的单词是一个个生成的，不是全部vector一起输入

![image-20240711160639749](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711160639749.png)

## Transformer

本质:seq2seq model

应用：

- 语音辨识：输入声音信号，输出文字
- 机器翻译
- 语音翻译：输入某些方言，输出可理解的文字
- 语音合成：输入文字，输出某些方言语音
- 聊天机器人
- 文法剖析
- 多标签分类
- 目标检测

### Encoder

NX代表包含N个编码器线性堆叠，通过多个编码器，对词向量一步步增强

![image-20240711155610535](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711155610535.png)

encoder的具体实现

- self-attention

- 残差网络，防止梯度消失等
- LayerNorm，防止梯度爆炸等
- feed forward，非线性变换

![image-20240711155545309](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711155545309.png)

### Decoder

#### Autoregressive（AT）

要准备**开始符号**和**结束符号**

![image-20240711160216909](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711160216909.png)

上一个时刻的输出作为下一刻时刻的输入

![image-20240711160436123](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711160436123.png)

#### non Autoaggresive model(NAT)

输入是多个begin token，输出是对应长度word vector

![image-20240711161522075](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711161522075.png)

### Encoder与Decoder的传递

- cross attention

![image-20240711161649576](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711161649576.png)

![image-20240711161839836](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711161839836.png)

- 思考：有多层encoder和decoder,decoder不管哪层都是拿encoder最后一层的输出来做cross attention，能否有其他的连接方式

### 训练

以语音辨识为例，输入begin token之后希望接下来输出的单词概率中”机“的概率最大，和GT做cross entropy损失，希望所有输出单词和GT的交叉熵损失总和最小，还要包括end token

注意，Decoder在训练时的输入还包括Ground Truth

![image-20240711163113019](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711163113019.png)

#### Tips

- Copy Mechanism:没有必要自己创造，而是从输入复制一些东西到输出，如聊天机器人，摘要撰写

- Guided Attention:要求模型计算attention时需要有先后顺序，比如语音合成

- Beam Search：seq2seq本身是贪心算法，但不一定是最优算法，但在需要有创造性的task如故事续写方面表现不好

- optimize evalution metric

- scheduled sampling:推理的时候decoder输入是上一个word的预测输出，训练的时候decoder输入是ground truth,造成一个dismatch

  

## 多模态

### 图文对比学习

https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247538505&idx=3&sn=b8257395c4b50e47f01d78696172cf85&chksm=ebb76f9ddcc0e68b24e08aba4ed59a16e8ce95ca289fc9cb74a5e92124db16bfc249d868a22e&scene=27

那对比学习采用的具体思想是什么呢？顾名思义，即将样例与与它语义相似的例子（正样例）和与它语义不相似的例子（负样例）进行对比，希望通过设计模型结构和对比损失，使语义相近的例子对应的表示在表示空间更接近，语义不相近的例子对应的表示距离更远，以达到类似聚类的效果

目标：使模型通过识别正样本和负样本学会判别哪些图文是配对的

- 正负样例：对于有监督的数据，正负样例很容易构造，同一标签下的例子互为正样例，不同标签下的例子互为负样例，但对于无标签的数据，目前的主流做法是对所有样例增加扰动，产生一些新的样例，同一个样例扰动产生的所有样例之间互为正样例，不同样例扰动产生的样例彼此之间互为负样例。

![image-20240628212959379](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240628212959379.png)

![image-20240628213233473](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240628213233473.png)

- 对比损失：在有了正负例之后，我们需要给模型信号，激励他们拉近正样例间的距离，拉远负样例间的距离，这就要通过设计对比损失来完成。

![image-20240628213525568](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240628213525568.png)

相似样本会使loss变小，不相似样本会使loss变大

![image-20240628234847372](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240628234847372.png)

### CLIP

目标：使模型学会识别图片中描述的东西

- 如何进行训练

优化目标是相似矩阵对角线上得分值越高，非对角线上得分值越低

![image-20240628235910609](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240628235910609.png)

- 如何进行推理

推理需要什么：一些提示文本(prompt),一个待预测的图片，一个训练好的模型，根据每种提示跟图片特征向量算相似度，找到概率最高的

![image-20240628235929153](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240628235929153.png)

### BLIP

# 知识蒸馏

## A Fast Knowledge Distillation Framework for Visual Recognition

![请添加图片描述](https://img-blog.csdnimg.cn/59a36c79306d45b29a45c55eb5a1ee21.png)

## Re-labeling ImageNet: from Single to Multi-Labels, from Global to Localized Label

![img](https://img-blog.csdnimg.cn/d372d86c831641b19a27c9baca89449d.png)

## Knowledge Distillation via Instance Relationship Graph

#### 方法

设计实例关系图IRG，用于保存图像的特征和不同图像特征之间的关系，网络每层都有一个IRG

![image-20240907155925805](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240907155925805.png)

IRG的变换IRG-t，图像从L1层传递到L2层时其对应的特征图和邻接矩阵都会发生变化

![image-20240907155936295](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240907155936295.png)

设计了三种loss

- IRGloss:
- IRG-tloss：变换的损失
- MTKloss

#### IRG损失（$L_{IRG}$）

IRG损失定义为教师网络和学生网络的IRG之间的差异。具体计算方法如下：

1. **定义IRG**：
   
   - 设$IRG^T_L$为教师网络在第$L$层的IRG，$IRG^S_{l_M}$为学生网络在第$l_M$层的IRG。
   
2. **损失计算** ：
   - IRG损失的公式为：
   $$
   L_{IRG}(x) = Dist(IRG^T_L, IRG^S_{l_M}) = \lambda_1 \cdot Dist(V^T_L, V^S_{l_M}) + \lambda_2 \cdot Dist(E^T_L, E^S_{l_M})
   $$
   - 其中，$Dist(V^T_L, V^S_{l_M})$计算顶点（实例特征）之间的欧几里得距离，$Dist(E^T_L, E^S_{l_M})$计算边（实例关系）之间的欧几里得距离。
   - 具体展开为：
   $$
   L_{IRG}(x) = \lambda_1 \cdot \sum_{i=1}^{I} ||f^T_L(x_i) - f^S_{l_M}(x_i)||_2^2 + \lambda_2 \cdot ||A^T_L - A^S_{l_M}||_2^2
   $$
   - 这里，$\lambda_1$和$\lambda_2$是平衡两个损失项的惩罚系数。

#### IRG变换损失（$L_{IRG-t}$）

IRG变换损失表示实例特征空间的变换，包含顶点变换和边变换。计算方法如下：

1. **定义IRG变换** ：
   - 设$IRG-t_{l_1 l_2}$为从第$l_1$层到第$l_2$层的IRG变换。

2. **损失计算** ：
   
   - IRG变换损失的公式为：
   $$
   L_{IRG-t}(x) = Dist(IRG-t^T_{l_1 l_2}, IRG-t^S_{l_3 l_4}) = Dist(Trans(V^T_{l_1}, V^T_{l_2}), Trans(V^S_{l_3}, V^S_{l_4}))
   $$
   - 其中，$\Lambda^T_{l_1 l_2}$和$\Theta^T_{l_1 l_2}$分别表示教师的顶点和边变换，$\Lambda^S_{l_3 l_4}$和$\Theta^S_{l_3 l_4}$表示学生的顶点和边变换。
   - 最终的损失函数为：
   
   $$
   L_{IRG-t}(x) = ||\Lambda^T_{l_1 l_2} - \Lambda^S_{l_3 l_4}||_2^2
   $$

#### 多类型知识损失（$L_{MTK}$）

多类型知识损失结合了实例特征损失、IRG损失和IRG变换损失，具体计算方法如下：

1. **损失计算** ：
   
   - 多类型知识损失的公式为：
   $$
   L_{MTK}(x) = L_{GT}(x) + L_{IRG}(x) + \lambda_3 \cdot L_{IRG-t}(x)
   $$
   - 结合之前的损失项，最终的损失函数为：
   
   $$
   L_{MTK}(x) = L_{GT}(x) + \lambda_1 \cdot L_{logits}(x) + \lambda_2 \cdot \sum_{l_M \in L_M} ||A^T_L - A^S_{l_M}||_2^2 + \lambda_3 \cdot \sum_{l_1 l_2 l_3 l_4 \in L_{Tran}} ||\Lambda^T_{l_1 l_2} - \Lambda^S_{l_3 l_4}||_2^2
   $$

通过以上三种损失的设计，学生网络能够有效地从教师网络中提取和学习不同类型的知识。

#### 效果

![image-20240907161459395](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240907161459395.png)

#### 思考

如何将其运用到数据蒸馏上



## 硬标签和软标签

![img](https://img-blog.csdnimg.cn/ba820aa63c8f4d3381b4fcf569897429.png)

硬标签：使用会比较多一点，用于非是即非的任务上，例如是猫就是猫，是狗就是狗；

软标签：

1. 用于模棱两可的情况；

2. 用于蒸馏，例如，计算teacher模型的参数于student模型参数的loss，用硬标签会过于绝对，不利于student模型更好地学习teacher参数的分布和teacher的决策行为，当然也可以软硬标签都用上，不过在大规模无监督的蒸馏背景下，用soft label更为何时。

https://mp.weixin.qq.com/s/P6LVeSlVGXNnCvQbwrFQbQ

soft label难以获取，可以通过标签正则的方式构造

## 交叉熵损失函数

# 扩散模型

模拟液体扩散过程，通过逐步denoise还原图像

## 原理部分

#### 高斯噪声

服从标准正态分布的随机采样

#### 前向加噪

重参数化将两个标准正态分布的联合转化为一个另一个正态分布，然后用递推式计算最终表达式

![image-20240708143032391](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240708143032391.png)

![image-20240708143819599](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240708143819599.png)

#### 反向过程

最后的XT相当于一个标准正态分布

![image-20240708160023630](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240708160023630.png)

![image-20240708160150311](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240708160150311.png)

## 神经网络部分

### 从image Generator到Text-to-image Generator

#### image Generator

训练noise predicter

#### ![image-20240708150602933](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240708150602933.png)

#### Text-to-image Generator

训练noise predicter的时候多加入文本信息

![image-20240708151202023](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240708151202023.png)

#### 训练的loss

https://zhuanlan.zhihu.com/p/607293445

构造一个分布，让他去拟合后验分布$q(x_t-1|x_t,x_0$)，采用KL作为损失函数，拟合两个分布，让模型去预测在前向时间步 `t` 所添加的高斯噪声 `ε`，模型输入是 `xt` 和 时间步 `t`

![image-20240708151457050](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240708151457050.png)



#### 生成网络框架

![image-20240708151833557](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240708151833557.png)

##### text-encoder

训练

##### generation model

generation model的中间产物：小图、latent representation

==latent representation如何产生：Auto-encoder==

generation model需要训练的部分：noise predicter

- input:添加噪声的latent-representation+text-embedding+step
- gt:添加的噪声

![image-20240708153514142](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240708153514142.png)

##### Decoder

不需要文本资料就可以训练

##### 生图过程



![image-20240708154025117](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240708154025117.png)

#### 评价指标

- FID 
- CLIP score

# Zero-shot learing

目标：希望模型具有推理能力，希望我们的模型能够对其从没见过的类别进行分类，让机器具有推理能力

从见过的类别（第一列）中提取特征（如：外形像马、条纹、黑白），然后根据对未知类别特征的描述，预测训练集未见过的类别。

![img](https://pic3.zhimg.com/v2-d8efa9870a3ce5ee028277ec57033036_b.webp?consumer=ZHI_MENG)

![image-20240711203916984](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711203916984.png)

![image-20240711204132065](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711204132065.png)

![image-20240711204341115](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711204341115.png)

![image-20240711204450095](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711204450095.png)

![image-20240711204646103](C:\Users\ASUS-PC\AppData\Roaming\Typora\typora-user-images\image-20240711204646103.png)

