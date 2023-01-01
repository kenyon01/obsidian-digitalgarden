> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [mp.weixin.qq.com](https://mp.weixin.qq.com/s/78EP4HOzb8yoBef6j6uM9g)

“他山之石，可以攻玉”，站在巨人的肩膀才能看得更高，走得更远。在科研的道路上，更需借助东风才能更快前行。为此，我们特别搜集整理了一些实用的代码链接，数据集，软件，编程技巧等，开辟 “他山之石” 专栏，助你乘风破浪，一路奋勇向前，敬请关注。

来源：CSDN—mingo_敏

地址：https://blog.csdn.net/shanglianlm/article/details/85019768

**01**  

**基本用法**

```
criterion = LossCriterion() #构造函数有自己的参数
loss = criterion(x, y) #调用标准时也有参数
```

**02**  

**损失函数**

**2-1 L1 范数损失 L1Loss**

计算 output 和 target 之差的绝对值。

```
torch.nn.L1Loss(reduction='mean')
```

参数：

reduction - 三个值，none: 不使用约简；mean: 返回 loss 和的平均值；sum: 返回 loss 的和。默认：mean。

**2-2 均方误差损失 MSELoss**

计算 output 和 target 之差的均方差。

```
torch.nn.MSELoss(reduction='mean')
```

参数：

reduction - 三个值，none: 不使用约简；mean: 返回 loss 和的平均值；sum: 返回 loss 的和。默认：mean。

**2-3 交叉熵损失 CrossEntropyLoss**

当训练有 C 个类别的分类问题时很有效. 可选参数 weight 必须是一个 1 维 Tensor, 权重将被分配给各个类别. 对于不平衡的训练集非常有效。

在多分类任务中，经常采用 softmax 激活函数 + 交叉熵损失函数，因为交叉熵描述了两个概率分布的差异，然而神经网络输出的是向量，并不是概率分布的形式。所以需要 softmax 激活函数将一个向量进行 “归一化” 成概率分布的形式，再采用交叉熵损失函数计算 loss。

![](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjOjyYsAGibaR4cQ2LERVN4nyzZfpic8mJANLibLAR68qw2laIRKKDPY9D1BVOl3fXadSYgbSf0h6Z66w/640?wx_fmt=png)

```
torch.nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean')
```

参数：

weight (Tensor, optional) – 自定义的每个类别的权重. 必须是一个长度为 C 的 Tensor

ignore_index (int, optional) – 设置一个目标值, 该目标值会被忽略, 从而不会影响到 输入的梯度。

reduction - 三个值，none: 不使用约简；mean: 返回 loss 和的平均值；sum: 返回 loss 的和。默认：mean。

**2-4 KL 散度损失 KLDivLoss**

计算 input 和 target 之间的 KL 散度。KL 散度可用于衡量不同的连续分布之间的距离, 在连续的输出分布的空间上 (离散采样) 上进行直接回归时很有效.

```
torch.nn.KLDivLoss(reduction='mean')
```

参数：

reduction - 三个值，none: 不使用约简；mean: 返回 loss 和的平均值；sum: 返回 loss 的和。默认：mean。

**2-5 二进制交叉熵损失 BCELoss**

二分类任务时的交叉熵计算函数。用于测量重构的误差, 例如自动编码机. 注意目标的值 t[i] 的范围为 0 到 1 之间.

```
torch.nn.BCELoss(weight=None, reduction='mean')
```

参数：

weight (Tensor, optional) – 自定义的每个 batch 元素的 loss 的权重. 必须是一个长度为 “nbatch” 的 的 Tensor

pos_weight(Tensor, optional) – 自定义的每个正样本的 loss 的权重. 必须是一个长度 为 “classes” 的 Tensor

**2-6 BCEWithLogitsLoss**

BCEWithLogitsLoss 损失函数把 Sigmoid 层集成到了 BCELoss 类中. 该版比用一个简单的 Sigmoid 层和 BCELoss 在数值上更稳定, 因为把这两个操作合并为一个层之后, 可以利用 log-sum-exp 的 技巧来实现数值稳定.

```
torch.nn.BCEWithLogitsLoss(weight=None, reduction='mean', pos_weight=None)
```

参数：

weight (Tensor, optional) – 自定义的每个 batch 元素的 loss 的权重. 必须是一个长度 为 “nbatch” 的 Tensor

pos_weight(Tensor, optional) – 自定义的每个正样本的 loss 的权重. 必须是一个长度 为 “classes” 的 Tensor

**2-7 MarginRankingLoss**

```
torch.nn.MarginRankingLoss(margin=0.0, reduction='mean')
```

对于 mini-batch(小批量) 中每个实例的损失函数如下:

![](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjOjyYsAGibaR4cQ2LERVN4nyPTI5iaicfMJq1jaiccfY9Xppytdt1jnv0fTEqaZX5PYtuOYhEdpjkb6ibA/640?wx_fmt=png)

参数：

margin: 默认值 0

**2-8 HingeEmbeddingLoss**

```
torch.nn.HingeEmbeddingLoss(margin=1.0,  reduction='mean')
```

对于 mini-batch(小批量) 中每个实例的损失函数如下:

参数：

![](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjOjyYsAGibaR4cQ2LERVN4nyWOrnqvicpNIBG5kCJ8xHCM1L1z5njT4gNWXC357KepggE2micg4rPNrg/640?wx_fmt=png)

margin: 默认值 1

**2-9 多标签分类损失 MultiLabelMarginLoss**

```
torch.nn.MultiLabelMarginLoss(reduction='mean')
```

对于 mini-batch(小批量) 中的每个样本按如下公式计算损失:

![](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjOjyYsAGibaR4cQ2LERVN4nyFQ0jhVyXKo9JGUX6VTndicSNPKSzYribhoGiagnLt6w9hIuJyticiaNcrnA/640?wx_fmt=png)

**2-10 平滑版 L1 损失 SmoothL1Loss**

也被称为 Huber 损失函数。

```
torch.nn.SmoothL1Loss(reduction='mean')
```

![](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjOjyYsAGibaR4cQ2LERVN4nyAmHYbF4R03XNelOeTVv5DkS2icroSFvuGwTs66UnHS3mzWuGK9nC4kg/640?wx_fmt=png)

其中

![](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjOjyYsAGibaR4cQ2LERVN4nymCuY20ibL7KyI2gibZbky2tMgaAoErtMNldyqmqrCmNibp6uNPw139eicQ/640?wx_fmt=png)

**2-11 2 分类的 logistic 损失 SoftMarginLoss**

```
torch.nn.SoftMarginLoss(reduction='mean')
```

![](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjOjyYsAGibaR4cQ2LERVN4nyACz4RNpN8sPBrq0fCjc8PRNrL2594FAsJeTzz8OyUqbnIlN5CjBib6w/640?wx_fmt=png)

**2-12 多标签 one-versus-all 损失 MultiLabelSoftMarginLoss**

```
torch.nn.MultiLabelSoftMarginLoss(weight=None, reduction='mean')
```

**![](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjOjyYsAGibaR4cQ2LERVN4nykGUjsW4NX4FO80ugfL9ouAb8LH8qIz4LkcBFLDk2sAVg4Wlf8ViaqrA/640?wx_fmt=png)**

**2-13 cosine 损失 CosineEmbeddingLoss**

```
torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
```

![](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjOjyYsAGibaR4cQ2LERVN4nyAicbCMXT9yicJYUia0H1AdSxMMMboevLDFIzYZFOM3h20rs2Pg5FP6Edg/640?wx_fmt=png)

参数：

margin: 默认值 0

**2-14 多类别分类的 hinge 损失 MultiMarginLoss**

```
torch.nn.MultiMarginLoss(p=1, margin=1.0, weight=None,  reduction='mean')
```

![](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjOjyYsAGibaR4cQ2LERVN4nyzEmyRAok8eaKic7D860apvPCSrnTTjiavficyG9DwBYfNiaYIhuVRVaHQw/640?wx_fmt=png)

参数：

p=1 或者 2 默认值：1

margin: 默认值 1

**2-15 三元组损失 TripletMarginLoss**

```
torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, reduction='mean')
```

![](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjOjyYsAGibaR4cQ2LERVN4nyG4udbPsFn2nVunkOFD0k43NdSOlAiael5Sz0sb4FgozLUfUfg14giaGw/640?wx_fmt=png)

其中：

![](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjOjyYsAGibaR4cQ2LERVN4nybl1iaedlvZSF9wwwD4HMxUAxXhPs2BbqTYXaaekwMWpFerv5xfia0w3Q/640?wx_fmt=png)

**2-16 连接时序分类损失 CTCLoss**

CTC 连接时序分类损失，可以对没有对齐的数据进行自动对齐，主要用在没有事先对齐的序列化数据训练上。比如语音识别、ocr 识别等等。

```
torch.nn.CTCLoss(blank=0, reduction='mean')
```

参数：

reduction - 三个值，none: 不使用约简；mean: 返回 loss 和的平均值；sum: 返回 loss 的和。默认：mean。

**2-17 负对数似然损失 NLLLoss**

负对数似然损失. 用于训练 C 个类别的分类问题.

```
torch.nn.NLLLoss(weight=None, ignore_index=-100,  reduction='mean')
```

参数：

weight (Tensor, optional) – 自定义的每个类别的权重. 必须是一个长度为 C 的 Tensor

ignore_index (int, optional) – 设置一个目标值, 该目标值会被忽略, 从而不会影响到 输入的梯度.

**2-18 NLLLoss2d**

对于图片输入的负对数似然损失. 它计算每个像素的负对数似然损失.

```
torch.nn.NLLLoss2d(weight=None, ignore_index=-100, reduction='mean')
```

参数：

weight (Tensor, optional) – 自定义的每个类别的权重. 必须是一个长度为 C 的 Tensor

reduction - 三个值，none: 不使用约简；mean: 返回 loss 和的平均值；sum: 返回 loss 的和。默认：mean。

**2-19 PoissonNLLLoss**

目标值为泊松分布的负对数似然损失

```
torch.nn.PoissonNLLLoss(log_input=True, full=False,  eps=1e-08,  reduction='mean')
```

参数：

log_input (bool, optional) – 如果设置为 True , loss 将会按照公 式 exp(input) - target * input 来计算, 如果设置为 False , loss 将会按照 input - target * log(input+eps) 计算.

full (bool, optional) – 是否计算全部的 loss, i. e. 加上 Stirling 近似项 target * log(target) - target + 0.5 * log(2 * pi * target).

eps (float, optional) – 默认值: 1e-8

参考资料

http://www.voidcn.com/article/p-rtzqgqkz-bpg.html  

![](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjPGpqqn4kanBxtfkqtUGbhVU2aTMdKRhiaztfnL4rYwDicAo4LDSuXvxd9VDicsDicYH9BOEWg8P8HwqA/640?wx_fmt=png)

[![](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjP2Tzsolk9cIZLw4GL8M2Gico1aJ9fZub1SxvuyDL5nlE7zL8k2LbNFosSl7ajTvsGuV6ibaic790ZJg/640?wx_fmt=png)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzIzNjc0MTMwMA==&action=getalbum&album_id=2193669496408735748&scene=21#wechat_redirect)

[![](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjM65DtFPKbNv2mVV5UT6RU9vicN1Mn9yNicRgLibJmSrktL28r1icHqvZ2JYtRqJXfDzpQm9AzuWOng4w/640?wx_fmt=png)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzIzNjc0MTMwMA==&action=getalbum&album_id=2228408918878289922#wechat_redirect)

本文目的在于学术交流，并不代表本公众号赞同其观点或对其内容真实性负责，版权归原作者所有，如有侵权请告知删除。  

****“**他山之****石****”**** **历史**文章**

  

*   [浅谈 PyTorch 中的 tensor 及使用](http://mp.weixin.qq.com/s?__biz=MzIzNjc0MTMwMA==&mid=2247578686&idx=2&sn=e2675806acebff4d5d3fed8aa2a14019&chksm=e8d0c3e5dfa74af35b943a6d0b72faefa45cdf0411eacc24caee08906e8d6afb12d03452cd47&scene=21#wechat_redirect)  
    
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
*   [最小路径问题 | Dijkstra 算法详解（附代码）](http://mp.weixin.qq.com/s?__biz=MzIzNjc0MTMwMA==&mid=2247578377&idx=2&sn=e12e19dfee0276a5ef641f9ca514bee4&chksm=e8d0c4d2dfa74dc43204f7c1b0ceaf677c7c227e5c573845e92a6d7f6a2d4b682eace1b6704f&scene=21#wechat_redirect)  
    
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
*   [Pytorch - 弹性训练极简实现 (附源码)](http://mp.weixin.qq.com/s?__biz=MzIzNjc0MTMwMA==&mid=2247578203&idx=2&sn=d8cabcbb725d42661c2de6ee561950a8&chksm=e8d0c580dfa74c96564e140238c928ce52944066928de8bda166cf088826884b61eba00fcd62&scene=21#wechat_redirect)  
    
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
*   [称霸 Kaggle 的十大深度学习技巧](http://mp.weixin.qq.com/s?__biz=MzIzNjc0MTMwMA==&mid=2247578160&idx=2&sn=1b391098a621e9e75e9f845e93076347&chksm=e8d0c5ebdfa74cfd560416b2fe24c22be40033eafd964587975ebe2196588f3dd4b4d653b1af&scene=21#wechat_redirect)  
    
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
*   [python 装饰器工程实例及关键点总结](http://mp.weixin.qq.com/s?__biz=MzIzNjc0MTMwMA==&mid=2247577786&idx=1&sn=5abe3d250d0d0189fdb307da9bb8d3b3&chksm=e8d0c761dfa74e77042782dac3366fdfd3e6bd5fc11510e807e7e470772c6b7d32a1c0507c0f&scene=21#wechat_redirect)  
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
*   [语义分割中的 loss function 最全面汇总](http://mp.weixin.qq.com/s?__biz=MzIzNjc0MTMwMA==&mid=2247577739&idx=1&sn=857e16cc7f8fe1948280cc163357c829&chksm=e8d0c750dfa74e46eb9736a91022df47793c47d92fbd001c6f07252c125415be22f0a737657c&scene=21#wechat_redirect)  
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
*   [基于 PyTorch 的卷积神经网络经典 BackBone(骨干网络) 复现](http://mp.weixin.qq.com/s?__biz=MzIzNjc0MTMwMA==&mid=2247577689&idx=1&sn=2cfabaeea490f3c7ced126eeaf86cead&chksm=e8d0c782dfa74e9472230996aa5b5d495f28bcdb327344ff9c65504106e6de2574859b1d554d&scene=21#wechat_redirect)  
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
*   [Best Practice in PyTorch: 如何控制 dataloader 的随机 shuffle](http://mp.weixin.qq.com/s?__biz=MzIzNjc0MTMwMA==&mid=2247577525&idx=2&sn=c842582d96a92aa0ca900c218544d6f7&chksm=e8d0d86edfa751785e27458997b48713af4191faf6ff61ef4c298b4a4e22ff81d56923a1093d&scene=21#wechat_redirect)  
    
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
*   [基于神经机器翻译 (NMT) 的语法纠错算法](http://mp.weixin.qq.com/s?__biz=MzIzNjc0MTMwMA==&mid=2247577442&idx=2&sn=e69417a4bb6358794a6ea4046440444e&chksm=e8d0d8b9dfa751af6d324d8f7bfe112d1cfa8661aa7d9945dbcceb3000045b1f9954f5f2d6a5&scene=21#wechat_redirect)  
    
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
*   [Pytorch+CNN + 猫狗分类实战](http://mp.weixin.qq.com/s?__biz=MzIzNjc0MTMwMA==&mid=2247577288&idx=2&sn=3531d17c914297f64be9bd5e59fd266c&chksm=e8d0d913dfa75005b1395155a4f53a6047070a204e74a2ce686529bd2e9489aec6811a20c340&scene=21#wechat_redirect)  
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
*   [半精度（FP16）调试血泪总结](http://mp.weixin.qq.com/s?__biz=MzIzNjc0MTMwMA==&mid=2247577137&idx=1&sn=3a0e7895b6a2becfbecb650e4b9e0058&chksm=e8d0d9eadfa750fc5ab79667b8cabdd893b73213ba0663043e24ccbe3a38c87c00b4a6135857&scene=21#wechat_redirect)  
    
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
*   [Transformer 中的 position encoding](http://mp.weixin.qq.com/s?__biz=MzIzNjc0MTMwMA==&mid=2247577098&idx=2&sn=5383b5e1d3e8a283e4ffe39ebb99adcc&chksm=e8d0d9d1dfa750c7baf944cf1c251605f1226589c1493478c6131d68a8587446cc1192936664&scene=21#wechat_redirect)  
    
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
*   [PyTorch 常用代码段合集](http://mp.weixin.qq.com/s?__biz=MzIzNjc0MTMwMA==&mid=2247576846&idx=1&sn=b243bc3c06274c519404aabdb0328755&chksm=e8d0dad5dfa753c3745772915ac7149e722dad2027dfd572c2c7cdbe64c5b72e881adf318639&scene=21#wechat_redirect)  
    
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
*   [PyTorch Cookbook（常用代码段整理合集）](http://mp.weixin.qq.com/s?__biz=MzIzNjc0MTMwMA==&mid=2247576787&idx=2&sn=43e5e9d30578fede37ce68071054b0c8&chksm=e8d0db08dfa7521edad75447fac20b067c8570aa9ab3ab7d0037e2d59108e4f60784440bdbd5&scene=21#wechat_redirect)  
    
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
*   [神经网络调参技巧：warmup 策略](http://mp.weixin.qq.com/s?__biz=MzIzNjc0MTMwMA==&mid=2247576757&idx=2&sn=c81e41ec47b77d8e59b4a9a39666b03a&chksm=e8d0db6edfa75278fa797d9cef39bf32f7e24f8b83b41a159e1780b36e0d9ae2fe9ab109fac8&scene=21#wechat_redirect)
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    

更多他山之石专栏文章，
-----------

请点击文章底部 “**阅读原文**” 查看
---------------------

![图片](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjNxkjrUrohP2lyXWW9J4Fcj8XbORR9bX3RBZwwXHUjnFWd06KuCYbdr62gVlH8lBQjI8vN3UGgS9g/640?wx_fmt=png)

![图片](https://mmbiz.qpic.cn/mmbiz_gif/AIR6eRePgjOcJ80hvMcoKX3QNNxrSyqoic7eicAssa7ia6yTSq4xH8HM085Gm8H4hDwkKQ3rkl7qX5zR8mdoPyjeQ/640?wx_fmt=gif)

**分享、点赞、在看，给个三连击呗！**