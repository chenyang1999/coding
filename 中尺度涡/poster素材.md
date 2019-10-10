- **题目（title）**

    基于深度区域提取网络对中尺度涡的检测算法

    **Detection algorithm of mesoscale eddy based on deep region extraction network**

- **作者信息**

    陈扬1 ，杨琛1 ，刘博文1 

    1 中国海洋大学 青岛 (chenyang8484@stu.ouc.edu.cn) 

    Chen Yang1 , Yang Chen1 , Liu Bowen1
    1Ocean University of China, Qingdao
    (chenyang8484@stu.ouc.edu.cn)

    

- 摘要（Abstract）

    中尺度涡是海洋科学领域一个重要的研究课题。其中，中尺度涡的检测是中尺度涡研究中重要的研究方
    向，有着非常重要的科学意义。近年以来，人工智能领域中的深度神经网络高速发展，其被广泛应用于解决计
    算机视觉中许多实际问题。本文将深度学习中的目标检测算法应用于中尺度涡检测，相比较于传统的中尺度涡
    检测方法只能检测中尺度涡的位置和大小，其能更好地利用多模态信息进行定位、分类和实例分割。本文基于
    Mask-RCNN 算法，提出一种结合多模态卫星遥感图像数据的目标检测算法，对海洋中的中尺度涡进行识别、
    分类和分割。实验结果表明我们的方法能够有效提取中尺度涡的特征，并对其进行精确检测和定位，同时我们
    的方法获得了较高准确率。

    关键词: 中尺度涡，目标检测算法，深度学习，多模态数据融合

- **简介（Introduction/Background）**

    在海洋领域，中尺度涡是指在海洋中持续时间为2到10个月、半径约为10到100千米的涡旋。研究中尺度涡的研究具有非常重要的科学意义和渔业价值。 

    传统算法利用海表高度、温度等数据，基于流场几何特征、边缘检测以及拉格朗日随机模型进行中尺度涡检测，这些算法误检率较高。

    我们将探索使用人工智能中的深度学习算法，对涡旋进行检测、分类和实例分割，也就是结合深度卷积神经网络和多模态海洋卫星遥感数据，实现对中尺度涡的精确检测。我们首先使用多模态数据融合，对卫星遥感数据，如海洋表面高度(SSH)、温度(SST))及流速数据(SSV)进行融合学习,接下来残差神经网络部分负责学习中尺度涡的特征表示，区域生成网络生成含有中尺度涡的区域并提取特征，头网络部分负责中尺度涡类别和范围预测，整体网络能通过端到端的方式进行训练。

- 数据来源

    本文实验的数据是GLORYS2V4中2000年01月16日到2009年12月16日共计十年的温度、高度以及流速数据，其中这三种数据的维度分别为681 × 1440 × 120，681 为维度的维度，1440 为经度的维度，120 表示数据来自连续的 120 个月。

    ![image-20190926134327051](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-054328.png)

- 多模态数据融合

    - 数据归一化

        ![image-20190926134530890](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-054531.png)

    - 使用卷积神经网络进行特征融合
    
  ![image-20190926134546612](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-054634.png)

    ![image-20190926134437282](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-054437.png)

- 基于深度区域提取网络的中尺度涡检测算法

    - 深度残差网络

        本文中针对中尺度涡的检测问题难度相比较于复杂场景目标检测任务来说，难度较低，所以我们选择了在 coco 数据集下预训练好的ResNet101作为特征提取网络。由于我们在 RestNet101使用了残差连接，可以有效地减少网络的过拟合，同时减少梯度在反向传播的时候产生梯度弥散现象。

        ![image-20190926135005077](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-055005.png)

    - 区域生成网络RPN

        区域生成网络PRN(Region Proposal Networks)网络，作用是找出多模态数据中可能存在旋涡的位置(proposals)。

        ![image-20190926135028550](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-055028.png)

        

        产生proposals需要用到3×3的滑动窗口(slides-window)，每一个滑动窗口需要产生 9 个锚点框(anchors boxs)，每一个锚点框由6个变量组成，分别是:x,y,w,h,p,p̂, 分别表示的是对输入数据中可能存在涡旋的位置的坐标(x,y)，以及以该坐标为几何中心的宽为w，高为h的锚点框，p表示该锚点框为旋涡的概率，p̂ 表示不是旋涡的概率。通过softmax判断anchors box属于positive或者negative

        ![image-20190926135653886](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-055654.png)

        ​            接下来通过比较 anchor box 和 ground truth 的 IOU是否大于置信度求得 P*，对预测为旋涡的 anchor box 的![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-055639.gif)做回归。

        先计算positive anchor与ground truth之间的平移量![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-055640.gif)和![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-055642.gif):

        ![image-20190926135711231](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-055711.png)

        设计回归的损失函数为：

        ![image-20190926135740815](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-055741.png)

        近似于优化目标![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-055750.gif):

        ![image-20190926135755878](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-055756.png)

        我们在 RPN 层在通过设立分类和回归两个任务，找出 feature maps 在对应位置可能含有旋涡的锚点位置及其的anchor box的大小。

        

    - ROI池化

        ![image-20190926135243581](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-055243.png)

        其前向传播公式为：

        ![image-20190926135330334](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-55332.png)

        

        ​            其反向传播公式为:

        

        ![image-20190926135336961](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-055337.png)

        

        ​            利用 ROIAlign 下采样方法，我们很大程度上解决了传统池化方法中像素点精度造成的Misalignment 对齐问题。

    - 定位、分类和实例分割

        ![image-20190926135907700](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-055908.png)

    - 第一部分负责判断中尺度涡的类别。上一步中的 RPN 网络已经帮我们检测出来该ROI区域内存在旋涡，这一部分的全连接网络的输出是一个仍然是一个二维的向量，每一维分别表示气旋涡和反气旋涡的概率。其损失函数![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-055934.gif)的计算公式为:

        ![image-20190926135946243](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-055946.png)

    - 第二部分为预测中尺度涡的位置，其损失函数![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-060014.gif)表示预测的旋涡所在的位置。其计算公式为为: 

        ![image-20190926140021580](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-060022.png)

    - 第三部分为掩码分支，其作用在于对目标区域的中尺度涡进行实例分割。通过逐像素计算其平均二值交叉熵得到其掩码损失函数![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-060103.gif)为:

        ![image-20190926140107754](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-060107.png)

        通过上面三个损失函数，我们求得了多模态数据中的存在的中尺度涡的分类、定位以及实例分割

- **结果（Result）**  

    - 功能性：

    ![image-20190926140225223](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-060225.png)

    ​       在功能实现方面，DeepEddy 仅能实现涡旋分类，而 Eddynet 在涡旋分类的基础上实现了涡旋分割，本研究基于多模态遥感数据，可以实现涡旋定位、分类和实例分割。

    - 准确率：

        ![image-20190926140250707](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-26-060251.png)

        在准确率上，本方法比同为深度学习算法的EddyNet提升了34.72%，效果显著。

- **结论/总结（Conclusion/Summary）**

    1. 本研究提出了基于深度区域提取网络的涡旋检测模型。在功能方面，实现了在卫星遥感数据上的中尺度涡的定位，分类和实例分割。

    2. 本研究使用了多模态数据融合。在中尺度涡检测的过程中使用到了海表面高度、海表面温度，和海水流速数据。通过实验找到了多模态数据融合的最佳策略。

    3. 本研究验证了中尺度涡检测模型的有效性。本研究提出了一个中尺度涡检测数据集。数据集制作中采用了传统非深度学习算法，对遥感卫星数据中每个数据点的类别进行精准标注;并在中尺度涡检测数据集上进行学习，得到了很好的准确的

- 致谢（Acknowledgement）

    你看着写

- 未来展望（Future Works）

  你看着吹
  
    