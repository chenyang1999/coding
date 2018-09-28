#optimizer
from keras import optimizers

model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
# 传入优化器名称: 默认参数将被采用
model.compile(loss='mean_squared_error', optimizer='sgd')
from keras import optimizers

# 所有参数梯度将被裁剪，让其l2范数最大为1：g * 1 / max(1, l2_norm)
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
from keras import optimizers

# 所有参数d 梯度将被裁剪到数值范围内：
# 最大值0.5
# 最小值-0.5
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
#SGD
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#随机梯度下降优化器
#包含扩展功能的支持： - 动量（momentum）优化, - 学习率衰减（每次参数更新后） - Nestrov动量(NAG)优化
#
#参数
#lr: float >= 0. 学习率
#momentum: float >= 0. 参数，用于加速SGD在相关方向上前进，并抑制震荡
#decay: float >= 0. 每次参数更新后学习率衰减值.
#nesterov: boolean. 是否使用Nesterov动量.
#RMSprop
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#RMSProp优化器。
#
#建议使用优化器的默认参数（除了学习率lr，它可以被自由调节）
#
#这个优化器通常是训练循环神经网络RNN的不错选择。
#
#参数
#
#lr：float> = 0.学习率。
#rho：float> = 0. RMSProp梯度平方的移动均值的衰减率。
#epsilon：float> = 0.模糊因子。若为None，默认为K.epsilon()。
#衰变：float> = 0.每次参数更新后学习率衰减值。
#引用
#
#rmsprop：将梯度除以其最近幅度的运行平均值
#[资源]
#
#Adagrad
keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
#Adagrad优化器。
#
#Adagrad是一种具有特定参数学习率的优化器，它根据参数在训练期间的更新频率进行自适应调整。参数接收的更新越多，更新越小。
#
#建议使用优化器的默认参数。
#
#参数
#
#lr：float> = 0.学习率。
#epsilon：float> = 0.若为None，默认为K.epsilon()。
#衰变：float> = 0.每次参数更新后学习率衰减值。
#引用
#
#Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
#[source]
#
#Adadelta
keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
#Adadelta优化器.
#
#Adadelta是Adagrad的一个具有更强鲁棒性的的扩展版本，它不是累积所有过去的梯度，而是根据渐变更新的移动窗口调整学习速率。 这样，即使进行了许多更新，Adadelta仍在继续学习。 与Adagrad相比，在Adadelta的原始版本中，您无需设置初始学习率。 在此版本中，与大多数其他Keras优化器一样，可以设置初始学习速率和衰减因子。
#
#建议使用优化器的默认参数。
#
#参数
#
#lr: float >= 0. 学习率，建议保留默认值.
#rho: float >= 0. Adadelta梯度平方移动均值的衰减率
#epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon().
#decay: float >= 0. 每次参数更新后学习率衰减值.
#引用
#
#Adadelta - an adaptive learning rate method
#[source]
#
#Adam
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#Adam优化器.
#
#默认参数遵循原论文中提供的值。
#
#参数
#
#lr: float >= 0. 学习率.
#beta_1: float, 0 < beta < 1. 通常接近于 1.
#beta_2: float, 0 < beta < 1. 通常接近于 1.
#epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon().
#decay: float >= 0. 每次参数更新后学习率衰减值.
#amsgrad: boolean. 是否应用此算法的AMSGrad变种，来自论文"On the Convergence of Adam and Beyond".
#引用
#
#Adam - A Method for Stochastic Optimization
#On the Convergence of Adam and Beyond
#[source]
#
#Adamax
keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#Adamax优化器，来自Adam论文的第七小节.
#
#它是Adam算法基于无穷范数（infinity norm）的变种。 默认参数遵循论文中提供的值。
#
#参数
#
#lr: float >= 0. 学习率.
#beta_1/beta_2: floats, 0 < beta < 1. 通常接近于 1.
#epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon().
#decay: float >= 0. 每次参数更新后学习率衰减值.
#引用
#
#Adam - A Method for Stochastic Optimization
#[source]
#
#Nadam
keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#Nesterov版本Adam优化器.
#
#正像Adam本质上是RMSProp与动量momentum的结合， Nadam是采用Nesterov momentum版本的Adam优化器。
#
#默认参数遵循论文中提供的值。 建议使用优化器的默认参数。
#
#参数
#
#lr: float >= 0. 学习率.
#beta_1/beta_2: floats, 0 < beta < 1. 通常接近于 1.
#epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon().
#引用
#
#Nadam report
#On the importance of initialization and momentum in deep learning
#[source]
#
#TFOptimizer
keras.optimizers.TFOptimizer(optimizer)
#原生Tensorlfow优化器的包装类（wrapper class）。