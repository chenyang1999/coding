#共享网络层
#函数式 API 的另一个用途是使用共享网络层的模型。我们来看看共享层。
#
#来考虑推特推文数据集。我们想要建立一个模型来分辨两条推文是否来自同一个人（例如，通过推文的相似性来对用户进行比较）。
#
#实现这个目标的一种方法是建立一个模型，将两条推文编码成两个向量，连接向量，然后添加逻辑回归层；这将输出两条推文来自同一作者的概率。模型将接收一对对正负表示的推特数据。
#
#由于这个问题是对称的，编码第一条推文的机制应该被完全重用来编码第二条推文。这里我们使用一个共享的 LSTM 层来编码推文。
#
#让我们使用函数式 API 来构建它。首先我们将一条推特转换为一个尺寸为 (140, 256) 的矩阵，即每条推特 140 字符，每个字符为 256 维的 one-hot 编码 （取 256 个常用字符）。

import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

tweet_a = Input(shape=(140, 256))
tweet_b = Input(shape=(140, 256))
#要在不同的输入上共享同一个层，只需实例化该层一次，然后根据需要传入你想要的输入即可：

# 这一层可以输入一个矩阵，并返回一个 64 维的向量
shared_lstm = LSTM(64)

# 当我们重用相同的图层实例多次，图层的权重也会被重用 (它其实就是同一层)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# 然后再连接两个向量：
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

# 再在上面添加一个逻辑回归层
predictions = Dense(1, activation='sigmoid')(merged_vector)

# 定义一个连接推特输入和预测的可训练的模型
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
							loss='binary_crossentropy',
							metrics=['accuracy'])
#model.fit([data_a, data_b], labels, epochs=10)
model.summary()	
#让我们暂停一会，看看如何读取共享层的输出或输出尺寸。
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from keras.utils import plot_model
plot_model(model,to_file='example.png')
lena = mpimg.imread('example.png') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()
