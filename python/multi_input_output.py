#多输入多输出模型
#以下是函数式 API 的一个很好的例子：具有多个输入和输出的模型。函数式 API 使处理大量交织的数据流变得容易。
#
#来考虑下面的模型。我们试图预测 Twitter 上的一条新闻标题有多少转发和点赞数。模型的主要输入将是新闻标题本身，即一系列词语，但是为了增添趣味，我们的模型还添加了其他的辅助输入来接收额外的数据，例如新闻标题的发布的时间等。 该模型也将通过两个损失函数进行监督学习。较早地在模型中使用主损失函数，是深度学习模型的一个良好正则方法。
#
#模型结构如下图所示：
#
#multi-input-multi-output-graph
#
#让我们用函数式 API 来实现它。
#
#主要输入接收新闻标题本身，即一个整数序列（每个整数编码一个词）。 这些整数在 1 到 10,000 之间（10,000 个词的词汇表），且序列长度为 100 个词。

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras
# 标题输入：接收一个含有 100 个整数的序列，每个整数在 1 到 10000 之间。
# 注意我们可以通过传递一个 `name` 参数来命名任何层。
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# Embedding 层将输入序列编码为一个稠密向量的序列，每个向量维度为 512。
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# LSTM 层把向量序列转换成单个向量，它包含整个序列的上下文信息
lstm_out = LSTM(32)(x)
#在这里，我们插入辅助损失，使得即使在模型主损失很高的情况下，LSTM 层和 Embedding 层都能被平稳地训练。

auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
#此时，我们将辅助输入数据与 LSTM 层的输出连接起来，输入到模型中：

auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# 堆叠多个全连接网络层
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# 最后添加主要的逻辑回归层
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
#然后定义一个具有两个输入和两个输出的模型：

model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
#现在编译模型，并给辅助损失分配一个 0.2 的权重。如果要为不同的输出指定不同的 loss_weights 或 loss，可以使用列表或字典。 在这里，我们给 loss 参数传递单个损失函数，这个损失将用于所有的输出。

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
							loss_weights=[1., 0.2])
#我们可以通过传递输入数组和目标数组的列表来训练模型：

#model.fit([headline_data, additional_data], [labels, labels],
#					epochs=50, batch_size=32)
#由于输入和输出均被命名了（在定义时传递了一个 name 参数），我们也可以通过以下方式编译模型：

model.compile(optimizer='rmsprop',
							loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
							loss_weights={'main_output': 1., 'aux_output': 0.2})

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
# 然后使用以下方式训练：
#model.fit({'main_input': headline_data, 'aux_input': additional_data},
#					{'main_output': labels, 'aux_output': labels},
#					epochs=50, batch_size=32)