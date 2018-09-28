#层「节点」的概念
#每当你在某个输入上调用一个层时，都将创建一个新的张量（层的输出），并且为该层添加一个「节点」，将输入张量连接到输出张量。当多次调用同一个图层时，该图层将拥有多个节点索引 (0, 1, 2...)。
#
#在之前版本的 Keras 中，可以通过 layer.get_output() 来获得层实例的输出张量，或者通过  layer.output_shape 来获取其输出形状。现在你依然可以这么做（除了 get_output() 已经被 output 属性替代）。但是如果一个层与多个输入连接呢？
#
#只要一个层只连接到一个输入，就不会有困惑，.output 会返回层的唯一输出：
import keras
from keras.layers import Input, LSTM, Dense,Conv2D
from keras.models import Model
a = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)

assert lstm.output == encoded_a
#但是如果该层有多个输入，那就会出现问题：

a = Input(shape=(140, 256))
b = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)
encoded_b = lstm(b)
#
#lstm.output
#>> AttributeError: Layer lstm_1 has multiple inbound nodes,
#hence the notion of "layer output" is ill-defined.
#Use `get_output_at(node_index)` instead.
#好吧，通过下面的方法可以解决：

assert lstm.get_output_at(0) == encoded_a
assert lstm.get_output_at(1) == encoded_b
#够简单，对吧？
#
#input_shape 和 output_shape 这两个属性也是如此：只要该层只有一个节点，或者只要所有节点具有相同的输入/输出尺寸，那么「层输出/输入尺寸」的概念就被很好地定义，并且将由 layer.output_shape /  layer.input_shape 返回。但是比如说，如果将一个 Conv2D 层先应用于尺寸为 (32，32，3) 的输入，再应用于尺寸为 (64, 64, 3) 的输入，那么这个层就会有多个输入/输出尺寸，你将不得不通过指定它们所属节点的索引来获取它们：

a = Input(shape=(32, 32, 3))
b = Input(shape=(64, 64, 3))

conv = Conv2D(16, (3, 3), padding='same')
conved_a = conv(a)

# 到目前为止只有一个输入，以下可行：
assert conv.input_shape == (None, 32, 32, 3)

conved_b = conv(b)
# 现在 `.input_shape` 属性不可行，但是这样可以：
assert conv.get_input_shape_at(0) == (None, 32, 32, 3)
assert conv.get_input_shape_at(1) == (None, 64, 64, 3)
#show
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from keras.utils import plot_model
plot_model(model,to_file='example.png')
lena = mpimg.imread('example.png') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
#plt.show()