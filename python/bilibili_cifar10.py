import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from keras.utils import plot_model
model.summary()
plot_model(model,to_file='example.svg')
plot_model(model,to_file='example.png')
lena = mpimg.imread('example.png') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理Nginx 
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
# plt.show()
from IPython.display import SVG, display
display(SVG('example.svg'))
