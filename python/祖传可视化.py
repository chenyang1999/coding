#!/usr/bin/python

x=z
y=x
sess = tf.InteractiveSession()  # 创建一个新的计算图
sess.run(tf.global_variables_initializer())  # 初始化所有参数
#z_cont = z[:, num_category:num_category+num_cont]
px = sess.run(x)
py = sess.run(y)
px=px.flatten()
py=py.flatten()
np.set_printoptions(threshold=np.nan)
print('pxshape=',px)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
#z = tf.random_normal([2, 3])
color = np.arctan2(px, py)
plt.scatter(px, py, s = 2, c = color, alpha = 0.5)
# 设置坐标轴范围
plt.xlim((-2, 2))
plt.ylim((-2,2 ))

# 不显示坐标轴的值

plt.show()
