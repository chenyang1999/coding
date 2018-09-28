#!/usr/bin/python

import tensorflow as tf
import numpy as np

#from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import scipy.ndimage
import  os
batch_size = 32  # batch size
num_category = 10  # total categorical factor
#num_cont = 2  # total continuous factor
num_dim = 50  # total latent dimension
T_num = 25
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_flag = True
sample_flag = True
load_flag = False
save_flag = True

z_cat = tf.random_uniform([batch_size],minval=0,maxval=10,dtype=tf.int32)
z_cat = tf.one_hot(z_cat, num_category)

multi_dist = tf.contrib.distributions.StudentT(df=[2.0]*40,loc=[0.0]*40,scale=[1.0]*40)
tmp_noise  = multi_dist.sample([batch_size])   #[32,40]
sess = tf.InteractiveSession()  # 创建一个新的计算图
sess.run(tf.global_variables_initializer())  # 初始化所有参数
#z_cont = z[:, num_category:num_category+num_cont]
px = sess.run(tmp_noise)
py = sess.run(tmp_noise)
px=px.flatten()
py=py.flatten()
np.set_printoptions(threshold=np.nan)
print('pxshape=',px)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
#z = tf.random_normal([2, 3])
color = np.arctan2(px, py)
plt.figure('gaoshi')
plt.scatter(px, py, s = 10, c = color, alpha = 0.7)
# 设置坐标轴范围

plt.xlim((-10, 10))
plt.ylim((-10,10 ))

# 不显示坐标轴的值
#plt.xticks(())
#plt.yticks(())
plt.savefig("noise")



print(z_cat)
print(tmp_noise)
noise = tmp_noise
#multi_dist = tf.random_normal([batch_size,40],mean=0.0,stddev=1.0)
#tmp_noise =multi_dist
with tf.variable_scope('weight'):
    mu = tf.get_variable("mu",[T_num,batch_size,40],initializer=tf.random_uniform_initializer(-1,1))
    sigma = tf.get_variable("sigma",[T_num,batch_size,40],initializer=tf.constant_initializer([0.05]))
    noise = tmp_noise*mu+sigma
    weight = tf.get_variable("weight",[T_num,1,1],initializer=tf.constant_initializer(2))   #5个T分布[5,1,1]
    weight = tf.tile(weight,[1,batch_size,40])   #weight扩到每个参数上[5,32,40]
#	noise = noise * weight
    noise = noise * weight
    noise = tf.reduce_mean(noise,axis=0)

'''
	w_noise=tf.transpose(noise, perm=[1, 0, 2]) 
	w_noise=tf.reshape(w_noise, [batch_size, -1])

	h_w = tf.layers.dense(w_noise , 640 , activation=tf.nn.relu)
	h_w2 = tf.layers.dense(h_w , 320 , activation=tf.nn.relu)

	h_w3= tf.layers.dense(h_w2 , T_num , activation=tf.nn.sigmoid) 
	h_w3=tf.reshape(h_w3 , [batch_size , T_num , 1])
	h_w3= tf.tile(h_w3 , [1 , 1 , 40 ]) 
	w_noise=tf.reshape(w_noise, [batch_size, T_num ,40]) 
	w_noise= w_noise*h_w3

	w_noise = tf.reduce_mean(w_noise,axis=1)'''

z = tf.concat([z_cat,noise],1)
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
plt.figure('afterT')
plt.scatter(px, py, s = 10, c = color, alpha = 0.7)
# 设置坐标轴范围

plt.xlim((-10, 10))
plt.ylim((-10,10 ))

# 不显示坐标轴的值

plt.savefig("cmcT")
##高斯
n = 1024    # data size
px = np.random.normal(-1, 1, n) # 每一个点的X值
py = np.random.normal(-1, 1, n) # 每一个点的Y值
np.set_printoptions(threshold=np.nan)
print('pxshape=',px)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
#z = tf.random_normal([2, 3])
color = np.arctan2(px, py)
plt.figure('gaosi')
plt.scatter(px, py, s = 10, c = color, alpha = 0.7)
# 设置坐标轴范围

plt.xlim((-10, 10))
plt.ylim((-10,10 ))

# 不显示坐标轴的值

plt.savefig('gaoshi')

'''
true_images = tf.placeholder(tf.float32, [batch_size,28,28,1])
true_labels = tf.placeholder(tf.float32, [batch_size,num_category])

with tf.variable_scope('generator'):
	print("generator")
	h0 = tf.layers.dense(z,1024)
	h0 = tf.nn.relu(tf.layers.batch_normalization(h0,training = True))
	print(h0.shape)
	
	h1 = tf.layers.dense(h0,7*7*32)
	h1 = tf.nn.relu(tf.layers.batch_normalization(h1,training = True))
	h1 = tf.reshape(h1,[-1,7,7,32])
	print(h1.shape)

	h2 = tf.layers.conv2d_transpose(h1,16,[4,4],strides=2,padding="same")
	h2 = tf.nn.relu(tf.layers.batch_normalization(h2,training = True))
	print(h2.shape)

	h3 = tf.layers.conv2d_transpose(h2,1,[4,4],strides=2,padding="same")
	h3 = tf.nn.sigmoid(h3)
	print(h3.shape)

def d(xx,reuse=False):
	with tf.variable_scope('discriminator', reuse=reuse):
		print("discriminator")
		h0 = tf.layers.conv2d(xx,16,[4,4],strides=2,padding="same")
		h0 = tf.nn.crelu(h0)
		print(h0.shape)

		h1 = tf.layers.conv2d(h0,32,[4,4],strides=2,padding="same")
		h1 = tf.nn.crelu(tf.layers.batch_normalization(h1,training = True))
		print(h1.shape)

		h12 = tf.layers.conv2d(h1,64,[4,4],strides=2,padding="same")
		h12 = tf.nn.crelu(tf.layers.batch_normalization(h12,training = True))
		print(h12.shape)

		h2 = tf.nn.max_pool(h1, ksize=[1,4,4,1], strides=[1,1,1,1],padding='VALID')
		h2 = tf.contrib.layers.flatten(h1)
		h2 = tf.layers.dense(h2,1024)
		h2 = tf.nn.crelu(tf.layers.batch_normalization(h2,training = True))
		print(h2.shape)

		disc = tf.layers.dense(h2,1)
		disc = tf.squeeze(disc)
		print(disc.shape)

		h3 = tf.layers.dense(h2,128)
		h3 = tf.nn.crelu(h3)

		class_cat = tf.layers.dense(h3,10)
		#class_cont = tf.layers.dense(h3,2)
		#class_cont = tf.nn.sigmoid(class_cont)
	return disc, class_cat#, class_cont

def merge_images(images):
	ret = np.zeros((8,8,28,28))  
	for i in range(8):
		for j in range(8):
			ret[i][j] = images[i*8+j].reshape(28,28)
	return ret

real_disc,real_class = d(true_images)
fake_disc,fake_class = d(h3,reuse = True)

loss_disc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones([batch_size]),logits=real_disc)) + \
			tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros([batch_size]),logits=fake_disc))
loss_class = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_labels,logits=real_class)) + \
			 tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=z_cat,logits=fake_class)) #+ \
			 #tf.reduce_mean(tf.nn.l2_loss(z_cont-fake_cont))
loss_gen = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones([batch_size]),fake_disc))

disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
train_disc = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9).minimize(loss_disc+loss_class,var_list = disc_vars)

gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator") or var.name.startswith("weight")]
train_gen = tf.train.AdamOptimizer(learning_rate=0.001, beta1 = 0.9).minimize(loss_gen+loss_class,var_list = gen_vars)

(train_data, train_labels), (x_test, y_test) = mnist.load_data()

train_data=train_data.reshape(-1,28,28,1)
print (train_data.shape)
x_test = x_test.astype('float32')
# 多分类标签生成
train_labels = keras.utils.to_categorical(train_labels, 10)
y_test = keras.utils.to_categorical(y_test, 10)

epoch_num = 10									#epoch_num遍历数据集的次数
iteration_num = train_data.shape[0]//batch_size #iteration_num=训练数据/batch_size
num_steps = 10000 

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
'''

#PAC



#
#if load_flag:
#	saver.restore(sess,"checkpoint/")
#i = 0
#if train_flag:
#	for epoch in range(epoch_num):		#epoch_num遍历数据集的次数
#		for it_n in range(iteration_num):	#iteration_num=batch_size
#			batch_images = train_data[it_n*batch_size:(it_n+1)*batch_size]
#			batch_labels = train_labels[it_n*batch_size:(it_n+1)*batch_size]
#			
#			l_disc = sess.run([loss_disc,train_disc],feed_dict={ true_images:batch_images,true_labels:batch_labels})[0]
#			l_gen = sess.run([loss_gen,train_gen])[0]
#			i= i+1
#			if i % 1000 == 0 or i == 1:
#				print("epoch:",epoch,"interation_num:",it_n,"l_disc:",l_disc,"l_gen:",l_gen)
#
#	if save_flag:
#		saver.save(sess,"checkpoint/")
#
#if sample_flag:
#	for i in range(100):
#		images = sess.run(h3).reshape(-1,28,28)
#		images_1 = sess.run(h3).reshape(-1,28,28)
#		last = np.concatenate((images,images_1),axis=0).reshape(8,8,28,28)
#		last_image = np.zeros((28*8,28*8))
#		for _ in range(8):
#			for __ in range(8):
#				last_image[_*28:(_+1)*28, __*28:(__+1)*28] = last[_][__]
#		print(last_image.shape)
#		imsave("samples/mnist/64_%d.png"%i,last_image)
#
#
#	labels = np.argmax(sess.run(z_cat),axis=1)
#
#	for i in range(32):
#		print("samples/mnist/%d.png %d"%(i,labels[i]))
#		imsave("samples/mnist/%d.png"%(i),images[i])
#		#imsave("samples")