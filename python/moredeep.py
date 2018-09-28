import tensorflow as tf
import numpy as np
import os
from scipy.misc import *
#from read_cifar10 import *
from keras.datasets import cifar10
from keras.utils import np_utils
import keras
batch_size = 100   # batch size
num_category = 10  # total categorical factor
#num_cont = 10 # total continuous factor
num_dim = 100  # total latent dimension
T_num = 50
train_flag = True
sample_flag = True
load_flag = False
save_flag = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
z_cat = tf.random_uniform([batch_size],minval=0,maxval=10,dtype=tf.int32)
z_cat = tf.one_hot(z_cat, 10)

multi_dist = tf.contrib.distributions.StudentT(df=[2.0]*100,loc=[0.0]*100,scale=[1.0]*100)
tmp_noise = multi_dist.sample([T_num,batch_size])

with tf.variable_scope('weight'):
	mu = tf.get_variable("mu",[T_num,batch_size,100])
	sigma = tf.get_variable("sigma",[T_num,batch_size,100])
	noise = tmp_noise*mu+sigma

	noise = tf.transpose(noise, perm=[1, 0, 2])
	noise = tf.reshape(noise, [batch_size, -1])
	h_w = tf.layers.dense(noise, 1024, activation=tf.nn.relu)
	h_w2 = tf.layers.dense(h_w, 640, activation=tf.nn.relu)

	h_w3 = tf.layers.dense(h_w2, 320, activation=tf.nn.relu)
	h_w4 = tf.layers.dense(h_w3, T_num, activation=tf.nn.sigmoid)
	h_w4 = tf.reshape(h_w4, [batch_size, T_num, 1])
	weight = tf.tile(h_w4, [1, 1, 100])
	noise = tf.reshape(noise, [batch_size, T_num, 100])
	noise = noise* weight

	noise = tf.reduce_mean(noise,axis=1)

z = tf.concat([z_cat,noise],1)
#z_cont = z[:, num_category:num_category+num_cont]
#z=noise

true_images = tf.placeholder(tf.float32, [batch_size,32,32,3])
true_labels = tf.placeholder(tf.float32, [batch_size,10])

def generator(z):
	with tf.variable_scope('generator'):
		print("generator")
		h0 = tf.layers.dense(z,4*4*512)
		h0 = tf.nn.relu(tf.layers.batch_normalization(h0,training = True))  #(32, 1024)
		print(h0.shape)
		h1 = tf.layers.dense(h0,8*8*256)  
		h1 = tf.nn.relu(tf.layers.batch_normalization(h1,training = True))
		h1 = tf.reshape(h1,[-1,8,8,256])
		h1 = tf.nn.dropout(h1,0.6)		#(32, 8, 8, 258)
		print(h1.shape)
		#l2__________
		h2 = tf.layers.conv2d_transpose(h1,256,[3,3],strides=2,padding="same")
		print(h2.shape)
		h2 = tf.reshape(h2,[-1,16,16,256,1])
		print(h2.shape)
		h2 = tf.layers.max_pooling3d(h2,(1,1,3),strides=(1,1,2),padding="same")
		print(h2.shape)
		h2 = tf.reshape(h2,[-1,16,16,128])
		print(h2.shape)
#		h2 = tf.layers.conv2d_transpose(h2,128,[3,3],strides=1,padding="same")
		h2 = tf.nn.relu(tf.layers.batch_normalization(h2,training = True))	#(32, 16, 16, 128)
		h2 = tf.nn.dropout(h2,0.6)	
		print(h2.shape)
		#l3___________
		h3 = tf.layers.conv2d_transpose(h2,9,[3,3],strides=2,padding="same")
		h3 = tf.reshape(h3,[-1,32,32,9,1])
		print(h3.shape)
		h3 = tf.layers.average_pooling3d(h3,(1,1,3),strides=(1,1,3),padding="same")
		print(h3.shape)
		h3 = tf.reshape(h3,[-1,32,32,3])
		print(h3.shape)
#		h3 = tf.layers.max_pooling3d(h3,(1,1,2),2,padding="same")
#		h3 = tf.layers.conv2d(h3,3,[3,3],strides=1,padding="same")
		h3 = tf.nn.tanh(h3) 
		h3 = tf.nn.dropout(h3,0.6)	      
		print(h3.shape)      #(32, 32, 32, 3)
		return(h3)

def discriminator(image,reuse=False):
	with tf.variable_scope('discriminator', reuse=reuse):
		#h02 = tf.layers.conv2d(xx,16,[4,4],strides=2,padding="same")
		#h02 = tf.nn.leaky_relu(h02)

		#h01 = tf.layers.conv2d(h02,32,[4,4],strides=2,padding="same")
		#h01 = tf.nn.leaky_relu(h01)
		print("discriminator")
		print(image.shape)
		h0 = tf.layers.conv2d(image,64,[3,3],strides=2,padding="same")
		h0 = tf.nn.relu(h0)   	#(32, 16, 16, 64)
		h0 = tf.nn.dropout(h0,0.8)	
		print(h0.shape)

		h1 = tf.layers.conv2d(h0,384,[3,3],strides=2,padding="same")
		h1 = tf.reshape(h1,[-1,8,8,384,1])
		print(h1.shape)
		h1 = tf.layers.max_pooling3d(h1,(1,1,3),strides=(1,1,3),padding="same")
		print(h1.shape)
		h1 = tf.reshape(h1,[-1,8,8,128])
		print(h1.shape)
		h1 = tf.nn.relu(h1)		#(32, 8, 8, 128)
		print(h1.shape)

		h11 = tf.layers.conv2d(h1,256,[3,3],strides=2,padding="same")
		h11 = tf.nn.relu(h11)
		h11 = tf.nn.dropout(h11,0.6)	    	#(32, 4, 4, 256)
		print(h11.shape)

		h2 = tf.contrib.layers.flatten(h11)
		h2 = tf.layers.dense(h2, 1024)    
		h2 = tf.nn.relu(h2)
		h2 = tf.nn.dropout(h2,0.6)	      # (batch_size, 1024)
		print(h2.shape)

		disc = tf.layers.dense(h2,1)
		disc = tf.squeeze(disc)
		print(disc.shape)

		h3 = tf.layers.dense(h2,128)
		h3 = tf.nn.relu(h3)        #(batch_size, 128)
		print(h3.shape)

		class_cat = tf.layers.dense(h3,num_category)
		#class_cont = tf.layers.dense(h3,num_cont)
		#class_cont = tf.nn.sigmoid(class_cont)
	return disc, class_cat#, class_cont

G = generator(z)
real_disc,real_class= discriminator(true_images)
fake_disc,fake_class= discriminator(G,reuse = True)

loss_disc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones([batch_size]),logits=real_disc)) + \
			tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros([batch_size]),logits=fake_disc))
loss_class = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_labels,logits=real_class)) + \
			 tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=z_cat,logits=fake_class)) #+ \
			 #tf.reduce_mean(tf.nn.l2_loss(z_cont-fake_cont))'''
loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones([batch_size]),logits=fake_disc))

disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
train_disc = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9).minimize(loss_disc,var_list = disc_vars)

gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator") or var.name.startswith("weight")]
train_gen = tf.train.AdamOptimizer(learning_rate=0.001, beta1 = 0.9).minimize(loss_gen,var_list = gen_vars)

(train_data, train_labels),(_,_) = cifar10.load_data()
train_labels=np_utils.to_categorical(train_labels,num_classes=10)
#train_labels=tf.contrib.distributions.OneHotCategorical(train_labels)
epoch_num = 100
iteration_num = train_data.shape[0]//batch_size

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

if load_flag:
	saver.restore(sess,"checkpoint/")

if train_flag:
	for epoch in range(epoch_num):
		for it_n in range(iteration_num):
			batch_images = train_data[it_n*batch_size:(it_n+1)*batch_size]
			batch_labels = train_labels[it_n*batch_size:(it_n+1)*batch_size]
			l_disc = sess.run([loss_disc,train_disc],feed_dict={true_images:batch_images, true_labels:batch_labels})[0]
			l_gen = sess.run([loss_gen,train_gen])[0]
			print("epoch: %d, iteration_num:%d, dloss:%f, gloss:%f "%(epoch, it_n, l_disc, l_gen))

	if save_flag:
		saver.save(sess,"checkpoint/")

if sample_flag:
	images = sess.run(G)
	images = images.reshape(-1,32,32,3)
	for i in range(batch_size):
		imsave("samples/cifar/%d.png"%(i),images[i])
