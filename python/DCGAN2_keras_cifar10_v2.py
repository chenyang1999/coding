from __future__ import print_function, division

from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
#from keras.layers.advanced_activations import LeakyReLU
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.misc import *

import sys

import numpy as np

class DCGAN():
	def __init__(self):
		# Input shape
		self.img_rows = 32
		self.img_cols = 32
		self.channels = 3
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		self.latent_dim = 100

		optimizer = Adam(0.0002, 0.5)

		# Build and compile the discriminator
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='binary_crossentropy',
			optimizer=optimizer,
			metrics=['accuracy'])

		# Build the generator
		self.generator = self.build_generator()

		# The generator takes noise as input and generates imgs
		z = Input(shape=(100,))
		img = self.generator(z)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The discriminator takes generated images as input and determines validity
		valid = self.discriminator(img)

		# The combined model  (stacked generator and discriminator)
		# Trains the generator to fool the discriminator
		self.combined = Model(z, valid)
		self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

	def build_generator(self):

		model = Sequential()

		model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
		model.add(Reshape((8, 8, 128)))
		model.add(UpSampling2D())
		model.add(Conv2D(128, kernel_size=3, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("relu"))
		model.add(UpSampling2D())
		model.add(Conv2D(64, kernel_size=3, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("relu"))
		model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
		model.add(Activation("tanh"))

		model.summary()

		noise = Input(shape=(self.latent_dim,))
		img = model(noise)

		return Model(noise, img)

	def build_discriminator(self):

		model = Sequential()

		model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same", trainable=False))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
		model.add(ZeroPadding2D(padding=((0,1),(0,1))))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid'))

		model.summary()

		img = Input(shape=self.img_shape)
		validity = model(img)

		return Model(img, validity)

	def train(self, epochs, batch_size=128, save_interval=100):

		# Load the dataset
		(X_train, _), (_, _) = cifar10.load_data()

		# Rescale -1 to 1
		X_train = X_train / 127.5 - 1.
		print(X_train.shape)
#		X_train = np.expand_dims(X_train, axis=3)
#		print(X_train.shape)
		# Adversarial ground truths
		valid = np.ones((batch_size, 1))
		fake = np.zeros((batch_size, 1))

		for epoch in range(epochs):

			# ---------------------
			#  Train Discriminator
			# ---------------------

			# Select a random half of images
			idx = np.random.randint(0, X_train.shape[0], batch_size)
			imgs = X_train[idx]

			# Sample noise and generate a batch of new images
			noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
			gen_imgs = self.generator.predict(noise)

			# Train the discriminator (real classified as ones and generated as zeros)
			d_loss_real = self.discriminator.train_on_batch(imgs, valid)
			d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			# ---------------------
			#  Train Generator
			# ---------------------

			# Train the generator (wants discriminator to mistake images as real)
			g_loss = self.combined.train_on_batch(noise, valid)

			# Plot the progress
			if(epoch%100==0):	
				print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

			# If at save interval => save generated image samples
			if epoch % save_interval == 0:
				self.save_imgs(epoch)

	def save_imgs(self, epoch):
		r, c = 1, 1
		noise = np.random.normal(0, 1, (r * c, self.latent_dim))
		gen_imgs = self.generator.predict(noise)

		# Rescale images 0 - 1
		gen_imgs = 0.5 * gen_imgs + 0.5
#		print(len(gen_imgs))
		for image_idx in range(1):
#			plt.subplot(3, 3, image_idx+1)
			#generated_image = unnormalize_display(train_data[image_idx]).transpose(1,2,0)
#			print(gen_imgs.shape)
#			print(gen_imgs)
#			generated_image = generated_images[image_idx].transpose(1,2,0)
#			print(generated_image.shape)
			imsave("images/cifar10_%d.png" % epoch,gen_imgs[0])


if __name__ == '__main__':
	dcgan = DCGAN()
	dcgan.train(epochs=12000, batch_size=32, save_interval=50)
