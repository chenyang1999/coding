#from __future__ import print_function
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalMaxPooling2D
#from lsuv_init import LSUVinit

batch_size = 32 
num_classes = 10
epochs = 1600
data_augmentation = True

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train[1])
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("y_train's shape=",y_train.shape)
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(48, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(48, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(80, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(GlobalMaxPooling2D())
model.add(Dropout(0.25))

#model.add(ZeroPadding2D((1, 1)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dropout(0.2))
'''
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
'''

model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from keras.utils import plot_model
plot_model(model,to_file='example.png',show_shapes=True)
lena = mpimg.imread('example.png') # lena.png

lena.shape #(512, 512, 3)
plt.imshow(lena) # 
plt.axis('off') # 
#plt.show()
# initiate RMSprop optimizer
opt = keras.optimizers.Adam(lr=0.0001)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
							optimizer=opt,
							metrics=['accuracy'])


#model = LSUVinit(model,x_train[:batch_size,:,:,:]) 
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph2', histogram_freq=0, write_graph=True, write_images=True)

if not data_augmentation:
		print('Not using data augmentation.')
		model.fit(x_train, y_train,
							batch_size=batch_size,
							epochs=epochs,
							validation_data=(x_test, y_test),
							shuffle=True, callbacks=[tbCallBack])
else:
		print('Using real-time data augmentation.')
		# This will do preprocessing and realtime data augmentation:
		'''
		datagen = ImageDataGenerator(
				featurewise_center=False,  # set input mean to 0 over the dataset
				samplewise_center=False,  # set each sample mean to 0
				featurewise_std_normalization=False,  # divide inputs by std of the dataset
				samplewise_std_normalization=False,  # divide each input by its std
				zca_whitening=False,  # apply ZCA whitening
				rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
				width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
				horizontal_flip=True,  # randomly flip images
				vertical_flip=False)  # randomly flip images
		'''
		datagen = ImageDataGenerator(
				featurewise_center=False,  # set input mean to 0 over the dataset
				samplewise_center=False,  # set each sample mean to 0
				featurewise_std_normalization=False,  # divide inputs by std of the dataset
				samplewise_std_normalization=False,  # divide each input by its std
				zca_whitening=False,  # apply ZCA whitening
				rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
				width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
				horizontal_flip=True,  # randomly flip images
				vertical_flip=False)  # randomly flip images


		# Compute quantities required for feature-wise normalization
		# (std, mean, and principal components if ZCA whitening is applied).
		datagen.fit(x_train)

		# Fit the model on the batches generated by datagen.flow().
		model.fit_generator(datagen.flow(x_train, y_train,
																		 batch_size=batch_size),
												steps_per_epoch=x_train.shape[0] // batch_size,
												epochs=epochs,
												validation_data=(x_test, y_test), callbacks=[tbCallBack])