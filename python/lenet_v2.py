#coding=utf-8  
from keras.models import Sequential  
from keras.layers import Dense,Flatten  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical  
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np  
seed = 7  
np.random.seed(seed)  
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape,y_train.shape)
X_train = X_train.reshape(-1,28, 28,1)/255.
X_test = X_test.reshape(-1,28, 28,1)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print(X_train.shape,y_train.shape)
model = Sequential()  
model.add(Conv2D(6,(3,3),strides=(1,1),input_shape=(28,28,1),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(16,(3,3),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(120,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform')) 
model.add(Flatten())    
model.add(Dense(84,activation='relu'))  
model.add(Dense(10,activation='softmax'))  
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])  
model.summary()  
  
print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=10, batch_size=64,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
