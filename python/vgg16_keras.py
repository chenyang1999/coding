#coding=utf-8  
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils import to_categorical
import numpy as np  
seed = 7  
np.random.seed(seed)  
from keras.datasets import mnist,cifar10
(X_train,y_train),(X_test,y_test)=cifar10.load_data()
print(X_train.shape,y_train.shape)
for _ in range(100):
	print(y_train[_])
#X_train = X_train.reshape(-1, 32, 32,1)
#X_test = X_test.reshape(-1, 28, 28,1)
##print(X_train.shape,y_train.shape)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print(X_train.shape,y_train.shape)
for _ in range(100):
	print(y_train[_])
model = Sequential()  
model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(32,32,3),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Flatten())  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(1024,activation='relu'))  
model.add(Dropout(0.5)) 
model.add(Dense(256,activation='relu'))  
model.add(Dropout(0.5)) 
model.add(Dense(10,activation='softmax'))  
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
model.summary()  
model.fit(X_train,y_train,epochs=5,batch_size=64)
#VGG-16
#[python] view plain copy
##coding=utf-8  
#from keras.models import Sequential  
#from keras.layers import Dense,Flatten,Dropout  
#from keras.layers.convolutional import Conv2D,MaxPooling2D  
#import numpy as np  
#seed = 7  
#np.random.seed(seed)  
#  
#model = Sequential()  
#model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(224,224,3),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(MaxPooling2D(pool_size=(2,2)))  
#model.add(Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(MaxPooling2D(pool_size=(2,2)))  
#model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(MaxPooling2D(pool_size=(2,2)))  
#model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(MaxPooling2D(pool_size=(2,2)))  
#model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(MaxPooling2D(pool_size=(2,2)))  
#model.add(Flatten())  
#model.add(Dense(4096,activation='relu'))  
#model.add(Dropout(0.5))  
#model.add(Dense(4096,activation='relu'))  
#model.add(Dropout(0.5))  
#model.add(Dense(1000,activation='softmax'))  
#model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
#model.summary()  