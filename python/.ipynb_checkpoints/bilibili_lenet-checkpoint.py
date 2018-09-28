#coding=utf-8
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras.utils import plot_model
(X_train,y_train),(X_test,y_test)=mnist.load_data()
#print(X_train.shape,y_train.shape)
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)
#print(X_train.shape,y_train.shape)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print(X_train.shape,y_train.shape)
model = Sequential()
#layer2
model.add(Conv2D(6, (3,3),strides=(1,1),input_shape=X_train.shape[1:],data_format='channels_first',padding='valid',activation='relu',kernel_initializer='uniform'))
#layer3
model.add(MaxPooling2D((2,2)))
#layer4
model.add(Conv2D(16, (3,3),strides=(1,1),data_format='channels_first',padding='valid',activation='relu',kernel_initializer='uniform'))
#layer5
model.add(MaxPooling2D(2,2))
#layer6
model.add(Conv2D(120, (5,5),strides=(1,1),data_format='channels_first',padding='valid',activation='relu',kernel_initializer='uniform'))
model.add(Flatten())
#layer7
model.add(Dense(84,activation='relu'))
#layer8
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from keras.utils import plot_model
plot_model(model,to_file='example.png',show_shapes=True)
lena = mpimg.imread('example.png') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()
for _ in range(len(model.layers)):
	print (model.layers[_])
print("______________________________________")
print(model.inputs)
print("______________________________________")
print(model.outputs)
print("______________________________________")
config = model.get_config()
model = model.from_config(config)
from keras.models import model_from_json
json_string = model.to_json()
print(json_string)
print("______________________________________")
model = model_from_json(json_string)
#print("train____________")
#model.fit(X_train,y_train,epochs=1,batch_size=128,)
#print("test_____________")
#loss,acc=model.evaluate(X_test,y_test)
#print("loss=",loss)
#print("accuracy=",acc)









