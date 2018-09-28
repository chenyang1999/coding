# 多分类问题
model.compile(optimizer='rmsprop',
							loss='categorical_crossentropy',
							metrics=['accuracy'])

# 二分类问题
model.compile(optimizer='rmsprop',
							loss='binary_crossentropy',
							metrics=['accuracy'])

# 均方误差回归问题
model.compile(optimizer='rmsprop',
							loss='mse')

# 自定义评估标准函数
import keras.backend as K

def mean_pred(y_true, y_pred):
		return K.mean(y_pred)

model.compile(optimizer='rmsprop',
							loss='binary_crossentropy',
							metrics=['accuracy', mean_pred])
