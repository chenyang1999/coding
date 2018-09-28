#你可以使用 model.save(filepath) 将 Keras 模型保存到单个 HDF5 文件中，该文件将包含：
#
#模型的结构，允许重新创建模型
#模型的权重
#训练配置项（损失函数，优化器）
#优化器状态，允许准确地从你上次结束的地方继续训练。
#你可以使用 keras.models.load_model(filepath) 重新实例化模型。load_model 还将负责使用保存的训练配置项来编译模型（除非模型从未编译过）。
#
#例子：
#
from keras.models import load_model

model.save('my_model.h5')  # 创建 HDF5 文件 'my_model.h5'
del model  # 删除现有模型
# 返回一个编译好的模型
# 与之前那个相同
model = load_model('my_model.h5')

#只保存/加载 模型的结构
#如果您只需要保存模型的结构，而非其权重或训练配置项，则可以执行以下操作：
#
# 保存为 JSON
json_string = model.to_json()

# 保存为 YAML
yaml_string = model.to_yaml()
#生成的 JSON/YAML 文件是人类可读的，如果需要还可以手动编辑。

#你可以从这些数据建立一个新的模型：

# 从 JSON 重建模型：
from keras.models import model_from_json
model = model_from_json(json_string)

# 从 YAML 重建模型：
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)
#只保存/加载 模型的权重
#、如果您只需要 模型的权重，可以使用下面的代码以 HDF5 格式进行保存。
#请注意，我们首先需要安装 HDF5 和 Python 库 h5py，它们不包含在 Keras 中。

model.save_weights('my_model_weights.h5')
#假设你有用于实例化模型的代码，则可以将保存的权重加载到具有相同结构的模型中：

model.load_weights('my_model_weights.h5')
#如果你需要将权重加载到不同的结构（有一些共同层）的模型中，例如微调或迁移学习，则可以按层的名字来加载权重：

model.load_weights('my_model_weights.h5', by_name=True)
#例如：
#
#"""
#假设原始模型如下所示：
#	model = Sequential()
#	model.add(Dense(2, input_dim=3, name='dense_1'))
#	model.add(Dense(3, name='dense_2'))
#	...
#	model.save_weights(fname)
#"""

# 新模型
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # 将被加载
model.add(Dense(10, name='new_dense'))  # 将不被加载

# 从第一个模型加载权重；只会影响第一层，dense_1
model.load_weights(fname, by_name=True)