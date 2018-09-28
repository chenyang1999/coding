#如何在 Keras 中使用预训练的模型？
我们提供了以下图像分类模型的代码和预训练的权重：

	Xception
	VGG16
	VGG19
	ResNet50
	Inception v3
	Inception-ResNet v2
	MobileNet v1
它们可以使用 keras.applications 模块进行导入：

	from keras.applications.xception import Xception
	from keras.applications.vgg16 import VGG16
	from keras.applications.vgg19 import VGG19
	from keras.applications.resnet50 import ResNet50
	from keras.applications.inception_v3 import InceptionV3
	from keras.applications.inception_resnet_v2 import InceptionResNetV2
	from keras.applications.mobilenet import MobileNet

	model = VGG16(weights='imagenet', include_top=True)