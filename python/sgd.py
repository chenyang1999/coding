#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier


# 加载MNIST数据集
def load_dataset():
	from sklearn.datasets import fetch_mldata
	mnist = fetch_mldata('MNIST original', data_home='dataset')
	X, y = mnist['data'], mnist['target']
	X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
	shuffle_index = np.random.permutation(60000)
	X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
	print('load mnist successfully\n', 'X_train shape is: ', X_train.shape, 'X_test shape is:', X_test.shape)
	return X_train, X_test, y_train, y_test


# 展示数据集的样本
def show_data(dataset, labels, index):
	sample = dataset[index]
	sample_img = sample.reshape(28, 28)
	print('The label of this image is:', labels[index])
	plt.imshow(sample_img)
	plt.axis('off')
	plt.show()


# 单个数字的随机梯度下降二分类器
def single_number_classify(X_train,  y_train, number):
	# 重构数据标签，该数字的标签为1其它数字为0
	y_train_i = (y_train == number)
	# y_test_i = (y_test == number)
	# 创建随机梯度下降分类器实例
	sgd_clf = SGDClassifier(random_state=42)
	sgd_clf.fit(X_train, y_train_i)
	return sgd_clf, y_train_i


# 单个数字的随机梯度下降二分类器预测
def snc_predict(sgd_clf, samples):
	predict = sgd_clf.predict(samples)
	print(' Predicted as:', predict)


# 单个数字的随机梯度下降二分类器性能评估,用验证集评估不是测试集
def snc_assess(sgd_clf, X_train, y_train_i):
	# K折交叉验证评分，指标为精度, 设置为3折
	from sklearn.model_selection import cross_val_score
	crs = cross_val_score(sgd_clf, X_train, y_train_i, cv=3, scoring="accuracy")
	print('3折交叉验证精度为：', crs)

	# 计算混淆矩阵, 每一行表示一个实际的类, 而每一列表示一个预测的类, 值为样本的个数
	from sklearn.model_selection import cross_val_predict
	from sklearn.metrics import confusion_matrix
	y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_i, cv=3)
	confu_matrix = confusion_matrix(y_train_i, y_train_pred)
	print('单数字二分类器混淆矩阵为：', confu_matrix)

	# 计算准确率、召回率和F1值
	from sklearn.metrics import precision_score, recall_score, f1_score
	precision = precision_score(y_train_i, y_train_pred)
	recall = recall_score(y_train_i, y_train_pred)
	f1_sco = f1_score(y_train_i, y_train_pred)

	print('准确率为：', precision, '召回率为', recall, 'F1值为：', f1_sco)

	# 获得准确率、召回率、阈值数据
	from sklearn.metrics import precision_recall_curve
	y_scores = cross_val_predict(sgd_clf, X_train, y_train_i, cv=3, method="decision_function")
	precisions, recalls, thresholds = precision_recall_curve(y_train_i, y_scores)
	# 绘制曲线
	plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
	plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
	plt.xlabel("Threshold")
	plt.legend(loc="upper left")
	plt.ylim([0, 1])
	plt.show()

	# ROC曲线，即真正例率（true positive rate，另一个名字叫做召回率）对假正例率（false positive rate, FPR）的曲线
	from sklearn.metrics import roc_curve
	fpr, tpr, thresholds = roc_curve(y_train_i, y_scores)
	# 绘制ROC曲线
	plt.plot(fpr, tpr, linewidth=2, label=None)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.axis([0, 1, 0, 1])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.show()


# 手写数字的随机梯度下降多分类器，默认为OvA/OvR（一对所有/一对其它）
def number_classify_ova(X_train, y_train):
	# 创建随机梯度下降多分类器实例
	sgd_clf = SGDClassifier(random_state=42)
	sgd_clf.fit(X_train, y_train)
	# 预测样本
	sample = X_train[100]
	predict = sgd_clf.predict([sample])
	# 查看该样本在各类中的得分
	digit_scores = sgd_clf.decision_function([sample])
	print('OvA的随机梯度下降分类器预测结果为：', predict, '该样本的各类得分：', digit_scores)
	return sgd_clf


# 手写数字的随机梯度下降多分类器，使用OvO（一对一）
def number_classify_ovo(X_train, y_train):
	# 创建OvO的随机梯度下降多分类器实例
	from sklearn.multiclass import OneVsOneClassifier
	ovo_sgd_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
	ovo_sgd_clf.fit(X_train, y_train)
	# 预测样本
	sample = X_train[100]
	predict = ovo_sgd_clf.predict([sample])
	print('OvO的随机梯度下降分类器预测结果为：', predict)


# 手写数字的随机森林（Random Forest）多分类器
def number_classify_rf(X_train, y_train):
	# 创建随机森林多分分类器
	from sklearn.ensemble import RandomForestClassifier
	forest_clf = RandomForestClassifier(random_state=42)
	forest_clf.fit(X_train, y_train)
	# 预测
	sample = X_train[100]
	predict = forest_clf.predict([sample])
	print('随机森林预测分类器结果为', predict)


# 输入正则化后的手写数字随机梯度下降多分类器的结果
def input_scaled_sgd(sgd_clf, X_train, y_train):
	# 对训练集进行标准的缩放
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
	# 评估分类器的性能
	from sklearn.model_selection import cross_val_predict
	from sklearn.metrics import confusion_matrix
	from sklearn.model_selection import cross_val_score
	scores = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
	print('输入正则化后的手写数字随机梯度下降多分类器的3折交叉验证精度为:', scores)
	y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
	conf_mx = confusion_matrix(y_train, y_train_pred)
	print('输入正则化后的手写数字随机梯度下降多分类器的混淆矩阵:\n', conf_mx)
	# 绘制混淆矩阵图
	# 换算出各元素在行中所占的比例
	row_sums = conf_mx.sum(axis=1, keepdims=True)
	norm_conf_mx = conf_mx / row_sums
	# 对角线置0，为了更好地观察不好的点（亮度高），
	np.fill_diagonal(norm_conf_mx, 0)
	plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
	plt.show()


if __name__ == '__main__':
	# 加载数据集
	X_train, X_test, y_train, y_test = load_dataset()
	# 展示数据集的某个样本
	# show_data(X_train, y_train, 100)
	# 创建单个数字的随机梯度下降二分类器
	# sgd_clf, y_train_i = single_number_classify(X_train, y_train, 5)
	# 用单个数字的随机梯度下降二分类器测试样本
	# snc_predict(sgd_clf, X_train[:3])
	# 评估单个数字的随机梯度下降二分类器的性能
	# snc_assess(sgd_clf, X_train, y_train_i)
	# 使用手写数字的随机梯度下降多分类器，OvA
	sgd_clf_ova = number_classify_ova(X_train, y_train)
	# 使用手写数字的随机梯度下降多分类器，OvO
	# number_classify_ovo(X_train, y_train)
	# 使用手写数字的随机森林多分类器
	# number_classify_rf(X_train, y_train)
	# 输入正则化后手写数字的随机梯度下降多分类器的性能评估
	input_scaled_sgd(sgd_clf_ova, X_train, y_train)