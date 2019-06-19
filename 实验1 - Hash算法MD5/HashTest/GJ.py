import math
import numpy as np  
from numpy import * 
import matplotlib.pyplot as plt
def Fun(n):                       #函数形式 
	a=np.random.randint(low=1, high=100, size=(n, n+1))  #产生随机数组
	return a
def GJ(a):                     #高斯-约旦法，除对角元素外全是零，
	row=a.shape[0]
	print(a)    
	for j in range(0,row):
		if j<row:
			b=FindLarge(a[j:,j])     #最后一个元素不用找最大值；只找对角线下方的主元
		else:
			b=0
		b1=b+j                      #   主元和对角线所在行交换
		c= np.copy(a[b1,:])  #  更方便的方法：   a[[b1,j], :] = a[[j,b1], :]
		a[b1,:]=a[j,:]
		a[j,:]=c
		for i in range(j, row):
			if i==j:
				continue
			a[i,:]=a[i,:]-a[j,:]*a[i,j]/a[j,j]
			print(a)
			print("---------------------------------------------------------")
	return a
def FindLarge(a0):          #寻找主元
	b0=np.argmax(a0)
#	print(b0)
	return b0
a=Fun(10)
a1=GJ(a)
print(a1)

