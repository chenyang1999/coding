'''
题目：列表排序及连接。

程序分析：排序可使用 sort() 方法，连接可以使用 + 号或 extend() 方法。
'''
import random
a=[]
for i in range(10):
	x=random.randint(1,100)
	a.append(x)
print(a)
a.sort()
print(a)