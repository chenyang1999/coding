'''
题目：输入3个数a,b,c，按大小顺序输出。　　　
'''
a=[]
for i in range(3):
	x=int(input())
	a.append(x)
a=set(a)
for _ in a:
	print(_,end=" ")
