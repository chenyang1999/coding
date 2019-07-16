#题目：求输入数字的平方，如果平方运算后小于 50 则退出。
x=int(input())
while x**2<50:
	print(x**2)
	x=int(input())
print(x**2)