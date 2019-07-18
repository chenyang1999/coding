'''
题目：输入一个奇数，然后判断最少几个 9 除于该数的结果为整数。

程序分析：999999 / 13 = 76923。
'''
z=int(input())
a=0
sum=0
for _ in range(100):
	a=a*10+9
	if a%z==0:
		sum+=1
		print(sum)
		break