'''
题目：给一个不多于5位的正整数，要求：一、求它是几位数，二、逆序打印出各位数字。

程序分析：学会分解出每一位数。
'''
x=int(input("input a number: "))
a=[]
while x>0:
	a.append(x%10)
	x//=10
a.reverse()
print(a)