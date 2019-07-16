'''
题目：一个5位数，判断它是不是回文数。即12321是回文数，个位与万位相同，十位与千位相同。
'''
x=str(input())
for i in range(len(x)//2):
	if x[i]!=x[-i-1]:
		print(x+"不是一个回文数")
		exit
print (x,"是一个回文数!" )