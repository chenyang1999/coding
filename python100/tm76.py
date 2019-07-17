'''
题目：编写一个函数，输入n为偶数时，调用函数求1/2+1/4+...+1/n,当输入n为奇数时，调用函数1/1+1/3+...+1/n
'''
n=int(input("input a number: "))
if n%2==0:
	x=2
else:
	x=1
sum=0
while x<=n:
	sum+=1/x
	x+=2
print(sum)