'''
题目：求s=a+aa+aaa+aaaa+aa...a的值，其中a是一个数字。例如2+22+222+2222+22222(此时共有5个数相加)，几个数相加由键盘控制。

程序分析：关键是计算出每一项的值。
'''
a=int(input("a: "))
sn=[]
t=0
n=int(input("n number: "))
for i in range(n):
	t+=a;
	sn.append(t)
	t*=10

print(sum(sn))