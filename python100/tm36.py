'''
题目：求100之内的素数。
'''
for i in range(2,101):
	bj=1
	for j in range(2,i):
		if (i%j==0):
			bj=0
			break
	if(bj):print(i)