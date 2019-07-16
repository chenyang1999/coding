'''
两个 3 行 3 列的矩阵，实现其对应位置的数据相加，并返回一个新矩阵：
'''
X = [[12,7,3],
	[4 ,5,6],
	[7 ,8,9]]
 
Y = [[5,8,1],
	[6,7,3],
	[4,5,9]]
 
result = [[0,0,0],
		 [0,0,0],
		 [0,0,0]]
# 迭代输出行
for i in range(len(X)):
	# 迭代输出列
	for j in range(len(X[0])):
		 result[i][j] = X[i][j] + Y[i][j]
 
for r in result:
	print(r)