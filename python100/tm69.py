'''
题目：有n个人围成一圈，顺序排号。从第一个人开始报数（从1到3报数），凡报到3的人退出圈子，问最后留下的是原来第几号的那位。
'''
class Solution:
	def LastRemaining_Solution(self, n, m):
		# write code here
		# 用列表来模拟环，新建列表range(n)，是n个小朋友的编号
		if not n or not m:
			return -1
		lis = list(range(n))
		i = 0
		while len(lis)>1:
			i = (m-1 + i)%len(lis) # 递推公式
			lis.pop(i)
		return lis[0]
a=Solution
print(a.LastRemaining_Solution(a, 34, 3)+1)