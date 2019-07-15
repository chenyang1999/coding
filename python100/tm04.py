'''
题目：输入某年某月某日，判断这一天是这一年的第几天？

程序分析：以3月5日为例，应该先把前两个月的加起来，然后再加上5天即本年的第几天，特殊情况，闰年且输入月份大于2时需考虑多加一天：

程序源代码：
'''

y=int(input("year:"))
m=int(input("month:"))
d=int(input("day:"))

months = (31,28,31,30,31,30,31,31,30,31,30,31)
sum=0
if 0 < m <= 12:
	for i in range(m-1):
		sum+=months[i]
else:
	print ('data error')

sum+=d
leap =0
if (y % 400 == 0) or ((y % 4 == 0) and (y % 100 != 0)):
	leap = 1
if (leap == 1) and (m > 2):
	sum += 1
print ('it is the %dth day.' % sum)