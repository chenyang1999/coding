'''
题目：时间函数举例2。
'''
if __name__ == '__main__':
	import time
	start = time.time()
	for i in range(10000000):
#		print (i)
		pass
	end = time.time()
 
	print (end - start)