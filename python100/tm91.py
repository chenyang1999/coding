'''
题目：时间函数举例1。
'''
if __name__ == '__main__':
	import time
	print (time.ctime(time.time()))
	print (time.asctime(time.localtime(time.time())))
	print (time.asctime(time.gmtime(time.time())))