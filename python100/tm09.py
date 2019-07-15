#题目：暂停一秒输出。
#
#程序分析：使用 time 模块的 sleep() 函数。
import time
 
myD = {1: 'a', 2: 'b',3:'c',4:'d'}
for key, value in dict.items(myD):
	print( key, value)
	time.sleep(1) # 暂停 1 秒