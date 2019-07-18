'''
题目：有两个磁盘文件A和B,各存放一行字母,要求把这两个文件中的信息合并(按字母顺序排列), 输出到一个新文件C中。
'''
with open('test1.txt') as f1, open('test2.txt') as f2, open('2.txt', 'w') as f3:
	for a in f1:
		b = f2.read()
	c = list(a + b)
	c.sort()
	d = ''
	d = d.join(c)
	f3.write(d)