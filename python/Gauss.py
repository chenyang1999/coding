
# -*- coding: UTF-8 -*- 
class ClassGauss(object):
	def __init__(self, a, b):	
		super(ClassGauss, self).__init__()
		self.a = a
		self.b = b
		self.n = len(self.b)
 
	def max(self, max_i, max_v, i, j):# get max
		a = self.a
		abs_of_a = abs(a[i][j])
 
		if max_v < abs_of_a:
			max_v = abs_of_a
			max_i = i
		return max_i, max_v
 
	def swap(self, ai, j):# change line
		a = self.a
		b = self.b
		n = self.n
		for i in range(0, n):
			temp = a[ai][i]
			a[ai][i] = a[j][i]
			a[j][i] = temp
 
			tempb = b[ai]
			b[ai] = b[j]
			b[j] = tempb
 
	def gauss(self):
		n = self.n
		max_i = 0 # line num of max value
		max_v = m = self.a[0][0]
		for j in range(0, n-1): 
			for i in range(j, n): 
				max_i, max_v = self.max(max_i, max_v, i, j)
			if max_v == 0:
				raise ValueError('no unique solution')
			if debug:
				print('max_v = %f' % max_v)
				print('max_i = %f , j = %f' % (max_i, j))
			if max_i != j:
				# jiaohuan ai hang he ajhang
				self.swap(max_i, j)
			if debug:
				print('SWAP*******')
				print(self.a)
				print(self.b)
			for p in range(j+1, n):
				l =  a[p][j] / a[j][j]
				# print('l = %f' % (l))
				b[p] -= l * b[j]
				for q in range(j, n):
					a[p][q] -= l * a[j][q]
			if debug:		
				print('CAL_a******')
				print(self.a)
				print(self.b)
			max_v = m
		if debug:
			print("************************")
			print(self.a)
			print(self.b)
		self.calculate()
 
	def calculate(self):
		n = self.n - 1
		xn = b[n] / a[n][n]
		print('xn = %f'% xn)
					
    
   
if __name__ == '__main__':
	# example1
    a = [[0.012, 0.0100, 0.1670],
         [1.000, 0.8334, 5.910],
         [3200, 1200, 4.200]]
    b = [0.6781, 12.10, 983.3]
    debug = False
    # example2
    # a = [[2.00, -4.00, 2.00],[1.00, 2.00, 3.00],[-3.00, -2.00, 5.00]]
    # b = [2.00, 3.00, 1.00]
    g = ClassGauss(a,b)
    g.gauss()