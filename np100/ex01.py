import numpy as np
a=np.array([1,2,3,4])
print(a,a.dtype,a.shape)

a.shape=2,2
print(a,a.dtype,a.shape)

d=a.reshape(2,2)
print(d)

a[1,1]=100
print(a)
print(d)

b= np.array([[1, 2, 3, 4],[4, 5, 6, 7], [7, 8, 9, 10]], dtype=np.float)
print(b)

c= np.arange(0,10,1)
print(c)

d= np.logspace(0, 2, 20)
print(d)

s = "abcdefgh"
print(np.fromstring(s, dtype=np.int16))

def func(i):
	return i%4+1
print(np.fromfunction(func, (10,)))

def func2(i, j):
	return (i+1) * (j+1)
print(np.fromfunction(func2, (9,9)))