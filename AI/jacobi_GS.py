import numpy as np

def Gauss_Sheidel(A,b):
	n1 = len(A)
	n2 = len(A.T)
	x = np.ones([n1,1])
	N=0
	if (n1 != n2):
		print( "The input matrix is not squre matrix")
	else :
		B = np.tril(A)
		U = np.triu(A,k=1)
		G = -np.dot(np.linalg.inv(B),U)
		d = np.dot(np.linalg.inv(B),b)
		x1 = x
		x2 = np.dot(G,x)+d
		while (np.linalg.norm(x1-x2,np.inf) > 10e-10):
			x1= x2
			x2 = np.dot(G,x1)+d
			N = N + 1
		print ("迭代次数=",N)
	return x1

def jacobi (A,b):
	n1 = len(A)
	n2 = len(A.T)
	x = np.ones([n1,1])
	N=0
	if (n1 != n2):
		print ("The input matrix is not squre matrix")
	else :
		B = np.zeros([n1,n1])
		D = np.zeros([n1,n1])
		d = np.ones([n1,1])
		B = A.copy()
		for i in range(n1):
			B[i,i] = 0.0
			D[i,i] = A[i,i].copy()
		B = np.dot(-np.linalg.inv(D),B)
		d = np.dot(np.linalg.inv(D),b)
		x1 = x
		x2 = np.dot(B,x)+d
		while (np.linalg.norm(x1-x2,np.inf) > 10e-10):
			x1= x2
			x2 = np.dot(B,x1)+d
			N = N + 1
		print("迭代次数=",N)
	return x2

x=np.array([[5,1,-1,-2],[2,8,1,3],[1,-2,-4,-1],[-1,3,2,7]])
print("Jacobi")
y=np.array([[-2],[-6],[6],[12]])
a=jacobi(x,y)
print("解:",a)

print("GS")
b=Gauss_Sheidel(x,y)
print( b)
