def gaussSeidel(A, b, x, N, tol):
	maxIterations = 10000
	xprev = [0.0 for i in range(N)]
	for i in range(maxIterations):
		for j in range(N):
			xprev[j] = x[j]
		for j in range(N):
			summ = 0.0
			for k in range(N):
				if (k != j):
					summ = summ + A[j][k] * x[k]
			x[j] = (b[j] - summ) / A[j][j]
		diff1norm = 0.0
		oldnorm = 0.0
		for j in range(N):
			diff1norm = diff1norm + abs(x[j] - xprev[j])
			oldnorm = oldnorm + abs(xprev[j])  
		if oldnorm == 0.0:
			oldnorm = 1.0
		norm = diff1norm / oldnorm
		if (norm < tol) and i != 0:
			print("Sequence converges to [", end="")
			for j in range(N - 1):
				print(x[j], ",", end="")
			print(x[N - 1], "]. \n迭代次数: ", i + 1 )
			return
	print("Doesn't converge.")

guess = [0.0, 0.0 , 0.0 ]
mx = [[8,-3,2],[4,11,-1],[6,3,12]]
mr = [20,33,36]

gaussSeidel(mx,mr, guess,3, 0.000001)