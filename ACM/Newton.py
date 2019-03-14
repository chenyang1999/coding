import sympy
import numpy as np
from matplotlib import pyplot as plt


def f(X):
	return 1 / (X ** 2 + 1)


def ff(X=list()):
	if len(X) < 2:
		raise ValueError('X\'s length must be bigger than 2')
	ans = 0
	for i in range(len(X)):
		temp = 1.0
		for j in range(len(X)):
			if j == i:
				continue
			temp *= (X[i] - X[j])
		ans += (f(X[i]) / temp)
	return ans


def draw():
	plt.rcParams['font.sans-serif'] = ['SimHei']
	plt.rcParams['axes.unicode_minus'] = False
	X = np.linspace(-10, 10, 100)

	TargetY = f(X)
	GetY = [Px.subs(x, i) for i in X]

	plt.plot(X, TargetY, label=r'$\frac{1}{x^2+1}$')
	plt.plot(X, GetY, label='$L(x)$')
	plt.legend()
	plt.show()


def generatePx(DataX):
	ans = f(DataX[0])
	if len(DataX) == 1:
		return ans
	else:
		temp = 1
		for i in range(len(DataX) - 1):
			temp *= (x - DataX[i])
			ans += ff(DataX[:i + 2]) * temp
		return ans


if __name__ == '__main__':
	x = sympy.symbols('x')

	DataX = np.linspace(-10, 10, 11)  # 插值点

	Px = sympy.expand(generatePx(DataX))
	draw()