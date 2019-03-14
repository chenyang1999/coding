import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange,interp1d
def f(X):
	return 1 / (X ** 2 + 1)

x = np.linspace(-10, 10, num=100, endpoint=True)
y = f(x)
f = lagrange(x, y)
f2 = interp1d(x, y, kind='cubic')

xnew = np.linspace(-10, 10, num=11, endpoint=True)

plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()
