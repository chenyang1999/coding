from matplotlib import pylab
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tkinter import font
#font.set_size(20)
def initialCondition(x):
	return 37.0
xArray = np.linspace(0,1.0,50)
yArray = map(initialCondition, xArray)
plt.figure(figsize = (12,6))
plt.plot(xArray, yArray)
plt.xlabel('$x$', fontsize = 15)
plt.ylabel('$f(x)$', fontsize = 15)
plt.title(u'一维热传导方程初值条件', fontproperties = font)
