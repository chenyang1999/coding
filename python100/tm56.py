'''
题目：画图，学用circle画圆形。
'''
from tkinter import *
if __name__ == '__main__':

 
	canvas = Canvas(width=800, height=600, bg='yellow')  
	canvas.pack(expand=YES, fill=BOTH)                
	k = 1
	j = 1
	for i in range(0,26):
		canvas.create_oval(310 - k,250 - k,310 + k,250 + k, width=1)
		k += j
		j += 0.3
 
	mainloop()