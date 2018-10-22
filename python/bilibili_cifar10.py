import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from keras.utils import plot_model
model.summary()
plot_model(model,to_file='example.svg')
plot_model(model,to_file='example.png')
lena = mpimg.imread('example.png') 

lena.shape #(512, 512, 3)
plt.imshow(lena) # 
plt.axis('off') # 
# plt.show()
from IPython.display import SVG, display
display(SVG('example.svg'))
