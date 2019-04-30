from imgaug import augmenters as iaa
from PIL import Image
import numpy as np
import torch
from torchvision import transforms,datasets
class BLUR(object):

	def __init__(self, param=0.5):
		self.param=0.5
		self.seq=iaa.Sequential([iaa.Snowflakes(0.4),
		iaa.Fog()])

	def __call__(self, img):

		img=np.array(img)
		img=self.seq.augment_images([img])
		img=Image.fromarray((img[0]))
		return img
	

	def __repr__(self):
		return self.__class__.__name__ + '()'
topil=transforms.ToPILImage()
tsfm=transforms.Compose([
				transforms.Resize((232, 232)),  # force resize
				#transforms.RandomCrop(224),
				#transforms.RandomHorizontalFlip(0.05),
				#transforms.RandomGrayscale(0.02),
				#transforms.RandomRotation(10),
				transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
				 BLUR(),
				transforms.ToTensor()])
				
image_datasets=datasets.ImageFolder('/home/test',tsfm)
dataloader=torch.utils.data.DataLoader(image_datasets,num_workers=2,batch_size=4)
index=0
for batch_idx, data in enumerate(dataloader):
	inputs, labels = data
	for i in range(4):
		topil(inputs[i]).save('/home/output/'+str(index)+'.jpg')
		index+=1
		
