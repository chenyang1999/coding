from scipy.misc import *
def sample_generation(iter_num):
	global generator
	sample_noise = np.random.normal(loc=0.0, scale=1.0, size=[9, 100])
	generated_images = generator.predict(sample_noise)
	generated_images = unnormalize_display(generated_images)
	for image_idx in range(len(generated_images)):
		plt.subplot(3, 3, image_idx+1)
		#generated_image = unnormalize_display(train_data[image_idx]).transpose(1,2,0)
		generated_image = generated_images[image_idx].transpose(1,2,0)
		print(generated_image.shape)
		imsave("samples/mnist/64_%d.png"%i,generated_image)
		plt.imshow(generated_image)
	#plt.show(block=False)
	plt.savefig('Run1/results/sample_'+str(iter_num)+'.png')
	#time.sleep(3)
	#plt.close('all')

for image_idx in range(len(generated_images)):
	plt.subplot(3, 3, image_idx+1)
	#generated_image = unnormalize_display(train_data[image_idx]).transpose(1,2,0)
	print(generated_image.shape)
	generated_image = generated_images[image_idx].transpose(1,2,0)
	print(generated_image.shape)
	imsave("images/cifar10_%d.png" % epoch,generated_image)
