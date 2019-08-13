
# coding: utf-8

# # Mask R-CNN - Train on Eddy Dataset
# 
# The code of the *Eddy* dataset is included below. 

# In[1]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# ## Configurations

# In[2]:


class EddyConfig(Config):
    """Configuration for training on the eddy detection dataset.
    Derives from the base Config class and overrides values specific
    to the eddy detection dataset.
    """
    # Give the configuration a recognizable name
    NAME = "eddies"
    
    # Eddy data only one channel
    IMAGE_CHANNEL_COUNT = 3

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  
    # 1(Non eddy/land/no data) + 2 eddy(anticyclonic eddy, cyclonic eddy)

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
    RPN_NMS_THRESHOLD = 0.8
    
e_config = EddyConfig()
e_config.display()


# ## Notebook Preferences

# In[3]:


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# ## Dataset
# 
# Create a synthetic dataset
# 
# Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
# 
# * load_image()
# * load_mask()
# * image_reference()

# from scipy import io
# 
# mat = io.loadmat("dataset.mat")
# 
# #label = mat['anset'].transpose(2,0,1)
# #ssh = mat['daset'].transpose(2,0,1)
# label = mat['anset']
# ssh = mat['daset']
# 
# print(ssh.shape)

# In[4]:


from scipy import io

matv2 = io.loadmat("eddyset.mat")
for i in matv2:
    print(i)

label = matv2['aset1']
ssh=matv2['dset1']

##### 利用原始数据进行可视化 ######
orig_label = matv2['aset1']       #
orig_ssh=matv2['dset1']           #
###################################

print(label.shape)
print(ssh.shape)
print(ssh[:,:,:].mean())


# In[5]:


import copy

def find_all(label_copy, i, j, num, location, mask):
    location.append([i, j])
    label_copy[i, j] = 0
    mask[i, j] = 1
    
    if 0<i<127 and 0<j<127 and label_copy[i+1, j] == num:
        find_all(label_copy, i+1, j, num, location, mask)
    if 0<i<127 and 0<j<127 and label_copy[i-1, j] == num:
        find_all(label_copy, i-1, j, num, location, mask)
    if 0<i<127 and 0<j<127 and label_copy[i, j+1] == num:
        find_all(label_copy, i, j+1, num, location, mask)
    if 0<i<127 and 0<j<127 and label_copy[i, j-1] == num:
        find_all(label_copy, i, j-1, num, location, mask)
    if 0<i<127 and 0<j<127 and label_copy[i+1, j+1] == num:
        find_all(label_copy, i+1, j+1, num, location, mask)
    if 0<i<127 and 0<j<127 and label_copy[i+1, j-1] == num:
        find_all(label_copy, i+1, j-1, num, location, mask)
    if 0<i<127 and 0<j<127 and label_copy[i-1, j+1] == num:
        find_all(label_copy, i-1, j+1, num, location, mask)
    if 0<i<127 and 0<j<127 and label_copy[i-1, j-1] == num:
        find_all(label_copy, i-1, j-1, num, location, mask)

def eddy_mask(label, num=1):
    width, height = label.shape
#     print(width, height)
    label_copy = copy.deepcopy(label)
    eddy_num = 0
    
    # 声明所有的位置集合，最终找到左上和右下角
    all_location = []
    
    # 声明蒙版列表
    mask_list = []
    
    for i in range(1, width-1):
        for j in range(1, height-1):
            if label_copy[i, j] == num:
                location = []
                mask = np.zeros([height, width])
                eddy_num += 1              
                find_all(label_copy, i, j, num, location, mask)
                all_location.append(location)
                mask_list.append(mask)
#    print("------------------eddy_mask-----------------------")
#    print("number of eddy is %d"%len(all_location))
#    print("------------------eddy_mask-----------------------")
    
    """将所有位置的【左上，右下】返回"""
    eddy_location = []
    hhh = []
    vvv = []
    for loca in all_location:
        hhh = [la[0] for la in loca]
        vvv = [la[1] for la in loca]
    eddy_location.append([[min(hhh), min(vvv)], [max(hhh), max(vvv)]])
    all_mask = np.stack([i for i in mask_list], axis = 2)
    # print(all_mask.shape)
    return eddy_location, all_mask

def mask_check(label, num=1):
    width, height = label.shape
#     print(width, height)
    label_copy = copy.deepcopy(label)
    eddy_num = 0
    
    # 声明所有的位置集合，最终找到左上和右下角
    all_location = []
    
    # 声明蒙版列表
    mask_list = []
    
    for i in range(1, width-1):
        for j in range(1, height-1):
            if label_copy[i, j] == num:
                
                location = []
                mask = np.zeros([height, width])
                eddy_num += 1
                
                find_all(label_copy, i, j, num, location, mask)
#                 print(location)
                all_location.append(location)
                mask_list.append(mask)
#    print("------------------mask_check-----------------------")
#    print("number of eddy is %d"%len(all_location))
#    print(label_copy.max())
#    print("------------------mask_check-----------------------")

    if len(all_location) == 0:
        return 0
    else:
        return 1


# In[6]:


# numpy 三维形状测试

np.arange(20).reshape(2,2,5)
np.arange(20).reshape(1, 4, 5)

arr = np.arange(20).reshape(2,2,5)
np.delete(arr, [1,2], axis = 2)

np.array([1 for i in range(3)]+[2 for i in range(4)])


# In[7]:


"""mask 8 边缘有1或-1：不扫描边缘； 
    mask 36 没有涡旋：将没有涡旋的mask和ssh删掉"""

# 清除没有涡的 ssh 以及 mask
empty = []
for i in range(512):
    if mask_check(label[:, :, i], -1) == 0 or mask_check(label[:, :, i], 1) == 0:
        empty.append(i)

label = np.delete(label, empty, axis=2)
ssh = np.delete(ssh, empty, axis=2)

# 将整体的数值范围缩放至 [0-255]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,255))
for i in range(ssh.shape[2]):
    ssh[:,:,i] = scaler.fit_transform(ssh[:,:,i])
ssh=ssh.astype(int)


# In[8]:


print(label.shape)

location, mask_list = eddy_mask(label[:,:, 80], -1)
# ssh[:, :, 1].reshape(128, 128, 1)
len(location)


# In[9]:


# from scipy import io

# mat = io.loadmat("dataset.mat")

# label = mat['anset']
# ssh = mat['daset']

class EddyDataset(utils.Dataset):
    """Generates the eddy synthetic dataset. The dataset consists of 
    anticyclonic eddy and cyclonic eddy placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    
    by WHX
    """
    # TODO (whx): get data from eddy data set
    
    def load_eddy(self, start, end, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes ---- [add_class(self, source, class_id, class_name)]
        self.add_class("eddy", 1, "anticyclonic")
        self.add_class("eddy", 2, "cyclonic")
        
        for i in range(start, end):
#             bg_color, shapes = self.random_image(height, width)
            self.add_image("eddy", image_id=i, path=None,
                           width=width, height=height)
# --------------------------------------------------------------------------
#         for i in range(count):
# #             bg_color, shapes = self.random_image(height, width)
#             self.add_image("eddy", image_id=i, path=None,
#                            width=width, height=height,
#                            bg_color=bg_color, shapes=shapes)
# --------------------------------------------------------------------------

#     def load_image(self, image_id):
#         """Generate an image from the specs of the given image ID.
#         Typically this function loads the image from a file, but
#         in this case it generates the image on the fly from the
#         specs in image_info.
#         """
#         info = self.image_info[image_id]
#         bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
#         image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
#         image = image * bg_color.astype(np.uint8)
#         for shape, color, dims in info['shapes']:
#             image = self.draw_shape(image, shape, dims, color)
# #             print("load_image form EddyDataset class")
# #         print(image.shape )
#         return image
    
    def load_image(self, image_id):
        """直接返回 image_id 这个通道的ssh的值，reshape成为1个通道
        """
        info = self.image_info[image_id]
        
        # ssh_img = ssh[image_id, :, :].reshape(1,info['height'], info['width']).transpose(1, 2, 0)
        
        ssh_img = ssh[:, :, image_id] #(128,128)
        # ssh_img.reshape(128,128,1)
        return np.stack((ssh_img, ssh_img, ssh_img), axis=2)
    
    def load_orig_image(self, image_id):
        """直接返回 image_id 这个通道的ssh的值，reshape成为1个通道
        """
        info = self.image_info[image_id]
        
        # ssh_img = ssh[image_id, :, :].reshape(1,info['height'], info['width']).transpose(1, 2, 0)
        
        ssh_img = orig_ssh[:, :, image_id] #(128,128)
        # ssh_img.reshape(128,128,1)
        return np.stack((ssh_img, ssh_img, ssh_img), axis=2)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)
            
            
    def load_mask(self, image_id):
        """Generate instance masks for eddies of the given image ID.
        """
        info = self.image_info[image_id]
        
        # image_id 这个样本上的 eddy 总数
        anticy_loc, anticy_mask = eddy_mask(label[ :, :, image_id], 1)
        cy_loc, cy_mask = eddy_mask(label[ :, :, image_id], -1)
        
        mask = np.concatenate((anticy_mask, cy_mask), axis = 2)
        # mask = np.stack((anticy_mask, cy_mask), axis = 0)

        class_ids = np.array([1 for i in range(anticy_mask.shape[2])]
                             +[2 for i in range(cy_mask.shape[2])])
        
        return mask.astype(np.bool), class_ids.astype(np.int32)
        
        

#     def load_mask(self, image_id):
#         """Generate instance masks for shapes of the given image ID.
#         """
#         info = self.image_info[image_id]
#         shapes = info['shapes']
#         count = len(shapes)
#         mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
#         for i, (shape, _, dims) in enumerate(info['shapes']):
#             mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
#                                                 shape, dims, 1)
            
#         print(mask.shape)
#         # Handle occlusions
#         occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
#         for i in range(count-2, -1, -1):
#             mask[:, :, i] = mask[:, :, i] * occlusion
#             occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
#         # Map class names to class IDs.
#         class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
#         return mask.astype(np.bool), class_ids.astype(np.int32)

#     def draw_shape(self, image, shape, dims, color):
#         """Draws a shape from the given specs."""
#         # Get the center x, y and the size s
#         x, y, s = dims
#         if shape == 'anticyclonic':
#             cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
#         elif shape == "cyclonic":
#             cv2.circle(image, (x, y), s, color, -1)

#         return image

#     def random_shape(self, height, width):
#         """Generates specifications of a random eddy that lies within
#         the given height and width boundaries.
#         Returns a tuple of three valus:
#         * The eddy name (anticyclonic, cyclonic)
#         * Shape color: a tuple of 3 values, RGB.
#         * Shape dimensions: A tuple of values that define the shape size
#                             and location. Differs per shape type.
#         """
#         # Eddy
#         shape = random.choice(["anticyclonic", "cyclonic"])
#         # Color
#         color = tuple([random.randint(0, 255) for _ in range(3)])
#         # Center x, y
#         buffer = 20
#         y = random.randint(buffer, height - buffer - 1)
#         x = random.randint(buffer, width - buffer - 1)
#         # Size
#         s = random.randint(buffer, height//4)
#         return shape, color, (x, y, s)

#     def random_image(self, height, width):
#         """Creates random specifications of an image with multiple shapes.
#         Returns the background color of the image and a list of shape
#         specifications that can be used to draw the image.
#         """
#         # Pick random background color
#         bg_color = np.array([random.randint(0, 255) for _ in range(3)])
#         # Generate a few random shapes and record their
#         # bounding boxes
#         shapes = []
#         boxes = []
#         N = random.randint(1, 4)
#         for _ in range(N):
#             shape, color, dims = self.random_shape(height, width)
#             shapes.append((shape, color, dims))
#             x, y, s = dims
#             boxes.append([y-s, x-s, y+s, x+s])
#         # Apply non-max suppression wit 0.3 threshold to avoid
#         # shapes covering each other
#         keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
#         shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
#         return bg_color, shapes


# In[10]:


# whx
# Training dataset
eddy_train = EddyDataset()
eddy_train.load_eddy(0, 350, e_config.IMAGE_SHAPE[0], e_config.IMAGE_SHAPE[1])
eddy_train.prepare()

# Validation dataset
eddy_val = EddyDataset()
eddy_val.load_eddy(350, 430, e_config.IMAGE_SHAPE[0], e_config.IMAGE_SHAPE[1])
eddy_val.prepare()


# In[11]:


eddy_train.image_info[1]
a = np.arange(12).reshape(3,4)
a


# In[12]:


a = a.transpose(1,0)
a


# In[13]:


b = np.arange(4).reshape(2, 2).reshape(2,2,1).reshape(2,2).reshape(2,2)
b


# In[14]:


def display_eddy(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        # print(image.shape)
        # X, Y = np.meshgrid(np.arange(128), np.arange(128))
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.contourf(image, 8, alpha=.50, cmap=plt.cm.winter)
        # plt.imshow(image.astype(np.uint8), cmap=cmap,plt.cm.hot
                   #norm=norm, interpolation=interpolation)
        i += 1
    plt.show()

def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_eddy(to_display, titles=titles, cols=limit + 1, cmap="PuBu")


# In[15]:


np.arange(20).reshape(2,2,5)[:,:,1].shape


# In[16]:


def image_save(image):
    plt.contourf(image, 8, alpha=.50, cmap=plt.cm.winter)
    fig = plt.gcf()
    fig.set_size_inches(1.28/3,1.28/3) #dpi = 300, output = 128*128 pixels
    
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)

    fig.savefig('./eddy.jpg', format='jpg', transparent=False, dpi=300, bbox_inches=None, pad_inches=0)


# In[17]:


# 利用原始数据进行可视化

orig_label = matv2['aset1']
orig_ssh=matv2['dset1']

orig_label = np.delete(orig_label, empty, axis=2)
orig_ssh = np.delete(orig_ssh, empty, axis=2)

# Load and display random samples
eddy_image_ids = np.random.choice(eddy_train.image_ids, 3)
# eddy_image_ids = np.hstack((eddy_image_ids, np.arange(2)))
print(eddy_image_ids)
for eddy_image_id in eddy_image_ids:
    image = eddy_train.load_orig_image(eddy_image_id)
    print(type(image))
    print(image.shape)
    mask, class_ids = eddy_train.load_mask(eddy_image_id)
    print(type(mask))
    print(mask.shape)
    #display_top_masks(image[:,:,1], mask, class_ids, eddy_train.class_names)
image = eddy_train.load_orig_image(eddy_image_ids[-1])
print("---------------")
image_save(image[:,:,1])


# In[18]:


# 本方法通过使用单通道的数据将


def ssh_convert_to_image(ssh_data):
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(1.28/3, 1.28/3) #dpi = 300, output = 700*700 pixels
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.contourf(ssh_data[:,:,0], 9, alpha=.50, cmap=plt.cm.winter)
    plt.savefig('temp_ssh.png', transparent=False, dpi=300, pad_inches=0)

    ssh_img = Image.open('temp_ssh.png').convert("RGB")
    
    ssh_img = ssh_img.resize((128, 128))
    ssh_img = ssh_img.transpose(Image.FLIP_TOP_BOTTOM)
    ssh_img_array = np.array(ssh_img)
    print(ssh_img_array.shape)
    return ssh_img_array


# In[19]:


ssh_data = eddy_train.load_orig_image(30)
img_data = ssh_convert_to_image(ssh_data)

print(img_data.shape)


# In[20]:


# whx
# Load and display random samples
# eddy_image_ids = np.random.choice(eddy_train.image_ids, 4)
# eddy_image_ids = np.hstack((eddy_image_ids, np.arange(2)))
print(eddy_image_ids)
for eddy_image_id in eddy_image_ids:
    image = eddy_train.load_image(eddy_image_id)
    print(type(image))
    print(image.shape)
    mask, class_ids = eddy_train.load_mask(eddy_image_id)
    print(type(mask))
    print(mask.shape)
    #display_top_masks(image[:,:,1], mask, class_ids, eddy_train.class_names)


# ## Create Model

# In[21]:


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=e_config,
                          model_dir=MODEL_DIR)


# In[22]:


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["conv1", "bn_conv1", "mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


# ## Training
# 
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.

# In[23]:


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(eddy_train, eddy_val, 
            learning_rate=e_config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')
model.train(eddy_train, eddy_val, 
            learning_rate=e_config.LEARNING_RATE / 10,
            epochs=1, 
            layers="all")


# In[23]:


# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(eddy_train, eddy_val, 
            learning_rate=e_config.LEARNING_RATE / 10,
            epochs=1, 
            layers="all")


# In[24]:


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_eddy.h5")
# model.keras_model.save_weights(model_path)


# ## Detection

# In[24]:


class InferenceConfig(EddyConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# In[34]:


# Test on a random image
# image_id = random.choice(eddy_val.image_ids)
image_id = 60
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(eddy_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)
print(original_image.shape)
print(gt_bbox.shape)
print(gt_mask.shape)
print(gt_class_id.shape)

original_image = eddy_val.load_orig_image(image_id)
original_image1 = ssh_convert_to_image(original_image)

#visualize.display_instances(original_image1, gt_bbox, gt_mask, gt_class_id, 
                           # eddy_train.class_names, figsize=(8, 8))


# In[36]:


train_ssh = eddy_train.load_image(image_id)
results = model.detect([train_ssh], verbose=1)

r = results[0]
#print(r)
confidence = 0.75
scores = r['scores']
del_index = []

for i in range(len(scores)):
    if scores[i] < confidence:
        del_index.append(i)
#print(del_index)

r['rois'] = np.delete(r['rois'],del_index,axis = 0)
r['masks'] = np.delete(r['masks'],del_index,axis = -1)
r['class_ids'] = np.delete(r['class_ids'],del_index,axis = 0)
r['scores'] = np.delete(r['scores'],del_index,axis = 0)


#visualize.display_instances(original_image1, r['rois'], r['masks'], r['class_ids'], 
                            #eddy_val.class_names, r['scores'], ax=get_ax())


# In[37]:


# fake EddyNet

train_ssh = eddy_train.load_image(image_id)
results = model.detect([train_ssh], verbose=1)

r = results[0]
# print(r)
confidence = 0.80
scores = r['scores']
del_index = []

for i in range(len(scores)):
    if scores[i] < confidence:
        del_index.append(i)
#print(del_index)

r['rois'] = np.delete(r['rois'],del_index,axis = 0)
r['masks'] = np.delete(r['masks'],del_index,axis = -1)
r['class_ids'] = np.delete(r['class_ids'],del_index,axis = 0)
r['scores'] = np.delete(r['scores'],del_index,axis = 0)


#visualize.display_instances(original_image1, r['rois'], r['masks'], r['class_ids'], 
                            #eddy_val.class_names, r['scores'], ax=get_ax())


# ## Evaluation

# In[39]:


# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(eddy_val.image_ids, 150)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =        modellib.load_image_gt(eddy_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))


# # anchor scale 没改之前
# mAP:  0.0811868091779902
# 
# epoch：
# # anchor scale 更改之后
# mAP:  0.2939656484731303
# 
# epoch： 10+20

# In[38]:


70----52
80----55


# image_id = 45 confidence = 0.80 涡旋预测完全正确
