"""Cifar10 dataset preprocessing and specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import numpy as np
from six import Pickle
from six import urllib
import tensorflow as tf

REMOTE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
LOCAL_DIR = os.path.join("data/cifar10/")
ARCHIVE_NAME = "cifar-10-python.tar.gz"
DATA_DIR = "cifar-10-batches-py/"
TRAIN_BATCHES = ["data_batch_%d" % (i + 1) for i in range(5)]
TEST_BATCHES = ["test_batch"]

IMAGE_SIZE = 32
NUM_CLASSES = 10

def get_params():
    """Return dataset parameters."""
    return {
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
    }

def prepare():
    """Download the cifar dataset."""
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)
    if not os.path.exists(LOCAL_DIR + ARCHIVE_NAME):
        print("Downloading...")
        urllib.request.urlretrieve(REMOTE_URL, LOCAL_DIR + ARCHIVE_NAME)
    if not os.path.exists(LOCAL_DIR + DATA_DIR):
        print("Extracting files...")
        tar = tarfile.open(LOCAL_DIR + ARCHIVE_NAME)
        tar.extractall(LOCAL_DIR)
        tar.close()

def read(split):
    """Create an instance of the dataset object."""
    """An iterator that reads and returns images and labels from cifar."""
    batches = {
        tf.estimator.ModeKeys.TRAIN: TRAIN_BATCHES,
        tf.estimator.ModeKeys.EVAL: TEST_BATCHES
    }[split]

    all_images = []
    all_labels = []

    for batch in batches:
        with open("%s%s%s" % (LOCAL_DIR, DATA_DIR, batch), "rb") as fo:
            dict = cPickle.load(fo)
            images = np.array(dict["data"])
            labels = np.array(dict["labels"])

            num = images.shape[0]
            images = np.reshape(images, [num, 3, IMAGE_SIZE, IMAGE_SIZE])
            images = np.transpose(images, [0, 2, 3, 1])
            print("Loaded %d examples." % num)

            all_images.append(images)
            all_labels.append(labels)

    all_images = np.concatenate(all_images)
    all_labels = np.concatenate(all_labels)

    return tf.contrib.data.Dataset.from_tensor_slices((all_images, all_labels))

def parse(image, label):
    """Parse input record to features and labels."""
    image = tf.to_float(image) / 255.0
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    return {"image": image}, {"label": label}
