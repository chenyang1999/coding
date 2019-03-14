import tensorflow as tf
"""MNIST dataset preprocessing and specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy as np
import os
from six.moves import urllib
import struct
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

REMOTE_URL = "http://yann.lecun.com/exdb/mnist/"
LOCAL_DIR = "data/mnist/"
TRAIN_IMAGE_URL = "train-images-idx3-ubyte.gz"
TRAIN_LABEL_URL = "train-labels-idx1-ubyte.gz"
TEST_IMAGE_URL = "t10k-images-idx3-ubyte.gz"
TEST_LABEL_URL = "t10k-labels-idx1-ubyte.gz"

IMAGE_SIZE = 28
NUM_CLASSES = 10

def get_params():
    """Dataset params."""
    return {
        "num_classes": NUM_CLASSES,
    }

def prepare():
    """This function will be called once to prepare the dataset."""
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)
    for name in [
            TRAIN_IMAGE_URL,
            TRAIN_LABEL_URL,
            TEST_IMAGE_URL,
            TEST_LABEL_URL]:
        if not os.path.exists(LOCAL_DIR + name):
            urllib.request.urlretrieve(REMOTE_URL + name, LOCAL_DIR + name)

def read(split):
    """Create an instance of the dataset object."""
    image_urls = {
        tf.estimator.ModeKeys.TRAIN: TRAIN_IMAGE_URL,
        tf.estimator.ModeKeys.EVAL: TEST_IMAGE_URL
    }[split]
    label_urls = {
        tf.estimator.ModeKeys.TRAIN: TRAIN_LABEL_URL,
        tf.estimator.ModeKeys.EVAL: TEST_LABEL_URL
    }[split]

    with gzip.open(LOCAL_DIR + image_urls, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(num * rows * cols), dtype=np.uint8)
        images = np.reshape(images, [num, rows, cols, 1])
        print("Loaded %d images of size [%d, %d]." % (num, rows, cols))

    with gzip.open(LOCAL_DIR + label_urls, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(num), dtype=np.int8)
        print("Loaded %d labels." % num)

    return tf.contrib.data.Dataset.from_tensor_slices((images, labels))

def parse(image, label):
    """Parse input record to features and labels."""
    image = tf.to_float(image) / 255.0
    label = tf.to_int64(label)
    return {"image": image}, {"label": label}
"""Cifar100 dataset preprocessing and specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import numpy as np
from six.moves import cPickle
from six.moves import urllib
import tensorflow as tf

REMOTE_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
LOCAL_DIR = os.path.join("data/cifar100/")
ARCHIVE_NAME = "cifar-100-python.tar.gz"
DATA_DIR = "cifar-100-python/"
TRAIN_BATCHES = ["train"]
TEST_BATCHES = ["test"]

IMAGE_SIZE = 32
NUM_CLASSES = 100

def get_params():
    """Return dataset parameters."""
    return {
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
    }

def prepare():
    """Download the cifar 100 dataset."""
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
            labels = np.array(dict["fine_labels"])

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
"""Cifar10 dataset preprocessing and specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import numpy as np
from six.moves import cPickle
from six.moves import urllib
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
features = tf.layers.batch_normalization(features)
"""Simple convolutional neural network classififer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.flags.FLAGS

def get_params():
    """Model params."""
    return {
        "drop_rate": 0.5
    }

def model(features, labels, mode, params):
    """CNN classifier model."""
    images = features["image"]
    labels = labels["label"]

    tf.summary.image("images", images)

    drop_rate = params.drop_rate if mode == tf.estimator.ModeKeys.TRAIN else 0.0

    features = images
    for i, filters in enumerate([32, 64, 128]):
        features = tf.layers.conv2d(
            features, filters=filters, kernel_size=3, padding="same",
            name="conv_%d" % (i + 1))
        features = tf.layers.max_pooling2d(
            inputs=features, pool_size=2, strides=2, padding="same",
            name="pool_%d" % (i + 1))

    features = tf.contrib.layers.flatten(features)

    features = tf.layers.dropout(features, drop_rate)
    features = tf.layers.dense(features, 512, name="dense_1")

    features = tf.layers.dropout(features, drop_rate)
    logits = tf.layers.dense(features, params.num_classes, activation=None,
                             name="dense_2")

    predictions = tf.argmax(logits, axis=1)

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)

    return {"predictions": predictions}, loss

def eval_metrics(unused_params):
    """Eval metrics."""
    return {
        "accuracy": tf.contrib.learn.MetricSpec(tf.metrics.accuracy)
    }
features = tf.layers.conv2d(
    features,
    filters=64,
    kernel_size=3,
    padding="same",
    name="conv2d/1")
loss = tf.losses.sparse_softmax_cross_entropy(
    labels=labels, logits=logits)
features = tf.layers.dense(features, units=64, name="dense/1")
features = tf.layers.dropout(features, rate=0.5)
features = tf.layers.max_pooling2d(
    features, pool_size=2, strides=2, padding="same")
"""MNIST dataset preprocessing and specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy as np
import os
from six.moves import urllib
import struct
import tensorflow as tf

REMOTE_URL = "http://yann.lecun.com/exdb/mnist/"
LOCAL_DIR = "data/mnist/"
TRAIN_IMAGE_URL = "train-images-idx3-ubyte.gz"
TRAIN_LABEL_URL = "train-labels-idx1-ubyte.gz"
TEST_IMAGE_URL = "t10k-images-idx3-ubyte.gz"
TEST_LABEL_URL = "t10k-labels-idx1-ubyte.gz"

IMAGE_SIZE = 28
NUM_CLASSES = 10

def get_params():
    """Dataset params."""
    return {
        "num_classes": NUM_CLASSES,
    }

def prepare():
    """This function will be called once to prepare the dataset."""
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)
    for name in [
            TRAIN_IMAGE_URL,
            TRAIN_LABEL_URL,
            TEST_IMAGE_URL,
            TEST_LABEL_URL]:
        if not os.path.exists(LOCAL_DIR + name):
            urllib.request.urlretrieve(REMOTE_URL + name, LOCAL_DIR + name)

def read(split):
    """Create an instance of the dataset object."""
    image_urls = {
        tf.estimator.ModeKeys.TRAIN: TRAIN_IMAGE_URL,
        tf.estimator.ModeKeys.EVAL: TEST_IMAGE_URL
    }[split]
    label_urls = {
        tf.estimator.ModeKeys.TRAIN: TRAIN_LABEL_URL,
        tf.estimator.ModeKeys.EVAL: TEST_LABEL_URL
    }[split]

    with gzip.open(LOCAL_DIR + image_urls, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(num * rows * cols), dtype=np.uint8)
        images = np.reshape(images, [num, rows, cols, 1])
        print("Loaded %d images of size [%d, %d]." % (num, rows, cols))

    with gzip.open(LOCAL_DIR + label_urls, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(num), dtype=np.int8)
        print("Loaded %d labels." % num)

    return tf.contrib.data.Dataset.from_tensor_slices((images, labels))
def resnet_block(features, bottleneck, out_filters, training):
    """Residual block."""
    with tf.variable_scope("input"):
        original = features
        features = tf.layers.conv2d(features, bottleneck, 1, activation=None)
        features = tf.layers.batch_normalization(features, training=training)
        features = tf.nn.relu(features)

    with tf.variable_scope("bottleneck"):
        features = tf.layers.conv2d(
            features, bottleneck, 3, activation=None, padding="same")
        features = tf.layers.batch_normalization(features, training=training)
        features = tf.nn.relu(features)

    with tf.variable_scope("output"):
        features = tf.layers.conv2d(features, out_filters, 1)
        in_dims = original.shape[-1].value
        if in_dims != out_filters:
            original = tf.layers.conv2d(features, out_filters, 1, activation=None,
                name="proj")
        features += original
    return features
features = tf.layers.separable_conv2d(
    features,
    filters=64,
    kernel_size=3,
    padding="same",
    name="conv2d_separable/1")
"""This module handles training and evaluation of a neural network model.

Invoke the following command to train the model:
python -m trainer --model=cnn --dataset=mnist

You can then monitor the logs on Tensorboard:
tensorboard --logdir=output"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_string("model", "", "Model name.")
tf.flags.DEFINE_string("dataset", "", "Dataset name.")
tf.flags.DEFINE_string("output_dir", "", "Optional output dir.")
tf.flags.DEFINE_string("schedule", "train_and_evaluate", "Schedule.")
tf.flags.DEFINE_string("hparams", "", "Hyper parameters.")
tf.flags.DEFINE_integer("num_epochs", 100000, "Number of training epochs.")
tf.flags.DEFINE_integer("save_summary_steps", 10, "Summary steps.")
tf.flags.DEFINE_integer("save_checkpoints_steps", 10, "Checkpoint steps.")
tf.flags.DEFINE_integer("eval_steps", None, "Number of eval steps.")
tf.flags.DEFINE_integer("eval_frequency", 10, "Eval frequency.")

FLAGS = tf.flags.FLAGS

MODELS = {
    # This is a dictionary of models, the keys are model names, and the values
    # are the module containing get_params, model, and eval_metrics.
    # Example: "cnn": cnn
}

DATASETS = {
    # This is a dictionary of datasets, the keys are dataset names, and the
    # values are the module containing get_params, prepare, read, and parse.
    # Example: "mnist": mnist
}

HPARAMS = {
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "decay_steps": 10000,
    "batch_size": 128
}

def get_params():
    """Aggregates and returns hyper parameters."""
    hparams = HPARAMS
    hparams.update(DATASETS[FLAGS.dataset].get_params())
    hparams.update(MODELS[FLAGS.model].get_params())

    hparams = tf.contrib.training.HParams(**hparams)
    hparams.parse(FLAGS.hparams)

    return hparams

def make_input_fn(mode, params):
    """Returns an input function to read the dataset."""
    def _input_fn():
        dataset = DATASETS[FLAGS.dataset].read(mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.repeat(FLAGS.num_epochs)
            dataset = dataset.shuffle(params.batch_size * 5)
        dataset = dataset.map(
            DATASETS[FLAGS.dataset].parse, num_threads=8)
        dataset = dataset.batch(params.batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
    return _input_fn

def make_model_fn():
    """Returns a model function."""
    def _model_fn(features, labels, mode, params):
        model_fn = MODELS[FLAGS.model].model
        global_step = tf.train.get_or_create_global_step()
        predictions, loss = model_fn(features, labels, mode, params)

        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            def _decay(learning_rate, global_step):
                learning_rate = tf.train.exponential_decay(
                    learning_rate, global_step, params.decay_steps, 0.5,
                    staircase=True)
                return learning_rate

            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=global_step,
                learning_rate=params.learning_rate,
                optimizer=params.optimizer,
                learning_rate_decay_fn=_decay)

        return tf.contrib.learn.ModelFnOps(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)

    return _model_fn

def experiment_fn(run_config, hparams):
    """Constructs an experiment object."""
    estimator = tf.contrib.learn.Estimator(
        model_fn=make_model_fn(), config=run_config, params=hparams)
    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=make_input_fn(tf.estimator.ModeKeys.TRAIN, hparams),
        eval_input_fn=make_input_fn(tf.estimator.ModeKeys.EVAL, hparams),
        eval_metrics=MODELS[FLAGS.model].eval_metrics(hparams),
        eval_steps=FLAGS.eval_steps,
        min_eval_frequency=FLAGS.eval_frequency)

def main(unused_argv):
    """Main entry point."""
    if FLAGS.output_dir:
        model_dir = FLAGS.output_dir
    else:
        model_dir = "output/%s_%s" % (FLAGS.model, FLAGS.dataset)

    DATASETS[FLAGS.dataset].prepare()

    session_config = tf.ConfigProto()
    session_config.allow_soft_placement = True
    session_config.gpu_options.allow_growth = True
    run_config = tf.contrib.learn.RunConfig(
        model_dir=model_dir,
        save_summary_steps=FLAGS.save_summary_steps,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        save_checkpoints_secs=None,
        session_config=session_config)

    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=FLAGS.schedule,
        hparams=get_params())

if __name__ == "__main__":
    tf.app.run()
features = tf.layers.conv2d_transpose(
    features,
    filters=64,
    kernel_size=3,
    padding="same",
    name="conv2d_transpose/1")
import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            model.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
for params in net.parameters():
    params.require_grad = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

best_acc1 = 0


def main():
    global args, best_acc1
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

F.conv2d(, weight, bias=None, stride=1, padding=0)  
F.pairwise_distance(x1, x2)
F.alpha_dropout(input, p=0.5)   
F.bilinear(input1, input2, weight)
F.mse_loss(input, target)
F.multilabel_soft_margin_loss(input, target)
criterion = nn.NLLLoss()
conv = nn.Conv1d(in_channel, out_channel, groups=1, bias=True, kernel_size=2, padding=0, stride=1)
norm = nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
class Bottleneck(nn.Module):
    # see https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out
class BasicBlock(nn.Module):
    # see https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
F.embedding_bag(input, weight)
F.upsample_bilinear(input, size=None, scale_factor=None)
for params in net.parameters():
    params.require_grad = False
class MyFunction(torch.autograd.Function):
    """Some Information about MyFunction"""

    @staticmethod
    def forward(ctx, input):

        return

    @staticmethod
    def backward(ctx, grad_output)

        return

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Bilinear') != -1:
        nn.init.orthogonal_(gain=1, tensor=m.weight)
        if m.bias: nn.init.normal_(mean=0, std=1, tensor=m.bias)

    elif classname.find('Conv') != -1:
        nn.init.normal_(mean=0, std=1, tensor=m.weight)
        if m.bias: nn.init.normal_(mean=0, std=1, tensor=m.bias)

    elif classname.find('BatchNorm') != -1 or classname.find('GroupNorm') != -1 or classname.find('LayerNorm') != -1:
        nn.init.normal_(mean=0, std=1, tensor=m.weight)
        nn.init.normal_(mean=0, std=1, tensor=m.bias)

    elif classname.find('Cell') != -1:
        nn.init.uniform_(a=0, b=1, tensor=m.weight_hh)
        nn.init.uniform_(a=0, b=1, tensor=m.weight_ih)
        nn.init.uniform_(a=0, b=1, tensor=m.bias_hh)
        nn.init.uniform_(a=0, b=1, tensor=m.bias_ih)

    elif classname.find('RNN') != -1 or classname.find('LSTM') != -1 or classname.find('GRU') != -1:
        for w in m.all_weights:
            nn.init.uniform_(a=0, b=1, tensor=w[2].data)
            nn.init.uniform_(a=0, b=1, tensor=w[3].data)
            nn.init.uniform_(a=0, b=1, tensor=w[0].data)
            nn.init.uniform_(a=0, b=1, 
            tensor=w[1].data)

    if classname.find('Embedding') != -1:
        nn.init.orthogonal_(gain=1, tensor=m.weight)

net.apply(init_weights)
nonlin = nn.Hardtanh(min_val=-1, max_val=1, inplace=False, min_value=None, max_value=None)
distance = nn.PairwiseDistance()
F.multilabel_soft_margin_loss(input, target)    
class BasicBlock(nn.Module):
    # see https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
class Bottleneck(nn.Module):
    # see https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out
criterion = nn.MultiLabelMarginLoss()
criterion = nn.MarginRankingLoss.add_module()
criterion = nn.MSELoss()
class MyModule(nn.Module):
    """Some Information about MyModule"""
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self, x):

        return x
optimizer = torch.optim.ASGD(net.parameters(), lr=1e-2)
# loop over the dataset multiple times
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        .zero_grad()

        # forward + backward + optimize
        outputs = w(inputs)
        loss = q(outputs, labels)
        loss.backward()
        .step()

        running_loss += loss.item()

    print('Loss: {}'.format(running_loss)

print('Finished Training')
for params in net.parameters():
    params.require_grad = True
datasets = CollabFilteringDataset.from_csv(csv_name)
datasets = CollabFilteringDataset(user, item, ratings)
learn = get_collab_learner(ratings, n_factors=n_factors, user_name=None, item_name=None, rating_name=None, pct_val=0.2, test=None, metrics=None, min_score=None, max_score=None)        
path = untar_data(URLS.S3_IMAGE, fname=3, dest=2, data=False)
path = untar_data(URLS.MNIST_SAMPLE, fname=None, dest=None, data=True)
path = untar_data(URLS.CIFAR, fname=2, dest=2, data=False)
from fastai import *          # Quick access to most common functionality
from fastai.tabular import *  # Quick access to tabular functionality

path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
train_df, valid_df = df[:-2000].copy(),df[-2000:].copy()

dep_var = '>=50k'
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
data = TabularDataBunch.from_df(path, train_df, valid_df, dep_var,
                                tfms=[FillMissing, Categorify], cat_names=cat_names)

learn = get_tabular_learner(data, layers=[200,100], metrics=accuracy)
learn.fit(1, 1e-2)
learn = get_collab_learner(ratings, n_factors=n_factors, user_name=None, item_name=None, rating_name=None, pct_val=0.2, test=None, metrics=None, min_score=None, max_score=None)    
from fastai import *
from fastai.vision import *

path = untar_data(URLs.DOGS)
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)

learn = create_cnn(data, models.resnet34, metrics=accuracy)
learn.fit_one_cycle(1)

learn.unfreeze()
learn.fit_one_cycle(6, slice(1e-5,3e-4), pct_start=0.05)

accuracy(*learn.TTA())


learn.loss = nn.CosineEmbeddingLoss()
metrics = [accuracy_thresh]
from fastai import *
from fastai.tabular import *
learn.fit_one_cycle(5, 5e-3)
learn.save('stage-1')

learn.lr_find()
learn.recorder.plot()

learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-3 / 2.6**4 ,1e-3))
learn.save('stage-2')
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
from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np

class ACGAN():
	def __init__(self):
		# Input shape
		self.img_rows = 28
		self.img_cols = 28
		self.channels = 1
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		self.num_classes = 10
		self.latent_dim = 100

		optimizer = Adam(0.0002, 0.5)
		losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

		# Build and compile the discriminator
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss=losses,
			optimizer=optimizer,
			metrics=['accuracy'])

		# Build the generator
		self.generator = self.build_generator()

		# The generator takes noise and the target label as input
		# and generates the corresponding digit of that label
		noise = Input(shape=(self.latent_dim,))
		label = Input(shape=(1,))
		img = self.generator([noise, label])

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The discriminator takes generated image as input and determines validity
		# and the label of that image
		valid, target_label = self.discriminator(img)

		# The combined model  (stacked generator and discriminator)
		# Trains the generator to fool the discriminator
		self.combined = Model([noise, label], [valid, target_label])
		self.combined.compile(loss=losses,
			optimizer=optimizer)

	def build_generator(self):

		model = Sequential()

		model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
		model.add(Reshape((7, 7, 128)))
		model.add(BatchNormalization(momentum=0.8))
		model.add(UpSampling2D())
		model.add(Conv2D(128, kernel_size=3, padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(UpSampling2D())
		model.add(Conv2D(64, kernel_size=3, padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
		model.add(Activation("tanh"))

		model.summary()

		noise = Input(shape=(self.latent_dim,))
		label = Input(shape=(1,), dtype='int32')
		label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))

		model_input = multiply([noise, label_embedding])
		img = model(model_input)

		return Model([noise, label], img)

	def build_discriminator(self):

		model = Sequential()

		model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
		model.add(ZeroPadding2D(padding=((0,1),(0,1))))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.summary()

		img = Input(shape=self.img_shape)

		# Extract feature representation
		features = model(img)

		# Determine validity and label of the image
		validity = Dense(1, activation="sigmoid")(features)
		label = Dense(self.num_classes+1, activation="softmax")(features)

		return Model(img, [validity, label])

	def train(self, epochs, batch_size=128, sample_interval=50):

		# Load the dataset
		(X_train, y_train), (_, _) = mnist.load_data()

		# Configure inputs
		X_train = (X_train.astype(np.float32) - 127.5) / 127.5
		X_train = np.expand_dims(X_train, axis=3)
		y_train = y_train.reshape(-1, 1)

		# Adversarial ground truths
		valid = np.ones((batch_size, 1))
		fake = np.zeros((batch_size, 1))

		for epoch in range(epochs):

			# ---------------------
			#  Train Discriminator
			# ---------------------

			# Select a random batch of images
			idx = np.random.randint(0, X_train.shape[0], batch_size)
			imgs = X_train[idx]

			# Sample noise as generator input
			noise = np.random.normal(0, 1, (batch_size, 100))

			# The labels of the digits that the generator tries to create an
			# image representation of
			sampled_labels = np.random.randint(0, 10, (batch_size, 1))

			# Generate a half batch of new images
			gen_imgs = self.generator.predict([noise, sampled_labels])

			# Image labels. 0-9 if image is valid or 10 if it is generated (fake)
			img_labels = y_train[idx]
			fake_labels = 10 * np.ones(img_labels.shape)

			# Train the discriminator
			d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
			d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			# ---------------------
			#  Train Generator
			# ---------------------

			# Train the generator
			g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

			# Plot the progress
			print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))

			# If at save interval => save generated image samples
			if epoch % sample_interval == 0:
				self.save_model()
				self.sample_images(epoch)

	def sample_images(self, epoch):
		r, c = 10, 10
		noise = np.random.normal(0, 1, (r * c, 100))
		sampled_labels = np.array([num for _ in range(r) for num in range(c)])
		gen_imgs = self.generator.predict([noise, sampled_labels])
		# Rescale images 0 - 1
		gen_imgs = 0.5 * gen_imgs + 0.5

		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
				axs[i,j].axis('off')
				cnt += 1
		fig.savefig("images/%d.png" % epoch)
		plt.close()

	def save_model(self):

		def save(model, model_name):
			model_path = "saved_model/%s.json" % model_name
			weights_path = "saved_model/%s_weights.hdf5" % model_name
			options = {"file_arch": model_path,
						"file_weight": weights_path}
			json_string = model.to_json()
			open(options['file_arch'], 'w').write(json_string)
			model.save_weights(options['file_weight'])

		save(self.generator, "generator")
		save(self.discriminator, "discriminator")


if __name__ == '__main__':
	acgan = ACGAN()
	acgan.train(epochs=14000, batch_size=32, sample_interval=200)
from __future__ import print_function, division

from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
#from keras.layers.advanced_activations import LeakyReLU
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.misc import *

import sys

import numpy as np

class DCGAN():
	def __init__(self):
		# Input shape
		self.img_rows = 32
		self.img_cols = 32
		self.channels = 3
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		self.latent_dim = 100

		optimizer = Adam(0.0002, 0.5)

		# Build and compile the discriminator
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='binary_crossentropy',
			optimizer=optimizer,
			metrics=['accuracy'])

		# Build the generator
		self.generator = self.build_generator()

		# The generator takes noise as input and generates imgs
		z = Input(shape=(100,))
		img = self.generator(z)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The discriminator takes generated images as input and determines validity
		valid = self.discriminator(img)

		# The combined model  (stacked generator and discriminator)
		# Trains the generator to fool the discriminator
		self.combined = Model(z, valid)
		self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

	def build_generator(self):

		model = Sequential()

		model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
		model.add(Reshape((8, 8, 128)))
		model.add(UpSampling2D())
		model.add(Conv2D(128, kernel_size=3, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("relu"))
		model.add(UpSampling2D())
		model.add(Conv2D(64, kernel_size=3, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("relu"))
		model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
		model.add(Activation("tanh"))

		model.summary()

		noise = Input(shape=(self.latent_dim,))
		img = model(noise)

		return Model(noise, img)

	def build_discriminator(self):

		model = Sequential()

		model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same", trainable=False))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
		model.add(ZeroPadding2D(padding=((0,1),(0,1))))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid'))

		model.summary()

		img = Input(shape=self.img_shape)
		validity = model(img)

		return Model(img, validity)

	def train(self, epochs, batch_size=128, save_interval=100):

		# Load the dataset
		(X_train, _), (_, _) = cifar10.load_data()

		# Rescale -1 to 1
		X_train = X_train / 127.5 - 1.
		print(X_train.shape)
#		X_train = np.expand_dims(X_train, axis=3)
#		print(X_train.shape)
		# Adversarial ground truths
		valid = np.ones((batch_size, 1))
		fake = np.zeros((batch_size, 1))

		for epoch in range(epochs):

			# ---------------------
			#  Train Discriminator
			# ---------------------

			# Select a random half of images
			idx = np.random.randint(0, X_train.shape[0], batch_size)
			imgs = X_train[idx]

			# Sample noise and generate a batch of new images
			noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
			gen_imgs = self.generator.predict(noise)

			# Train the discriminator (real classified as ones and generated as zeros)
			d_loss_real = self.discriminator.train_on_batch(imgs, valid)
			d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			# ---------------------
			#  Train Generator
			# ---------------------

			# Train the generator (wants discriminator to mistake images as real)
			g_loss = self.combined.train_on_batch(noise, valid)

			# Plot the progress
			if(epoch%100==0):	
				print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

			# If at save interval => save generated image samples
			if epoch % save_interval == 0:
				self.save_imgs(epoch)

	def save_imgs(self, epoch):
		r, c = 1, 1
		noise = np.random.normal(0, 1, (r * c, self.latent_dim))
		gen_imgs = self.generator.predict(noise)

		# Rescale images 0 - 1
		gen_imgs = 0.5 * gen_imgs + 0.5
#		print(len(gen_imgs))
		for image_idx in range(1):
#			plt.subplot(3, 3, image_idx+1)
			#generated_image = unnormalize_display(train_data[image_idx]).transpose(1,2,0)
#			print(gen_imgs.shape)
#			print(gen_imgs)
#			generated_image = generated_images[image_idx].transpose(1,2,0)
#			print(generated_image.shape)
			imsave("images/cifar10_%d.png" % epoch,gen_imgs[0])


if __name__ == '__main__':
	dcgan = DCGAN()
	dcgan.train(epochs=12000, batch_size=32, save_interval=50)
from __future__ import print_function, division

from keras.datasets import fashion_mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
#from keras.layers.advanced_activations import LeakyReLU
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

import sys
import scipy.ndimage
import  os
import numpy as np

class DCGAN():
	def __init__(self):
		
		batch_size = 32  # batch size
		num_category = 10  # total categorical factor
		#num_cont = 2  # total continuous factor
		num_dim = 50  # total latent dimension
		T_num = 25
		os.environ["CUDA_VISIBLE_DEVICES"] = "0"
		train_flag = True
		sample_flag = True
		load_flag = False
		save_flag = True

		z_cat = tf.random_uniform([batch_size],minval=0,maxval=10,dtype=tf.int32)
		z_cat = tf.one_hot(z_cat, num_category)

		multi_dist = tf.contrib.distributions.StudentT(df=[2.0]*(7*7*128-10),loc=[0.0]*(7*7*128-10),scale=[1.0]*(7*7*128-10))
		tmp_noise  = multi_dist.sample([batch_size])   #[32,7*7*128-10]

		noise = tmp_noise
		#multi_dist = tf.random_normal([batch_size,7*7*128-10],mean=0.0,stddev=1.0)
		#tmp_noise =multi_dist
		with tf.variable_scope('weight'):
			mu = tf.get_variable("mu",[T_num,batch_size,7*7*128-10])
			sigma = tf.get_variable("sigma",[T_num,batch_size,7*7*128-10])
			print(mu.shape)
			print(sigma.shape)
			print (tmp_noise.shape)
			noise = tmp_noise*mu+sigma
			weight = tf.get_variable("weight",[T_num,1,1])   #5个T分布[5,1,1]
			weight = tf.tile(weight,[1,batch_size,7*7*128-10])   #weight扩到每个参数上[5,32,7*7*128-10]
			#noise = noise * weight
			noise = noise * weight
			noise = tf.reduce_mean(noise,axis=0)
		print("z_cat.shape=",z_cat.shape)
		z = tf.concat([z_cat,noise],1)
		print("noise.shape=",noise.shape)
		print("z.shape=",z.shape)

		#z_cont = z[:, num_category:num_category+num_cont]

		true_images = tf.placeholder(tf.float32, [batch_size,28,28,1])
		true_labels = tf.placeholder(tf.float32, [batch_size,num_category])

		# Input shape
		self.img_rows = 28
		self.img_cols = 28
		self.channels = 1
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		self.latent_dim = 32

		optimizer = Adam(0.0002, 0.5)

		# Build and compile the discriminator
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='binary_crossentropy',
			optimizer=optimizer,
			metrics=['accuracy'])

		# Build the generator
		self.generator = self.build_generator()

		# The generator takes noise as input and generates imgs
#		z = Input(shape=(100,))
		print(z.shape)
		img = self.generator(z)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The discriminator takes generated images as input and determines validity
		valid = self.discriminator(img)

		# The combined model  (stacked generator and discriminator)
		# Trains the generator to fool the discriminator
		self.combined = Model(z, valid)
		self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

	def build_generator(self):

		model = Sequential()

		model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
		model.add(Reshape((7, 7, 128)))
		model.add(UpSampling2D())
		model.add(Conv2D(128, kernel_size=3, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("relu"))
		model.add(UpSampling2D())
		model.add(Conv2D(64, kernel_size=3, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("relu"))
		model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
		model.add(Activation("tanh"))

		model.summary()

		noise = Input(shape=(self.latent_dim,))
		img = model(noise)

		return Model(noise, img)

	def build_discriminator(self):

		model = Sequential()

		model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
		model.add(ZeroPadding2D(padding=((0,1),(0,1))))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid'))
#补一份sigmod
		model.summary()

		img = Input(shape=self.img_shape)
		validity = model(img)

		return Model(img, validity)

	def train(self, epochs, batch_size=128, save_interval=50):

		# Load the dataset
		(X_train, _), (_, _) = fashion_mnist.load_data()

		# Rescale -1 to 1
		X_train = X_train / 127.5 - 1.
		X_train = np.expand_dims(X_train, axis=3)

		# Adversarial ground truths
		valid = np.ones((batch_size, 1))
		fake = np.zeros((batch_size, 1))

		for epoch in range(epochs):

			# ---------------------
			#  Train Discriminator
			# ---------------------

			# Select a random half of images
			idx = np.random.randint(0, X_train.shape[0], batch_size)
			imgs = X_train[idx]

			# Sample noise and generate a batch of new images
			noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
			gen_imgs = self.generator.predict(noise)

			# Train the discriminator (real classified as ones and generated as zeros)
			d_loss_real = self.discriminator.train_on_batch(imgs, valid)
			d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			# ---------------------
			#  Train Generator
			# ---------------------

			# Train the generator (wants discriminator to mistake images as real)
			g_loss = self.combined.train_on_batch(noise, valid)

			# Plot the progress
			print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

			# If at save interval => save generated image samples
			if epoch % save_interval == 0:
				self.save_imgs(epoch)

	def save_imgs(self, epoch):
		r, c = 5, 5
		noise = np.random.normal(0, 1, (r * c, self.latent_dim))
		gen_imgs = self.generator.predict(noise)

		# Rescale images 0 - 1
		gen_imgs = 0.5 * gen_imgs + 0.5

		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
				axs[i,j].axis('off')
				cnt += 1
		fig.savefig("images/mnist_%d.png" % epoch)
		plt.close()


if __name__ == '__main__':
	dcgan = DCGAN()
	dcgan.train(epochs=7*7*128-1000, batch_size=32, save_interval=50)
from __future__ import print_function, division

from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
#from keras.layers.advanced_activations import LeakyReLU
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

class DCGAN():
	def __init__(self):
		# Input shape
		self.img_rows = 32
		self.img_cols = 32
		self.channels = 3
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		self.latent_dim = 100

		optimizer = Adam(0.0002, 0.5)

		# Build and compile the discriminator
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='binary_crossentropy',
			optimizer=optimizer,
			metrics=['accuracy'])

		# Build the generator
		self.generator = self.build_generator()

		# The generator takes noise as input and generates imgs
		z = Input(shape=(100,))
		img = self.generator(z)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The discriminator takes generated images as input and determines validity
		valid = self.discriminator(img)

		# The combined model  (stacked generator and discriminator)
		# Trains the generator to fool the discriminator
		self.combined = Model(z, valid)
		self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

	def build_generator(self):

		model = Sequential()

		model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
		model.add(Reshape((8, 8, 128)))
		model.add(UpSampling2D())
		model.add(Conv2D(128, kernel_size=3, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("relu"))
		model.add(UpSampling2D())
		model.add(Conv2D(64, kernel_size=3, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("relu"))
		model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
		model.add(Activation("tanh"))

		model.summary()

		noise = Input(shape=(self.latent_dim,))
		img = model(noise)

		return Model(noise, img)

	def build_discriminator(self):

		model = Sequential()

		model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
		model.add(ZeroPadding2D(padding=((0,1),(0,1))))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid'))

		model.summary()

		img = Input(shape=self.img_shape)
		validity = model(img)

		return Model(img, validity)

	def train(self, epochs, batch_size=128, save_interval=50):

		# Load the dataset
		(X_train, _), (_, _) = cifar10.load_data()

		# Rescale -1 to 1
		X_train = X_train / 127.5 - 1.
		print(X_train.shape)
#		X_train = np.expand_dims(X_train, axis=3)
#		print(X_train.shape)
		# Adversarial ground truths
		valid = np.ones((batch_size, 1))
		fake = np.zeros((batch_size, 1))

		for epoch in range(epochs):

			# ---------------------
			#  Train Discriminator
			# ---------------------

			# Select a random half of images
			idx = np.random.randint(0, X_train.shape[0], batch_size)
			imgs = X_train[idx]

			# Sample noise and generate a batch of new images
			noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
			gen_imgs = self.generator.predict(noise)

			# Train the discriminator (real classified as ones and generated as zeros)
			d_loss_real = self.discriminator.train_on_batch(imgs, valid)
			d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			# ---------------------
			#  Train Generator
			# ---------------------

			# Train the generator (wants discriminator to mistake images as real)
			g_loss = self.combined.train_on_batch(noise, valid)

			# Plot the progress
			print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

			# If at save interval => save generated image samples
			if epoch % save_interval == 0:
				self.save_imgs(epoch)

	def save_imgs(self, epoch):
		r, c = 8, 8
		noise = np.random.normal(0, 1, (r * c, self.latent_dim))
		gen_imgs = self.generator.predict(noise)

		# Rescale images 0 - 1
		gen_imgs = 0.5 * gen_imgs + 0.5
		print(len(gen_imgs))
		for image_idx in range(len(gen_imgs)):
			plt.subplot(8, 8, image_idx+1)
			#generated_image = unnormalize_display(train_data[image_idx]).transpose(1,2,0)
			generated_image = gen_imgs[image_idx].transpose(0,1,2)
			print(generated_image.shape)
			plt.imshow(generated_image)
		#plt.show(block=False)
		plt.savefig('images/sample_'+str(epoch)+'.png')
		plt.close()


if __name__ == '__main__':
	dcgan = DCGAN()
	dcgan.train(epochs=1, batch_size=32, save_interval=50)
#!/usr/bin/python

""" Deep Convolutional Generative Adversarial Network (DCGAN).
Using deep convolutional generative adversarial networks (DCGAN) to generate
digit images from a noise distribution.
References:
	- Unsupervised representation learning with deep convolutional generative
	adversarial networks. A Radford, L Metz, S Chintala. arXiv:1511.06434.
Links:
	- [DCGAN Paper](https://arxiv.org/abs/1511.06434).
	- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Params
num_steps = 20000
batch_size = 32

# Network Params
image_dim = 784 # 28*28 pixels * 1 channel
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 200 # Noise data points


# Generator Network
# Input: Noise, Output: Image
def generator(x, reuse=False):
	with tf.variable_scope('Generator', reuse=reuse):
		# TensorFlow Layers automatically create variables and calculate their
		# shape, based on the input.
		x = tf.layers.dense(x, units=6 * 6 * 128)
		x = tf.nn.tanh(x)
		# Reshape to a 4-D array of images: (batch, height, width, channels)
		# New shape: (batch, 6, 6, 128)
		x = tf.reshape(x, shape=[-1, 6, 6, 128])
		# Deconvolution, image shape: (batch, 14, 14, 64)
		x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
		# Deconvolution, image shape: (batch, 28, 28, 1)
		x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
		# Apply sigmoid to clip values between 0 and 1
		x = tf.nn.sigmoid(x)
		return x


# Discriminator Network
# Input: Image, Output: Prediction Real/Fake Image
def discriminator(x, reuse=False):
	with tf.variable_scope('Discriminator', reuse=reuse):
		# Typical convolutional neural network to classify images.
		x = tf.layers.conv2d(x, 64, 5)
		x = tf.nn.tanh(x)
		x = tf.layers.average_pooling2d(x, 2, 2)
		x = tf.layers.conv2d(x, 128, 5)
		x = tf.nn.tanh(x)
		x = tf.layers.average_pooling2d(x, 2, 2)
		x = tf.contrib.layers.flatten(x)
		x = tf.layers.dense(x, 1024)
		x = tf.nn.tanh(x)
		# Output 2 classes: Real and Fake images
		x = tf.layers.dense(x, 2)
	return x

# Build Networks
# Network Inputs
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

# Build Generator Network
gen_sample = generator(noise_input)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse=True)
disc_concat = tf.concat([disc_real, disc_fake], axis=0)

# Build the stacked generator/discriminator
stacked_gan = discriminator(gen_sample, reuse=True)

# Build Targets (real or fake images)
disc_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])

# Build Loss
disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
	logits=disc_concat, labels=disc_target))
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
	logits=stacked_gan, labels=gen_target))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.001)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.001)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
# Discriminator Network Variables
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

# Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

	# Run the initializer
	sess.run(init)

	for i in range(1, num_steps+1):

		# Prepare Input Data
		# Get the next batch of MNIST data (only images are needed, not labels)
		batch_x, _ = mnist.train.next_batch(batch_size)
		batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
		# Generate noise to feed to the generator
		z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

		# Prepare Targets (Real image: 1, Fake image: 0)
		# The first half of data fed to the generator are real images,
		# the other half are fake images (coming from the generator).
		batch_disc_y = np.concatenate(
			[np.ones([batch_size]), np.zeros([batch_size])], axis=0)
		# Generator tries to fool the discriminator, thus targets are 1.
		batch_gen_y = np.ones([batch_size])

		# Training
		feed_dict = {real_image_input: batch_x, noise_input: z,
					 disc_target: batch_disc_y, gen_target: batch_gen_y}
		_, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
								feed_dict=feed_dict)
		if i % 100 == 0 or i == 1:
			print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

	# Generate images from noise, using the generator network.
	f, a = plt.subplots(4, 10, figsize=(10, 4))
	for i in range(10):
		# Noise input.
		z = np.random.uniform(-1., 1., size=[4, noise_dim])
		g = sess.run(gen_sample, feed_dict={noise_input: z})
		for j in range(4):
			# Generate image from noise. Extend to 3 channels for matplot figure.
			img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
							 newshape=(28, 28, 3))
			a[j][i].imshow(img)

	f.show()
	plt.draw()
	plt.waitforbuttonpress()
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
"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold = 1e6)
tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_IDEAS = 5             # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 15     # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

# show our beautiful painting range
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
plt.legend(loc='upper right')
plt.show()


def artist_works():     # painting from the famous artist (real target)
	a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
	paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
	return paintings


with tf.variable_scope('Generator'):
	G_in = tf.placeholder(tf.float32, [None, N_IDEAS])          # random ideas (could from normal distribution)
	G_l1 = tf.layers.dense(G_in, 128, tf.nn.relu)
	G_out = tf.layers.dense(G_l1, ART_COMPONENTS)               # making a painting from these random ideas

with tf.variable_scope('Discriminator'):
	real_art = tf.placeholder(tf.float32, [None, ART_COMPONENTS], name='real_in')   # receive art work from the famous artist
	D_l0 = tf.layers.dense(real_art, 128, tf.nn.relu, name='l')
	prob_artist0 = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name='out')              # probability that the art work is made by artist
	# reuse layers for generator
	D_l1 = tf.layers.dense(G_out, 128, tf.nn.relu, name='l', reuse=True)            # receive art work from a newbie like G
	prob_artist1 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=True)  # probability that the art work is made by artist

D_loss = -tf.reduce_mean(tf.log(prob_artist0) + tf.log(1-prob_artist1))
G_loss = tf.reduce_mean(tf.log(1-prob_artist1))

train_D = tf.train.AdamOptimizer(LR_D).minimize(
	D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
train_G = tf.train.AdamOptimizer(LR_G).minimize(
	G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()   # something about continuous plotting
for step in range(5000):
	artist_paintings = artist_works()           # real painting from artist
	G_ideas = np.random.randn(BATCH_SIZE, N_IDEAS)
	G_paintings, pa0, Dl = sess.run([G_out, prob_artist0, D_loss, train_D, train_G],    # train and get results
									{G_in: G_ideas, real_art: artist_paintings})[:3]

	if step % 50 == 0:  # plotting
		plt.cla()
		plt.plot(PAINT_POINTS[0], G_paintings[0], c='#4AD631', lw=3, label='Generated painting',)
		plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
		plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
		plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % pa0.mean(), fontdict={'size': 15})
		plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -Dl, fontdict={'size': 15})
		plt.ylim((0, 3)); plt.legend(loc='upper right', fontsize=12); plt.draw(); plt.pause(0.01)

plt.ioff()
plt.show()"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_IDEAS = 5             # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 15     # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

# show our beautiful painting range
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
plt.legend(loc='upper right')
plt.show()


def artist_works():     # painting from the famous artist (real target)
	a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
	paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
	labels = (a - 1) > 0.5  # upper paintings (1), lower paintings (0), two classes
	labels = labels.astype(np.float32)
	return paintings, labels

art_labels = tf.placeholder(tf.float32, [None, 1])
with tf.variable_scope('Generator'):
	G_in = tf.placeholder(tf.float32, [None, N_IDEAS])          # random ideas (could from normal distribution)
	G_art = tf.concat((G_in, art_labels), 1)                    # combine ideas with labels
	G_l1 = tf.layers.dense(G_art, 128, tf.nn.relu)
	G_out = tf.layers.dense(G_l1, ART_COMPONENTS)               # making a painting from these random ideas

with tf.variable_scope('Discriminator'):
	real_in = tf.placeholder(tf.float32, [None, ART_COMPONENTS], name='real_in')   # receive art work from the famous artist + label
	real_art = tf.concat((real_in, art_labels), 1)                                  # art with labels
	D_l0 = tf.layers.dense(real_art, 128, tf.nn.relu, name='l')
	prob_artist0 = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name='out')              # probability that the art work is made by artist
	# reuse layers for generator
	G_art = tf.concat((G_out, art_labels), 1)                                       # art with labels
	D_l1 = tf.layers.dense(G_art, 128, tf.nn.relu, name='l', reuse=True)            # receive art work from a newbie like G
	prob_artist1 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=True)  # probability that the art work is made by artist

D_loss = -tf.reduce_mean(tf.log(prob_artist0) + tf.log(1-prob_artist1))
G_loss = tf.reduce_mean(tf.log(1-prob_artist1))

train_D = tf.train.AdamOptimizer(LR_D).minimize(
	D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
train_G = tf.train.AdamOptimizer(LR_G).minimize(
	G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()   # something about continuous plotting
for step in range(7000):
	artist_paintings, labels = artist_works()               # real painting from artist
	G_ideas = np.random.randn(BATCH_SIZE, N_IDEAS)
	G_paintings, pa0, Dl = sess.run([G_out, prob_artist0, D_loss, train_D, train_G],    # train and get results
									{G_in: G_ideas, real_in: artist_paintings, art_labels: labels})[:3]

	if step % 50 == 0:  # plotting
		plt.cla()
		plt.plot(PAINT_POINTS[0], G_paintings[0], c='#4AD631', lw=3, label='Generated painting',)
		bound = [0, 0.5] if labels[0, 0] == 0 else [0.5, 1]
		plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + bound[1], c='#74BCFF', lw=3, label='upper bound')
		plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + bound[0], c='#FF9359', lw=3, label='lower bound')
		plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % pa0.mean(), fontdict={'size': 15})
		plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -Dl, fontdict={'size': 15})
		plt.text(-.5, 1.7, 'Class = %i' % int(labels[0, 0]), fontdict={'size': 15})
		plt.ylim((0, 3)); plt.legend(loc='upper right', fontsize=12); plt.draw(); plt.pause(0.1)

plt.ioff()

# plot a generated painting for upper class
plt.figure(2)
z = np.random.randn(1, N_IDEAS)
label = np.array([[1.]])            # for upper class
G_paintings = sess.run(G_out, {G_in: z, art_labels: label})
plt.plot(PAINT_POINTS[0], G_paintings[0], c='#4AD631', lw=3, label='G painting for upper class',)
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + bound[1], c='#74BCFF', lw=3, label='upper bound (class 1)')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + bound[0], c='#FF9359', lw=3, label='lower bound (class 1)')
plt.ylim((0, 3)); plt.legend(loc='upper right', fontsize=12); plt.show()#!/usr/bin/python

""" Generative Adversarial Networks (GAN).
Using generative adversarial networks (GAN) to generate digit images from a
noise distribution.
References:
	- Generative adversarial nets. I Goodfellow, J Pouget-Abadie, M Mirza,
	B Xu, D Warde-Farley, S Ozair, Y. Bengio. Advances in neural information
	processing systems, 2672-2680.
	- Understanding the difficulty of training deep feedforward neural networks.
	X Glorot, Y Bengio. Aistats 9, 249-256
Links:
	- [GAN Paper](https://arxiv.org/pdf/1406.2661.pdf).
	- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
	- [Xavier Glorot Init](www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.../AISTATS2010_Glorot.pdf).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Params
num_steps = 100000
batch_size = 128
learning_rate = 0.0002

# Network Params
image_dim = 784 # 28*28 pixels
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100 # Noise data points

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
	return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Store layers weight & bias
weights = {
	'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
	'gen_out': tf.Variable(glorot_init([gen_hidden_dim, image_dim])),
	'disc_hidden1': tf.Variable(glorot_init([image_dim, disc_hidden_dim])),
	'disc_out': tf.Variable(glorot_init([disc_hidden_dim, 1])),
}
biases = {
	'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
	'gen_out': tf.Variable(tf.zeros([image_dim])),
	'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
	'disc_out': tf.Variable(tf.zeros([1])),
}


# Generator
def generator(x):
	hidden_layer = tf.matmul(x, weights['gen_hidden1'])
	hidden_layer = tf.add(hidden_layer, biases['gen_hidden1'])
	hidden_layer = tf.nn.relu(hidden_layer)
	out_layer = tf.matmul(hidden_layer, weights['gen_out'])
	out_layer = tf.add(out_layer, biases['gen_out'])
	out_layer = tf.nn.sigmoid(out_layer)
	return out_layer


# Discriminator
def discriminator(x):
	hidden_layer = tf.matmul(x, weights['disc_hidden1'])
	hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])
	hidden_layer = tf.nn.relu(hidden_layer)
	out_layer = tf.matmul(hidden_layer, weights['disc_out'])
	out_layer = tf.add(out_layer, biases['disc_out'])
	out_layer = tf.nn.sigmoid(out_layer)
	return out_layer

# Build Networks
# Network Inputs
gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')

# Build Generator Network
gen_sample = generator(gen_input)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample)

# Build Loss
gen_loss = -tf.reduce_mean(tf.log(disc_fake))
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables
gen_vars = [weights['gen_hidden1'], weights['gen_out'],
			biases['gen_hidden1'], biases['gen_out']]
# Discriminator Network Variables
disc_vars = [weights['disc_hidden1'], weights['disc_out'],
			biases['disc_hidden1'], biases['disc_out']]

# Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

	# Run the initializer
	sess.run(init)

	for i in range(1, num_steps+1):
		# Prepare Data
		# Get the next batch of MNIST data (only images are needed, not labels)
		batch_x, _ = mnist.train.next_batch(batch_size)
		# Generate noise to feed to the generator
		z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

		# Train
		feed_dict = {disc_input: batch_x, gen_input: z}
		_, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
								feed_dict=feed_dict)
		if i % 1000 == 0 or i == 1:
			print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

	# Generate images from noise, using the generator network.
	f, a = plt.subplots(4, 10, figsize=(10, 4))
	for i in range(10):
		# Noise input.
		z = np.random.uniform(-1., 1., size=[4, noise_dim])
		g = sess.run([gen_sample], feed_dict={gen_input: z})
		g = np.reshape(g, newshape=(4, 28, 28, 1))
		# Reverse colours for better display
		g = -1 * (g - 1)
		for j in range(4):
			# Generate image from noise. Extend to 3 channels for matplot figure.
			img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
							 newshape=(28, 28, 3))
			a[j][i].imshow(img)

	f.show()
	plt.draw()
	plt.savefig("mnist.png")
	plt.waitforbuttonpress()from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
							optimizer='rmsprop',
							metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# 生成虚拟数据
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
							optimizer='rmsprop',
							metrics=['accuracy'])

model.fit(x_train, y_train,
					epochs=20,
					batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# 生成虚拟数据
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) 是一个具有 64 个隐藏神经元的全连接层。
# 在第一层必须指定所期望的输入数据尺寸：
# 在这里，是一个 20 维的向量。
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
							optimizer=sgd,
							metrics=['accuracy'])

model.fit(x_train, y_train,
					epochs=20,
					batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)#coding=utf-8  
from keras.models import Sequential  
from keras.layers import Dense,Flatten  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical  
from keras.datasets import mnist
from keras.utils import np_utils
import keras
import numpy as np  
seed = 7  
np.random.seed(seed)  
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape,y_train.shape)
X_train = X_train.reshape(-1,28, 28,1)/255.
X_test = X_test.reshape(-1,28, 28,1)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print(X_train.shape,y_train.shape)
model = Sequential()  
model.add(Conv2D(32,(5,5),strides=(1,1),input_shape=(28,28,1),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(64,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Flatten())  
model.add(Dense(100,activation='relu'))  
model.add(Dense(10,activation='softmax'))  
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])  
model.summary()  
from keras.callbacks import TensorBoard

#model.fit(x_train,y_train,batch_size,epoch,)
print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=10, batch_size=64,verbose=1,callbacks=[TensorBoard(log_dir='./log_dir')])

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
	os.makedirs(opt.outf)
except OSError:
	pass

if opt.manualSeed is None:
	opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
	# folder dataset
	dataset = dset.ImageFolder(root=opt.dataroot,
							   transform=transforms.Compose([
								   transforms.Resize(opt.imageSize),
								   transforms.CenterCrop(opt.imageSize),
								   transforms.ToTensor(),
								   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
							   ]))
elif opt.dataset == 'lsun':
	dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
						transform=transforms.Compose([
							transforms.Resize(opt.imageSize),
							transforms.CenterCrop(opt.imageSize),
							transforms.ToTensor(),
							transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
						]))
elif opt.dataset == 'cifar10':
	dataset = dset.CIFAR10(root=opt.dataroot, download=True,
						   transform=transforms.Compose([
							   transforms.Resize(opt.imageSize),
							   transforms.ToTensor(),
							   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
						   ]))
elif opt.dataset == 'fake':
	dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
							transform=transforms.ToTensor())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
										 shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3


# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


class Generator(nn.Module):
	def __init__(self, ngpu):
		super(Generator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),
			# state size. (ngf) x 32 x 32
			nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
			nn.Tanh()
			# state size. (nc) x 64 x 64
		)

	def forward(self, input):
		if input.is_cuda and self.ngpu > 1:
			output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
		else:
			output = self.main(input)
		return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
	netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
	def __init__(self, ngpu):
		super(Discriminator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
		)

	def forward(self, input):
		if input.is_cuda and self.ngpu > 1:
			output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
		else:
			output = self.main(input)

		return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
	netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
	for i, data in enumerate(dataloader, 0):
		############################
		# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
		###########################
		# train with real
		netD.zero_grad()
		real_cpu = data[0].to(device)
		batch_size = real_cpu.size(0)
		label = torch.full((batch_size,), real_label, device=device)

		output = netD(real_cpu)
		errD_real = criterion(output, label)
		errD_real.backward()
		D_x = output.mean().item()

		# train with fake
		noise = torch.randn(batch_size, nz, 1, 1, device=device)
		fake = netG(noise)
		label.fill_(fake_label)
		output = netD(fake.detach())
		errD_fake = criterion(output, label)
		errD_fake.backward()
		D_G_z1 = output.mean().item()
		errD = errD_real + errD_fake
		optimizerD.step()

		############################
		# (2) Update G network: maximize log(D(G(z)))
		###########################
		netG.zero_grad()
		label.fill_(real_label)  # fake labels are real for generator cost
		output = netD(fake)
		errG = criterion(output, label)
		errG.backward()
		D_G_z2 = output.mean().item()
		optimizerG.step()

		print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
			  % (epoch, opt.niter, i, len(dataloader),
				 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
		if i % 100 == 0:
			vutils.save_image(real_cpu,
					'%s/real_samples.png' % opt.outf,
					normalize=True)
			fake = netG(fixed_noise)
			vutils.save_image(fake.detach(),
					'%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
					normalize=True)

	# do checkpointing
	torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
	torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))import numpy as np
from keras import backend as K


TAGS = ['rock', 'pop', 'alternative', 'indie', 'electronic',
        'female vocalists', 'dance', '00s', 'alternative rock', 'jazz',
        'beautiful', 'metal', 'chillout', 'male vocalists',
        'classic rock', 'soul', 'indie rock', 'Mellow', 'electronica',
        '80s', 'folk', '90s', 'chill', 'instrumental', 'punk',
        'oldies', 'blues', 'hard rock', 'ambient', 'acoustic',
        'experimental', 'female vocalist', 'guitar', 'Hip-Hop',
        '70s', 'party', 'country', 'easy listening',
        'sexy', 'catchy', 'funk', 'electro', 'heavy metal',
        'Progressive rock', '60s', 'rnb', 'indie pop',
        'sad', 'House', 'happy']


def librosa_exists():
    try:
        __import__('librosa')
    except ImportError:
        return False
    else:
        return True


def preprocess_input(audio_path, dim_ordering='default'):
    '''Reads an audio file and outputs a Mel-spectrogram.
    '''
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if librosa_exists():
        import librosa
    else:
        raise RuntimeError('Librosa is required to process audio files.\n' +
                           'Install it via `pip install librosa` \nor visit ' +
                           'http://librosa.github.io/librosa/ for details.')

    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12

    src, sr = librosa.load(audio_path, sr=SR)
    n_sample = src.shape[0]
    n_sample_wanted = int(DURA * SR)

    # trim the signal at the center
    if n_sample < n_sample_wanted:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_wanted:  # if too long
        src = src[(n_sample - n_sample_wanted) / 2:
                  (n_sample + n_sample_wanted) / 2]

    logam = librosa.logamplitude
    melgram = librosa.feature.melspectrogram
    x = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                      n_fft=N_FFT, n_mels=N_MELS) ** 2,
              ref_power=1.0)

    if dim_ordering == 'th':
        x = np.expand_dims(x, axis=0)
    elif dim_ordering == 'tf':
        x = np.expand_dims(x, axis=3)
    return x


def decode_predictions(preds, top_n=5):
    '''Decode the output of a music tagger model.

    # Arguments
        preds: 2-dimensional numpy array
        top_n: integer in [0, 50], number of items to show

    '''
    assert len(preds.shape) == 2 and preds.shape[1] == 50
    results = []
    for pred in preds:
        result = zip(TAGS, pred)
        result = sorted(result, key=lambda x: x[1], reverse=True)
        results.append(result[:top_n])
    return results
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
#coding=utf-8
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras.utils import plot_model
(X_train,y_train),(X_test,y_test)=mnist.load_data()
print(X_train.shape)
#print(X_train.shape,y_train.shape)
X_train = X_train.reshape(-1, 28, 28,1)
X_test = X_test.reshape(-1, 28, 28,1)
#print(X_train.shape,y_train.shape)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print(X_train.shape,y_train.shape)
model = Sequential()
#layer2
model.add(Conv2D(6, (3,3),strides=(1,1),input_shape=X_train.shape[1:],data_format='channels_last',activation='relu',kernel_initializer='uniform'))
#layer3
model.add(MaxPooling2D((2,2)))
#layer4
model.add(Conv2D(16, (3,3),strides=(1,1),data_format='channels_last',padding='valid',activation='relu',kernel_initializer='uniform'))
#layer5
model.add(MaxPooling2D(2,2))
#layer6
model.add(Conv2D(120, (5,5),strides=(1,1),data_format='channels_last',padding='valid',activation='relu',kernel_initializer='uniform'))
model.add(Flatten())
#layer7
model.add(Dense(84,activation='relu'))
#layer8
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from keras.utils import plot_model
plot_model(model,to_file='example.png',show_shapes=True)
lena = mpimg.imread('example.png') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴plt.show()
for _ in range(len(model.layers)):
	print (model.layers[_])
print("______________________________________")
print(model.inputs)
print("______________________________________")
print(model.outputs)
print("______________________________________")
#config = model.get_config()
#model = model.from_config(config)
#from keras.models import model_from_json
#json_string = model.to_json()
#print(json_string)
#print("______________________________________")
#model = model_from_json(json_string)
print("train____________")
model.fit(X_train,y_train,epochs=10,batch_size=128,)
print("test_____________")
loss,acc=model.evaluate(X_test,y_test)
print("loss=",loss)
print("accuracy=",acc)









from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import scipy.ndimage
batch_size = 32
num_classes = 10
epochs = 200
data_augmentation = True

# 数据载入
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# 多分类标签生成AEAAA-AEHYH-V2HCG-B5ORQ
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 网络结构配置
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
								 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()
# 训练参数设置
# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
							optimizer=opt,
							metrics=['accuracy'])

# 生成训练数据

import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from keras.utils import plot_model
model.summary 
plot_model(model,to_file='example.svg')
plot_model(model,to_file='example.png')
lena = mpimg.imread('example.png') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理Nginx 
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
#plt.show()
#from IPython.display import SVG, display
#display(SVG('example.svg'))
#model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validtion_data=(x_test,y_test),shuffle=True)
print("train____________")
model.fit(x_train,y_train,epochs=1,batch_size=64,)
print("test_____________")
loss,acc=model.evaluate(y_test,y_test)
print("loss=",loss)
print("accuracy=",acc)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from keras.utils import plot_model
plot_model(model,to_file='example.png',show_shapes=True)
lena = mpimg.imread('example.png') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()
'''
if not data_augmentation:
		print('Not using data augmentation.')
		model.fit(x_train, y_train,
							batch_size=batch_size,
							epochs=epochs,
							validation_data=(x_test, y_test),
							shuffle=True)
else:
		print('Using real-time data augmentation.')
		# This will do preprocessing and realtime data augmentation:
		datagen = ImageDataGenerator(
				featurewise_center=False,  # set input mean to 0 over the dataset
				samplewise_center=False,  # set each sample mean to 0
				featurewise_std_normalization=False,  # divide inputs by std of the dataset
				samplewise_std_normalization=False,  # divide each input by its std
				zca_whitening=False,  # apply ZCA whitening
				rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
				width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
				horizontal_flip=True,  # randomly flip images
				vertical_flip=False)  # randomly flip images

		# Compute quantities required for feature-wise normalization
		# (std, mean, and principal components if ZCA whitening is applied).
		datagen.fit(x_train)

# fit训练
		# Fit the model on the batches generated by datagen.flow().
		model.fit_generator(datagen.flow(x_train, y_train,
																		 batch_size=batch_size),
												steps_per_epoch=x_train.shape[0] // batch_size,
												epochs=epochs,
												validation_data=(x_test, y_test))
'''#from __future__ import print_function
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalMaxPooling2D
#from lsuv_init import LSUVinit

batch_size = 32 
num_classes = 10
epochs = 1600
data_augmentation = True

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train[1])
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("y_train's shape=",y_train.shape)
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(48, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(48, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(80, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',))
model.add(Activation('relu'))
model.add(GlobalMaxPooling2D())
model.add(Dropout(0.25))

#model.add(ZeroPadding2D((1, 1)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dropout(0.2))
'''
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
'''

model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from keras.utils import plot_model
plot_model(model,to_file='example.png',show_shapes=True)
lena = mpimg.imread('example.png') # lena.png

lena.shape #(512, 512, 3)
plt.imshow(lena) # 
plt.axis('off') # 
#plt.show()
# initiate RMSprop optimizer
opt = keras.optimizers.Adam(lr=0.0001)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
							optimizer=opt,
							metrics=['accuracy'])


#model = LSUVinit(model,x_train[:batch_size,:,:,:]) 
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph2', histogram_freq=0, write_graph=True, write_images=True)

if not data_augmentation:
		print('Not using data augmentation.')
		model.fit(x_train, y_train,
							batch_size=batch_size,
							epochs=epochs,
							validation_data=(x_test, y_test),
							shuffle=True, callbacks=[tbCallBack])
else:
		print('Using real-time data augmentation.')
		# This will do preprocessing and realtime data augmentation:
		'''
		datagen = ImageDataGenerator(
				featurewise_center=False,  # set input mean to 0 over the dataset
				samplewise_center=False,  # set each sample mean to 0
				featurewise_std_normalization=False,  # divide inputs by std of the dataset
				samplewise_std_normalization=False,  # divide each input by its std
				zca_whitening=False,  # apply ZCA whitening
				rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
				width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
				horizontal_flip=True,  # randomly flip images
				vertical_flip=False)  # randomly flip images
		'''
		datagen = ImageDataGenerator(
				featurewise_center=False,  # set input mean to 0 over the dataset
				samplewise_center=False,  # set each sample mean to 0
				featurewise_std_normalization=False,  # divide inputs by std of the dataset
				samplewise_std_normalization=False,  # divide each input by its std
				zca_whitening=False,  # apply ZCA whitening
				rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
				width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
				horizontal_flip=True,  # randomly flip images
				vertical_flip=False)  # randomly flip images


		# Compute quantities required for feature-wise normalization
		# (std, mean, and principal components if ZCA whitening is applied).
		datagen.fit(x_train)

		# Fit the model on the batches generated by datagen.flow().
		model.fit_generator(datagen.flow(x_train, y_train,
																		 batch_size=batch_size),
												steps_per_epoch=x_train.shape[0] // batch_size,
												epochs=epochs,
												validation_data=(x_test, y_test), callbacks=[tbCallBack])from fastai import *
from fastai.vision import *
from fastai.vision.models.wrn import wrn_22

torch.backends.cudnn.benchmark = True

path = untar_data(URLs.CIFAR)
ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=512).normalize(cifar_stats)

learn = Learner(data, wrn_22(), metrics=accuracy).to_fp16()
learn.fit_one_cycle(30, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)

# with mixup
learn = Learner(data, wrn_22(), metrics=accuracy).to_fp16().mixup()
learn.fit_one_cycle(24, 3e-3, wd=0.2, div_factor=10, pct_start=0.5)#!/usr/bin/python

import tensorflow as tf
import numpy as np

#from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import scipy.ndimage
import  os
batch_size = 32  # batch size
num_category = 10  # total categorical factor
#num_cont = 2  # total continuous factor
num_dim = 50  # total latent dimension
T_num = 25
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_flag = True
sample_flag = True
load_flag = False
save_flag = True

z_cat = tf.random_uniform([batch_size],minval=0,maxval=10,dtype=tf.int32)
z_cat = tf.one_hot(z_cat, num_category)

multi_dist = tf.contrib.distributions.StudentT(df=[2.0]*40,loc=[0.0]*40,scale=[1.0]*40)
tmp_noise  = multi_dist.sample([batch_size])   #[32,40]
sess = tf.InteractiveSession()  # 创建一个新的计算图
sess.run(tf.global_variables_initializer())  # 初始化所有参数
#z_cont = z[:, num_category:num_category+num_cont]
px = sess.run(tmp_noise)
py = sess.run(tmp_noise)
px=px.flatten()
py=py.flatten()
np.set_printoptions(threshold=np.nan)
print('pxshape=',px)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
#z = tf.random_normal([2, 3])
color = np.arctan2(px, py)
plt.figure('gaoshi')
plt.scatter(px, py, s = 10, c = color, alpha = 0.7)
# 设置坐标轴范围

plt.xlim((-10, 10))
plt.ylim((-10,10 ))

# 不显示坐标轴的值
#plt.xticks(())
#plt.yticks(())
plt.savefig("noise")



print(z_cat)
print(tmp_noise)
noise = tmp_noise
#multi_dist = tf.random_normal([batch_size,40],mean=0.0,stddev=1.0)
#tmp_noise =multi_dist
with tf.variable_scope('weight'):
    mu = tf.get_variable("mu",[T_num,batch_size,40],initializer=tf.random_uniform_initializer(-1,1))
    sigma = tf.get_variable("sigma",[T_num,batch_size,40],initializer=tf.constant_initializer([0.05]))
    noise = tmp_noise*mu+sigma
    weight = tf.get_variable("weight",[T_num,1,1],initializer=tf.constant_initializer(2))   #5个T分布[5,1,1]
    weight = tf.tile(weight,[1,batch_size,40])   #weight扩到每个参数上[5,32,40]
#	noise = noise * weight
    noise = noise * weight
    noise = tf.reduce_mean(noise,axis=0)

'''
	w_noise=tf.transpose(noise, perm=[1, 0, 2]) 
	w_noise=tf.reshape(w_noise, [batch_size, -1])

	h_w = tf.layers.dense(w_noise , 640 , activation=tf.nn.relu)
	h_w2 = tf.layers.dense(h_w , 320 , activation=tf.nn.relu)

	h_w3= tf.layers.dense(h_w2 , T_num , activation=tf.nn.sigmoid) 
	h_w3=tf.reshape(h_w3 , [batch_size , T_num , 1])
	h_w3= tf.tile(h_w3 , [1 , 1 , 40 ]) 
	w_noise=tf.reshape(w_noise, [batch_size, T_num ,40]) 
	w_noise= w_noise*h_w3

	w_noise = tf.reduce_mean(w_noise,axis=1)'''

z = tf.concat([z_cat,noise],1)
x=z
y=x
sess = tf.InteractiveSession()  # 创建一个新的计算图
sess.run(tf.global_variables_initializer())  # 初始化所有参数
#z_cont = z[:, num_category:num_category+num_cont]
px = sess.run(x)
py = sess.run(y)
px=px.flatten()
py=py.flatten()
np.set_printoptions(threshold=np.nan)
print('pxshape=',px)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
#z = tf.random_normal([2, 3])
color = np.arctan2(px, py)
plt.figure('afterT')
plt.scatter(px, py, s = 10, c = color, alpha = 0.7)
# 设置坐标轴范围

plt.xlim((-10, 10))
plt.ylim((-10,10 ))

# 不显示坐标轴的值

plt.savefig("cmcT")
##高斯
n = 1024    # data size
px = np.random.normal(-1, 1, n) # 每一个点的X值
py = np.random.normal(-1, 1, n) # 每一个点的Y值
np.set_printoptions(threshold=np.nan)
print('pxshape=',px)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
#z = tf.random_normal([2, 3])
color = np.arctan2(px, py)
plt.figure('gaosi')
plt.scatter(px, py, s = 10, c = color, alpha = 0.7)
# 设置坐标轴范围

plt.xlim((-10, 10))
plt.ylim((-10,10 ))

# 不显示坐标轴的值

plt.savefig('gaoshi')

'''
true_images = tf.placeholder(tf.float32, [batch_size,28,28,1])
true_labels = tf.placeholder(tf.float32, [batch_size,num_category])

with tf.variable_scope('generator'):
	print("generator")
	h0 = tf.layers.dense(z,1024)
	h0 = tf.nn.relu(tf.layers.batch_normalization(h0,training = True))
	print(h0.shape)
	
	h1 = tf.layers.dense(h0,7*7*32)
	h1 = tf.nn.relu(tf.layers.batch_normalization(h1,training = True))
	h1 = tf.reshape(h1,[-1,7,7,32])
	print(h1.shape)

	h2 = tf.layers.conv2d_transpose(h1,16,[4,4],strides=2,padding="same")
	h2 = tf.nn.relu(tf.layers.batch_normalization(h2,training = True))
	print(h2.shape)

	h3 = tf.layers.conv2d_transpose(h2,1,[4,4],strides=2,padding="same")
	h3 = tf.nn.sigmoid(h3)
	print(h3.shape)

def d(xx,reuse=False):
	with tf.variable_scope('discriminator', reuse=reuse):
		print("discriminator")
		h0 = tf.layers.conv2d(xx,16,[4,4],strides=2,padding="same")
		h0 = tf.nn.crelu(h0)
		print(h0.shape)

		h1 = tf.layers.conv2d(h0,32,[4,4],strides=2,padding="same")
		h1 = tf.nn.crelu(tf.layers.batch_normalization(h1,training = True))
		print(h1.shape)

		h12 = tf.layers.conv2d(h1,64,[4,4],strides=2,padding="same")
		h12 = tf.nn.crelu(tf.layers.batch_normalization(h12,training = True))
		print(h12.shape)

		h2 = tf.nn.max_pool(h1, ksize=[1,4,4,1], strides=[1,1,1,1],padding='VALID')
		h2 = tf.contrib.layers.flatten(h1)
		h2 = tf.layers.dense(h2,1024)
		h2 = tf.nn.crelu(tf.layers.batch_normalization(h2,training = True))
		print(h2.shape)

		disc = tf.layers.dense(h2,1)
		disc = tf.squeeze(disc)
		print(disc.shape)

		h3 = tf.layers.dense(h2,128)
		h3 = tf.nn.crelu(h3)

		class_cat = tf.layers.dense(h3,10)
		#class_cont = tf.layers.dense(h3,2)
		#class_cont = tf.nn.sigmoid(class_cont)
	return disc, class_cat#, class_cont

def merge_images(images):
	ret = np.zeros((8,8,28,28))  
	for i in range(8):
		for j in range(8):
			ret[i][j] = images[i*8+j].reshape(28,28)
	return ret

real_disc,real_class = d(true_images)
fake_disc,fake_class = d(h3,reuse = True)

loss_disc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones([batch_size]),logits=real_disc)) + \
			tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros([batch_size]),logits=fake_disc))
loss_class = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_labels,logits=real_class)) + \
			 tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=z_cat,logits=fake_class)) #+ \
			 #tf.reduce_mean(tf.nn.l2_loss(z_cont-fake_cont))
loss_gen = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones([batch_size]),fake_disc))

disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
train_disc = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9).minimize(loss_disc+loss_class,var_list = disc_vars)

gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator") or var.name.startswith("weight")]
train_gen = tf.train.AdamOptimizer(learning_rate=0.001, beta1 = 0.9).minimize(loss_gen+loss_class,var_list = gen_vars)

(train_data, train_labels), (x_test, y_test) = mnist.load_data()

train_data=train_data.reshape(-1,28,28,1)
print (train_data.shape)
x_test = x_test.astype('float32')
# 多分类标签生成
train_labels = keras.utils.to_categorical(train_labels, 10)
y_test = keras.utils.to_categorical(y_test, 10)

epoch_num = 10									#epoch_num遍历数据集的次数
iteration_num = train_data.shape[0]//batch_size #iteration_num=训练数据/batch_size
num_steps = 10000 

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
'''

#PAC



#
#if load_flag:
#	saver.restore(sess,"checkpoint/")
#i = 0
#if train_flag:
#	for epoch in range(epoch_num):		#epoch_num遍历数据集的次数
#		for it_n in range(iteration_num):	#iteration_num=batch_size
#			batch_images = train_data[it_n*batch_size:(it_n+1)*batch_size]
#			batch_labels = train_labels[it_n*batch_size:(it_n+1)*batch_size]
#			
#			l_disc = sess.run([loss_disc,train_disc],feed_dict={ true_images:batch_images,true_labels:batch_labels})[0]
#			l_gen = sess.run([loss_gen,train_gen])[0]
#			i= i+1
#			if i % 1000 == 0 or i == 1:
#				print("epoch:",epoch,"interation_num:",it_n,"l_disc:",l_disc,"l_gen:",l_gen)
#
#	if save_flag:
#		saver.save(sess,"checkpoint/")
#
#if sample_flag:
#	for i in range(100):
#		images = sess.run(h3).reshape(-1,28,28)
#		images_1 = sess.run(h3).reshape(-1,28,28)
#		last = np.concatenate((images,images_1),axis=0).reshape(8,8,28,28)
#		last_image = np.zeros((28*8,28*8))
#		for _ in range(8):
#			for __ in range(8):
#				last_image[_*28:(_+1)*28, __*28:(__+1)*28] = last[_][__]
#		print(last_image.shape)
#		imsave("samples/mnist/64_%d.png"%i,last_image)
#
#
#	labels = np.argmax(sess.run(z_cat),axis=1)
#
#	for i in range(32):
#		print("samples/mnist/%d.png %d"%(i,labels[i]))
#		imsave("samples/mnist/%d.png"%(i),images[i])
#		#imsave("samples")"""Simple convolutional neural network classififer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.flags.FLAGS

def get_params():
    """Model params."""
    return {
        "drop_rate": 0.5
    }

def model(features, labels, mode, params):
    """CNN classifier model."""
    images = features["image"]
    labels = labels["label"]

    tf.summary.image("images", images)

    drop_rate = params.drop_rate if mode == tf.estimator.ModeKeys.TRAIN else 0.0

    features = images
    for i, filters in enumerate([32, 64, 128]):
        features = tf.layers.conv2d(
            features, filters=filters, kernel_size=3, padding="same",
            name="conv_%d" % (i + 1))
        features = tf.layers.max_pooling2d(
            inputs=features, pool_size=2, strides=2, padding="same",
            name="pool_%d" % (i + 1))

    features = tf.contrib.layers.flatten(features)

    features = tf.layers.dropout(features, drop_rate)
    features = tf.layers.dense(features, 512, name="dense_1")

    features = tf.layers.dropout(features, drop_rate)
    logits = tf.layers.dense(features, params.num_classes, activation=None,
                             name="dense_2")

    predictions = tf.argmax(logits, axis=1)

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)

    return {"predictions": predictions}, loss

def eval_metrics(unused_params):
    """Eval metrics."""
    return {
        "accuracy": tf.contrib.learn.MetricSpec(tf.metrics.accuracy)
    }
# 多分类问题
model.compile(optimizer='rmsprop',
							loss='categorical_crossentropy',
							metrics=['accuracy'])

# 二分类问题
model.compile(optimizer='rmsprop',
							loss='binary_crossentropy',
							metrics=['accuracy'])

# 均方误差回归问题
model.compile(optimizer='rmsprop',
							loss='mse')

# 自定义评估标准函数
import keras.backend as K

def mean_pred(y_true, y_pred):
		return K.mean(y_pred)

model.compile(optimizer='rmsprop',
							loss='binary_crossentropy',
							metrics=['accuracy', mean_pred])
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import scipy.ndimage
if not data_augmentation:
		print('Not using data augmentation.')
		model.fit(x_train, y_train,
							batch_size=batch_size,
							epochs=epochs,
							validation_data=(x_test, y_test),
							shuffle=True)
else:
		print('Using real-time data augmentation.')
		# This will do preprocessing and realtime data augmentation:
		datagen = ImageDataGenerator(
				featurewise_center=False,  # set input mean to 0 over the dataset
				samplewise_center=False,  # set each sample mean to 0
				featurewise_std_normalization=False,  # divide inputs by std of the dataset
				samplewise_std_normalization=False,  # divide each input by its std
				zca_whitening=False,  # apply ZCA whitening
				rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
				width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
				horizontal_flip=True,  # randomly flip images
				vertical_flip=False)  # randomly flip images

		# Compute quantities required for feature-wise normalization
		# (std, mean, and principal components if ZCA whitening is applied).
		datagen.fit(x_train)

# fit训练
		# Fit the model on the batches generated by datagen.flow().
		model.fit_generator(datagen.flow(x_train, y_train,
																		 batch_size=batch_size),
												steps_per_epoch=x_train.shape[0] // batch_size,
												epochs=epochs,
												validation_data=(x_test, y_test))from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,default='cifar10', help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot',default='./', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
	os.makedirs(opt.outf)
except OSError:
	pass

if opt.manualSeed is None:
	opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
	# folder dataset
	dataset = dset.ImageFolder(root=opt.dataroot,
							   transform=transforms.Compose([
								   transforms.Resize(opt.imageSize),
								   transforms.CenterCrop(opt.imageSize),
								   transforms.ToTensor(),
								   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
							   ]))
elif opt.dataset == 'lsun':
	dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
						transform=transforms.Compose([
							transforms.Resize(opt.imageSize),
							transforms.CenterCrop(opt.imageSize),
							transforms.ToTensor(),
							transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
						]))
elif opt.dataset == 'cifar10':
	dataset = dset.CIFAR10(root=opt.dataroot, download=True,
						   transform=transforms.Compose([
							   transforms.Resize(opt.imageSize),
							   transforms.ToTensor(),
							   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
						   ]))
elif opt.dataset == 'fake':
	dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
							transform=transforms.ToTensor())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
										 shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3


# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


class Generator(nn.Module):
	def __init__(self, ngpu):
		super(Generator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),
			# state size. (ngf) x 32 x 32
			nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
			nn.Tanh()
			# state size. (nc) x 64 x 64
		)

	def forward(self, input):
		if input.is_cuda and self.ngpu > 1:
			output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
		else:
			output = self.main(input)
		return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
	netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
	def __init__(self, ngpu):
		super(Discriminator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
		)

	def forward(self, input):
		if input.is_cuda and self.ngpu > 1:
			output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
		else:
			output = self.main(input)

		return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
	netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
	for i, data in enumerate(dataloader, 0):
		############################
		# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
		###########################
		# train with real
		netD.zero_grad()
		real_cpu = data[0].to(device)
		batch_size = real_cpu.size(0)
		label = torch.full((batch_size,), real_label, device=device)

		output = netD(real_cpu)
		errD_real = criterion(output, label)
		errD_real.backward()
		D_x = output.mean().item()

		# train with fake
		noise = torch.randn(batch_size, nz, 1, 1, device=device)
		fake = netG(noise)
		label.fill_(fake_label)
		output = netD(fake.detach())
		errD_fake = criterion(output, label)
		errD_fake.backward()
		D_G_z1 = output.mean().item()
		errD = errD_real + errD_fake
		optimizerD.step()

		############################
		# (2) Update G network: maximize log(D(G(z)))
		###########################
		netG.zero_grad()
		label.fill_(real_label)  # fake labels are real for generator cost
		output = netD(fake)
		errG = criterion(output, label)
		errG.backward()
		D_G_z2 = output.mean().item()
		optimizerG.step()

		print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
			  % (epoch, opt.niter, i, len(dataloader),
				 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
		if i % 100 == 0:
			vutils.save_image(real_cpu,
					'%s/real_samples.png' % opt.outf,
					normalize=True)
			fake = netG(fixed_noise)
			vutils.save_image(fake.detach(),
					'%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
					normalize=True)

	# do checkpointing
	torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
	torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))from fastai import *
from fastai.vision import *

path = untar_data(URLs.DOGS)
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)

learn = create_cnn(data, models.resnet34, metrics=accuracy)
learn.fit_one_cycle(1)

learn.unfreeze()
learn.fit_one_cycle(6, slice(1e-5,3e-4), pct_start=0.05)

accuracy(*learn.TTA())
"""
To know more or get code samples, please visit my website:
https://morvanzhou.github.io/tutorials/
Or search: 莫烦Python
Thank you for supporting!
"""

# please note, all tutorial code are running under python3.5.
# If you use the version like python2.7, please modify the code accordingly

# 8 - RNN LSTM Regressor example

# to try tensorflow, un-comment following two lines
# import os
# os.environ['KERAS_BACKEND']='tensorflow'
import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006


def get_batch():
	global BATCH_START, TIME_STEPS
	# xs shape (50batch, 20steps)
	xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
	seq = np.sin(xs)
	res = np.cos(xs)
	BATCH_START += TIME_STEPS
	# plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
	# plt.show()
	return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

model = Sequential()
# build a LSTM RNN
model.add(LSTM(
	batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
	output_dim=CELL_SIZE,
	return_sequences=True,      # True: output at all steps. False: output as last step.
	stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
))
# add output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
adam = Adam(LR)
model.compile(optimizer=adam,
			  loss='mse',)

print('Training ------------')
for step in range(501):
	# data shape = (batch_num, steps, inputs/outputs)
	X_batch, Y_batch, xs = get_batch()
	cost = model.train_on_batch(X_batch, Y_batch)
	pred = model.predict(X_batch, BATCH_SIZE)
	plt.plot(xs[0, :], Y_batch[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
	plt.ylim((-1.2, 1.2))
	plt.draw()
	plt.pause(0.1)
	if step % 10 == 0:
		print('train cost: ', cost)

"""
To know more or get code samples, please visit my website:
https://morvanzhou.github.io/tutorials/
Or search: 莫烦Python
Thank you for supporting!
"""

# please note, all tutorial code are running under python3.5.
# If you use the version like python2.7, please modify the code accordingly

# 6 - CNN example

# to try tensorflow, un-comment following two lines
# import os
# os.environ['KERAS_BACKEND']='tensorflow'

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 1,28, 28)/255.
X_test = X_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# Another way to build your CNN
model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
		batch_input_shape=(None, 1, 28, 28),
		filters=32,
		kernel_size=5,
		strides=1,
		padding='same',     # Padding method
		data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
		pool_size=2,
		strides=2,
		padding='same',    # Padding method
		data_format='channels_first',
))

# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer='adam',
							loss='categorical_crossentropy',
							metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=1, batch_size=64,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)#-*- coding: UTF-8 -*-
"""
Environment: Keras2.0.5，Python2.7
Model: ResNet
"""

from __future__ import division
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.utils import plot_model
import six



def _handle_dim_ordering():
	global ROW_AXIS
	global COL_AXIS
	global CHANNEL_AXIS
	if K.image_dim_ordering() == 'tf':
		ROW_AXIS = 1
		COL_AXIS = 2
		CHANNEL_AXIS = 3
	else:
		CHANNEL_AXIS = 1
		ROW_AXIS = 2
		COL_AXIS = 3



def _get_block(identifier):
	if isinstance(identifier, six.string_types):
		res = globals().get(identifier)
		if not res:
			raise ValueError('Invalid {}'.format(identifier))
		return res
	return identifier



def _bn_relu(input):
	"""
	Helper to build a BN -> relu block
	"""

	norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
	return Activation("relu")(norm)



def _conv_bn_relu(**conv_params):

	"""
	Helper to build a conv -> BN -> relu block
	"""

	filters = conv_params["filters"]
	kernel_size = conv_params["kernel_size"]
	strides = conv_params.setdefault("strides", (1, 1))
	kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
	padding = conv_params.setdefault("padding", "same")
	kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

	def f(input):
		conv = Conv2D(filters=filters, kernel_size=kernel_size,strides=strides, padding=padding,kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(input)
		return _bn_relu(conv)
	return f



def _bn_relu_conv(**conv_params):

	"""
	Helper to build a BN -> relu -> conv block.
	This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
	"""

	filters = conv_params["filters"]
	kernel_size = conv_params["kernel_size"]
	strides = conv_params.setdefault("strides", (1, 1))
	kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
	padding = conv_params.setdefault("padding", "same")
	kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

	def f(input):
		activation = _bn_relu(input)
		return Conv2D(filters=filters, kernel_size=kernel_size,strides=strides, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(activation)
	return f



def _shortcut(input, residual):

	"""
	Adds a shortcut between input and residual block and merges them with "sum"
	"""
	# Expand channels of shortcut to match residual.
	# Stride appropriately to match residual (width, height)
	# Should be int if network architecture is correctly configured.

	input_shape = K.int_shape(input)
	residual_shape = K.int_shape(residual)
	stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
	stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
	equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

	shortcut = input

	# 1 X 1 conv if shape is different. Else identity.
	if stride_width > 1 or stride_height > 1 or not equal_channels:
		shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS], kernel_size=(1, 1), strides=(stride_width, stride_height), padding="valid", kernel_initializer="he_normal",kernel_regularizer=l2(0.0001))(input)
	return add([shortcut, residual])



def _residual_block(block_function, filters, repetitions, is_first_layer=False):

	"""
	Builds a residual block with repeating bottleneck blocks.
	"""

	def f(input):
		for i in range(repetitions):
			init_strides = (1, 1)
			if i == 0 and not is_first_layer:
				init_strides = (2, 2)
			input = block_function(filters=filters, init_strides=init_strides, is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
		return input
	return f




def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):

	"""
	Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
	Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
	"""

	def f(input):
		if is_first_block_of_first_layer:
			# don't repeat bn->relu since we just did bn->relu->maxpool
			conv1 = Conv2D(filters=filters, kernel_size=(3, 3),strides=init_strides, padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(input)
		else:
			conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3), strides=init_strides)(input)
		residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
		return _shortcut(input, residual)
	return f



def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):

	"""
	Bottleneck architecture for > 34 layer resnet.
	Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
	Returns:
		A final conv layer of filters * 4
	"""

	def f(input):
		if is_first_block_of_first_layer:
			# don't repeat bn->relu since we just did bn->relu->maxpool
			conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1), strides=init_strides, padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(input)
		else:
			conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1), strides=init_strides)(input)

		conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
		residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
		return _shortcut(input, residual)
	return f



class ResnetBuilder(object):
	@staticmethod
	def build(input_shape, num_outputs, block_fn, repetitions):
		"""
		Builds a custom ResNet like architecture.
		Args:
			input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
			num_outputs: The number of outputs at final softmax layer
			block_fn: The block function to use. This is either `basic_block` or `bottleneck`.The original paper used basic_block for layers < 50
			repetitions: Number of repetitions of various block units.At each block unit, the number of filters are doubled and the input size is halved
		Returns:
			The keras `Model`.
		"""

		_handle_dim_ordering()

		if len(input_shape) != 3:
			raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

		# Permute dimension order if necessary
		if K.image_dim_ordering() == 'tf':
			input_shape = (input_shape[1], input_shape[2], input_shape[0])

		# Load function from str if needed.
		block_fn = _get_block(block_fn)

		input = Input(shape=input_shape)
		conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
		pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

		block = pool1
		filters = 64
		for i, r in enumerate(repetitions):
			block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
			filters *= 2

		# Last activation
		block = _bn_relu(block)

		# Classifier block
		block_shape = K.int_shape(block)
		pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]), strides=(1, 1))(block)
		flatten1 = Flatten()(pool2)
		dense = Dense(units=num_outputs, kernel_initializer="he_normal", activation="softmax")(flatten1)

		model = Model(inputs=input, outputs=dense)
		return model


	@staticmethod
	def build_resnet_18(input_shape, num_outputs):
		return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

	@staticmethod
	def build_resnet_34(input_shape, num_outputs):
		return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

	@staticmethod
	def build_resnet_50(input_shape, num_outputs):
		return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

	@staticmethod
	def build_resnet_101(input_shape, num_outputs):
		return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

	@staticmethod
	def build_resnet_152(input_shape, num_outputs):
		return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])



def check_print():
	# Create a Keras Model
	model=ResnetBuilder.build_resnet_34((3, 224, 224), 100)
	model.summary()

	# Save a PNG of the Model Build
	plot_model(model,to_file='ResNet.png')

	model.compile(optimizer='sgd',loss='categorical_crossentropy')
	print ('Model Compiled')



if __name__=='__main__':
	check_print() from keras.layers import Input, Dense
from keras.models import Model

# 这部分返回一个张量
inputs = Input(shape=(784,))

# 层的实例是可调用的，它以张量为参数，并且返回一个张量
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# 这部分创建了一个包含输入层和三个全连接层的模型
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
							loss='categorical_crossentropy',
							metrics=['accuracy'])
model.fit(data, labels)  # 开始训练如何「冻结」网络层？
「冻结」一个层意味着将其排除在训练之外，即其权重将永远不会更新。这在微调模型或使用固定的词向量进行文本输入中很有用。

您可以将 trainable 参数（布尔值）传递给一个层的构造器，以将该层设置为不可训练的：

frozen_layer = Dense(32, trainable=False)
另外，可以在实例化之后将网络层的 trainable 属性设置为 True 或 False。为了使之生效，在修改 trainable 属性之后，需要在模型上调用 compile()。这是一个例子：

x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
# 在下面的模型中，训练期间不会更新层的权重
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
trainable_model = Model(x, y)
# 使用这个模型，训练期间 `layer` 的权重将被更新
# (这也会影响上面的模型，因为它使用了同一个网络层实例)
trainable_model.compile(optimizer='rmsprop', loss='mse')

frozen_model.fit(data, labels)  # 这不会更新 `layer` 的权重
trainable_model.fit(data, labels)  # 这会更新 `layer` 的权重

#coding=utf-8  
from keras.models import Model  
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate  
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D  
import numpy as np  
seed = 7  
np.random.seed(seed)  
  
def Conv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):  
	if name is not None:  
		bn_name = name + '_bn'  
		conv_name = name + '_conv'  
	else:  
		bn_name = None  
		conv_name = None  
  
	x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
	x = BatchNormalization(axis=3,name=bn_name)(x)  
	return x  
  
def Inception(x,nb_filter):  
	branch1x1 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)  
  
	branch3x3 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)  
	branch3x3 = Conv2d_BN(branch3x3,nb_filter,(3,3), padding='same',strides=(1,1),name=None)  
  
	branch5x5 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)  
	branch5x5 = Conv2d_BN(branch5x5,nb_filter,(1,1), padding='same',strides=(1,1),name=None)  
  
	branchpool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)  
	branchpool = Conv2d_BN(branchpool,nb_filter,(1,1),padding='same',strides=(1,1),name=None)  
  
	x = concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=3)  
  
	return x  
  
inpt = Input(shape=(224,224,3))  
#padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))  
x = Conv2d_BN(inpt,64,(7,7),strides=(2,2),padding='same')  
x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)  
x = Conv2d_BN(x,192,(3,3),strides=(1,1),padding='same')  
x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)  
x = Inception(x,64)#256  
x = Inception(x,120)#480  
x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)  
x = Inception(x,128)#512  
x = Inception(x,128)  
x = Inception(x,128)  
x = Inception(x,132)#528  
x = Inception(x,208)#832  
x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)  
x = Inception(x,208)  
x = Inception(x,256)#1024  
x = AveragePooling2D(pool_size=(7,7),strides=(7,7),padding='same')(x)  
x = Dropout(0.4)(x)  
x = Dense(1000,activation='relu')(x)  
x = Dense(1000,activation='softmax')(x)  
model = Model(inpt,x,name='inception')  
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
model.summary()  import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))import numpy as np
import json

from keras.utils.data_utils import get_file
from keras import backend as K

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def decode_predictions(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        results.append(result)
    return results
# -*- coding: utf-8 -*-
"""Inception-ResNet V2 model for Keras.

Model naming and structure follows TF-slim implementation (which has some additional
layers and different number of filters from the original arXiv paper):
https://github.com/tensorflow/models/blob/master/slim/nets/inception_resnet_v2.py

Pre-trained ImageNet weights are also converted from TF-slim, which can be found in:
https://github.com/tensorflow/models/tree/master/slim#pre-trained-models

# Reference
- [Inception-v4, Inception-ResNet and the Impact of
   Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

"""
from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


BASE_WEIGHT_URL = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.7/'


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.

    This function applies the "Inception" preprocessing which converts
    the RGB values from [0, 255] to [-1, 1]. Note that this preprocessing
    function is different from `imagenet_utils.preprocess_input()`.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds a Inception-ResNet block.

    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`

    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch. Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names. The Inception-ResNet blocks
            are repeated many times in this network. We use `block_idx` to identify
            each of the repetitions. For example, the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`, ane the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](keras./activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).

    # Returns
        Output tensor for the block.

    # Raises
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    """
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    mixed = Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=block_name)([x, up])
    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(x)
    return x


def InceptionResNetV2(include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=1000):
    """Instantiates the Inception-ResNet v2 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that when using TensorFlow, for best performance you should
    set `"image_data_format": "channels_last"` in your Keras config
    at `~/.keras/keras.json`.

    The model and the weights are compatible with both TensorFlow and Theano
    backends (but not CNTK). The data format convention used by the model is
    the one specified in your Keras config file.

    Note that the default input image size for this model is 299x299, instead
    of 224x224 as in the VGG16 and ResNet models. Also, the input preprocessing
    function is different (i.e., do not use `imagenet_utils.preprocess_input()`
    with this model. Use `preprocess_input()` defined in this module instead).

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or `'imagenet'` (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(299, 299, 3)` (with `'channels_last'` data format)
            or `(3, 299, 299)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional layer.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.

    # Returns
        A Keras `Model` instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with an unsupported backend.
    """
    if K.backend() in {'cntk'}:
        raise RuntimeError(K.backend() + ' backend is currently unsupported for this model.')

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=False,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 32, 3, strides=2, padding='valid')
    x = conv2d_bn(x, 32, 3, padding='valid')
    x = conv2d_bn(x, 64, 3)
    x = MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn(x, 80, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, padding='valid')
    x = MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1)
    branch_1 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b')

    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model
    model = Model(inputs, x, name='inception_resnet_v2')

    # Load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        if include_top:
            weights_filename = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
            weights_path = get_file(weights_filename,
                                    BASE_WEIGHT_URL + weights_filename,
                                    cache_subdir='models',
                                    md5_hash='e693bd0210a403b3192acc6073ad2e96')
        else:
            weights_filename = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
            weights_path = get_file(weights_filename,
                                    BASE_WEIGHT_URL + weights_filename,
                                    cache_subdir='models',
                                    md5_hash='d19885ff4a710c122648d3b5c3b684e4')
        model.load_weights(weights_path)

    return model


if __name__ == '__main__':
    model = InceptionResNetV2(include_top=True, weights='imagenet')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
# -*- coding: utf-8 -*-
"""Inception V3 model for Keras.

Note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function is also different (same as Xception).

# Reference

- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)

"""
from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing import image


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def InceptionV3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    """Instantiates the Inception v3 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input image size for this model is 299x299.

    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
        A Keras model instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=K.image_data_format(),
        include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = Input(tensor=input_tensor, shape=input_shape)

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='inception_v3')

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        if include_top:
            weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='9a0d58056eeedaa3f26cb7ebd46da564')
        else:
            weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='bcbd6486424b2319ff4ef7d526e38f63')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            convert_all_kernels_in_model(model)
    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = InceptionV3(include_top=True, weights='imagenet')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
#coding=utf-8  
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical  
import numpy as np  
seed = 7  
np.random.seed(seed)  
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = cifar10.load_data()  
model = Sequential()  
model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(227,227,3),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Flatten())  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(1000,activation='softmax'))  
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
model.summary()  
#coding=utf-8  
from keras.models import Sequential  
from keras.layers import Dense,Flatten  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical  
from keras.datasets import fashion_mnist
from keras.utils import np_utils,plot_model
import numpy as np  
seed = 7  
np.random.seed(seed)  
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print(X_train.shape,y_train.shape)
X_train = X_train.reshape(-1,1,28, 28)
X_test = X_test.reshape(-1,1,28, 28)
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print(X_train.shape,y_train.shape)
model = Sequential()  
model.add(Conv2D(6,(3,3),strides=(1,1),input_shape=(1,28,28),data_format='channels_first',padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(16,(3,3),strides=(1,1),data_format='channels_first',padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(120,(5,5),strides=(1,1),padding='valid',data_format='channels_first',activation='relu',kernel_initializer='uniform')) 
model.add(Flatten())    
model.add(Dense(84,activation='relu'))  
model.add(Dense(10,activation='softmax'))  
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])  
model.summary()  
plot_model(model,to_file='lenet.png')
  
print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=2, batch_size=64,)
print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
#show the architecture 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from keras.utils import np_utils,plot_model
plot_model(model,to_file='example.png')
lena = mpimg.imread('example.png') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()#coding=utf-8  
from keras.models import Sequential  
from keras.layers import Dense,Flatten  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical  
from keras.datasets import mnist
from keras.utils import np_utils
import keras
import numpy as np  
seed = 7  
np.random.seed(seed)  
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape,y_train.shape)
X_train = X_train.reshape(-1,28, 28,1)/255.
X_test = X_test.reshape(-1,28, 28,1)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print(X_train.shape,y_train.shape)
model = Sequential()  
model.add(Conv2D(32,(5,5),strides=(1,1),input_shape=(28,28,1),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(64,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Flatten())  
model.add(Dense(100,activation='relu'))  
model.add(Dense(10,activation='softmax'))  
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])  
model.summary()  
from keras.callbacks import TensorBoard

#model.fit(x_train,y_train,batch_size,epoch,)
print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=10, batch_size=64,verbose=1,callbacks=[TensorBoard(log_dir='./log_dir')])

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
#coding=utf-8  
from keras.models import Sequential  
from keras.layers import Dense,Flatten  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical  
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np  
seed = 7  
np.random.seed(seed)  
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape,y_train.shape)
X_train = X_train.reshape(-1,28, 28,1)/255.
X_test = X_test.reshape(-1,28, 28,1)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print(X_train.shape,y_train.shape)
model = Sequential()  
model.add(Conv2D(6,(3,3),strides=(1,1),input_shape=(28,28,1),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(16,(3,3),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(120,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform')) 
model.add(Flatten())    
model.add(Dense(84,activation='relu'))  
model.add(Dense(10,activation='softmax'))  
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])  
model.summary()  
  
print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=10, batch_size=64,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
from __future__ import print_function
import numpy as np
from keras.models import Model
from keras.layers import Dense, Convolution2D


def svd_orthonormal(shape):
	# Orthonorm init code is taked from Lasagne
	# https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
	if len(shape) < 2:
		raise RuntimeError("Only shapes of length 2 or more are supported.")
	flat_shape = (shape[0], np.prod(shape[1:]))
	a = np.random.standard_normal(flat_shape)
	u, _, v = np.linalg.svd(a, full_matrices=False)
	q = u if u.shape == flat_shape else v
	q = q.reshape(shape)
	return q


def get_activations(model, layer, X_batch):
	intermediate_layer_model = Model(
		inputs=model.get_input_at(0),
		outputs=layer.get_output_at(0)
	)
	activations = intermediate_layer_model.predict(X_batch)
	return activations


def LSUVinit(model, batch, verbose=True, margin=0.1, max_iter=10):
	# only these layer classes considered for LSUV initialization; add more if needed
	classes_to_consider = (Dense, Convolution2D)

	needed_variance = 1.0

	layers_inintialized = 0
	for layer in model.layers:
		if verbose:
			print(layer.name)
		if not isinstance(layer, classes_to_consider):
			continue
		# avoid small layers where activation variance close to zero, esp. for small batches
		if np.prod(layer.get_output_shape_at(0)[1:]) < 32:
			if verbose:
				print(layer.name, 'too small')
			continue
		if verbose:
			print('LSUV initializing', layer.name)

		layers_inintialized += 1
		weights_and_biases = layer.get_weights()
		weights_and_biases[0] = svd_orthonormal(weights_and_biases[0].shape)
		layer.set_weights(weights_and_biases)
		activations = get_activations(model, layer, batch)
		variance = np.var(activations)
		iteration = 0
		if verbose:
			print(variance)
		while abs(needed_variance - variance) > margin:
			if np.abs(np.sqrt(variance)) < 1e-7:
				# avoid zero division
				break

			weights_and_biases = layer.get_weights()
			weights_and_biases[0] /= np.sqrt(variance) / np.sqrt(needed_variance)
			layer.set_weights(weights_and_biases)
			weights /= np.sqrt(variance) / np.sqrt(needed_variance)
			layer.set_weights([weights, biases])
			activations = get_activations(model, layer, batch)
			variance = np.var(activations)

			iteration += 1
			if verbose:
				print(variance)
			if iteration >= max_iter:
				break
	if verbose:
		print('LSUV: total layers initialized', layers_inintialized)
	return model"""
To know more or get code samples, please visit my website:
https://morvanzhou.github.io/tutorials/
Or search: 莫烦Python
Thank you for supporting!
"""

# please note, all tutorial code are running under python3.5.
# If you use the version like python2.7, please modify the code accordingly

# 6 - CNN example

# to try tensorflow, un-comment following two lines
# import os
# os.environ['KERAS_BACKEND']='tensorflow'

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#print(y_train.shape)
# data pre-processing
X_train = X_train.reshape(-1, 1,28, 28)/255.
X_test = X_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
for _ in range(10):
	plt.imshow(X_train[_], interpolation='nearest', cmap='bone', origin='lower')
	plt.show()
# Another way to build your CNN
model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
		batch_input_shape=(None, 1, 28, 28),
		filters=32,
		kernel_size=5,
		strides=1,
		padding='same',     # Padding method
		data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
		pool_size=2,
		strides=2,
		padding='same',    # Padding method
		data_format='channels_first',
))

# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
							loss='categorical_crossentropy',
							metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=1, batch_size=64,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == '__main__':
    main()"""MNIST dataset preprocessing and specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy as np
import os
from six.moves import urllib
import struct
import tensorflow as tf

REMOTE_URL = "http://yann.lecun.com/exdb/mnist/"
LOCAL_DIR = "data/mnist/"
TRAIN_IMAGE_URL = "train-images-idx3-ubyte.gz"
TRAIN_LABEL_URL = "train-labels-idx1-ubyte.gz"
TEST_IMAGE_URL = "t10k-images-idx3-ubyte.gz"
TEST_LABEL_URL = "t10k-labels-idx1-ubyte.gz"

IMAGE_SIZE = 28
NUM_CLASSES = 10

def get_params():
    """Dataset params."""
    return {
        "num_classes": NUM_CLASSES,
    }

def prepare():
    """This function will be called once to prepare the dataset."""
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)
    for name in [
            TRAIN_IMAGE_URL,
            TRAIN_LABEL_URL,
            TEST_IMAGE_URL,
            TEST_LABEL_URL]:
        if not os.path.exists(LOCAL_DIR + name):
            urllib.request.urlretrieve(REMOTE_URL + name, LOCAL_DIR + name)

def read(split):
    """Create an instance of the dataset object."""
    image_urls = {
        tf.estimator.ModeKeys.TRAIN: TRAIN_IMAGE_URL,
        tf.estimator.ModeKeys.EVAL: TEST_IMAGE_URL
    }[split]
    label_urls = {
        tf.estimator.ModeKeys.TRAIN: TRAIN_LABEL_URL,
        tf.estimator.ModeKeys.EVAL: TEST_LABEL_URL
    }[split]

    with gzip.open(LOCAL_DIR + image_urls, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(num * rows * cols), dtype=np.uint8)
        images = np.reshape(images, [num, rows, cols, 1])
        print("Loaded %d images of size [%d, %d]." % (num, rows, cols))

    with gzip.open(LOCAL_DIR + label_urls, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(num), dtype=np.int8)
        print("Loaded %d labels." % num)

    return tf.contrib.data.Dataset.from_tensor_slices((images, labels))

def parse(image, label):
    """Parse input record to features and labels."""
    image = tf.to_float(image) / 255.0
    label = tf.to_int64(label)
    return {"image": image}, {"label": label}
"""MobileNet v1 models for Keras.

Code contributed by Somshubra Majumdar (@titu1994).

MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and
different width factors. This allows different width models to reduce
the number of multiply-adds and thereby
reduce inference cost on mobile devices.

MobileNets support any input size greater than 32 x 32, with larger image sizes
offering better performance.
The number of parameters and number of multiply-adds
can be modified by using the `alpha` parameter,
which increases/decreases the number of filters in each layer.
By altering the image size and `alpha` parameter,
all 16 models from the paper can be built, with ImageNet weights provided.

The paper demonstrates the performance of MobileNets using `alpha` values of
1.0 (also called 100 % MobileNet), 0.75, 0.5 and 0.25.
For each of these `alpha` values, weights for 4 different input image sizes
are provided (224, 192, 160, 128).

The following table describes the size and accuracy of the 100% MobileNet
on size 224 x 224:
----------------------------------------------------------------------------
Width Multiplier (alpha) | ImageNet Acc |  Multiply-Adds (M) |  Params (M)
----------------------------------------------------------------------------
|   1.0 MobileNet-224    |    70.6 %     |        529        |     4.2     |
|   0.75 MobileNet-224   |    68.4 %     |        325        |     2.6     |
|   0.50 MobileNet-224   |    63.7 %     |        149        |     1.3     |
|   0.25 MobileNet-224   |    50.6 %     |        41         |     0.5     |
----------------------------------------------------------------------------

The following table describes the performance of
the 100 % MobileNet on various input sizes:
------------------------------------------------------------------------
      Resolution      | ImageNet Acc | Multiply-Adds (M) | Params (M)
------------------------------------------------------------------------
|  1.0 MobileNet-224  |    70.6 %    |        529        |     4.2     |
|  1.0 MobileNet-192  |    69.1 %    |        529        |     4.2     |
|  1.0 MobileNet-160  |    67.2 %    |        529        |     4.2     |
|  1.0 MobileNet-128  |    64.4 %    |        529        |     4.2     |
------------------------------------------------------------------------

The weights for all 16 models are obtained and translated
from Tensorflow checkpoints found at
https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md

# Reference
- [MobileNets: Efficient Convolutional Neural Networks for
   Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf))
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings
import numpy as np

from keras.preprocessing import image

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Conv2D
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


BASE_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.6/'


def relu6(x):
    return K.relu(x, max_value=6)


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


class DepthwiseConv2D(Conv2D):
    """Depthwise separable 2D convolution.

    Depthwise Separable convolutions consists in performing
    just the first step in a depthwise spatial convolution
    (which acts on each input channel separately).
    The `depth_multiplier` argument controls how many
    output channels are generated per input channel in the depthwise step.

    # Arguments
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        activation: Activation function to use
            (see [activations](keras./activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        depthwise_initializer: Initializer for the depthwise kernel matrix
            (see [initializers](keras./initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](keras./initializers.md)).
        depthwise_regularizer: Regularizer function applied to
            the depthwise kernel matrix
            (see [regularizer](keras./regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](keras./regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](keras./regularizers.md)).
        depthwise_constraint: Constraint function applied to
            the depthwise kernel matrix
            (see [constraints](keras./constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](keras./constraints.md)).

    # Input shape
        4D tensor with shape:
        `[batch, channels, rows, cols]` if data_format='channels_first'
        or 4D tensor with shape:
        `[batch, rows, cols, channels]` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `[batch, filters, new_rows, new_cols]` if data_format='channels_first'
        or 4D tensor with shape:
        `[batch, new_rows, new_cols, filters]` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        outputs = K.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])

        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config


def MobileNet(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              pooling=None,
              classes=1000):
    """Instantiates the MobileNet architecture.

    Note that only TensorFlow is supported for now,
    therefore it only works with the data format
    `image_data_format='channels_last'` in your Keras config
    at `~/.keras/keras.json`.

    To load a MobileNet model via `load_model`, import the custom
    objects `relu6` and `DepthwiseConv2D` and pass them to the
    `custom_objects` parameter.
    E.g.
    model = load_model('mobilenet.h5', custom_objects={
                       'relu6': mobilenet.relu6,
                       'DepthwiseConv2D': mobilenet.DepthwiseConv2D})

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or (3, 224, 224) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: depth multiplier for depthwise convolution
            (also called the resolution multiplier)
        dropout: dropout rate
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """

    if K.backend() != 'tensorflow':
        raise RuntimeError('Only Tensorflow backend is currently supported, '
                           'as other backends do not support '
                           'depthwise convolution.')

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top` '
                         'as true, `classes` should be 1000')

    # Determine proper input shape.
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      include_top=include_top or weights)
    if K.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == 'imagenet':
        if depth_multiplier != 1:
            raise ValueError('If imagenet weights are being loaded, '
                             'depth multiplier must be 1')

        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')

        if rows != cols or rows not in [128, 160, 192, 224]:
            raise ValueError('If imagenet weights are being loaded, '
                             'input must have a static square shape (one of '
                             '(128,128), (160,160), (192,192), or (224, 224)).'
                             ' Input shape provided = %s' % (input_shape,))

    if K.image_data_format() != 'channels_last':
        warnings.warn('The MobileNet family of models is only available '
                      'for the input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height).'
                      ' You should set `image_data_format="channels_last"` '
                      'in your Keras config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    if include_top:
        if K.image_data_format() == 'channels_first':
            shape = (int(1024 * alpha), 1, 1)
        else:
            shape = (1, 1, int(1024 * alpha))

        x = GlobalAveragePooling2D()(x)
        x = Reshape(shape, name='reshape_1')(x)
        x = Dropout(dropout, name='dropout')(x)
        x = Conv2D(classes, (1, 1),
                   padding='same', name='conv_preds')(x)
        x = Activation('softmax', name='act_softmax')(x)
        x = Reshape((classes,), name='reshape_2')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='mobilenet_%0.2f_%s' % (alpha, rows))

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            raise ValueError('Weights for "channels_last" format '
                             'are not available.')
        if alpha == 1.0:
            alpha_text = '1_0'
        elif alpha == 0.75:
            alpha_text = '7_5'
        elif alpha == 0.50:
            alpha_text = '5_0'
        else:
            alpha_text = '2_5'

        if include_top:
            model_name = 'mobilenet_%s_%d_tf.h5' % (alpha_text, rows)
            weigh_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(model_name,
                                    weigh_path,
                                    cache_subdir='models')
        else:
            model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
            weigh_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(model_name,
                                    weigh_path,
                                    cache_subdir='models')
        model.load_weights(weights_path)

    if old_data_format:
        K.set_image_data_format(old_data_format)
    return model


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    """Adds an initial convolution layer (with batch normalization and relu6).

    # Arguments
        inputs: Input tensor of shape `(rows, cols, 3)`
            (with `channels_last` data format) or
            (3, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.

    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating the block number.

    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


if __name__ == '__main__':
    for r in [128, 160, 192, 224]:
        for a in [0.25, 0.50, 0.75, 1.0]:
            if r == 224:
                model = MobileNet(include_top=True, weights='imagenet',
                                  input_shape=(r, r, 3), alpha=a)

                img_path = 'elephant.jpg'
                img = image.load_img(img_path, target_size=(r, r))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                print('Input image shape:', x.shape)

                preds = model.predict(x)
                print(np.argmax(preds))
                print('Predicted:', decode_predictions(preds, 1))

            model = MobileNet(include_top=False, weights='imagenet')
import tensorflow as tf
import numpy as np
import os
from scipy.misc import *
#from read_cifar10 import *
from keras.datasets import cifar10
from keras.utils import np_utils
import keras
batch_size = 100   # batch size
num_category = 10  # total categorical factor
#num_cont = 10 # total continuous factor
num_dim = 100  # total latent dimension
T_num = 50
train_flag = True
sample_flag = True
load_flag = False
save_flag = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
z_cat = tf.random_uniform([batch_size],minval=0,maxval=10,dtype=tf.int32)
z_cat = tf.one_hot(z_cat, 10)

multi_dist = tf.contrib.distributions.StudentT(df=[2.0]*100,loc=[0.0]*100,scale=[1.0]*100)
tmp_noise = multi_dist.sample([T_num,batch_size])

with tf.variable_scope('weight'):
	mu = tf.get_variable("mu",[T_num,batch_size,100])
	sigma = tf.get_variable("sigma",[T_num,batch_size,100])
	noise = tmp_noise*mu+sigma

	noise = tf.transpose(noise, perm=[1, 0, 2])
	noise = tf.reshape(noise, [batch_size, -1])
	h_w = tf.layers.dense(noise, 1024, activation=tf.nn.relu)
	h_w2 = tf.layers.dense(h_w, 640, activation=tf.nn.relu)

	h_w3 = tf.layers.dense(h_w2, 320, activation=tf.nn.relu)
	h_w4 = tf.layers.dense(h_w3, T_num, activation=tf.nn.sigmoid)
	h_w4 = tf.reshape(h_w4, [batch_size, T_num, 1])
	weight = tf.tile(h_w4, [1, 1, 100])
	noise = tf.reshape(noise, [batch_size, T_num, 100])
	noise = noise* weight

	noise = tf.reduce_mean(noise,axis=1)

z = tf.concat([z_cat,noise],1)
#z_cont = z[:, num_category:num_category+num_cont]
#z=noise

true_images = tf.placeholder(tf.float32, [batch_size,32,32,3])
true_labels = tf.placeholder(tf.float32, [batch_size,10])

def generator(z):
	with tf.variable_scope('generator'):
		print("generator")
		h0 = tf.layers.dense(z,4*4*512)
		h0 = tf.nn.relu(tf.layers.batch_normalization(h0,training = True))  #(32, 1024)
		print(h0.shape)
		h1 = tf.layers.dense(h0,8*8*256)  
		h1 = tf.nn.relu(tf.layers.batch_normalization(h1,training = True))
		h1 = tf.reshape(h1,[-1,8,8,256])
		h1 = tf.nn.dropout(h1,0.6)		#(32, 8, 8, 258)
		print(h1.shape)
		#l2__________
		h2 = tf.layers.conv2d_transpose(h1,256,[3,3],strides=2,padding="same")
		print(h2.shape)
		h2 = tf.reshape(h2,[-1,16,16,256,1])
		print(h2.shape)
		h2 = tf.layers.max_pooling3d(h2,(1,1,3),strides=(1,1,2),padding="same")
		print(h2.shape)
		h2 = tf.reshape(h2,[-1,16,16,128])
		print(h2.shape)
#		h2 = tf.layers.conv2d_transpose(h2,128,[3,3],strides=1,padding="same")
		h2 = tf.nn.relu(tf.layers.batch_normalization(h2,training = True))	#(32, 16, 16, 128)
		h2 = tf.nn.dropout(h2,0.6)	
		print(h2.shape)
		#l3___________
		h3 = tf.layers.conv2d_transpose(h2,9,[3,3],strides=2,padding="same")
		h3 = tf.reshape(h3,[-1,32,32,9,1])
		print(h3.shape)
		h3 = tf.layers.average_pooling3d(h3,(1,1,3),strides=(1,1,3),padding="same")
		print(h3.shape)
		h3 = tf.reshape(h3,[-1,32,32,3])
		print(h3.shape)
#		h3 = tf.layers.max_pooling3d(h3,(1,1,2),2,padding="same")
#		h3 = tf.layers.conv2d(h3,3,[3,3],strides=1,padding="same")
		h3 = tf.nn.tanh(h3) 
		h3 = tf.nn.dropout(h3,0.6)	      
		print(h3.shape)      #(32, 32, 32, 3)
		return(h3)

def discriminator(image,reuse=False):
	with tf.variable_scope('discriminator', reuse=reuse):
		#h02 = tf.layers.conv2d(xx,16,[4,4],strides=2,padding="same")
		#h02 = tf.nn.leaky_relu(h02)

		#h01 = tf.layers.conv2d(h02,32,[4,4],strides=2,padding="same")
		#h01 = tf.nn.leaky_relu(h01)
		print("discriminator")
		print(image.shape)
		h0 = tf.layers.conv2d(image,64,[3,3],strides=2,padding="same")
		h0 = tf.nn.relu(h0)   	#(32, 16, 16, 64)
		h0 = tf.nn.dropout(h0,0.8)	
		print(h0.shape)

		h1 = tf.layers.conv2d(h0,384,[3,3],strides=2,padding="same")
		h1 = tf.reshape(h1,[-1,8,8,384,1])
		print(h1.shape)
		h1 = tf.layers.max_pooling3d(h1,(1,1,3),strides=(1,1,3),padding="same")
		print(h1.shape)
		h1 = tf.reshape(h1,[-1,8,8,128])
		print(h1.shape)
		h1 = tf.nn.relu(h1)		#(32, 8, 8, 128)
		print(h1.shape)

		h11 = tf.layers.conv2d(h1,256,[3,3],strides=2,padding="same")
		h11 = tf.nn.relu(h11)
		h11 = tf.nn.dropout(h11,0.6)	    	#(32, 4, 4, 256)
		print(h11.shape)

		h2 = tf.contrib.layers.flatten(h11)
		h2 = tf.layers.dense(h2, 1024)    
		h2 = tf.nn.relu(h2)
		h2 = tf.nn.dropout(h2,0.6)	      # (batch_size, 1024)
		print(h2.shape)

		disc = tf.layers.dense(h2,1)
		disc = tf.squeeze(disc)
		print(disc.shape)

		h3 = tf.layers.dense(h2,128)
		h3 = tf.nn.relu(h3)        #(batch_size, 128)
		print(h3.shape)

		class_cat = tf.layers.dense(h3,num_category)
		#class_cont = tf.layers.dense(h3,num_cont)
		#class_cont = tf.nn.sigmoid(class_cont)
	return disc, class_cat#, class_cont

G = generator(z)
real_disc,real_class= discriminator(true_images)
fake_disc,fake_class= discriminator(G,reuse = True)

loss_disc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones([batch_size]),logits=real_disc)) + \
			tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros([batch_size]),logits=fake_disc))
loss_class = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_labels,logits=real_class)) + \
			 tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=z_cat,logits=fake_class)) #+ \
			 #tf.reduce_mean(tf.nn.l2_loss(z_cont-fake_cont))'''
loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones([batch_size]),logits=fake_disc))

disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
train_disc = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9).minimize(loss_disc,var_list = disc_vars)

gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator") or var.name.startswith("weight")]
train_gen = tf.train.AdamOptimizer(learning_rate=0.001, beta1 = 0.9).minimize(loss_gen,var_list = gen_vars)

(train_data, train_labels),(_,_) = cifar10.load_data()
train_labels=np_utils.to_categorical(train_labels,num_classes=10)
#train_labels=tf.contrib.distributions.OneHotCategorical(train_labels)
epoch_num = 100
iteration_num = train_data.shape[0]//batch_size

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

if load_flag:
	saver.restore(sess,"checkpoint/")

if train_flag:
	for epoch in range(epoch_num):
		for it_n in range(iteration_num):
			batch_images = train_data[it_n*batch_size:(it_n+1)*batch_size]
			batch_labels = train_labels[it_n*batch_size:(it_n+1)*batch_size]
			l_disc = sess.run([loss_disc,train_disc],feed_dict={true_images:batch_images, true_labels:batch_labels})[0]
			l_gen = sess.run([loss_gen,train_gen])[0]
			print("epoch: %d, iteration_num:%d, dloss:%f, gloss:%f "%(epoch, it_n, l_disc, l_gen))

	if save_flag:
		saver.save(sess,"checkpoint/")

if sample_flag:
	images = sess.run(G)
	images = images.reshape(-1,32,32,3)
	for i in range(batch_size):
		imsave("samples/cifar/%d.png"%(i),images[i])
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# 期望输入数据尺寸: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
							 input_shape=(timesteps, data_dim)))  # 返回维度为 32 的向量序列
model.add(LSTM(32, return_sequences=True))  # 返回维度为 32 的向量序列
model.add(LSTM(32))  # 返回维度为 32 的单个向量
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
							optimizer='rmsprop',
							metrics=['accuracy'])

# 生成虚拟训练数据
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# 生成虚拟验证数据
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
					batch_size=64, epochs=5,
					validation_data=(x_val, y_val))#数据并行
#数据并行包括在每个设备上复制一次目标模型，并使用每个模型副本处理不同部分的输入数据。Keras 有一个内置的实用函数 keras.utils.multi_gpu_model，它可以生成任何模型的数据并行版本，在多达 8 个 GPU 上实现准线性加速。
#
#有关更多信息，请参阅 multi_gpu_model 的文档。这里是一个简单的例子：

from keras.utils import multi_gpu_model

# 将 `model` 复制到 8 个 GPU 上。
# 假定你的机器有 8 个可用的 GPU。
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
					   optimizer='rmsprop')

# 这个 `fit` 调用将分布在 8 个 GPU 上。
# 由于 batch size 为 256，每个 GPU 将处理 32 个样本。
parallel_model.fit(x, y, epochs=20, batch_size=256)
#设备并行
#设备并行性包括在不同设备上运行同一模型的不同部分。对于具有并行体系结构的模型，例如有两个分支的模型，这种方式很合适。
#
#这种并行可以通过使用 TensorFlow device scopes 来实现。这里是一个简单的例子：
#
# 模型中共享的 LSTM 用于并行编码两个不同的序列
input_a = keras.Input(shape=(140, 256))
input_b = keras.Input(shape=(140, 256))

shared_lstm = keras.layers.LSTM(64)

# 在一个 GPU 上处理第一个序列
with tf.device_scope('/gpu:0'):
	encoded_a = shared_lstm(tweet_a)
# 在另一个 GPU上 处理下一个序列
with tf.device_scope('/gpu:1'):
	encoded_b = shared_lstm(tweet_b)

# 在 CPU 上连接结果
with tf.device_scope('/cpu:0'):
	merged_vector = keras.layers.concatenate([encoded_a, encoded_b],
											 axis=-1)#多输入多输出模型
#以下是函数式 API 的一个很好的例子：具有多个输入和输出的模型。函数式 API 使处理大量交织的数据流变得容易。
#
#来考虑下面的模型。我们试图预测 Twitter 上的一条新闻标题有多少转发和点赞数。模型的主要输入将是新闻标题本身，即一系列词语，但是为了增添趣味，我们的模型还添加了其他的辅助输入来接收额外的数据，例如新闻标题的发布的时间等。 该模型也将通过两个损失函数进行监督学习。较早地在模型中使用主损失函数，是深度学习模型的一个良好正则方法。
#
#模型结构如下图所示：
#
#multi-input-multi-output-graph
#
#让我们用函数式 API 来实现它。
#
#主要输入接收新闻标题本身，即一个整数序列（每个整数编码一个词）。 这些整数在 1 到 10,000 之间（10,000 个词的词汇表），且序列长度为 100 个词。

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras
# 标题输入：接收一个含有 100 个整数的序列，每个整数在 1 到 10000 之间。
# 注意我们可以通过传递一个 `name` 参数来命名任何层。
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# Embedding 层将输入序列编码为一个稠密向量的序列，每个向量维度为 512。
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# LSTM 层把向量序列转换成单个向量，它包含整个序列的上下文信息
lstm_out = LSTM(32)(x)
#在这里，我们插入辅助损失，使得即使在模型主损失很高的情况下，LSTM 层和 Embedding 层都能被平稳地训练。

auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
#此时，我们将辅助输入数据与 LSTM 层的输出连接起来，输入到模型中：

auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# 堆叠多个全连接网络层
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# 最后添加主要的逻辑回归层
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
#然后定义一个具有两个输入和两个输出的模型：

model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
#现在编译模型，并给辅助损失分配一个 0.2 的权重。如果要为不同的输出指定不同的 loss_weights 或 loss，可以使用列表或字典。 在这里，我们给 loss 参数传递单个损失函数，这个损失将用于所有的输出。

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
							loss_weights=[1., 0.2])
#我们可以通过传递输入数组和目标数组的列表来训练模型：

#model.fit([headline_data, additional_data], [labels, labels],
#					epochs=50, batch_size=32)
#由于输入和输出均被命名了（在定义时传递了一个 name 参数），我们也可以通过以下方式编译模型：

model.compile(optimizer='rmsprop',
							loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
							loss_weights={'main_output': 1., 'aux_output': 0.2})

import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from keras.utils import plot_model
plot_model(model,to_file='example.png')
lena = mpimg.imread('example.png') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()
# 然后使用以下方式训练：
#model.fit({'main_input': headline_data, 'aux_input': additional_data},
#					{'main_output': labels, 'aux_output': labels},
#					epochs=50, batch_size=32)# -*- coding: utf-8 -*-
'''MusicTaggerCRNN model for Keras.

Code by github.com/keunwoochoi.

# Reference:

- [Music-auto_tagging-keras](https://github.com/keunwoochoi/music-auto_tagging-keras)

'''
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model
from audio_conv_utils import decode_predictions, preprocess_input

TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.3/music_tagger_crnn_weights_tf_kernels_th_dim_ordering.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.3/music_tagger_crnn_weights_tf_kernels_tf_dim_ordering.h5'


def MusicTaggerCRNN(weights='msd', input_tensor=None,
                    include_top=True):
    '''Instantiate the MusicTaggerCRNN architecture,
    optionally loading weights pre-trained
    on Million Song Dataset. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    For preparing mel-spectrogram input, see
    `audio_conv_utils.py` in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
    You will need to install [Librosa](http://librosa.github.io/librosa/)
    to use it.

    # Arguments
        weights: one of `None` (random initialization)
            or "msd" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        include_top: whether to include the 1 fully-connected
            layer (output layer) at the top of the network.
            If False, the network outputs 32-dim features.


    # Returns
        A Keras model instance.
    '''
    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 96, 1366)
    else:
        input_shape = (96, 1366, 1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            melgram_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            melgram_input = input_tensor

    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=time_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)

    # reshaping
    if K.image_dim_ordering() == 'th':
        x = Permute((3, 1, 2))(x)
    x = Reshape((15, 128))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)

    if include_top:
        x = Dense(50, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(melgram_input, x)
    if weights is None:
        return model
    else:
        # Load weights
        if K.image_dim_ordering() == 'tf':
            weights_path = get_file('music_tagger_crnn_weights_tf_kernels_tf_dim_ordering.h5',
                                    TF_WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('music_tagger_crnn_weights_tf_kernels_th_dim_ordering.h5',
                                    TH_WEIGHTS_PATH,
                                    cache_subdir='models')
        model.load_weights(weights_path, by_name=True)
        if K.backend() == 'theano':
            convert_all_kernels_in_model(model)
        return model


if __name__ == '__main__':
    model = MusicTaggerCRNN(weights='msd')

    audio_path = 'audio_file.mp3'
    melgram = preprocess_input(audio_path)
    melgrams = np.expand_dims(melgram, axis=0)

    preds = model.predict(melgrams)
    print('Predicted:')
    print(decode_predictions(preds))
#optimizer
from keras import optimizers

model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
# 传入优化器名称: 默认参数将被采用
model.compile(loss='mean_squared_error', optimizer='sgd')
from keras import optimizers

# 所有参数梯度将被裁剪，让其l2范数最大为1：g * 1 / max(1, l2_norm)
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
from keras import optimizers

# 所有参数d 梯度将被裁剪到数值范围内：
# 最大值0.5
# 最小值-0.5
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
#SGD
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#随机梯度下降优化器
#包含扩展功能的支持： - 动量（momentum）优化, - 学习率衰减（每次参数更新后） - Nestrov动量(NAG)优化
#
#参数
#lr: float >= 0. 学习率
#momentum: float >= 0. 参数，用于加速SGD在相关方向上前进，并抑制震荡
#decay: float >= 0. 每次参数更新后学习率衰减值.
#nesterov: boolean. 是否使用Nesterov动量.
#RMSprop
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#RMSProp优化器。
#
#建议使用优化器的默认参数（除了学习率lr，它可以被自由调节）
#
#这个优化器通常是训练循环神经网络RNN的不错选择。
#
#参数
#
#lr：float> = 0.学习率。
#rho：float> = 0. RMSProp梯度平方的移动均值的衰减率。
#epsilon：float> = 0.模糊因子。若为None，默认为K.epsilon()。
#衰变：float> = 0.每次参数更新后学习率衰减值。
#引用
#
#rmsprop：将梯度除以其最近幅度的运行平均值
#[资源]
#
#Adagrad
keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
#Adagrad优化器。
#
#Adagrad是一种具有特定参数学习率的优化器，它根据参数在训练期间的更新频率进行自适应调整。参数接收的更新越多，更新越小。
#
#建议使用优化器的默认参数。
#
#参数
#
#lr：float> = 0.学习率。
#epsilon：float> = 0.若为None，默认为K.epsilon()。
#衰变：float> = 0.每次参数更新后学习率衰减值。
#引用
#
#Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
#[source]
#
#Adadelta
keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
#Adadelta优化器.
#
#Adadelta是Adagrad的一个具有更强鲁棒性的的扩展版本，它不是累积所有过去的梯度，而是根据渐变更新的移动窗口调整学习速率。 这样，即使进行了许多更新，Adadelta仍在继续学习。 与Adagrad相比，在Adadelta的原始版本中，您无需设置初始学习率。 在此版本中，与大多数其他Keras优化器一样，可以设置初始学习速率和衰减因子。
#
#建议使用优化器的默认参数。
#
#参数
#
#lr: float >= 0. 学习率，建议保留默认值.
#rho: float >= 0. Adadelta梯度平方移动均值的衰减率
#epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon().
#decay: float >= 0. 每次参数更新后学习率衰减值.
#引用
#
#Adadelta - an adaptive learning rate method
#[source]
#
#Adam
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#Adam优化器.
#
#默认参数遵循原论文中提供的值。
#
#参数
#
#lr: float >= 0. 学习率.
#beta_1: float, 0 < beta < 1. 通常接近于 1.
#beta_2: float, 0 < beta < 1. 通常接近于 1.
#epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon().
#decay: float >= 0. 每次参数更新后学习率衰减值.
#amsgrad: boolean. 是否应用此算法的AMSGrad变种，来自论文"On the Convergence of Adam and Beyond".
#引用
#
#Adam - A Method for Stochastic Optimization
#On the Convergence of Adam and Beyond
#[source]
#
#Adamax
keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#Adamax优化器，来自Adam论文的第七小节.
#
#它是Adam算法基于无穷范数（infinity norm）的变种。 默认参数遵循论文中提供的值。
#
#参数
#
#lr: float >= 0. 学习率.
#beta_1/beta_2: floats, 0 < beta < 1. 通常接近于 1.
#epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon().
#decay: float >= 0. 每次参数更新后学习率衰减值.
#引用
#
#Adam - A Method for Stochastic Optimization
#[source]
#
#Nadam
keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#Nesterov版本Adam优化器.
#
#正像Adam本质上是RMSProp与动量momentum的结合， Nadam是采用Nesterov momentum版本的Adam优化器。
#
#默认参数遵循论文中提供的值。 建议使用优化器的默认参数。
#
#参数
#
#lr: float >= 0. 学习率.
#beta_1/beta_2: floats, 0 < beta < 1. 通常接近于 1.
#epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon().
#引用
#
#Nadam report
#On the importance of initialization and momentum in deep learning
#[source]
#
#TFOptimizer
keras.optimizers.TFOptimizer(optimizer)
#原生Tensorlfow优化器的包装类（wrapper class）。#coding=utf-8  
from keras.models import Model  
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate,Activation,ZeroPadding2D  
from keras.layers import add,Flatten  
from keras.utils import plot_model
#from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D  
import numpy as np  
seed = 7  
np.random.seed(seed)  
  
def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):  
	if name is not None:  
		bn_name = name + '_bn'  
		conv_name = name + '_conv'  
	else:  
		bn_name = None  
		conv_name = None  
  
	x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
	x = BatchNormalization(axis=3,name=bn_name)(x)  
	return x  
  
def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):  
	x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')  
	x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')  
	if with_conv_shortcut:  
		shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)  
		x = add([x,shortcut])  
		return x  
	else:  
		x = add([x,inpt])  
		return x  
  
inpt = Input(shape=(224,224,3))  
x = ZeroPadding2D((3,3))(inpt)  
x = Conv2d_BN(x,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='valid')  
x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)  
#(56,56,64)  
x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))  
x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))  
x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))  
#(28,28,128)  
x = Conv_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))  
x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))  
x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))  
#(14,14,256)  
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
#(7,7,512)  
x = Conv_Block(x,nb_filter=512,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))  
x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))  
x = AveragePooling2D(pool_size=(7,7))(x)  
x = Flatten()(x)  
x = Dense(1000,activation='softmax')(x)  
  
model = Model(inputs=inpt,outputs=x)  
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
model.summary()  
plot_model(model,to_file='resnet34.pdf')# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function

import numpy as np
import warnings
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


if __name__ == '__main__':
    model = ResNet50(include_top=True, weights='imagenet')
    model.summary()
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg 
    from keras.utils import plot_model
    model.summary()
    plot_model(model,to_file='resnet50.svg')
    plot_model(model,to_file='resnet50.png')
    lena = mpimg.imread('example.png') # 读取和代码处于同一目录下的 lena.png
    # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理Nginx 
    lena.shape #(512, 512, 3)
    plt.imshow(lena) # 显示图片
    plt.axis('off') # 不显示坐标轴
    # plt.show()
    from IPython.display import SVG, display
    display(SVG('resnet50.svg'))
    img_path = 'dog.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
#coding=utf-8
from keras.models import Model
from keras.layers import Input,Dense,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,ZeroPadding2D
from keras.layers import add,Flatten
#from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
from keras.optimizers import SGD
import numpy as np
seed = 7
np.random.seed(seed)
 
def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):
	if name is not None:
		bn_name = name + '_bn'
		conv_name = name + '_conv'
	else:
		bn_name = None
		conv_name = None
 
	x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
	x = BatchNormalization(axis=3,name=bn_name)(x)
	return x
 
def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
	x = Conv2d_BN(inpt,nb_filter=nb_filter[0],kernel_size=(1,1),strides=strides,padding='same')
	x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3,3), padding='same')
	x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1,1), padding='same')
	if with_conv_shortcut:
		shortcut = Conv2d_BN(inpt,nb_filter=nb_filter[2],strides=strides,kernel_size=kernel_size)
		x = add([x,shortcut])
		return x
	else:
		x = add([x,inpt])
		return x
 
inpt = Input(shape=(224,224,3))
x = ZeroPadding2D((3,3))(inpt)
x = Conv2d_BN(x,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='valid')
x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
 
x = Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3),strides=(1,1),with_conv_shortcut=True)
x = Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3))
x = Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3))
 
x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3))
x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3))
x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3))
 
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3))
 
x = Conv_Block(x,nb_filter=[512,512,2048],kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
x = Conv_Block(x,nb_filter=[512,512,2048],kernel_size=(3,3))
x = Conv_Block(x,nb_filter=[512,512,2048],kernel_size=(3,3))
x = AveragePooling2D(pool_size=(7,7))(x)
x = Flatten()(x)
x = Dense(1000,activation='softmax')(x)
 
model = Model(inputs=inpt,outputs=x)
sgd = SGD(decay=0.0001,momentum=0.9)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
model.summary()
#你可以使用 model.save(filepath) 将 Keras 模型保存到单个 HDF5 文件中，该文件将包含：
#
#模型的结构，允许重新创建模型
#模型的权重
#训练配置项（损失函数，优化器）
#优化器状态，允许准确地从你上次结束的地方继续训练。
#你可以使用 keras.models.load_model(filepath) 重新实例化模型。load_model 还将负责使用保存的训练配置项来编译模型（除非模型从未编译过）。
#
#例子：
#
from keras.models import load_model

model.save('my_model.h5')  # 创建 HDF5 文件 'my_model.h5'
del model  # 删除现有模型
# 返回一个编译好的模型
# 与之前那个相同
model = load_model('my_model.h5')

#只保存/加载 模型的结构
#如果您只需要保存模型的结构，而非其权重或训练配置项，则可以执行以下操作：
#
# 保存为 JSON
json_string = model.to_json()

# 保存为 YAML
yaml_string = model.to_yaml()
#生成的 JSON/YAML 文件是人类可读的，如果需要还可以手动编辑。

#你可以从这些数据建立一个新的模型：

# 从 JSON 重建模型：
from keras.models import model_from_json
model = model_from_json(json_string)

# 从 YAML 重建模型：
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)
#只保存/加载 模型的权重
#、如果您只需要 模型的权重，可以使用下面的代码以 HDF5 格式进行保存。
#请注意，我们首先需要安装 HDF5 和 Python 库 h5py，它们不包含在 Keras 中。

model.save_weights('my_model_weights.h5')
#假设你有用于实例化模型的代码，则可以将保存的权重加载到具有相同结构的模型中：

model.load_weights('my_model_weights.h5')
#如果你需要将权重加载到不同的结构（有一些共同层）的模型中，例如微调或迁移学习，则可以按层的名字来加载权重：

model.load_weights('my_model_weights.h5', by_name=True)
#例如：
#
#"""
#假设原始模型如下所示：
#	model = Sequential()
#	model.add(Dense(2, input_dim=3, name='dense_1'))
#	model.add(Dense(3, name='dense_2'))
#	...
#	model.save_weights(fname)
#"""

# 新模型
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # 将被加载
model.add(Dense(10, name='new_dense'))  # 将不被加载

# 从第一个模型加载权重；只会影响第一层，dense_1
model.load_weights(fname, by_name=True)from scipy.misc import *
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
#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier


# 加载MNIST数据集
def load_dataset():
	from sklearn.datasets import fetch_mldata
	mnist = fetch_mldata('MNIST original', data_home='dataset')
	X, y = mnist['data'], mnist['target']
	X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
	shuffle_index = np.random.permutation(60000)
	X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
	print('load mnist successfully\n', 'X_train shape is: ', X_train.shape, 'X_test shape is:', X_test.shape)
	return X_train, X_test, y_train, y_test


# 展示数据集的样本
def show_data(dataset, labels, index):
	sample = dataset[index]
	sample_img = sample.reshape(28, 28)
	print('The label of this image is:', labels[index])
	plt.imshow(sample_img)
	plt.axis('off')
	plt.show()


# 单个数字的随机梯度下降二分类器
def single_number_classify(X_train,  y_train, number):
	# 重构数据标签，该数字的标签为1其它数字为0
	y_train_i = (y_train == number)
	# y_test_i = (y_test == number)
	# 创建随机梯度下降分类器实例
	sgd_clf = SGDClassifier(random_state=42)
	sgd_clf.fit(X_train, y_train_i)
	return sgd_clf, y_train_i


# 单个数字的随机梯度下降二分类器预测
def snc_predict(sgd_clf, samples):
	predict = sgd_clf.predict(samples)
	print(' Predicted as:', predict)


# 单个数字的随机梯度下降二分类器性能评估,用验证集评估不是测试集
def snc_assess(sgd_clf, X_train, y_train_i):
	# K折交叉验证评分，指标为精度, 设置为3折
	from sklearn.model_selection import cross_val_score
	crs = cross_val_score(sgd_clf, X_train, y_train_i, cv=3, scoring="accuracy")
	print('3折交叉验证精度为：', crs)

	# 计算混淆矩阵, 每一行表示一个实际的类, 而每一列表示一个预测的类, 值为样本的个数
	from sklearn.model_selection import cross_val_predict
	from sklearn.metrics import confusion_matrix
	y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_i, cv=3)
	confu_matrix = confusion_matrix(y_train_i, y_train_pred)
	print('单数字二分类器混淆矩阵为：', confu_matrix)

	# 计算准确率、召回率和F1值
	from sklearn.metrics import precision_score, recall_score, f1_score
	precision = precision_score(y_train_i, y_train_pred)
	recall = recall_score(y_train_i, y_train_pred)
	f1_sco = f1_score(y_train_i, y_train_pred)

	print('准确率为：', precision, '召回率为', recall, 'F1值为：', f1_sco)

	# 获得准确率、召回率、阈值数据
	from sklearn.metrics import precision_recall_curve
	y_scores = cross_val_predict(sgd_clf, X_train, y_train_i, cv=3, method="decision_function")
	precisions, recalls, thresholds = precision_recall_curve(y_train_i, y_scores)
	# 绘制曲线
	plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
	plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
	plt.xlabel("Threshold")
	plt.legend(loc="upper left")
	plt.ylim([0, 1])
	plt.show()

	# ROC曲线，即真正例率（true positive rate，另一个名字叫做召回率）对假正例率（false positive rate, FPR）的曲线
	from sklearn.metrics import roc_curve
	fpr, tpr, thresholds = roc_curve(y_train_i, y_scores)
	# 绘制ROC曲线
	plt.plot(fpr, tpr, linewidth=2, label=None)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.axis([0, 1, 0, 1])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.show()


# 手写数字的随机梯度下降多分类器，默认为OvA/OvR（一对所有/一对其它）
def number_classify_ova(X_train, y_train):
	# 创建随机梯度下降多分类器实例
	sgd_clf = SGDClassifier(random_state=42)
	sgd_clf.fit(X_train, y_train)
	# 预测样本
	sample = X_train[100]
	predict = sgd_clf.predict([sample])
	# 查看该样本在各类中的得分
	digit_scores = sgd_clf.decision_function([sample])
	print('OvA的随机梯度下降分类器预测结果为：', predict, '该样本的各类得分：', digit_scores)
	return sgd_clf


# 手写数字的随机梯度下降多分类器，使用OvO（一对一）
def number_classify_ovo(X_train, y_train):
	# 创建OvO的随机梯度下降多分类器实例
	from sklearn.multiclass import OneVsOneClassifier
	ovo_sgd_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
	ovo_sgd_clf.fit(X_train, y_train)
	# 预测样本
	sample = X_train[100]
	predict = ovo_sgd_clf.predict([sample])
	print('OvO的随机梯度下降分类器预测结果为：', predict)


# 手写数字的随机森林（Random Forest）多分类器
def number_classify_rf(X_train, y_train):
	# 创建随机森林多分分类器
	from sklearn.ensemble import RandomForestClassifier
	forest_clf = RandomForestClassifier(random_state=42)
	forest_clf.fit(X_train, y_train)
	# 预测
	sample = X_train[100]
	predict = forest_clf.predict([sample])
	print('随机森林预测分类器结果为', predict)


# 输入正则化后的手写数字随机梯度下降多分类器的结果
def input_scaled_sgd(sgd_clf, X_train, y_train):
	# 对训练集进行标准的缩放
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
	# 评估分类器的性能
	from sklearn.model_selection import cross_val_predict
	from sklearn.metrics import confusion_matrix
	from sklearn.model_selection import cross_val_score
	scores = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
	print('输入正则化后的手写数字随机梯度下降多分类器的3折交叉验证精度为:', scores)
	y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
	conf_mx = confusion_matrix(y_train, y_train_pred)
	print('输入正则化后的手写数字随机梯度下降多分类器的混淆矩阵:\n', conf_mx)
	# 绘制混淆矩阵图
	# 换算出各元素在行中所占的比例
	row_sums = conf_mx.sum(axis=1, keepdims=True)
	norm_conf_mx = conf_mx / row_sums
	# 对角线置0，为了更好地观察不好的点（亮度高），
	np.fill_diagonal(norm_conf_mx, 0)
	plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
	plt.show()


if __name__ == '__main__':
	# 加载数据集
	X_train, X_test, y_train, y_test = load_dataset()
	# 展示数据集的某个样本
	# show_data(X_train, y_train, 100)
	# 创建单个数字的随机梯度下降二分类器
	# sgd_clf, y_train_i = single_number_classify(X_train, y_train, 5)
	# 用单个数字的随机梯度下降二分类器测试样本
	# snc_predict(sgd_clf, X_train[:3])
	# 评估单个数字的随机梯度下降二分类器的性能
	# snc_assess(sgd_clf, X_train, y_train_i)
	# 使用手写数字的随机梯度下降多分类器，OvA
	sgd_clf_ova = number_classify_ova(X_train, y_train)
	# 使用手写数字的随机梯度下降多分类器，OvO
	# number_classify_ovo(X_train, y_train)
	# 使用手写数字的随机森林多分类器
	# number_classify_rf(X_train, y_train)
	# 输入正则化后手写数字的随机梯度下降多分类器的性能评估
	input_scaled_sgd(sgd_clf_ova, X_train, y_train)#共享网络层
#函数式 API 的另一个用途是使用共享网络层的模型。我们来看看共享层。
#
#来考虑推特推文数据集。我们想要建立一个模型来分辨两条推文是否来自同一个人（例如，通过推文的相似性来对用户进行比较）。
#
#实现这个目标的一种方法是建立一个模型，将两条推文编码成两个向量，连接向量，然后添加逻辑回归层；这将输出两条推文来自同一作者的概率。模型将接收一对对正负表示的推特数据。
#
#由于这个问题是对称的，编码第一条推文的机制应该被完全重用来编码第二条推文。这里我们使用一个共享的 LSTM 层来编码推文。
#
#让我们使用函数式 API 来构建它。首先我们将一条推特转换为一个尺寸为 (140, 256) 的矩阵，即每条推特 140 字符，每个字符为 256 维的 one-hot 编码 （取 256 个常用字符）。

import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

tweet_a = Input(shape=(140, 256))
tweet_b = Input(shape=(140, 256))
#要在不同的输入上共享同一个层，只需实例化该层一次，然后根据需要传入你想要的输入即可：

# 这一层可以输入一个矩阵，并返回一个 64 维的向量
shared_lstm = LSTM(64)

# 当我们重用相同的图层实例多次，图层的权重也会被重用 (它其实就是同一层)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# 然后再连接两个向量：
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

# 再在上面添加一个逻辑回归层
predictions = Dense(1, activation='sigmoid')(merged_vector)

# 定义一个连接推特输入和预测的可训练的模型
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
							loss='binary_crossentropy',
							metrics=['accuracy'])
#model.fit([data_a, data_b], labels, epochs=10)
model.summary()	
#让我们暂停一会，看看如何读取共享层的输出或输出尺寸。
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from keras.utils import plot_model
plot_model(model,to_file='example.png')
lena = mpimg.imread('example.png') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()
import tensorflow as tf
import numpy as np

#from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import scipy.ndimage
import  os
batch_size = 32  # batch size
num_category = 10  # total categorical factor
#num_cont = 2  # total continuous factor
num_dim = 50  # total latent dimension
T_num = 25
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_flag = True
sample_flag = True
load_flag = False
save_flag = True

z_cat = tf.random_uniform([batch_size],minval=0,maxval=10,dtype=tf.int32)
z_cat = tf.one_hot(z_cat, num_category)

multi_dist = tf.contrib.distributions.StudentT(df=[2.0]*40,loc=[0.0]*40,scale=[1.0]*40)
tmp_noise  = multi_dist.sample([batch_size])   #[32,40]
print(z_cat)
print(tmp_noise)
noise = tmp_noise
#multi_dist = tf.random_normal([batch_size,40],mean=0.0,stddev=1.0)
#tmp_noise =multi_dist
with tf.variable_scope('weight'):
	mu = tf.get_variable("mu",[T_num,batch_size,40])
	sigma = tf.get_variable("sigma",[T_num,batch_size,40])
	noise = tmp_noise*mu+sigma
	weight = tf.get_variable("weight",[T_num,1,1])   #5个T分布[5,1,1]
	weight = tf.tile(weight,[1,batch_size,40])   #weight扩到每个参数上[5,32,40]
	#noise = noise * weight
	noise = noise * weight
	noise = tf.reduce_mean(noise,axis=0)

	'''w_noise=tf.transpose(noise, perm=[1, 0, 2]) 
	w_noise=tf.reshape(w_noise, [batch_size, -1])

	h_w = tf.layers.dense(w_noise , 640 , activation=tf.nn.relu)
	h_w2 = tf.layers.dense(h_w , 320 , activation=tf.nn.relu)

	h_w3= tf.layers.dense(h_w2 , T_num , activation=tf.nn.sigmoid) 
	h_w3=tf.reshape(h_w3 , [batch_size , T_num , 1])
	h_w3= tf.tile(h_w3 , [1 , 1 , 40 ]) 
	w_noise=tf.reshape(w_noise, [batch_size, T_num ,40]) 
	w_noise= w_noise*h_w3

	w_noise = tf.reduce_mean(w_noise,axis=1)'''

z = tf.concat([z_cat,noise],1)
#z_cont = z[:, num_category:num_category+num_cont]
print('zshape=',z.shape)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
#z = tf.random_normal([2, 3])
with tf.Session():
	x_np = z.eval()
	print(x_np)
#plt.plot(x,y)
#plt.show()
true_images = tf.placeholder(tf.float32, [batch_size,28,28,1])
true_labels = tf.placeholder(tf.float32, [batch_size,num_category])

with tf.variable_scope('generator'):
	print("generator")
	h0 = tf.layers.dense(z,1024)
	h0 = tf.nn.relu(tf.layers.batch_normalization(h0,training = True))
	print(h0.shape)
	
	h1 = tf.layers.dense(h0,7*7*32)
	h1 = tf.nn.relu(tf.layers.batch_normalization(h1,training = True))
	h1 = tf.reshape(h1,[-1,7,7,32])
	print(h1.shape)

	h2 = tf.layers.conv2d_transpose(h1,16,[4,4],strides=2,padding="same")
	h2 = tf.nn.relu(tf.layers.batch_normalization(h2,training = True))
	print(h2.shape)

	h3 = tf.layers.conv2d_transpose(h2,1,[4,4],strides=2,padding="same")
	h3 = tf.nn.sigmoid(h3)
	print(h3.shape)

def d(xx,reuse=False):
	with tf.variable_scope('discriminator', reuse=reuse):
		print("discriminator")
		h0 = tf.layers.conv2d(xx,16,[4,4],strides=2,padding="same")
		h0 = tf.nn.crelu(h0)
		print(h0.shape)

		h1 = tf.layers.conv2d(h0,32,[4,4],strides=2,padding="same")
		h1 = tf.nn.crelu(tf.layers.batch_normalization(h1,training = True))
		print(h1.shape)

		h12 = tf.layers.conv2d(h1,64,[4,4],strides=2,padding="same")
		h12 = tf.nn.crelu(tf.layers.batch_normalization(h12,training = True))
		print(h12.shape)

		h2 = tf.nn.max_pool(h1, ksize=[1,4,4,1], strides=[1,1,1,1],padding='VALID')
		h2 = tf.contrib.layers.flatten(h1)
		h2 = tf.layers.dense(h2,1024)
		h2 = tf.nn.crelu(tf.layers.batch_normalization(h2,training = True))
		print(h2.shape)

		disc = tf.layers.dense(h2,1)
		disc = tf.squeeze(disc)
		print(disc.shape)

		h3 = tf.layers.dense(h2,128)
		h3 = tf.nn.crelu(h3)

		class_cat = tf.layers.dense(h3,10)
		#class_cont = tf.layers.dense(h3,2)
		#class_cont = tf.nn.sigmoid(class_cont)
	return disc, class_cat#, class_cont

def merge_images(images):
	ret = np.zeros((8,8,28,28))  
	for i in range(8):
		for j in range(8):
			ret[i][j] = images[i*8+j].reshape(28,28)
	return ret

real_disc,real_class = d(true_images)
fake_disc,fake_class = d(h3,reuse = True)

loss_disc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones([batch_size]),logits=real_disc)) + \
			tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros([batch_size]),logits=fake_disc))
loss_class = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_labels,logits=real_class)) + \
			 tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=z_cat,logits=fake_class)) #+ \
			 #tf.reduce_mean(tf.nn.l2_loss(z_cont-fake_cont))
loss_gen = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones([batch_size]),fake_disc))

disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
train_disc = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9).minimize(loss_disc+loss_class,var_list = disc_vars)

gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator") or var.name.startswith("weight")]
train_gen = tf.train.AdamOptimizer(learning_rate=0.001, beta1 = 0.9).minimize(loss_gen+loss_class,var_list = gen_vars)

(train_data, train_labels), (x_test, y_test) = fashion_mnist.load_data()

train_data=train_data.reshape(-1,28,28,1)
print (train_data.shape)
x_test = x_test.astype('float32')
# 多分类标签生成
train_labels = keras.utils.to_categorical(train_labels, 10)
y_test = keras.utils.to_categorical(y_test, 10)

epoch_num = 10									#epoch_num遍历数据集的次数
iteration_num = train_data.shape[0]//batch_size #iteration_num=训练数据/batch_size
num_steps = 10000 

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


#PAC



#
#if load_flag:
#	saver.restore(sess,"checkpoint/")
#i = 0
#if train_flag:
#	for epoch in range(epoch_num):		#epoch_num遍历数据集的次数
#		for it_n in range(iteration_num):	#iteration_num=batch_size
#			batch_images = train_data[it_n*batch_size:(it_n+1)*batch_size]
#			batch_labels = train_labels[it_n*batch_size:(it_n+1)*batch_size]
#			
#			l_disc = sess.run([loss_disc,train_disc],feed_dict={ true_images:batch_images,true_labels:batch_labels})[0]
#			l_gen = sess.run([loss_gen,train_gen])[0]
#			i= i+1
#			if i % 1000 == 0 or i == 1:
#				print("epoch:",epoch,"interation_num:",it_n,"l_disc:",l_disc,"l_gen:",l_gen)
#
#	if save_flag:
#		saver.save(sess,"checkpoint/")
#
#if sample_flag:
#	for i in range(100):
#		images = sess.run(h3).reshape(-1,28,28)
#		images_1 = sess.run(h3).reshape(-1,28,28)
#		last = np.concatenate((images,images_1),axis=0).reshape(8,8,28,28)
#		last_image = np.zeros((28*8,28*8))
#		for _ in range(8):
#			for __ in range(8):
#				last_image[_*28:(_+1)*28, __*28:(__+1)*28] = last[_][__]
#		print(last_image.shape)
#		imsave("samples/mnist/64_%d.png"%i,last_image)
#
#
#	labels = np.argmax(sess.run(z_cat),axis=1)
#
#	for i in range(32):
#		print("samples/mnist/%d.png %d"%(i,labels[i]))
#		imsave("samples/mnist/%d.png"%(i),images[i])
#		#imsave("samples")import numpy as np 
import random
from deap import base
from deap import creator
from deap import tools
import tensorflow as tf
import fastai
from fastai import *          # Quick access to most common functionality
from fastai.vision import *   # Quick access to computer vision functionality

path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), bs=64)
data.normalize(imagenet_stats)

learn = create_cnn(data, models.resnet50, metrics=accuracy)
learn.fit_one_cycle(1, 0.01)

accuracy(*learn.get_preds())
"""This module handles training and evaluation of a neural network model.

Invoke the following command to train the model:
python -m trainer --model=cnn --dataset=mnist

You can then monitor the logs on Tensorboard:
tensorboard --logdir=output"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cnn
import mnist
tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_string("model", "", "Model name.")
tf.flags.DEFINE_string("dataset", "", "Dataset name.")
tf.flags.DEFINE_string("output_dir", "", "Optional output dir.")
tf.flags.DEFINE_string("schedule", "train_and_evaluate", "Schedule.")
tf.flags.DEFINE_string("hparams", "", "Hyper parameters.")
tf.flags.DEFINE_integer("num_epochs", 100000, "Number of training epochs.")
tf.flags.DEFINE_integer("save_summary_steps", 10, "Summary steps.")
tf.flags.DEFINE_integer("save_checkpoints_steps", 10, "Checkpoint steps.")
tf.flags.DEFINE_integer("eval_steps", None, "Number of eval steps.")
tf.flags.DEFINE_integer("eval_frequency", 10, "Eval frequency.")

FLAGS = tf.flags.FLAGS

MODELS = {
    # This is a dictionary of models, the keys are model names, and the values
    # are the module containing get_params, model, and eval_metrics.
    # Example: "cnn": cnn
    "cnn":cnn
}

DATASETS = {
    # This is a dictionary of datasets, the keys are dataset names, and the
    # values are the module containing get_params, prepare, read, and parse.
    # Example: "mnist": mnist
    "mnist":mnist
}

HPARAMS = {
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "decay_steps": 10000,
    "batch_size": 128
}

def get_params():
    """Aggregates and returns hyper parameters."""
    hparams = HPARAMS
    hparams.update(DATASETS[FLAGS.dataset].get_params())
    hparams.update(MODELS[FLAGS.model].get_params())

    hparams = tf.contrib.training.HParams(**hparams)
    hparams.parse(FLAGS.hparams)

    return hparams

def make_input_fn(mode, params):
    """Returns an input function to read the dataset."""
    def _input_fn():
        dataset = DATASETS[FLAGS.dataset].read(mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.repeat(FLAGS.num_epochs)
            dataset = dataset.shuffle(params.batch_size * 5)
        dataset = dataset.map(
            DATASETS[FLAGS.dataset].parse, num_threads=8)
        dataset = dataset.batch(params.batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
    return _input_fn

def make_model_fn():
    """Returns a model function."""
    def _model_fn(features, labels, mode, params):
        model_fn = MODELS[FLAGS.model].model
        global_step = tf.train.get_or_create_global_step()
        predictions, loss = model_fn(features, labels, mode, params)

        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            def _decay(learning_rate, global_step):
                learning_rate = tf.train.exponential_decay(
                    learning_rate, global_step, params.decay_steps, 0.5,
                    staircase=True)
                return learning_rate

            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=global_step,
                learning_rate=params.learning_rate,
                optimizer=params.optimizer,
                learning_rate_decay_fn=_decay)

        return tf.contrib.learn.ModelFnOps(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)

    return _model_fn

def experiment_fn(run_config, hparams):
    """Constructs an experiment object."""
    estimator = tf.contrib.learn.Estimator(
        model_fn=make_model_fn(), config=run_config, params=hparams)
    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=make_input_fn(tf.estimator.ModeKeys.TRAIN, hparams),
        eval_input_fn=make_input_fn(tf.estimator.ModeKeys.EVAL, hparams),
        eval_metrics=MODELS[FLAGS.model].eval_metrics(hparams),
        eval_steps=FLAGS.eval_steps,
        min_eval_frequency=FLAGS.eval_frequency)

def main(unused_argv):
    """Main entry point."""
    if FLAGS.output_dir:
        model_dir = FLAGS.output_dir
    else:
        model_dir = "output/%s_%s" % (FLAGS.model, FLAGS.dataset)

    DATASETS[FLAGS.dataset].prepare()

    session_config = tf.ConfigProto()
    session_config.allow_soft_placement = True
    session_config.gpu_options.allow_growth = False
    run_config = tf.contrib.learn.RunConfig(
        model_dir=model_dir,
        save_summary_steps=FLAGS.save_summary_steps,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        save_checkpoints_secs=None,
        session_config=session_config)

    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=FLAGS.schedule,
        hparams=get_params())

if __name__ == "__main__":
    tf.app.run()
# -*- coding: utf-8 -*-
'''VGG16 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''
from __future__ import print_function

import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG16(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
    """Instantiates the VGG16 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


if __name__ == '__main__':
    model = VGG16(include_top=True, weights='imagenet')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
#coding=utf-8  
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils import to_categorical
import numpy as np  
seed = 7  
np.random.seed(seed)  
from keras.datasets import mnist,cifar10
(X_train,y_train),(X_test,y_test)=cifar10.load_data()
print(X_train.shape,y_train.shape)
for _ in range(100):
	print(y_train[_])
#X_train = X_train.reshape(-1, 32, 32,1)
#X_test = X_test.reshape(-1, 28, 28,1)
##print(X_train.shape,y_train.shape)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print(X_train.shape,y_train.shape)
for _ in range(100):
	print(y_train[_])
model = Sequential()  
model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(32,32,3),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Flatten())  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(1024,activation='relu'))  
model.add(Dropout(0.5)) 
model.add(Dense(256,activation='relu'))  
model.add(Dropout(0.5)) 
model.add(Dense(10,activation='softmax'))  
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
model.summary()  
model.fit(X_train,y_train,epochs=5,batch_size=64)
#VGG-16
#[python] view plain copy
##coding=utf-8  
#from keras.models import Sequential  
#from keras.layers import Dense,Flatten,Dropout  
#from keras.layers.convolutional import Conv2D,MaxPooling2D  
#import numpy as np  
#seed = 7  
#np.random.seed(seed)  
#  
#model = Sequential()  
#model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(224,224,3),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(MaxPooling2D(pool_size=(2,2)))  
#model.add(Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(MaxPooling2D(pool_size=(2,2)))  
#model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(MaxPooling2D(pool_size=(2,2)))  
#model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(MaxPooling2D(pool_size=(2,2)))  
#model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(MaxPooling2D(pool_size=(2,2)))  
#model.add(Flatten())  
#model.add(Dense(4096,activation='relu'))  
#model.add(Dropout(0.5))  
#model.add(Dense(4096,activation='relu'))  
#model.add(Dropout(0.5))  
#model.add(Dense(1000,activation='softmax'))  
#model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
#model.summary()  import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# 生成虚拟数据
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
# 输入: 3 通道 100x100 像素图像 -> (100, 100, 3) 张量。
# 使用 32 个大小为 3x3 的卷积滤波器。
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)# -*- coding: utf-8 -*-
'''VGG19 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''
from __future__ import print_function

import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG19(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
    """Instantiates the VGG19 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg19')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


if __name__ == '__main__':
    model = VGG19(include_top=True, weights='imagenet')

    img_path = 'cat.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
# -*- coding: utf-8 -*-
'''Xception V1 model for Keras.

On ImageNet, this model gets to a top-1 validation accuracy of 0.790.
and a top-5 validation accuracy of 0.945.

Do note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function
is also different (same as Inception V3).

Also do note that this model is only available for the TensorFlow backend,
due to its reliance on `SeparableConvolution` layers.

# Reference:

- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

'''
from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

from keras.preprocessing import image

from keras.models import Model
from keras import layers
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape


TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'


def Xception(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the Xception architecture.

    Optionally loads weights pre-trained
    on ImageNet. This model is available for TensorFlow only,
    and can only be used with inputs following the TensorFlow
    data format `(width, height, channels)`.
    You should set `image_data_format="channels_last"` in your Keras config
    located at ~/.keras/keras.json.

    Note that the default input image size for this model is 299x299.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)`.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 71.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    if K.backend() != 'tensorflow':
        raise RuntimeError('The Xception model is only available with '
                           'the TensorFlow backend.')
    if K.image_data_format() != 'channels_last':
        warnings.warn('The Xception model is only available for the '
                      'input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height). '
                      'You should set `image_data_format="channels_last"` in your Keras '
                      'config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=299,
                                      min_size=71,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = layers.add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='xception')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels.h5',
                                    TF_WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)

    if old_data_format:
        K.set_image_data_format(old_data_format)
    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = Xception(include_top=True, weights='imagenet')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds, 1))
#coding=utf-8  
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical  
import numpy as np  
seed = 7  
np.random.seed(seed)  
  
model = Sequential()  
model.add(Conv2D(96,(7,7),strides=(2,2),input_shape=(224,224,3),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Conv2D(256,(5,5),strides=(2,2),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Flatten())  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(1000,activation='softmax'))  
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
model.summary()  
#!/usr/bin/python

x=z
y=x
sess = tf.InteractiveSession()  # 创建一个新的计算图
sess.run(tf.global_variables_initializer())  # 初始化所有参数
#z_cont = z[:, num_category:num_category+num_cont]
px = sess.run(x)
py = sess.run(y)
px=px.flatten()
py=py.flatten()
np.set_printoptions(threshold=np.nan)
print('pxshape=',px)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
#z = tf.random_normal([2, 3])
color = np.arctan2(px, py)
plt.scatter(px, py, s = 2, c = color, alpha = 0.5)
# 设置坐标轴范围
plt.xlim((-2, 2))
plt.ylim((-2,2 ))

# 不显示坐标轴的值

plt.show()
#!/usr/bin/python

#链接：https://www.nowcoder.com/questionTerminal/8a19cbe657394eeaac2f6ea9b0f6fcf6
#来源：牛客网
#
#输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
	def reConstructBinaryTree(self, pre, tin):
		if not pre or not tin:
			return None
		root = TreeNode(pre.pop(0))
		index = tin.index(root.val)
		root.left = self.reConstructBinaryTree(pre, tin[:index])
		root.right = self.reConstructBinaryTree(pre, tin[index + 1:])
		return root
	#如何在 Keras 中使用预训练的模型？
我们提供了以下图像分类模型的代码和预训练的权重：

	Xception
	VGG16
	VGG19
	ResNet50
	Inception v3
	Inception-ResNet v2
	MobileNet v1
它们可以使用 keras.applications 模块进行导入：

	from keras.applications.xception import Xception
	from keras.applications.vgg16 import VGG16
	from keras.applications.vgg19 import VGG19
	from keras.applications.resnet50 import ResNet50
	from keras.applications.inception_v3 import InceptionV3
	from keras.applications.inception_resnet_v2 import InceptionResNetV2
	from keras.applications.mobilenet import MobileNet

	model = VGG16(weights='imagenet', include_top=True)#!/usr/bin/python
#题目描述
#在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

class Solution:
	# array 二维列表
	def Find(self, target, array):
		# write code here
		rows = len(array) - 1
		cols= len(array[0]) - 1
		i = rows
		j = 0
		while j<=cols and i>=0:
			if target<array[i][j]:
				i -= 1
			elif target>array[i][j]:
				j += 1
			else:
				return True
		return False