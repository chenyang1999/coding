from fastai import *
from fastai.vision import *

path = untar_data(URLs.DOGS)
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)

learn = create_cnn(data, models.resnet34, metrics=accuracy)
learn.fit_one_cycle(1)

learn.unfreeze()
learn.fit_one_cycle(6, slice(1e-5,3e-4), pct_start=0.05)

accuracy(*learn.TTA())