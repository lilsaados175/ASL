import os
import random
import numpy as np
import tensorflow as tf

class CFG:
    batch_size = 64
    img_height = 64
    img_width = 64
    epochs = 30
    num_classes = 29
    img_channels = 3


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)