import skimage.io as io
import os, sys
from skimage import data_dir
import numpy as np

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
path = "D:/train_data/mona/mona/"
dirs = os.listdir(path)
x_all = []
for file in dirs:
    try:
        print(path + file)
        img = load_img(path + file)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，形状为 (1, 3, 150, 150)
        x_all.append(x)
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir='C:/Users/76067/.keras/datasets/mona/mona_aug'):
            i += 1
            if i > 5:  # 数据扩充倍数，此处为数据扩充50倍
                break  # 否则生成器会退出循环
    except BaseException:
        print("Error")







