from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import os

# 隐藏tensorflow的输出信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow_datasets as tfds
import random
import tensorflow as tf

tfds.disable_progress_bar()
import pathlib
import matplotlib.pyplot as plt
from tensorflow import keras

# 设置pyplot字体，显示中文
plt.rcParams['font.family'] = 'SimHei'
# 读取模型
model = tf.keras.models.load_model('saved_model/mona')
# 查看网络的所有层
model.summary()

# 限制图片显示尺寸
img_size = 512


# 验证模型


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


# MoeLoader +1s

# 导入验证图片
data_test_orig = tf.keras.utils.get_file(origin='C:/Users/76067/.keras/datasets/MoeLoader +1s.zip',
                                         fname='MoeLoader +1s')
data_test = pathlib.Path(data_test_orig)
# 解析
all_test_image_paths = list(data_test.glob('*/*'))
all_test_image_paths = [str(path) for path in all_test_image_paths]
random.shuffle(all_test_image_paths)
# 图片总数
image_test_count = len(all_test_image_paths)
print("Image count: " + str(image_test_count))

true_mona_pic_conunt = 0
test_mona_pic_conunt = 0
err_mona_pic_conunt = 0

for test_paths_index in all_test_image_paths:
    img = keras.preprocessing.image.load_img(
        test_paths_index, target_size=(img_size, img_size)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    '''
    print(
        "这张图片最接近分类 {} ，置信度为 {:.2f} %"
            .format(np.argmax(score), 100 * np.max(score))
    )
    '''
    print(test_paths_index)
    target_img_path = test_paths_index.split('\\')[len(test_paths_index.split('\\')) - 1]
    if target_img_path.count('mona') >= 1:
        true_mona_pic_conunt = true_mona_pic_conunt + 1
    # copyfile(test_paths_index, target)

    if (np.argmax(score) == 0) and (np.max(score) > 0.7):
        # plt.title("莫娜")
        # print('D:/Python_Project/Mona/Be/mona/' + target_img_path)
        test_mona_pic_conunt = test_mona_pic_conunt + 1
        if target_img_path.count('mona') == 0:
            err_mona_pic_conunt = err_mona_pic_conunt + 1
        copyfile(test_paths_index, 'C:/Users/76067/Pictures/Be/mona/' + str(np.max(score)) + '_' + target_img_path)
    # else:
        # plt.title("别的女人")
        # print('D:/Python_Project/Mona/Be/other/' + target_img_path)
        # copyfile(test_paths_index, 'C:/Users/76067/Pictures/Be/other/' + str(np.max(score)) + '_' + target_img_path)

    '''
    plt.imshow(load_and_preprocess_image(test_paths_index))
    plt.grid(False)
    plt.xlabel(
        "置信度: {:.2f} %"
            .format(100 * np.max(score))
    )

    plt.show()
    '''
    print()

print('图片总数：'+str(image_test_count))
print('真实mona类总数：'+str(true_mona_pic_conunt))
print('模型分类出的mona类总数：'+str(test_mona_pic_conunt))
print('分类错误的数量：'+str(err_mona_pic_conunt))
print('准确度：'+str(round(((test_mona_pic_conunt - err_mona_pic_conunt) / true_mona_pic_conunt) * 100, 2)))
