from PIL import Image
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib
import random
import os

# 导入数据
data_root_orig = tf.keras.utils.get_file(origin='C:/Users/76067/.keras/datasets/mona.zip', fname='mona')
data_root = pathlib.Path(data_root_orig)
# 解析
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
# 图片总数
image_count = len(all_image_paths)

for img_index in range(len(all_image_paths)):
    img_is_png = all_image_paths[img_index].endswith(".png")
    if img_is_png :
        #print(all_image_paths[img_index].replace(".png",".jpg"))
        image = Image.open(all_image_paths[img_index])
        image_rgb = image.convert('RGB')
        image_rgb.save(all_image_paths[img_index].replace(".png",".jpg"))
        # 移除
        os.remove(all_image_paths[img_index])
        #print(all_image_paths[img_index])

print("Done")
