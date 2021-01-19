import matplotlib.pyplot as plt
import numpy as np
import os
# 隐藏tensorflow的输出信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import random
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# 导入训练数据集
data_dir = tf.keras.utils.get_file(origin='C:/Users/76067/.keras/datasets/mona.zip', fname='mona')
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
# 格式化训练数据
batch_size = 32
img_height = 300
img_width = 300
# 拆分出训练数据集
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
# 拆分出验证数据集
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)
# 您将通过传递这些数据集来训练使用这些数据集的模型model.fit。
# 如果愿意，还可以手动遍历数据集并检索成批图像：
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
# 暂时不知道是干啥的，照抄官网教程
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# 标准化数据
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# 数据扩充
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(img_height,
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)
# 创建模型
# 该模型由三个卷积块组成，每个卷积块中都有一个最大池层。有一个完全连接的层，上面有128个单元。
# 可以通过relu激活功能激活。尚未针对高精度调整此模型，本教程的目的是展示一种标准方法。
num_classes = 2
# Dropout
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
# 对于本教程，选择optimizers.Adam优化器和losses.SparseCategoricalCrossentropy损失函数。
# 要查看每个训练时期的训练和验证准确性，请传递metrics参数。
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# 查看网络的所有层
model.summary()
#开始训练
epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
# 可视化训练结果
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# 验证模型


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [300, 300])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


# 导入验证图片
data_test_orig = tf.keras.utils.get_file(origin='C:/Users/76067/.keras/datasets/mona-test.zip', fname='mona-test')
data_test = pathlib.Path(data_test_orig)
# 解析
all_test_image_paths = list(data_test.glob('*/*'))
all_test_image_paths = [str(path) for path in all_test_image_paths]
random.shuffle(all_test_image_paths)
# 图片总数
image_test_count = len(all_test_image_paths)
print("Image count: " + str(image_test_count))

for test_paths_index in all_test_image_paths:
    img = keras.preprocessing.image.load_img(
        test_paths_index, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "这张图片最接近分类 {} ，置信度为 {:.2f} %"
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    plt.imshow(load_and_preprocess_image(test_paths_index))
    plt.grid(False)
    plt.xlabel(
        "percent confidence: {:.2f} %"
        .format(100 * np.max(score))
    )
    plt.title(class_names[np.argmax(score)])
    plt.show()
    print()



