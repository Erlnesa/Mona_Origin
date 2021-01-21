# 莫娜图片分类器
莫娜图片分类器的开发版本

## 功能介绍

    能将大量图片批量分为：“画面里有莫娜” 与 “画面里没有莫娜”两类。
    
    
## 各文件功能

### mona.py
        
    最初的稳定运行版本，拥有最基本的分类功能。但无法保存模型。
        
### deep_danbooru_test.py
    
    能将本地图片送入基于deep_danbooru网络训练的模型进行tag标记。尚未整合进项目。
        
### keras_data_aug.py
    
    能批量对图像进行数据增强。包括镜像，平移，旋转等。并能选择扩充的倍数。
        
### load_model.py
    
    读取save_model.py训练并保存好的模型，并从指定文件夹抽取图片，根据模型进行分类，然后分别放入两个不同的文件夹。
        
### save_model.py
    
    利用数据集训练卷积神经网络并保存，同时可视化训练过程，给出每次迭代的训练损失和训练准确度已经验证准确度。
        
### png_to_jpg.py
    
    批量将png图片转化为同名的jpg图片，并删除原文件。
        
### saved_model.7z
    
    已经训练好的初代模型，拥有不错的识别准确度。
