"""
由于部分图片读取时出现 ibpng warning: iCCP: known incorrect sRGB profile
检测其中哪些图片会出现这个warning
对于出现这些warning的图片，用windows上的画图软件打开后再保存一下就没有warning了
但是这个warning不影响程序运行，可忽视
"""
import os
from skimage import io

path = "D:\ChromeDownload\\fewshotlogodetection_round1_train_202204/train/images/"
filelist = os.listdir(path)
for i in filelist:
    try:
        image = io.imread(i)
    except:
        print(i)
