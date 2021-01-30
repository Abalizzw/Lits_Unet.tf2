#part1
import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
from HDF5DatasetWriter import HDF5DatasetWriter
from HDF5DatasetGenerator import HDF5DatasetGenerator

###函数部分###
def get_pixels_hu(scans):
    '''
    dicom转换为hu
    :param scans:
    :return:
    '''
    #type(scans[0].pixel_array)
    #Out[15]: numpy.ndarray
    #scans[0].pixel_array.shape
    #Out[16]: (512, 512)
    # image.shape: (129,512,512)
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def transform_ctdata(self, windowWidth, windowCenter, normal=False):
    """
    注意，这个函数的self.image一定得是float类型的，否则就无效！
    return: trucated image according to window center and window width
    """
    minWindow = float(windowCenter) - 0.5*float(windowWidth)
    newimg = (self.image - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg

def getRangImageDepth(image):
    """
    提取出包含目标切片的数据
    args:
    image ndarray of shape (depth, height, weight)
    """
    # 得到轴向上出现过目标（label>=1)的切片
    z = np.any(image, axis=(1,2)) # z.shape:(depth,)
    startposition,endposition = np.where(z)[0][[0,-1]]
    return startposition, endposition

def clahe_equalized(imgs,start,end):
   '''
   直方图均值化
   :param imgs: 图像
   :param start: 数据起点
   :param end: 数据终点
   :return: 均值化图片
   '''
   assert (len(imgs.shape)==3)  #3D arrays
   #create a CLAHE object (Arguments are optional).
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   imgs_equalized = np.empty(imgs.shape)
   for i in range(start, end+1):
       imgs_equalized[i,:,:] = clahe.apply(np.array(imgs[i,:,:], dtype = np.uint8))
   return imgs_equalized


###函数部分结束###

###制作训练集###
dataset = HDF5DatasetWriter(image_dims=(2782, 512, 512, 1),
                            mask_dims=(2782, 512, 512, 1),
                            outputPath="data_train/train_liver.h5")

for i in range(1,18): # 前17个人作为测试集
   full_images = [] # 后面用来存储目标切片的列表
   full_livers = [] # 功能同上
   label_path = '../3Dircadb1/3Dircadb1.%d/MASKS_DICOM/liver'%i
   data_path = '../3Dircadb1/3Dircadb1.%d/PATIENT_DICOM'%i
   liver_slices = [pydicom.dcmread(label_path + '/' + s) for s in os.listdir(label_path)]
   # 注意需要排序，即使文件夹中显示的是有序的，读进来后就是随机的了
   liver_slices.sort(key = lambda x: int(x.InstanceNumber))
   # s.pixel_array 获取dicom格式中的像素值
   livers = np.stack([s.pixel_array for s in liver_slices])
   image_slices = [pydicom.dcmread(data_path + '/' + s) for s in os.listdir(data_path)]
   image_slices.sort(key = lambda x: int(x.InstanceNumber))

   images = get_pixels_hu(image_slices)

   images = transform_ctdata(images,500,150)

   start,end = getRangImageDepth(livers)
   images = clahe_equalized(images,start,end)

   images /= 255.
   # 仅提取腹部所有切片中包含了肝脏的那些切片，其余的不要

   total = (end - 4) - (start+4) +1
   print("%d person, total slices %d"%(i,total))
   # 首和尾目标区域都太小，舍弃
   images = images[start+5:end-5]
   print("%d person, images.shape:(%d,)"%(i,images.shape[0]))

   livers[livers>0] = 1

   livers = livers[start+5:end-5]

   full_images.append(images)
   full_livers.append(livers)

   full_images = np.vstack(full_images)
   full_images = np.expand_dims(full_images,axis=-1)
   full_livers = np.vstack(full_livers)
   full_livers = np.expand_dims(full_livers,axis=-1)

   dataset.add(full_images, full_livers)
   #dataset.add(x, y) #此步骤为数据增强，暂时不进行数据增强
# end of lop
dataset.close()

###制作测试数据###
full_images2 = []
full_livers2 = []
for i in range(18,21):#后3个人作为测试样本
    label_path = '../3Dircadb1/3Dircadb1.%d/MASKS_DICOM/liver'%i
    data_path = '../3Dircadb1/3Dircadb1.%d/PATIENT_DICOM'%i
    liver_slices = [pydicom.dcmread(label_path + '/' + s) for s in os.listdir(label_path)]
    liver_slices.sort(key = lambda x: int(x.InstanceNumber))
    livers = np.stack([s.pixel_array for s in liver_slices])
    start,end = getRangImageDepth(livers)
    total = (end - 4) - (start+4) +1
    print("%d person, total slices %d"%(i,total))

    image_slices = [pydicom.dcmread(data_path + '/' + s) for s in os.listdir(data_path)]
    image_slices.sort(key = lambda x: int(x.InstanceNumber))

    images = get_pixels_hu(image_slices)
    images = transform_ctdata(images,500,150)
    images = clahe_equalized(images,start,end)
    images /= 255.
    images = images[start+5:end-5]
    print("%d person, images.shape:(%d,)"%(i,images.shape[0]))
    livers[livers>0] = 1
    livers = livers[start+5:end-5]

    full_images2.append(images)
    full_livers2.append(livers)

full_images2 = np.vstack(full_images2)
full_images2 = np.expand_dims(full_images2,axis=-1)
full_livers2 = np.vstack(full_livers2)
full_livers2 = np.expand_dims(full_livers2,axis=-1)

dataset = HDF5DatasetWriter(image_dims=(full_images2.shape[0], full_images2.shape[1], full_images2.shape[2], 1),
                            mask_dims=(full_images2.shape[0], full_images2.shape[1], full_images2.shape[2], 1),
                            outputPath="data_train/val_liver.h5")


dataset.add(full_images2, full_livers2)

print("total images in val ",dataset.close())
