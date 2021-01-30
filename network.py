import os
import sys
import numpy as np
import random
import math
import tensorflow as tf
from HDF5DatasetGenerator import HDF5DatasetGenerator
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from skimage import io


K.set_image_data_format('channels_last')

def dice_coef(y_true, y_pred):
    #定义dice函数
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    #定义dice损失
    return -dice_coef(y_true, y_pred)

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    print(target.shape)
    print(refer._keras_shape)
    cw = (target._keras_shape[2] - refer._keras_shape[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target._keras_shape[1] - refer._keras_shape[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

def get_unet():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH , IMG_CHANNELS))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up_conv5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)

    ch, cw = get_crop_shape(conv4, up_conv5)

    crop_conv4 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up_conv6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)

    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv3)

    up7 = concatenate([up_conv6, crop_conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up_conv7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv2)

    up8 = concatenate([up_conv7, crop_conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up_conv8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv1)


    up9 = concatenate([up_conv8, crop_conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

IMG_WIDTH = 512 #图像size
IMG_HEIGHT = 512 #图像size
IMG_CHANNELS = 1 #通道数
TOTAL = 2782 # 总共的训练数据
TOTAL_VAL = 152 # 总共的测试数据
outputPath = '../data_train/train_liver.h5' # 训练文件
val_outputPath = '../data_train/val_liver.h5'# 测试文件
BATCH_SIZE = 8

class UnetModel:
    def train_and_predict(self):

        reader = HDF5DatasetGenerator(dbPath=outputPath,batchSize=BATCH_SIZE)
        train_iter = reader.generator()

        test_reader = HDF5DatasetGenerator(dbPath=val_outputPath,batchSize=BATCH_SIZE)
        test_iter = test_reader.generator()
        fixed_test_images, fixed_test_masks = test_iter.__next__()
#

        model = get_unet()
        model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

        model.fit_generator(train_iter,steps_per_epoch=int(TOTAL/BATCH_SIZE),verbose=1,epochs=500,shuffle=True,
                            validation_data=(fixed_test_images, fixed_test_masks),callbacks=[model_checkpoint])
#
        reader.close()
        test_reader.close()


        print('-'*30)
        print('Loading and preprocessing test data...')
        print('-'*30)

        print('-'*30)
        print('Loading saved weights...')
        print('-'*30)
        model.load_weights('weights.h5')

        print('-'*30)
        print('Predicting masks on test data...')
        print('-'*30)


        imgs_mask_test = model.predict(fixed_test_images, verbose=1)
        np.save('imgs_mask_test.npy', imgs_mask_test)

        print('-' * 30)
        print('Saving predicted masks to files...')
        print('-' * 30)
        pred_dir = 'preds'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        i = 0


        for image in imgs_mask_test:
            image = (image[:, :, 0] * 255.).astype(np.uint8)
            gt = (fixed_test_masks[i,:,:,0] * 255.).astype(np.uint8)
            ini = (fixed_test_images[i,:,:,0] *255.).astype(np.uint8)
            io.imsave(os.path.join(pred_dir, str(i) + '_ini.png'), ini)
            io.imsave(os.path.join(pred_dir, str(i) + '_pred.png'), image)
            io.imsave(os.path.join(pred_dir, str(i) + '_gt.png'), gt)
            i += 1

unet = UnetModel()
unet.train_and_predict()
