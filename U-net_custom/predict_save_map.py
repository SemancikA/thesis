from model_2 import multi_unet_model #Uses softmax 
from functions import split_image_into_patches, split_image_into_patches_resized, split_masks_into_patches, split_masks_into_patches_resized
from functions import GradientMonitoringCallback, NaNDebugCallback, CustomCallback
import os
from tensorflow.keras.utils import normalize
import numpy as np
import tensorflow as tf
#from matplotlib import pyplot as plt
#from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import datetime
from tensorflow.keras.callbacks import TensorBoard, Callback
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import MeanIoU


####Global variables###
SIZE_X = 512
SIZE_Y = 512   
n_classes=3 #Number of classes for segmentation

test_images_path = "/home/student515/Documents/thesis/Dataset/Unet/test_images"
test_masks_path = "/home/student515/Documents/thesis/Dataset/Unet/test_masks"
test_model_weights = "/home/student515/Documents/thesis/thesis/U-net_custom/model_test/Unet50.hdf5"

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=SIZE_X, IMG_WIDTH=SIZE_Y, IMG_CHANNELS=1)

model = get_model()
optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

model.load_weights(test_model_weights)