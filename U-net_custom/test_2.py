from model import unet
from model_batch_norm import unet_batch_norm
from functions import split_images_masks_into_patches
from functions import GradientMonitoringCallback, NaNDebugCallback, CustomCallback
import os
from tensorflow.keras.utils import normalize
import numpy as np
import tensorflow as tf
#from matplotlib import pyplot as plt
#from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
import datetime
from tensorflow.keras.callbacks import TensorBoard, Callback
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras.metrics import MeanIoU

####iterative experiment setup####
####Global variables###
SIZE_X = 512
SIZE_Y = 512
n_classes=3 #Number of classes for segmentation
resize_factor = 4 #4, None
model_depth = 4 #2,4,8,16
model_type=unet #unet, unet_batch_norm

#model_name = "20240421-212247_image_size-512x512_batch-32_epochs-1334_resize_factor-4_model_depth-4_model_type-unet_experiment-15.hdf5"
model_name = "20240422-175014_image_size-512x512_batch-32_epochs-1257_resize_factor-4_model_depth-4_model_type-unet_experiment-16-with-scaling.hdf5"

test_image_path = "/home/student515/Documents/thesis/Dataset/Unet/test_images"
test_mask_path = "/home/student515/Documents/thesis/Dataset/Unet/test_masks"
test_prediction_save_path = "/home/student515/Documents/thesis/Dataset/Unet/test_predictions/"+ model_name
weights_path = "/home/student515/Documents/thesis/thesis/U-net_custom/model_test/" + model_name

os.makedirs(test_prediction_save_path + "/X", exist_ok=True)
os.makedirs(test_prediction_save_path + "/Y", exist_ok=True)

#split images and masks into patches
test_images, test_masks = split_images_masks_into_patches(test_image_path, test_mask_path, patch_height=SIZE_X, patch_width=SIZE_Y, resize_factor=resize_factor)

test_images = np.array(test_images)
test_images = np.expand_dims(test_images, axis=3)

test_masks = np.array(test_masks)

for i in range (len(test_images)):
    img=test_images[i,:,:,0]
    img=img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(test_prediction_save_path + "/X/image_" + str(i) + ".tif")

    mask=test_masks[i,:,:,0]*127.5
    mask=mask.astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.save(test_prediction_save_path + "/Y/mask_" + str(i) + ".tif")


X_test = normalize(test_images, axis=1)

test_masks_cat = to_categorical(test_masks, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((test_masks.shape[0], test_masks.shape[1], test_masks.shape[2], n_classes))

IMG_HEIGHT = X_test.shape[1]
IMG_WIDTH  = X_test.shape[2]
IMG_CHANNELS = X_test.shape[3]

def get_model():
    return model_type(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS, lay=model_depth)

model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.load_weights(weights_path)

#evaluate model
_, acc = model.evaluate(X_test, y_test_cat)
print("Accuracy: ", acc)

#predict
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)

#IoU
iou = MeanIoU(num_classes=n_classes)
iou.update_state(test_masks[:,:,:,0],y_pred_argmax)
print("IoU: ", iou.result().numpy())
#per class IoU
iou_values = np.array(iou.get_weights()).reshape(n_classes, n_classes)
print("IoU values: ", iou_values)
class1_IoU = iou_values[0,0]/(iou_values[0,0]+iou_values[0,1]+iou_values[0,2])
class2_IoU = iou_values[1,1]/(iou_values[1,0]+iou_values[1,1]+iou_values[1,2])
class3_IoU = iou_values[2,2]/(iou_values[2,0]+iou_values[2,1]+iou_values[2,2])
per_class_IoU = [class1_IoU, class2_IoU, class3_IoU]
#print("Per-Class IoU:", per_class_IoU)

#per class accuracy
cm=confusion_matrix(test_masks[:,:,:,0].flatten(), y_pred_argmax.flatten())

per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
print("Per-Class Accuracy:", per_class_accuracy)

#Save the predicted masks to create .gif
for i in range(len(y_pred_argmax)):
    pred_image = y_pred_argmax[i]*127
    pred_image = pred_image.astype(np.uint8)
    pred_image = Image.fromarray(pred_image)
    pred_image.save(os.path.join(test_prediction_save_path, f"pred_mask_{i}.tif"))



