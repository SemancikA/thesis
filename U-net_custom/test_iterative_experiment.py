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
import glob


#Global variables
n_classes=3
model_type=unet
test_image_path = "/home/student515/Documents/thesis/Dataset/Unet/test_images"
test_mask_path = "/home/student515/Documents/thesis/Dataset/Unet/test_masks"

test_weights_path = "/home/student515/Documents/thesis/thesis/U-net_custom/model_test/exp-18"

weights_paths = sorted(glob.glob(os.path.join(test_weights_path, "*.hdf5")))

test_save_path="/home/student515/Documents/thesis/Dataset/Unet/test_predictions/exp-18"

test_text_file_path = test_save_path + "/test_results.txt"

with open(test_text_file_path, 'w') as file:
    file.write('')

for weights_path in weights_paths:

    weight_name = os.path.basename(weights_path)
    SIZE_X = int(weight_name.split("_")[2].split("x")[1])
    SIZE_Y = SIZE_X

    resize_factor = int(weight_name.split("_")[6].split("-")[1])
    model_depth = int(weight_name.split("_")[8].split("-")[1])

    test_prediction_save_path = test_save_path + "/" + weight_name

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

    #Save the predicted masks
    for i in range(len(y_pred_argmax)):
        pred_image = y_pred_argmax[i]*127
        pred_image = pred_image.astype(np.uint8)
        pred_image = Image.fromarray(pred_image)
        pred_image.save(os.path.join(test_prediction_save_path, f"pred_mask_{i}.tif"))

    data_to_write = f"Accuracy_{acc}_IoU:_{iou.result().numpy()}_Per-Class Accuracy_{per_class_accuracy[0]}_{per_class_accuracy[1]}_{per_class_accuracy[2]}_Weight_{weight_name}"

    with open(test_text_file_path, 'a') as file:
        file.write(data_to_write + "\n")