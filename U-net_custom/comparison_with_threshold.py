import os
import glob
import cv2
import numpy as np

from sklearn.metrics import confusion_matrix
from PIL import Image


image_path="Dataset/Unet/test_predictions/image_size-512x512_batch-32_epochs-1257_resize_factor-4_model_depth-4_model_type-unet_experiment-16-with-scaling.hdf5/X"
mask_path="Dataset/Unet/test_predictions/image_size-512x512_batch-32_epochs-1257_resize_factor-4_model_depth-4_model_type-unet_experiment-16-with-scaling.hdf5/Y"
prediction_path="Dataset/Unet/test_predictions/image_size-512x512_batch-32_epochs-1257_resize_factor-4_model_depth-4_model_type-unet_experiment-16-with-scaling.hdf5"

image_paths = sorted(glob.glob(os.path.join(image_path, "*.tif")))
mask_paths = sorted(glob.glob(os.path.join(mask_path, "*.tif")))
prediction_paths = sorted(glob.glob(os.path.join(prediction_path, "*.tif")))

if len(image_paths) != len(mask_paths) or len(image_paths) != len(prediction_paths):
    raise ValueError("The number of images, masks, and predictions must be the same.")

masks=[]
preds=[]
thresholded_images=[]

for img_path, msk_path, pred_path in zip(image_paths, mask_paths, prediction_paths):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
    
    mask[mask < 255] = 0
    pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
    pred[pred > 240] = 255
    pred[pred < 240] = 0

    #Threshold the image with otsu method
    thresh_val, otsu_thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    masks.append(mask)
    preds.append(pred)
    thresholded_images.append(otsu_thresh_img)

masks = np.array(masks)
preds = np.array(preds)
thresholded_images = np.array(thresholded_images)

#Calculate accuracy of thresholded image on masks
cm_threshold=confusion_matrix(masks.flatten(), thresholded_images.flatten())
#print(cm_threshold)

per_class_ac_thresh = cm_threshold.diagonal() / cm_threshold.sum(axis=1)
print("Per-Class Accuracy of threshold :", per_class_ac_thresh)

#Calculate accuracy of prediction on masks
cm_pred=confusion_matrix(masks.flatten(), preds.flatten())
#print(cm_pred)

per_class_ac_pred = cm_pred.diagonal() / cm_pred.sum(axis=1)
print("Per-Class Accuracy of prediction :", per_class_ac_pred)

   
