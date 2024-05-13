import os
import glob
import cv2
import numpy as np
from tensorflow.keras.metrics import MeanIoU

from sklearn.metrics import confusion_matrix
from PIL import Image
import rapidjson as rjson


image_path="/home/student515/Documents/thesis/Dataset/Unet/test_predictions/final_model_Al_test/X"
mask_path="/home/student515/Documents/thesis/Dataset/Unet/test_predictions/final_model_Al_test/Y"
prediction_path="/home/student515/Documents/thesis/Dataset/Unet/test_predictions/final_model_Al_test"

thresholded_images_save_path="/home/student515/Documents/thesis/Dataset/Unet/test_predictions/final_model_Al_test/thresholded_images"
os.makedirs(thresholded_images_save_path, exist_ok=True)

test_text_file_path = prediction_path + "/test_results_threshold.json"

with open(test_text_file_path, 'w') as file:
    rjson.dump([],file)

with open(test_text_file_path, 'r') as file:
    data_list = rjson.load(file)

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


iou_threshold_images = thresholded_images/255
iou_masks= masks/255
iou_preds = preds/255


#IoU of thresholded image with masks
iou_threshold =MeanIoU(num_classes=2)
iou_threshold.update_state(iou_masks,iou_threshold_images)
print("IoU of thresholded image with masks: ", iou_threshold.result().numpy())
iou_threshold_values = np.array(iou_threshold.get_weights()).reshape(2, 2)
print("IoU values of thresholded image with masks: ", iou_threshold_values)
class0_IoU_threshold = iou_threshold_values[0,0]/(iou_threshold_values[0,0]+iou_threshold_values[0,1])
class1_IoU_threshold = iou_threshold_values[1,1]/(iou_threshold_values[1,0]+iou_threshold_values[1,1])
per_class_IoU_threshold = [class0_IoU_threshold, class1_IoU_threshold]
print("Per-Class IoU of thresholded image with masks: ", per_class_IoU_threshold)

#IoU of prediction with masks
iou_pred =MeanIoU(num_classes=2)
iou_pred.update_state(iou_masks,iou_preds)
print("IoU of prediction with masks: ", iou_pred.result().numpy())
iou_pred_values = np.array(iou_pred.get_weights()).reshape(2, 2)
print("IoU values of prediction with masks: ", iou_pred_values)
class0_IoU_pred = iou_pred_values[0,0]/(iou_pred_values[0,0]+iou_pred_values[0,1])
class1_IoU_pred = iou_pred_values[1,1]/(iou_pred_values[1,0]+iou_pred_values[1,1])
per_class_IoU_pred = [class0_IoU_pred, class1_IoU_pred]
print("Per-Class IoU of prediction with masks: ", per_class_IoU_pred)

data_to_write={"Per-Class-Accuracy-of-threshold_":per_class_ac_thresh.tolist(),"_Per-Class-Accuracy-of-prediction_":per_class_ac_pred.tolist(),"_Per-Class-IoU-of-thresholded-image-with-masks_":per_class_IoU_threshold, "_Per-Class-IoU-of-prediction-with-masks_":per_class_IoU_pred}
data_list.append(data_to_write)

with open(test_text_file_path, 'w') as file:
    rjson.dump(data_list, file, indent=4)

for i in range(len(thresholded_images)):
    thresh_img=thresholded_images[i]
    thresh_img=thresh_img.astype(np.uint8)
    thresh_img = Image.fromarray(thresh_img)
    thresh_img.save(thresholded_images_save_path + "/thresholded_image_" + str(i) + ".tif")

   
