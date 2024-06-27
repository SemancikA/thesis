from model import multi_unet_model
from functions import split_images_masks_into_patches
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

#iterative experiment setup
####Global variables###
SIZE_X = 512
SIZE_Y = 512
n_classes=3 #Number of classes for segmentation
batch_size = 4
epochs = 10000
resize_factor = 4

light=16
mid=32
deep=64

model_depth = light #light, mid, deep


#String to be appended to the direcories
string= "_light"
#directories
images_path = "/home/student515/Documents/thesis/Dataset/Unet/images"
masks_path = "/home/student515/Documents/thesis/Dataset/Unet/masks"

images_patches_save_path = "/home/student515/Documents/thesis/Dataset/Unet/train_patches"
masks_patches_save_path = "/home/student515/Documents/thesis/Dataset/Unet/train_masks_patches"
model_save_dir='/home/student515/Documents/thesis/thesis/U-net_custom/model_save/model_save-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +"_image_size-"+str(SIZE_X) +"x"+ str(SIZE_Y) + "_batch-"+str(batch_size) + "_epochs-"+str(epochs) + "_resize_factor-" + str(resize_factor) + "_model_depth-"+ str(model_depth) + string
log_dir = "/home/student515/Documents/thesis/thesis/U-net_custom/logs/"  + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +"_image_size-"+str(SIZE_X) +"x"+ str(SIZE_Y) + "_batch-"+str(batch_size) + "_epochs-"+str(epochs) + "_resize_factor-" + str(resize_factor) + "_model_depth-"+ str(model_depth) + string
val_pred_save_dir = "/home/student515/Documents/thesis/Dataset/Unet/save_prediction_patches_train/"  + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +"_image_size-"+str(SIZE_X) +"x"+ str(SIZE_Y) + "_batch-"+str(batch_size) + "_epochs-"+str(epochs) + "_resize_factor-" + str(resize_factor) + "_model_depth-"+ str(model_depth) + string
os.makedirs(val_pred_save_dir, exist_ok=True)
os.makedirs(val_pred_save_dir + "/X", exist_ok=True)
os.makedirs(val_pred_save_dir + "/Y", exist_ok=True)


###############################################
###Capture training images and masks info as a list
train_images, train_masks = split_images_masks_into_patches(images_path, masks_path, images_patches_save_path, masks_patches_save_path, SIZE_X, SIZE_Y, resize_factor)

#Convert list to array for machine learning processing        
train_images = np.array(train_images)
train_images = np.expand_dims(train_images, axis=3)
#train_images = normalize(train_images, axis=1)
         
train_masks = np.array(train_masks,dtype=np.uint8)

#Picking 15% for validation and remaining for training
X_train, X_val_test, y_train, y_val_test = train_test_split(train_images, train_masks, test_size = 0.3, random_state = 0)

X_val, X_test, y_val, y_tesy = train_test_split(X_val_test, y_val_test, test_size = 0.5, random_state = 0)

# Save all validation images and masks to val_pred_save_dir
for i in range(X_val.shape[0]):
    img =X_val[i,:,:,0]
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(val_pred_save_dir + "/X/image_" + str(i) + ".tif")

    mask= y_val[i,:,:,0]*(255/2)
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.save(val_pred_save_dir + "/Y/mask_" + str(i) + ".tif")

X_train = normalize(X_train, axis=1)
X_val = normalize(X_val, axis=1)
X_test = normalize(X_test, axis=1)

train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

val_masks_cat = to_categorical(y_val, num_classes=n_classes)
y_val_cat = val_masks_cat.reshape((y_val.shape[0], y_val.shape[1], y_val.shape[2], n_classes))

#Compute class weights
train_masks_reshaped_encoded = train_masks.reshape(-1,1)
classes = np.unique(train_masks_reshaped_encoded)
class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=classes,y=train_masks_reshaped_encoded.flatten())
#class_weights_dict = {class_id: weight for class_id, weight in zip(classes, class_weights)}



IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS, lay=model_depth)

model = get_model()
#optimizer = Adam(learning_rate=0.01, clipvalue=0.5)
#optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=optimizer, loss=categorical_focal_loss(gamma=2., alpha=0.25), metrics=['accuracy'])

print(model.summary())

##########################################################
#check for NaN in the training images
if np.isnan(train_images).any():
    print("NaN found in train_images")
#check for NaN in the training masks
if np.isnan(train_masks).any():
    print("NaN found in train_masks_input")

print("Number of images in X_train: ", X_train.shape)
print("Number of images in y_train: ", y_train.shape)
print("Number of images in X_test: ", X_val.shape)
print("Number of images in y_test: ", y_val.shape)
print("Number of images in y_train_cat: ", y_train_cat.shape)
print("Number of images in y_test_cat: ", y_val_cat.shape)
print("Class values in the dataset are ... ", np.unique(y_train))
print("Class values in the dataset are ... ", np.unique(y_val))

print("Class weights are...:", class_weights)
#print("Class weights are...:", class_weights_dict)

print("Maximum value in the training images: ", np.max(X_train))
print("Minimum value in the training images: ", np.min(X_train))
print("Maximum value in the training masks: ", np.max(y_train_cat))
print("Minimum value in the training masks: ", np.min(y_train_cat))
print("Maximum value in the testing images: ", np.max(X_val))
print("Minimum value in the testing images: ", np.min(X_val))

if np.isnan(X_train).any():
    print("NaN found in X_train")
if np.isnan(y_train_cat).any():
    print("NaN found in y_train_cat")


if np.isinf(X_train).any():
    print("Inf found in X_train")
if np.isinf(y_train_cat).any():
    print("Inf found in y_train_cat")

print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))



###########################################
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
custom_callback = CustomCallback(model_save_dir=model_save_dir, X_val=X_val, y_val_cat=y_val_cat, tensorboard_log_dir=log_dir, val_pred_save_dir=val_pred_save_dir)
nan_debug_callback = NaNDebugCallback()
Gradient_monitoring_callback = GradientMonitoringCallback()
############################################

#Data augmentation
datagen = ImageDataGenerator(
    rotation_range=360,             # Degree range for random rotations
    horizontal_flip=True,           # To mirror images horizontally
    vertical_flip=True,             # To mirror images vertically
    brightness_range=[0.9, 1.1]     # Adjusts brightness, simulating an intensity shift. Values <1 darken the image, values >1 brighten it.
)

train_generator = datagen.flow(X_train, y_train_cat, batch_size=batch_size)

# Train the model
history = model.fit(
                    #train_generator,
                    X_train, 
                    y_train_cat,
                    batch_size=batch_size,
                    verbose=1,
                    epochs=epochs,
                    validation_data=(X_val, y_val_cat),
                    #class_weight=class_weights_dict,
                    shuffle=True,
                    callbacks=[tensorboard_callback, 
                               custom_callback, 
                               #nan_debug_callback,
                               #Gradient_monitoring_callback
                               ])


