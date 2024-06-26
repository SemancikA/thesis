from model import unet
from model_batch_norm import unet_batch_norm
from functions import split_images_masks_into_patches
from functions import GradientMonitoringCallback, NaNDebugCallback, CustomCallback
from functions import categorical_focal_loss
import os
from tensorflow.keras.utils import normalize
import numpy as np
import tensorflow as tf
#from matplotlib import pyplot as plt
#from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import datetime
from tensorflow.keras.callbacks import TensorBoard, Callback, EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

####iterative experiment setup####
####Global variables###
n_classes=3 #Number of classes for segmentation
epochs = 1500

dataset_size = 320
batch_sizes = [4,8,16,32] #max batch size = 32, pri 64 crash
resize_factors = [4] #4, None
model_depths = [2,4,8,16] #2,4,8 => model depth 4 best loss
model_types=[unet] #unet, unet_batch_norm => unet best loss
string="_exp-21-DOE-focal_loss"
image_sizes = [512,1024]

for resize_factor in resize_factors:
    for SIZE_X in image_sizes:
        for model_depth in model_depths:
            for batch_size in batch_sizes:
                for model_type in model_types:

                    if SIZE_X == 512 and model_depth == 16 and (batch_size == 16 or batch_size == 32):
                        continue

                    if SIZE_X==1024 and batch_size == 32:
                        continue

                    if SIZE_X==1024 and batch_size == 16 and model_depth != 2:
                        continue

                    if SIZE_X==1024 and batch_size == 8 and (model_depth == 8 or model_depth == 16):
                        continue       

                    SIZE_Y = SIZE_X
                    time_now=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    
                    images_path = "/home/student515/Documents/thesis/Dataset/Unet/images/Al"
                    masks_path = "/home/student515/Documents/thesis/Dataset/Unet/masks/Al"

                    images_patches_save_path = "/home/student515/Documents/thesis/Dataset/Unet/train_patches"
                    masks_patches_save_path = "/home/student515/Documents/thesis/Dataset/Unet/train_masks_patches"
                    model_save_dir='/home/student515/Documents/thesis/thesis/U-net_custom/model_save/' + time_now +"_image_size-"+str(SIZE_X) +"x"+ str(SIZE_Y) + "_batch-"+str(batch_size) + "_epochs-"+str(epochs) + "_resize_factor-" + str(resize_factor) + "_model_depth-"+ str(model_depth) + "_model_type-" + str(model_type)[10:-19] + string
                    log_dir = "/home/student515/Documents/thesis/thesis/U-net_custom/logs/"  + time_now +"_image_size-"+str(SIZE_X) +"x"+ str(SIZE_Y) + "_batch-"+str(batch_size) + "_epochs-"+str(epochs) + "_resize_factor-" + str(resize_factor) + "_model_depth-"+ str(model_depth) + "_model_type-" + str(model_type)[10:-19] + string
                    val_pred_save_dir = "/home/student515/Documents/thesis/Dataset/Unet/save_prediction_patches_train/"  + time_now +"_image_size-"+str(SIZE_X) +"x"+ str(SIZE_Y) + "_batch-"+str(batch_size) + "_epochs-"+str(epochs) + "_resize_factor-" + str(resize_factor) + "_model_depth-"+ str(model_depth) + "_model_type-" + str(model_type)[10:-19] + string
                    
                    os.makedirs(model_save_dir, exist_ok=True)
                    os.makedirs(val_pred_save_dir, exist_ok=True)
                    os.makedirs(val_pred_save_dir + "/X", exist_ok=True)
                    os.makedirs(val_pred_save_dir + "/Y", exist_ok=True)

                    ###############################################
                    ###Capture training images and masks info as a list
                    train_images, train_masks = split_images_masks_into_patches(images_path, masks_path, images_patches_save_path, masks_patches_save_path, SIZE_X, SIZE_Y, resize_factor)

                    #Convert list to array for machine learning processing        
                    train_images = np.array(train_images)
                    train_images = np.expand_dims(train_images, axis=3)
                            
                    train_masks = np.array(train_masks,dtype=np.uint8)

                    train_masks_flatten = train_masks.flatten()
                    labels, counts = np.unique(train_masks_flatten, return_counts=True)
                    #Compute class weights
                    class_weights = class_weight.compute_class_weight(
                        class_weight='balanced', 
                        classes=labels, 
                        y=train_masks_flatten)
                    
                    #Compute percentual ratio of each class in dataset
                    class_weights = class_weights/counts

                    print("Counts: ", counts)
                    print("Class weights: ", class_weights)

                    #Picking 15% for validation and remaining for training
                    X_train, X_val, y_train, y_val = train_test_split(train_images, train_masks, test_size = 0.15, random_state = 0)

                    #X_val, X_test, y_val, y_tesy = train_test_split(X_val_test, y_val_test, test_size = 0.5, random_state = 0)

                    # Save all validation images and masks to val_pred_save_dir
                    # for i in range(X_val.shape[0]):
                    #     img =X_val[i,:,:,0]
                    #     img = img.astype(np.uint8)
                    #     img = Image.fromarray(img)
                    #     img.save(val_pred_save_dir + "/X/image_" + str(i) + ".tif")

                    #     mask= y_val[i,:,:,0]*(255/2)
                    #     mask = mask.astype(np.uint8)
                    #     mask = Image.fromarray(mask)
                    #     mask.save(val_pred_save_dir + "/Y/mask_" + str(i) + ".tif")

                    X_train = normalize(X_train, axis=1)
                    X_val = normalize(X_val, axis=1)

                    train_masks_cat = to_categorical(y_train, num_classes=n_classes)
                    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

                    val_masks_cat = to_categorical(y_val, num_classes=n_classes)
                    y_val_cat = val_masks_cat.reshape((y_val.shape[0], y_val.shape[1], y_val.shape[2], n_classes))

                    IMG_HEIGHT = X_train.shape[1]
                    IMG_WIDTH  = X_train.shape[2]
                    IMG_CHANNELS = X_train.shape[3]

                    def get_model():
                        return model_type(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS, lay=model_depth)

                    model = get_model()
                    model.compile(optimizer='adam', loss=categorical_focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])

                    print(model.summary())

                    print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))

                    ###########################################
                    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
                    custom_callback = CustomCallback(model_save_dir=model_save_dir, X_val=X_val, y_val_cat=y_val_cat, tensorboard_log_dir=log_dir, val_pred_save_dir=val_pred_save_dir)
                    nan_debug_callback = NaNDebugCallback()
                    Gradient_monitoring_callback = GradientMonitoringCallback()
                    early_stop=EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')
                    checkpoint=ModelCheckpoint(
                        #filepath=model_save_dir + "/model-{epoch:02d}.hdf5",
                        filepath=f"{model_save_dir}/{time_now}_image_size-{SIZE_X}x{SIZE_Y}_batch-{batch_size}_epoch-{{epoch:02d}}_resize_factor-{resize_factor}_model_depth-{model_depth}_{string}.hdf5",
                        monitor='val_loss', 
                        verbose=0, 
                        save_best_only=True, 
                        mode='auto')
                    ############################################
                    time_now +"_image_size-"+str(SIZE_X) +"x"+ str(SIZE_Y) + "_batch-"+str(batch_size) + "_epochs-"+str(epochs) + "_resize_factor-" + str(resize_factor) + "_model_depth-"+ str(model_depth) + string

                    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))

                    def augment(image, label):
                        original_size = tf.shape(image)[0]

                        original_image_size = tf.shape(image)[:2]
                        original_label_size = tf.shape(label)[:2]
                        image_channels = tf.shape(image)[-1]
                        label_channels = tf.shape(label)[-1]

                        # Random brightness +-10%
                        image = tf.image.random_brightness(image, max_delta=0.1)
                        # Random flipping of the image and label
                        if tf.random.uniform(()) > 0.5:
                            image = tf.image.flip_left_right(image)
                            label = tf.image.flip_left_right(label)
                        if tf.random.uniform(()) > 0.5:
                            image = tf.image.flip_up_down(image)
                            label = tf.image.flip_up_down(label)
                        # Random rotation of image and label
                        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
                        image = tf.image.rot90(image, k)
                        label = tf.image.rot90(label, k)

                        # Scaling images to zoom in, in range 1.0 to 1.5
                        scale = tf.random.uniform([], 1.0, 1.5)  # Generate a scaling factor between 1.0 and 1.5
                        scaled_image_size = tf.cast(tf.cast(original_image_size, tf.float32) * scale, tf.int32)
                        scaled_label_size = tf.cast(tf.cast(original_label_size, tf.float32) * scale, tf.int32)

                        image = tf.image.resize(image, scaled_image_size, method=tf.image.ResizeMethod.BILINEAR)
                        label = tf.image.resize(label, scaled_label_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                        
                        # Crop back to original size at a random location using the same seed to ensure the same random location for image and label
                        seed = np.random.randint(1, 10000)
                        image_crop_size = tf.concat([original_image_size, [image_channels]], axis=0)
                        label_crop_size = tf.concat([original_label_size, [label_channels]], axis=0)
                        image = tf.image.random_crop(image, image_crop_size, seed=seed)
                        label = tf.image.random_crop(label, label_crop_size, seed=seed)

                        return image, label
                    
                    augmented_dataset = dataset.map(augment).repeat()
                    augmented_dataset = augmented_dataset.shuffle(buffer_size=(dataset_size)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

                    # #save few augmented images for verification
                    # i=0
                    # augmented_dir = "/home/student515/Documents/thesis/Dataset/Unet/augmented"
                    # os.makedirs(augmented_dir, exist_ok=True)

                    # #dataset verification
                    # augmented_dataset_save = augmented_dataset
                    # def save_augmented_images(dataset, num_samples=2):
                    #     for i, (images, masks) in enumerate(dataset.take(num_samples)):
                    #         images = images.numpy()  # Convert to numpy array
                    #         masks = masks.numpy()  # Convert to numpy array
                    #         for j in range(images.shape[0]):
                    #             image = images[j]
                    #             mask = masks[j]
                    #             # Normalize and convert image to uint8
                    #             img= ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                    #             img= img[:,:,0]
                    #             img = Image.fromarray(img)
                    #             img.save(os.path.join(augmented_dir, f"image_{i}_{j}.tif"))

                    #             # Normalize and convert mask to uint8
                    #             mask = tf.argmax(mask, axis=-1)
                    #             mask = mask.numpy()
                    #             mask = (mask * 255/2).astype(np.uint8)
                    #             mask = Image.fromarray(mask)
                    #             mask.save(os.path.join(augmented_dir, f"mask_{i}_{j}.tif"))
                    
                    # save_augmented_images(augmented_dataset_save)

                    # Train the model
                    history = model.fit(
                                        augmented_dataset,
                                        steps_per_epoch=dataset_size // batch_size,
                                        verbose=1,
                                        epochs=epochs,
                                        validation_data=(X_val, y_val_cat),
                                        shuffle=True,
                                        callbacks=[tensorboard_callback, 
                                                custom_callback,
                                                #early_stop,
                                                checkpoint, 
                                                ])


