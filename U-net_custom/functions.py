import cv2
import glob
import os
import nrrd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import Callback
from PIL import Image

from tensorflow.keras.callbacks import Callback
import os

class CustomCallback(Callback):
    def __init__(self, model_save_dir,val_pred_save_dir, X_val, y_val_cat, tensorboard_log_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_save_dir = model_save_dir
        self.val_pred_save_dir = val_pred_save_dir
        self.X_val = X_val  # Using X_test as validation data
        self.y_val_cat = y_val_cat  # Using y_test_cat as validation labels
        self.tensorboard_log_dir = tensorboard_log_dir + "/IoU"
        self.writer = tf.summary.create_file_writer(tensorboard_log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            #model_path = os.path.join(self.model_save_dir, f"Unet{epoch+1}.hdf5")
            #self.model.save(model_path)
            #print(f"Model saved at {model_path}")

            # Calculate IoU for the test data
            preds = self.model.predict(self.X_val)
            iou, iou_values = self.calculate_iou_and_save_pred(self.y_val_cat, preds, epoch, self.val_pred_save_dir)
            
            # Log IoU to TensorBoard
            with self.writer.as_default():
                tf.summary.scalar('IoU', iou, step=epoch)
                self.writer.flush()

            print(f"Logging info at epoch {epoch+1}: Loss: {logs.get('loss')}, Accuracy: {logs.get('accuracy')}, IoU: {iou}")
            print(f"IoU values: {iou_values}")


    def calculate_iou_and_save_pred(self, y_true, y_pred, epoch,val_pred_save_dir, n_classes=3):            
        y_pred_argmax=np.argmax(y_pred, axis=-1)
        y_true_argmax = tf.argmax(y_true, axis=-1)
        IOU_keras = MeanIoU(num_classes=n_classes)
        IOU_keras.update_state(y_true_argmax,y_pred_argmax)
        iou = IOU_keras.result().numpy()
        iou_values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
        
        # Save the predicted masks to create .gif
        # for i in range(len(y_pred_argmax)):
        #     pred_image = y_pred_argmax[i]*127
        #     pred_image = pred_image.astype(np.uint8)
        #     pred_image = Image.fromarray(pred_image)
        #     pred_image.save(os.path.join(val_pred_save_dir, f"pred_mask_{i}_{epoch+1}.tif"))

        return iou, iou_values
    

class NaNDebugCallback(Callback):
    def on_batch_end(self, batch, logs=None):
        if np.isnan(logs.get('loss')):
            print(f'NaN detected in loss at end of batch {batch}')
            self.model.stop_training = True

        for layer in self.model.layers:
            weights_layer = layer.get_weights()
            if any(np.isnan(w).any() for w in weights_layer):
                print(f'NaN detected in layer weights at end of batch {batch}, layer {layer.name}')
                self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        if np.isnan(logs.get('loss')):
            print(f'NaN detected in loss at epoch {epoch}')
            self.model.stop_training = True

class GradientMonitoringCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Access the model
        model = self.model
        # Select the loss and optimizer
        loss_fn = model.loss
        optimizer = model.optimizer
        
        # Iterate through batches in the dataset (example using training data)
        for x_batch_train, y_batch_train in model.train_dataset:
            with tf.GradientTape() as tape:
                # Watch the model's inputs
                tape.watch(model.trainable_variables)
                # Compute the predictions and loss
                predictions = model(x_batch_train, training=True)
                loss = loss_fn(y_batch_train, predictions)
            
            # Compute the gradients of the loss w.r.t. the model's weights
            gradients = tape.gradient(loss, model.trainable_variables)
            # Calculate the gradient norms
            gradient_norms = [tf.norm(grad).numpy() for grad in gradients if grad is not None]
            
            # Log or print the gradient norms
            print(f"Epoch {epoch}: Gradient norms: {gradient_norms}")
            break  # Remove or modify depending on how many batches you want to process per epoch


def split_images_masks_into_patches(image_path, mask_path, image_save_path=None, mask_save_path=None, patch_height=512, patch_width=512, resize_factor=None):
    image_paths = sorted(glob.glob(os.path.join(image_path, "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(mask_path, "*.nrrd")))
    image_patches = []
    mask_patches = []

    if len(image_paths) != len(mask_paths):
        raise ValueError("The number of images and masks must be the same.")

    for img_path, msk_path in zip(image_paths, mask_paths):
        img_array = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_height, img_width = img_array.shape[:2]
        mask_array, header = nrrd.read(msk_path)
        mask_array = np.array(mask_array, dtype=np.uint8)
        mask_array = np.rot90(mask_array, axes=(1, 0))
        mask_array = np.flip(mask_array, axis=1)
        mask_array = mask_array-1
        mask_height, mask_width = mask_array.shape[:2]

        if resize_factor:
            patch_height_resized, patch_width_resized = int(patch_height * resize_factor), int(patch_width * resize_factor)
        else:
            patch_height_resized, patch_width_resized = patch_height, patch_width

        n, m = min(img_height, mask_height) // patch_height_resized, min(img_width, mask_width) // patch_width_resized

        for i in range(n):
            for j in range(m):
                start_row, start_col = i * patch_height_resized, j * patch_width_resized
                end_row, end_col = start_row + patch_height_resized, start_col + patch_width_resized
                
                # Process and save image patch
                if resize_factor:
                    img_patch = cv2.resize(img_array[start_row:end_row, start_col:end_col], (patch_width, patch_height), interpolation=cv2.INTER_CUBIC)
                else:
                    img_patch = img_array[start_row:end_row, start_col:end_col]
                
                image_patches.append(img_patch)

                # Process and save mask patch
                if resize_factor:
                    mask_patch = cv2.resize(mask_array[start_row:end_row, start_col:end_col], (patch_width, patch_height), interpolation=cv2.INTER_NEAREST)
                    mask_patch = np.expand_dims(mask_patch, axis=2)
                else:
                    mask_patch = mask_array[start_row:end_row, start_col:end_col]
                
                if image_save_path:
                    img_patch_path = os.path.join(image_save_path, f"img_patch_{i}_{j}.tif")
                    cv2.imwrite(img_patch_path, img_patch)    
                    mask_patch_path = os.path.join(mask_save_path, f"mask_patch_{i}_{j}.tif")
                    cv2.imwrite(mask_patch_path, (mask_patch*127))

                mask_patches.append(mask_patch)

    return image_patches, mask_patches


def categorical_focal_loss(gamma=2., alpha=0.25):
    """
    Focal loss for multi-class classification.
    This loss function is designed to address class imbalance by focusing more on hard-to-classify examples.
    
    Parameters:
        gamma (float): Focusing parameter. Default is 2.0.
        alpha (float): Balancing factor. Default is 0.25.
        
    Returns:
        loss (function): A loss function taking (y_true, y_pred) as arguments and returning a loss value.
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        
        # Calculate loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        return K.mean(K.sum(loss, axis=-1))
    
    return focal_loss_fixed