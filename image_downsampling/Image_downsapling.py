
import os
from PIL import Image

input_folder_train = "/home/student515/Documents/thesis/Dataset/SRGAN_train_aug"
output_folder_train = "/home/student515/Documents/thesis/Dataset/SRGAN_train_aug_LR"

input_folder_val = "/home/student515/Documents/thesis/Dataset/SRGAN_val"
output_folder_val = "/home/student515/Documents/thesis/Dataset/SRGAN_val_LR"

# Loop through all files in input folder
for filename in os.listdir(input_folder_train):
    # Open image
    with Image.open(os.path.join(input_folder_train, filename)) as img:
        # Downsample by 4x using nearest neighbor interpolation
        img = img.resize((img.width // 4, img.height // 4), resample=Image.NEAREST)
        # Save downsampled image to output folder
        img.save(os.path.join(output_folder_train, filename))

# Loop through all files in input folder
for filename in os.listdir(input_folder_val):
    # Open image
    with Image.open(os.path.join(input_folder_val, filename)) as img:
        # Downsample by 4x using nearest neighbor interpolation
        img = img.resize((img.width // 4, img.height // 4), resample=Image.NEAREST)
        # Save downsampled image to output folder
        img.save(os.path.join(output_folder_val, filename))        
