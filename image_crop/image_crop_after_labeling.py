
import os
import random
import shutil

# Define input and output folder paths
input_folder = "/home/student515/Documents/thesis/Dataset/Image_crop_for_labeling/Al/"
train_folder = "/home/student515/Documents/thesis/Dataset/SRGAN_train"
test_folder = "/home/student515/Documents/thesis/Dataset/SRGAN_test"
val_folder = "//home/student515/Documents/thesis/Dataset/SRGAN_val"

# Define ratio of images to be split into train, test, and validation sets
train_ratio = 0.7
test_ratio = 0.2
val_ratio = 0.1

# Create output folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Get list of all image file names in input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

# Shuffle list of image file names randomly
random.shuffle(image_files)

# Calculate number of images for each set based on defined ratios
num_train = int(len(image_files) * train_ratio)
num_test = int(len(image_files) * test_ratio)
num_val = int(len(image_files) * val_ratio)

# Loop through shuffled list of image file names and copy them to appropriate output folder
for i, image_file in enumerate(image_files):
    if i < num_train:
        shutil.copy(os.path.join(input_folder, image_file), train_folder)
    elif i < num_train + num_test:
        shutil.copy(os.path.join(input_folder, image_file), test_folder)
    else:
        shutil.copy(os.path.join(input_folder, image_file), val_folder)
