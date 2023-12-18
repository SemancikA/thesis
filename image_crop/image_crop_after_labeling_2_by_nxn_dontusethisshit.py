import os
import random
import shutil
from PIL import Image

# Define input and output folder paths
input_folder = "/home/student515/Documents/thesis/Dataset/Image_crop_for_labeling/Al/"
train_folder = "/home/student515/Documents/thesis/Dataset/SRGAN_train"
test_folder = "/home/student515/Documents/thesis/Dataset/SRGAN_test"
val_folder = "/home/student515/Documents/thesis/Dataset/SRGAN_val"

# Define ratio of images to be split into train, test, and validation sets
train_ratio = 0.7
test_ratio = 0.2
val_ratio = 0.1

# Size of the smaller images (n x n)
split_size = 1200 # Change this to your desired size

# Create output folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Function to split image into smaller segments
def split_image(image_path, split_size):
    img = Image.open(image_path)
    width, height = img.size

    # Calculate the number of splits along width and height
    h_splits = width // split_size
    v_splits = height // split_size

    # List to store the segments
    segments = []

    for i in range(h_splits):
        for j in range(v_splits):
            left = i * split_size
            upper = j * split_size
            right = (i + 1) * split_size
            lower = (j + 1) * split_size

            segment = img.crop((left, upper, right, lower))
            segments.append(segment)

    return segments

# Get list of all image file names in input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

# Shuffle list of image file names randomly
random.shuffle(image_files)

# Calculate number of images for each set based on defined ratios
num_train = int(len(image_files) * train_ratio)
num_test = int(len(image_files) * test_ratio)
num_val = int(len(image_files) * val_ratio)

# Loop through shuffled list of image file names and process them
for i, image_file in enumerate(image_files):
    segments = split_image(os.path.join(input_folder, image_file), split_size)

    for idx, segment in enumerate(segments):
        output_file = f"{os.path.splitext(image_file)[0]}_segment_{idx}.tif"
        if i < num_train:
            segment.save(os.path.join(train_folder, output_file))
        elif i < num_train + num_test:
            segment.save(os.path.join(test_folder, output_file))
        else:
            segment.save(os.path.join(val_folder, output_file))
