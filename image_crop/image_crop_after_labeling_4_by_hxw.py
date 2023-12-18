import os
import random
import cv2

# Define input and output folder paths
input_folder = "/home/student515/Documents/thesis/Dataset/Image_crop_for_labeling/Al"
train_folder = "/home/student515/Documents/thesis/Dataset/SRGAN_train_test"
test_folder = "/home/student515/Documents/thesis/Dataset/SRGAN_test"
val_folder = "/home/student515/Documents/thesis/Dataset/SRGAN_val_test"

# Define ratio of images to be split into train, test, and validation sets
train_ratio = 0.7
test_ratio = 0.15
val_ratio = 0.15

# Number of segments in x and y directions
x_segments = 4  # Number of horizontal segments
y_segments = 4  # Number of vertical segments

# Create output folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Function to split image into smaller segments based on x and y segments
def split_image(image_path, x_segments, y_segments):
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Calculate the width and height of each segment
    segment_width = width // x_segments
    segment_height = height // y_segments

    # List to store the segments
    segments = []

    for i in range(x_segments):
        for j in range(y_segments):
            left = i * segment_width
            upper = j * segment_height
            right = (i + 1) * segment_width
            lower = (j + 1) * segment_height

            segment = img[upper:lower, left:right]
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
    segments = split_image(os.path.join(input_folder, image_file), x_segments, y_segments)

    for idx, segment in enumerate(segments):
        output_file = f"{os.path.splitext(image_file)[0]}_segment_{idx}.tif"
        if i < num_train:
            cv2.imwrite(os.path.join(train_folder, output_file), segment)
        elif i < num_train + num_test:
            cv2.imwrite(os.path.join(test_folder, output_file), segment)
        else:
            cv2.imwrite(os.path.join(val_folder, output_file), segment)
