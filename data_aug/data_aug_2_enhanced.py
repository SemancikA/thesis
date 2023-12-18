import os
import random
import logging
from PIL import Image, ImageEnhance

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s: %(message)s')

# Define input and output directories
input_dir = "/home/student515/Documents/thesis/Dataset/SRGAN_train"
output_dir = "/home/student515/Documents/thesis/Dataset/SRGAN_train_aug"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define brightness shift value range
bshif_min = 0.90
bshif_max = 1.10

# Helper function to enhance and save images
def enhance_and_save_image(img, filename, transformation):
    try:
        enhancer = ImageEnhance.Brightness(img)
        img_enhanced = enhancer.enhance(random.uniform(bshif_min, bshif_max))
        new_filename = f"{transformation}_{filename}"
        img_enhanced.save(os.path.join(output_dir, new_filename))
        logging.info(f"Saved {new_filename}")
    except Exception as e:
        logging.error(f"Failed to enhance and save image {filename}: {e}")

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".tif", ".png")):
        try:
            with Image.open(os.path.join(input_dir, filename)) as img:
                # Original image with brightness shift
                enhance_and_save_image(img, filename, "original_bshif")

                # Horizontal flip with brightness shift
                img_horizontal_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
                enhance_and_save_image(img_horizontal_flip, filename, "horizontal_flip")

                # Perform rotations with brightness shift and horizontal flip
                for angle in [90, 180, 270]:
                    img_rotated = img.rotate(angle)
                    enhance_and_save_image(img_rotated, filename, f"rotated_{angle}")
                    img_rotated_flip = img_rotated.transpose(Image.FLIP_LEFT_RIGHT)
                    enhance_and_save_image(img_rotated_flip, filename, f"rotated_{angle}_horizontal_flip")

        except IOError as e:
            logging.error(f"Cannot open or process image {filename}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
