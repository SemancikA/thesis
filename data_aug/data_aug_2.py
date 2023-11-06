import os
from PIL import Image
import random
from PIL import ImageEnhance

# Define input and output directories
input_dir = "/home/adam/Documents/thesis/thesis/data_aug/input"
output_dir = "/home/adam/Documents/thesis/thesis/data_aug/output"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#Create variables for brightness shift value
bshif_min = 0.90
bshif_max = 1.10

# Iterate through all images in input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".tif") or filename.endswith(".png"):
        # Open original image, apply brightness shift, and save to output directory
        img = Image.open(os.path.join(input_dir, filename))
        enhancer = ImageEnhance.Brightness(img)
        img_bshif = enhancer.enhance(random.uniform(bshif_min, bshif_max))
        img_bshif.save(os.path.join(output_dir, filename))

        img_horizontal_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        enhancer = ImageEnhance.Brightness(img_horizontal_flip)
        img_horizontal_flip = enhancer.enhance(random.uniform(bshif_min, bshif_max))
        img_horizontal_flip.save(os.path.join(output_dir, "horizontal_flip_" + filename))

        # Rotate image by 90 degrees and save to output directory, horizontal flip 90 degrees and save to output directory
        img_rotated_90 = img.rotate(90)
        enhancer = ImageEnhance.Brightness(img_rotated_90)
        img_rotated_90_bshif = enhancer.enhance(random.uniform(bshif_min, bshif_max))
        img_rotated_90_bshif.save(os.path.join(output_dir, "rotated_90_" + filename))

        img_rotated_90_horizontal_flip = img_rotated_90.transpose(Image.FLIP_LEFT_RIGHT)
        enhancer = ImageEnhance.Brightness(img_rotated_90_horizontal_flip)
        img_rotated_90_horizontal_flip_bshif = enhancer.enhance(random.uniform(bshif_min, bshif_max))
        img_rotated_90_horizontal_flip_bshif.save(os.path.join(output_dir, "rotated_90_horizontal_flip_" + filename))

        # Rotate image by 180 degrees and save to output directory, horizontal flip 180 degrees and save to output directory
        img_rotated_180 = img.rotate(180)
        enhancer = ImageEnhance.Brightness(img_rotated_180)
        img_rotated_180_bshif = enhancer.enhance(random.uniform(bshif_min, bshif_max))
        img_rotated_180_bshif.save(os.path.join(output_dir, "rotated_180_" + filename))

        img_rotated_180_horizontal_flip = img_rotated_180.transpose(Image.FLIP_LEFT_RIGHT)
        enhancer = ImageEnhance.Brightness(img_rotated_180_horizontal_flip)
        img_rotated_180_horizontal_flip_bshif = enhancer.enhance(random.uniform(bshif_min, bshif_max))
        img_rotated_180_horizontal_flip_bshif.save(os.path.join(output_dir, "rotated_180_horizontal_flip_" + filename))

        # Rotate image by 270 degrees and save to output directory, horizontal flip 270 degrees and save to output directory
        img_rotated_270 = img.rotate(270)
        enhancer = ImageEnhance.Brightness(img_rotated_270)
        img_rotated_270_bshif = enhancer.enhance(random.uniform(bshif_min, bshif_max))
        img_rotated_270_bshif.save(os.path.join(output_dir, "rotated_270_" + filename))

        img_rotated_270_horizontal_flip = img_rotated_270.transpose(Image.FLIP_LEFT_RIGHT)
        enhancer = ImageEnhance.Brightness(img_rotated_270_horizontal_flip)
        img_rotated_270_horizontal_flip_bshif = enhancer.enhance(random.uniform(bshif_min, bshif_max))
        img_rotated_270_horizontal_flip_bshif.save(os.path.join(output_dir, "rotated_270_horizontal_flip_" + filename))
