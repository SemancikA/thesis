from PIL import Image
import os

def split_image(image_path, output_folder, width, height):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the image
    image = Image.open(image_path)
    image_width, image_height = image.size
    
    # Calculate the number of smaller images needed
    num_cols = image_width // width
    num_rows = image_height // height
    
    # Split the image and save the smaller gray scale images
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * width
            y = row * height
            box = (x, y, x + width, y + height)
            cropped_image = image.crop(box)
            output_path = os.path.join(output_folder, f"image_{row}_{col}.png")
            grayscale_image = cropped_image.convert("L")
            grayscale_image.save(output_path)
    
    print(f"Image split into {num_rows * num_cols} smaller images.")

# Example usage
image_path = "/home/press/Documents/thesis/thesis/image_split/input/Cu99_015_TD_01_L_1000x.tif"  # Path to the input image
output_folder = "/home/press/Documents/thesis/thesis/image_split/output"  # Path to the output folder
width = 2048  # Width of the smaller images
height = 2048  # Height of the smaller images

split_image(image_path, output_folder, width, height)
