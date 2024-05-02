from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None

def crop_image(image_path, output_folder,pixels_to_crop_from_all_sides):
    # Load the image
    image = Image.open(image_path)

    # Crop the image and save the smaller gray scale images
    cropped_image = image.crop((pixels_to_crop_from_all_sides, pixels_to_crop_from_all_sides, image.size[0]-pixels_to_crop_from_all_sides, image.size[1]-pixels_to_crop_from_all_sides))
    grayscale_image = cropped_image.convert("L")
    
    # Split grayscale image in to num_cols/num_rows smaller images
    width, height = grayscale_image.size
    num_cols = 4
    num_rows = 4
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * width/num_cols
            y = row * height/num_rows
            box = (x, y, x + width/num_cols, y + height/num_rows)
            cropped_image = grayscale_image.crop(box)
            output_path = os.path.join(output_folder, f"{row}_{col}_{os.path.basename(image_path)}")
            cropped_image.save(output_path)

    # Save the cropped image to output folder with new name
    #grayscale_image.save(os.path.join(output_folder, f"cropped_{pixels_to_crop_from_all_sides}_{os.path.basename(image_path)}"))

#example usage
image_path = "/home/student515/Documents/thesis/Dataset/Keyence_output/Al/test"  # Path to the input image
output_folder = "/home/student515/Documents/thesis/Dataset/Image_crop_for_labeling/Al/test"  # Path to output folder
pixels_to_crop_from_all_sides = 50  # Number of pixels to crop from all sides

#crop_image(image_path, output_folder, pixels_to_crop_from_all_sides)

#Use crop_image function on all images in image_path
for image in os.listdir(image_path):
    if image.endswith(".tif"):
        crop_image(os.path.join(image_path, image), output_folder, pixels_to_crop_from_all_sides)
        print(f"{image} cropped and saved to {output_folder}")
