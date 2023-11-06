import os
import cv2

def rotate_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # Rotate each image 4 times and save to the output folder
    for file_name in image_files:
        # Read the image
        image_path = os.path.join(input_folder, file_name)
        image = cv2.imread(image_path)

        # Rotate and save the image 4 times
        for i in range(4):
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE * (i + 1))

            # Generate the output file name
            output_file_name = f"rotated_{i+1}_{file_name}.png"
            output_file_path = os.path.join(output_folder, output_file_name)

            # Save the rotated image to the output folder
            cv2.imwrite(output_file_path, rotated_image)

            print(f"Rotated {file_name} {i+1} time(s) and saved as {output_file_name}")


# Specify the input and output folders
input_folder = "/home/press/Documents/thesis/thesis/data aug/input"
output_folder = "/home/press/Documents/thesis/thesis/data aug/output"

# Call the function to rotate and save the images
rotate_images(input_folder, output_folder)