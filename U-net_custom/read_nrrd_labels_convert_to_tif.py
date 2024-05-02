# This script reads the .nrrd mask files and converts them to .tif images using cv2.
import nrrd
import os
import glob
import cv2

#Capture training labels info as a list
mask_path = "/home/student515/Documents/thesis/Dataset/Unet/mask_verification"

# Read the .nrrd mask as a NumPy array
for directory_path in glob.glob(mask_path):
    for msk_path in glob.glob(os.path.join(directory_path, "*.nrrd")):
        mask_array, header = nrrd.read(msk_path)
        #Save masks as .tif images using cv2
        cv2.imwrite(msk_path[:-5]+".tif", mask_array)