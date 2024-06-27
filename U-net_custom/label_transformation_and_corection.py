# This script reads the .nrrd mask files and converts them to .tif images using cv2.
import nrrd
import os
import glob
import cv2
import numpy as np

#Capture training labels info as a list
mask_path = "/home/student515/Documents/thesis/Dataset/Unet/masks"

# Read the .nrrd mask as a NumPy array
for directory_path in glob.glob(mask_path):
    for msk_path in glob.glob(os.path.join(directory_path, "*.nrrd")):
        mask_array, header = nrrd.read(msk_path)
        #rotate the mask array 90 degrees anticlockwise
        mask_array = np.rot90(mask_array, axes=(1, 0))
        #flip the mask array along the y-axis
        mask_array = np.flip(mask_array, axis=1)
        #Save masks as .tif images using cv2
        cv2.imwrite(msk_path[:-5]+"_transformed.tif", mask_array)
        #save the mask array in nrrd format
        nrrd.write(msk_path[:-5]+"_transformed.nrrd", mask_array, header)