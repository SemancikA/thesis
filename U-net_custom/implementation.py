import tkinter as tk
from tkinter import filedialog, messagebox
import os
from model import unet
import glob
import cv2
import numpy as np
from tensorflow.keras.utils import normalize
from PIL import Image

def browse_path(entry):
    """ Open a dialog to choose a directory and update the entry with the selected path. """
    directory = filedialog.askdirectory()
    if directory:
        entry.delete(0, tk.END)
        entry.insert(0, directory)

def run_processing():
    input_path = input_entry.get()
    output_path = output_entry.get()
    if os.path.exists(input_path) and os.path.exists(output_path):
        results = image_processing(input_path, output_path)
        LoF = results[0]*100
        Gas = results[1]*100
        Base = results[2]*100
        result_label.config(text=f"Lack of Fusion: {str(LoF)}%, Gas porosity: {str(Gas)}%, Base material: {str(Base)}%")
    else:
        messagebox.showerror("Error", "Check the input and output paths.")

def image_processing(input_path, output_path):
    n_classes = 3
    model_type=unet
    weights_paths="/home/student515/Documents/thesis/thesis/U-net_custom/model_final/20240430-184855_image_size-512x512_batch-4_epochs-1312_resize_factor-4_model_depth-4_model_type-unet_exp-20-DoE.hdf5"
    SIZE_X = 512
    SIZE_Y = SIZE_X
    model_depth = 4

    #load images
    image_paths = sorted(glob.glob(os.path.join(input_path, "*.tif")))
    image_patches = []

    #split images into patches to load them into the model
    for img_path in image_paths:
        img_array = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_height, img_width = img_array.shape[:2]
        patch_height, patch_width = SIZE_X, SIZE_Y
        n= img_height // patch_height
        m= img_width // patch_width

        for i in range(n):
            for j in range(m):
                start_row, start_col = i * patch_height, j * patch_width
                end_row, end_col = start_row + patch_height, start_col + patch_width
                img_patch = img_array[start_row:end_row, start_col:end_col]
                image_patches.append(img_patch)

    image_patches = np.array(image_patches)
    image_patches = np.expand_dims(image_patches, axis=3)

    #normalize images
    X_images = normalize(image_patches, axis=1)

    IMG_HEIGHT = X_images.shape[1]
    IMG_WIDTH  = X_images.shape[2]
    IMG_CHANNELS = X_images.shape[3]

    def get_model():
        return model_type(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS, lay=model_depth)
    
    model = get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    #load weights and predict
    model.load_weights(weights_paths)
    y_pred = model.predict(X_images)
    y_pred_argmax = np.argmax(y_pred, axis=3)
    
    #save predicted images and input images
    for i in range (len(image_patches)):
        img=image_patches[i,:,:,0]
        img=img.astype(np.uint8)
        img = Image.fromarray(img)
        img.save(output_path + "/image_" + str(i) + ".tif")

        pred_image=y_pred_argmax[i,]*127.5
        pred_image=pred_image.astype(np.uint8)
        pred_image = Image.fromarray(pred_image)
        pred_image.save(output_path + "/pred_" + str(i) + ".tif")  

    #Compute percentual ratio of each predicted class
    pred_flatten = y_pred_argmax.flatten()
    labels, counts = np.unique(pred_flatten, return_counts=True)
    ratios = counts / len(pred_flatten)
    results = ratios
    return results
    

# Set up the main GUI window
root = tk.Tk()
root.title("Metallography image processor")

# Text to explain the purpose of the program
explanation = tk.Label(root, text="This program processes metallography images to detect defects. Program expects input images in .tif format ONLY.")
explanation.pack(pady=5)

explanation2 = tk.Label(root, text="Use 250X magnification lense on KEYENCE microscope in order to work correctly.")
explanation2.pack(pady=5)

explanation3 = tk.Label(root, text="Program processes multiple images in input folder at once. Resulted predictions are saved in output folder.")
explanation3.pack(pady=5)

# Create entries and buttons for directory paths
input_entry = tk.Entry(root, width=50)
input_entry.pack(pady=10)
input_button = tk.Button(root, text="Browse image input folder", command=lambda: browse_path(input_entry))
input_button.pack()

output_entry = tk.Entry(root, width=50)
output_entry.pack(pady=10)
output_button = tk.Button(root, text="Browse image save folder", command=lambda: browse_path(output_entry))
output_button.pack()

# Create a button to run the processing
run_button = tk.Button(root, text="Run Processing", command=run_processing)
run_button.pack(pady=20)

# Label to display results
result_label = tk.Label(root, text="Percentual ratio of each detected defect displays here.")
result_label.pack(pady=20)

root.mainloop()