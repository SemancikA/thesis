import customtkinter as ctk
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
        entry.delete(0, ctk.END)
        entry.insert(0, directory)

def run_processing():
    input_path = input_entry.get()
    output_path = output_entry.get()
    if os.path.exists(input_path) and os.path.exists(output_path):
        results = image_processing(input_path, output_path)
        LoF = results[0] * 100
        Gas = results[1] * 100
        Base = results[2] * 100
        result_label.config(text=f"Lack of Fusion: {LoF:.2f}%, Gas porosity: {Gas:.2f}%, Base material: {Base:.2f}%")
    else:
        messagebox.showerror("Error", "Check the input and output paths.")

def image_processing(input_path, output_path):
    n_classes = 3
    model_type = unet
    weights_path = "/home/student515/Documents/thesis/thesis/U-net_custom/model_final/20240430-184855_image_size-512x512_batch-4_epochs-1312_resize_factor-4_model_depth-4_model_type-unet_exp-20-DoE.hdf5"
    SIZE_X = 512
    SIZE_Y = SIZE_X
    model_depth = 4

    image_paths = sorted(glob.glob(os.path.join(input_path, "*.tif")))
    image_patches = []

    for img_path in image_paths:
        img_array = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_height, img_width = img_array.shape[:2]
        patch_height, patch_width = SIZE_X, SIZE_Y
        n = img_height // patch_height
        m = img_width // patch_width

        for i in range(n):
            for j in range(m):
                start_row, start_col = i * patch_height, j * patch_width
                end_row, end_col = start_row + patch_height, start_col + patch_width
                img_patch = img_array[start_row:end_row, start_col:end_col]
                image_patches.append(img_patch)

    image_patches = np.array(image_patches)
    image_patches = np.expand_dims(image_patches, axis=3)

    X_images = normalize(image_patches, axis=1)

    IMG_HEIGHT = X_images.shape[1]
    IMG_WIDTH = X_images.shape[2]
    IMG_CHANNELS = X_images.shape[3]

    def get_model():
        return model_type(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS, lay=model_depth)

    model = get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(weights_path)
    y_pred = model.predict(X_images)
    y_pred_argmax = np.argmax(y_pred, axis=3)

    for i in range(len(image_patches)):
        img = image_patches[i, :, :, 0]
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(output_path, f"image_{i}.tif"))

        pred_image = y_pred_argmax[i, :] * 127.5
        pred_image = pred_image.astype(np.uint8)
        pred_image = Image.fromarray(pred_image)
        pred_image.save(os.path.join(output_path, f"pred_{i}.tif"))

    pred_flatten = y_pred_argmax.flatten()
    labels, counts = np.unique(pred_flatten, return_counts=True)
    ratios = counts / len(pred_flatten)
    return ratios

# Set up the main GUI window
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Metallography Image Processor")
root.geometry("700x400")

# Frame for explanations
explanation_frame = ctk.CTkFrame(root)
explanation_frame.pack(pady=10, padx=10, fill="x")

explanation_text = (
    "This program processes metallography images to detect defects. Program expects input images in .tif format ONLY.\n"
    "Use 250X magnification lens on KEYENCE microscope in order to work correctly.\n"
    "Program processes multiple images in input folder at once. Resulted predictions are saved in output folder."
)

explanation_box = ctk.CTkTextbox(explanation_frame, width=550, height=100, wrap='word')
explanation_box.insert("1.0", explanation_text)
explanation_box.configure(state='disabled')
explanation_box.pack(pady=5)

# Frame for input and output paths
path_frame = ctk.CTkFrame(root)
path_frame.pack(pady=10, padx=10, fill="x")

input_label = ctk.CTkLabel(path_frame, text="Input Folder:")
input_label.grid(row=0, column=0, pady=5, padx=5, sticky="w")
input_entry = ctk.CTkEntry(path_frame, width=400)
input_entry.grid(row=0, column=1, pady=5, padx=5)
input_button = ctk.CTkButton(path_frame, text="Browse", command=lambda: browse_path(input_entry))
input_button.grid(row=0, column=2, pady=5, padx=5)

output_label = ctk.CTkLabel(path_frame, text="Output Folder:")
output_label.grid(row=1, column=0, pady=5, padx=5, sticky="w")
output_entry = ctk.CTkEntry(path_frame, width=400)
output_entry.grid(row=1, column=1, pady=5, padx=5)
output_button = ctk.CTkButton(path_frame, text="Browse", command=lambda: browse_path(output_entry))
output_button.grid(row=1, column=2, pady=5, padx=5)

# Run processing button
run_button = ctk.CTkButton(root, text="Run Processing", command=run_processing)
run_button.pack(pady=20)

# Label to display results
result_label = ctk.CTkLabel(root, text="Percentual ratio of each detected defect displays here.")
result_label.pack(pady=20)

root.mainloop()
