import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import capture as ai
import shutil
import os


selected_image_path = None  # Store the selected image path
process_image = None  # Store the processed image

def run (file_path):
    return ai.runModel(file_path)  # Should return processed image path


def upload_image():
    """Opens a file dialog to select an image and displays it."""
    global selected_image_path
    global process_image
    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.gif *.bmp")]
    )
    if file_path:
        selected_image_path = file_path
        try:
            image = Image.open(file_path)
            image.thumbnail((200, 200))
            img_tk = ImageTk.PhotoImage(image)
            image_label.config(image=img_tk)
            image_label.image = img_tk
            messagebox.showinfo("Selected", f"Selected image: {file_path}")
            process_image = run(file_path)  # Save returned path
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    else:
        messagebox.showwarning("Cancelled", "Image selection cancelled.")

def save_image():
    """Saves the selected image to a chosen directory and removes it from the app screen."""
    global process_image
    global selected_image_path
    if not process_image:
        messagebox.showwarning("No Image", "Please upload an image first.")
        return

    save_directory = filedialog.askdirectory(title="Select Save Directory")
    if save_directory:
        try:
            file_name = os.path.basename(process_image)
            destination_path = os.path.join(save_directory, file_name)
            shutil.copy(process_image, destination_path)
            messagebox.showinfo("Success", f"Image saved to: {destination_path}")
            # Remove image from app screen
            image_label.config(image='')
            image_label.image = None
            selected_image_path = None
            process_image = None  # Clear after saving
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    else:
        messagebox.showwarning("Cancelled", "Image save cancelled.")

# Create the main application window
root = tk.Tk()
root.title("Image Uploader and Saver")
root.geometry("400x400")  # Set start app size to 400x400

# Image display label
image_label = tk.Label(root)
image_label.pack(pady=10)

# Upload and Save buttons
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=5)

save_button = tk.Button(root, text="Save Image", command=save_image)
save_button.pack(pady=5)

root.mainloop()