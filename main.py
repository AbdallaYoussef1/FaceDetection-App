import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Load cascade files
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

# Error check
if face_cascade.empty():
    raise IOError("Face cascade file not found")
if eye_cascade.empty():
    raise IOError("Eye cascade file not found")
if nose_cascade.empty():
    raise IOError("Nose cascade file not found")

# Load filter images
hat_image = cv2.imread('hat.png', cv2.IMREAD_UNCHANGED)
mustache_image = cv2.imread('mustache.png', cv2.IMREAD_UNCHANGED)

# Error check
if hat_image is None:
    raise FileNotFoundError("Hat image file 'hat.png' not found")
if mustache_image is None:
    raise FileNotFoundError("Mustache image file 'mustache.png' not found")

# Maximum width and height
MAX_WIDTH = 800
MAX_HEIGHT = 600

# Flags for filter states
add_filters_flag = False
add_freckles_flag = False

# Original image without any filters
original_image = None

def resize_image(image, max_width, max_height):
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        image = cv2.resize(image, (int(width * scaling_factor), int(height * scaling_factor)))
    return image

def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[0], overlay.shape[1]
    if x >= background.shape[1] or y >= background.shape[0]:
        return background

    if x + w > background.shape[1]:
        w = background.shape[1] - x
        overlay = overlay[:, :w]

    if y + h > background.shape[0]:
        h = background.shape[0] - y
        overlay = overlay[:h]

    if overlay.shape[2] == 4:
        alpha_overlay = overlay[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha_overlay

        for c in range(0, 3):
            background[y:y + h, x:x + w, c] = (alpha_overlay * overlay[:, :, c] +
                                               alpha_background * background[y:y + h, x + w, c])

    return background

def add_filters(image):
    global add_filters_flag
    global add_freckles_flag

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Add hat and mustache
        if add_filters_flag:
            hat_resized = cv2.resize(hat_image, (w, int(w * hat_image.shape[0] / hat_image.shape[1])))
            hat_y_offset = y - hat_resized.shape[0] + int(hat_resized.shape[0] * 0.15)  # Slightly down from the top of the forehead
            for i in range(hat_resized.shape[0]):
                for j in range(hat_resized.shape[1]):
                    if hat_resized.shape[2] == 4 and hat_resized[i, j][3] != 0:
                        if hat_y_offset + i >= 0:
                            image[hat_y_offset + i, x + j] = hat_resized[i, j][:3]

            # Add nose and mustache
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]
            nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (nx, ny, nw, nh) in nose:
                mustache_resized = cv2.resize(mustache_image, (nw, int(nw * mustache_image.shape[0] / mustache_image.shape[1])))
                mustache_y_offset = ny + nh - 25  # Slightly up from the bottom of the nose
                for i in range(mustache_resized.shape[0]):
                    for j in range(mustache_resized.shape[1]):
                        if mustache_resized.shape[2] == 4 and mustache_resized[i, j][3] != 0:
                            if mustache_y_offset + i < h and nx + j < w:
                                roi_color[mustache_y_offset + i, nx + j] = mustache_resized[i, j][:3]
                break  # Apply to the first detected nose

        # Add freckles
        if add_freckles_flag:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                # Add freckles below the eyes on the cheeks
                for _ in range(5):
                    cx = np.random.randint(x + ex, x + ex + ew)
                    cy = np.random.randint(y + ey + eh + 2, y + ey + eh + 13)  # Below the eyes
                    cv2.circle(image, (cx, cy), 2, (0, 0, 255), -1)

            nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (nx, ny, nw, nh) in nose:
                # Add freckles around the nose area
                for _ in range(5):
                    cx = np.random.randint(x + nx - nw // 2, x + nx + nw + nw // 2)  # Around the nose
                    cy = np.random.randint(y + ny + nh // 2, y + ny + nh + nh // 2)  # Below the nose
                    cv2.circle(image, (cx, cy), 2, (0, 0, 255), -1)

    return image

def open_image():
    global img_with_filters
    global original_image
    global add_filters_flag
    global add_freckles_flag

    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        if img is not None:
            img = resize_image(img, MAX_WIDTH, MAX_HEIGHT)
            img_with_filters = img.copy()
            original_image = img.copy()  # Store the original image

            # Convert OpenCV image to a format that can be displayed with Tkinter
            img_rgb = cv2.cvtColor(img_with_filters, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=im_pil)

            label.config(image=imgtk)
            label.image = imgtk
        else:
            print("Image file could not be loaded.")

def toggle_filters():
    global add_filters_flag
    global add_freckles_flag
    add_filters_flag = not add_filters_flag
    add_freckles_flag = False
    update_image()

def toggle_freckles():
    global add_filters_flag
    global add_freckles_flag
    add_freckles_flag = not add_freckles_flag
    add_filters_flag = False
    update_image()

def clear_updates():
    global img_with_filters
    global add_filters_flag
    global add_freckles_flag
    global original_image
    add_filters_flag = False
    add_freckles_flag = False
    img_with_filters = original_image.copy()  # Reset to the original image
    update_image()

def update_image():
    global img_with_filters
    if img_with_filters is not None:
        img_with_filters = add_filters(img_with_filters.copy())
        # Convert OpenCV image to a format that can be displayed with Tkinter
        img_rgb = cv2.cvtColor(img_with_filters, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=im_pil)
        label.config(image=imgtk)
        label.image = imgtk

# Create a Tkinter interface
root = Tk()
root.title("Face Filter Application")

label = Label(root)
label.pack()

btn_select = Button(root, text="Select Image", command=open_image)
btn_select.pack()

btn_filters = Button(root, text="Add Hat & Mustache", command=toggle_filters)
btn_filters.pack()

btn_freckles = Button(root, text="Add Freckles", command=toggle_freckles)
btn_freckles.pack()

btn_clear = Button(root, text="Clear Updates", command=clear_updates)
btn_clear.pack()

root.mainloop()
