
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps, ImageTk
import numpy as np
import tensorflow as tf

# Load trained AI model
model = tf.keras.models.load_model("MH_AI_Vision.keras")

def recognize_digit(image_path):
    image = Image.open(image_path).convert("L")
    image = ImageOps.invert(image)
    image = image.point(lambda p: 255 if p > 80 else 0)

    bbox = image.getbbox()
    if bbox is None:
        return None, 0, None

    image = image.crop(bbox)
    image.thumbnail((20, 20))

    canvas = Image.new("L", (28, 28), 0)

    x = (28 - image.width) // 2
    y = (28 - image.height) // 2
    canvas.paste(image, (x, y))

    data = np.array(canvas).astype("float32") / 255.0
    data = data.reshape(1, 28, 28, 1)

    prediction = model.predict(data, verbose=0)

    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    return digit, confidence, canvas


def choose_image():
    path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )

    if not path:
        return

    digit, confidence, processed = recognize_digit(path)

    if digit is None:
        result_label.config(text="No digit found!")
        return

    preview = ImageTk.PhotoImage(processed.resize((224, 224)))

    image_label.config(image=preview)
    image_label.image = preview

    result_label.config(
        text=f"Detected Number: {digit}\nConfidence: {confidence:.2f}%"
    )


# Main window
root = tk.Tk()
root.title("MH AI Vision")
root.geometry("500x550")
root.configure(bg="#101522")

title = tk.Label(
    root,
    text="MH AI VISION",
    font=("Arial", 25, "bold"),
    fg="#38BDF8",
    bg="#101522"
)
title.pack(pady=25)

image_label = tk.Label(root, bg="#101522")
image_label.pack(pady=15)

choose_button = tk.Button(
    root,
    text="Choose Image",
    command=choose_image,
    font=("Arial", 14, "bold"),
    bg="#2563EB",
    fg="white",
    padx=25,
    pady=10
)
choose_button.pack(pady=15)

result_label = tk.Label(
    root,
    text="Upload an image to start",
    font=("Arial", 16, "bold"),
    fg="white",
    bg="#101522"
)
result_label.pack(pady=20)

root.mainloop()
