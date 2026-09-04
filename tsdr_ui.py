import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model("model/traffic_sign_model.h5")
with open("labels/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Webcam
cap = cv2.VideoCapture(0)

# Tkinter setup
root = tk.Tk()
root.title("Traffic Sign Detection and Tracking")
root.geometry("900x700")
root.configure(bg="#1e1e1e")

title = Label(root, text="Traffic Sign Detection and Tracking", font=("Segoe UI", 22, "bold"), fg="white", bg="#1e1e1e")
title.pack(pady=10)

video_label = Label(root, bg="#1e1e1e")
video_label.pack()

prediction_label = Label(root, text="", font=("Segoe UI", 18, "bold"), fg="#00ff99", bg="#1e1e1e")
prediction_label.pack(pady=10)

running = False

def preprocess(frame):
    img = cv2.resize(frame, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def detect_sign_region(frame):
    """Detect red or blue regions (common in traffic signs) and return bounding box"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red color range
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Blue color range
    lower_blue = np.array([90, 70, 50])
    upper_blue = np.array([130, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = mask_red + mask_blue

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            x, y, w, h = cv2.boundingRect(largest)
            return x, y, w, h
    return None

def update_frame():
    global running
    if running:
        ret, frame = cap.read()        
        if ret:
            box = detect_sign_region(frame)
            if box is not None:
                x, y, w, h = box
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    processed = preprocess(roi)
                    pred = model.predict(processed)
                    pred_class = np.argmax(pred)
                    confidence = np.max(pred)
                    if confidence > 0.8:
                        pred_label = labels[pred_class] if pred_class < len(labels) else "Unknown"
                        prediction_label.config(text=f"Detected: {pred_label}")
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.putText(frame, pred_label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        prediction_label.config(text="No confident detection")
                else:
                    prediction_label.config(text="No sign detected")
            else:
                prediction_label.config(text="No sign detected")

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

        video_label.after(10, update_frame)

def start_camera():
    global running
    running = True
    update_frame()

def stop_camera():
    global running
    running = False
    cap.release()
    video_label.config(image="")
    prediction_label.config(text="")

# Buttons
button_frame = tk.Frame(root, bg="#1e1e1e")
button_frame.pack(pady=20)

Button(button_frame, text="Start", command=start_camera,
       font=("Segoe UI", 14, "bold"), bg="#00cc66", fg="white", width=10).grid(row=0, column=0, padx=10)

Button(button_frame, text="Stop", command=stop_camera,
       font=("Segoe UI", 14, "bold"), bg="#ff3333", fg="white", width=10).grid(row=0, column=1, padx=10)

Button(button_frame, text="Exit", command=root.destroy,
       font=("Segoe UI", 14, "bold"), bg="#444", fg="white", width=10).grid(row=0, column=2, padx=10)

root.mainloop()
