import cv2
import numpy as np
from keras.models import load_model

# Load model and labels
model = load_model('model/traffic_sign_model.h5')

with open('labels.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def preprocess(img):
    img = cv2.resize(img, (30, 30))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

cap = cv2.VideoCapture(0)
print("Starting camera... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    x1, y1, x2, y2 = w//2 - 100, h//2 - 100, w//2 + 100, h//2 + 100
    roi = frame[y1:y2, x1:x2]

    img = preprocess(roi)
    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    label = f"{classes[class_id]} ({confidence*100:.1f}%)"
    color = (0,255,0) if confidence > 0.7 else (0,0,255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Traffic Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
