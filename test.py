# Mengimpor modul yang diperlukan
import cv2
import numpy as np
# from yolov8 import YOLOv8 # Library ultralytic untuk YOLOv8
from ultralytics import YOLO

# Membuat objek YOLOv8 dengan model yang sudah dilatih
yolo = YOLO('yolov8s.pt')

# Membuka kamera webcam
cap = cv2.VideoCapture(0)

# Melakukan loop selama kamera aktif
while cap.isOpened():
  # Membaca frame dari kamera
  ret, frame = cap.read()
  if not ret:
    break
  
  # Mendeteksi objek pada frame dengan YOLOv8
  results = yolo.predict(frame)

  # Menampilkan hasil deteksi pada frame
  # results.render() # Menggambar kotak dan label pada framsse

  # Menampilkan frame pada jendela
  cv2.imshow("YOLOv8 Object Detection", results.imgs[0])

  # Menunggu tombol 'q' untuk keluar dari loop
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Melepaskan sumber kamera dan menutup jendela
cap.release()
cv2.destroyAllWindows()