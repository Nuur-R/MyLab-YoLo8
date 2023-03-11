import cv2

# program python untuk mengambil gambar dari webcam dan keyboard q untuk menghentikan program
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break