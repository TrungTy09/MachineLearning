import numpy as np
import cv2
import time
import os

# Label: 00000 là ko cầm tiền, còn lại là các mệnh giá  nhớ chỉnh lại cái nhãn lưu ý nên để số kí tự trong nhãn là như nhau
label = "050000"

i = 0
max_images = 2000  

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

while(True):
    i += 1
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow('frame', frame)

    if i >= 61 and i <= 2060:  
        print("Số ảnh capture = ", i - 60)
        if not os.path.exists('data/' + str(label)):
            os.mkdir('data/' + str(label))

        cv2.imwrite('data/' + str(label) + "/" + str(i) + ".png", frame)

    if i >= 2060:
        print("Đã chụp đủ 2000 ảnh, chương trình dừng.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
