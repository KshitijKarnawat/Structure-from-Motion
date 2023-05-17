import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def downSample(image, factor):
    for i in range(0, factor):
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape
        image = cv.pyrDown(image, dstsize=(col // 2, row // 2))
    return image


model_path = "../models/"
model_name = "model-small.onnx"
model = cv.dnn.readNet(model_path + model_name)

if model.empty():
    print("Model not loaded")


frame = cv.imread("../Data/rotate/left/IMG_20230516_234015.jpg")

cv.imshow("frame", frame)
h, w, c = frame.shape

blob = cv.dnn.blobFromImage(frame, 1/255., (256,256), (123.675, 116.28, 103.53), True, False)

# Set input to the model
model.setInput(blob)

output = model.forward()
output = output[0,:,:]
output = cv.resize(output, (w,h))
output = cv.normalize(output, None, 0, 255, norm_type=cv.NORM_MINMAX, dtype= cv.CV_32F)

# frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

cv.imwrite('colorimg.jpg',frame)
cv.imwrite('depth.png',output)

plt.imshow(output, 'jet')
plt.show()
cv.waitKey()
cv.destroyAllWindows()