import os
import cv2
import face_alignment
from skimage import io
import numpy as np
def makeGaussian(size, fwhm = 2, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
list1 = os.listdir('./data/label/AFLW2000-3D-Reannotated')

for i in range(len(list1)):
    img = cv2.imread('./data/LS3D-W/AFLW2000-3D-Reannotated/%s.jpg'%list1[i][:-4])
    h, w, c = img.shape
    print(h,w,list1[i])
    preds = np.load('./data/label/AFLW2000-3D-Reannotated/%s'%list1[i])
    label = np.zeros((64, 64, 68), dtype=np.float32)
    for j in range(68):
        xmin = int(preds[0][j][0] / w * 64) if int(preds[0][j][0] / w * 64) < 64 else 63
        ymin = int(preds[0][j][1] / h * 64) if int(preds[0][j][1] / h * 64) < 64 else 63
        # print(xmin, ymin)
        label[:, :, j:j + 1] = makeGaussian(64, center=(xmin, ymin)).reshape(64, 64, 1)
    np.save('./data2/label/AFLW2000-3D-Reannotated/%s'%list1[i],label)

