import torchlm
import cv2
import numpy as np
transform = torchlm.LandmarksCompose([
    torchlm.LandmarksRandomScale(prob=0.5),
    torchlm.LandmarksRandomMask(prob=0.5),
    torchlm.LandmarksRandomBlur(kernel_range=(5, 25), prob=0.5),
    torchlm.LandmarksRandomBrightness(prob=0.),
    torchlm.LandmarksRandomRotate(40, prob=0.5, bins=8),
    torchlm.LandmarksRandomCenterCrop((0.5, 1.0), (0.5, 1.0), prob=0.5)
])



preds = np.load('./data/label/AFLW2000-3D-Reannotated/image00204.npy').reshape(68,2)
# preds = preds.astype(np.float32)
img = cv2.imread('./data/LS3D-W/AFLW2000-3D-Reannotated/image00204.jpg')


img1 , lm = transform(img,preds)
# print(lm)
for i in range(68):
    cv2.circle(img,(int(preds[i][0]),int(preds[i][1])),1,(0,0,255),1)
    cv2.circle(img1, (int(lm[i][0]), int(lm[i][1])), 1, (0, 0, 255), 1)

print(img.shape)
print(img1.shape)
cv2.imshow('1',img)
cv2.imshow('2',img1)
cv2.waitKey(0)