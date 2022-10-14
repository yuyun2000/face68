import os
import cv2
import face_alignment
from skimage import io
import numpy as np
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,device='cpu')

list1 = os.listdir('./data/LS3D-W/300VW-3D/CatA')

for i in range(len(list1)):
    if not os.path.exists('./data/label/CatA/%s'%list1[i]):
        os.makedirs('./data/label/CatA/%s'%list1[i])
    list = os.listdir('./data/LS3D-W/300VW-3D/CatA/%s'%list1[i])
    list_old = os.listdir('./data/label/CatA/%s'%list1[i])

    for j in range(len(list)):
        if list[j][:-4]+'.npy' in list_old:
            continue
        if 'jpg' in list[j]:
            input = io.imread('./data/LS3D-W/300VW-3D/CatA/%s/%s'%(list1[i],list[j]))
            preds = fa.get_landmarks(input)
            if preds is None:
                continue
            label = np.array(preds, dtype=np.int32)
            np.save('./data/label/CatA/%s/%s'%(list1[i],list[j][:-4]), label)


