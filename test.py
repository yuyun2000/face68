import numpy
import tensorflow as tf
import cv2
import numpy as np
from Flops import try_count_flops
def my_softmax(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x)

model = tf.keras.models.load_model("./face68-125-up.h5")
flop = try_count_flops(model)
print(flop/1000000)


# imagegt = cv2.imread('./test/indoor_003.png')
# # imagegt = np.concatenate((imagegt.reshape(523,930,1),imagegt.reshape(523,930,1),imagegt.reshape(523,930,1)),2)
#
# # imagegt = np.pad(imagegt,((300,300),(300,300),(0,0)))
# h,w,c = imagegt.shape
# # imagegt = np.concatenate((imagegt.reshape(h,w,1),imagegt.reshape(h,w,1),imagegt.reshape(h,w,1)),2)
#
# imagegt = cv2.resize(imagegt,(256,256))
# image = imagegt.astype(np.float32)
# img = image / 255
# img = img.reshape(1,256,256,3)
# out = model(img,training=False)
# out = np.array(tf.reshape(out[0:1,:,:,:],(64,64,68)))
#
# for k in range(68):
#     max = np.max(out[:,:,k:k+1].flatten())
#     # print(max)
#     for i in range(64):
#         for j in range(64):
#             if out[i][j][k] >=max:
#                 # cv2.rectangle(imagegt, (j * 4, i * 4), (j * 4 + 4, i * 4 + 4), (255, 0, 0), 1)
#                 cv2.circle(imagegt, (j * 4 + 2, i * 4 + 2), 1, (0, 0, 255), 2)
#
# imagegt = cv2.resize(imagegt,(256,256))
# cv2.imshow('1',imagegt)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


vid = cv2.VideoCapture('./test/2.mp4')
fourcc = cv2.VideoWriter_fourcc(*'I420')
outv = cv2.VideoWriter('output.avi',fourcc,20,(256,256))
while True:
    flag,img = vid.read(0)
    if not flag:
        break
    img = np.pad(img, ((100, 100), (100, 100), (0, 0)))
    img0 = cv2.resize(img,(256,256))
    img = img0.astype(np.float32)
    img = img / 255
    img = img.reshape(1, 256, 256, 3)
    out = model(img, training=False)

    out = np.array(tf.reshape(out[0:1,:,:,:],(64,64,68)))
    for k in range(68):
        max = np.max(out[:, :, k:k + 1].flatten())
        # print(max)
        for i in range(64):
            for j in range(64):
                if out[i][j][k] >= max:
                    # if max < 0.2:
                        # break
                    # cv2.rectangle(img0, (j * 4, i * 4), (j * 4 + 4, i * 4 + 4), (255, 0, 0), 1)
                    cv2.circle(img0,(j*4+2,i*4+2),1,(0,0,255),2)

    img0 = cv2.resize(img0, (256, 256))
    outv.write(img0)
    cv2.imshow('1', img0)
    if ord('q') == cv2.waitKey(1):
        break
vid.release()
outv.release()
#销毁所有的数据
cv2.destroyAllWindows()
