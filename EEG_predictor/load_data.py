import numpy as np
from scipy.io import loadmat
import scipy.signal as signal
import time

def load_data(path,num_label_class):
    print('load data: '+str(path))
    EEGdata = loadmat(file_name=path)
    EEGdata = EEGdata['MI_data']
    #
    EEGdata2 = loadmat(file_name='new_data1/XJQ_data2.mat')
    EEGdata2 = EEGdata2['MI_data']
    #
    EEGdata = np.concatenate([EEGdata2,EEGdata],axis=2)
    EEGdata = np.concatenate([EEGdata[:,:,:,i] for i in range(4)],axis=-1)
    EEGdata = np.swapaxes(EEGdata,0,-1)
    EEGdata = np.swapaxes(EEGdata, 1,2)

    label = np.concatenate([np.array([[j] for i in range(num_label_class)]) for j in range(4)],axis=0)
    EEGdata = pre_processing(EEGdata)
    return EEGdata,label
def pre_processing(EEGdata):
    a1, b1 = signal.butter(4,[0.003925,0.3125],'bandpass') #0.0625,0.235
    EEGdata = signal.filtfilt(a1,b1,EEGdata, axis=-1)
    mean_val = EEGdata.mean(axis=-1)
    std_val = EEGdata.std(axis=-1)
    mean_val = np.stack([mean_val for _ in range(1800)],axis=-1)
    std_val = np.stack([std_val for _ in range(1800)], axis=-1)
    EEGdata = EEGdata-mean_val
    EEGdata = EEGdata /std_val
    return EEGdata



# #导入工具包
# import cv2
# import numpy as np
# import math
# from matplotlib import pyplot as plt
#
# # 读取图像
# img = cv2.imread('C:/Users/lenovo/Desktop/lena.jpg', 0)#0的含义:将图像转化为单通道灰度图像
#
# def ideal_bandpass_filter(img,D0,w):
#     img_float32 = np.float32(img)#转换为np.float32格式，这是oppencv官方要求，咱们必须这么做
#     dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)#傅里叶变换,得到频谱图
#     dft_shift = np.fft.fftshift(dft)#将频谱图低频部分转到中间位置，三维(263,263,2)
#     rows, cols = img.shape #得到每一维度的数量
#     crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
#     mask = np.ones((rows, cols,2), np.uint8)  #对滤波器初始化，长宽和上面图像的一样
#     for i in range(0, rows): #遍历图像上所有的点
#         for j in range(0, cols):
#             d = math.sqrt(pow(i - crow, 2) + pow(j - ccol, 2)) # 计算(i, j)到中心点的距离
#             if D0 - w / 2 < d < D0 + w / 2:
#                 mask[i, j,0]=mask[i,j,1] = 1
#             else:
#                 mask[i, j,0]=mask[i,j,1] = 0
#     f = dft_shift * mask  # 滤波器和频谱图像结合到一起，是1的就保留下来，是0的就全部过滤掉
#     ishift = np.fft.ifftshift(f) #上面处理完后，低频部分在中间，所以傅里叶逆变换之前还需要将频谱图低频部分移到左上角
#     iimg = cv2.idft(ishift) #傅里叶逆变换
#     res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1]) #结果还不能看，因为逆变换后的是实部和虚部的双通道的一个结果，这东西还不是一个图像，为了让它显示出来我们还需要对实部和虚部进行一下处理才可以
#     return res
#
# new_image1=ideal_bandpass_filter(img,D0=6,w=10)
# new_image2=ideal_bandpass_filter(img,D0=15,w=10)
# new_image3=ideal_bandpass_filter(img,D0=25,w=10)
#
# # 显示原始图像和带通滤波处理图像
# title=['Source Image','D0=6','D0=15','D0=25']
# images=[img,new_image1,new_image2,new_image3]
# for i in np.arange(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(title[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()




if __name__ == '__main__':
    EEGdata,label = load_data(path)