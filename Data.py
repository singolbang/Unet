import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

dir_data = r'PATH' #data save path

name_label = r'PATH' #label data path
name_input = r'PATH' #input data path

img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size
nframe = img_label.n_frames

nframe_train = 30 # train : 30, test : 30

dir_save_train = os.path.join(dir_data, 'test') #폴더 생성


if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)


id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

offset_nframe = 0

for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

offset_nframe = nframe_train

plt.subplot(122)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(121)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()

