"change the image to video"

import cv2
import os
from tqdm import tqdm,trange

img_root = r'H:\Some_proj\vehicle-speed-estimation\vehicle-speed-estimation-master\tools\image'
fps = 30    #保存视频的FPS
size=(640,480)
#可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('./test1.avi',fourcc,fps,size)#最后一个是保存图片的尺寸

files = os.listdir(img_root)
os.chdir(img_root)
# for f in tqdm(files,colour='green'):
for image in trange(len(files),colour = 'green'):
    fpath = 'frame_' + str(image) + '.jpg'

    # fpath = os.path.join(img_root,f)
    frame = cv2.imread(fpath)
    videoWriter.write(frame)
videoWriter.release()
