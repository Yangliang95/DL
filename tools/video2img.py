import os
import cv2
from PIL import Image
from tqdm import  tqdm,trange
import imageio
import os.path as osp

def video2img(sp,dp):
    """ 将视频转换成图片
        sp: 视频路径 """
    cap = cv2.VideoCapture(sp)
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0 #-1

    while suc:
        frame_count += 1
        suc, frame = cap.read()
        params = []
        params.append(2)  # params.append(1)
        if suc:
            cv2.imwrite(dp+'\\frame_%d.jpg' % frame_count, frame, params)
    cap.release()

    print('unlock image: ', frame_count)


def jpg2video(ip,sp, fps):
    """ 将图片合成视频. ip: 图片路径， sp: 视频路径，fps: 帧率 """

    fourcc = cv2.VideoWriter_fourcc(*'XVID') #*'XVID'  *"MJPG"
    images = os.listdir(ip)
    im = Image.open(ip + '/' + images[0])
    vw = cv2.VideoWriter(sp, fourcc, fps, im.size)
    os.chdir(ip)
    for image in trange(len(images),colour='green'):
        # Image.open(str(image)+'.jpg').convert("RGB").save(str(image)+'.jpg')
        jpgfile = 'frame_' + str(image) + '.jpg'
        # jpgfile = str(image) + '.jpg'
        try:
            frame = cv2.imread(jpgfile)
            vw.write(frame)
        except Exception as exc:
            print(jpgfile, exc)
    vw.release()
    print(sp, 'Synthetic success!')


def img2gif(img_dir,gif_path,duration,start_time=0,end_time=5):
    """

    :param img_dir: 包含图片的文件夹
    :param gif_path: 输出的gif的路径
    :param duration: 每张图片切换的时间间隔，与fps的关系：duration = 1 / fps
    :return:
    """

    gif_path = osp.join(gif_path, 'output' + str(start_time) + '_' + str(end_time) + '.gif')

    frames = []
    # f =os.listdir(img_dir)
    # f.sort(key=lambda x: int(x[6:-4]))
    for idx in tqdm(sorted(os.listdir(img_dir),key=lambda x: int(x[6:-4]))[start_time:end_time]):
        img = osp.join(img_dir, idx)
        frames.append(imageio.imread(img))

    imageio.mimsave(gif_path, frames, 'GIF', duration=duration)
    print('Finish changing!')



if __name__ == '__main__':
    sp = r"H:\Some_proj\vehicle-speed-estimation\speedchallenge-master\data\train.mp4"
    dp= r"H:\Some_proj\vehicle-speed-estimation\speedchallenge-master\data\train_img"
    sp_new = 'val.mp4'
    ip = r'H:\Some_proj\vehicle-speed-estimation\vehicle-speed-estimation-master\tools\image'

    img_dir = r'H:\Some_proj\vehicle-speed-estimation\vehicle-speed-estimation-master\tools\image'
    par_dir = osp.dirname(img_dir)
    img2gif(img_dir=img_dir,gif_path=par_dir, duration=0.1,start_time=1000,end_time=2000) #图片转GIF
    # video2img(sp,dp)  # 视频转图片
    # jpg2video(ip,sp_new, 30)  # 图片转视频
