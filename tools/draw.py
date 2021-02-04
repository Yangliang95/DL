"draw the output in image"
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import os

def draw_text(filepath,text1,text2):
    "draw the prediction and label in image"
    ":param filepath  the path of the image "
    ":param text1 , text2 the text which will be draw"
    bk_img = cv2.imread(filepath)
    #设置需要显示的字体
    fontpath = "font/simsun.ttc"
    font = ImageFont.truetype(fontpath, 24)
    img_pil = Image.fromarray(bk_img)
    draw = ImageDraw.Draw(img_pil)
    #绘制文字信息
    error_str = str(float(text1)-float(text2))
    draw.text((40, 30),  'prediction_speed: ' + text1, font = font, fill = (0, 0, 255)) #bgr
    draw.text((40, 60),  'groundtruth_speed: ' + text2, font = font, fill = (0, 255, 0))
    draw.text((40, 90), 'Error: ' + (error_str[0:6] if len(error_str)>=6 else error_str), font=font, fill=(255, 0, 255))
    bk_img = np.array(img_pil)

    # cv2.imshow("add_text",bk_img)
    # cv2.waitKey()
    cv2.imwrite("./image/"+os.path.split(filepath)[1],bk_img)

if __name__=="__main__":
    filepath = r'H:\Some_proj\vehicle-speed-estimation\speedchallenge-master\data\train_img'
    labelpath = r'H:\Some_proj\vehicle-speed-estimation\vehicle-speed-estimation-master\res.txt'
    files = os.listdir(filepath)
    files.sort(key=lambda x: int(x[6:-4]))
    with open(labelpath,'r') as lab:

        for f in files:
            fpath = os.path.join(filepath,f)
            text_pre,text_lab = lab.readline().split(',')
            text_pre, text_lab = text_pre[0:6],text_lab[0:6]
            draw_text(fpath,text_pre,text_lab)


