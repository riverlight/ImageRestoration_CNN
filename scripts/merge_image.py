# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np


def merge_video():
    work_dir = "D:/workroom/testroom/"
    lst_video = ['48.mp4',
                 '48-ir-old.mp4',
                 '48-ir-old-ir.mp4']
    lst_video_final = list()
    lst_cap = list()
    width = 0
    height = 0
    image_res = None
    for i, video_file in enumerate(lst_video):
        video_file = os.path.join(work_dir, video_file)
        lst_video_final.append(video_file)
        cap = cv2.VideoCapture(video_file)
        lst_cap.append(cap)

    cap = cv2.VideoCapture(os.path.join(work_dir, lst_video[0]))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(work_dir+'v-merge.avi', fourcc, fps, (width*len(lst_video), height))

    count = 0
    while True:
        if count % 25 == 0:
            print('frame id ', count)

        break_flag = False
        lst_img = list()
        for cap in lst_cap:
            ret, frame = cap.read()
            if ret is not True:
                break_flag = True
                break
            lst_img.append(frame)

        if break_flag:
            break

        image_res = None
        for img in lst_img:
            if image_res is None:
                image_res = img
            else:
                image_res = np.hstack((image_res, img))
        out.write(image_res)
        count += 1

    out.release()
    for cap in lst_cap:
        cap.release()


def merge_pic():
    work_dir = "D:/workroom/project/riverlight/ImageRestoration_CNN/sample/"
    lst_image = ['gotham.jpg',
                 'gotham-q15.jpg',
                 'gotham-q15-ir-old.jpg',
                 'gotham-q15-ir.jpg']

    lst_image_final = list()
    width = 0
    height = 0
    image_res = None
    for i, jpg_file in enumerate(lst_image):
        jpg_file = os.path.join(work_dir, jpg_file)
        lst_image_final.append(jpg_file)
        img = cv2.imread(jpg_file)
        if i == 0:
            print(img.shape)
            width = img.shape[1]
            height = img.shape[0]
            image_res = np.copy(img)
            continue
        if height != img.shape[0]:
            img = cv2.resize(img, (int(height * img.shape[1] / img.shape[0]), height))
        print(image_res.shape, img.shape)
        image_res = np.hstack((image_res, img))

    cv2.imwrite(os.path.join(work_dir, ("res.jpg")), image_res)

def main():
    merge_pic()
    # merge_video()


if __name__ == "__main__":
    main()
