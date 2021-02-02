# -*- coding: utf-8 -*-

import os
import cv2
import random

"""video to jpg"""

g_interval = 25*30
g_workdir = "D:\\workroom\\tools\\dataset\\test-seq"
g_dataset_dir = "D:\\workroom\\tools\\dataset\\IR-dataset\\eval\\"
g_tmpdir = os.path.join(g_workdir, 'tmp')
g_video_basename = "ToS-4k-1920.mp4"
g_option = 'v'
g_local_video = os.path.join(g_workdir, g_video_basename)
g_lst_idx = [0, 150, 300, 450]

def video_snapshot(video_name):
    basename = os.path.basename(video_name)
    ext = file_ext(video_name)

    cap = cv2.VideoCapture(video_name)
    count = 0
    while True:
        ret, frame = cap.read()
        if ret is not True:
            break

        # h, w, _ = frame.shape
        # frame = frame[int(h/16):int(h*15/16), 32:w-32,]

        if count % g_interval == 0:
            s_jpg_name = os.path.join(g_tmpdir, basename.replace(ext, '-%04d.jpg' % count))
            cv2.imwrite(s_jpg_name, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            print(frame.shape)
            # exit(0)
        count += 1
        if count > 25*60*30:
            break

def copy_to_dir(video_name):
    basename = os.path.basename(video_name)
    ext = file_ext(video_name)

    for id in g_lst_idx:
        s_jpg_name = os.path.join(g_tmpdir, 's-' + basename.replace(ext, '-%04d.jpg' % id))
        n_jpg_name = os.path.join(g_tmpdir, 'n-' + basename.replace(ext, '-%04d-q*.jpg' % id))
        copy_cmd = 'copy {} {}'.format(s_jpg_name, g_dataset_dir + 'source\\')
        os.system(copy_cmd)
        copy_cmd = 'copy {} {}'.format(n_jpg_name, g_dataset_dir + 'noise\\')
        os.system(copy_cmd)
    os.chdir(g_tmpdir)
    os.system('rm -rf *.jpg')


def file_ext(file_name):
    ext = file_name.split('.')[-1]
    return '.' + ext

def main():
    if g_option == 'v':
        video_snapshot(g_local_video)
    else:
        copy_to_dir(g_local_video)
    print('done')


if __name__ == "__main__":
    main()
