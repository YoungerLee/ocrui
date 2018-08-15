import os
import numpy as np
import math
import cv2 as cv

root = '/home/deeple/project/ctpn/text-detection-ctpn/data/0730/'
path = root + 'errdata'
gt_path = root + 'errlabel_txt'
out_path = 're_image'
if not os.path.exists(out_path):
    os.makedirs(out_path)
files = os.listdir(path)
files.sort()
#files=files[:100]
for file in files:
    _, basename = os.path.split(file)
    if basename.lower().split('.')[-1] not in ['jpg', 'png']:
        continue
    stem, ext = os.path.splitext(basename)
    gt_file = os.path.join(gt_path, '' + stem + '.txt')
    img_path = os.path.join(path, file)
    print(img_path)
    img = cv.imread(img_path)
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    # im_scale = float(600) / float(im_size_min)
    # if np.round(im_scale * im_size_max) > 1200:
    #     im_scale = float(1200) / float(im_size_max)
    # re_im = cv.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv.INTER_LINEAR)
    # re_size = re_im.shape
    cv.imwrite(os.path.join(out_path, stem) + '.jpg', img)

    with open(gt_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        tmp = line.strip('\n').split(',')
        xmin, ymin, xmax, ymax = [int(i) for i in tmp]
        width = xmax - xmin
        height = ymax - ymin

        # reimplement
        step = 16.0
        x_left = []
        x_right = []
        x_left.append(xmin)
        x_left_start = int(math.ceil(xmin / 16.0) * 16.0)
        if x_left_start == xmin:
            x_left_start = xmin + 16
        for i in np.arange(x_left_start, xmax, 16):
            x_left.append(i)
        x_left = np.array(x_left)

        x_right.append(x_left_start - 1)
        for i in range(1, len(x_left) - 1):
            x_right.append(x_left[i] + 15)
        x_right.append(xmax)
        x_right = np.array(x_right)

        idx = np.where(x_left == x_right)
        x_left = np.delete(x_left, idx, axis=0)
        x_right = np.delete(x_right, idx, axis=0)

        if not os.path.exists('label_tmp'):
            os.makedirs('label_tmp')
        with open(os.path.join('label_tmp', stem) + '.txt', 'a') as f:
            for i in range(len(x_left)):
                f.writelines("text\t")
                f.writelines(str(int(x_left[i])))
                f.writelines("\t")
                f.writelines(str(int(ymin)))
                f.writelines("\t")
                f.writelines(str(int(x_right[i])))
                f.writelines("\t")
                f.writelines(str(int(ymax)))
                f.writelines("\n")
