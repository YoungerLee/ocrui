import os
import cv2
import shutil
IoU_threshold = 0.7
img_dir = '/home/deeple/project/ctpn/text-detection-ctpn/data/blurdata'
err_dir = '/home/deeple/project/ctpn/text-detection-ctpn/data/errdata'
def computeIoU(A, B):
    A_xmin, A_ymin, A_xmax, A_ymax = A
    B_xmin, B_ymin, B_xmax, B_ymax = B
    W = min(A_xmax, B_xmax) - max(A_xmin, B_xmin)
    H = min(A_ymax, B_ymax) - max(A_ymin, B_ymin)
    if W <= 0 or H <= 0:
        return 0
    area_A = (A_xmax - A_xmin) * (A_ymax - A_ymin)
    area_B = (B_xmax - B_xmin) * (B_ymax - B_ymin)
    area_inter = W * H
    return area_inter*1.0 / (area_A + area_B - area_inter)

def evaluation(gt_dir, pred_dir):
    TP, FP, FN = 0, 0, 0
    for item in os.listdir(pred_dir):
        pred_path = os.path.join(pred_dir, item)
        gt_item = str(item.split('_blur_')[0]) + '.txt'
        gt_path = os.path.join(gt_dir, gt_item)
        f = open(pred_path, 'r')
        g = open(gt_path, 'r')
        lines_pred = f.readlines()
        lines_gt = g.readlines()
        gt_bbox = [int(i) for i in lines_gt[0].strip('\n').split(',')]
        if not lines_pred:
            FN += 1
            img_path = os.path.join(img_dir, item.replace('txt', 'jpg'))
            dst_path = os.path.join(err_dir, item.replace('txt', 'jpg'))
            shutil.copy(img_path, dst_path)
        elif len(lines_pred) <= 1:
            pred_bbox = [int(i) for i in lines_pred[0].strip('\n').split(',')]
            if computeIoU(pred_bbox, gt_bbox) >= IoU_threshold:
                TP += 1
            else:
                FP += 1
                img_path = os.path.join(img_dir, item.replace('txt', 'jpg'))
                dst_path = os.path.join(err_dir, item.replace('txt', 'jpg'))
                shutil.copy(img_path, dst_path)
        else:
            FP += 1
            img_path = os.path.join(img_dir, item.replace('txt', 'jpg'))
            dst_path = os.path.join(err_dir, item.replace('txt', 'jpg'))
            shutil.copy(img_path, dst_path)
    print('TP = %d FP = %d FN = %d' %(TP, FP, FN))
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F = 2.0 * P * R / (P + R)
    return P, R, F

def pick_out_abnormal(src_dir):
    img_dir = '/home/deeple/project/ctpn/text-detection-ctpn/data/samples0730_resize'
    err_dir = '/home/deeple/project/ctpn/text-detection-ctpn/data/errdata'
    for item in os.listdir(src_dir):
        label = item.split('_')[0]
        pred_path = os.path.join(src_dir, item)
        f = open(pred_path, 'r')
        lines_pred = f.readlines()
        if len(lines_pred) != 1:
            print(item, lines_pred)
            img_name = item.replace('txt', 'jpg')
            img_path = os.path.join(img_dir, label, img_name)
            shutil.copyfile(img_path, os.path.join(err_dir, img_name))



if __name__ == '__main__':
    gt_dir = '/home/deeple/project/ctpn/text-detection-ctpn/data/results/gtlabel'
    pred_dir = '/home/deeple/project/ctpn/text-detection-ctpn/data/results/predictions'
    # P, R, F = evaluation(gt_dir, pred_dir)
    # print('precision = %f, recall = %f, F-score = %f' % (P, R, F))
    pick_out_abnormal(pred_dir)
