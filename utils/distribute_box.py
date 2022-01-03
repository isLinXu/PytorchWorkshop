import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def IOU(rec1, rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    rec1 = (rec1[0], rec1[1], rec1[0] + rec1[2], rec1[1] + rec1[3])
    rec2 = (rec2[0], rec2[1], rec2[0] + rec2[2], rec2[1] + rec2[3])
    left_column_max = max(rec1[0], rec2[0])
    right_column_min = min(rec1[2], rec2[2])
    up_row_max = max(rec1[1], rec2[1])
    down_row_min = min(rec1[3], rec2[3])
    # 两矩形无相交区域的情况
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        S = min(S1, S2)
        return S_cross / S


def distance_bbox(root_path):
    for name in os.listdir(root_path):
        img_path = os.path.join(root_path, name)
        img = cv2.imread(img_path)
        src = np.copy(img)
        img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.Canny(img, 130, 100)
        img_size = img.shape
        cv2.imwrite(os.path.join(root_path, "res", name), img)

        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(root_path, "res", "bin" + name + ".jpg"), img)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        # 开操作
        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=5)
        # cv2.imwrite(os.path.join(root_path, "res", "open"+name+".jpg"), img)
        # 闭操作
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=5)
        # cv2.imwrite(os.path.join(root_path, "res", "close" + name + ".jpg"), img)

        img = cv2.dilate(img, kernel)
        # img = cv2.erode(img,kernel)
        cv2.imwrite(os.path.join(root_path, "res", "dilate" + name + ".jpg"), img)

        _, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        for stat in stats:
            if stat[4] < 3500:
                if stat[2] > stat[3]:
                    r = stat[2]
                else:
                    r = stat[3]
                cv2.rectangle(img, tuple(stat[0:2]), tuple(stat[0:2] + stat[2:4]), 0, thickness=-1)

        cv2.imwrite(os.path.join(root_path, "res", name + ".jpg"), img)

        contour, hirarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        src1 = np.copy(src)

        # square = []
        # circle = []
        # for cnt in contour:
        #     approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        #     if len(approx) == 4:
        #         square.append(cnt)
        #     if len(approx) >15:
        #         circle.append(cnt)
        #
        #     x,y,w,h = cv2.boundingRect(cnt)
        #     cv2.rectangle(src1, (x, y), (x + w, y + h), (0, 0, 255))
        cv2.drawContours(src1, contour, -1, (255, 255, 0), 3)
        cv2.imwrite(os.path.join(root_path, "res", name + ".jpg.jpg.jpg.jpg"), src1)

        regon = []
        for index, cnt in enumerate(contour):
            x, y, w, h = cv2.boundingRect(cnt)
            if w / h > 10 or h / w > 10:
                continue
            if cv2.contourArea(cnt) / float(img_size[0] * img_size[1]) > 0.333:
                continue
            regon.append(cv2.contourArea(cnt))
        regon = sorted(regon, reverse=True)

        reg = 0
        for index, a in enumerate(regon):
            if index + 1 == len(regon):
                reg = regon[0]
                break
            if a < regon[index + 1] * 2:
                reg = a
                break

        therhold = 1000

        boxs = []
        temp = {}
        for index, cnt in enumerate(contour):
            if len(cnt) in temp:
                temp[len(cnt)] += 1
            else:
                temp[len(cnt)] = 1
            # if hirarchy[0][index][-1] == -1:
            x, y, w, h = cv2.boundingRect(cnt)
            if w / h > 10 or h / w > 10:
                continue
            if cv2.contourArea(cnt) < therhold or w * h / float(img_size[0] * img_size[1]) > 0.2 or h > 0.8 * img_size[
                0] or w > 0.8 * img_size[1]:
                continue
            boxs.append((x, y, w, h))
        # print(temp)
        iou_th = 0.3
        right_box = []
        for box1 in boxs:
            add = True
            for box2 in boxs:
                iou = IOU(box1, box2)
                if iou > iou_th and box1[2] * box1[3] < box2[2] * box2[3]:
                    add = False
                    break
            if not add:
                continue
            right_box.append(box1)
        # right_box = boxs

        for x, y, w, h in right_box:
            cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255))
        # cv2.drawContours(img,contour,-1,(255,255,0),3)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # cv2.imwrite(os.path.join(root_path, "res", name + ".jpg.jpg"), src)
        # cv2.imwrite(os.path.join("/home/hx/tangmy", name), src)

if __name__ == '__main__':
    root_path = ''
    distance_bbox(root_path=root_path)