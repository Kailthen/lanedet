import cv2
import os
import os.path as osp

def imshow_lanes(img, lanes, show=False, out_file=None, scores=None):
    for lane in lanes:
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 4, (255, 0, 0), 2)
    if scores is not None:
        for lane, score in zip(lanes, scores):
            x, y = lane[1]
            x, y = int(x), int(y)
            cv2.putText(img, str(score), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if show:
        cv2.imshow('view', img)
        cv2.waitKey(0)

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)

