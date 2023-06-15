import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import json
import argparse
from PIL import Image
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network
from pathlib import Path
from tqdm import tqdm

from evadd import io_utils as IO

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, img_path):
        ori_img = cv2.imread(img_path)
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'img_path':img_path, 'ori_img':ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.get_lanes(data)
        return data

    def show(self, data):
        out_file = self.cfg.savedir 
        if out_file:
            out_file = osp.join(out_file, "vis", osp.basename(data['img_path']))
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        id_classes = [lane.metadata['id_class'] for lane in data['lanes']]
        scores = [lane.metadata['score'] for lane in data['lanes']]
        imshow_lanes(data['ori_img'], lanes, show=self.cfg.show, out_file=out_file, scores=scores)

        # write to json
        img_bfn,_ = osp.splitext(osp.basename(data['img_path']))
        json_fn = f"{self.cfg.savedir}/{img_bfn}.json"

        j_lanes = []
        for i, l_arr in enumerate(lanes):
            l_dict = {
                "image_name": osp.basename(data['img_path']),
                "points": np.around(l_arr, decimals=3).tolist(),
                "class": id_classes[i],
                "class_name": "unknown",
                "color": 0,
                "color_name": "unknown",
                "score": scores[i],
            }
            j_lanes.append(l_dict)
        # json.dump(j_lanes, open(json_fn, 'wt'))
        IO.save_to_json(json_fn, j_lanes)

    def run(self, data):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]
        if self.cfg.show or self.cfg.savedir:
            self.show(data)
        return data

def get_img_paths(path):
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return paths 

def process(args):
    cfg = Config.fromfile(args.config)
    cfg.show = args.show
    cfg.savedir = args.savedir
    cfg.load_from = args.load_from
    detect = Detect(cfg)

    paths = get_img_paths(args.img)
    for p in tqdm(paths):
        detect.run(p)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('--img',  help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true', 
            help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default=None, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args()

    paths = get_img_paths(args.img)
    im_ = Image.open(paths[0])
    # reset width,height
    ori_img_h = im_.height
    ori_img_w = im_.width
    
    cfg_dir = os.path.dirname(args.config)
    cfg_fn, _ = os.path.splitext(os.path.basename(args.config))
    new_cfg_fn = f"{cfg_dir}/_tmp_{cfg_fn}_{ori_img_w}x{ori_img_h}.py"

    if not os.path.exists(new_cfg_fn):
        cfg_lines = []
        with open(args.config, 'rt') as f:
            for l in f.readlines():
                if l.startswith("ori_img_h"):
                    l = f"ori_img_h={ori_img_h}\n"
                elif l.startswith("ori_img_w"):
                    l = f"ori_img_w={ori_img_w}\n"
                cfg_lines.append(l)
        with open(new_cfg_fn, 'wt') as f:
            f.writelines(cfg_lines)
    args.config = new_cfg_fn
    process(args)
