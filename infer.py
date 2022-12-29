"""
脱钩检测 model server
"""
import sys
import time
import math
import json
import cv2
import os
import torch
import numpy as np
import requests
import argparse
import ipdb;pdb=ipdb.set_trace
import openvino



class HookDet(object):
    def __init__(self):
        self.class_names = ["HookBox", "Headlights"]
        self.conf_thres = 0.3
        self.iou_thres = 0.5
        self.img_size = [640, 640]
        self.half = 0
        self.initialized = False
        self.device = "cpu"

    
    def model_load(self):
        ''' Model switch to deploy status '''
        from pyopenvino.inference_engine import IECore
        ie = IECore()                                        # Create core object
        net = ie.read_network(model+'.xml', model+'.bin')    # Read model file
        print('inputs:', net.inputs)
        print('outputs:', net.outputs)
        output_node_name = net.outputs[0]['name']
        exenet = ie.load_network(net, 'CPU') 
        return exenet

    def initialize(self, ctx):
        self.initialized = True
        self.exenet = self.model_load()
        
    def preprocess(self, data):
        image = data[0].get("body")
        image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
        image_rgb = np.array(image)
        self.image = image_rgb
        img, img_src = self.precess_image(self.image, self.img_size, self.stride, self.half)
        img = img.to(self.device)
        return img, img_src

    @staticmethod
    def precess_image(img_src, img_size, stride, half):
        '''Process image before image inference.'''
        image = letterbox(img_src, img_size, stride=stride)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0
        return image, img_src

    def inference(self, res):
        img, img_src = res
        inblob = img
        exenet = self.exenet
        exenet.kernel_type = 'special'    # Set kernel implementation type ('naive', 'numpy' or 'special')
        atime = 0
        nitr = 1
        print('Kernel type:', exenet.kernel_type)
        for i in range(nitr):
            stime = time.time()
            res = exenet.infer({'data':inblob}, verbose=True)    # Run inference
            etime = time.time()
            atime += etime-stime
        print(atime/nitr, 'sec/inf')

        m = np.argsort(res[output_node_name][0])[::-1]    # Sort results
        print('Result:', m[:10])


    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2
        return boxes

_service = HookDet()
def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
    if data is None:
        return None
    data = _service.preprocess(data)
    data = _service.inference(data)
    return data


def draw_img(image, det):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    for *xyxy, conf, cls in reversed(det):
        box = xyxy
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, (0,0,255), thickness=lw, lineType=cv2.LINE_AA)
    cv2.resize(image, (720, 360))
    cv2.imwrite('tmp.jpg', image)

if __name__ == '__main__':
    path = '15.png'
    f = open(path, 'rb')
    image_bytes = f.read()
    prop = {"model_dir": {}, "gpu_id": 0}
    ctx = {"system_properties": prop}
    ctx=argparse.Namespace(**ctx)

    data = [{"body": image_bytes}]
    data = handle(data, ctx)
    det = eval(data[0]['results']) 
    # draw_img(cv2.imread(path), det)
