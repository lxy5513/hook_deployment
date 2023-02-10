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
from openvino.inference_engine import IECore, Blob, TensorDesc
from serve.hook_det.yolov6.utils.nms import non_max_suppression
import argparse
import ipdb;pdb=ipdb.set_trace

CONFIG = {'weights': 'last_ckpt.pt', 'source': '15.png', 'conf_thres': 0.2, 'iou_thres': 0.2, 'max_det': 1000, 'device': 'cpu', 'save_txt': False, 'save_img': True, 'save_dir': None, 'view_img': False, 'classes': None, 'agnostic_nms': False, 'project': 'runs/inference', 'name': 'exp', 'hide_labels': False, 'hide_conf': False, 'half': False}

def letterbox(img, size=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    w, h = size

    # Scale ratio (new / old)
    r = min(h / shape[0], w / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = w - new_unpad[0], h - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (w, h)
        ratio = w / shape[1], h / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    top2, bottom2, left2, right2 = 0, 0, 0, 0
    if img.shape[0] != h:
        top2 = (h - img.shape[0])//2
        bottom2 = top2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
    elif img.shape[1] != w:
        left2 = (w - img.shape[1])//2
        right2 = left2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
    return img

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
        XML_PATH = "serve/best_stop_aug_ckpt.xml"
        BIN_PATH = "serve/best_stop_aug_ckpt.bin"
        ie_core_handler = IECore()
        network = ie_core_handler.read_network(model=XML_PATH, weights=BIN_PATH)
        self.executable_network = ie_core_handler.load_network(network, device_name='CPU', num_requests=1)
    
    def initialize(self, ctx):
        self.initialized = True
        self.model_load()
        
    def preprocess(self, data):
        image = data[0].get("body")
        image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
        image_rgb = np.array(image)
        self.image = image_rgb
        img, img_src = self.precess_image(self.image, self.img_size, 32, self.half)
        return img, img_src

    @staticmethod
    def precess_image(img_src, img_size, stride, half):
        '''Process image before image inference.'''
        image = letterbox(img_src)
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # image = torch.from_numpy(np.ascontiguousarray(image))
        # image = image.half() if half else image.float()  # uint8 to fp16/32
        image = image/255  # 0 - 255 to 0.0 - 1.0
        return image, img_src

    def inference(self, res):
        img, img_src = res
        if len(img.shape) == 3:
            img = img[None]
            # expand for batch dim
        
        inference_request = self.executable_network.requests[0]
        random_input_data = img.astype(np.float32)
        tensor_description = TensorDesc(precision="FP32", dims=img.shape, layout='NCHW')
        input_blob = Blob(tensor_description, random_input_data)
        print(inference_request.input_blobs)
        input_blob_name = next(iter(inference_request.input_blobs)) # next(iter(input_blobs))
        inference_request.set_blob(blob_name=input_blob_name, blob=input_blob)
        inference_request.infer()
        output_blob_name = next(iter(inference_request.output_blobs))
        output = inference_request.output_blobs[output_blob_name].buffer
        output = torch.from_numpy(output)
        det = non_max_suppression(output, self.conf_thres, self.iou_thres, None, False, max_det=10)[0]
        det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
        res = det.cpu().numpy().tolist()
        results = {"results": str(res)}
        print(results)
        return [results]

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
    import tqdm
    for _ in tqdm.tqdm(range(20)):
        f = open(path, 'rb')
        image_bytes = f.read()
        prop = {"model_dir": {}, "gpu_id": 0}
        ctx = {"system_properties": prop}
        ctx=argparse.Namespace(**ctx)

        data = [{"body": image_bytes}]
        data = handle(data, ctx)
        det = eval(data[0]['results']) 

    # draw_img(cv2.imread(path), det)
