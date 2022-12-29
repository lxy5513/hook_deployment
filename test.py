from openvino.inference_engine import IECore, Blob, TensorDesc
import numpy as np

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


inference_request = executable_network.requests[0]
random_input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
tensor_description = TensorDesc(precision="FP32", dims=(1, 3, 640, 640), layout='NCHW')
input_blob = Blob(tensor_description, random_input_data)
print(inference_request.input_blobs)
input_blob_name = next(iter(inference_request.input_blobs)) # next(iter(input_blobs))
inference_request.set_blob(blob_name=input_blob_name, blob=input_blob)
inference_request.infer()
output_blob_name = next(iter(inference_request.output_blobs))
output = inference_request.output_blobs[output_blob_name].buffer

import torch 
import ipdb;ipdb.set_trace()
output = torch.from_numpy(output)
from serve.hook_det.yolov6.utils.nms import non_max_suppression
det = non_max_suppression(output, 0.5, 0.5, None, False, max_det=10)

