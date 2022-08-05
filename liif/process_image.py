import argparse
import os
import os.path as osp
from PIL import Image

import torch
from torchvision import transforms

from liif.models import models

from .utils import make_coord
from .test import batched_predict

import logging

def get_model(model_name="base"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "base": 
        model_name = r'models/liif_base.py'
    else:
        model_name = r'models/liif_large.pth'

        
    # current_location = os.path.abspath(__file__)
    # logging.warning("current_location "+current_location)
    # print("current_location "+current_location)
    
    current_location = osp.dirname(__file__)
    # logging.warning("current_location "+current_location)
    # print("current_location "+current_location)

    
    model_path = osp.join(current_location, model_name)
    
    # logging.warning("model_path "+model_path)
    # print("model_path "+model_path)
    
    model = torch.load(model_path, map_location=device)['model']
    model = models.make(model, load_sd=True).to(device)
    return model


def get_onnx_model(model_name="base"):
    from liif.models.liif_onnx import LIIF_ONNX

    if model_name == "base": 
        encoder = r'models/base_encoder.onnx.py'
        imnet = r'models/base_imnet.onnx.py'
    else:
        encoder = r'models/base_encoder.onnx.py'
        imnet = r'models/base_imnet.onnx.py'
        
    current_location = osp.dirname(__file__)

    encoder = osp.join(current_location, encoder)
    imnet = osp.join(current_location, imnet)
    model = LIIF_ONNX(encoder, imnet)
    return model

        
def process_image(model, pil_image, resolution):

    img = frame_to_tensor(pil_image)
    # logging.warning("shape "+str(img.shape))
    # logging.warning("img "+str(img))
    
    pred = process_frame(model, img, resolution)
    
    img = tensor_to_frame(pred)
    
    return img

# def process_frame(model, img, resolution):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if type(resolution) == str:
#         h, w = list(map(int, resolution.split(',')))
#     else:
#         h, w = resolution
#     coord = make_coord((h, w)).to(device)
#     cell = torch.ones_like(coord)
#     cell[:, 0] *= 2 / h
#     cell[:, 1] *= 2 / w
#     pred = batched_predict(model, ((img - 0.5) / 0.5).to(device).unsqueeze(0),
#         coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
#     pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    
#     return pred

def process_frame(model, img, resolution):
    if type(resolution) == str:
        h, w = list(map(int, resolution.split(',')))
    else:
        h, w = resolution
    coord = make_coord((h, w))
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, ((img - 0.5) / 0.5).unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    
    return pred

def frame_to_tensor(pil_image):
    img = transforms.ToTensor()(pil_image.convert('RGB'))
    return img


def tensor_to_frame(pil_image):
    img = transforms.ToPILImage()(pil_image)
    return img