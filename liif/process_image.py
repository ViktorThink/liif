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

def get_model(model_name="large"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "base": 
        model_name = r'models/liif_base.py'
    else:
        model_name = r'models/liif_large.pth'

        
    current_location = os.path.abspath(__file__)
    logging.info("current_location "+current_location)
    print("current_location "+current_location)
    
    current_location = osp.dirname(__file__)
    logging.info("current_location "+current_location)
    print("current_location "+current_location)

    
    model_path = osp.join("/".join(current_location.split("/")[:-1]), model_name)

    
    model = torch.load(model_path, map_location=device)['model']
    model = models.make(model, load_sd=True).to(device)
    return model

def process_frame(model, pil_image, resolution, device=None):
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    img = frame_to_tensor(pil_image)

    h, w = list(map(int, resolution.split(',')))
    coord = make_coord((h, w)).to(device)
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, ((img - 0.5) / 0.5).to(device).unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    
    img = tensor_to_frame(pred)
    return img



def frame_to_tensor(pil_image):
    img = transforms.ToTensor()(pil_image.convert('RGB'))
    return img


def tensor_to_frame(pil_image):
    img = transforms.ToPILImage()(pil_image)
    return img