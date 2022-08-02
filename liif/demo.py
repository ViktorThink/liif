import argparse
import os
import os.path as osp
from PIL import Image

import torch
from torchvision import transforms

from liif.models import models
from .utils import make_coord
from .test import batched_predict




    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model', default=0)
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='large')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))
    print("model",args.model)
    if args.model == "base": 
        model_name = r'models/liif_base.pth'
    else:
        model_name = r'models/liif_large.pth'
        
    dir_name = osp.dirname(__file__) or "."

    model_path = osp.join("/".join(os.path.abspath(__file__).split("/")[0:]), model_name)
    print("path",model_path)
    print("current directory", os.getcwd())
    print("Absolute path", os.path.abspath(__file__))
    print("Real path",os.path.realpath(__file__))
    
    model = torch.load(model_path, map_location=device)['model']
    model = models.make(model, load_sd=True).to(device)

    h, w = list(map(int, args.resolution.split(',')))
    coord = make_coord((h, w)).to(device)
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, ((img - 0.5) / 0.5).to(device).unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(args.output)
