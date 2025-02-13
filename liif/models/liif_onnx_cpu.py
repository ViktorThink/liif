import torch
import torch.nn as nn
import torch.nn.functional as F

from . import models
from .models import register
from ..utils import make_coord

import numpy as np
import onnxruntime as ort


@register('liif')
class LIIF_ONNX_CPU(nn.Module):

    def __init__(self, encoder_path, imnet_path, local_ensemble=True, feat_unfold=True, cell_decode=True, providers=None, sess_options=None):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        
        if sess_options == None:
            sess_options = ort.SessionOptions()
            
            sess_options.intra_op_num_threads = 10
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        
        if providers != None:
            self.encoder = ort.InferenceSession(encoder_path,providers=providers, sess_options=sess_options)
            self.imnet = ort.InferenceSession(imnet_path, providers=providers, sess_options=sess_options)
        elif ort.get_device() =="GPU":
            self.encoder = ort.InferenceSession(encoder_path,providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'], sess_options=sess_options)
            self.imnet = ort.InferenceSession(imnet_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'], sess_options=sess_options)
        else:
            self.encoder = ort.InferenceSession(encoder_path)
            self.imnet = ort.InferenceSession(imnet_path)



    def gen_feat(self, inp):
        
        inp = inp.numpy().astype(np.float32)
        out = self.encoder.run(None,{"input":inp})[0]
        self.feat = torch.from_numpy(out)
        
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        

        feat_coord = make_coord(feat.shape[-2:], flatten=False) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                
                imnet_input = inp.view(bs * q, -1).numpy().astype(np.float32)
                out = self.imnet.run(None,{"input":imnet_input})[0]
                pred = torch.from_numpy(out)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
