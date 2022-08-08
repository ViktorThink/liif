import torch
import torch.nn as nn
import torch.nn.functional as F

from . import models
from .models import register
from ..utils import make_coord

import numpy as np
import onnxruntime as ort


@register('liif')
class LIIF_ONNX(nn.Module):

    def __init__(self, encoder_path, imnet_path, local_ensemble=True, feat_unfold=True, cell_decode=True, providers=None, sess_options=None):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        
        if sess_options == None:
            sess_options = ort.SessionOptions()

        
        if providers != None:
            self.encoder = ort.InferenceSession(encoder_path,providers=providers, sess_options=sess_options)
            self.imnet = ort.InferenceSession(imnet_path, providers=providers, sess_options=sess_options)
        elif ort.get_device() =="GPU":
            self.encoder = ort.InferenceSession(encoder_path,providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'], sess_options=sess_options)
            self.imnet = ort.InferenceSession(imnet_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'], sess_options=sess_options)
        else:
            self.encoder = ort.InferenceSession(encoder_path)
            self.imnet = ort.InferenceSession(imnet_path)
        self.encoder_binding = self.encoder.io_binding()
        self.imnet_binding = self.imnet.io_binding()




    def gen_feat(self, inp):
        inp = inp.cuda()
        
        binding = self.encoder.io_binding()
        binding.bind_input(
            name='input',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(inp.shape),
            buffer_ptr=inp.data_ptr(),
            )
        
        
        Y_shape = list(tuple(inp.shape)) # You need to specify the output PyTorch tensor shape
        Y_shape[1]=64
        Y_tensor = torch.empty(Y_shape, dtype=torch.float32, device='cuda').contiguous()
        binding.bind_output(
            name='output',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(Y_shape),
            buffer_ptr=Y_tensor.data_ptr(),
        )

        self.encoder.run_with_iobinding(binding)
        out = binding.copy_outputs_to_cpu()[0]
        
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
                
                inp = inp.view(bs * q, -1)
                
                
                inp = inp.cuda()
                
                binding = self.imnet.io_binding()
                binding.bind_input(
                    name='input',
                    device_type='cuda',
                    device_id=0,
                    element_type=np.float32,
                    shape=tuple(inp.shape),
                    buffer_ptr=inp.data_ptr(),
                    )
                
                
                Y_shape = list(tuple(inp.shape)) # You need to specify the output PyTorch tensor shape
                Y_shape[1]=3
                Y_tensor = torch.empty(Y_shape, dtype=torch.float32, device='cuda').contiguous()
                binding.bind_output(
                    name='output',
                    device_type='cuda',
                    device_id=0,
                    element_type=np.float32,
                    shape=tuple(Y_shape),
                    buffer_ptr=Y_tensor.data_ptr(),
                )        
                
                
                
                self.imnet.run_with_iobinding(binding)
                out = binding.copy_outputs_to_cpu()[0]
                
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
