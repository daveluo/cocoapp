from pathlib import Path
import json, pdb
import PIL, os, numpy as np, math, collections, threading, json, random, scipy, cv2
import pandas as pd, pickle, sys, itertools, string, sys, re, datetime, time, shutil, copy
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms

from torch import nn, cuda, backends, FloatTensor, LongTensor, optim
from torch.autograd import Variable
import torch.nn.functional as F

cats = {
    1: 'ground',
    2: 'coconut_tree'
}

id2cat = list(cats.values())
sz = 224

def open_image(fn):
    """ Opens an image using OpenCV given the file path.

    Arguments:
        fn: the file path of the image

    Returns:
        The image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
    """
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        #res = np.array(Image.open(fn), dtype=np.float32)/255
        #if len(res.shape)==2: res = np.repeat(res[...,None],3,2)
        #return res
        try:
            im = cv2.imread(str(fn), flags).astype(np.float32)/255
            if im is None: raise OSError(f'File not recognized by opencv: {fn}')
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e

class StdConv(nn.Module):
    def __init__(self, n_in,n_out,stride=2,dp = 0.1):
        super().__init__()
        self.conv = nn.Conv2d(n_in,n_out,3,stride=stride,padding=1)
        self.bn = nn.BatchNorm2d(n_out)
        self.dropout = nn.Dropout(dp)
        
    def forward(self,x):
        return self.dropout(self.bn(F.relu(self.conv(x))))
    
def flatten_conv(x,k):
    bs,nf,gx,gy = x.size()
    x = x.permute(0,3,2,1).contiguous()
    return x.view(bs,-1,nf//k) 

class OutConv(nn.Module):
    def __init__(self, k, n_in, bias):
        super().__init__()
        self.k = k
        self.oconv1 = nn.Conv2d(n_in, (len(id2cats)+1) * k, 3, padding=1)
        self.oconv2 = nn.Conv2d(n_in, 4 * k, 3, padding = 1)
        self.oconv1.bias.data.zero_().add_(bias)
        
    def forward(self,x):
        return [flatten_conv(self.oconv1(x), self.k),
                flatten_conv(self.oconv2(x), self.k)]

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()

def get_base():
    layers = cut_model(f_model(True), cut)
    return nn.Sequential(*layers)

class SSD_Custom_noFPN1(nn.Module):
    def __init__(self, m_base, k, bias, drop):
        super().__init__()
        self.m_base = m_base
        
        # bottom up 
        self.sfs = [SaveFeatures(m_base[i]) for i in [5,6]] # 28x28 & 14x14
        self.drop = nn.Dropout(drop)
        self.sconv1 = StdConv(512,256, dp=drop, stride=1) # 7x7
        self.sconv2 = StdConv(256,256, dp=drop) # 4x4
        self.sconv3 = StdConv(256,256, dp=drop) # 2x2
        self.sconv4 = StdConv(256,256, dp=drop) # 1x1
                  
        # lateral
        self.lat1 = nn.Conv2d(128,256, kernel_size=1, stride=1, padding=0)

        # outconvs
        self.out1 = OutConv(k, 256, bias)
        self.out2 = OutConv(k, 256, bias)
        self.out3 = OutConv(k, 256, bias)
        self.out4 = OutConv(k, 256, bias)
        self.out5 = OutConv(k, 256, bias)
        self.out6 = OutConv(k, 256, bias)
        
    def forward(self, x):
#         pdb.set_trace()
        x = self.drop(F.relu(self.m_base(x))) 
        
        c1 = self.lat1(self.sfs[0].features) # 128, 28, 28
        c2 = self.sfs[1].features # 256, 14, 14     
        c3 = self.sconv1(x)         # 256, 7, 7
        c4 = self.sconv2(c3)       # 256, 4, 4
        c5 = self.sconv3(c4)      # 256, 2, 2
        c6 = self.sconv4(c5)      # 256, 1, 1
            
        o1c,o1l = self.out1(c1)
        o2c,o2l = self.out2(c2)
        o3c,o3l = self.out3(c3)
        o4c,o4l = self.out4(c4)
        o5c,o5l = self.out5(c5)
#        o6c,o6l = self.out6(p6)
        
        return [torch.cat([o1c,o2c,o3c,o4c,o5c], dim=1),
                torch.cat([o1l,o2l,o3l,o4l,o5l], dim=1)]
    

def preproc_img(img):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img)
    img_tensor.unsqueeze_(0)
    return img_tensor

def load_model(path):
    learn = torch.load(path)
    return learn

def gen_anchors(anc_grids, anc_zooms, anc_ratios):
    anchor_scales = [(anz*i,anz*j) for anz in anc_zooms for (i,j) in anc_ratios]
    k = len(anchor_scales)
    anc_offsets = [1/(o*2) for o in anc_grids]
    anc_x = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag) for ao,ag in zip(anc_offsets,anc_grids)])
    anc_y = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag) for ao,ag in zip(anc_offsets,anc_grids)])
    anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), k, axis=0)
    anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales]) for ag in anc_grids])
    grid_sizes_np = np.concatenate([np.array([ 1/ag for i in range(ag*ag) for o,p in anchor_scales]) for ag in anc_grids])
    anchors_np = np.concatenate([anc_ctrs, anc_sizes], axis=1)
    anchors = Variable(torch.FloatTensor(anchors_np))
    grid_sizes = Variable(torch.FloatTensor(grid_sizes_np)).unsqueeze(1)
    return anchors, grid_sizes

#gen ancs
anc_grids = [28,14,7,4,2]
anc_zooms = [2**(0/3),2**(1/3),2**(2/3)]
anc_ratios = [(1.,1.), (.5,1.), (1.,.5)]
anchors, grid_sizes = gen_anchors(anc_grids, anc_zooms, anc_ratios)

def hw2corners(ctr, hw): return torch.cat([ctr-hw/2, ctr+hw/2], dim=1)

def actn_to_bb(actn, anchors, grid_sizes):
    actn_bbs = torch.tanh(actn)
    actn_centers = (actn_bbs[:,:2]/2 * grid_sizes) + anchors[:,:2]
    actn_hw = (actn_bbs[:,2:]/2+1) * anchors[:,2:]
    return hw2corners(actn_centers, actn_hw)

def to_np(v):
    if isinstance(v, (np.ndarray, np.generic)): return v
    if isinstance(v, (list,tuple)): return [to_np(o) for o in v]
    if isinstance(v, Variable): v=v.data
    if isinstance(v, torch.cuda.HalfTensor): v=v.float()
    return v.cpu().numpy()

def pred2dict(bb_np,score,cat_str):
    # convert to top left x,y bottom right x,y
    return {"x1": bb_np[1],
            "x2": bb_np[3],
            "y1": bb_np[0],
            "y2": bb_np[2],
            "score": score,
            "category": cat_str}

def get_predictions(img):
    img_t = preproc_img(img)

    #load model
    m_path = 'cocomodel_01.pt'
    model = load_model(m_path)

    #make predictions
    p_cl, p_bb = model(Variable(img_t))

    #convert bb and clas
    a_ic = actn_to_bb(p_bb[0], anchors, grid_sizes)
    clas_pr, clas_ids = p_cl[0].max(1)
    clas_pr = clas_pr.sigmoid()
    clas_ids = to_np(clas_ids)

    preds = []
    for i,a in enumerate(a_ic):
        cat_str = 'bg' if clas_ids[i]==len(id2cat) else id2cat[clas_ids[i]]
        score = to_np(clas_pr[i])[0].astype('float64')*100
        bb_np = to_np(a).astype('float64')
        preds.append(pred2dict(bb_np,score,cat_str))

    return {
        "bboxes": preds     
        }