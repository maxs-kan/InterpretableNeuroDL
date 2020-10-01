import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

def jittering(img, jit, C, D, H, W) :
    return np.pad(img, [(0, 0), (0, jit), (0, jit), (0, jit)], mode='constant')
def upsample(mask, img_size):
    x = F.interpolate(mask, size=img_size, mode='trilinear', align_corners=False)
    return x
def np_to_torch(X, device, img_size, requires_grad=False):
    output = torch.tensor(X, requires_grad=requires_grad, device=device)
    return  output.reshape(img_size)
def tv_norm(x, beta = 1):
    d1 = torch.mean(torch.abs(x[:,:,:-1,:,:] - x[:,:,1:,:,:]).pow(beta))
    d2 = torch.mean(torch.abs(x[:,:,:,:-1,:] - x[:,:,:,1:,:]).pow(beta))
    d3 = torch.mean(torch.abs(x[:,:,:,:,:-1] - x[:,:,:,:,1:]).pow(beta))
    tv =  d1 + d2 + d3
    return tv
class GaussianFilter(nn.Module):
    def __init__(self, k_size, device, g_filter):
        super(GaussianFilter, self).__init__()
        self.device = device
        pad = (k_size - 1) // 2
        self.k_size = k_size
        self.conv =  nn.Conv3d(1, 1, k_size,padding=(pad, pad, pad), bias=None)
        self.conv.to(device)
        self.g_filter = g_filter
    def forward(self, x, sigma):
        n= np.zeros((self.k_size, self.k_size, self.k_size))
        n[self.k_size // 2 + 1, self.k_size // 2 + 1, self.k_size // 2 + 1] = 1
        k = self.g_filter(n, sigma=sigma)[None, None,:,:,:]
        self.conv.weight = torch.nn.Parameter(torch.from_numpy(k).float().to(self.device))
        for param in self.conv.parameters():
            param.requires_grad = False
        return self.conv(x)

class MeanPertrub():
    def __init__(self, device,mask_scale=4, blur_img=10, blur_mask=10, max_iter=300, 
                 l1_coef=3, tv_coef=1, tv_beta=7, rep=10, jit=5, k_size=5, lr=0.3):
        self.device = device
        self.lr = lr
        self.mask_scale = 4
        self.blur_img = blur_img
        self.blur_mask = blur_mask
        self.max_iter = max_iter
        self.l1_coef = l1_coef
        self.tv_coef = tv_coef
        self.tv_beta = tv_beta
        self.rep = rep
        self.jit = jit
        self.filter_gaus = GaussianFilter(k_size, device, gaussian_filter)
    
    def get_masks(self, X, pred, model):
        res = []
        N, C, D, H, W = X.shape
        rw_max = self.max_iter // 5
        for i, img in tqdm(enumerate(X), total=X.shape[0]):
            model_ans = pred[i]
            mask = torch.ones((1, C, D // self.mask_scale, H // self.mask_scale, W // self.mask_scale), requires_grad=True, device=self.device)
            optimizer = Adam([mask], lr=self.lr, betas=(0.9, 0.99), amsgrad=True)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            best_loss, best_mask = float('inf'), None
            for epoch in range(self.max_iter):
                mask_up = upsample(mask, img_size=(D, H, W))
                mask_up = self.filter_gaus(mask_up, self.blur_mask)
                total_pred_loss = 0
                for _ in range(self.rep):
                    img_jit = jittering(img, self.jit, C, D, H, W)
                    j0 = np.random.randint(self.jit)
                    j1 = np.random.randint(self.jit)
                    j2 = np.random.randint(self.jit)
                    img_jit = img_jit[:, j0:(D + j0), j1:(H + j1), j2:(W + j2)] 
                    img_torch = np_to_torch(img_jit, self.device, img_size=(1, C, D, H, W), requires_grad=False)
                    blur = self.filter_gaus(img_torch, self.blur_img)
                    perturbated_input = img_torch.mul(mask_up) + blur.mul(1 - mask_up)
                    outputs = model(perturbated_input.float())
                    prob = torch.exp(outputs)
                    total_pred_loss += F.relu(prob[0, model_ans] - 0.05)
                reg_loss = self.l1_coef * torch.mean(torch.abs(1 - mask)) + self.tv_coef * tv_norm(mask_up, self.tv_beta)
                rw = 1 if epoch > rw_max else epoch / rw_max
                loss = total_pred_loss / self.rep + rw * reg_loss
                
                if epoch > 50 and loss.item() <= best_loss:
                    best_loss = loss.item()
                    best_mask = mask.clone().detach()
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                mask.data.clamp_(0, 1)
            res_mask = upsample((1 - best_mask), img_size=(D, H, W))
            res.append(res_mask.cpu().numpy())
        X_mask = np.concatenate(res, axis=0)
        X_mask =  X_mask.squeeze(axis=1)
        return X_mask