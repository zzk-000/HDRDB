import numpy as np
import cv2
from scipy.signal import convolve2d
import torch.nn as nn
import torch
import torch.nn.functional as F
import kornia


def lengthMask(img):
    h, w = np.shape(img)
    Co = 21
    ret, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    binary = binary.astype('uint8')
    contours, hierarchy = cv2.findContours(binary, 2, cv2.CHAIN_APPROX_NONE)
    canva = np.zeros((h, w))

    for i in range(len(contours)):
        if len(contours[i]) > Co:
            wc = len(contours[i])
            cv2.drawContours(canva, contours[i], -1, wc, 1)
    return canva


def directMask(img):
    wx1 = np.array([[-1, 0, 1]])
    wy1 = np.array([[-1], [0], [1]])
    grd_x = convolve2d(img, wx1, mode='same')
    grd_y = convolve2d(img, wy1, mode='same')

    grd_xx = grd_x * grd_x
    grd_yy = grd_y * grd_y
    grd_xy = grd_x * grd_y

    sum_xx = cv2.blur(grd_xx, ksize=(3, 3))
    sum_yy = cv2.blur(grd_yy, ksize=(3, 3))
    sum_xy = cv2.blur(grd_xy, ksize=(3, 3))

    tmp = ((sum_xx - sum_yy) ** 2 + 4 * sum_xy ** 2) ** (1 / 2)
    e1 = 0.5 * (sum_xx + sum_yy + tmp)
    e2 = 0.5 * (sum_xx + sum_yy - tmp)
    directmap = (e1 - e2) / (e1 + e2 + 1e-16)
    return directmap


def gradMask(img):
    def get_gx(img):
        wx1 = np.array([[-1, 0, 1]])
        wx2 = np.array([[-1, 0, 0, 0, 1]])
        wx3 = np.array([[-1, 0, 0, 0, 0, 0, 1]])
        gx1 = convolve2d(img, wx1, mode='same')
        gx2 = convolve2d(img, wx2, mode='same')
        gx3 = convolve2d(img, wx3, mode='same')

        gx_merge = (gx1 + gx2 + gx3) / 3
        gx = np.abs(gx1.copy())
        gx[np.abs(gx_merge - gx1) > 0.5/1023] = 0

        gx[np.abs(gx) > 4/1023] = 0
        gx[np.abs(gx) <= 1/1023] = 0

        return gx

    def get_gy(img):
        wy1 = np.array([[-1], [0], [1]])
        wy2 = np.array([[-1], [0], [0], [0], [1]])
        wy3 = np.array([[-1], [0], [0], [0], [0], [0], [1]])
        gy1 = convolve2d(img, wy1, mode='same')
        gy2 = convolve2d(img, wy2, mode='same')
        gy3 = convolve2d(img, wy3, mode='same')
        gy_merge = (gy1 + gy2 + gy3) / 3
        gy = np.abs(gy1.copy())
        gy[np.abs(gy_merge - gy1) > 0.5/1023] = 0

        gy[np.abs(gy) > 4/1023] = 0
        gy[np.abs(gy) <= 1/1023] = 0
        return gy

    gx = get_gx(img)
    gy = get_gy(img)
    grd_merge = np.zeros([img.shape[0], img.shape[1], 2])
    grd_merge[:, :, 0] = gx
    grd_merge[:, :, 1] = gy
    gradmap = np.max(np.abs(grd_merge), 2) * 1023 / 4
    return gradmap


class gradmap(nn.Module):
    def __init__(self, device):
        super(gradmap, self).__init__()
        self.device = device

    def get_gradmap_final(self, EY):
        gradmap = gradMask(EY)
        directmap = directMask(gradmap)
        lengthmap = lengthMask(gradmap)
        gradmap_final = gradmap * directmap * lengthmap
        return gradmap_final

    def forward(self, base):
        base_np = base.detach().cpu().numpy()
        b, n, h, w = base_np.shape
        gradmap_final_RGB = np.zeros([b, n, h, w])
        for i in range(b):
            for j in range(n):
                gradmap_final_RGB[i, j] = self.get_gradmap_final(base_np[i, j])
        gradmap_final_RGB = torch.clamp(torch.from_numpy(gradmap_final_RGB).float().to(self.device), 0, 1).detach()
        return gradmap_final_RGB


class gradmap_cuda(nn.Module):
    def __init__(self, device):
        super(gradmap_cuda, self).__init__()
        self.device = device
        self.grd_bottom = 1 / 1023
        self.grd_up = 4 / 1023

        wx1 = torch.FloatTensor([[-1, 0, 1]]).unsqueeze(0).unsqueeze(0).to(self.device)
        wx2 = torch.FloatTensor([[-1, 0, 0, 0, 1]]).unsqueeze(0).unsqueeze(0).to(self.device)
        wx3 = torch.FloatTensor([[-1, 0, 0, 0, 0, 0, 1]]).unsqueeze(0).unsqueeze(0).to(self.device)
        wy1 = torch.FloatTensor([[-1], [0], [1]]).unsqueeze(0).unsqueeze(0).to(self.device)
        wy2 = torch.FloatTensor([[-1], [0], [0], [0], [1]]).unsqueeze(0).unsqueeze(0).to(self.device)
        wy3 = torch.FloatTensor([[-1], [0], [0], [0], [0], [0], [1]]).unsqueeze(0).unsqueeze(0).to(self.device)
        self.wx1 = nn.Parameter(data=wx1, requires_grad=False)
        self.wx2 = nn.Parameter(data=wx2, requires_grad=False)
        self.wx3 = nn.Parameter(data=wx3, requires_grad=False)
        self.wy1 = nn.Parameter(data=wy1, requires_grad=False)
        self.wy2 = nn.Parameter(data=wy2, requires_grad=False)
        self.wy3 = nn.Parameter(data=wy3, requires_grad=False)

        k_closing = torch.ones(3, 3).float().to(self.device)
        self.k_closing = nn.Parameter(data=k_closing, requires_grad=False)

    def get_directmap(self, img):
        grd_x = F.conv2d(img, self.wx1, padding=(0, 1))
        grd_y = F.conv2d(img, self.wy1, padding=(1, 0))
        grd_xx = grd_x * grd_x
        grd_yy = grd_y * grd_y
        grd_xy = grd_x * grd_y

        sum_xx = kornia.filters.box_blur(grd_xx, kernel_size=(3, 3), border_type='replicate') * (3 * 3)
        sum_yy = kornia.filters.box_blur(grd_yy, kernel_size=(3, 3), border_type='replicate') * (3 * 3)
        sum_xy = kornia.filters.box_blur(grd_xy, kernel_size=(3, 3), border_type='replicate') * (3 * 3)

        tmp = torch.sqrt((sum_xx - sum_yy).pow(2) + 4 * sum_xy.pow(2))
        e1 = 0.5 * (sum_xx + sum_yy + tmp)
        e2 = 0.5 * (sum_xx + sum_yy - tmp)
        directmap = (e1 - e2) / (e1 + e2 + 1e-16)
        return directmap

    def get_gradmap_final(self, img):
        gx1 = F.conv2d(img, self.wx1, padding=(0, 1))
        gx2 = F.conv2d(img, self.wx2, padding=(0, 2))
        gx3 = F.conv2d(img, self.wx3, padding=(0, 3))
        gx_merge1 = (gx1 + gx2 + gx3) / 3 - 0.5 / 1023
        gx_merge2 = (gx1 + gx2 + gx3) / 3 + 0.5 / 1023
        gx = gx1.clone().detach()
        gx[(gx1 < gx_merge1) | (gx1 > gx_merge2)] = 0

        gy1 = F.conv2d(img, self.wy1, padding=(1, 0))
        gy2 = F.conv2d(img, self.wy2, padding=(2, 0))
        gy3 = F.conv2d(img, self.wy3, padding=(3, 0))
        gy_merge1 = (gy1 + gy2 + gy3) / 3 - 0.5 / 1023
        gy_merge2 = (gy1 + gy2 + gy3) / 3 + 0.5 / 1023
        gy = gy1.clone().detach()
        gy[(gy1 < gy_merge1) | (gy1 > gy_merge2)] = 0

        gx[gx.abs() < self.grd_bottom] = 0
        gx[gx.abs() > self.grd_up] = 0
        gy[gy.abs() < self.grd_bottom] = 0
        gy[gy.abs() > self.grd_up] = 0

        grd_merge = torch.cat((gx.abs().unsqueeze(-1), gy.abs().unsqueeze(-1)), -1)
        gradmap = torch.max(grd_merge, -1)[0] * (1 / self.grd_up)
        return gradmap

    def forward(self, img):
        img_R, img_G, img_B = torch.split(img, split_size_or_sections=1, dim=1)
        gradmap_R = self.get_gradmap_final(img_R)
        gradmap_G = self.get_gradmap_final(img_G)
        gradmap_B = self.get_gradmap_final(img_B)
        gradmap_RGB = torch.cat([gradmap_R, gradmap_G, gradmap_B], dim=1).detach()

        gradmap_RGB4 = kornia.morphology.closing(gradmap_RGB,self.k_closing)
        return gradmap_RGB4


class flatmap_cuda(nn.Module):
    def __init__(self, device):
        super(flatmap_cuda, self).__init__()
        self.device = device
        wx1 = torch.FloatTensor([[-1, 0, 1]]).unsqueeze(0).unsqueeze(0).to(self.device)
        wy1 = torch.FloatTensor([[-1], [0], [1]]).unsqueeze(0).unsqueeze(0).to(self.device)
        self.wx1 = nn.Parameter(data=wx1, requires_grad=False)
        self.wy1 = nn.Parameter(data=wy1, requires_grad=False)

    def get_flatmap_final(self, img):
        grd_x = F.conv2d(img, self.wx1, padding=(0, 1))
        grd_y = F.conv2d(img, self.wy1, padding=(1, 0))
        flatmask = img.clone().detach()
        flatmask[:,:,:,:] = 0
        flatmask[grd_x.abs()>=2 / 1023] = 1
        flatmask[grd_y.abs()>=2 / 1023] = 1
        return flatmask

    def forward(self, img):
        img_R, img_G, img_B = torch.split(img, split_size_or_sections=1, dim=1)
        flatmask_R = self.get_flatmap_final(img_R)
        flatmask_G = self.get_flatmap_final(img_G)
        flatmask_B = self.get_flatmap_final(img_B)
        flatmask_RGB = torch.cat([flatmask_R, flatmask_G, flatmask_B], dim=1).detach()
        return flatmask_RGB
