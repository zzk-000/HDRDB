import torch.nn as nn
import torch
import kornia
from featuremap import gradmap_cuda, flatmap_cuda
from ShapeNet import KNet
from PositionNet import MNet


def mean_init_RGB(base):
    mean_init = kornia.filters.box_blur(input=base, kernel_size=(21, 21), border_type='replicate')

    return mean_init


class PNet(nn.Module):
    def __init__(self, device):
        super(PNet, self).__init__()
        self.MNet = MNet(channels=64, ch_in=9, ch_out=3, res_num=8)
        self.gradmap_cuda = gradmap_cuda(device)
        self.flatmap_cuda = flatmap_cuda(device)

    def forward(self, base):
        mean21 = mean_init_RGB(base)

        diffmap = torch.abs(base - mean21).detach()
        gradmap = self.gradmap_cuda(base)
        flatmap = self.flatmap_cuda(base)
        feature_map = torch.cat([flatmap, gradmap, diffmap], dim=1).detach()

        mask = self.MNet(feature_map)

        MNet_K21 = mean21 * mask + base * (1 - mask)

        result = {
                'MNet_K21': MNet_K21,
                  }
        return result


class PNet_SNet(nn.Module):
    def __init__(self, device, ksize):
        super(PNet_SNet, self).__init__()
        self.MNet = MNet(channels=64, ch_in=9, ch_out=3, res_num=8)
        self.KNet = KNet(channels=64, ch_in=9, ch_out=3, res_num=8, ksize=ksize)
        self.gradmap_cuda = gradmap_cuda(device)
        self.flatmap_cuda = flatmap_cuda(device)

    def forward(self, base):
        mean21, mask02 = mean_init_RGB(base)

        diffmap = torch.abs(base - mean21).detach()
        gradmap = self.gradmap_cuda(base)
        flatmap = self.flatmap_cuda(base)
        feature_map = torch.cat([flatmap, gradmap, diffmap], dim=1).detach()

        mask = self.MNet(feature_map)

        mean = self.KNet(feature_map, base)

        MNet_KNet = mean * mask + base * (1 - mask)

        result = {
                'MNet_KNet': MNet_KNet,
                  }

        return result
