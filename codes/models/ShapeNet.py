import torch.nn as nn
import torch
import torch.nn.functional as F


class KB(nn.Module):
    def __init__(self, channels=64):
        super(KB, self).__init__()
        self.kernel_func = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.kernel_func(x) + x


class KNet(nn.Module):
    def __init__(self, channels, ch_in, ch_out, res_num, ksize):
        super(KNet, self).__init__()
        self.ksize = ksize
        self.psize = self.ksize//2
        self.ch_out_final = 2*ch_out*self.ksize

        layers = []
        for _ in range(res_num):
            layers.append(KB(channels))
        self.KBs = nn.Sequential(*layers)
        self.conv_first = nn.Sequential(nn.Conv2d(ch_in, channels, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=False),
                                        nn.ReLU(inplace=True))
        self.conv_last = nn.Sequential(nn.Conv2d(channels,self.ch_out_final, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False))

    def forward(self, feature_map, base):
        b, _, h, w = base.shape

        y = self.conv_first(feature_map)
        y = self.KBs(y)
        y = self.conv_last(y)
        weight = y.reshape(b, 3, self.ch_out_final//3, h, w).permute(0, 1, 3, 4, 2)

        weight_h, weight_w = torch.split(weight,dim=4,split_size_or_sections=self.ch_out_final//3//2)

        weight_h = torch.softmax(weight_h, dim=4)
        weight_w = torch.softmax(weight_w, dim=4)

        base_unfold_h = F.pad(base, [0, 0, self.psize, self.psize]).unfold(2, self.ksize, 1)
        base_filtered_h = torch.mul(base_unfold_h, weight_h).sum(dim=-1, keepdim=True).squeeze(dim=-1)

        base_unfold_w = F.pad(base_filtered_h, [self.psize, self.psize, 0, 0]).unfold(3, self.ksize, 1)
        base_filtered_w = torch.mul(base_unfold_w, weight_w).sum(dim=-1, keepdim=True).squeeze(dim=-1)

        return base_filtered_w
