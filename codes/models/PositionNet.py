import torch.nn as nn


class MB(nn.Module):
    def __init__(self, channels=64):
        super(MB, self).__init__()
        self.mask_func = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.mask_func(x)


class MNet(nn.Module):
    def __init__(self, channels=64, ch_in=9, ch_out=3, res_num=8):
        super(MNet, self).__init__()
        layers = []
        for _ in range(res_num):
            layers.append(MB(channels))
        self.MBs = nn.Sequential(*layers)
        self.conv_first = nn.Sequential(nn.Conv2d(ch_in, channels, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=False),
                                        nn.ReLU(inplace=True))
        self.conv_last = nn.Sequential(nn.Conv2d(channels, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                       nn.Sigmoid())

    def forward(self, x):
        y = self.conv_first(x)
        y = self.MBs(y)
        y = self.conv_last(y)
        return y
