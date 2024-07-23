import time
import numpy as np
import tqdm
import cv2
import os
import torch
from codes.models.model_all import PNet_SNet
import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


class Net_test:
    def __init__(self, opt):
        self.ksize = opt.ksize
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.load_dir = opt.load_dir
        self.model = PNet_SNet(self.device, self.ksize).to(self.device)
        self.load_network()

    def load_network(self):
        if self.load_dir is not None:
            cp = torch.load(self.load_dir, map_location=self.device)
            self.model.MNet.load_state_dict(cp['MNet'])
            self.model.KNet.load_state_dict(cp['KNet'])
            print('--完成权重加载:{}--'.format(self.load_dir))
        self.model.eval()


def test(opt):
    torch.manual_seed(901)
    Net = Net_test(opt)
    list_info = []
    list_psnr_out = []

    start = time.time()
    name_list = os.listdir(opt.base_path)
    for name in tqdm.tqdm(name_list):
        base_file = opt.base_path + '/' + name
        gt_file = opt.GT_path + '/' + name

        ERGB_base = cv2.imread(base_file, flags=-1)[:, :, ::-1] / 65535
        ERGB_gt = cv2.imread(gt_file, flags=-1)[:, :, ::-1] / 65535

        input = torch.from_numpy(ERGB_base).float().permute(2, 0, 1).unsqueeze(0).to(Net.device)

        with torch.no_grad():
            res = Net.model(input)
            out = res['MNet_KNet'].squeeze(0).permute(1, 2, 0).cpu().numpy()

            if opt.save_output:
                out_file = opt.OUT_path + '/' + name
                cv2.imwrite(out_file, np.round(np.clip(out, a_min=0, a_max=1) * 65535)[:, :, ::-1].astype('uint16'))

        hdr_out = np.round(np.clip(out, a_min=0, a_max=1) * 65535)/65535
        psnr_out = compare_psnr(ERGB_gt, hdr_out)
        list_psnr_out.append(psnr_out)

        info = 'name:%s, psnr_out:%.4f\n' % (name, psnr_out)
        list_info.append(info)
        print(info)
    info = 'time:%.3f s, psnr_out:%.4f\n' % (time.time() - start, np.mean(list_psnr_out))
    list_info.append(info)
    print(info)

    with open(opt.PSNR_txt, 'w', encoding='utf-8') as f:
        for i in list_info:
            f.write(i)


def test_param():
    parser = argparse.ArgumentParser(description='')
    opt = parser.parse_args()
    opt.ksize = 21
    opt.gpu_ids = [0]
    opt.load_dir = ''
    opt.base_path = ''
    opt.GT_path = ''
    opt.PSNR_txt = ''
    opt.OUT_path = ''

    opt.save_output = True
    if opt.save_output and not os.path.exists(opt.OUT_path):
        os.makedirs(opt.OUT_path)

    test(opt)


if __name__ == '__main__':
    test_param()
