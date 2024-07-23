import time
import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from torch.nn import init
from codes.dataset.dataset import create_dataset
from codes.models.model_all import PNet
import argparse
import os
import shutil


class Net_train:
    def __init__(self, opt):
        self.gpu_ids = opt.gpu_ids
        self.device = opt.device
        self.lr = opt.lr
        self.gamma = opt.gamma
        self.step = opt.step
        self.milestones = opt.milestones
        self.save_dir = opt.save_dir
        self.load_dir = opt.load_dir
        self.tarin_status()
        self.tbnum = opt.tbnum
        if opt.tbnum != 0:
            self.tb = SummaryWriter(opt.logdir)
            self.tbstep = 0

    def tarin_status(self):
        self.model = PNet(self.device).to(self.device)
        self.set_loss_optimizer_scheduler()
        self.load_network()
        if self.gpu_ids:
            self.model = nn.DataParallel(self.model, self.gpu_ids)

    def set_loss_optimizer_scheduler(self):
        self.optim = optim.Adam(self.model.MNet.parameters(), lr=self.lr)
        self.sche = lr_scheduler.MultiStepLR(self.optim, milestones=self.milestones, gamma=self.gamma)
        self.L1 = nn.L1Loss().to(self.device)
        self.optimizers = []
        self.optimizers.append(self.optim)

    def get_current_lr(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups][0]

    def schedulers_step(self):
        self.sche.step()

    def load_network(self):
        self.init_weight(self.model, 'xavier')
        if self.load_dir is not None:
            checkpoint = torch.load(self.load_dir, map_location=self.device)
            self.model.MNet.load_state_dict(checkpoint['MNet'])
            print('--完成权重加载:{}--'.format(self.load_dir))
        self.model.train()

    def save_network(self, epoch):
        save_path = self.save_dir + 'model_{}.pth'.format(epoch)
        state = {
            'MNet': self.model.module.MNet.state_dict(),
        }
        torch.save(state, save_path)

    def init_weight(self, net, init_type):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data)
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data)
                else:
                    raise NotImplementedError('initialization method {} is not implemented'.format(init_type))
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data)
                init.constant_(m.bias.data, 0.0)

        print('--initialize network with {}'.format(init_type))
        net.apply(init_func)

    def train_step(self, data):
        self.optim.zero_grad()

        """set data"""
        self.base = data['base'].to(self.device)
        self.gt = data['gt'].to(self.device)

        """forward"""
        self.result = self.model(self.base)
        self.MNet_K21 = self.result['MNet_K21']

        """cal loss"""
        self.loss_MNet_K21 = self.L1(self.MNet_K21, self.gt)
        self.psnr_MNet_K21 = kornia.losses.psnr_loss(self.MNet_K21, self.gt, max_val=1)

        self.loss_use = self.loss_MNet_K21

        """back"""
        self.loss_use.backward()
        self.optim.step()

    def tensorboard(self, iter_num):
        loss_MNet_K21 = self.loss_MNet_K21.item()
        psnr_MNet_K21 = self.psnr_MNet_K21.item()

        if self.tbnum != 0 and iter_num % 1 == 0:
            self.tbstep += 1
            self.tb.add_scalar('loss_MNet_K21', loss_MNet_K21, global_step=self.tbstep)
            self.tb.add_scalar('psnr_MNet_K21', psnr_MNet_K21, global_step=self.tbstep)
        return loss_MNet_K21, psnr_MNet_K21


def train(opt):
    torch.manual_seed(901)
    train_loader = create_dataset(opt)
    Net = Net_train(opt)

    for epoch in range(opt.epoch_start, opt.epoch_end + 1):
        list_loss_MNet_K21 = []
        list_psnr_MNet_K21 = []

        start = time.time()
        lr = Net.get_current_lr()
        for i, data in enumerate(train_loader, 1):
            Net.train_step(data)
            loss_MNet_K21, psnr_MNet_K21 = Net.tensorboard(i)

            list_loss_MNet_K21.append(loss_MNet_K21)
            list_psnr_MNet_K21.append(psnr_MNet_K21)

        epoch_message = 'epoch:%d,time:%.1f,bsize:%d,lr:%.7f --- lMNet_K21:%.5f --- pMNet_K21:%.3f\n' \
                        % (epoch, (time.time() - start) / 60, opt.batch_size, lr, np.mean(list_loss_MNet_K21), np.mean(list_psnr_MNet_K21))
        print(epoch_message)
        print('------------')
        with open(opt.loss_file, 'a', encoding='utf-8') as f:
            f.write(epoch_message)
            f.write('\n')

        Net.schedulers_step()
        if epoch % opt.save_epoch == 0:
            Net.save_network(epoch=epoch)
    if opt.tbnum != 0:
        Net.tb.close()


def check_dir(opt):
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    if opt.tbnum != 0:
        if os.path.exists(opt.logdir):
            shutil.rmtree(opt.logdir)
        os.makedirs(opt.logdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    opt = parser.parse_args()
    GPU = '3090'

    if GPU == 'A100':
        opt.train_base = ''
        opt.train_gt = ''
        opt.num = 0
        opt.gpu_ids = [0]
        opt.num_workers = 32
        opt.batch_size = 16
    else:
        opt.train_base = ''
        opt.train_gt = ''
        opt.num = 0
        opt.gpu_ids = [0]
        opt.num_workers = 16
        opt.batch_size = 8

    opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    opt.tbnum = 1
    opt.lr = 2e-4
    opt.gamma = 0.5
    opt.epoch_start = 1
    opt.epoch_end = 50
    opt.milestones = [10, 20, 30, 40]
    opt.save_epoch = 1
    opt.load_dir = None
    opt.save_dir = ''
    opt.loss_file = opt.save_dir + 'loss.txt'
    opt.logdir = opt.save_dir + 'logs'

    check_dir(opt)

    with open(opt.loss_file, 'w', encoding='utf-8') as f:
        timestr = time.strftime('  %Y-%m-%d : %H:%M:%S', time.localtime())
        f.write('%s\n'%(timestr))
        print(time)
        for i in vars(opt).keys():
            line = "%12s : "%i + "%s"%vars(opt)[i]
            f.write('%s\n'%(line))
            print(line)
        f.write('\n')

    train(opt)
