import sys
import os

import numpy as np
import torch
from tqdm.auto import tqdm

import losses
from util import get_perf_stats
from JetEfpDataSet import JetEfpDataset

class Trainer:
    def __init__(self,
            data_bkg,
            data_sig,
            num_samples = 100000,
            num_test = 10000,
            batchsize=1024,
            SB=0.0,
            use_cuda=False,
            preprocessing=None,
            std=0.025):

        self.SB = SB

        num_samples_sig = int(SB*num_samples)
        self.train_dataset = JetEfpDataset(
            data_bkg,
            data_sig,
            start_bkg=0,
            stop_bkg=num_samples,
            start_sig=0,
            stop_sig=num_samples_sig,
            preprocessing=preprocessing,
            std=std)

        self.test_dataset = JetEfpDataset(
            data_bkg,
            data_sig,
            start_bkg=num_samples,
            stop_bkg=num_samples+num_test,
            start_sig=num_samples_sig,
            stop_sig=num_samples_sig+num_test,
            preprocessing=self.train_dataset.preprocessing)

        sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.sampler.RandomSampler(self.train_dataset),
            batch_size=batchsize,
            drop_last=False)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=None)

        self.num_dim = self.test_dataset.data.shape[1]
        self.use_cuda = use_cuda
        self.std = std

        self.inn_loss = losses.INN_loss(self.num_dim)

    def train(self,
            inn,
            num_epochs=200,
            lr=0.001,
            betas=(0.9, 0.99),
            gamma=0.98,
            use_ste_decay=True,
            gamma2=0.5,
            lambda_l2=0.0,
            result_dir=None,
            disable_tqdm=True):

        self.optimizer = torch.optim.Adam(inn.parameters(), weight_decay=lambda_l2, lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

        losses_over_epochs_train = []
        losses_over_epochs_test = []

        self.train_loader.dataset.std = self.std

        for epoch in tqdm(range(1,num_epochs+1), desc='Epoch', disable=disable_tqdm):
            inn.train()
            train_inn_loss = 0
            for x, lable in self.train_loader:
                if self.use_cuda:
                    x = x.cuda()
                    lable = lable.cuda()
                self.optimizer.zero_grad()
                z, log_jac_det = inn(x)
                inn_loss = self.inn_loss(z, log_jac_det)
                inn_loss.backward()
                self.optimizer.step()
                train_inn_loss += inn_loss.detach().cpu().numpy()*len(x)

            self.scheduler.step()
            train_inn_loss /= len(self.train_loader.dataset)
            if use_ste_decay:
                self.train_loader.dataset.std *= gamma2

            inn.eval()
            with torch.no_grad():
                x, lable = self.test_dataset[:]
                if self.use_cuda:
                    x = x.cuda()
                    lable = lable.cuda()
                z, log_jac_det = inn(x)
                losses = 0.5*torch.sum(z**2, 1) - log_jac_det
                losses /= self.num_dim
                score = torch.norm(z, dim=1)

                loss = ((1.0*losses[lable==0].mean() + self.SB*losses[lable==1].mean())/(1.0+self.SB)).cpu().numpy()
                losses_over_epochs_train.append(train_inn_loss)
                losses_over_epochs_test.append(loss)

                auc_sc, imtafe_sc = get_perf_stats(lable.cpu().numpy(), score.cpu().numpy(), flip=False)
                auc_l, imtafe_l = get_perf_stats(lable.cpu().numpy(), losses.cpu().numpy(), flip=False)

                tqdm.write('\n=== epoch {} ==='.format(epoch))
                tqdm.write('inn loss (train): {}'.format(train_inn_loss))
                tqdm.write('inn loss (test): {}'.format(loss))
                tqdm.write('auc (score): {}'.format(auc_sc))
                tqdm.write('imtafe (score): {}'.format(imtafe_sc))
                tqdm.write('auc (loss): {}'.format(auc_l))
                tqdm.write('imtafe (loss): {}'.format(imtafe_l))
                tqdm.write('lr: {}'.format(self.scheduler.get_last_lr()[0]))
                tqdm.write('std: {}'.format(self.train_loader.dataset.std))
                sys.stdout.flush()

        if result_dir:
            np.save(os.path.join(result_dir, 'losses_over_epochs_train.npy'), np.array(losses_over_epochs_train))
            np.save(os.path.join(result_dir, 'losses_over_epochs_test.npy'), np.array(losses_over_epochs_test))
